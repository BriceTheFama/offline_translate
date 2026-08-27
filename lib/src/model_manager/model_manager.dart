import 'dart:io';

import 'package:crypto/crypto.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

import '../core/language.dart';
import '../core/model_info.dart';
import '../exceptions/exceptions.dart';
import 'model_source.dart';

/// Installs, verifies and removes translation models on the device.
abstract class ModelManager {
  /// Whether the model for [pair] is installed and passes a quick check.
  Future<bool> isInstalled(LanguagePair pair);

  /// Downloads and verifies the model for [pair].
  ///
  /// This is the only operation in the package that needs network access, and
  /// only when the source is a remote one. Reports progress through
  /// [onProgress]. Installing an already-installed model of the same version is
  /// a no-op unless [force] is set.
  Future<ModelInfo> install(
    LanguagePair pair, {
    void Function(InstallProgress progress)? onProgress,
    bool force = false,
  });

  /// Deletes the installed model for [pair]. Does nothing when absent.
  Future<void> delete(LanguagePair pair);

  /// Lists every installed model.
  Future<List<ModelInfo>> installedModels();

  /// Returns the installed model for [pair], or `null`.
  Future<ModelInfo?> getModel(LanguagePair pair);

  /// Verifies every file of an installed model against its checksum.
  ///
  /// Throws [ModelCorruptedException] on the first mismatch.
  Future<void> verify(LanguagePair pair);

  /// Directory holding installed models.
  Future<Directory> modelsDirectory();
}

/// Filesystem-backed [ModelManager].
///
/// Models live in `<application support>/offline_translator/models/<pair>`.
/// Downloads land in a sibling `.tmp` directory and are only moved into place
/// after every checksum matches, so an interrupted install can never leave a
/// half-written model that later fails at inference time.
class FileModelManager implements ModelManager {
  /// Creates a manager fetching from [source].
  ///
  /// [source] may be omitted when the models are already on disk — sideloaded
  /// by the application, or shipped with it. Everything except [install] works
  /// without one, which is what lets a fully offline application be built with
  /// no download path at all.
  ///
  /// [rootPath] overrides the default location, which is useful in tests.
  FileModelManager({this.source, String? rootPath})
      : _rootPathOverride = rootPath;

  /// Where model files are fetched from, when they are fetched at all.
  final ModelSource? source;

  ModelSource _requireSource(LanguagePair pair) {
    final configured = source;
    if (configured == null) {
      throw ModelNotInstalledException(
          '${pair.id}" and this translator has no model source; pass '
          '`modelSource:` to initialize() to download models, or install them '
          'on disk yourself');
    }
    return configured;
  }

  final String? _rootPathOverride;
  Directory? _root;

  /// Extra free space required beyond the model size, as a safety margin.
  static const int _freeSpaceMargin = 32 * 1024 * 1024;

  @override
  Future<Directory> modelsDirectory() async {
    final cached = _root;
    if (cached != null) return cached;
    final base = _rootPathOverride ??
        p.join((await getApplicationSupportDirectory()).path,
            'offline_translate', 'models');
    final dir = Directory(base);
    if (!dir.existsSync()) await dir.create(recursive: true);
    _root = dir;
    return dir;
  }

  Future<Directory> _modelDir(LanguagePair pair) async =>
      Directory(p.join((await modelsDirectory()).path, pair.id));

  @override
  Future<bool> isInstalled(LanguagePair pair) async =>
      await getModel(pair) != null;

  @override
  Future<ModelInfo?> getModel(LanguagePair pair) async {
    final dir = await _modelDir(pair);
    final manifest = File(p.join(dir.path, 'manifest.json'));
    if (!manifest.existsSync()) return null;
    ModelInfo info;
    try {
      info = ModelInfo.parse(await manifest.readAsString(), path: dir.path);
    } catch (_) {
      return null;
    }
    // Cheap structural check: every listed file must exist with the right size.
    for (final file in info.files) {
      final f = File(p.join(dir.path, file.name));
      if (!f.existsSync() || f.lengthSync() != file.size) return null;
    }
    return info;
  }

  @override
  Future<List<ModelInfo>> installedModels() async {
    final root = await modelsDirectory();
    final out = <ModelInfo>[];
    for (final entity in root.listSync()) {
      if (entity is! Directory) continue;
      final pair = _parsePairId(p.basename(entity.path));
      if (pair == null) continue;
      final info = await getModel(pair);
      if (info != null) out.add(info);
    }
    out.sort((a, b) => a.id.compareTo(b.id));
    return out;
  }

  static LanguagePair? _parsePairId(String id) {
    final parts = id.split('-');
    if (parts.length != 2) return null;
    final from = Language.tryParse(parts[0]);
    final to = Language.tryParse(parts[1]);
    if (from == null || to == null) return null;
    return LanguagePair(from, to);
  }

  @override
  Future<ModelInfo> install(
    LanguagePair pair, {
    void Function(InstallProgress progress)? onProgress,
    bool force = false,
  }) async {
    void report(InstallStage stage, int received, int total, [String? file]) {
      onProgress?.call(InstallProgress(
        pair: pair,
        stage: stage,
        receivedBytes: received,
        totalBytes: total,
        currentFile: file,
      ));
    }

    report(InstallStage.manifest, 0, 0);
    final remote = _requireSource(pair);
    final manifestText = await remote.fetchManifest(pair);
    final ModelInfo wanted;
    try {
      wanted = ModelInfo.parse(manifestText);
    } catch (e) {
      throw ModelDownloadException(pair.id, 'malformed manifest ($e)');
    }
    if (wanted.pair != pair) {
      throw ModelDownloadException(
          pair.id, 'manifest describes ${wanted.id}, not ${pair.id}');
    }

    final existing = await getModel(pair);
    if (!force && existing != null && existing.checksum == wanted.checksum) {
      report(InstallStage.done, existing.size, existing.size);
      return existing;
    }

    final total = wanted.size;
    await _requireFreeSpace(pair, total);

    final dir = await _modelDir(pair);
    final staging = Directory('${dir.path}.tmp');
    if (staging.existsSync()) await staging.delete(recursive: true);
    await staging.create(recursive: true);

    try {
      var received = 0;
      for (final file in wanted.files) {
        report(InstallStage.downloading, received, total, file.name);
        final destination = File(p.join(staging.path, file.name));
        await remote.fetchFile(pair, file.name, destination, onBytes: (delta) {
          received += delta;
          report(InstallStage.downloading, received, total, file.name);
        });
      }

      for (final file in wanted.files) {
        report(InstallStage.verifying, total, total, file.name);
        final destination = File(p.join(staging.path, file.name));
        if (!destination.existsSync()) {
          throw ModelCorruptedException(pair.id, file.name, 'file missing');
        }
        final size = destination.lengthSync();
        if (size != file.size) {
          throw ModelCorruptedException(
              pair.id, file.name, 'expected ${file.size} bytes, got $size');
        }
        final digest = await _sha256(destination);
        if (digest != file.sha256) {
          throw ModelCorruptedException(pair.id, file.name,
              'checksum mismatch (expected ${file.sha256}, got $digest)');
        }
      }

      await File(p.join(staging.path, 'manifest.json'))
          .writeAsString(manifestText);

      if (dir.existsSync()) await dir.delete(recursive: true);
      await staging.rename(dir.path);
      report(InstallStage.done, total, total);
      return wanted.withPath(dir.path);
    } catch (_) {
      if (staging.existsSync()) {
        await staging.delete(recursive: true);
      }
      rethrow;
    }
  }

  Future<void> _requireFreeSpace(LanguagePair pair, int needed) async {
    // `File.statSync` cannot report free space; shell out only where it is both
    // available and cheap, and skip the check when it is not.
    try {
      final root = await modelsDirectory();
      final result = Process.runSync('df', <String>['-k', root.path]);
      if (result.exitCode != 0) return;
      final lines = (result.stdout as String).trim().split('\n');
      if (lines.length < 2) return;
      final columns = lines[1].split(RegExp(r'\s+'));
      if (columns.length < 4) return;
      final availableKb = int.tryParse(columns[3]);
      if (availableKb == null) return;
      final available = availableKb * 1024;
      if (available < needed + _freeSpaceMargin) {
        throw InsufficientStorageException(
            needed + _freeSpaceMargin, available);
      }
    } on InsufficientStorageException {
      rethrow;
    } catch (_) {
      // Free-space reporting is best effort; a failed probe must not block an
      // install that would otherwise succeed.
    }
  }

  @override
  Future<void> verify(LanguagePair pair) async {
    final dir = await _modelDir(pair);
    final manifest = File(p.join(dir.path, 'manifest.json'));
    if (!manifest.existsSync()) throw ModelNotInstalledException(pair.id);
    final info = ModelInfo.parse(await manifest.readAsString(), path: dir.path);
    for (final file in info.files) {
      final f = File(p.join(dir.path, file.name));
      if (!f.existsSync()) {
        throw ModelCorruptedException(pair.id, file.name, 'file missing');
      }
      final digest = await _sha256(f);
      if (digest != file.sha256) {
        throw ModelCorruptedException(pair.id, file.name, 'checksum mismatch');
      }
    }
  }

  @override
  Future<void> delete(LanguagePair pair) async {
    final dir = await _modelDir(pair);
    if (dir.existsSync()) await dir.delete(recursive: true);
    final staging = Directory('${dir.path}.tmp');
    if (staging.existsSync()) await staging.delete(recursive: true);
  }

  static Future<String> _sha256(File file) async {
    final digest = await file.openRead().transform(sha256).first;
    return digest.toString();
  }
}
