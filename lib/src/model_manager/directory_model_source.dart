import 'dart:io';

import 'package:path/path.dart' as p;

import '../core/language.dart';
import '../exceptions/exceptions.dart';
import 'model_source.dart';

/// Installs model bundles from a directory already present on the device.
///
/// Useful when the application ships its models itself (side-loaded, bundled in
/// an expansion file, or pushed by an MDM) and must never touch the network,
/// and for integration tests. The expected layout is the same as
/// [HttpModelSource]: one directory per direction holding `manifest.json` and
/// the files it lists.
class DirectoryModelSource implements ModelSource {
  /// Creates a source rooted at [rootPath].
  const DirectoryModelSource(this.rootPath);

  /// Directory holding one subdirectory per language direction.
  final String rootPath;

  File _file(LanguagePair pair, String name) =>
      File(p.join(rootPath, pair.id, name));

  @override
  Future<String> fetchManifest(LanguagePair pair) async {
    final file = _file(pair, 'manifest.json');
    if (!file.existsSync()) {
      throw ModelDownloadException(pair.id, 'no manifest at ${file.path}');
    }
    return file.readAsString();
  }

  @override
  Future<void> fetchFile(
    LanguagePair pair,
    String name,
    File destination, {
    void Function(int delta)? onBytes,
  }) async {
    final source = _file(pair, name);
    if (!source.existsSync()) {
      throw ModelDownloadException(pair.id, 'missing file ${source.path}');
    }
    final sink = destination.openWrite();
    try {
      await for (final chunk in source.openRead()) {
        sink.add(chunk);
        onBytes?.call(chunk.length);
      }
      await sink.flush();
    } finally {
      await sink.close();
    }
  }
}
