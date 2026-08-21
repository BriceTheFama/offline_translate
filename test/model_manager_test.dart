import 'dart:convert';
import 'dart:io';

import 'package:crypto/crypto.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:offline_translate/offline_translate.dart';
import 'package:path/path.dart' as p;

/// Builds a throwaway model bundle whose "ONNX" files are just bytes. The
/// manager never opens them, it only checks sizes and checksums.
Directory buildFakeBundle(Directory root, LanguagePair pair,
    {String version = '1.0.0', int payloadSize = 2048}) {
  final dir = Directory(p.join(root.path, pair.id))
    ..createSync(recursive: true);
  final files = <Map<String, dynamic>>[];
  for (final name in <String>[
    'encoder.onnx',
    'decoder.onnx',
    'source.spm',
    'vocab.json',
  ]) {
    final bytes = List<int>.generate(
        payloadSize, (i) => (name.codeUnitAt(0) + i + version.length) % 256);
    File(p.join(dir.path, name)).writeAsBytesSync(bytes);
    files.add(<String, dynamic>{
      'name': name,
      'size': bytes.length,
      'sha256': sha256.convert(bytes).toString(),
    });
  }
  final checksum = sha256
      .convert(utf8.encode(
          (files.map((f) => f['sha256']! as String).toList()..sort()).join()))
      .toString();
  File(p.join(dir.path, 'manifest.json')).writeAsStringSync(jsonEncode({
    'from': pair.from.code,
    'to': pair.to.code,
    'version': version,
    'checksum': checksum,
    'base_model': 'Helsinki-NLP/opus-mt-${pair.id}',
    'license': 'Apache-2.0',
    'quantization': 'int8',
    'architecture': <String, dynamic>{
      'decoder_layers': 6,
      'decoder_attention_heads': 8,
      'head_dimension': 64,
      'decoder_start_token_id': 59513,
      'eos_token_id': 0,
      'pad_token_id': 59513,
      'max_position_embeddings': 512,
      'vocab_size': 59514,
    },
    'files': files,
  }));
  return dir;
}

/// A source that can be told to fail, to corrupt, or to count its calls.
class ScriptedSource implements ModelSource {
  ScriptedSource(this.delegate);

  final ModelSource delegate;
  int manifestCalls = 0;
  int fileCalls = 0;
  bool failManifest = false;
  bool failFiles = false;
  String? corruptFile;

  @override
  Future<String> fetchManifest(LanguagePair pair) {
    manifestCalls++;
    if (failManifest) {
      throw ModelDownloadException(pair.id, 'scripted failure');
    }
    return delegate.fetchManifest(pair);
  }

  @override
  Future<void> fetchFile(LanguagePair pair, String name, File destination,
      {void Function(int delta)? onBytes}) async {
    fileCalls++;
    if (failFiles) {
      throw ModelDownloadException(pair.id, 'scripted failure on $name');
    }
    await delegate.fetchFile(pair, name, destination, onBytes: onBytes);
    if (corruptFile == name) {
      final bytes = destination.readAsBytesSync();
      bytes[0] = bytes[0] ^ 0xFF;
      destination.writeAsBytesSync(bytes);
    }
  }
}

void main() {
  const enFr = LanguagePair(Language.en, Language.fr);
  const frEn = LanguagePair(Language.fr, Language.en);

  late Directory tmp;
  late Directory remote;
  late Directory installRoot;
  late ScriptedSource source;
  late FileModelManager manager;

  setUp(() {
    tmp = Directory.systemTemp.createTempSync('ot-mm-test');
    remote = Directory(p.join(tmp.path, 'remote'))..createSync();
    installRoot = Directory(p.join(tmp.path, 'installed'))..createSync();
    buildFakeBundle(remote, enFr);
    buildFakeBundle(remote, frEn);
    source = ScriptedSource(DirectoryModelSource(remote.path));
    manager = FileModelManager(source: source, rootPath: installRoot.path);
  });

  tearDown(() => tmp.deleteSync(recursive: true));

  test('reports nothing installed on a fresh device', () async {
    expect(await manager.isInstalled(enFr), isFalse);
    expect(await manager.getModel(enFr), isNull);
    expect(await manager.installedModels(), isEmpty);
  });

  test('installs, verifies and lists a model', () async {
    final stages = <InstallStage>[];
    final progress = <int>[];
    final info = await manager.install(enFr, onProgress: (e) {
      stages.add(e.stage);
      progress.add(e.receivedBytes);
    });

    expect(info.id, 'en-fr');
    expect(info.license, 'Apache-2.0');
    expect(info.baseModel, 'Helsinki-NLP/opus-mt-en-fr');
    expect(info.architecture.decoderLayers, 6);
    expect(info.files, hasLength(4));
    expect(info.size, 4 * 2048);
    expect(info.path, endsWith('en-fr'));

    expect(stages.first, InstallStage.manifest);
    expect(stages, contains(InstallStage.downloading));
    expect(stages, contains(InstallStage.verifying));
    expect(stages.last, InstallStage.done);
    expect(progress.last, 4 * 2048);

    expect(await manager.isInstalled(enFr), isTrue);
    expect(
        (await manager.installedModels()).map((m) => m.id), <String>['en-fr']);
    await manager.verify(enFr);
  });

  test('skips a re-install of the same version', () async {
    await manager.install(enFr);
    final calls = source.fileCalls;
    await manager.install(enFr);
    expect(source.fileCalls, calls, reason: 'no files should be refetched');
  });

  test('force re-downloads even when up to date', () async {
    await manager.install(enFr);
    final calls = source.fileCalls;
    await manager.install(enFr, force: true);
    expect(source.fileCalls, greaterThan(calls));
  });

  test('installs a newer version over an older one', () async {
    await manager.install(enFr);
    final before = (await manager.getModel(enFr))!.version;
    Directory(p.join(remote.path, enFr.id)).deleteSync(recursive: true);
    buildFakeBundle(remote, enFr, version: '2.0.0', payloadSize: 4096);
    final after = await manager.install(enFr);
    expect(before, '1.0.0');
    expect(after.version, '2.0.0');
    expect(after.size, 4 * 4096);
    await manager.verify(enFr);
  });

  test('rejects a corrupted download and leaves nothing behind', () async {
    source.corruptFile = 'decoder.onnx';
    await expectLater(
        manager.install(enFr), throwsA(isA<ModelCorruptedException>()));
    expect(await manager.isInstalled(enFr), isFalse);
    expect(
        Directory(p.join(installRoot.path, 'en-fr.tmp')).existsSync(), isFalse);
  });

  test('keeps a working model when a re-install fails', () async {
    await manager.install(enFr);
    source.corruptFile = 'encoder.onnx';
    await expectLater(manager.install(enFr, force: true),
        throwsA(isA<ModelCorruptedException>()));
    expect(await manager.isInstalled(enFr), isTrue,
        reason: 'the previous install must survive a failed update');
    await manager.verify(enFr);
  });

  test('surfaces a download failure', () async {
    source.failManifest = true;
    await expectLater(
        manager.install(enFr), throwsA(isA<ModelDownloadException>()));
    source
      ..failManifest = false
      ..failFiles = true;
    await expectLater(
        manager.install(enFr), throwsA(isA<ModelDownloadException>()));
  });

  test('detects a model corrupted after installation', () async {
    final info = await manager.install(enFr);
    final file = File(p.join(info.path, 'vocab.json'));
    file.writeAsBytesSync(<int>[1, 2, 3]);
    // The cheap size check already rejects it.
    expect(await manager.getModel(enFr), isNull);
    await expectLater(
        manager.verify(enFr), throwsA(isA<ModelCorruptedException>()));
  });

  test('detects tampering that preserves the file size', () async {
    final info = await manager.install(enFr);
    final file = File(p.join(info.path, 'vocab.json'));
    final bytes = file.readAsBytesSync();
    bytes[10] = bytes[10] ^ 0xFF;
    file.writeAsBytesSync(bytes);
    expect(await manager.getModel(enFr), isNotNull,
        reason: 'the size check cannot catch this');
    await expectLater(
        manager.verify(enFr), throwsA(isA<ModelCorruptedException>()));
  });

  test('verify throws when nothing is installed', () async {
    await expectLater(
        manager.verify(enFr), throwsA(isA<ModelNotInstalledException>()));
  });

  test('deletes a model and tolerates deleting twice', () async {
    await manager.install(enFr);
    await manager.delete(enFr);
    expect(await manager.isInstalled(enFr), isFalse);
    await manager.delete(enFr);
  });

  test('lists several models in a stable order', () async {
    await manager.install(frEn);
    await manager.install(enFr);
    expect((await manager.installedModels()).map((m) => m.id).toList(),
        <String>['en-fr', 'fr-en']);
  });

  test('ignores unrelated directories in the models folder', () async {
    await manager.install(enFr);
    Directory(p.join(installRoot.path, 'not-a-pair')).createSync();
    Directory(p.join(installRoot.path, 'xx-yy')).createSync();
    File(p.join(installRoot.path, 'stray.txt')).writeAsStringSync('hi');
    expect((await manager.installedModels()).map((m) => m.id).toList(),
        <String>['en-fr']);
  });

  test('rejects a manifest that describes another direction', () async {
    final manifest = File(p.join(remote.path, enFr.id, 'manifest.json'));
    final json =
        jsonDecode(manifest.readAsStringSync()) as Map<String, dynamic>;
    json['to'] = 'de';
    manifest.writeAsStringSync(jsonEncode(json));
    await expectLater(
        manager.install(enFr), throwsA(isA<ModelDownloadException>()));
  });

  test('rejects a malformed manifest', () async {
    File(p.join(remote.path, enFr.id, 'manifest.json'))
        .writeAsStringSync('{not json');
    await expectLater(
        manager.install(enFr), throwsA(isA<ModelDownloadException>()));
  });
}
