// Verifies the published model bundles, end to end, exactly as an application
// would consume them.
//
//   flutter test tool/verify_published_test.dart
//
// This is the one check that cannot be done offline, and the only one that
// exercises the whole published path: it downloads from
// `HttpModelSource.official()` into a throwaway directory, verifies every file
// against the SHA-256 in its manifest, loads the model, translates, and then
// proves the result still holds with the network taken away.
//
// It lives in `tool/` rather than `test/` on purpose: `flutter test` must never
// depend on a network or on a third-party host being up, so this one has to be
// asked for by name.
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:offline_translate/offline_translate.dart';
import 'package:path/path.dart' as p;

void main() {
  const pair = LanguagePair(Language.en, Language.fr);
  final runtimeLibrary = Platform.environment['OT_TEST_ORT_LIB'] ??
      p.join('third_party', 'onnxruntime', 'lib', _libraryName);

  late Directory root;
  late OfflineTranslator translator;
  late ModelInfo info;

  setUpAll(() async {
    OfflineTranslator.onnxRuntimeLibraryPath =
        File(runtimeLibrary).absolute.path;
    root = Directory.systemTemp.createTempSync('ot-published-');
    translator = await OfflineTranslator.initialize(
      languages: const {Language.en, Language.fr},
      defaultLanguage: Language.fr,
      modelManager: FileModelManager(
          source: HttpModelSource.official(), rootPath: root.path),
    );
  });

  tearDownAll(() async {
    await translator.dispose();
    if (root.existsSync()) root.deleteSync(recursive: true);
  });

  test('installs en-fr from the published repository', () async {
    final stages = <InstallStage>[];
    info = await translator.installModel(
      from: Language.en,
      to: Language.fr,
      onProgress: (progress) => stages.add(progress.stage),
    );
    expect(info.id, 'en-fr');
    expect(info.license, 'MPL-2.0');
    expect(info.architecture.family, ModelFamily.tinySsru);
    expect(info.size, lessThan(35 * 1024 * 1024));
    expect(stages, contains(InstallStage.verifying));
    expect(stages.last, InstallStage.done);
  }, timeout: const Timeout(Duration(minutes: 10)));

  test('every downloaded file matches its manifest checksum', () async {
    // install() already checked this on the way in; this re-runs it against
    // what actually landed on disk.
    await translator.modelManager.verify(pair);
    for (final file in info.files) {
      expect(File(p.join(info.path, file.name)).lengthSync(), file.size,
          reason: file.name);
    }
  });

  test('translates, with every HTTP client in the process disarmed', () async {
    await translator.preload(from: Language.en, to: Language.fr);
    final previous = HttpOverrides.current;
    HttpOverrides.global = _NoNetwork();
    try {
      const expected = <String, String>{
        'Hello, how are you?': 'Bonjour, comment allez-vous ?',
        'This is a simple test.': "C'est un test simple.",
        'I love programming and mobile development.':
            "J'adore la programmation et le développement mobile.",
      };
      expected.forEach((source, target) {
        expect(translator.translate(source).translatedText, target,
            reason: source);
      });
      final long = await translator.translateLong(
          'The network is switched off.\n\nThis is translated on the device.');
      expect(long.chunkCount, greaterThan(1));
      expect(long.translatedText, isNotEmpty);
    } finally {
      HttpOverrides.global = previous;
    }
  });
}

class _NoNetwork extends HttpOverrides {
  @override
  HttpClient createHttpClient(SecurityContext? context) =>
      throw StateError('the network was used after installation');
}

String get _libraryName => Platform.isMacOS
    ? 'libonnxruntime.dylib'
    : (Platform.isWindows ? 'onnxruntime.dll' : 'libonnxruntime.so');
