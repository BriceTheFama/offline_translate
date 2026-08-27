import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:offline_translate/offline_translate.dart';
import 'package:path/path.dart' as p;

/// The end-to-end test: the real model, the real ONNX Runtime, the real public
/// API, on the host.
///
/// The rest of the suite runs against a fake engine, which is the right trade
/// for testing the plumbing but proves nothing about translation. This one
/// answers the only question that actually matters — does
/// `translate('Hello, how are you?')` return `Bonjour, comment allez-vous ?`
/// — and does it through `OfflineTranslator`, not through an internal.
///
/// It needs two things that are too large to keep in the repository, so it
/// skips rather than fails when they are absent:
///
/// ```sh
/// python3 tool/build_tiny_model.py --pair en-fr   # -> ~/ot-models-tiny/en-fr
/// tool/fetch_onnxruntime.sh                       # -> third_party/onnxruntime
/// flutter test test/end_to_end_test.dart
/// ```
///
/// **There is no model source.** `FileModelManager` is built without one, so
/// this test could not reach the network even if something in the stack wanted
/// to: `installModel` would throw. Everything below runs off the files on disk.
void main() {
  final modelsRoot = Platform.environment['OT_TEST_MODEL_DIR'] ??
      p.join(_home, 'ot-models-tiny');
  final runtimeLibrary = Platform.environment['OT_TEST_ORT_LIB'] ??
      p.join('third_party', 'onnxruntime', 'lib', _libraryName);

  final ready = Directory(p.join(modelsRoot, 'en-fr')).existsSync() &&
      File(runtimeLibrary).existsSync();
  final skip = ready
      ? null
      : 'needs a bundle in $modelsRoot/en-fr and a runtime at '
          '$runtimeLibrary — see the doc comment';

  group('en -> fr, on the real engine', () {
    late OfflineTranslator translator;

    setUpAll(() async {
      if (!ready) return;
      OfflineTranslator.onnxRuntimeLibraryPath =
          File(runtimeLibrary).absolute.path;
      translator = await OfflineTranslator.initialize(
        languages: const {Language.en, Language.fr},
        defaultLanguage: Language.fr,
        modelManager: FileModelManager(rootPath: modelsRoot),
      );
    });

    tearDownAll(() async {
      if (ready) await translator.dispose();
    });

    test('initialize loads the model that is installed', () {
      expect(translator.loadedModels,
          contains(const LanguagePair(Language.en, Language.fr)));
      expect((translator.modelManager as FileModelManager).source, isNull,
          reason: 'this test has no network path at all');
    }, skip: skip);

    test('translate is synchronous and returns the expected French', () {
      // No `await`. This is the shape the API promises, and the reason
      // initialize() is the only asynchronous step.
      final result = translator.translate(
        'Hello, how are you?',
        from: Language.en,
        to: Language.fr,
      );
      expect(result, isA<TranslationResult>());
      expect(result.translatedText, 'Bonjour, comment allez-vous ?');
      expect(result.sourceLanguage, Language.en);
      expect(result.targetLanguage, Language.fr);
    }, skip: skip);

    test('translate uses the declared default direction', () {
      // Two languages and a default target leave nothing to disambiguate.
      // "Bonjour monde" rather than "Bonjour le monde" is the int8 model's
      // answer; the float32 rebuild keeps the article. It is the one fixture
      // out of seven where quantisation changes the output, and it is recorded
      // here as the model's real behaviour rather than smoothed over.
      expect(
          translator.translate('Hello world').translatedText, 'Bonjour monde');
    }, skip: skip);

    test('translates the rest of the fixture sentences', () {
      const cases = <String, String>{
        'This is a simple test.': "C'est un test simple.",
        'I love programming and mobile development.':
            "J'adore la programmation et le développement mobile.",
        'The quick brown fox jumps over the lazy dog.':
            'Le renard brun rapide saute par-dessus le chien paresseux.',
      };
      cases.forEach((source, expected) {
        expect(translator.translate(source).translatedText, expected,
            reason: 'for "$source"');
      });
    }, skip: skip);

    test('the model is loaded once and reused across translations', () {
      final before = translator.loadedModels;
      for (var i = 0; i < 5; i++) {
        expect(
            translator.translate('Good morning.').translatedText, isNotEmpty);
      }
      expect(translator.loadedModels, before,
          reason: 'translating must never load or unload a model');
    }, skip: skip);

    test('empty and whitespace-only input come back unchanged', () {
      expect(translator.translate('').translatedText, '');
      expect(translator.translate('   \n ').translatedText, '   \n ');
      expect(translator.translate('').chunkCount, 0);
    }, skip: skip);

    test('input outside the vocabulary survives as UTF-8', () {
      // Byte fallback, end to end: none of these characters is a piece of an
      // English source vocabulary, and the model still copies them through.
      final result = translator.translate('Send it to 東京 by 🚀, ¿vale?');
      expect(result.translatedText, contains('東京'));
      expect(result.translatedText, contains('🚀'));
    }, skip: skip);

    test('translateLong is asynchronous and chunks a document', () async {
      const document = '''
Machine translation has changed a great deal over the last decade. Early
systems relied on hand written rules and large phrase tables.

Neural models replaced those pipelines with a single network trained end to
end. The encoder reads the source sentence and the decoder produces the target
sentence one token at a time.

Running such a model on a phone used to be out of the question. Quantisation
and smaller architectures have made it practical.
''';
      final future = translator.translateLong(document);
      expect(future, isA<Future<TranslationResult>>());
      final result = await future;
      expect(result.chunkCount, greaterThan(1),
          reason: 'a document must be split');
      expect(result.truncated, isFalse);
      // Paragraph structure is preserved, so the shape of the document
      // survives the round trip through the chunker.
      expect('\n\n'.allMatches(result.translatedText).length,
          '\n\n'.allMatches(document.trim()).length);
      expect(result.translatedText, contains('traduction'));
    }, skip: skip);

    test('a language outside the declared set is refused, not downloaded',
        () async {
      expect(
        () => translator.translate('Hello', from: Language.en, to: Language.de),
        throwsA(isA<UnsupportedLanguageException>()),
      );
    }, skip: skip);

    test('translates with every HTTP client in the process disarmed', () async {
      // The strongest offline proof that can be made without unplugging a
      // cable: for the duration of this test, constructing an HttpClient
      // anywhere in the isolate throws. `package:http` — the only networking
      // dependency this package has — goes through `HttpOverrides`, so if any
      // part of loading a model, tokenizing, running inference or reassembling
      // a document reached for the network, these calls would fail rather than
      // quietly succeed.
      final previous = HttpOverrides.current;
      HttpOverrides.global = _NoNetwork();
      try {
        expect(translator.translate('Hello, how are you?').translatedText,
            'Bonjour, comment allez-vous ?');
        final long = await translator.translateLong(
            'The network is switched off.\n\nThis is translated on the '
            'device.');
        expect(long.translatedText, isNotEmpty);
        expect(_NoNetwork.attempts, 0);
      } finally {
        HttpOverrides.global = previous;
      }
    }, skip: skip);

    test('the bundle is the tiny SSRU family and is under 35 MB', () async {
      final info = await translator.modelManager
          .getModel(const LanguagePair(Language.en, Language.fr));
      expect(info, isNotNull);
      expect(info!.architecture.family, ModelFamily.tinySsru);
      expect(info.architecture.decoderStartTokenId, -1,
          reason: 'the SSRU decoder starts from a zero embedding');
      expect(info.license, 'MPL-2.0');
      expect(info.size, lessThan(35 * 1024 * 1024));
    }, skip: skip);
  });
}

/// An `HttpOverrides` that refuses to hand out a client at all.
class _NoNetwork extends HttpOverrides {
  static int attempts = 0;

  @override
  HttpClient createHttpClient(SecurityContext? context) {
    attempts++;
    throw StateError('the network was used during an offline translation');
  }
}

String get _home =>
    Platform.environment['HOME'] ?? Platform.environment['USERPROFILE'] ?? '.';

String get _libraryName => Platform.isMacOS
    ? 'libonnxruntime.dylib'
    : (Platform.isWindows ? 'onnxruntime.dll' : 'libonnxruntime.so');
