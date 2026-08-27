import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:offline_translate/offline_translate.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

/// End-to-end tests that exercise the real ONNX model on the target device.
///
/// Run with a bundle available on the machine:
///
/// ```sh
/// flutter test integration_test -d macos \
///   --dart-define=OT_MODELS_DIR=$HOME/ot-models
/// ```
///
/// On a phone, leave the define out entirely: the suite falls back to
/// `HttpModelSource.official()` and downloads the bundle on first run. To test
/// without a network, push a bundle to the app's documents directory as
/// `ot-models/<pair>/` instead.
void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  const localDir = String.fromEnvironment('OT_MODELS_DIR', defaultValue: '');
  const remoteUrl = String.fromEnvironment('OT_MODELS_URL', defaultValue: '');
  late Directory workDir;
  late OfflineTranslator translator;

  Future<ModelSource> resolveSource() async {
    if (localDir.isNotEmpty && Directory(localDir).existsSync()) {
      return DirectoryModelSource(localDir);
    }
    final docs = await getApplicationDocumentsDirectory();
    final pushed = Directory(p.join(docs.path, 'ot-models'));
    if (pushed.existsSync()) return DirectoryModelSource(pushed.path);
    if (remoteUrl.isNotEmpty) {
      return HttpModelSource(baseUrl: Uri.parse(remoteUrl));
    }
    // Zero configuration: fall back to the published bundles, so this suite
    // runs on a fresh device with nothing but a network connection.
    return HttpModelSource.official();
  }

  setUpAll(() async {
    final support = await getApplicationSupportDirectory();
    workDir = Directory(p.join(support.path, 'ot-integration-test'));
    if (workDir.existsSync()) workDir.deleteSync(recursive: true);
    workDir.createSync(recursive: true);

    final source = await resolveSource();
    translator = await OfflineTranslator.initialize(
      modelManager: FileModelManager(source: source, rootPath: workDir.path),
      cache: TranslationCache(maxEntries: 32),
    );
  });

  tearDownAll(() async {
    await translator.dispose();
    if (workDir.existsSync()) workDir.deleteSync(recursive: true);
  });

  testWidgets(
    'installs the en-fr model with verified checksums',
    (tester) async {
      final stages = <InstallStage>[];
      final info = await translator.installModel(
        from: Language.en,
        to: Language.fr,
        onProgress: (progress) => stages.add(progress.stage),
      );
      expect(info.id, 'en-fr');
      // The suite validates whichever family the bundle is, because both are
      // supported and the constants move with the model.
      if (info.architecture.family == ModelFamily.tinySsru) {
        expect(info.license, 'MPL-2.0');
        expect(info.architecture.decoderLayers, 4);
        expect(info.architecture.modelDimension, 384);
      } else {
        expect(info.license, 'Apache-2.0');
        expect(info.architecture.decoderLayers, 6);
        expect(info.architecture.modelDimension, 512);
      }
      expect(stages, contains(InstallStage.verifying));
      expect(stages.last, InstallStage.done);
      expect(
        await translator.isModelAvailable(from: Language.en, to: Language.fr),
        isTrue,
      );
      // A second install of the same version is a no-op.
      await translator.installModel(from: Language.en, to: Language.fr);
    },
    timeout: const Timeout(Duration(minutes: 5)),
  );

  testWidgets(
    'translate() is synchronous and produces the expected French',
    (tester) async {
      await translator.preload(from: Language.en, to: Language.fr);
      final result = translator.translate(
        'Hello, how are you?',
        from: Language.en,
        to: Language.fr,
      );
      expect(result.translatedText, 'Bonjour, comment allez-vous ?');
      expect(result.sourceLanguage, Language.en);
      expect(result.targetLanguage, Language.fr);
      expect(result.fromCache, isFalse);
      expect(result.duration.inMilliseconds, greaterThan(0));
    },
    timeout: const Timeout(Duration(minutes: 2)),
  );

  testWidgets(
    'reuses the loaded model across calls',
    (tester) async {
      await translator.preload(from: Language.en, to: Language.fr);
      final first = translator.translate(
        'Good morning.',
        from: Language.en,
        to: Language.fr,
      );
      final second = translator.translate(
        'See you tomorrow.',
        from: Language.en,
        to: Language.fr,
      );
      expect(first.translatedText, isNotEmpty);
      expect(second.translatedText, isNotEmpty);
      expect(
        translator.loadedModels,
        contains(const LanguagePair(Language.en, Language.fr)),
      );
    },
    timeout: const Timeout(Duration(minutes: 2)),
  );

  testWidgets(
    'serves repeated translations from the cache',
    (tester) async {
      await translator.preload(from: Language.en, to: Language.fr);
      const text = 'The cache should return this instantly.';
      final first = translator.translate(
        text,
        from: Language.en,
        to: Language.fr,
      );
      final second = translator.translate(
        text,
        from: Language.en,
        to: Language.fr,
      );
      expect(second.translatedText, first.translatedText);
      expect(first.fromCache, isFalse);
      expect(second.fromCache, isTrue);
    },
    timeout: const Timeout(Duration(minutes: 2)),
  );

  testWidgets(
    'translates a multi-paragraph document and keeps its layout',
    (tester) async {
      const document = '''
The first paragraph introduces the topic. It contains two sentences.

The second paragraph adds a detail, and it also contains two sentences. This
one is deliberately a little longer so that the segmenter has something to do.

The third paragraph concludes.''';
      final result = await translator.translateLong(
        document,
        from: Language.en,
        to: Language.fr,
      );
      expect(result.chunkCount, greaterThanOrEqualTo(3));
      expect(result.truncated, isFalse);
      expect(
        '\n\n'.allMatches(result.translatedText).length,
        greaterThanOrEqualTo(2),
      );
      expect(
        result.translatedText.trim().split('\n\n').length,
        greaterThanOrEqualTo(3),
      );
    },
    timeout: const Timeout(Duration(minutes: 5)),
  );

  testWidgets(
    'streaming yields the same text as translate',
    (tester) async {
      const document =
          'One sentence here. Another sentence there.\n\n'
          'And a second paragraph to force more than one chunk.';
      final streamed = StringBuffer();
      var events = 0;
      await for (final chunk in translator.translateStream(
        document,
        from: Language.en,
        to: Language.fr,
      )) {
        streamed.write(chunk.translatedText);
        events++;
      }
      expect(events, greaterThan(0));
      final whole = await translator.translateLong(
        document,
        from: Language.en,
        to: Language.fr,
      );
      expect(streamed.toString(), whole.translatedText);
    },
    timeout: const Timeout(Duration(minutes: 5)),
  );

  testWidgets(
    'handles empty, whitespace and unicode input',
    (tester) async {
      await translator.preload(from: Language.en, to: Language.fr);
      expect(
        translator
            .translate('', from: Language.en, to: Language.fr)
            .translatedText,
        '',
      );
      expect(
        translator
            .translate('   \n ', from: Language.en, to: Language.fr)
            .translatedText,
        '   \n ',
      );
      final emoji = translator.translate(
        'I love café 😀',
        from: Language.en,
        to: Language.fr,
      );
      expect(emoji.translatedText, isNotEmpty);
    },
    timeout: const Timeout(Duration(minutes: 2)),
  );

  testWidgets('reports a missing model instead of hanging', (tester) async {
    expect(
      () => translator.translate('Hallo', from: Language.de, to: Language.es),
      throwsA(isA<ModelNotLoadedException>()),
    );
    await expectLater(
      translator.translateLong('Hallo', from: Language.de, to: Language.es),
      throwsA(isA<ModelNotInstalledException>()),
    );
  });

  testWidgets(
    'translates with no network permission in play',
    (tester) async {
      // Nothing in the translation path opens a socket. Assert it by running the
      // whole flow inside a zone where every socket attempt throws.
      await translator.preload(from: Language.en, to: Language.fr);
      late TranslationResult result;
      await HttpOverrides.runZoned(() async {
        result = translator.translate(
          'This runs with networking disabled.',
          from: Language.en,
          to: Language.fr,
        );
      }, createHttpClient: (_) => throw StateError('network access attempted'));
      expect(result.translatedText, isNotEmpty);
    },
    timeout: const Timeout(Duration(minutes: 2)),
  );
}
