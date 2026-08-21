import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:offline_translate/offline_translate.dart';
import 'package:path/path.dart' as p;

import 'model_manager_test.dart' show buildFakeBundle;

/// A stand-in engine that "translates" by uppercasing each word and prefixing
/// the direction. It is deterministic, instant, and records what it was asked
/// to do, which is what the orchestration tests need — the real inference path
/// is covered by `example/integration_test`.
class FakeEngine implements TranslationEngine {
  FakeEngine(this.model);

  @override
  final ModelInfo model;

  static int loadCount = 0;
  static int disposeCount = 0;
  static final List<String> generated = <String>[];

  bool _loaded = false;
  final List<String> _vocabulary = <String>[];

  @override
  bool get isLoaded => _loaded;

  @override
  Future<void> load(
      {GenerationConfig config = const GenerationConfig()}) async {
    loadCount++;
    // Pretend loading is slow enough that concurrent callers can overlap.
    await Future<void>.delayed(const Duration(milliseconds: 20));
    _loaded = true;
  }

  @override
  Int32List encodeText(String text) {
    // One token per word, plus the end-of-sequence token, mirroring the shape
    // of the real tokenizer closely enough for the segmenter.
    final words =
        text.trim().isEmpty ? <String>[] : text.trim().split(RegExp(r'\s+'));
    final ids = Int32List(words.length + 1);
    for (var i = 0; i < words.length; i++) {
      var index = _vocabulary.indexOf(words[i]);
      if (index < 0) {
        _vocabulary.add(words[i]);
        index = _vocabulary.length - 1;
      }
      ids[i] = index + 1;
    }
    return ids;
  }

  @override
  String decodeTokens(List<int> tokens) =>
      tokens.map((t) => _vocabulary[t - 1].toUpperCase()).join(' ');

  @override
  GenerationOutput generate(Int32List inputIds, GenerationConfig config) {
    if (!_loaded) throw StateError('not loaded');
    final tokens = inputIds.where((id) => id != 0).toList();
    generated.add(decodeTokens(tokens));
    return GenerationOutput(
      tokens: tokens,
      truncated: false,
      encodeMicros: 1,
      decodeMicros: tokens.length,
    );
  }

  @override
  Stream<ChunkTranslation> generateChunks(
      List<String> chunks, GenerationConfig config) async* {
    // The real engine hands this to a worker isolate; the fake stays here but
    // still yields between chunks, which is what the callers rely on.
    for (var i = 0; i < chunks.length; i++) {
      final ids = encodeText(chunks[i]);
      final out = generate(ids, config);
      yield ChunkTranslation(i, decodeTokens(out.tokens),
          truncated: out.truncated);
      await Future<void>.delayed(Duration.zero);
    }
  }

  @override
  Future<void> dispose() async {
    disposeCount++;
    _loaded = false;
  }
}

void main() {
  const enFr = LanguagePair(Language.en, Language.fr);
  const frEn = LanguagePair(Language.fr, Language.en);
  const enEs = LanguagePair(Language.en, Language.es);

  late Directory tmp;
  late FileModelManager manager;

  Future<OfflineTranslator> makeTranslator({
    Language? from,
    Language? to,
    TranslationCache? cache,
    int maxLoadedModels = 2,
    GenerationConfig config = const GenerationConfig(),
  }) =>
      OfflineTranslator.initialize(
        from: from,
        to: to,
        modelManager: manager,
        cache: cache,
        maxLoadedModels: maxLoadedModels,
        generationConfig: config,
        engineFactory: FakeEngine.new,
      );

  setUp(() async {
    FakeEngine.loadCount = 0;
    FakeEngine.disposeCount = 0;
    FakeEngine.generated.clear();
    tmp = Directory.systemTemp.createTempSync('ot-api-test');
    final remote = Directory(p.join(tmp.path, 'remote'))..createSync();
    for (final pair in <LanguagePair>[enFr, frEn, enEs]) {
      buildFakeBundle(remote, pair);
    }
    manager = FileModelManager(
      source: DirectoryModelSource(remote.path),
      rootPath: p.join(tmp.path, 'installed'),
    );
    for (final pair in <LanguagePair>[enFr, frEn, enEs]) {
      await manager.install(pair);
    }
  });

  tearDown(() => tmp.deleteSync(recursive: true));

  group('initialize', () {
    test('requires a source or a manager', () {
      expect(OfflineTranslator.initialize, throwsA(isA<ArgumentError>()));
    });

    test('rejects a half-specified default direction', () {
      expect(
        () => OfflineTranslator.initialize(
            from: Language.en, modelManager: manager),
        throwsA(isA<ArgumentError>()),
      );
    });

    test('preloads the direction it is given', () async {
      final translator =
          await makeTranslator(from: Language.en, to: Language.fr);
      expect(FakeEngine.loadCount, 1);
      expect(translator.loadedModels, <LanguagePair>[enFr]);
      await translator.dispose();
    });

    test('loads nothing when no direction is given', () async {
      final translator = await makeTranslator();
      expect(FakeEngine.loadCount, 0);
      expect(translator.loadedModels, isEmpty);
      await translator.dispose();
    });
  });

  group('translateSync', () {
    test('translates a short text without a Future', () async {
      final translator =
          await makeTranslator(from: Language.en, to: Language.fr);
      final result = translator.translateSync(text: 'hello world');
      expect(result, isA<TranslationResult>());
      expect(result.translatedText, 'HELLO WORLD');
      expect(result.sourceText, 'hello world');
      expect(result.sourceLanguage, Language.en);
      expect(result.targetLanguage, Language.fr);
      expect(result.chunkCount, 1);
      expect(result.fromCache, isFalse);
      await translator.dispose();
    });

    test('throws when the model is not loaded', () async {
      final translator = await makeTranslator();
      expect(
        () => translator.translateSync(
            text: 'hello', from: Language.en, to: Language.fr),
        throwsA(isA<ModelNotLoadedException>()),
      );
      await translator.dispose();
    });

    test('reuses the loaded model across calls', () async {
      final translator =
          await makeTranslator(from: Language.en, to: Language.fr);
      translator.translateSync(text: 'one');
      translator.translateSync(text: 'two');
      translator.translateSync(text: 'three');
      expect(FakeEngine.loadCount, 1, reason: 'the model must load once');
      await translator.dispose();
    });

    test('returns whitespace-only input unchanged', () async {
      final translator =
          await makeTranslator(from: Language.en, to: Language.fr);
      expect(translator.translateSync(text: '').translatedText, '');
      expect(translator.translateSync(text: '  \n ').translatedText, '  \n ');
      expect(translator.translateSync(text: '').chunkCount, 0);
      await translator.dispose();
    });

    test('chunks text that exceeds the input window', () async {
      final translator = await makeTranslator(
        from: Language.en,
        to: Language.fr,
        config: const GenerationConfig(maxInputTokens: 4),
      );
      final result = translator.translateSync(
          text: 'one two three. four five six. seven eight nine.');
      expect(result.chunkCount, greaterThan(1));
      expect(result.translatedText, contains('ONE'));
      expect(result.translatedText, contains('NINE'));
      await translator.dispose();
    });
  });

  group('translate', () {
    test('loads the model on demand', () async {
      final translator = await makeTranslator();
      final result = await translator.translate(
          text: 'hello', from: Language.en, to: Language.fr);
      expect(result.translatedText, 'HELLO');
      expect(FakeEngine.loadCount, 1);
      await translator.dispose();
    });

    test('preserves paragraph structure', () async {
      final translator = await makeTranslator(
        from: Language.en,
        to: Language.fr,
        config: const GenerationConfig(maxInputTokens: 3),
      );
      final result = await translator.translate(text: 'alpha\n\nbeta\n\ngamma');
      expect(result.translatedText, 'ALPHA\n\nBETA\n\nGAMMA');
      expect(result.chunkCount, 3);
      await translator.dispose();
    });

    test('is safe to call concurrently for the same direction', () async {
      final translator = await makeTranslator();
      final results =
          await Future.wait<TranslationResult>(<Future<TranslationResult>>[
        translator.translate(text: 'one', from: Language.en, to: Language.fr),
        translator.translate(text: 'two', from: Language.en, to: Language.fr),
        translator.translate(text: 'three', from: Language.en, to: Language.fr),
      ]);
      expect(results.map((r) => r.translatedText).toList(),
          <String>['ONE', 'TWO', 'THREE']);
      expect(FakeEngine.loadCount, 1,
          reason: 'concurrent callers must share one load');
      await translator.dispose();
    });

    test('throws when the model is not installed', () async {
      await manager.delete(enEs);
      final translator = await makeTranslator();
      await expectLater(
        translator.translate(text: 'hi', from: Language.en, to: Language.es),
        throwsA(isA<ModelNotInstalledException>()),
      );
      await translator.dispose();
    });
  });

  group('translateStream', () {
    test('emits one event per chunk and concatenates to translate()', () async {
      final translator = await makeTranslator(
        from: Language.en,
        to: Language.fr,
        config: const GenerationConfig(maxInputTokens: 3),
      );
      const text = 'alpha\n\nbeta\n\ngamma';
      final buffer = StringBuffer();
      var events = 0;
      await for (final chunk in translator.translateStream(text: text)) {
        buffer.write(chunk.translatedText);
        events++;
      }
      expect(events, 3);
      final whole = await translator.translate(text: text);
      expect(buffer.toString(), whole.translatedText);
      await translator.dispose();
    });

    test('emits nothing for blank input', () async {
      final translator =
          await makeTranslator(from: Language.en, to: Language.fr);
      expect(await translator.translateStream(text: '   ').toList(), isEmpty);
      await translator.dispose();
    });
  });

  group('cache', () {
    test('serves an identical request without touching the engine', () async {
      final translator = await makeTranslator(
        from: Language.en,
        to: Language.fr,
        cache: TranslationCache(maxEntries: 8),
      );
      translator.translateSync(text: 'hello');
      final calls = FakeEngine.generated.length;
      final second = translator.translateSync(text: 'hello');
      expect(second.fromCache, isTrue);
      expect(FakeEngine.generated.length, calls);
      await translator.dispose();
    });

    test('is bypassed when no cache is configured', () async {
      final translator =
          await makeTranslator(from: Language.en, to: Language.fr);
      translator.translateSync(text: 'hello');
      final second = translator.translateSync(text: 'hello');
      expect(second.fromCache, isFalse);
      expect(FakeEngine.generated.length, 2);
      await translator.dispose();
    });

    test('is dropped for a direction when its model is deleted', () async {
      final translator = await makeTranslator(
        from: Language.en,
        to: Language.fr,
        cache: TranslationCache(),
      );
      translator.translateSync(text: 'hello');
      expect(translator.cache!.length, 1);
      await translator.deleteModel(from: Language.en, to: Language.fr);
      expect(translator.cache!.isEmpty, isTrue);
      await translator.dispose();
    });
  });

  group('model residency', () {
    test('evicts the least recently used model past the limit', () async {
      final translator = await makeTranslator(maxLoadedModels: 2);
      await translator.preload(from: Language.en, to: Language.fr);
      await translator.preload(from: Language.fr, to: Language.en);
      expect(translator.loadedModels, hasLength(2));

      // Touch en-fr so fr-en becomes the eviction candidate.
      await translator.translate(text: 'x', from: Language.en, to: Language.fr);
      await translator.preload(from: Language.en, to: Language.es);

      expect(translator.loadedModels, hasLength(2));
      expect(translator.loadedModels, contains(enFr));
      expect(translator.loadedModels, contains(enEs));
      expect(translator.loadedModels, isNot(contains(frEn)));
      expect(FakeEngine.disposeCount, 1);
      await translator.dispose();
    });

    test('does not load the whole catalogue', () async {
      final translator = await makeTranslator(maxLoadedModels: 1);
      await translator.preload(from: Language.en, to: Language.fr);
      await translator.preload(from: Language.fr, to: Language.en);
      await translator.preload(from: Language.en, to: Language.es);
      expect(translator.loadedModels, hasLength(1));
      await translator.dispose();
    });

    test('unload frees memory but keeps the files', () async {
      final translator =
          await makeTranslator(from: Language.en, to: Language.fr);
      await translator.unload(from: Language.en, to: Language.fr);
      expect(translator.loadedModels, isEmpty);
      expect(
          await translator.isModelAvailable(from: Language.en, to: Language.fr),
          isTrue);
      await translator.dispose();
    });

    test('deleteModel unloads and removes the files', () async {
      final translator =
          await makeTranslator(from: Language.en, to: Language.fr);
      await translator.deleteModel(from: Language.en, to: Language.fr);
      expect(translator.loadedModels, isEmpty);
      expect(
          await translator.isModelAvailable(from: Language.en, to: Language.fr),
          isFalse);
      await translator.dispose();
    });
  });

  group('dispose', () {
    test('releases every engine', () async {
      final translator = await makeTranslator();
      await translator.preload(from: Language.en, to: Language.fr);
      await translator.preload(from: Language.fr, to: Language.en);
      await translator.dispose();
      expect(FakeEngine.disposeCount, 2);
    });

    test('is idempotent', () async {
      final translator =
          await makeTranslator(from: Language.en, to: Language.fr);
      await translator.dispose();
      await translator.dispose();
      expect(FakeEngine.disposeCount, 1);
    });

    test('rejects further use', () async {
      final translator =
          await makeTranslator(from: Language.en, to: Language.fr);
      await translator.dispose();
      expect(() => translator.translateSync(text: 'hi'),
          throwsA(isA<TranslatorDisposedException>()));
      await expectLater(translator.translate(text: 'hi'),
          throwsA(isA<TranslatorDisposedException>()));
      await expectLater(translator.installModel(),
          throwsA(isA<TranslatorDisposedException>()));
    });

    test('a fresh translator works after the previous one was disposed',
        () async {
      final first = await makeTranslator(from: Language.en, to: Language.fr);
      first.translateSync(text: 'hello');
      await first.dispose();

      final second = await makeTranslator(from: Language.en, to: Language.fr);
      expect(second.translateSync(text: 'hello').translatedText, 'HELLO');
      await second.dispose();
    });
  });

  group('direction resolution', () {
    test('falls back to the default direction', () async {
      final translator =
          await makeTranslator(from: Language.en, to: Language.fr);
      expect(translator.translateSync(text: 'hi').targetLanguage, Language.fr);
      await translator.dispose();
    });

    test('requires an explicit direction when there is no default', () async {
      final translator = await makeTranslator();
      expect(() => translator.translateSync(text: 'hi'),
          throwsA(isA<ArgumentError>()));
      await translator.dispose();
    });
  });
}
