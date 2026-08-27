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
    Set<Language>? languages,
    Language? defaultLanguage,
    TranslationCache? cache,
    int maxLoadedModels = 2,
    GenerationConfig config = const GenerationConfig(),
  }) =>
      OfflineTranslator.initialize(
        languages: languages,
        defaultLanguage: defaultLanguage,
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
    test('rejects an empty language set', () {
      expect(
        () => OfflineTranslator.initialize(
            languages: const <Language>{}, modelManager: manager),
        throwsA(isA<ArgumentError>()),
      );
    });

    test('rejects a default language outside the declared set', () {
      expect(
        () => OfflineTranslator.initialize(
          languages: const {Language.en, Language.fr},
          defaultLanguage: Language.de,
          modelManager: manager,
        ),
        throwsA(isA<ArgumentError>()),
      );
    });

    test('preloads every installed direction inside the language set',
        () async {
      final translator = await makeTranslator(
          languages: const {Language.en, Language.fr},
          defaultLanguage: Language.fr);
      // Both directions of a two-language set, and the default first.
      expect(FakeEngine.loadCount, 2);
      expect(translator.loadedModels, <LanguagePair>[enFr, frEn]);
      await translator.dispose();
    });

    test('never loads more than maxLoadedModels', () async {
      final translator = await makeTranslator(
        languages: const {Language.en, Language.fr, Language.es},
        maxLoadedModels: 1,
      );
      expect(FakeEngine.loadCount, 1);
      await translator.dispose();
    });

    test('skips directions whose model is not installed', () async {
      await manager.delete(frEn);
      final translator = await makeTranslator(
          languages: const {Language.en, Language.fr},
          defaultLanguage: Language.fr);
      expect(translator.loadedModels, <LanguagePair>[enFr]);
      await translator.dispose();
    });

    test('loads nothing when no languages are declared', () async {
      final translator = await makeTranslator();
      expect(FakeEngine.loadCount, 0);
      expect(translator.loadedModels, isEmpty);
      await translator.dispose();
    });

    test('a declared set makes every other language an error', () async {
      final translator = await makeTranslator(
          languages: const {Language.en, Language.fr},
          defaultLanguage: Language.fr);
      expect(
        () => translator.translate('hi', from: Language.en, to: Language.es),
        throwsA(isA<UnsupportedLanguageException>()),
      );
      await expectLater(
        translator.translateLong('hi', from: Language.es, to: Language.fr),
        throwsA(isA<UnsupportedLanguageException>()),
      );
      // ...and the model for it was never needed.
      expect(FakeEngine.loadCount, 2);
      await translator.dispose();
    });
  });

  group('translate (synchronous)', () {
    test('translates a short text without a Future', () async {
      final translator = await makeTranslator(
          languages: const {Language.en, Language.fr},
          defaultLanguage: Language.fr);
      final result = translator.translate('hello world');
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
        () => translator.translate('hello', from: Language.en, to: Language.fr),
        throwsA(isA<ModelNotLoadedException>()),
      );
      await translator.dispose();
    });

    test('reuses the loaded model across calls', () async {
      final translator = await makeTranslator(
          languages: const {Language.en, Language.fr},
          defaultLanguage: Language.fr);
      final loadsAfterInitialize = FakeEngine.loadCount;
      translator.translate('one');
      translator.translate('two');
      translator.translate('three');
      expect(FakeEngine.loadCount, loadsAfterInitialize,
          reason: 'translating must never load a model');
      await translator.dispose();
    });

    test('returns whitespace-only input unchanged', () async {
      final translator = await makeTranslator(
          languages: const {Language.en, Language.fr},
          defaultLanguage: Language.fr);
      expect(translator.translate('').translatedText, '');
      expect(translator.translate('  \n ').translatedText, '  \n ');
      expect(translator.translate('').chunkCount, 0);
      await translator.dispose();
    });

    test('chunks text that exceeds the input window', () async {
      final translator = await makeTranslator(
        languages: const {Language.en, Language.fr},
        defaultLanguage: Language.fr,
        config: const GenerationConfig(maxInputTokens: 4),
      );
      final result = translator
          .translate('one two three. four five six. seven eight nine.');
      expect(result.chunkCount, greaterThan(1));
      expect(result.translatedText, contains('ONE'));
      expect(result.translatedText, contains('NINE'));
      await translator.dispose();
    });
  });

  group('translate', () {
    test('loads the model on demand', () async {
      final translator = await makeTranslator();
      final result = await translator.translateLong('hello',
          from: Language.en, to: Language.fr);
      expect(result.translatedText, 'HELLO');
      expect(FakeEngine.loadCount, 1);
      await translator.dispose();
    });

    test('preserves paragraph structure', () async {
      final translator = await makeTranslator(
        languages: const {Language.en, Language.fr},
        defaultLanguage: Language.fr,
        config: const GenerationConfig(maxInputTokens: 3),
      );
      final result = await translator.translateLong('alpha\n\nbeta\n\ngamma');
      expect(result.translatedText, 'ALPHA\n\nBETA\n\nGAMMA');
      expect(result.chunkCount, 3);
      await translator.dispose();
    });

    test('is safe to call concurrently for the same direction', () async {
      final translator = await makeTranslator();
      final results =
          await Future.wait<TranslationResult>(<Future<TranslationResult>>[
        translator.translateLong('one', from: Language.en, to: Language.fr),
        translator.translateLong('two', from: Language.en, to: Language.fr),
        translator.translateLong('three', from: Language.en, to: Language.fr),
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
        translator.translateLong('hi', from: Language.en, to: Language.es),
        throwsA(isA<ModelNotInstalledException>()),
      );
      await translator.dispose();
    });
  });

  group('translateStream', () {
    test('emits one event per chunk and concatenates to translate()', () async {
      final translator = await makeTranslator(
        languages: const {Language.en, Language.fr},
        defaultLanguage: Language.fr,
        config: const GenerationConfig(maxInputTokens: 3),
      );
      const text = 'alpha\n\nbeta\n\ngamma';
      final buffer = StringBuffer();
      var events = 0;
      await for (final chunk in translator.translateStream(text)) {
        buffer.write(chunk.translatedText);
        events++;
      }
      expect(events, 3);
      final whole = await translator.translateLong(text);
      expect(buffer.toString(), whole.translatedText);
      await translator.dispose();
    });

    test('emits nothing for blank input', () async {
      final translator = await makeTranslator(
          languages: const {Language.en, Language.fr},
          defaultLanguage: Language.fr);
      expect(await translator.translateStream('   ').toList(), isEmpty);
      await translator.dispose();
    });
  });

  group('cache', () {
    test('serves an identical request without touching the engine', () async {
      final translator = await makeTranslator(
        languages: const {Language.en, Language.fr},
        defaultLanguage: Language.fr,
        cache: TranslationCache(maxEntries: 8),
      );
      translator.translate('hello');
      final calls = FakeEngine.generated.length;
      final second = translator.translate('hello');
      expect(second.fromCache, isTrue);
      expect(FakeEngine.generated.length, calls);
      await translator.dispose();
    });

    test('is bypassed when no cache is configured', () async {
      final translator = await makeTranslator(
          languages: const {Language.en, Language.fr},
          defaultLanguage: Language.fr);
      translator.translate('hello');
      final second = translator.translate('hello');
      expect(second.fromCache, isFalse);
      expect(FakeEngine.generated.length, 2);
      await translator.dispose();
    });

    test('is dropped for a direction when its model is deleted', () async {
      final translator = await makeTranslator(
        languages: const {Language.en, Language.fr},
        defaultLanguage: Language.fr,
        cache: TranslationCache(),
      );
      translator.translate('hello');
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
      await translator.translateLong('x', from: Language.en, to: Language.fr);
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
      final translator = await makeTranslator(
          languages: const {Language.en, Language.fr},
          defaultLanguage: Language.fr);
      await translator.unload(from: Language.en, to: Language.fr);
      expect(translator.loadedModels, isNot(contains(enFr)));
      expect(
          await translator.isModelAvailable(from: Language.en, to: Language.fr),
          isTrue);
      await translator.dispose();
    });

    test('deleteModel unloads and removes the files', () async {
      final translator = await makeTranslator(
          languages: const {Language.en, Language.fr},
          defaultLanguage: Language.fr);
      await translator.deleteModel(from: Language.en, to: Language.fr);
      expect(translator.loadedModels, isNot(contains(enFr)));
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
      final translator = await makeTranslator(
          languages: const {Language.en, Language.fr},
          defaultLanguage: Language.fr);
      final loaded = FakeEngine.loadCount;
      await translator.dispose();
      await translator.dispose();
      expect(FakeEngine.disposeCount, loaded,
          reason: 'every engine is released exactly once');
    });

    test('rejects further use', () async {
      final translator = await makeTranslator(
          languages: const {Language.en, Language.fr},
          defaultLanguage: Language.fr);
      await translator.dispose();
      expect(() => translator.translate('hi'),
          throwsA(isA<TranslatorDisposedException>()));
      await expectLater(translator.translateLong('hi'),
          throwsA(isA<TranslatorDisposedException>()));
      await expectLater(translator.installModel(),
          throwsA(isA<TranslatorDisposedException>()));
    });

    test('a fresh translator works after the previous one was disposed',
        () async {
      final first = await makeTranslator(
          languages: const {Language.en, Language.fr},
          defaultLanguage: Language.fr);
      first.translate('hello');
      await first.dispose();

      final second = await makeTranslator(
          languages: const {Language.en, Language.fr},
          defaultLanguage: Language.fr);
      expect(second.translate('hello').translatedText, 'HELLO');
      await second.dispose();
    });
  });

  group('direction resolution', () {
    test('falls back to the default direction', () async {
      final translator = await makeTranslator(
          languages: const {Language.en, Language.fr},
          defaultLanguage: Language.fr);
      expect(translator.translate('hi').targetLanguage, Language.fr);
      await translator.dispose();
    });

    test('requires an explicit direction when there is no default', () async {
      final translator = await makeTranslator();
      expect(() => translator.translate('hi'), throwsA(isA<ArgumentError>()));
      await translator.dispose();
    });
  });
}
