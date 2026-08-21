import 'dart:async';

import 'package:meta/meta.dart';

import '../cache/translation_cache.dart';
import '../engine/onnx_marian_engine.dart';
import '../engine/translation_engine.dart';
import '../exceptions/exceptions.dart';
import '../model_manager/model_manager.dart';
import '../model_manager/model_source.dart';
import '../utils/text_segmenter.dart';
import 'generation_config.dart';
import 'language.dart';
import 'runtime_config.dart';
import 'model_info.dart';
import 'translation_result.dart';

/// Builds the engine for an installed model. Overridable for tests.
typedef EngineFactory = TranslationEngine Function(ModelInfo model);

/// The entry point of the package.
///
/// One instance owns a [ModelManager], a bounded pool of loaded engines and an
/// optional [TranslationCache]. Models are loaded once and reused; nothing here
/// touches the network except [installModel].
///
/// ```dart
/// final translator = await OfflineTranslator.initialize(
///   modelSource: HttpModelSource(baseUrl: Uri.parse('https://cdn.example.com/models')),
/// );
/// await translator.installModel(from: Language.en, to: Language.fr);
///
/// final short = translator.translateSync(
///   text: 'Hello, how are you?',
///   from: Language.en,
///   to: Language.fr,
/// );
///
/// final long = await translator.translate(
///   text: article,
///   from: Language.en,
///   to: Language.fr,
/// );
///
/// await translator.dispose();
/// ```
class OfflineTranslator {
  OfflineTranslator._({
    required this.modelManager,
    required this.generationConfig,
    required this.runtimeConfig,
    required this.maxLoadedModels,
    required TranslationCache? cache,
    required EngineFactory engineFactory,
    required LanguagePair? defaultPair,
  })  : _cache = cache,
        _engineFactory = engineFactory,
        _defaultPair = defaultPair;

  /// Manages installed models on disk.
  final ModelManager modelManager;

  /// Decoder settings applied to every translation.
  final GenerationConfig generationConfig;

  /// Inference backend settings, applied when a model is loaded.
  ///
  /// The defaults are what `doc/performance.md` was measured with. See
  /// [RuntimeConfig.lowMemory] before reaching for it: most of what looks
  /// tunable there is measured to change nothing, or to make things worse.
  final RuntimeConfig runtimeConfig;

  /// How many models may stay resident at once. The least recently used engine
  /// is unloaded when the limit is exceeded.
  final int maxLoadedModels;

  final TranslationCache? _cache;
  final EngineFactory _engineFactory;
  final LanguagePair? _defaultPair;

  static const TextSegmenter _segmenter = TextSegmenter();

  /// Loaded engines, ordered least-recently-used first.
  final Map<String, TranslationEngine> _engines = <String, TranslationEngine>{};
  final Map<String, Future<TranslationEngine>> _loading =
      <String, Future<TranslationEngine>>{};

  bool _disposed = false;

  /// Creates a translator and, when [from] and [to] are given, preloads that
  /// direction so the first [translateSync] does not pay the load cost.
  ///
  /// [modelSource] tells the manager where model bundles come from; it is only
  /// consulted by [installModel]. Pass [modelManager] to supply your own.
  static Future<OfflineTranslator> initialize({
    Language? from,
    Language? to,
    ModelSource? modelSource,
    ModelManager? modelManager,
    GenerationConfig generationConfig = const GenerationConfig(),
    RuntimeConfig runtimeConfig = const RuntimeConfig(),
    TranslationCache? cache,
    int maxLoadedModels = 2,
    @visibleForTesting EngineFactory? engineFactory,
  }) async {
    if (modelManager == null && modelSource == null) {
      throw ArgumentError(
          'Provide either a modelSource or a modelManager to initialize().');
    }
    if ((from == null) != (to == null)) {
      throw ArgumentError('Pass both `from` and `to`, or neither.');
    }
    final pair = from != null && to != null ? LanguagePair(from, to) : null;
    final translator = OfflineTranslator._(
      modelManager: modelManager ?? FileModelManager(source: modelSource!),
      generationConfig: generationConfig,
      runtimeConfig: runtimeConfig,
      maxLoadedModels: maxLoadedModels,
      cache: cache,
      engineFactory: engineFactory ??
          (info) => OnnxMarianEngine(info, runtimeConfig: runtimeConfig),
      defaultPair: pair,
    );
    if (pair != null) await translator._engineFor(pair);
    return translator;
  }

  /// The translation cache, when one was supplied to [initialize].
  TranslationCache? get cache => _cache;

  /// Directions currently held in memory.
  List<LanguagePair> get loadedModels => _engines.keys
      .map((id) => _engines[id]!.model.pair)
      .toList(growable: false);

  // ---------------------------------------------------------------- models --

  /// Installs the model for the given direction. Needs network access when the
  /// configured source is remote; afterwards translation is fully offline.
  Future<ModelInfo> installModel({
    Language? from,
    Language? to,
    void Function(InstallProgress progress)? onProgress,
    bool force = false,
  }) async {
    _checkNotDisposed();
    return modelManager.install(_resolvePair(from, to),
        onProgress: onProgress, force: force);
  }

  /// Whether the model for the given direction is installed.
  Future<bool> isModelAvailable({Language? from, Language? to}) async {
    _checkNotDisposed();
    return modelManager.isInstalled(_resolvePair(from, to));
  }

  /// Unloads and deletes the model for the given direction.
  Future<void> deleteModel({Language? from, Language? to}) async {
    _checkNotDisposed();
    final pair = _resolvePair(from, to);
    await unload(from: pair.from, to: pair.to);
    _cache?.clear(pair: pair);
    await modelManager.delete(pair);
  }

  /// Lists installed models.
  Future<List<ModelInfo>> installedModels() async {
    _checkNotDisposed();
    return modelManager.installedModels();
  }

  /// Loads the model for the given direction into memory ahead of time.
  Future<void> preload({Language? from, Language? to}) async {
    _checkNotDisposed();
    await _engineFor(_resolvePair(from, to));
  }

  /// Releases the in-memory engine for a direction, keeping the files on disk.
  Future<void> unload({Language? from, Language? to}) async {
    final pair = _resolvePair(from, to);
    final engine = _engines.remove(pair.id);
    await engine?.dispose();
  }

  // ----------------------------------------------------------- translation --

  /// Translates a short text on the calling isolate and returns immediately.
  ///
  /// The model must already be loaded — call [initialize] with a direction, or
  /// [preload], or await one [translate] first; otherwise this throws
  /// [ModelNotInstalledException] rather than blocking on I/O.
  ///
  /// Inference runs synchronously on the caller's thread. On a phone that is
  /// roughly 10-25 ms per generated token, so a short sentence costs tens of
  /// milliseconds — fine from a button handler, but see [translate] for
  /// anything longer than a sentence or two. Text that does not fit the model's
  /// input window is chunked and the chunks are translated in sequence, still
  /// synchronously.
  TranslationResult translateSync({
    required String text,
    Language? from,
    Language? to,
  }) {
    _checkNotDisposed();
    final pair = _resolvePair(from, to);
    final engine = _engines[pair.id];
    if (engine == null || !engine.isLoaded) {
      throw ModelNotLoadedException(pair.id);
    }
    final watch = Stopwatch()..start();

    final cached = _cache?.get(pair, text);
    if (cached != null) {
      return TranslationResult(
        sourceText: text,
        translatedText: cached,
        sourceLanguage: pair.from,
        targetLanguage: pair.to,
        duration: watch.elapsed,
        fromCache: true,
      );
    }
    if (text.trim().isEmpty) {
      return TranslationResult(
        sourceText: text,
        translatedText: text,
        sourceLanguage: pair.from,
        targetLanguage: pair.to,
        duration: watch.elapsed,
        chunkCount: 0,
      );
    }

    final chunks = _chunk(engine, text);
    final parts = <String>[];
    var truncated = false;
    for (final chunk in chunks) {
      final out = _translateChunk(engine, chunk.text);
      parts.add(out.text);
      truncated |= out.truncated;
    }
    final translated = _segmenter.reassemble(chunks, parts);
    _cache?.put(pair, text, translated);

    return TranslationResult(
      sourceText: text,
      translatedText: translated,
      sourceLanguage: pair.from,
      targetLanguage: pair.to,
      duration: watch.elapsed,
      chunkCount: chunks.length,
      truncated: truncated,
    );
  }

  /// Translates text of any length without blocking the caller between chunks.
  ///
  /// Loads the model on demand, splits the text along paragraph and sentence
  /// boundaries, and yields to the event loop between chunks so the UI keeps
  /// running. Each individual chunk is still one synchronous inference; use
  /// [translateStream] when you want output as it is produced.
  Future<TranslationResult> translate({
    required String text,
    Language? from,
    Language? to,
  }) async {
    _checkNotDisposed();
    final pair = _resolvePair(from, to);
    final watch = Stopwatch()..start();

    final cached = _cache?.get(pair, text);
    if (cached != null) {
      return TranslationResult(
        sourceText: text,
        translatedText: cached,
        sourceLanguage: pair.from,
        targetLanguage: pair.to,
        duration: watch.elapsed,
        fromCache: true,
      );
    }

    final engine = await _engineFor(pair);
    if (text.trim().isEmpty) {
      return TranslationResult(
        sourceText: text,
        translatedText: text,
        sourceLanguage: pair.from,
        targetLanguage: pair.to,
        duration: watch.elapsed,
        chunkCount: 0,
      );
    }

    final chunks = _chunk(engine, text);
    final parts = List<String>.filled(chunks.length, '');
    var truncated = false;
    // Inference runs on a worker isolate attached to the same native sessions,
    // so the UI isolate stays free for the whole document.
    await for (final chunk in engine.generateChunks(
        chunks.map((c) => c.text).toList(growable: false), generationConfig)) {
      parts[chunk.index] = chunk.text;
      truncated |= chunk.truncated;
    }
    final translated = _segmenter.reassemble(chunks, parts);
    _cache?.put(pair, text, translated);

    return TranslationResult(
      sourceText: text,
      translatedText: translated,
      sourceLanguage: pair.from,
      targetLanguage: pair.to,
      duration: watch.elapsed,
      chunkCount: chunks.length,
      truncated: truncated,
    );
  }

  /// Streams the translation chunk by chunk, so long documents can be shown as
  /// they are produced.
  ///
  /// Each event carries the translation of one chunk together with the source
  /// text it came from, and the separator that followed it in the source is
  /// already appended, so concatenating [TranslationResult.translatedText]
  /// across the stream reproduces exactly what [translate] would return.
  Stream<TranslationResult> translateStream({
    required String text,
    Language? from,
    Language? to,
  }) async* {
    _checkNotDisposed();
    final pair = _resolvePair(from, to);
    final engine = await _engineFor(pair);
    if (text.trim().isEmpty) return;

    final chunks = _chunk(engine, text);
    final watch = Stopwatch()..start();
    var previous = Duration.zero;
    await for (final chunk in engine.generateChunks(
        chunks.map((c) => c.text).toList(growable: false), generationConfig)) {
      final elapsed = watch.elapsed;
      yield TranslationResult(
        sourceText: chunks[chunk.index].text,
        translatedText: chunk.text + chunks[chunk.index].separator,
        sourceLanguage: pair.from,
        targetLanguage: pair.to,
        duration: elapsed - previous,
        truncated: chunk.truncated,
      );
      previous = elapsed;
    }
  }

  // ----------------------------------------------------------- internals ----

  List<TextChunk> _chunk(TranslationEngine engine, String text) =>
      _segmenter.split(
        text,
        maxTokens: generationConfig.maxInputTokens,
        countTokens: (s) => engine.encodeText(s).length,
      );

  ({String text, bool truncated}) _translateChunk(
      TranslationEngine engine, String text) {
    final ids = engine.encodeText(text);
    final output = engine.generate(ids, generationConfig);
    return (
      text: engine.decodeTokens(output.tokens),
      truncated: output.truncated
    );
  }

  LanguagePair _resolvePair(Language? from, Language? to) {
    if (from != null && to != null) return LanguagePair(from, to);
    final fallback = _defaultPair;
    if (fallback == null) {
      throw ArgumentError(
          'No language pair given and this translator has no default. '
          'Pass `from:`/`to:`, or create it with OfflineTranslator.initialize('
          'from: ..., to: ...).');
    }
    if (from == null && to == null) return fallback;
    return LanguagePair(from ?? fallback.from, to ?? fallback.to);
  }

  Future<TranslationEngine> _engineFor(LanguagePair pair) {
    _checkNotDisposed();
    final loaded = _engines[pair.id];
    if (loaded != null) {
      // Refresh recency: reinserting moves the key to the end of the map.
      _engines
        ..remove(pair.id)
        ..[pair.id] = loaded;
      return Future<TranslationEngine>.value(loaded);
    }
    final pending = _loading[pair.id];
    if (pending != null) return pending;

    final future = _load(pair);
    _loading[pair.id] = future;
    return future.whenComplete(() => _loading.remove(pair.id));
  }

  Future<TranslationEngine> _load(LanguagePair pair) async {
    final info = await modelManager.getModel(pair);
    if (info == null) throw ModelNotInstalledException(pair.id);
    final engine = _engineFactory(info);
    await engine.load(config: generationConfig);
    _engines[pair.id] = engine;
    await _evictIfNeeded();
    return engine;
  }

  Future<void> _evictIfNeeded() async {
    while (_engines.length > maxLoadedModels) {
      final oldest = _engines.keys.first;
      final engine = _engines.remove(oldest);
      await engine?.dispose();
    }
  }

  void _checkNotDisposed() {
    if (_disposed) throw const TranslatorDisposedException();
  }

  /// Releases every loaded model and clears the cache.
  ///
  /// The instance cannot be used afterwards; create a new one with
  /// [initialize].
  Future<void> dispose() async {
    if (_disposed) return;
    _disposed = true;
    final engines = _engines.values.toList(growable: false);
    _engines.clear();
    for (final engine in engines) {
      await engine.dispose();
    }
    _cache?.clear();
  }
}
