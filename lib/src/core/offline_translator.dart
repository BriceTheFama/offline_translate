import 'dart:async';

import 'package:meta/meta.dart';

import '../cache/translation_cache.dart';
import '../engine/native/onnx_runtime.dart';
import '../engine/onnx_engine.dart';
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
///   defaultLanguage: Language.fr,
///   languages: {Language.en, Language.fr},
/// );
///
/// // Short text: synchronous, no await, the result is there.
/// final short = translator.translate(
///   'Hello, how are you?',
///   from: Language.en,
///   to: Language.fr,
/// );
/// print(short.translatedText); // Bonjour, comment allez-vous ?
///
/// // Long text: asynchronous, chunked, off the UI isolate.
/// final long = await translator.translateLong(article, from: Language.en);
///
/// await translator.dispose();
/// ```
///
/// **The sync/async split is the shape of the API, not an optimisation.**
/// [translate] returns a [TranslationResult] rather than a `Future`, so a
/// button handler can use its result immediately; that is only possible
/// because [initialize] has already put the model in memory, and it is why
/// [translate] throws [ModelNotLoadedException] instead of quietly loading one.
/// [translateLong] is the opposite trade: it may load a model, it splits the
/// text, and it runs inference on a worker isolate attached to the same native
/// sessions, so a long document never blocks a frame.
class OfflineTranslator {
  OfflineTranslator._({
    required this.modelManager,
    required this.generationConfig,
    required this.runtimeConfig,
    required this.maxLoadedModels,
    required TranslationCache? cache,
    required EngineFactory engineFactory,
    required Set<Language>? languages,
    required this.defaultLanguage,
  })  : _cache = cache,
        _engineFactory = engineFactory,
        _languages = languages;

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

  /// Target language used when a translation omits `to`.
  final Language? defaultLanguage;

  final TranslationCache? _cache;
  final EngineFactory _engineFactory;
  final Set<Language>? _languages;

  /// The languages this translator serves, or null when unrestricted.
  Set<Language>? get languages =>
      _languages == null ? null : Set<Language>.unmodifiable(_languages);

  /// The direction used when neither `from` nor `to` is given, if one can be
  /// worked out: a [defaultLanguage] target plus, when exactly two languages
  /// were declared, the other one as the source.
  LanguagePair? get _defaultPair {
    final target = defaultLanguage;
    if (target == null) return null;
    final languages = _languages;
    if (languages == null || languages.length != 2) return null;
    final source = languages.firstWhere((l) => l != target);
    return LanguagePair(source, target);
  }

  static const TextSegmenter _segmenter = TextSegmenter();

  /// Loaded engines, ordered least-recently-used first.
  final Map<String, TranslationEngine> _engines = <String, TranslationEngine>{};
  final Map<String, Future<TranslationEngine>> _loading =
      <String, Future<TranslationEngine>>{};

  bool _disposed = false;

  /// Creates a translator for [languages] and loads the models it already has.
  ///
  /// [languages] declares the set an application needs, and is the mechanism
  /// behind "you should not have to install models you will never use": any
  /// direction *within* the set is servable, everything outside it is rejected
  /// with [UnsupportedLanguageException] rather than sending you to download a
  /// model. Leaving it null declares no restriction.
  ///
  /// Every direction inside [languages] whose model is already installed is
  /// loaded here, up to [maxLoadedModels] — that is what makes [translate]
  /// synchronous afterwards. Directions that are not installed are skipped
  /// silently; call [installModel] for those.
  ///
  /// [defaultLanguage] is the target used when [translate] is called without
  /// `to`. When [languages] holds exactly two, the other one becomes the
  /// default source, so `translate(text)` needs no direction at all.
  ///
  /// [modelSource] tells the manager where model bundles come from; it is only
  /// consulted by [installModel], so it may be omitted entirely when the models
  /// are already on disk. Pass [modelManager] to supply your own.
  static Future<OfflineTranslator> initialize({
    Set<Language>? languages,
    Language? defaultLanguage,
    ModelSource? modelSource,
    ModelManager? modelManager,
    GenerationConfig generationConfig = const GenerationConfig(),
    RuntimeConfig runtimeConfig = const RuntimeConfig(),
    TranslationCache? cache,
    int maxLoadedModels = 2,
    @visibleForTesting EngineFactory? engineFactory,
  }) async {
    final declared = languages == null ? null : Set<Language>.of(languages);
    if (declared != null && declared.isEmpty) {
      throw ArgumentError.value(languages, 'languages',
          'Declare at least one language, or pass null');
    }
    if (declared != null &&
        defaultLanguage != null &&
        !declared.contains(defaultLanguage)) {
      throw ArgumentError.value(defaultLanguage, 'defaultLanguage',
          'is not one of the declared languages');
    }
    final translator = OfflineTranslator._(
      modelManager: modelManager ?? FileModelManager(source: modelSource),
      generationConfig: generationConfig,
      runtimeConfig: runtimeConfig,
      maxLoadedModels: maxLoadedModels,
      cache: cache,
      engineFactory: engineFactory ??
          (info) => OnnxEngine(info, runtimeConfig: runtimeConfig),
      languages: declared,
      defaultLanguage: defaultLanguage,
    );
    await translator._preloadInstalled();
    return translator;
  }

  /// Loads every declared direction that is already installed.
  ///
  /// A missing model is not an error here: an application may well initialize
  /// before its first download. It becomes an error only when [translate] is
  /// called for that direction, because a synchronous call cannot go to disk.
  Future<void> _preloadInstalled() async {
    for (final pair in _declaredPairs()) {
      if (_engines.length >= maxLoadedModels) break;
      try {
        if (await modelManager.isInstalled(pair)) await _engineFor(pair);
      } on OfflineTranslatorException {
        // A corrupted or unreadable bundle must not stop initialize(); the
        // failure resurfaces, with its own message, on first use.
      }
    }
  }

  /// Every direction inside the declared language set, default direction first.
  List<LanguagePair> _declaredPairs() {
    final languages = _languages;
    if (languages == null) {
      final fallback = _defaultPair;
      return fallback == null
          ? const <LanguagePair>[]
          : <LanguagePair>[fallback];
    }
    final pairs = <LanguagePair>[
      for (final from in languages)
        for (final to in languages)
          if (from != to) LanguagePair(from, to),
    ];
    final preferred = _defaultPair;
    if (preferred != null) {
      pairs
        ..removeWhere((p) => p == preferred)
        ..insert(0, preferred);
    }
    return pairs;
  }

  /// Where to load ONNX Runtime from, when it is not already reachable.
  ///
  /// Applications never need this. On Android the shared library comes from
  /// the plugin's Gradle configuration, and on iOS and macOS from the
  /// `onnxruntime-c` pod, so it is simply found.
  ///
  /// It exists for **host tests and desktop tooling**, where there is no pod
  /// and no APK: `flutter test` running against the real engine has to be told
  /// where a local runtime lives. Set it before [initialize]; ONNX Runtime is
  /// opened once per process, so changing it afterwards has no effect.
  ///
  /// ```dart
  /// OfflineTranslator.onnxRuntimeLibraryPath = 'third_party/.../libonnxruntime.dylib';
  /// ```
  static set onnxRuntimeLibraryPath(String? path) =>
      OrtLibrary.overrideLibraryPath = path;

  /// The path set by [onnxRuntimeLibraryPath], if any.
  static String? get onnxRuntimeLibraryPath => OrtLibrary.overrideLibraryPath;

  /// ONNX Runtime version backing this package, e.g. `1.29.0`.
  ///
  /// Reading this opens the runtime, so it throws with the same message as
  /// [initialize] would when none can be found.
  static String get onnxRuntimeVersion => OrtLibrary.instance.versionString;

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
  /// This is deliberately **not** a `Future`:
  ///
  /// ```dart
  /// final result = translator.translate('Hello, how are you?',
  ///     from: Language.en, to: Language.fr);
  /// print(result.translatedText); // Bonjour, comment allez-vous ?
  /// ```
  ///
  /// The model must already be loaded, which [initialize] takes care of for
  /// every installed direction it was told about. If it is not, this throws
  /// [ModelNotLoadedException] rather than blocking on I/O — call [preload] or
  /// [translateLong] first, both of which may load.
  ///
  /// Inference runs synchronously on the caller's thread: roughly 1 ms per
  /// generated token on a laptop and a few ms on a phone, so a sentence costs
  /// tens of milliseconds. That is fine from a button handler and wrong for a
  /// document — use [translateLong] past a sentence or two. Text that does not
  /// fit the model's input window is chunked and the chunks are translated in
  /// sequence, still synchronously.
  TranslationResult translate(
    String text, {
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

  /// Translates text of any length without blocking the UI isolate.
  ///
  /// Loads the model on demand, splits the text along paragraph and sentence
  /// boundaries, and runs inference on a worker isolate attached to the *same*
  /// native sessions — the model is loaded once and shared, never reloaded per
  /// chunk. Use [translateStream] when you want output as it is produced.
  ///
  /// ```dart
  /// final result = await translator.translateLong(article,
  ///     from: Language.en, to: Language.fr);
  /// ```
  Future<TranslationResult> translateLong(
    String text, {
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
  /// across the stream reproduces exactly what [translateLong] would return.
  Stream<TranslationResult> translateStream(
    String text, {
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
    final target = to ?? defaultLanguage;
    var source = from;
    if (source == null || target == null) {
      final fallback = _defaultPair;
      if (fallback == null) {
        throw ArgumentError(
            'No language pair given and this translator has no default. Pass '
            '`from:` and `to:`, or initialize() with a `defaultLanguage:` and '
            'exactly two `languages:`.');
      }
      source ??= fallback.from;
    }
    final pair = LanguagePair(source, target ?? _defaultPair!.to);
    _checkDeclared(pair);
    return pair;
  }

  /// Rejects a direction that leaves the language set given to [initialize].
  void _checkDeclared(LanguagePair pair) {
    final languages = _languages;
    if (languages == null) return;
    for (final language in <Language>[pair.from, pair.to]) {
      if (!languages.contains(language)) {
        throw UnsupportedLanguageException(language, languages);
      }
    }
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
