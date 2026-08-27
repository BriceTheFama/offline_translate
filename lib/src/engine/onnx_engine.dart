import 'dart:io';
import 'dart:typed_data';

import 'package:path/path.dart' as p;

import '../core/generation_config.dart';
import '../core/model_info.dart';
import '../core/runtime_config.dart';
import '../exceptions/exceptions.dart';
import '../tokenizer/marian_tokenizer.dart';
import 'decoding_runner.dart';
import 'native/onnx_runtime.dart';
import 'native/onnx_runtime_bindings.dart' as bg;
import 'native/onnx_runtime_session.dart';
import 'translation_engine.dart';
import 'translation_worker.dart';

/// ONNX Runtime backend for both model families this package ships.
///
/// Every bundle is a two-session split — `encoder.onnx` and `decoder.onnx` —
/// and every `decoder.onnx` carries a `next_token` output that performs the
/// greedy arg-max *inside* the graph, so the vocabulary-wide logits tensor
/// never crosses the FFI boundary. What differs between the families is the
/// shape of the decoding loop, and that is [DecodingRunner]'s problem, not this
/// class's: [createRunner] reads `manifest.architecture.family` and returns the
/// right one.
///
/// This engine owns the sessions and one runner on the isolate that called
/// [load]; that runner serves [generate], and therefore `translate()`. It also
/// lazily spawns a [TranslationWorker] that attaches to the *same* sessions
/// from a background isolate, which is what serves [generateChunks] and
/// therefore `translateLong()` and `translateStream()`. The model is loaded
/// once either way.
class OnnxEngine implements TranslationEngine {
  /// Creates an engine for an installed [model].
  OnnxEngine(this.model, {this.runtimeConfig = const RuntimeConfig()});

  @override
  final ModelInfo model;

  /// Backend tuning applied when the sessions are created.
  final RuntimeConfig runtimeConfig;

  OrtSession? _encoderSession;
  OrtSession? _decoderSession;
  DecodingRunner? _runner;
  TranslationWorker? _worker;
  Future<TranslationWorker?>? _workerStarting;
  GenerationConfig _loadConfig = const GenerationConfig();
  bool _workerUnavailable = false;

  @override
  bool get isLoaded => _runner != null;

  /// The tokenizer bound to this model, once [load] has run.
  MarianTokenizer get tokenizer => _requireRunner().tokenizer;

  /// ONNX Runtime version backing this engine, e.g. `1.29.0`.
  static String get runtimeVersion => OrtLibrary.instance.versionString;

  DecodingRunner _requireRunner() {
    final runner = _runner;
    if (runner == null) {
      throw const TranslationEngineException('Engine not loaded');
    }
    return runner;
  }

  @override
  Future<void> load(
      {GenerationConfig config = const GenerationConfig()}) async {
    if (isLoaded) return;
    _loadConfig = config;
    final dir = model.path;
    try {
      // `vocab.json` is optional: OPUS-MT maps SentencePiece pieces through a
      // separate shared vocabulary, the Firefox students use the SentencePiece
      // ids directly and ship no second mapping.
      final vocabFile = File(p.join(dir, 'vocab.json'));
      final tokenizer = MarianTokenizer.fromAssets(
        spmBytes: await File(p.join(dir, 'source.spm')).readAsBytes(),
        vocabJson:
            await vocabFile.exists() ? await vocabFile.readAsString() : null,
      );
      final sessionConfig = _toSessionConfig(config);
      final encoder =
          OrtSession.fromFile(p.join(dir, 'encoder.onnx'), sessionConfig);
      _encoderSession = encoder;
      final decoder =
          OrtSession.fromFile(p.join(dir, 'decoder.onnx'), sessionConfig);
      _decoderSession = decoder;
      _runner = createRunner(
        model: model,
        tokenizer: tokenizer,
        encoder: encoder,
        decoder: decoder,
        maxInputTokens: config.maxInputTokens,
      );
    } on OfflineTranslatorException {
      await dispose();
      rethrow;
    } catch (e) {
      await dispose();
      throw TranslationEngineException(
          'Failed to load model "${model.id}" from $dir', e);
    }
  }

  OrtSessionConfig _toSessionConfig(GenerationConfig config) {
    final threads = runtimeConfig.threads != 0
        ? runtimeConfig.threads
        : (config.threads != 0 ? config.threads : _defaultThreads());
    return OrtSessionConfig(
      intraOpThreads: threads,
      interOpThreads: runtimeConfig.interOpThreads,
      graphOptimizationLevel: switch (runtimeConfig.graphOptimization) {
        GraphOptimization.none => bg.GraphOptimizationLevel.ORT_DISABLE_ALL,
        GraphOptimization.basic => bg.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        GraphOptimization.extended =>
          bg.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        GraphOptimization.all => bg.GraphOptimizationLevel.ORT_ENABLE_ALL,
      },
      disablePrePacking: !runtimeConfig.prePackWeights,
      enableCpuMemArena: runtimeConfig.useMemoryArena,
      enableMemPattern: runtimeConfig.useMemoryPattern,
      executionProvider: switch (runtimeConfig.accelerator) {
        Accelerator.cpu => OrtExecutionProvider.cpu,
        Accelerator.xnnpack => OrtExecutionProvider.xnnpack,
        Accelerator.nnapi => OrtExecutionProvider.nnapi,
        Accelerator.coreml => OrtExecutionProvider.coreml,
      },
    );
  }

  static int _defaultThreads() {
    final cores = Platform.numberOfProcessors;
    // Marian is small; past four threads the synchronisation cost dominates.
    return cores <= 2 ? 1 : (cores >= 6 ? 4 : cores - 1);
  }

  @override
  Int32List encodeText(String text) => _requireRunner().tokenizer.encode(text);

  @override
  String decodeTokens(List<int> tokens) =>
      _requireRunner().tokenizer.decode(tokens);

  @override
  GenerationOutput generate(Int32List inputIds, GenerationConfig config) =>
      _requireRunner().generate(inputIds, config);

  @override
  Stream<ChunkTranslation> generateChunks(
      List<String> chunks, GenerationConfig config) async* {
    final runner = _requireRunner();
    if (chunks.isEmpty) return;

    final worker = await _ensureWorker();
    if (worker == null) {
      // No isolate available: fall back to translating here, yielding to the
      // event loop between chunks so the caller is at least not starved.
      for (var i = 0; i < chunks.length; i++) {
        final result = runner.translate(chunks[i], config);
        yield ChunkTranslation(i, result.text, truncated: result.truncated);
        await Future<void>.delayed(Duration.zero);
      }
      return;
    }
    await for (final chunk in worker.translate(chunks, config)) {
      yield ChunkTranslation(chunk.index, chunk.text,
          truncated: chunk.truncated);
    }
  }

  Future<TranslationWorker?> _ensureWorker() {
    final existing = _worker;
    if (existing != null) return Future<TranslationWorker?>.value(existing);
    if (_workerUnavailable) return Future<TranslationWorker?>.value();
    return _workerStarting ??= _startWorker();
  }

  Future<TranslationWorker?> _startWorker() async {
    try {
      final worker = await TranslationWorker.spawn(
        model: model,
        encoder: _encoderSession!,
        decoder: _decoderSession!,
        maxInputTokens: _loadConfig.maxInputTokens,
      );
      _worker = worker;
      return worker;
    } catch (_) {
      // Falling back to same-isolate generation is always correct, just less
      // smooth, so a worker that cannot start must not fail the translation.
      _workerUnavailable = true;
      return null;
    } finally {
      _workerStarting = null;
    }
  }

  @override
  Future<void> dispose() async {
    await _worker?.close();
    _worker = null;
    _workerStarting = null;
    _runner?.dispose();
    _runner = null;
    _encoderSession?.release();
    _decoderSession?.release();
    _encoderSession = null;
    _decoderSession = null;
  }
}
