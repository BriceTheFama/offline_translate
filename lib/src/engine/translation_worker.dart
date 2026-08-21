import 'dart:async';
import 'dart:io';
import 'dart:isolate';

import 'package:path/path.dart' as p;

import '../core/generation_config.dart';
import '../core/model_info.dart';
import '../exceptions/exceptions.dart';
import '../tokenizer/marian_tokenizer.dart';
import 'marian_runner.dart';
import 'native/onnx_runtime.dart';
import 'native/onnx_runtime_session.dart';

/// Everything a worker isolate needs to attach to an already-loaded model.
///
/// Only plain values cross the isolate boundary: the two session addresses,
/// the environment address, the model directory and the manifest. The worker
/// re-opens ONNX Runtime (a process-global library), adopts those handles
/// without taking ownership, and builds its own plans and tokenizer.
class WorkerBootstrap {
  /// Creates a bootstrap message.
  const WorkerBootstrap({
    required this.replyTo,
    required this.manifestJson,
    required this.modelPath,
    required this.envAddress,
    required this.encoderAddress,
    required this.decoderAddress,
    required this.maxInputTokens,
    required this.libraryPath,
  });

  /// Port the worker sends its own receive port to.
  final SendPort replyTo;

  /// Serialised [ModelInfo]; Dart objects cannot be shared, only copied.
  final String manifestJson;

  /// Directory holding `source.spm` and `vocab.json`.
  final String modelPath;

  /// Address of the shared `OrtEnv`.
  final int envAddress;

  /// Address of the shared encoder session.
  final int encoderAddress;

  /// Address of the shared decoder session.
  final int decoderAddress;

  /// Encoder input width the scratch tensors are sized for.
  final int maxInputTokens;

  /// Override for the ONNX Runtime library path, when one is in force.
  final String? libraryPath;
}

/// A request to translate a batch of chunks.
class _WorkerRequest {
  const _WorkerRequest(this.id, this.chunks, this.config);

  final int id;
  final List<String> chunks;
  final GenerationConfig config;
}

/// One translated chunk, sent back as soon as it is ready.
class WorkerChunk {
  /// Creates a chunk result.
  const WorkerChunk(this.id, this.index, this.text, {required this.truncated});

  /// Identifies the request this belongs to.
  final int id;

  /// Position of this chunk in the request.
  final int index;

  /// The translated text.
  final String text;

  /// Whether generation stopped on a length limit.
  final bool truncated;
}

/// Signals that a request finished, successfully or not.
class _WorkerDone {
  const _WorkerDone(this.id, this.error);

  final int id;
  final String? error;
}

class _WorkerShutdown {
  const _WorkerShutdown();
}

/// Runs inference on a background isolate against sessions another isolate
/// owns.
///
/// This is what keeps `translate()` from stalling the UI. `translateSync()`
/// deliberately stays on the caller's isolate — that is its contract — but a
/// long document goes through here, so the interface keeps rendering while the
/// decoder runs.
///
/// The model is *not* loaded twice. ONNX Runtime sessions are safe to run from
/// several threads, so the worker attaches to the resident sessions and only
/// adds its own run plans, scratch tensors and tokenizer.
class TranslationWorker {
  TranslationWorker._(this._isolate, this._toWorker, this._fromWorker);

  final Isolate _isolate;
  final SendPort _toWorker;
  final ReceivePort _fromWorker;

  final Map<int, StreamController<WorkerChunk>> _pending =
      <int, StreamController<WorkerChunk>>{};
  int _nextId = 0;
  bool _closed = false;

  /// Spawns a worker attached to the sessions described by [bootstrap].
  static Future<TranslationWorker> spawn({
    required ModelInfo model,
    required OrtSession encoder,
    required OrtSession decoder,
    required int maxInputTokens,
  }) async {
    final fromWorker = ReceivePort();
    final bootstrap = WorkerBootstrap(
      replyTo: fromWorker.sendPort,
      manifestJson: model.toJsonString(),
      modelPath: model.path,
      envAddress: OrtEnv.instance.address,
      encoderAddress: encoder.address,
      decoderAddress: decoder.address,
      maxInputTokens: maxInputTokens,
      libraryPath: OrtLibrary.overrideLibraryPath,
    );
    // An uncaught error in the worker would otherwise leave callers waiting
    // forever, so failures are routed back onto the same port and surfaced as
    // a stream error.
    final isolate = await Isolate.spawn(
      _workerMain,
      bootstrap,
      debugName: 'offline_translate:${model.id}',
      errorsAreFatal: true,
      onError: fromWorker.sendPort,
      onExit: fromWorker.sendPort,
    );

    final events = StreamQueue<Object?>(fromWorker);
    final first = await events.next;
    if (first is! SendPort) {
      isolate.kill(priority: Isolate.immediate);
      fromWorker.close();
      throw TranslationEngineException(
          'Translation worker failed to start: $first');
    }
    final worker = TranslationWorker._(isolate, first, fromWorker);
    unawaited(worker._listen(events));
    return worker;
  }

  Future<void> _listen(StreamQueue<Object?> events) async {
    while (await events.hasNext) {
      final message = await events.next;
      if (message is WorkerChunk) {
        _pending[message.id]?.add(message);
      } else if (message is _WorkerDone) {
        final controller = _pending.remove(message.id);
        final error = message.error;
        if (error != null) {
          controller?.addError(TranslationEngineException(error));
        }
        await controller?.close();
      } else if (message is List || message == null) {
        // `onError` sends [error, stackTrace]; `onExit` sends null. Either way
        // the worker is gone, so nothing else will answer the open requests.
        final reason = message is List && message.isNotEmpty
            ? '${message.first}'
            : 'the translation worker exited unexpectedly';
        for (final controller in _pending.values) {
          controller
            ..addError(TranslationEngineException(reason))
            ..close().ignore();
        }
        _pending.clear();
        _closed = true;
        return;
      }
    }
  }

  /// Translates [chunks], emitting each result as it is produced.
  Stream<WorkerChunk> translate(List<String> chunks, GenerationConfig config) {
    if (_closed) {
      throw const TranslationEngineException('Translation worker is closed');
    }
    final id = _nextId++;
    final controller = StreamController<WorkerChunk>();
    _pending[id] = controller;
    _toWorker.send(_WorkerRequest(id, chunks, config));
    return controller.stream;
  }

  /// Stops the worker.
  Future<void> close() async {
    if (_closed) return;
    _closed = true;
    _toWorker.send(const _WorkerShutdown());
    // Give the worker a moment to release its plans, then make sure it is gone.
    await Future<void>.delayed(const Duration(milliseconds: 50));
    for (final controller in _pending.values) {
      await controller.close();
    }
    _pending.clear();
    _fromWorker.close();
    _isolate.kill(priority: Isolate.beforeNextEvent);
  }
}

Future<void> _workerMain(WorkerBootstrap bootstrap) async {
  final commands = ReceivePort();
  MarianRunner? runner;
  OrtSession? encoder;
  OrtSession? decoder;
  try {
    if (bootstrap.libraryPath != null) {
      OrtLibrary.overrideLibraryPath = bootstrap.libraryPath;
    }
    OrtEnv.adopt(bootstrap.envAddress);
    final model =
        ModelInfo.parse(bootstrap.manifestJson, path: bootstrap.modelPath);
    final tokenizer = MarianTokenizer.fromAssets(
      spmBytes:
          File(p.join(bootstrap.modelPath, 'source.spm')).readAsBytesSync(),
      vocabJson:
          File(p.join(bootstrap.modelPath, 'vocab.json')).readAsStringSync(),
    );
    encoder = OrtSession.adopt(bootstrap.encoderAddress);
    decoder = OrtSession.adopt(bootstrap.decoderAddress);
    runner = MarianRunner.create(
      model: model,
      tokenizer: tokenizer,
      encoder: encoder,
      decoder: decoder,
      maxInputTokens: bootstrap.maxInputTokens,
    );
  } catch (e) {
    bootstrap.replyTo.send('$e');
    commands.close();
    return;
  }

  bootstrap.replyTo.send(commands.sendPort);

  await for (final message in commands) {
    if (message is _WorkerShutdown) break;
    if (message is! _WorkerRequest) continue;
    String? error;
    try {
      for (var i = 0; i < message.chunks.length; i++) {
        final result = runner.translate(message.chunks[i], message.config);
        bootstrap.replyTo.send(WorkerChunk(message.id, i, result.text,
            truncated: result.truncated));
      }
    } catch (e) {
      error = '$e';
    }
    bootstrap.replyTo.send(_WorkerDone(message.id, error));
  }

  runner.dispose();
  encoder.release();
  decoder.release();
  commands.close();
}

/// A minimal pull-based view over a [ReceivePort].
///
/// `package:async`'s StreamQueue would do, but it is not a dependency and this
/// is the only place that needs one.
class StreamQueue<T> {
  /// Wraps [stream].
  StreamQueue(Stream<T> stream) {
    _subscription = stream.listen((event) {
      final completer = _waiting.isEmpty ? null : _waiting.removeAt(0);
      if (completer != null) {
        completer.complete(event);
      } else {
        _buffer.add(event);
      }
    }, onDone: () {
      _done = true;
      for (final completer in _waiting) {
        completer.complete(null);
      }
      _waiting.clear();
    });
  }

  late final StreamSubscription<T> _subscription;
  final List<T> _buffer = <T>[];
  final List<Completer<T?>> _waiting = <Completer<T?>>[];
  bool _done = false;

  /// Whether another event may still arrive.
  Future<bool> get hasNext async => !_done || _buffer.isNotEmpty;

  /// The next event, or `null` once the stream is done.
  Future<T?> get next {
    if (_buffer.isNotEmpty) return Future<T?>.value(_buffer.removeAt(0));
    if (_done) return Future<T?>.value();
    final completer = Completer<T?>();
    _waiting.add(completer);
    return completer.future;
  }

  /// Stops listening.
  Future<void> cancel() => _subscription.cancel();
}
