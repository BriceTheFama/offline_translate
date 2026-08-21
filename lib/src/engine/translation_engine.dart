import 'dart:typed_data';

import '../core/generation_config.dart';
import '../core/model_info.dart';

/// Raw token-level output of one generation pass.
class GenerationOutput {
  /// Creates a generation output.
  const GenerationOutput({
    required this.tokens,
    required this.truncated,
    required this.encodeMicros,
    required this.decodeMicros,
  });

  /// Generated target token ids, without the end-of-sequence token.
  final List<int> tokens;

  /// Whether generation hit a length limit instead of end-of-sequence.
  final bool truncated;

  /// Time spent in the encoder, in microseconds.
  final int encodeMicros;

  /// Time spent in the autoregressive decoder, in microseconds.
  final int decodeMicros;
}

/// The translation of one chunk, produced by [TranslationEngine.generateChunks].
class ChunkTranslation {
  /// Creates a chunk translation.
  const ChunkTranslation(this.index, this.text, {required this.truncated});

  /// Position of the chunk in the request, so results can be reassembled even
  /// if a backend ever produces them out of order.
  final int index;

  /// The translated text.
  final String text;

  /// Whether generation stopped on a length limit rather than end-of-sequence.
  final bool truncated;
}

/// A swappable inference backend.
///
/// The rest of the package talks only to this interface, so the ONNX Runtime
/// implementation can be replaced (by a native backend, a test double, or a
/// future runtime) without touching the public API.
abstract class TranslationEngine {
  /// Metadata of the model this engine serves.
  ModelInfo get model;

  /// Whether [load] has completed and [generate] can be called.
  bool get isLoaded;

  /// Loads the tokenizer and inference sessions into memory.
  ///
  /// Must be called exactly once before [generate]. Implementations keep
  /// everything resident until [dispose] is called. [config] tells the backend
  /// how wide to size its reusable buffers.
  Future<void> load({GenerationConfig config});

  /// Tokenizes [text] into source ids, terminated by end-of-sequence.
  Int32List encodeText(String text);

  /// Decodes generated target ids back into text.
  String decodeTokens(List<int> tokens);

  /// Runs greedy generation synchronously on the calling thread.
  ///
  /// [inputIds] must already be truncated to the model's input limit.
  GenerationOutput generate(Int32List inputIds, GenerationConfig config);

  /// Translates [chunks] without blocking the caller's isolate.
  ///
  /// Implementations should move the work off the calling isolate — the ONNX
  /// backend runs it on a worker attached to the same native sessions — and
  /// emit each chunk as soon as it is ready. Falling back to same-isolate
  /// generation is acceptable, as long as the stream still yields between
  /// chunks.
  Stream<ChunkTranslation> generateChunks(
      List<String> chunks, GenerationConfig config);

  /// Releases native sessions and memory.
  Future<void> dispose();
}
