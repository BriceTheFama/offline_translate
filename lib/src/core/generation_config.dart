import 'package:meta/meta.dart';

/// Tuning knobs for the autoregressive decoder.
@immutable
class GenerationConfig {
  /// Creates a generation configuration.
  const GenerationConfig({
    this.maxInputTokens = 512,
    this.maxNewTokens = 512,
    this.lengthRatioLimit = 3.0,
    this.threads = 0,
  })  : assert(maxInputTokens > 0, 'maxInputTokens must be positive'),
        assert(maxNewTokens > 0, 'maxNewTokens must be positive');

  /// Hard cap on encoder input length. Marian models are trained with 512
  /// learned positions; longer inputs are chunked before they reach the engine.
  final int maxInputTokens;

  /// Hard cap on the number of tokens generated for a single chunk.
  final int maxNewTokens;

  /// Stops generation once the output exceeds this multiple of the input
  /// length. Guards against the degenerate repetition loops that greedy
  /// decoding can fall into.
  final double lengthRatioLimit;

  /// Number of intra-op threads for ONNX Runtime. `0` lets the runtime decide.
  final int threads;

  /// Returns a copy with the given fields replaced.
  GenerationConfig copyWith({
    int? maxInputTokens,
    int? maxNewTokens,
    double? lengthRatioLimit,
    int? threads,
  }) =>
      GenerationConfig(
        maxInputTokens: maxInputTokens ?? this.maxInputTokens,
        maxNewTokens: maxNewTokens ?? this.maxNewTokens,
        lengthRatioLimit: lengthRatioLimit ?? this.lengthRatioLimit,
        threads: threads ?? this.threads,
      );
}
