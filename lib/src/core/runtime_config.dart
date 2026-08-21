import 'package:meta/meta.dart';

/// How aggressively the inference backend may rewrite the model graph.
enum GraphOptimization {
  /// No rewriting. Loads fastest and uses the least memory; roughly 30 % slower.
  none,

  /// Cheap, always-safe rewrites.
  basic,

  /// Adds operator fusion.
  extended,

  /// Everything the backend offers. The default.
  all,
}

/// Which backend runs the model.
///
/// None of the accelerators is enabled by default, and the measurements say
/// none of them should be. Per translated sentence, on the reference devices:
///
/// | | Android (4 cores) | iOS (8 cores) |
/// |---|---|---|
/// | **cpu** | **236 ms** | **161 ms** |
/// | xnnpack | 236 ms | 163 ms |
/// | nnapi | 312 ms | not in the build |
/// | coreml | not in the build | 179 ms |
///
/// A dynamically quantised encoder-decoder whose sequence length grows every
/// step is simply the wrong shape for them. Full tables in
/// `doc/onnx-runtime.md` and `doc/performance.md`.
enum Accelerator {
  /// Plain CPU kernels. The measured default, and the fastest option on every
  /// device tested.
  cpu,

  /// XNNPACK — quantised CPU GEMM. Available on Android and iOS, not in Apple
  /// desktop builds. Measured indistinguishable from [cpu] here.
  xnnpack,

  /// Android NNAPI. Deprecated upstream since Android 15, and measured 32 %
  /// slower with a 1.8 s first inference while it partitions the graph.
  nnapi,

  /// Apple CoreML. Prefers static shapes; measured 8-11 % slower.
  coreml,
}

/// Backend tuning, independent of which engine implements it.
///
/// The defaults are what the benchmarks in `doc/performance.md` were taken
/// with. Two presets cover the usual choices: [RuntimeConfig.speed] and
/// [RuntimeConfig.lowMemory].
@immutable
class RuntimeConfig {
  /// Creates a runtime configuration.
  const RuntimeConfig({
    this.threads = 0,
    this.interOpThreads = 1,
    this.graphOptimization = GraphOptimization.all,
    this.prePackWeights = true,
    this.useMemoryArena = true,
    this.useMemoryPattern = true,
    this.accelerator = Accelerator.cpu,
  });

  /// Threads used inside a single operator. `0` picks from the core count.
  final int threads;

  /// Threads used to run independent operators in parallel.
  final int interOpThreads;

  /// Graph rewriting level.
  final GraphOptimization graphOptimization;

  /// Whether the backend may pack weights into its preferred GEMM layout.
  ///
  /// Leave this on. It reads like a memory/speed trade and it is not: with
  /// ONNX Runtime 1.29 turning it **off** costs about 130 MB *more* and 13 %
  /// throughput, because the unpacked int8 weights are then converted into
  /// arena buffers on every run instead of once. The measurements are in
  /// `doc/onnx-runtime.md`.
  final bool prePackWeights;

  /// Whether the backend keeps an arena of reusable buffers.
  ///
  /// Leave this on. On mobile, turning it off makes generation **3-4× slower**
  /// (236 → 903 ms per sentence on Android, 161 → 511 ms on iOS) because every
  /// intermediate buffer then goes back to the system allocator.
  final bool useMemoryArena;

  /// Whether the backend pre-plans intermediate buffers from observed shapes.
  final bool useMemoryPattern;

  /// Backend to run on.
  final Accelerator accelerator;

  /// Favours throughput. The default, and what the published benchmarks use.
  static const RuntimeConfig speed = RuntimeConfig();

  /// Skips graph optimisation: about 20 MB less resident memory, at a real
  /// cost in speed.
  ///
  /// Read the numbers before choosing this. It is **28 % slower on a laptop,
  /// 55 % slower on iOS and 60 % slower on Android** — 236 → 379 ms per
  /// sentence on the Android reference device — and it buys roughly 20 MB out
  /// of ~180 MB.
  ///
  /// It is offered because on a 1 GB device 20 MB can matter, not because it is
  /// generally a good trade. After the FFI engine rewrite the runtime no longer
  /// wastes memory, so there is no longer a large memory/speed dial to turn:
  /// every other combination measured either changed nothing or made both axes
  /// worse.
  static const RuntimeConfig lowMemory = RuntimeConfig(
    graphOptimization: GraphOptimization.none,
  );

  /// Returns a copy with the given fields replaced.
  RuntimeConfig copyWith({
    int? threads,
    int? interOpThreads,
    GraphOptimization? graphOptimization,
    bool? prePackWeights,
    bool? useMemoryArena,
    bool? useMemoryPattern,
    Accelerator? accelerator,
  }) =>
      RuntimeConfig(
        threads: threads ?? this.threads,
        interOpThreads: interOpThreads ?? this.interOpThreads,
        graphOptimization: graphOptimization ?? this.graphOptimization,
        prePackWeights: prePackWeights ?? this.prePackWeights,
        useMemoryArena: useMemoryArena ?? this.useMemoryArena,
        useMemoryPattern: useMemoryPattern ?? this.useMemoryPattern,
        accelerator: accelerator ?? this.accelerator,
      );

  @override
  String toString() => 'RuntimeConfig(threads: $threads, '
      'opt: ${graphOptimization.name}, prepack: $prePackWeights, '
      'arena: $useMemoryArena, memPattern: $useMemoryPattern, '
      'ep: ${accelerator.name})';
}
