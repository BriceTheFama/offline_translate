import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';
import 'package:meta/meta.dart';

import '../../exceptions/exceptions.dart';
import 'onnx_runtime.dart';
import 'onnx_runtime_allocator.dart';
import 'onnx_runtime_bindings.dart' as bg;
import 'onnx_runtime_tensor.dart';

/// Execution providers this package knows how to request.
enum OrtExecutionProvider {
  /// Plain CPU kernels. Always available; the default.
  cpu,

  /// XNNPACK, a quantised-GEMM CPU backend. Android and iOS.
  xnnpack,

  /// Android NNAPI. Deprecated upstream and usually a loss for dynamic shapes.
  nnapi,

  /// Apple CoreML. Wants static shapes, so rarely a win here.
  coreml,
}

/// Session tuning that `offline_translate` actually uses.
@immutable
class OrtSessionConfig {
  /// Creates a configuration.
  const OrtSessionConfig({
    this.intraOpThreads = 4,
    this.interOpThreads = 1,
    this.graphOptimizationLevel = bg.GraphOptimizationLevel.ORT_ENABLE_ALL,
    this.disablePrePacking = false,
    this.enableCpuMemArena = true,
    this.enableMemPattern = true,
    this.executionProvider = OrtExecutionProvider.cpu,
  });

  /// Threads used inside a single operator.
  final int intraOpThreads;

  /// Threads used to run independent operators in parallel.
  final int interOpThreads;

  /// How aggressively ONNX Runtime rewrites the graph at load time.
  final bg.GraphOptimizationLevel graphOptimizationLevel;

  /// Skip packing weights into the GEMM kernel's preferred layout.
  ///
  /// Pre-packing is what makes the first inference expensive in memory — it is
  /// worth roughly 180 MB on this model — and it buys about 30 % throughput.
  /// See `doc/onnx-runtime.md` for the measurements.
  final bool disablePrePacking;

  /// ONNX Runtime's arena allocator. Off means memory is returned sooner and
  /// peaks are lower, at the cost of more allocator traffic.
  final bool enableCpuMemArena;

  /// Pre-planning of intermediate buffers from the first run's shapes.
  final bool enableMemPattern;

  /// Backend to run the graph on.
  final OrtExecutionProvider executionProvider;

  /// Returns a copy with the given fields replaced.
  OrtSessionConfig copyWith({
    int? intraOpThreads,
    int? interOpThreads,
    bg.GraphOptimizationLevel? graphOptimizationLevel,
    bool? disablePrePacking,
    bool? enableCpuMemArena,
    bool? enableMemPattern,
    OrtExecutionProvider? executionProvider,
  }) =>
      OrtSessionConfig(
        intraOpThreads: intraOpThreads ?? this.intraOpThreads,
        interOpThreads: interOpThreads ?? this.interOpThreads,
        graphOptimizationLevel:
            graphOptimizationLevel ?? this.graphOptimizationLevel,
        disablePrePacking: disablePrePacking ?? this.disablePrePacking,
        enableCpuMemArena: enableCpuMemArena ?? this.enableCpuMemArena,
        enableMemPattern: enableMemPattern ?? this.enableMemPattern,
        executionProvider: executionProvider ?? this.executionProvider,
      );

  @override
  String toString() => 'OrtSessionConfig(intra: $intraOpThreads, '
      'inter: $interOpThreads, opt: ${graphOptimizationLevel.name}, '
      'prepack: ${!disablePrePacking}, arena: $enableCpuMemArena, '
      'memPattern: $enableMemPattern, ep: ${executionProvider.name})';
}

/// A loaded ONNX model.
class OrtSession {
  OrtSession._(this._library, this._ptr, this.inputNames, this.outputNames);

  final OrtLibrary _library;
  final ffi.Pointer<bg.OrtSession> _ptr;

  /// Input names declared by the graph, in order.
  final List<String> inputNames;

  /// Output names declared by the graph, in order.
  final List<String> outputNames;

  final List<OrtRunPlan> _plans = <OrtRunPlan>[];
  ffi.Pointer<bg.OrtRunOptions> _runOptions = ffi.nullptr;
  bool _released = false;
  bool _owned = true;

  /// Wraps a session that another isolate created and still owns.
  ///
  /// ONNX Runtime sessions are safe to call `Run` on from several threads, so
  /// a worker isolate can share the model already resident in memory instead
  /// of loading a second copy. The wrapper is non-owning: [release] frees the
  /// run options and plans this isolate created, never the session itself.
  static OrtSession adopt(int address) {
    final library = OrtLibrary.instance;
    final ptr = ffi.Pointer<bg.OrtSession>.fromAddress(address);
    final session = OrtSession._(
      library,
      ptr,
      _names(library, ptr, input: true),
      _names(library, ptr, input: false),
    );
    session._owned = false;
    final runOptions = calloc<ffi.Pointer<bg.OrtRunOptions>>();
    try {
      library.check(
        library.api.ref.CreateRunOptions.asFunction<
            ffi.Pointer<bg.OrtStatus> Function(
                ffi.Pointer<ffi.Pointer<bg.OrtRunOptions>>)>()(runOptions),
        'CreateRunOptions',
      );
      session._runOptions = runOptions.value;
    } finally {
      calloc.free(runOptions);
    }
    return session;
  }

  /// Address of the underlying session, for sharing with another isolate.
  int get address => _ptr.address;

  /// Loads the model at [path] with [config].
  static OrtSession fromFile(String path, OrtSessionConfig config) {
    final library = OrtLibrary.instance;
    final ref = library.api.ref;
    final options = _createOptions(library, config);
    final pathPtr = path.toNativeUtf8();
    final out = calloc<ffi.Pointer<bg.OrtSession>>();
    try {
      library.check(
        ref.CreateSession.asFunction<
            ffi.Pointer<bg.OrtStatus> Function(
                ffi.Pointer<bg.OrtEnv>,
                ffi.Pointer<ffi.Char>,
                ffi.Pointer<bg.OrtSessionOptions>,
                ffi.Pointer<ffi.Pointer<bg.OrtSession>>)>()(
          OrtEnv.instance.pointer,
          pathPtr.cast<ffi.Char>(),
          options,
          out,
        ),
        'CreateSession($path)',
      );
      final ptr = out.value;
      final session = OrtSession._(
        library,
        ptr,
        _names(library, ptr, input: true),
        _names(library, ptr, input: false),
      );
      final runOptions = calloc<ffi.Pointer<bg.OrtRunOptions>>();
      try {
        library.check(
          ref.CreateRunOptions.asFunction<
              ffi.Pointer<bg.OrtStatus> Function(
                  ffi.Pointer<ffi.Pointer<bg.OrtRunOptions>>)>()(runOptions),
          'CreateRunOptions',
        );
        session._runOptions = runOptions.value;
      } finally {
        calloc.free(runOptions);
      }
      return session;
    } finally {
      ref.ReleaseSessionOptions.asFunction<
          void Function(ffi.Pointer<bg.OrtSessionOptions>)>()(options);
      calloc
        ..free(pathPtr)
        ..free(out);
    }
  }

  static ffi.Pointer<bg.OrtSessionOptions> _createOptions(
      OrtLibrary library, OrtSessionConfig config) {
    final ref = library.api.ref;
    final out = calloc<ffi.Pointer<bg.OrtSessionOptions>>();
    late final ffi.Pointer<bg.OrtSessionOptions> options;
    try {
      library.check(
        ref.CreateSessionOptions.asFunction<
            ffi.Pointer<bg.OrtStatus> Function(
                ffi.Pointer<ffi.Pointer<bg.OrtSessionOptions>>)>()(out),
        'CreateSessionOptions',
      );
      options = out.value;
    } finally {
      calloc.free(out);
    }

    library.check(
      ref.SetIntraOpNumThreads.asFunction<
          ffi.Pointer<bg.OrtStatus> Function(ffi.Pointer<bg.OrtSessionOptions>,
              int)>()(options, config.intraOpThreads),
      'SetIntraOpNumThreads',
    );
    library.check(
      ref.SetInterOpNumThreads.asFunction<
          ffi.Pointer<bg.OrtStatus> Function(ffi.Pointer<bg.OrtSessionOptions>,
              int)>()(options, config.interOpThreads),
      'SetInterOpNumThreads',
    );
    library.check(
      ref.SetSessionGraphOptimizationLevel.asFunction<
          ffi.Pointer<bg.OrtStatus> Function(ffi.Pointer<bg.OrtSessionOptions>,
              int)>()(options, config.graphOptimizationLevel.value),
      'SetSessionGraphOptimizationLevel',
    );
    if (!config.enableCpuMemArena) {
      library.check(
        ref.DisableCpuMemArena.asFunction<
            ffi.Pointer<bg.OrtStatus> Function(
                ffi.Pointer<bg.OrtSessionOptions>)>()(options),
        'DisableCpuMemArena',
      );
    }
    if (!config.enableMemPattern) {
      library.check(
        ref.DisableMemPattern.asFunction<
            ffi.Pointer<bg.OrtStatus> Function(
                ffi.Pointer<bg.OrtSessionOptions>)>()(options),
        'DisableMemPattern',
      );
    }
    if (config.disablePrePacking) {
      _addConfigEntry(library, options, 'session.disable_prepacking', '1');
    }

    switch (config.executionProvider) {
      case OrtExecutionProvider.cpu:
        break;
      case OrtExecutionProvider.xnnpack:
        _appendProvider(library, options, 'XNNPACK', <String, String>{
          'intra_op_num_threads': '${config.intraOpThreads}',
        });
      case OrtExecutionProvider.nnapi:
        _appendNnapi(library, options);
      case OrtExecutionProvider.coreml:
        _appendProvider(library, options, 'CoreML', const <String, String>{});
    }
    return options;
  }

  static void _addConfigEntry(
    OrtLibrary library,
    ffi.Pointer<bg.OrtSessionOptions> options,
    String key,
    String value,
  ) {
    final keyPtr = key.toNativeUtf8();
    final valuePtr = value.toNativeUtf8();
    try {
      library.check(
        library.api.ref.AddSessionConfigEntry.asFunction<
            ffi.Pointer<bg.OrtStatus> Function(
                ffi.Pointer<bg.OrtSessionOptions>,
                ffi.Pointer<ffi.Char>,
                ffi.Pointer<ffi.Char>)>()(
          options,
          keyPtr.cast<ffi.Char>(),
          valuePtr.cast<ffi.Char>(),
        ),
        'AddSessionConfigEntry($key)',
      );
    } finally {
      calloc
        ..free(keyPtr)
        ..free(valuePtr);
    }
  }

  /// NNAPI is not reachable through the generic provider entry point: it has a
  /// dedicated `OrtSessionOptionsAppendExecutionProvider_Nnapi` free function
  /// that only exists in the Android build.
  static void _appendNnapi(
      OrtLibrary library, ffi.Pointer<bg.OrtSessionOptions> options) {
    final append = library.lookupOrNull<
        ffi.Pointer<bg.OrtStatus> Function(ffi.Pointer<bg.OrtSessionOptions>,
            ffi.Uint32)>('OrtSessionOptionsAppendExecutionProvider_Nnapi');
    if (append == null) {
      throw const TranslationEngineException(
          'NNAPI is not available in this ONNX Runtime build (Android only).');
    }
    library.check(
      append.asFunction<
          ffi.Pointer<bg.OrtStatus> Function(
              ffi.Pointer<bg.OrtSessionOptions>, int)>()(options, 0),
      'OrtSessionOptionsAppendExecutionProvider_Nnapi',
    );
  }

  static void _appendProvider(
    OrtLibrary library,
    ffi.Pointer<bg.OrtSessionOptions> options,
    String name,
    Map<String, String> providerOptions,
  ) {
    final namePtr = name.toNativeUtf8();
    final count = providerOptions.length;
    final keys = calloc<ffi.Pointer<ffi.Char>>(count == 0 ? 1 : count);
    final values = calloc<ffi.Pointer<ffi.Char>>(count == 0 ? 1 : count);
    var index = 0;
    providerOptions.forEach((key, value) {
      keys[index] = key.toNativeUtf8().cast<ffi.Char>();
      values[index] = value.toNativeUtf8().cast<ffi.Char>();
      index++;
    });
    try {
      library.check(
        library.api.ref.SessionOptionsAppendExecutionProvider.asFunction<
            ffi.Pointer<bg.OrtStatus> Function(
                ffi.Pointer<bg.OrtSessionOptions>,
                ffi.Pointer<ffi.Char>,
                ffi.Pointer<ffi.Pointer<ffi.Char>>,
                ffi.Pointer<ffi.Pointer<ffi.Char>>,
                int)>()(
          options,
          namePtr.cast<ffi.Char>(),
          keys,
          values,
          count,
        ),
        'SessionOptionsAppendExecutionProvider($name)',
      );
    } finally {
      for (var i = 0; i < count; i++) {
        calloc
          ..free(keys[i])
          ..free(values[i]);
      }
      calloc
        ..free(keys)
        ..free(values)
        ..free(namePtr);
    }
  }

  static List<String> _names(
      OrtLibrary library, ffi.Pointer<bg.OrtSession> session,
      {required bool input}) {
    final ref = library.api.ref;
    final allocator = OrtCpuAllocator.instance;
    final countOut = calloc<ffi.Size>();
    final nameOut = calloc<ffi.Pointer<ffi.Char>>();
    try {
      library.check(
        input
            ? ref.SessionGetInputCount.asFunction<
                ffi.Pointer<bg.OrtStatus> Function(ffi.Pointer<bg.OrtSession>,
                    ffi.Pointer<ffi.Size>)>()(session, countOut)
            : ref.SessionGetOutputCount.asFunction<
                ffi.Pointer<bg.OrtStatus> Function(ffi.Pointer<bg.OrtSession>,
                    ffi.Pointer<ffi.Size>)>()(session, countOut),
        input ? 'SessionGetInputCount' : 'SessionGetOutputCount',
      );
      final count = countOut.value;
      final names = <String>[];
      for (var i = 0; i < count; i++) {
        library.check(
          input
              ? ref.SessionGetInputName.asFunction<
                      ffi.Pointer<bg.OrtStatus> Function(
                          ffi.Pointer<bg.OrtSession>,
                          int,
                          ffi.Pointer<bg.OrtAllocator>,
                          ffi.Pointer<ffi.Pointer<ffi.Char>>)>()(
                  session, i, allocator.allocator, nameOut)
              : ref.SessionGetOutputName.asFunction<
                      ffi.Pointer<bg.OrtStatus> Function(
                          ffi.Pointer<bg.OrtSession>,
                          int,
                          ffi.Pointer<bg.OrtAllocator>,
                          ffi.Pointer<ffi.Pointer<ffi.Char>>)>()(
                  session, i, allocator.allocator, nameOut),
          input ? 'SessionGetInputName' : 'SessionGetOutputName',
        );
        names.add(nameOut.value.cast<Utf8>().toDartString());
        allocator.free(nameOut.value.cast<ffi.Void>());
      }
      return List<String>.unmodifiable(names);
    } finally {
      calloc
        ..free(countOut)
        ..free(nameOut);
    }
  }

  /// Builds a reusable plan for running this session with a fixed set of input
  /// and output names.
  ///
  /// The name strings are encoded once and kept for the life of the session.
  /// This is the difference that removes the per-step native leak: a decoding
  /// loop reuses one plan instead of re-encoding 42 C strings per token.
  OrtRunPlan plan(List<String> inputs, List<String> outputs) {
    final created = OrtRunPlan._(this, inputs, outputs);
    _plans.add(created);
    return created;
  }

  /// The run options shared by every plan on this session.
  ffi.Pointer<bg.OrtRunOptions> get runOptions => _runOptions;

  /// Releases the session, its run options and every plan built from it.
  void release() {
    if (_released) return;
    _released = true;
    for (final plan in _plans) {
      plan._release();
    }
    _plans.clear();
    final ref = _library.api.ref;
    if (_runOptions != ffi.nullptr) {
      ref.ReleaseRunOptions.asFunction<
          void Function(ffi.Pointer<bg.OrtRunOptions>)>()(_runOptions);
      _runOptions = ffi.nullptr;
    }
    if (_owned) {
      ref.ReleaseSession.asFunction<
          void Function(ffi.Pointer<bg.OrtSession>)>()(_ptr);
    }
  }
}

/// A pre-encoded call signature for [OrtSession].
///
/// Holds the native arrays for input names, input values, output names and
/// output values. Running a plan writes into those arrays instead of building
/// them, so a decoding step performs no allocation on the Dart or the native
/// side.
class OrtRunPlan {
  OrtRunPlan._(this._session, List<String> inputs, List<String> outputs)
      : inputNames = List<String>.unmodifiable(inputs),
        outputNames = List<String>.unmodifiable(outputs),
        _inputNamePtrs = calloc<ffi.Pointer<ffi.Char>>(inputs.length),
        _inputValuePtrs = calloc<ffi.Pointer<bg.OrtValue>>(inputs.length),
        _outputNamePtrs = calloc<ffi.Pointer<ffi.Char>>(outputs.length),
        _outputValuePtrs = calloc<ffi.Pointer<bg.OrtValue>>(outputs.length) {
    for (var i = 0; i < inputs.length; i++) {
      _inputNamePtrs[i] = inputs[i].toNativeUtf8().cast<ffi.Char>();
    }
    for (var i = 0; i < outputs.length; i++) {
      _outputNamePtrs[i] = outputs[i].toNativeUtf8().cast<ffi.Char>();
    }
    _index = <String, int>{
      for (var i = 0; i < inputs.length; i++) inputs[i]: i,
    };
  }

  final OrtSession _session;

  /// Input names, in the order the plan feeds them.
  final List<String> inputNames;

  /// Output names, in the order [run] returns them.
  final List<String> outputNames;

  final ffi.Pointer<ffi.Pointer<ffi.Char>> _inputNamePtrs;
  final ffi.Pointer<ffi.Pointer<bg.OrtValue>> _inputValuePtrs;
  final ffi.Pointer<ffi.Pointer<ffi.Char>> _outputNamePtrs;
  final ffi.Pointer<ffi.Pointer<bg.OrtValue>> _outputValuePtrs;
  late final Map<String, int> _index;
  bool _released = false;

  /// Binds [tensor] to the input at [position].
  void setInputAt(int position, OrtTensor tensor) {
    _inputValuePtrs[position] = tensor.pointer;
  }

  /// Binds [tensor] to the input called [name].
  void setInput(String name, OrtTensor tensor) {
    final position = _index[name];
    if (position == null) {
      throw ArgumentError.value(name, 'name', 'not an input of this plan');
    }
    _inputValuePtrs[position] = tensor.pointer;
  }

  /// Drops every bound input value.
  ///
  /// Called after a generation pass so that tensors ONNX Runtime has since
  /// freed cannot be handed back to it by a later run.
  void clearInputs() {
    for (var i = 0; i < inputNames.length; i++) {
      _inputValuePtrs[i] = ffi.nullptr;
    }
  }

  /// Runs the session, filling [into] with the outputs.
  ///
  /// [into] must have [outputNames]`.length` entries; each is replaced with a
  /// borrowed tensor that the caller is responsible for releasing.
  void run(List<OrtTensor?> into) {
    if (_released) {
      throw StateError('This run plan has been released.');
    }
    for (var i = 0; i < outputNames.length; i++) {
      _outputValuePtrs[i] = ffi.nullptr;
    }
    final library = OrtLibrary.instance;
    library.check(
      library.api.ref.Run.asFunction<
          ffi.Pointer<bg.OrtStatus> Function(
              ffi.Pointer<bg.OrtSession>,
              ffi.Pointer<bg.OrtRunOptions>,
              ffi.Pointer<ffi.Pointer<ffi.Char>>,
              ffi.Pointer<ffi.Pointer<bg.OrtValue>>,
              int,
              ffi.Pointer<ffi.Pointer<ffi.Char>>,
              int,
              ffi.Pointer<ffi.Pointer<bg.OrtValue>>)>()(
        _session._ptr,
        _session._runOptions,
        _inputNamePtrs,
        _inputValuePtrs,
        inputNames.length,
        _outputNamePtrs,
        outputNames.length,
        _outputValuePtrs,
      ),
      'Run',
    );
    for (var i = 0; i < outputNames.length; i++) {
      into[i] = OrtTensor.borrowed(_outputValuePtrs[i]);
    }
  }

  void _release() {
    if (_released) return;
    _released = true;
    for (var i = 0; i < inputNames.length; i++) {
      calloc.free(_inputNamePtrs[i]);
    }
    for (var i = 0; i < outputNames.length; i++) {
      calloc.free(_outputNamePtrs[i]);
    }
    calloc
      ..free(_inputNamePtrs)
      ..free(_inputValuePtrs)
      ..free(_outputNamePtrs)
      ..free(_outputValuePtrs);
  }
}
