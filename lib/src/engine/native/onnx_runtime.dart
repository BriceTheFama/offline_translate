import 'dart:ffi' as ffi;
import 'dart:io';

import 'package:ffi/ffi.dart';

import '../../exceptions/exceptions.dart';
import 'onnx_runtime_bindings.dart' as bg;

/// Resolves the ONNX Runtime shared library and exposes its C API table.
///
/// How the library is reached differs per platform:
///
/// * **Android** — `libonnxruntime.so` is extracted from the official AAR by
///   this package's `android/build.gradle` and packaged as a JNI library, so
///   it is opened by name.
/// * **iOS / macOS** — the `onnxruntime-c` pod vendors a *static* xcframework,
///   so the symbols end up inside the application binary and are looked up in
///   the current process. `src/ort_shim.c` holds a reference to
///   `OrtGetApiBase` so the linker cannot dead-strip the archive.
/// * **anywhere else** — set [overrideLibraryPath] before the first use, which
///   is how the pure-Dart harness and the desktop builds find a local copy.
class OrtLibrary {
  OrtLibrary._(this._library, this.bindings, this.api, this.version);

  final ffi.DynamicLibrary _library;

  /// Generated bindings bound to the resolved library.
  final bg.OrtBindings bindings;

  /// The API function table for [version].
  final ffi.Pointer<bg.OrtApi> api;

  /// The `ORT_API_VERSION` that was successfully requested.
  final int version;

  /// The API version this package's bindings were generated against.
  static const int builtAgainstApiVersion = 29;

  /// Explicit path to `libonnxruntime.{so,dylib}`, for platforms where the
  /// library is not already in the process. Must be set before [instance].
  static String? overrideLibraryPath;

  static OrtLibrary? _instance;

  /// The process-wide ONNX Runtime handle, opened on first use.
  static OrtLibrary get instance => _instance ??= _open();

  /// Whether the runtime has been resolved already.
  static bool get isOpen => _instance != null;

  static OrtLibrary _open() {
    final library = _resolve();
    final bindings = bg.OrtBindings(library);
    final base = bindings.OrtGetApiBase();
    if (base == ffi.nullptr) {
      throw const TranslationEngineException(
          'OrtGetApiBase() returned null; the ONNX Runtime library is present '
          'but unusable.');
    }
    final getApi =
        base.ref.GetApi.asFunction<ffi.Pointer<bg.OrtApi> Function(int)>();

    // Ask for the version the bindings were generated against, then walk back:
    // the C API table is append-only, so an older runtime still answers for an
    // older version and every offset this package uses stays valid.
    for (var version = builtAgainstApiVersion; version >= 16; version--) {
      final api = getApi(version);
      if (api != ffi.nullptr) {
        return OrtLibrary._(library, bindings, api, version);
      }
    }
    final actual = base.ref.GetVersionString
        .asFunction<ffi.Pointer<ffi.Char> Function()>()();
    throw TranslationEngineException(
        'This ONNX Runtime (${actual.cast<Utf8>().toDartString()}) is older '
        'than API version 16, which offline_translator requires.');
  }

  static ffi.DynamicLibrary _resolve() {
    final override = overrideLibraryPath;
    if (override != null) return ffi.DynamicLibrary.open(override);
    if (Platform.isAndroid) return ffi.DynamicLibrary.open('libonnxruntime.so');
    if (Platform.isIOS || Platform.isMacOS) return ffi.DynamicLibrary.process();
    if (Platform.isLinux) return ffi.DynamicLibrary.open('libonnxruntime.so');
    if (Platform.isWindows) return ffi.DynamicLibrary.open('onnxruntime.dll');
    throw TranslationEngineException(
        'Unsupported platform ${Platform.operatingSystem}. Set '
        'OrtLibrary.overrideLibraryPath to a local ONNX Runtime build.');
  }

  /// Runtime version string, e.g. `1.29.0`.
  String get versionString {
    final base = bindings.OrtGetApiBase();
    final ptr = base.ref.GetVersionString
        .asFunction<ffi.Pointer<ffi.Char> Function()>()();
    return ptr.cast<Utf8>().toDartString();
  }

  /// Looks up a symbol that only exists in some builds, such as the Android
  /// NNAPI entry point. Returns `null` when it is absent.
  ffi.Pointer<ffi.NativeFunction<T>>? lookupOrNull<T extends Function>(
      String symbol) {
    if (!_library.providesSymbol(symbol)) return null;
    return _library.lookup<ffi.NativeFunction<T>>(symbol);
  }

  /// Throws if [status] is an error, and always releases it.
  void check(ffi.Pointer<bg.OrtStatus> status, String what) {
    if (status == ffi.nullptr) return;
    final ref = api.ref;
    final code =
        ref.GetErrorCode.asFunction<int Function(ffi.Pointer<bg.OrtStatus>)>()(
            status);
    final message = ref.GetErrorMessage.asFunction<
            ffi.Pointer<ffi.Char> Function(ffi.Pointer<bg.OrtStatus>)>()(status)
        .cast<Utf8>()
        .toDartString();
    ref.ReleaseStatus.asFunction<void Function(ffi.Pointer<bg.OrtStatus>)>()(
        status);
    throw TranslationEngineException(
        '$what failed (ORT error $code): $message');
  }
}

/// The ONNX Runtime environment. One per process, created on first use.
///
/// Telemetry is switched off explicitly. ONNX Runtime's Apple and Android
/// builds do not phone home from the C API, but this package promises no
/// network activity after installation, so the switch is thrown anyway.
class OrtEnv {
  OrtEnv._(this._library, this._ptr, {this.owned = true});

  /// Whether this isolate created the environment and must release it.
  final bool owned;

  final OrtLibrary _library;
  final ffi.Pointer<bg.OrtEnv> _ptr;

  static OrtEnv? _instance;

  /// The process-wide environment.
  static OrtEnv get instance => _instance ??= _create();

  /// Adopts an environment created on another isolate.
  ///
  /// ONNX Runtime wants exactly one `OrtEnv` per process, so a worker isolate
  /// must reuse the one the owning isolate made rather than create a second.
  /// The adopted handle is non-owning.
  static void adopt(int address) {
    _instance ??= OrtEnv._(
        OrtLibrary.instance, ffi.Pointer<bg.OrtEnv>.fromAddress(address),
        owned: false);
  }

  /// Address of the underlying environment, for sharing with another isolate.
  int get address => _ptr.address;

  /// The raw handle, for passing to session creation.
  ffi.Pointer<bg.OrtEnv> get pointer => _ptr;

  static OrtEnv _create() {
    final library = OrtLibrary.instance;
    final out = calloc<ffi.Pointer<bg.OrtEnv>>();
    final name = 'offline_translate'.toNativeUtf8();
    try {
      library.check(
        library.api.ref.CreateEnv.asFunction<
            ffi.Pointer<bg.OrtStatus> Function(int, ffi.Pointer<ffi.Char>,
                ffi.Pointer<ffi.Pointer<bg.OrtEnv>>)>()(
          bg.OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR.value,
          name.cast<ffi.Char>(),
          out,
        ),
        'CreateEnv',
      );
      final env = OrtEnv._(library, out.value);
      library.check(
        library.api.ref.DisableTelemetryEvents.asFunction<
            ffi.Pointer<bg.OrtStatus> Function(
                ffi.Pointer<bg.OrtEnv>)>()(env._ptr),
        'DisableTelemetryEvents',
      );
      return env;
    } finally {
      calloc
        ..free(out)
        ..free(name);
    }
  }

  /// Releases the environment. Only meaningful at process shutdown; sessions
  /// must be released first.
  void release() {
    if (owned) {
      _library.api.ref.ReleaseEnv
          .asFunction<void Function(ffi.Pointer<bg.OrtEnv>)>()(_ptr);
    }
    _instance = null;
  }
}
