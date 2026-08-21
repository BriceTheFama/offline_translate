import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';

import 'onnx_runtime.dart';
import 'onnx_runtime_bindings.dart' as bg;

/// ONNX Runtime's CPU allocator and memory descriptor.
///
/// Both are process-wide and owned by the runtime: `GetAllocatorWithDefaultOptions`
/// hands back a singleton that must **not** be released, and one CPU
/// `OrtMemoryInfo` is enough for every tensor this package creates, so it is
/// built once instead of per call.
class OrtCpuAllocator {
  OrtCpuAllocator._(this._library, this.allocator, this.memoryInfo);

  final OrtLibrary _library;

  /// The default CPU allocator. Owned by ONNX Runtime; never release it.
  final ffi.Pointer<bg.OrtAllocator> allocator;

  /// CPU memory descriptor used when wrapping Dart-owned buffers as tensors.
  final ffi.Pointer<bg.OrtMemoryInfo> memoryInfo;

  static OrtCpuAllocator? _instance;

  /// The process-wide CPU allocator, created on first use.
  static OrtCpuAllocator get instance => _instance ??= _create();

  static OrtCpuAllocator _create() {
    final library = OrtLibrary.instance;
    final ref = library.api.ref;

    final allocatorOut = calloc<ffi.Pointer<bg.OrtAllocator>>();
    final memoryOut = calloc<ffi.Pointer<bg.OrtMemoryInfo>>();
    try {
      library.check(
        ref.GetAllocatorWithDefaultOptions.asFunction<
            ffi.Pointer<bg.OrtStatus> Function(
                ffi.Pointer<ffi.Pointer<bg.OrtAllocator>>)>()(allocatorOut),
        'GetAllocatorWithDefaultOptions',
      );
      library.check(
        ref.CreateCpuMemoryInfo.asFunction<
            ffi.Pointer<bg.OrtStatus> Function(
                int, int, ffi.Pointer<ffi.Pointer<bg.OrtMemoryInfo>>)>()(
          bg.OrtAllocatorType.OrtArenaAllocator.value,
          bg.OrtMemType.OrtMemTypeDefault.value,
          memoryOut,
        ),
        'CreateCpuMemoryInfo',
      );
      return OrtCpuAllocator._(library, allocatorOut.value, memoryOut.value);
    } finally {
      calloc
        ..free(allocatorOut)
        ..free(memoryOut);
    }
  }

  /// Frees a buffer that ONNX Runtime allocated through [allocator], such as
  /// the strings returned by `SessionGetInputName`.
  void free(ffi.Pointer<ffi.Void> pointer) {
    _library.api.ref.AllocatorFree.asFunction<
        ffi.Pointer<bg.OrtStatus> Function(ffi.Pointer<bg.OrtAllocator>,
            ffi.Pointer<ffi.Void>)>()(allocator, pointer);
  }
}
