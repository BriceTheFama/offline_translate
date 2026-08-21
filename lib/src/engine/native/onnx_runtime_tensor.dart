import 'dart:ffi' as ffi;
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import 'onnx_runtime.dart';
import 'onnx_runtime_allocator.dart';
import 'onnx_runtime_bindings.dart' as bg;

/// A tensor handed to or received from ONNX Runtime.
///
/// Two kinds of tensor exist here and they are freed differently:
///
/// * **owned** — created by [OrtTensor.int64], [OrtTensor.float32] or
///   [OrtTensor.boolean]. The data buffer is allocated by this package and
///   ONNX Runtime only borrows it, so [release] frees both the `OrtValue` and
///   the buffer. These are the tensors the engine keeps and *reuses*: the
///   buffer can be rewritten between runs with [asInt64List] and friends, so a
///   decoding step needs no allocation at all.
/// * **borrowed** — produced by [OrtSession.run]. ONNX Runtime owns the
///   memory; [release] only frees the `OrtValue`.
class OrtTensor {
  OrtTensor._(this._value, this._buffer, this._shape);

  final ffi.Pointer<bg.OrtValue> _value;
  final ffi.Pointer<ffi.Void> _buffer;
  final List<int> _shape;

  bool _released = false;

  /// The raw handle, for passing back into ONNX Runtime.
  ffi.Pointer<bg.OrtValue> get pointer => _value;

  /// Shape this tensor was created with. Empty for tensors returned by a run,
  /// whose shape is queried on demand with [shapeOf].
  List<int> get shape => _shape;

  /// Whether [release] has already been called.
  bool get isReleased => _released;

  static ffi.Pointer<bg.OrtValue> _create(
    ffi.Pointer<ffi.Void> data,
    int byteLength,
    List<int> shape,
    int elementType,
  ) {
    final library = OrtLibrary.instance;
    final dims = calloc<ffi.Int64>(shape.isEmpty ? 1 : shape.length);
    for (var i = 0; i < shape.length; i++) {
      dims[i] = shape[i];
    }
    final out = calloc<ffi.Pointer<bg.OrtValue>>();
    try {
      library.check(
        library.api.ref.CreateTensorWithDataAsOrtValue.asFunction<
            ffi.Pointer<bg.OrtStatus> Function(
                ffi.Pointer<bg.OrtMemoryInfo>,
                ffi.Pointer<ffi.Void>,
                int,
                ffi.Pointer<ffi.Int64>,
                int,
                int,
                ffi.Pointer<ffi.Pointer<bg.OrtValue>>)>()(
          OrtCpuAllocator.instance.memoryInfo,
          data,
          byteLength,
          dims,
          shape.length,
          elementType,
          out,
        ),
        'CreateTensorWithDataAsOrtValue',
      );
      return out.value;
    } finally {
      calloc
        ..free(dims)
        ..free(out);
    }
  }

  /// Allocates an `int64` tensor of [shape], zero-filled.
  factory OrtTensor.int64(List<int> shape) {
    final count = _elementCount(shape);
    final buffer = calloc<ffi.Int64>(count == 0 ? 1 : count);
    final value = _create(buffer.cast(), count * 8, shape,
        bg.ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64.value);
    return OrtTensor._(value, buffer.cast(), shape);
  }

  /// Allocates a `float32` tensor of [shape], zero-filled.
  factory OrtTensor.float32(List<int> shape) {
    final count = _elementCount(shape);
    final buffer = calloc<ffi.Float>(count == 0 ? 1 : count);
    final value = _create(buffer.cast(), count * 4, shape,
        bg.ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT.value);
    return OrtTensor._(value, buffer.cast(), shape);
  }

  /// Allocates a `bool` tensor of [shape] holding [value] everywhere.
  factory OrtTensor.boolean(List<int> shape, {required bool value}) {
    final count = _elementCount(shape);
    final buffer = calloc<ffi.Uint8>(count == 0 ? 1 : count);
    for (var i = 0; i < count; i++) {
      buffer[i] = value ? 1 : 0;
    }
    final tensor = _create(buffer.cast(), count, shape,
        bg.ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL.value);
    return OrtTensor._(tensor, buffer.cast(), shape);
  }

  /// Wraps a tensor produced by ONNX Runtime. The data belongs to the runtime.
  factory OrtTensor.borrowed(ffi.Pointer<bg.OrtValue> value) =>
      OrtTensor._(value, ffi.nullptr, const <int>[]);

  /// A second view of [source]'s buffer with a different [shape].
  ///
  /// Creating an `OrtValue` is a pointer wrap, not a copy, so a fixed
  /// maximum-width buffer can be allocated once and presented to the model at
  /// whatever length the current input needs. Releasing the view leaves the
  /// underlying buffer alone; [source] still owns it.
  factory OrtTensor.viewOf(OrtTensor source, List<int> shape,
      {required int bytesPerElement, int? elementType}) {
    final count = _elementCount(shape);
    final value = _create(
      source._buffer,
      count * bytesPerElement,
      shape,
      elementType ??
          bg.ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
              .value,
    );
    return OrtTensor._(value, ffi.nullptr, shape);
  }

  static int _elementCount(List<int> shape) =>
      shape.fold<int>(1, (a, b) => a * b);

  /// A view over this tensor's `int64` data. Writing through it changes what
  /// the next run sees, with no reallocation.
  Int64List asInt64List(int length) =>
      _dataPointer().cast<ffi.Int64>().asTypedList(length);

  /// A view over this tensor's `float32` data.
  Float32List asFloat32List(int length) =>
      _dataPointer().cast<ffi.Float>().asTypedList(length);

  /// Reads a single `int64` element at [index]. Used for the decoder's
  /// `next_token` output, so no list is materialised per generated token.
  int int64At(int index) => _dataPointer().cast<ffi.Int64>()[index];

  ffi.Pointer<ffi.Void> _dataPointer() {
    if (_buffer != ffi.nullptr) return _buffer;
    final library = OrtLibrary.instance;
    final out = calloc<ffi.Pointer<ffi.Void>>();
    try {
      library.check(
        library.api.ref.GetTensorMutableData.asFunction<
            ffi.Pointer<bg.OrtStatus> Function(ffi.Pointer<bg.OrtValue>,
                ffi.Pointer<ffi.Pointer<ffi.Void>>)>()(_value, out),
        'GetTensorMutableData',
      );
      return out.value;
    } finally {
      calloc.free(out);
    }
  }

  /// Queries the runtime for this tensor's shape.
  List<int> shapeOf() {
    final library = OrtLibrary.instance;
    final ref = library.api.ref;
    final infoOut = calloc<ffi.Pointer<bg.OrtTensorTypeAndShapeInfo>>();
    final countOut = calloc<ffi.Size>();
    try {
      library.check(
        ref.GetTensorTypeAndShape.asFunction<
            ffi.Pointer<bg.OrtStatus> Function(ffi.Pointer<bg.OrtValue>,
                ffi.Pointer<ffi.Pointer<bg.OrtTensorTypeAndShapeInfo>>)>()(
          _value,
          infoOut,
        ),
        'GetTensorTypeAndShape',
      );
      final info = infoOut.value;
      try {
        library.check(
          ref.GetDimensionsCount.asFunction<
              ffi.Pointer<bg.OrtStatus> Function(
                  ffi.Pointer<bg.OrtTensorTypeAndShapeInfo>,
                  ffi.Pointer<ffi.Size>)>()(info, countOut),
          'GetDimensionsCount',
        );
        final count = countOut.value;
        final dims = calloc<ffi.Int64>(count == 0 ? 1 : count);
        try {
          library.check(
            ref.GetDimensions.asFunction<
                ffi.Pointer<bg.OrtStatus> Function(
                    ffi.Pointer<bg.OrtTensorTypeAndShapeInfo>,
                    ffi.Pointer<ffi.Int64>,
                    int)>()(info, dims, count),
            'GetDimensions',
          );
          return <int>[for (var i = 0; i < count; i++) dims[i]];
        } finally {
          calloc.free(dims);
        }
      } finally {
        ref.ReleaseTensorTypeAndShapeInfo.asFunction<
            void Function(ffi.Pointer<bg.OrtTensorTypeAndShapeInfo>)>()(info);
      }
    } finally {
      calloc
        ..free(infoOut)
        ..free(countOut);
    }
  }

  /// Releases the `OrtValue`, and the data buffer when this package owns it.
  void release() {
    if (_released) return;
    _released = true;
    OrtLibrary.instance.api.ref.ReleaseValue
        .asFunction<void Function(ffi.Pointer<bg.OrtValue>)>()(_value);
    if (_buffer != ffi.nullptr) calloc.free(_buffer);
  }
}
