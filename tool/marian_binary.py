"""Reads Marian's native `.bin` parameter format.

Mozilla's Firefox Translations student models are published as Marian binaries,
not as PyTorch or ONNX. The format is documented only by its implementation
(`marian/src/common/binary.cpp`); this is a direct port of `loadItems`:

    size_t   binaryFileVersion            (== 1)
    size_t   numHeaders
    Header   headers[numHeaders]          { nameLength, type, shapeLength, dataLength }
    char     names[]                      numHeaders NUL-terminated strings
    int32    shapes[]                     shapeLength dims per item
    size_t   padOffset                    followed by padOffset bytes of padding
    char     data[]                       the blobs, in header order

`type` is Marian's `Type` enum: a class in the high bytes and the element size
in the low byte. Only two appear in a published model — plain `float32`
(0x0404) and `intgemm8` (0x4101), an int8 tensor with its quantisation
multiplier appended as a trailing float.
"""
from __future__ import annotations

import struct
from typing import Dict, List, Tuple

import numpy as np

FLOAT32 = 0x0404
INTGEMM_CLASS = 0x4100

Item = Tuple[int, List[int], bytes]  # (type, shape, blob)


def load(path: str) -> Tuple[int, Dict[str, Item]]:
    """Returns `(fileVersion, {name: (type, shape, blob)})`."""
    buf = open(path, "rb").read()
    off = 0

    def read(fmt: str):
        nonlocal off
        values = struct.unpack_from(fmt, buf, off)
        off += struct.calcsize(fmt)
        return values

    version, count = read("<QQ")
    if version != 1:
        raise ValueError(f"{path}: unsupported Marian binary version {version}")
    headers = [read("<QQQQ") for _ in range(count)]

    names = []
    for name_length, _, _, _ in headers:
        names.append(buf[off : off + name_length].rstrip(b"\0").decode("utf-8"))
        off += name_length

    shapes = []
    for _, _, shape_length, _ in headers:
        shapes.append(list(struct.unpack_from(f"<{shape_length}i", buf, off)))
        off += 4 * shape_length

    (padding,) = read("<Q")
    off += padding

    items: Dict[str, Item] = {}
    for name, header, shape in zip(names, headers, shapes):
        data_length = header[3]
        items[name] = (header[1], shape, buf[off : off + data_length])
        off += data_length
    return version, items


def config(items: Dict[str, Item]) -> str:
    """The YAML architecture description Marian stores inside the model."""
    return items["special:model.yml"][2].rstrip(b"\0").decode("utf-8")


def parameters(items: Dict[str, Item]) -> Dict[str, np.ndarray]:
    """Dequantises every parameter to float32, in the orientation Marian uses.

    `marian-conv --gemm-type intgemm8` writes matmul weights through
    `PrepareBQuantizedTransposed`, so an `intgemm8` blob holds the *transpose*
    of the shape recorded next to it. `Wemb` is the one exception: it is read by
    an embedding lookup rather than used as a matmul operand, so it keeps its
    row layout. Getting this backwards produces a model that still emits fluent
    French — just not French that has anything to do with the input — so it is
    worth restating that the rule was established by measurement, not by
    reading: under the wrong orientation the nearest neighbours of `▁Hello` are
    `dan`, `-1`, `garde`; under this one they are `▁Hell`, `▁Bonjour`, `ello`.
    """
    out: Dict[str, np.ndarray] = {}
    for name, (dtype, shape, blob) in items.items():
        if name.startswith("special:"):
            continue
        count = int(np.prod(shape)) if shape else 1
        if dtype & 0xFF00 == INTGEMM_CLASS:
            quantised = np.frombuffer(blob[:count], dtype=np.int8).astype(np.float32)
            (multiplier,) = struct.unpack_from("<f", blob, count)
            weight = quantised / multiplier
            if name != "Wemb" and len(shape) == 2 and min(shape) > 1:
                weight = np.ascontiguousarray(
                    weight.reshape(shape[1], shape[0]).T)
            else:
                weight = weight.reshape(shape)
        elif dtype == FLOAT32:
            weight = np.frombuffer(blob[: count * 4], dtype=np.float32).reshape(shape)
        else:
            raise ValueError(f"{name}: unhandled Marian type {dtype:#x}")
        out[name] = np.ascontiguousarray(weight, dtype=np.float32)
    return out
