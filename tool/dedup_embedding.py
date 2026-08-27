"""Stores the tied embedding once, shared by both graphs.

Marian ties the source embedding, the target embedding and the output
projection to one matrix. optimum exports encoder and decoder as separate ONNX
files, so that matrix is written twice — verified bit-identical, and worth
29 MB of the 104 MB bundle.

ONNX external data lets tensors in several files point at the same blob, so
both graphs are rewritten to reference one `embedding.data`. The saving is on
disk and download size: measured in fresh processes, resident memory is
unchanged (339-350 MB either way), because ONNX Runtime pre-packs weights into
its own buffers on first inference and those copies dominate the working set.

The data files are written here rather than through `onnx.save(
save_as_external_data=True)`, because that helper redirects *every* large
tensor to one location and would overwrite the shared pointer.
"""
import os, shutil, sys
import onnx
from onnx import numpy_helper

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)

EMB = {'encoder.onnx': 'embed_tokens.weight_quantized',
       'decoder.onnx': 'model.shared.weight_quantized_merged_0'}
ALIGN = 4096  # ONNX Runtime only memory-maps page-aligned external tensors


def point_at(init, location, offset, length):
    init.ClearField('raw_data')
    del init.external_data[:]
    init.data_location = onnx.TensorProto.EXTERNAL
    for key, value in (('location', location), ('offset', str(offset)),
                       ('length', str(length))):
        entry = init.external_data.add()
        entry.key, entry.value = key, value


enc = onnx.load(os.path.join(src, 'encoder.onnx'))
shared = None
for init in enc.graph.initializer:
    if init.name == EMB['encoder.onnx']:
        shared = numpy_helper.to_array(init).tobytes()
assert shared is not None
with open(os.path.join(dst, 'embedding.data'), 'wb') as fh:
    fh.write(shared)
print(f'embedding.data: {len(shared)/1048576:.1f} MB (shared by both graphs)')

for name in ('encoder.onnx', 'decoder.onnx'):
    model = onnx.load(os.path.join(src, name))
    base = name.replace('.onnx', '')
    blob = os.path.join(dst, f'{base}.data')
    written = 0
    with open(blob, 'wb') as fh:
        for init in model.graph.initializer:
            if init.name == EMB[name]:
                payload = numpy_helper.to_array(init).tobytes()
                assert payload == shared, f'{name}: embedding is not the shared one'
                point_at(init, 'embedding.data', 0, len(shared))
                continue
            payload = numpy_helper.to_array(init).tobytes()
            if len(payload) < 1024:
                # Small tensors — biases, layer norms, and the quantisation
                # scales and zero-points — go back inline. They must be reset
                # explicitly: onnx.load() populates raw_data but leaves
                # data_location EXTERNAL with the *source* file's offsets, and
                # saving that would silently read them from the wrong place in
                # the new blob. That corrupts the scales, which changes the
                # output without failing anything.
                init.data_location = onnx.TensorProto.DEFAULT
                del init.external_data[:]
                init.raw_data = payload
                continue
            pad = (-fh.tell()) % ALIGN
            fh.write(b'\0' * pad)
            offset = fh.tell()
            fh.write(payload)
            written += len(payload)
            point_at(init, f'{base}.data', offset, len(payload))
    onnx.save(model, os.path.join(dst, name))
    os.chmod(blob, 0o644)
    print(f'{name}: graph {os.path.getsize(os.path.join(dst, name))/1024:.0f} KB, '
          f'{base}.data {os.path.getsize(blob)/1048576:.1f} MB')

for extra in ('source.spm', 'vocab.json', 'manifest.json'):
    shutil.copyfile(os.path.join(src, extra), os.path.join(dst, extra))

after = sum(os.path.getsize(os.path.join(dst, f)) for f in os.listdir(dst))
before = sum(os.path.getsize(os.path.join(src, f)) for f in os.listdir(src))
print(f'\n{before/1048576:.1f} MB -> {after/1048576:.1f} MB '
      f'({100*(before-after)/before:.0f}% smaller)')
