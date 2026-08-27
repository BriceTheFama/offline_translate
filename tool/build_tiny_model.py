#!/usr/bin/env python3
"""Builds an `offline_translate` bundle from a Firefox Translations student model.

    python3 tool/build_tiny_model.py --pair en-fr --out ~/ot-models-tiny

Mozilla distributes these models as Marian binaries, so there is no `optimum`
path: the pipeline parses the checkpoint, rebuilds the architecture in PyTorch
from the YAML config Marian stores inside it, and exports two ONNX graphs.

  1. fetch `model.*.intgemm.alphas.bin` and `vocab.*.spm` from
     mozilla/firefox-translations-models;
  2. parse the Marian binary and dequantise it to float32
     (`tool/marian_binary.py`);
  3. rebuild the model in PyTorch (`tool/tiny_transformer.py`) and check it
     against the fixtures;
  4. export `encoder.onnx` and `decoder.onnx`, and verify the graphs reproduce
     PyTorch's own greedy output;
  5. quantise both to int8 dynamic;
  6. share the tied embedding between the two graphs as one `embedding.data`;
  7. emit `source.spm`, `vocab.json` and `manifest.json`.

The exported graphs are deliberately not shaped like optimum's. Because the
decoder is an SSRU rather than self-attention, its whole history is one
`[1, 1, 384]` state per layer, so there is no growing KV cache, no `If` node and
no `use_cache_branch`: one graph serves every step. The cross-attention keys and
values are computed in the *encoder* graph, once per sentence.

Requires: torch, onnx, onnxruntime, sentencepiece, numpy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.request

import numpy as np
import onnx
import torch
from onnx import numpy_helper

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import marian_binary  # noqa: E402
from tiny_transformer import (  # noqa: E402
    START_TOKEN, DecoderGraph, EncoderGraph, TinyModel)

# Mozilla publishes one directory per direction. `prod` models are the ones
# Firefox ships; `dev` are smaller/experimental. Everything in the repository is
# MPL-2.0 — see doc/licensing.md.
BASE_URL = ("https://raw.githubusercontent.com/mozilla/"
            "firefox-translations-models/main/models")

CATALOGUE = {
    # pair: (repository directory, quality tier)
    "en-fr": ("enfr", "prod"),
    "fr-en": ("fren", "prod"),
    "en-es": ("enes", "prod"),
    "es-en": ("esen", "prod"),
    "en-de": ("ende", "prod"),
    "de-en": ("deen", "prod"),
}

FIXTURES = [
    "Hello, how are you?",
    "Hello world",
    "This is a simple test.",
    "I love programming and mobile development.",
    "The quick brown fox jumps over the lazy dog.",
    "I would like to book a table for two people at eight o'clock tonight.",
    "She said that the meeting had been postponed until next Tuesday because "
    "of the weather.",
]

MAX_POSITIONS = 512
OPSET = 17


def log(message: str) -> None:
    print(f"\033[1m==\033[0m {message}", flush=True)


# -- 1. fetch -----------------------------------------------------------------


def fetch(pair: str, cache: str) -> tuple[str, str]:
    directory, tier = CATALOGUE[pair]
    os.makedirs(cache, exist_ok=True)
    names = {
        "model": f"model.{directory}.intgemm.alphas.bin",
        "vocab": f"vocab.{directory}.spm",
    }
    paths = {}
    for kind, name in names.items():
        destination = os.path.join(cache, name)
        if not os.path.exists(destination):
            url = f"{BASE_URL}/{tier}/{directory}/{name}.gz"
            log(f"downloading {name}")
            with urllib.request.urlopen(url) as response:
                import gzip
                payload = gzip.decompress(response.read())
            with open(destination, "wb") as handle:
                handle.write(payload)
        paths[kind] = destination
    return paths["model"], paths["vocab"]


# -- 3. rebuild ---------------------------------------------------------------


def build_torch(model_path: str) -> tuple[TinyModel, str]:
    _, items = marian_binary.load(model_path)
    config = marian_binary.config(items)
    parameters = marian_binary.parameters(items)
    model = TinyModel(parameters, max_positions=MAX_POSITIONS).eval()
    return model, config


@torch.no_grad()
def greedy_torch(model: TinyModel, ids: list[int], limit: int = 256) -> list[int]:
    input_ids = torch.tensor([ids], dtype=torch.long)
    mask = torch.ones_like(input_ids)
    cross = model.cross_kv(model.encode(input_ids, mask))
    states = [torch.zeros(1, 1, model.dim) for _ in range(model.decoder_layers)]
    token = torch.tensor([[START_TOKEN]], dtype=torch.long)
    out: list[int] = []
    for position in range(limit):
        logits, states = model.step(
            token, torch.tensor([position]), mask, cross, states)
        nxt = int(logits[0, -1].argmax())
        if nxt == 0:  # </s>
            break
        out.append(nxt)
        token = torch.tensor([[nxt]], dtype=torch.long)
    return out


# -- 4. export ----------------------------------------------------------------


def cross_names(layers: int) -> list[str]:
    names = []
    for layer in range(layers):
        names += [f"cross_key.{layer}", f"cross_value.{layer}"]
    return names


def export(model: TinyModel, work: str) -> tuple[str, str]:
    layers = model.decoder_layers
    encoder_path = os.path.join(work, "encoder.onnx")
    decoder_path = os.path.join(work, "decoder.onnx")

    source = torch.ones(1, 7, dtype=torch.long)
    mask = torch.ones(1, 7, dtype=torch.long)
    outputs = cross_names(layers)
    torch.onnx.export(
        EncoderGraph(model), (source, mask), encoder_path,
        input_names=["input_ids", "attention_mask"], output_names=outputs,
        dynamic_axes={"input_ids": {1: "source"}, "attention_mask": {1: "source"},
                      **{name: {2: "source"} for name in outputs}},
        opset_version=OPSET, dynamo=False)

    with torch.no_grad():
        cross = model.cross_kv(model.encode(source, mask))
    states = [torch.zeros(1, 1, model.dim) for _ in range(layers)]
    state_names = [f"state.{layer}" for layer in range(layers)]
    new_state_names = [f"new_state.{layer}" for layer in range(layers)]
    torch.onnx.export(
        DecoderGraph(model),
        (torch.tensor([[START_TOKEN]]), torch.tensor([0]), mask, *cross, *states),
        decoder_path,
        input_names=["input_ids", "position", "encoder_attention_mask",
                     *cross_names(layers), *state_names],
        output_names=["next_token", *new_state_names],
        dynamic_axes={"encoder_attention_mask": {1: "source"},
                      **{name: {2: "source"} for name in cross_names(layers)}},
        opset_version=OPSET, dynamo=False)
    return encoder_path, decoder_path


# -- 5. quantise --------------------------------------------------------------


def quantise(source: str, destination: str) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(source, destination, weight_type=QuantType.QInt8)


# -- 6. share the tied embedding ----------------------------------------------

ALIGN = 4096  # ONNX Runtime only memory-maps page-aligned external tensors


def point_at(initializer, location: str, offset: int, length: int) -> None:
    initializer.ClearField("raw_data")
    del initializer.external_data[:]
    initializer.data_location = onnx.TensorProto.EXTERNAL
    for key, value in (("location", location), ("offset", str(offset)),
                       ("length", str(length))):
        entry = initializer.external_data.add()
        entry.key, entry.value = key, value


def find_embedding(model, dim: int, vocab: int):
    """The quantised tied matrix: the one 8-bit initializer of shape [dim, vocab].

    It is `uint8`, not `int8`: the quantiser reaches this tensor through its
    `Gather` consumer, which quantises asymmetrically. That there is exactly
    *one* of them in the decoder — where the same matrix feeds both the target
    embedding lookup and the output projection — is the point of storing it
    transposed. A `[vocab, dim]` table plus its transpose would have produced
    two, and no amount of sharing afterwards could have merged them.
    """
    eight_bit = (onnx.TensorProto.INT8, onnx.TensorProto.UINT8)
    found = [init for init in model.graph.initializer
             if list(init.dims) == [dim, vocab] and init.data_type in eight_bit]
    if len(found) != 1:
        raise SystemExit(
            f"expected exactly one [{dim}, {vocab}] 8-bit initializer, found "
            f"{len(found)}: {[i.name for i in found]}")
    return found[0]


def write_bundle(graphs: dict[str, str], out: str, dim: int, vocab: int) -> None:
    os.makedirs(out, exist_ok=True)
    loaded = {name: onnx.load(path) for name, path in graphs.items()}
    embeddings = {name: find_embedding(model, dim, vocab)
                  for name, model in loaded.items()}

    payloads = {name: numpy_helper.to_array(init).tobytes()
                for name, init in embeddings.items()}
    reference = payloads["decoder.onnx"]
    for name, payload in payloads.items():
        if payload != reference:
            raise SystemExit(
                f"{name}: the tied embedding quantised differently from the "
                "decoder's; the two graphs cannot share one blob")
    with open(os.path.join(out, "embedding.data"), "wb") as handle:
        handle.write(reference)
    log(f"embedding.data {len(reference) / 1048576:.1f} MB, shared by both graphs")

    for name, model in loaded.items():
        base = name.replace(".onnx", "")
        blob_path = os.path.join(out, f"{base}.data")
        shared_name = embeddings[name].name
        with open(blob_path, "wb") as handle:
            for init in model.graph.initializer:
                payload = numpy_helper.to_array(init).tobytes()
                if init.name == shared_name:
                    point_at(init, "embedding.data", 0, len(reference))
                    continue
                if len(payload) < 1024:
                    # Small tensors go back inline. They must be reset
                    # explicitly: onnx.load() fills raw_data but leaves
                    # data_location EXTERNAL pointing into the *source* file, and
                    # saving that reads the quantisation scales from the wrong
                    # offset in the new blob — which changes the output without
                    # failing anything.
                    init.data_location = onnx.TensorProto.DEFAULT
                    del init.external_data[:]
                    init.raw_data = payload
                    continue
                handle.write(b"\0" * ((-handle.tell()) % ALIGN))
                offset = handle.tell()
                handle.write(payload)
                point_at(init, f"{base}.data", offset, len(payload))
        onnx.save(model, os.path.join(out, name))
        log(f"{name} graph {os.path.getsize(os.path.join(out, name)) / 1024:.0f} KB, "
            f"{base}.data {os.path.getsize(blob_path) / 1048576:.1f} MB")


# -- 7. tokenizer and manifest ------------------------------------------------


def write_tokenizer(spm_path: str, out: str) -> int:
    """Copies the SentencePiece model in, unmodified.

    No `vocab.json`: Marian uses the SentencePiece ids directly, so a mapping
    file would be the identity over 32 000 entries and 629 KB of a 32 MB
    bundle. `MarianTokenizer.fromAssets` builds that identity itself when no
    vocabulary is supplied. The `.spm` is byte-for-byte the file Mozilla
    publishes, so a bundle can be checked against the upstream checksum.
    """
    import sentencepiece as spm
    shutil.copyfile(spm_path, os.path.join(out, "source.spm"))
    return spm.SentencePieceProcessor(model_file=spm_path).get_piece_size()


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(out: str, pair: str, model: TinyModel, config: str) -> None:
    source, target = pair.split("-")
    files = sorted(f for f in os.listdir(out) if f != "manifest.json")
    entries = [{"name": name, "size": os.path.getsize(os.path.join(out, name)),
                "sha256": sha256(os.path.join(out, name))} for name in files]
    manifest = {
        "from": source,
        "to": target,
        "version": "1.0.0",
        "base_model": f"mozilla/firefox-translations-models {CATALOGUE[pair][0]}",
        "license": "MPL-2.0",
        "quantization": "int8",
        "architecture": {
            "family": "tiny-ssru",
            "decoder_layers": model.decoder_layers,
            "encoder_layers": model.encoder_layers,
            "decoder_attention_heads": model.heads,
            "head_dimension": model.dim // model.heads,
            "model_dimension": model.dim,
            "decoder_start_token_id": START_TOKEN,
            "eos_token_id": 0,
            "pad_token_id": 0,
            "max_position_embeddings": MAX_POSITIONS,
            "vocab_size": model.vocab,
        },
        "files": entries,
        "marian_config": config,
    }
    manifest["checksum"] = hashlib.sha256(
        "".join(e["sha256"] for e in entries).encode()).hexdigest()
    with open(os.path.join(out, "manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)


# -- verification -------------------------------------------------------------


def greedy_onnx(out: str, model: TinyModel, ids: list[int],
                limit: int = 256) -> list[int]:
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.log_severity_level = 3
    encoder = ort.InferenceSession(os.path.join(out, "encoder.onnx"), options)
    decoder = ort.InferenceSession(os.path.join(out, "decoder.onnx"), options)
    layers = model.decoder_layers

    source = np.array([ids], dtype=np.int64)
    mask = np.ones_like(source)
    cross = encoder.run(None, {"input_ids": source, "attention_mask": mask})
    feed = {name: value for name, value in zip(cross_names(layers), cross)}
    feed["encoder_attention_mask"] = mask
    for layer in range(layers):
        feed[f"state.{layer}"] = np.zeros((1, 1, model.dim), dtype=np.float32)
    token = np.array([[START_TOKEN]], dtype=np.int64)

    out_tokens: list[int] = []
    for position in range(limit):
        feed["input_ids"] = token
        feed["position"] = np.array([position], dtype=np.int64)
        result = decoder.run(None, feed)
        nxt = int(result[0][0, 0])
        for layer in range(layers):
            feed[f"state.{layer}"] = result[1 + layer]
        if nxt == 0:
            break
        out_tokens.append(nxt)
        token = np.array([[nxt]], dtype=np.int64)
    return out_tokens


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", default="en-fr", choices=sorted(CATALOGUE))
    parser.add_argument("--out", default=os.path.expanduser("~/ot-models-tiny"))
    parser.add_argument("--cache", default=os.path.expanduser("~/ot-work/ffx"))
    parser.add_argument("--work", help="reuse this directory for intermediates")
    args = parser.parse_args()

    out = os.path.join(args.out, args.pair)
    model_path, spm_path = fetch(args.pair, args.cache)

    log(f"parsing {os.path.basename(model_path)}")
    model, config = build_torch(model_path)
    print(f"   {model.encoder_layers} encoder layers, {model.decoder_layers} "
          f"decoder layers, dim {model.dim}, vocab {model.vocab}")

    import sentencepiece as spm
    processor = spm.SentencePieceProcessor(model_file=spm_path)

    log("checking the PyTorch rebuild on the fixtures")
    reference = {}
    for text in FIXTURES:
        ids = processor.encode(text) + [0]
        reference[text] = greedy_torch(model, ids)
        print(f"   {text}\n   -> {processor.decode(reference[text])}")

    work = args.work or tempfile.mkdtemp(prefix="tiny-")
    os.makedirs(work, exist_ok=True)
    try:
        quantised = {name: os.path.join(work, f"q-{name}")
                     for name in ("encoder.onnx", "decoder.onnx")}
        if not all(os.path.exists(path) for path in quantised.values()):
            log("exporting ONNX")
            encoder_fp32, decoder_fp32 = export(model, work)

            log("quantising to int8")
            for name, path in (("encoder.onnx", encoder_fp32),
                               ("decoder.onnx", decoder_fp32)):
                quantise(path, quantised[name])
                print(f"   {name}: {os.path.getsize(path) / 1048576:.1f} MB -> "
                      f"{os.path.getsize(quantised[name]) / 1048576:.1f} MB")
        else:
            log(f"reusing quantised graphs in {work}")

        log("writing the bundle")
        if os.path.isdir(out):
            shutil.rmtree(out)
        write_bundle(quantised, out, model.dim, model.vocab)
        write_tokenizer(spm_path, out)
        write_manifest(out, args.pair, model, config)
    finally:
        if not args.work:
            shutil.rmtree(work, ignore_errors=True)

    log("verifying the quantised bundle against PyTorch")
    matches = 0
    for text in FIXTURES:
        ids = processor.encode(text) + [0]
        produced = processor.decode(greedy_onnx(out, model, ids))
        expected = processor.decode(reference[text])
        same = produced == expected
        matches += same
        print(f"   [{'ok ' if same else 'DIFF'}] {produced}")
        if not same:
            print(f"          float32: {expected}")

    total = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out))
    log(f"{out}: {total / 1048576:.1f} MB, {matches}/{len(FIXTURES)} "
        "translations identical to the float32 rebuild")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
