#!/usr/bin/env python3
"""Builds an `offline_translate` model bundle from an upstream OPUS-MT checkpoint.

    python3 tool/build_model.py --pair en-fr --out build/models

Pipeline:
  1. export the Hugging Face checkpoint to ONNX with optimum (encoder +
     *merged* decoder, i.e. one graph serving both the first and the cached
     decoding steps);
  2. quantize both graphs to int8 dynamic;
  3. graft a `next_token` output onto the decoder: the greedy arg-max is done
     inside the graph, with the `bad_words_ids` (pad) logit masked out, so the
     59k-wide logits tensor never crosses the FFI boundary;
  4. copy the original, unmodified `source.spm` and `vocab.json`;
  5. emit `manifest.json` with per-file SHA-256 checksums and the architecture
     constants read from the checkpoint's `config.json`.

Requires: optimum-onnx, transformers, torch, onnx, onnxruntime, accelerate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

# Upstream checkpoints, with the license stated on their model card.
CATALOGUE = {
    "en-fr": ("Helsinki-NLP/opus-mt-en-fr", "Apache-2.0"),
    "fr-en": ("Helsinki-NLP/opus-mt-fr-en", "Apache-2.0"),
    "en-es": ("Helsinki-NLP/opus-mt-en-es", "Apache-2.0"),
    "es-en": ("Helsinki-NLP/opus-mt-es-en", "Apache-2.0"),
    "en-de": ("Helsinki-NLP/opus-mt-en-de", "CC-BY-4.0"),  # note: not Apache
    "de-en": ("Helsinki-NLP/opus-mt-de-en", "Apache-2.0"),
    "fr-es": ("Helsinki-NLP/opus-mt-fr-es", "Apache-2.0"),
    "es-fr": ("Helsinki-NLP/opus-mt-es-fr", "Apache-2.0"),
    "fr-de": ("Helsinki-NLP/opus-mt-fr-de", "Apache-2.0"),
    "de-fr": ("Helsinki-NLP/opus-mt-de-fr", "Apache-2.0"),
    "es-de": ("Helsinki-NLP/opus-mt-es-de", "Apache-2.0"),
    "de-es": ("Helsinki-NLP/opus-mt-de-es", "Apache-2.0"),
}

BUNDLE_VERSION = "1.0.0"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def export(model_id: str, workdir: str) -> str:
    out = os.path.join(workdir, "onnx")
    subprocess.run(
        [
            sys.executable, "-m", "optimum.commands.optimum_cli",
            "export", "onnx",
            "--model", model_id,
            "--task", "text2text-generation-with-past",
            "--opset", "17",
            out,
        ],
        check=True,
    )
    return out


def quantize(src: str, dst: str) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(src, dst, weight_type=QuantType.QInt8)


def quantized_merged_decoder(onnx_dir: str, workdir: str) -> str:
    """Quantizes the two decoder graphs, then merges them.

    Quantizing `decoder_model_merged.onnx` directly is a no-op for the weights
    that live inside the `If` branches, which leaves the graph at fp32 size
    (214 MB instead of 54 MB). Quantizing `decoder_model.onnx` and
    `decoder_with_past_model.onnx` separately and merging afterwards keeps the
    int8 weights and lets the merge pass deduplicate the shared initializers.
    """
    from optimum.onnx.graph_transformations import merge_decoders

    no_past = os.path.join(workdir, "decoder_q8.onnx")
    with_past = os.path.join(workdir, "decoder_with_past_q8.onnx")
    merged = os.path.join(workdir, "decoder_merged_q8.onnx")
    if not os.path.exists(no_past):
        quantize(os.path.join(onnx_dir, "decoder_model.onnx"), no_past)
    if not os.path.exists(with_past):
        quantize(os.path.join(onnx_dir, "decoder_with_past_model.onnx"), with_past)
    if not os.path.exists(merged):
        # `strict=False`: the cached graph does not re-emit the cross-attention
        # KV outputs, they are carried over from the first step instead.
        merge_decoders(no_past, with_past, save_path=merged, strict=False)
    return merged


def add_next_token(model: onnx.ModelProto, vocab_size: int, pad_id: int) -> None:
    """Adds `next_token = argmax(logits + bad_words_mask)` to the decoder.

    Doing the greedy pick inside the graph keeps the 59 514-wide logits tensor
    on the native side: the Dart engine only ever reads one int64 per step.
    The mask reproduces `generation_config.bad_words_ids`, which forbids the
    pad token that Marian also uses as `decoder_start_token_id`.
    """
    graph = model.graph
    names = {o.name for o in graph.output}
    if "next_token" in names:
        return
    if "logits" not in names:
        raise SystemExit(f"decoder has no `logits` output: {sorted(names)}")

    mask = np.zeros((vocab_size,), dtype=np.float32)
    mask[pad_id] = -1e9
    graph.initializer.append(numpy_helper.from_array(mask, name="ot_bad_words_mask"))
    graph.node.append(
        helper.make_node("Add", ["logits", "ot_bad_words_mask"],
                         ["ot_masked_logits"], name="ot_mask_add"))
    graph.node.append(
        helper.make_node("ArgMax", ["ot_masked_logits"], ["next_token"],
                         name="ot_argmax", axis=-1, keepdims=0))
    graph.output.append(
        helper.make_tensor_value_info(
            "next_token", TensorProto.INT64,
            ["batch_size", "decoder_sequence_length"]))
    onnx.checker.check_model(model, full_check=False)


def save_with_external_data(model: onnx.ModelProto, path: str) -> None:
    """Writes the graph next to a companion `.data` file holding the weights.

    ONNX Runtime memory-maps external initializers instead of copying them out
    of the protobuf, which cuts resident memory for this model roughly in half
    (466 MB -> 344 MB peak on the reference machine) at no measurable cost in
    inference speed.
    """
    location = os.path.basename(path).replace(".onnx", ".data")
    directory = os.path.dirname(path)
    stale = os.path.join(directory, location)
    if os.path.exists(stale):
        os.remove(stale)
    onnx.save(model, path, save_as_external_data=True,
              all_tensors_to_one_file=True, location=location,
              size_threshold=1024, convert_attribute=True)
    # onnx.save creates the companion file 0600; bundles are meant to be shipped.
    os.chmod(os.path.join(directory, location), 0o644)


def build(pair: str, out_root: str, keep_workdir: str | None) -> None:
    if pair not in CATALOGUE:
        raise SystemExit(f"unknown pair {pair}; known: {sorted(CATALOGUE)}")
    model_id, license_id = CATALOGUE[pair]
    src_lang, tgt_lang = pair.split("-")

    # Intermediates are large — the fp32 export alone is about 1.1 GB per
    # direction — so unless a workdir is pinned they go to a temporary one that
    # is removed as soon as the bundle is written.
    temporary = keep_workdir is None
    workdir = keep_workdir or tempfile.mkdtemp(prefix=f"ot-{pair}-")
    os.makedirs(workdir, exist_ok=True)
    onnx_dir = os.path.join(workdir, "onnx")
    if not os.path.isdir(onnx_dir):
        export(model_id, workdir)

    with open(os.path.join(onnx_dir, "config.json")) as fh:
        config = json.load(fh)

    dest = os.path.join(out_root, pair)
    os.makedirs(dest, exist_ok=True)

    enc_q = os.path.join(workdir, "encoder_q8.onnx")
    if not os.path.exists(enc_q):
        quantize(os.path.join(onnx_dir, "encoder_model.onnx"), enc_q)
    dec_q = quantized_merged_decoder(onnx_dir, workdir)

    save_with_external_data(onnx.load(enc_q), os.path.join(dest, "encoder.onnx"))
    decoder = onnx.load(dec_q)
    add_next_token(decoder, config["vocab_size"], config["pad_token_id"])
    save_with_external_data(decoder, os.path.join(dest, "decoder.onnx"))
    for name in ("source.spm", "vocab.json"):
        shutil.copyfile(os.path.join(onnx_dir, name), os.path.join(dest, name))

    files = []
    for name in ("encoder.onnx", "encoder.data", "decoder.onnx", "decoder.data",
                 "source.spm", "vocab.json"):
        path = os.path.join(dest, name)
        files.append({"name": name, "size": os.path.getsize(path),
                      "sha256": sha256(path)})

    bundle_checksum = hashlib.sha256(
        "".join(sorted(f["sha256"] for f in files)).encode()).hexdigest()

    heads = config["decoder_attention_heads"]
    manifest = {
        "from": src_lang,
        "to": tgt_lang,
        "version": BUNDLE_VERSION,
        "checksum": bundle_checksum,
        "base_model": model_id,
        "license": license_id,
        "quantization": "int8",
        "architecture": {
            "decoder_layers": config["decoder_layers"],
            "decoder_attention_heads": heads,
            "head_dimension": config["d_model"] // heads,
            "decoder_start_token_id": config["decoder_start_token_id"],
            "eos_token_id": config["eos_token_id"],
            "pad_token_id": config["pad_token_id"],
            "max_position_embeddings": config["max_position_embeddings"],
            "vocab_size": config["vocab_size"],
        },
        "files": files,
    }
    with open(os.path.join(dest, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")

    total = sum(f["size"] for f in files)
    print(f"built {pair}: {total / 1048576:.1f} MB -> {dest} "
          f"({manifest['architecture']['decoder_layers']} layers, "
          f"vocab {manifest['architecture']['vocab_size']}, {license_id})")

    if temporary:
        shutil.rmtree(workdir, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair", action="append", required=True,
                    help="language direction, e.g. en-fr; `all` builds the "
                         "whole catalogue (repeatable)")
    ap.add_argument("--out", default="build/models")
    ap.add_argument("--workdir", default=None,
                    help="reuse this directory for intermediate artefacts "
                         "instead of a temporary one that is cleaned up")
    ap.add_argument("--force", action="store_true",
                    help="rebuild directions that are already present in --out")
    args = ap.parse_args()

    pairs: list[str] = []
    for pair in args.pair:
        if pair == "all":
            pairs.extend(CATALOGUE)
        elif pair not in pairs:
            pairs.append(pair)

    failures = []
    for index, pair in enumerate(pairs, 1):
        manifest = os.path.join(args.out, pair, "manifest.json")
        if os.path.exists(manifest) and not args.force:
            print(f"[{index}/{len(pairs)}] {pair}: already built, skipping")
            continue
        print(f"[{index}/{len(pairs)}] building {pair} ...", flush=True)
        try:
            build(pair, args.out, args.workdir)
        except Exception as exc:  # noqa: BLE001 - one bad pair must not stop the batch
            print(f"[{index}/{len(pairs)}] {pair} FAILED: {exc}", flush=True)
            failures.append((pair, exc))

    if failures:
        print("\nfailed:", ", ".join(pair for pair, _ in failures))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
