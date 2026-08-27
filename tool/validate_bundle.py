#!/usr/bin/env python3
"""Runs a model bundle through the exact protocol the Dart engine uses.

    python3 tool/validate_bundle.py build/models/fr-en

Checks, in order:

  1. every file listed in `manifest.json` exists with the right size and
     SHA-256 (the same check the on-device ModelManager performs);
  2. the graphs load, and the decoder has the grafted `next_token` output;
  3. greedy generation produces sensible target-language text for a handful of
     source-language sentences, with per-sentence timings.

It deliberately mirrors the engine rather than calling `transformers.generate`:
the point is to catch a bundle that is subtly wrong for *this* decoding loop —
a missing `next_token`, cross-attention KV fed back on cached steps, a
mismatched vocabulary — before it reaches a device.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import onnxruntime as ort

# A few sentences per source language, so every direction gets real input.
SAMPLES = {
    "en": [
        "Hello, how are you?",
        "The quick brown fox jumps over the lazy dog.",
        "I would like to book a table for two people at eight o'clock tonight.",
        "She said that the meeting had been postponed until next Tuesday.",
    ],
    "fr": [
        "Bonjour, comment allez-vous ?",
        "Le renard brun rapide saute par-dessus le chien paresseux.",
        "Je voudrais réserver une table pour deux personnes à huit heures.",
        "Elle a dit que la réunion avait été reportée à mardi prochain.",
    ],
    "es": [
        "Hola, ¿cómo estás?",
        "El rápido zorro marrón salta sobre el perro perezoso.",
        "Me gustaría reservar una mesa para dos personas a las ocho.",
        "Dijo que la reunión se había aplazado hasta el próximo martes.",
    ],
    "de": [
        "Guten Tag, wie geht es Ihnen?",
        "Der schnelle braune Fuchs springt über den faulen Hund.",
        "Ich möchte einen Tisch für zwei Personen um acht Uhr reservieren.",
        "Sie sagte, dass die Sitzung auf nächsten Dienstag verschoben wurde.",
    ],
}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_files(directory: str, manifest: dict) -> int:
    total = 0
    for entry in manifest["files"]:
        path = os.path.join(directory, entry["name"])
        if not os.path.exists(path):
            raise SystemExit(f"missing file: {entry['name']}")
        size = os.path.getsize(path)
        if size != entry["size"]:
            raise SystemExit(f"{entry['name']}: size {size} != {entry['size']}")
        digest = sha256(path)
        if digest != entry["sha256"]:
            raise SystemExit(f"{entry['name']}: checksum mismatch")
        total += size
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bundle")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    directory = args.bundle
    with open(os.path.join(directory, "manifest.json")) as fh:
        manifest = json.load(fh)
    arch = manifest["architecture"]
    pair = f"{manifest['from']}-{manifest['to']}"

    total = verify_files(directory, manifest)

    layers = arch["decoder_layers"]
    heads = arch["decoder_attention_heads"]
    head_dim = arch["head_dimension"]
    start, eos = arch["decoder_start_token_id"], arch["eos_token_id"]

    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.log_severity_level = 4
    began = time.time()
    encoder = ort.InferenceSession(os.path.join(directory, "encoder.onnx"),
                                   options, providers=["CPUExecutionProvider"])
    decoder = ort.InferenceSession(os.path.join(directory, "decoder.onnx"),
                                   options, providers=["CPUExecutionProvider"])
    load_ms = (time.time() - began) * 1000

    outputs = {output.name for output in decoder.get_outputs()}
    if "next_token" not in outputs:
        raise SystemExit("decoder.onnx has no `next_token` output; this bundle "
                         "was not produced by this repository's tooling")

    family = manifest["architecture"].get("family", "marian")
    if family != "marian":
        # This checker drives optimum's merged decoder protocol — encoder hidden
        # states, a growing key/value cache, `use_cache_branch`. A `tiny-ssru`
        # bundle has none of those, and it is verified where it is built:
        # `tool/build_tiny_model.py` decodes every fixture through the finished
        # int8 graphs and compares against the float32 rebuild before writing
        # the bundle. `test/end_to_end_test.dart` then runs it through the Dart
        # engine and the public API.
        print(f"  files and checksums OK; {family} inference is verified by "
              "tool/build_tiny_model.py and test/end_to_end_test.dart")
        return

    present_decoder = [f"present.{l}.decoder.{kv}"
                       for l in range(layers) for kv in ("key", "value")]
    present_encoder = [f"present.{l}.encoder.{kv}"
                       for l in range(layers) for kv in ("key", "value")]
    past = lambda name: name.replace("present", "past_key_values")  # noqa: E731

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(manifest["base_model"])

    def greedy(text: str, max_new: int = 512):
        encoded = tokenizer(text, return_tensors="np")
        ids = encoded["input_ids"].astype(np.int64)
        mask = encoded["attention_mask"].astype(np.int64)
        began = time.time()
        hidden = encoder.run(["last_hidden_state"],
                             {"input_ids": ids, "attention_mask": mask})[0]
        encode_ms = (time.time() - began) * 1000
        empty = np.zeros((1, heads, 0, head_dim), np.float32)
        current = np.array([[start]], np.int64)
        decoder_kv = encoder_kv = None
        tokens = []
        began = time.time()
        for _ in range(max_new):
            feed = {"input_ids": current, "encoder_attention_mask": mask,
                    "encoder_hidden_states": hidden}
            if decoder_kv is None:
                for name in present_decoder + present_encoder:
                    feed[past(name)] = empty
                feed["use_cache_branch"] = np.array([False])
                out = decoder.run(["next_token"] + present_decoder + present_encoder,
                                  feed)
                decoder_kv = out[1:1 + len(present_decoder)]
                encoder_kv = out[1 + len(present_decoder):]
            else:
                for name, value in zip(present_decoder, decoder_kv):
                    feed[past(name)] = value
                for name, value in zip(present_encoder, encoder_kv):
                    feed[past(name)] = value
                feed["use_cache_branch"] = np.array([True])
                out = decoder.run(["next_token"] + present_decoder, feed)
                decoder_kv = out[1:]
            token = int(out[0][0, -1])
            if token == eos:
                break
            tokens.append(token)
            current = np.array([[token]], np.int64)
        decode_ms = (time.time() - began) * 1000
        return (tokenizer.decode(tokens, skip_special_tokens=True), len(tokens),
                encode_ms, decode_ms)

    print(f"{pair} v{manifest['version']}  {total / 1048576:.1f} MB  "
          f"{manifest['quantization']}  {manifest['license']}  "
          f"{layers}L/{heads}H  vocab {arch['vocab_size']}  load {load_ms:.0f} ms")

    samples = SAMPLES.get(manifest["from"])
    if not samples:
        raise SystemExit(f"no sample sentences for source language "
                         f"{manifest['from']}")
    empty_outputs = 0
    for text in samples:
        translated, count, encode_ms, decode_ms = greedy(text)
        if not translated.strip():
            empty_outputs += 1
        if not args.quiet:
            print(f"  {manifest['from'].upper()}: {text}")
            print(f"  {manifest['to'].upper()}: {translated}")
            print(f"     {count} tokens, enc {encode_ms:.0f} ms, "
                  f"dec {decode_ms:.0f} ms "
                  f"({decode_ms / max(count, 1):.1f} ms/tok)")
    if empty_outputs:
        raise SystemExit(f"{pair}: {empty_outputs} sample(s) produced no output")
    print(f"  OK", file=sys.stderr)


if __name__ == "__main__":
    main()
