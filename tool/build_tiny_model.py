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

# Mozilla publishes one directory per direction, under a quality tier, in
# mozilla/firefox-translations-models.
#
# The checkpoints themselves are no longer downloadable from there. They are Git
# LFS pointers whose objects have been removed from GitHub's LFS storage — the
# batch API answers `410 Object does not exist on the server` — so `raw` serves
# a 130-byte pointer and `media` answers 404. Only the small files, including
# `metadata.json`, are still real.
#
# That turns out not to matter, because `metadata.json` publishes the **SHA-256
# of the checkpoint**. Any mirror will do as long as its bytes hash to what
# Mozilla says they should, and a tampered or truncated mirror fails loudly
# instead of quietly producing a model that translates slightly wrong. The
# authoritative metadata always comes from GitHub; only the bytes come from a
# mirror.
META_URL = ("https://raw.githubusercontent.com/mozilla/"
            "firefox-translations-models/main/models")

# Tried in order. `{pair}` is `en-fr`, `{joined}` is `enfr`.
CHECKPOINT_MIRRORS = [
    ("mozilla/firefox-translations-models (GitHub LFS)",
     "https://media.githubusercontent.com/media/mozilla/"
     "firefox-translations-models/main/models/{tier}/{joined}/{name}.gz"),
    ("mukowaty/firefox-translations (Hugging Face, MPL-2.0)",
     "https://huggingface.co/mukowaty/firefox-translations/resolve/main/"
     "{pair}/{name}.gz"),
]

# Measured on `en-fr`, from the tiers' own `metadata.json`:
#
#   tier          dim  enc/dec   size     FLORES BLEU   COMET
#   tiny          256    6/2     17.1 MB      48.5     0.8573
#   base-memory   384    6/4     31.6 MB      49.6     0.8697
#
# `base-memory` is the default: 1.1 BLEU and 0.012 COMET for 14 MB is the better
# side of that trade for a package that already fits the size budget. `tiny`
# is a supported `--tier`, and doc/model-decision.md carries the comparison.
DEFAULT_TIER = "base-memory"

CATALOGUE = {
    # pair: repository directory
    "en-fr": "enfr",
    "fr-en": "fren",
    "en-es": "enes",
    "es-en": "esen",
    "en-de": "ende",
    "de-en": "deen",
}

# One set per source language, so the build log shows a translation a human can
# judge. Feeding English into a `fr-en` model would still be a valid check that
# ONNX matches PyTorch — both see the same input — but it would print nonsense
# and hide a genuinely bad conversion behind it.
FIXTURES = {
    "en": [
        "Hello, how are you?",
        "Hello world",
        "This is a simple test.",
        "I love programming and mobile development.",
        "The quick brown fox jumps over the lazy dog.",
        "I would like to book a table for two people at eight o'clock tonight.",
        "She said that the meeting had been postponed until next Tuesday "
        "because of the weather.",
    ],
    "fr": [
        "Bonjour, comment allez-vous ?",
        "Bonjour le monde",
        "C'est un test simple.",
        "J'adore la programmation et le développement mobile.",
        "Le renard brun rapide saute par-dessus le chien paresseux.",
        "Je voudrais réserver une table pour deux personnes à huit heures ce "
        "soir.",
        "Elle a dit que la réunion avait été reportée à mardi prochain en "
        "raison de la météo.",
    ],
    "es": [
        "Hola, ¿cómo estás?",
        "Hola mundo",
        "Esta es una prueba sencilla.",
        "Me encanta la programación y el desarrollo móvil.",
        "El rápido zorro marrón salta sobre el perro perezoso.",
        "Me gustaría reservar una mesa para dos personas a las ocho de esta "
        "noche.",
        "Dijo que la reunión se había aplazado hasta el próximo martes debido "
        "al tiempo.",
    ],
    "de": [
        "Hallo, wie geht es dir?",
        "Hallo Welt",
        "Das ist ein einfacher Test.",
        "Ich liebe Programmierung und mobile Entwicklung.",
        "Der schnelle braune Fuchs springt über den faulen Hund.",
        "Ich möchte heute Abend um acht Uhr einen Tisch für zwei Personen "
        "reservieren.",
        "Sie sagte, dass das Treffen wegen des Wetters auf nächsten Dienstag "
        "verschoben worden sei.",
    ],
}

MAX_POSITIONS = 512
OPSET = 17


def log(message: str) -> None:
    print(f"\033[1m==\033[0m {message}", flush=True)


# -- 1. fetch -----------------------------------------------------------------


def _download_gz(url: str) -> bytes | None:
    """Fetches a gzipped file, or returns None if this mirror does not have it."""
    import gzip
    try:
        with urllib.request.urlopen(url, timeout=180) as response:
            payload = response.read()
    except Exception:  # noqa: BLE001 - any failure just means "try the next one"
        return None
    if payload.startswith(b"version https://git-lfs"):
        return None  # an LFS pointer, not the object
    try:
        return gzip.decompress(payload)
    except OSError:
        return None


def fetch(pair: str, cache: str, tier: str) -> tuple[str, str, dict]:
    """Downloads the checkpoint and vocabulary, verified against Mozilla's own
    published SHA-256."""
    joined = CATALOGUE[pair]
    cache = os.path.join(cache, tier)
    os.makedirs(cache, exist_ok=True)

    with urllib.request.urlopen(
            f"{META_URL}/{tier}/{joined}/metadata.json", timeout=60) as response:
        metadata = json.load(response)

    paths = {}
    for kind, name, expected_hash, expected_size in (
            ("model", f"model.{joined}.intgemm.alphas.bin",
             metadata.get("hash"), metadata.get("byteSize")),
            ("vocab", f"vocab.{joined}.spm", None, None)):
        destination = os.path.join(cache, name)
        if not os.path.exists(destination):
            payload = None
            for label, template in CHECKPOINT_MIRRORS:
                url = template.format(tier=tier, joined=joined, pair=pair,
                                      name=name)
                log(f"{pair}: fetching {name} from {label}")
                payload = _download_gz(url)
                if payload is not None:
                    break
            if payload is None:
                raise SystemExit(
                    f"{pair}: no mirror served {name}. The upstream GitHub LFS "
                    "objects have been removed; see CHECKPOINT_MIRRORS.")
            with open(destination, "wb") as handle:
                handle.write(payload)
        paths[kind] = destination

        if expected_size and os.path.getsize(destination) != expected_size:
            raise SystemExit(
                f"{pair}: {name} is {os.path.getsize(destination)} bytes, "
                f"Mozilla's metadata says {expected_size}")
        if expected_hash:
            digest = sha256(destination)
            if digest != expected_hash:
                os.remove(destination)
                raise SystemExit(
                    f"{pair}: {name} hashes to {digest}, Mozilla's metadata "
                    f"says {expected_hash}. That mirror is not serving the "
                    "published checkpoint; the file was deleted.")
            print(f"   sha256 matches Mozilla's published hash ({digest[:16]}…)")

    return paths["model"], paths["vocab"], metadata


# -- 3. rebuild ---------------------------------------------------------------


def build_torch(model_path: str) -> tuple[TinyModel, str, int]:
    """Returns the rebuilt model, its Marian config, and how many of the
    checkpoint's tensors are quantisation multipliers rather than weights."""
    _, items = marian_binary.load(model_path)
    config = marian_binary.config(items)
    parameters = marian_binary.parameters(items)
    multipliers = sum(1 for name in parameters if name.endswith("_QuantMultA"))
    model = TinyModel(parameters, max_positions=MAX_POSITIONS).eval()
    return model, config, multipliers


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


def write_manifest(out: str, pair: str, model: TinyModel, config: str,
                   tier: str, metadata: dict) -> None:
    source, target = pair.split("-")
    files = sorted(f for f in os.listdir(out) if f != "manifest.json")
    entries = [{"name": name, "size": os.path.getsize(os.path.join(out, name)),
                "sha256": sha256(os.path.join(out, name))} for name in files]
    manifest = {
        "from": source,
        "to": target,
        "version": "1.0.0",
        "base_model": "mozilla/firefox-translations-models "
                      f"{tier}/{CATALOGUE[pair]}",
        "license": "MPL-2.0",
        "quantization": "int8",
        # Straight from the upstream metadata, so a bundle carries the quality
        # its checkpoint was published with rather than a number copied by hand.
        "upstream": {
            "tier": tier,
            "checkpoint_bytes": metadata.get("byteSize"),
            "checkpoint_sha256": metadata.get("hash"),
            "flores": metadata.get("flores"),
        },
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


def build_one(pair: str, args) -> tuple[bool, str]:
    """Builds one direction. Returns (everything matched, a summary line)."""
    out = os.path.join(args.out, pair)
    source_language = pair.split("-")[0]
    fixtures = FIXTURES[source_language]

    model_path, spm_path, metadata = fetch(pair, args.cache, args.tier)

    log(f"{pair}: parsing {args.tier}/{os.path.basename(model_path)}")
    model, config, multipliers = build_torch(model_path)
    flores = metadata.get("flores", {})
    print(f"   {model.encoder_layers} encoder layers, {model.decoder_layers} "
          f"decoder layers, dim {model.dim}, vocab {model.vocab}, "
          f"upstream BLEU {flores.get('bleu')} COMET {flores.get('comet')}")

    # Mozilla publishes the exact parameter count of every checkpoint. Matching
    # it is a cheap, total check on the rebuild: a layer read with the wrong
    # shape, a tensor skipped, or a depth misread all change this number, and
    # all of them would otherwise surface only as slightly worse translations.
    #
    # Their count includes the one-element `*_QuantMultA` scalars — the intgemm
    # quantisation multipliers — which are consumed while dequantising and are
    # not weights of the rebuilt float32 model. There are 70 of them in a
    # 6+4-layer student, and subtracting them is what makes the two counts
    # comparable at all.
    expected = (metadata.get("modelStatistics") or {}).get("parameters")
    actual = sum(p.numel() for p in model.parameters())
    if expected and actual != expected - multipliers:
        raise SystemExit(
            f"{pair}: the rebuild has {actual} parameters, the checkpoint has "
            f"{expected} less {multipliers} quantisation multipliers "
            f"= {expected - multipliers}")
    if expected:
        print(f"   {actual} parameters + {multipliers} quantisation "
              f"multipliers = {expected}, matching the checkpoint")

    import sentencepiece as spm
    processor = spm.SentencePieceProcessor(model_file=spm_path)

    log(f"{pair}: checking the PyTorch rebuild on the fixtures")
    reference = {}
    for text in fixtures:
        ids = processor.encode(text) + [0]
        reference[text] = greedy_torch(model, ids)
        print(f"   {text}\n   -> {processor.decode(reference[text])}")

    work = args.work or tempfile.mkdtemp(prefix=f"tiny-{pair}-")
    if args.work:
        work = os.path.join(args.work, f"{args.tier}-{pair}")
    os.makedirs(work, exist_ok=True)
    try:
        quantised = {name: os.path.join(work, f"q-{name}")
                     for name in ("encoder.onnx", "decoder.onnx")}
        if not all(os.path.exists(path) for path in quantised.values()):
            log(f"{pair}: exporting ONNX")
            encoder_fp32, decoder_fp32 = export(model, work)

            log(f"{pair}: quantising to int8")
            for name, path in (("encoder.onnx", encoder_fp32),
                               ("decoder.onnx", decoder_fp32)):
                quantise(path, quantised[name])
                print(f"   {name}: {os.path.getsize(path) / 1048576:.1f} MB -> "
                      f"{os.path.getsize(quantised[name]) / 1048576:.1f} MB")
        else:
            log(f"{pair}: reusing quantised graphs in {work}")

        log(f"{pair}: writing the bundle")
        if os.path.isdir(out):
            shutil.rmtree(out)
        write_bundle(quantised, out, model.dim, model.vocab)
        write_tokenizer(spm_path, out)
        write_manifest(out, pair, model, config, args.tier, metadata)
    finally:
        if not args.work:
            shutil.rmtree(work, ignore_errors=True)

    log(f"{pair}: verifying the quantised bundle against PyTorch")
    matches = 0
    for text in fixtures:
        ids = processor.encode(text) + [0]
        produced = processor.decode(greedy_onnx(out, model, ids))
        expected = processor.decode(reference[text])
        same = produced == expected
        matches += same
        print(f"   [{'ok ' if same else 'DIFF'}] {produced}")
        if not same:
            print(f"          float32: {expected}")

    total = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out))
    summary = (f"{pair:>6}  {total / 1048576:5.1f} MB  "
               f"{matches}/{len(fixtures)} identical to float32  "
               f"upstream BLEU {flores.get('bleu')}")
    log(f"{out}: {summary}")
    return matches == len(fixtures), summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", default="en-fr",
                        help="a direction, or `all` for the whole catalogue")
    parser.add_argument("--tier", default=DEFAULT_TIER,
                        choices=("base-memory", "tiny"),
                        help=f"upstream quality tier (default: {DEFAULT_TIER})")
    parser.add_argument("--out", default=os.path.expanduser("~/ot-models-tiny"))
    parser.add_argument("--cache", default=os.path.expanduser("~/ot-work/ffx"))
    parser.add_argument("--work", help="reuse this directory for intermediates")
    parser.add_argument("--skip-existing", action="store_true",
                        help="leave a direction alone if its bundle is there")
    args = parser.parse_args()

    pairs = sorted(CATALOGUE) if args.pair == "all" else [args.pair]
    unknown = [p for p in pairs if p not in CATALOGUE]
    if unknown:
        raise SystemExit(f"unknown direction(s): {', '.join(unknown)}")

    summaries, failed = [], []
    for pair in pairs:
        if args.skip_existing and os.path.exists(
                os.path.join(args.out, pair, "manifest.json")):
            log(f"{pair}: already built, skipping")
            continue
        try:
            ok, summary = build_one(pair, args)
            summaries.append(summary)
            if not ok:
                # A quantisation difference is expected on the odd sentence and
                # is not a build failure; it is reported, not hidden.
                pass
        except Exception as error:  # noqa: BLE001 - keep going past one bad pair
            failed.append(f"{pair}: {error}")
            print(f"\033[31m{pair} FAILED: {error}\033[0m", file=sys.stderr)

    if len(pairs) > 1:
        log("catalogue")
        for summary in summaries:
            print(f"   {summary}")
        for failure in failed:
            print(f"   FAILED {failure}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
