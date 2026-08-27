#!/usr/bin/env python3
"""Scores a model bundle on FLORES-200 devtest with BLEU and chrF++.

    python3 tool/eval_quality.py ~/ot-models/en-fr --flores ~/flores200_dataset

Uses the **engine's own greedy decoding loop**, not `transformers.generate`, so
the score reflects what the package actually produces on a device — including
the int8 quantisation and the in-graph arg-max. A number obtained any other way
would flatter the bundle.

FLORES-200 is CC-BY-SA-4.0 and is the set Mozilla publishes its Firefox
Translations scores against, which makes the two directly comparable. The usual
caveat applies: BLEU across different detokenisation conventions is only
roughly comparable, so treat gaps under ~1 point as noise.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import onnxruntime as ort

FLORES = {"en": "eng_Latn", "fr": "fra_Latn", "es": "spa_Latn", "de": "deu_Latn"}


def load_sessions(bundle: str, threads: int):
    with open(os.path.join(bundle, "manifest.json")) as fh:
        manifest = json.load(fh)
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.log_severity_level = 4
    began = time.time()
    encoder = ort.InferenceSession(os.path.join(bundle, "encoder.onnx"), options,
                                   providers=["CPUExecutionProvider"])
    decoder = ort.InferenceSession(os.path.join(bundle, "decoder.onnx"), options,
                                   providers=["CPUExecutionProvider"])
    return manifest, encoder, decoder, (time.time() - began) * 1000


def make_greedy(manifest, encoder, decoder, tokenizer):
    arch = manifest["architecture"]
    layers = arch["decoder_layers"]
    heads = arch["decoder_attention_heads"]
    head_dim = arch["head_dimension"]
    start, eos = arch["decoder_start_token_id"], arch["eos_token_id"]
    present_dec = [f"present.{l}.decoder.{kv}"
                   for l in range(layers) for kv in ("key", "value")]
    present_enc = [f"present.{l}.encoder.{kv}"
                   for l in range(layers) for kv in ("key", "value")]
    past = lambda n: n.replace("present", "past_key_values")  # noqa: E731
    empty = np.zeros((1, heads, 0, head_dim), np.float32)

    def greedy(text, max_new=256):
        encoded = tokenizer(text, return_tensors="np")
        ids = encoded["input_ids"].astype(np.int64)
        mask = encoded["attention_mask"].astype(np.int64)
        hidden = encoder.run(["last_hidden_state"],
                             {"input_ids": ids, "attention_mask": mask})[0]
        current = np.array([[start]], np.int64)
        decoder_kv = encoder_kv = None
        tokens = []
        for _ in range(max_new):
            feed = {"input_ids": current, "encoder_attention_mask": mask,
                    "encoder_hidden_states": hidden}
            if decoder_kv is None:
                for name in present_dec + present_enc:
                    feed[past(name)] = empty
                feed["use_cache_branch"] = np.array([False])
                out = decoder.run(["next_token"] + present_dec + present_enc, feed)
                decoder_kv = out[1:1 + len(present_dec)]
                encoder_kv = out[1 + len(present_dec):]
            else:
                for name, value in zip(present_dec, decoder_kv):
                    feed[past(name)] = value
                for name, value in zip(present_enc, encoder_kv):
                    feed[past(name)] = value
                feed["use_cache_branch"] = np.array([True])
                out = decoder.run(["next_token"] + present_dec, feed)
                decoder_kv = out[1:]
            token = int(out[0][0, -1])
            if token == eos:
                break
            tokens.append(token)
            current = np.array([[token]], np.int64)
        return tokenizer.decode(tokens, skip_special_tokens=True), len(tokens)

    return greedy


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bundle")
    ap.add_argument("--flores", default=os.path.expanduser("~/flores200_dataset"))
    ap.add_argument("--split", default="devtest", choices=["dev", "devtest"])
    ap.add_argument("--limit", type=int, default=0,
                    help="score only the first N sentences (0 = all 1012)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--label", default=None,
                    help="name for this bundle in the output line")
    args = ap.parse_args()

    manifest, encoder, decoder, load_ms = load_sessions(args.bundle, args.threads)
    src, tgt = manifest["from"], manifest["to"]
    if src not in FLORES or tgt not in FLORES:
        raise SystemExit(f"FLORES has no mapping for {src}-{tgt}")

    def read(lang):
        path = os.path.join(args.flores, args.split,
                            f"{FLORES[lang]}.{args.split}")
        with open(path, encoding="utf-8") as fh:
            return [line.rstrip("\n") for line in fh]

    sources, references = read(src), read(tgt)
    if args.limit:
        sources, references = sources[:args.limit], references[:args.limit]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(manifest["base_model"])
    greedy = make_greedy(manifest, encoder, decoder, tokenizer)

    hypotheses = []
    tokens = 0
    began = time.time()
    for index, text in enumerate(sources, 1):
        translated, count = greedy(text)
        hypotheses.append(translated)
        tokens += count
        if index % 100 == 0:
            print(f"  {index}/{len(sources)}", file=sys.stderr, flush=True)
    seconds = time.time() - began

    import sacrebleu
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    chrf = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2)

    size = sum(os.path.getsize(os.path.join(args.bundle, f))
               for f in os.listdir(args.bundle)
               if os.path.isfile(os.path.join(args.bundle, f)))
    label = args.label or os.path.basename(os.path.abspath(args.bundle))

    print(json.dumps({
        "label": label,
        "pair": f"{src}-{tgt}",
        "sentences": len(sources),
        "size_mb": round(size / 1048576, 1),
        "bleu": round(bleu.score, 2),
        "chrf2": round(chrf.score, 2),
        "load_ms": round(load_ms),
        "tokens": tokens,
        "seconds": round(seconds, 1),
        "tokens_per_second": round(tokens / seconds, 1),
    }))
    print(f"{label:28} {size / 1048576:6.1f} MB  BLEU {bleu.score:5.2f}  "
          f"chrF++ {chrf.score:5.2f}  {tokens / seconds:5.1f} tok/s",
          file=sys.stderr)


if __name__ == "__main__":
    main()
