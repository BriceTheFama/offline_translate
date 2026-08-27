#!/usr/bin/env python3
"""Builds fp32 / fp16 / int8 / int4 variants of one direction and sizes them.

    python3 tool/precision_sweep.py --pair en-fr --out build/sweep

Produces one bundle per precision so `tool/eval_quality.py` can score each with
the engine's own decoding loop. Quantisation only ever touches the weights; the
graph, the tokenizer and the grafted `next_token` output are identical across
variants, so any quality difference is attributable to precision alone.

Two limits are worth knowing before reading the numbers:

* **int4 reaches the MatMuls, not the embedding.** ONNX Runtime's 4-bit path is
  `MatMulNBits`, which replaces `MatMul` with a constant operand. The token
  embedding is consumed by `Gather`, so it stays int8. Since the embedding is
  39-58 % of a Marian bundle, int4 saves far less than "half of int8".
* **fp16 is not a speed win on CPU.** ONNX Runtime has no fp16 CPU kernels for
  most operators and inserts casts back to fp32, so fp16 is best read as a
  disk-size data point for a future GPU/NPU backend.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

import onnx


def run(*command: str) -> None:
    subprocess.run(list(command), check=True)


def export_fp32(model_id: str, workdir: str) -> str:
    out = os.path.join(workdir, "onnx")
    if not os.path.isdir(out):
        run(sys.executable, "-m", "optimum.commands.optimum_cli", "export",
            "onnx", "--model", model_id, "--task",
            "text2text-generation-with-past", "--opset", "17", out)
    return out


def size_of(directory: str) -> float:
    return sum(os.path.getsize(os.path.join(directory, f))
               for f in os.listdir(directory)
               if os.path.isfile(os.path.join(directory, f))) / 1048576


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair", default="en-fr")
    ap.add_argument("--out", default="build/sweep")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--precisions", default="fp32,fp16,int8,int4")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from build_model import (CATALOGUE, add_next_token, quantize,
                             quantized_merged_decoder, sha256,
                             save_with_external_data, BUNDLE_VERSION)

    model_id, license_id = CATALOGUE[args.pair]
    src_lang, tgt_lang = args.pair.split("-")
    workdir = args.workdir or os.path.join(args.out, "_work")
    os.makedirs(workdir, exist_ok=True)
    onnx_dir = export_fp32(model_id, workdir)
    with open(os.path.join(onnx_dir, "config.json")) as fh:
        config = json.load(fh)

    wanted = args.precisions.split(",")
    results = []

    for precision in wanted:
        dest = os.path.join(args.out, f"{args.pair}-{precision}")
        os.makedirs(dest, exist_ok=True)
        print(f"\n=== {args.pair} {precision} ===", flush=True)

        if precision == "fp32":
            encoder = onnx.load(os.path.join(onnx_dir, "encoder_model.onnx"))
            decoder = onnx.load(os.path.join(onnx_dir,
                                             "decoder_model_merged.onnx"))
        elif precision == "fp16":
            from onnxconverter_common import float16
            encoder = float16.convert_float_to_float16(
                onnx.load(os.path.join(onnx_dir, "encoder_model.onnx")),
                keep_io_types=True, disable_shape_infer=True)
            decoder = float16.convert_float_to_float16(
                onnx.load(os.path.join(onnx_dir, "decoder_model_merged.onnx")),
                keep_io_types=True, disable_shape_infer=True)
        elif precision == "int8":
            enc_q = os.path.join(workdir, "encoder_q8.onnx")
            if not os.path.exists(enc_q):
                quantize(os.path.join(onnx_dir, "encoder_model.onnx"), enc_q)
            encoder = onnx.load(enc_q)
            decoder = onnx.load(quantized_merged_decoder(onnx_dir, workdir))
        elif precision == "int4":
            # The class moved between ONNX Runtime releases; accept either.
            try:
                from onnxruntime.quantization.matmul_nbits_quantizer import (
                    MatMulNBitsQuantizer as _Quantiser, DefaultWeightOnlyQuantConfig)
                def _make(model):
                    return _Quantiser(model, algo_config=DefaultWeightOnlyQuantConfig(
                        block_size=32, is_symmetric=True, bits=4))
            except ImportError:
                from onnxruntime.quantization.matmul_4bits_quantizer import (
                    MatMul4BitsQuantizer as _Quantiser)
                def _make(model):
                    return _Quantiser(model, block_size=32, is_symmetric=True)

            def to_int4(path, out_path):
                if os.path.exists(out_path):
                    return onnx.load(out_path)
                quantiser = _make(onnx.load(path))
                quantiser.process()
                quantiser.model.save_model_to_file(out_path, True)
                return onnx.load(out_path)

            encoder = to_int4(os.path.join(onnx_dir, "encoder_model.onnx"),
                              os.path.join(workdir, "encoder_int4.onnx"))
            decoder = to_int4(os.path.join(onnx_dir,
                                           "decoder_model_merged.onnx"),
                              os.path.join(workdir, "decoder_int4.onnx"))
        else:
            raise SystemExit(f"unknown precision {precision}")

        save_with_external_data(encoder, os.path.join(dest, "encoder.onnx"))
        add_next_token(decoder, config["vocab_size"], config["pad_token_id"])
        save_with_external_data(decoder, os.path.join(dest, "decoder.onnx"))
        for name in ("source.spm", "vocab.json"):
            shutil.copyfile(os.path.join(onnx_dir, name),
                            os.path.join(dest, name))

        files = []
        for name in sorted(os.listdir(dest)):
            path = os.path.join(dest, name)
            if not os.path.isfile(path) or name == "manifest.json":
                continue
            files.append({"name": name, "size": os.path.getsize(path),
                          "sha256": sha256(path)})
        heads = config["decoder_attention_heads"]
        manifest = {
            "from": src_lang, "to": tgt_lang, "version": BUNDLE_VERSION,
            "checksum": "sweep", "base_model": model_id, "license": license_id,
            "quantization": precision,
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

        megabytes = size_of(dest)
        results.append((precision, megabytes, dest))
        print(f"  {precision}: {megabytes:.1f} MB -> {dest}", flush=True)

    print(f"\n{'precision':10} {'size':>10}")
    for precision, megabytes, _ in results:
        print(f"{precision:10} {megabytes:9.1f} MB")
    print("\nScore each with:")
    for _, _, dest in results:
        print(f"  python3 tool/eval_quality.py {dest} --limit 200")


if __name__ == "__main__":
    main()
