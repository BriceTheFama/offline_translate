#!/usr/bin/env python3
"""Emits tokenizer reference vectors for a built bundle.

    python3 tool/make_tokenizer_vectors.py build/models/fr-en vectors/fr-en.json

The vectors come from the real reference implementation, not a
reimplementation of it: `transformers.MarianTokenizer` for an OPUS-MT bundle,
and `sentencepiece` itself (`--spm`) for a Firefox Translations bundle, which
has no Hugging Face checkpoint and maps SentencePiece ids straight through.
The Dart tokenizer is then checked against them by

    dart run tool/ffi_harness.dart <bundle> <libonnxruntime> --tokens <file>

This is what keeps every new direction honest. The Dart tokenizer was validated
byte-for-byte on en→fr, but each OPUS-MT checkpoint ships its own SentencePiece
model and its own shared vocabulary, and they do not all have the same
normalizer settings or vocabulary size. A direction is not finished until its
own vectors pass.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

CASES = [
    "", " ", "   ", "\n", "\t\n ",
    "Hello", "Hello, how are you?",
    "Bonjour, comment allez-vous ?",
    "Hola, ¿cómo estás?",
    "Guten Tag, wie geht es Ihnen?",
    "The quick brown fox jumps over the lazy dog.",
    "  multiple   spaces\ttab\nnewline ",
    "Naïve café — ½ price! 😀",
    "Ⅻ ｱ ﬁne",
    "ＡＢＣ　１２３",
    "école",
    "Ｃafé — “quotes” and ‘apostrophes’",
    "Numbers: 3.14159, 1,000,000 and -42%.",
    "Email: john.doe@example.com, URL: https://example.com/path?a=1&b=2",
    "Emoji test 👨‍👩‍👧‍👦 family and 🇫🇷 flag.",
    "Mixed 中文 and English and русский текст.",
    "C++ / C# / .NET — <html>&amp;</html>",
    "A" * 300,
    "Straße, Grüße, Mädchen und Bücher.",
    "El niño comió una piña en la montaña.",
    "L'été où j'ai appris à nager, où était-il ?",
    " non-breaking space",
    "​zero width​",
    "ß ẞ ﬀ ﬃ",
    "١٢٣ عربي",
    "🦈🦖🦕🧐🧘",
]

WORDS = (
    "the quick brown fox jumps over lazy dog hello world translation offline "
    "flutter model onnx runtime café naïve über niño straße Ω π ∑ 日本語 中文 "
    "русский العربية emoji 😀 🚀 ½ Ⅻ ﬁ ｱ être où déjà mañana señor Grüße"
).split()
PUNCT = list(",.!?;:'\"()[]{}—–…/\\|@#$%^&*_+=<>~`")


def fuzz(count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        parts = []
        for _ in range(rng.randint(1, 25)):
            r = rng.random()
            if r < 0.6:
                parts.append(rng.choice(WORDS))
            elif r < 0.75:
                parts.append(str(rng.randint(0, 10 ** rng.randint(1, 9))))
            elif r < 0.85:
                parts.append(rng.choice(PUNCT))
            elif r < 0.95:
                parts.append("".join(chr(rng.randint(32, 0x2FFF))
                                     for _ in range(rng.randint(1, 6))))
            else:
                parts.append("".join(chr(rng.randint(0x1F300, 0x1FAFF))
                                     for _ in range(rng.randint(1, 4))))
        out.append(rng.choice([" ", "  ", "\t", "\n", " \n ", ""]).join(parts))
    # Uniformly random code points, to reach the corners of the normalizer.
    for _ in range(count // 5):
        out.append("".join(chr(rng.randint(1, 0x10FFFF))
                           for _ in range(rng.randint(1, 30))))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bundle", help="a built bundle directory, e.g. build/models/fr-en")
    ap.add_argument("out", help="where to write the vectors")
    ap.add_argument("--fuzz", type=int, default=1500)
    ap.add_argument("--spm", action="store_true",
                    help="use sentencepiece on the bundle's own source.spm, for "
                         "models whose ids are SentencePiece ids")
    args = ap.parse_args()

    with open(os.path.join(args.bundle, "manifest.json")) as fh:
        manifest = json.load(fh)
    base = manifest["base_model"]

    if args.spm:
        import sentencepiece
        processor = sentencepiece.SentencePieceProcessor(
            model_file=os.path.join(args.bundle, "source.spm"))
        eos = manifest["architecture"]["eos_token_id"]

        class tokenizer:  # noqa: N801 - a shim with the two methods used below
            @staticmethod
            def tokenize(text):
                return processor.encode(text, out_type=str)

            @staticmethod
            def __call__(text):
                return {"input_ids": processor.encode(text) + [eos]}

        tokenizer = tokenizer()
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(base)
        if type(tokenizer).__name__ != "MarianTokenizer":
            raise SystemExit(f"{base} does not use MarianTokenizer "
                             f"(got {type(tokenizer).__name__})")

    texts = list(CASES) + fuzz(args.fuzz, seed=hash(manifest["checksum"]) & 0xFFFF)
    vectors = []
    skipped = 0
    for text in texts:
        try:
            text.encode("utf-8")
        except UnicodeEncodeError:
            skipped += 1
            continue
        vectors.append({
            "text": text,
            "ids": tokenizer(text)["input_ids"],
            "pieces": tokenizer.tokenize(text),
        })

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({
            "pair": f"{manifest['from']}-{manifest['to']}",
            "base_model": base,
            "vocab_size": manifest["architecture"]["vocab_size"],
            "vectors": vectors,
        }, fh, ensure_ascii=False)
    print(f"{manifest['from']}-{manifest['to']}: {len(vectors)} vectors "
          f"({skipped} skipped) -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
