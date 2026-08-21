#!/usr/bin/env python3
"""Uploads built model bundles to a Hugging Face repository.

    pip install huggingface_hub
    huggingface-cli login
    python3 tool/upload_models.py ~/ot-models --repo <user>/offline-translate-models

The resulting URLs match `HttpModelSource` exactly, with no rewriting:

    https://huggingface.co/<user>/<repo>/resolve/main/en-fr/manifest.json
                          └──────── baseUrl ────────┘ └pair┘ └── file ──┘

```dart
HttpModelSource(baseUrl: Uri.parse(
    'https://huggingface.co/<user>/<repo>/resolve/main'))
```

Hugging Face is the default suggestion for three reasons: the path layout is
already the one this package expects, LFS redirects to its CDN are followed
transparently by `package:http` (verified against a real object), and it keeps
the models next to the Apache-2.0 / CC-BY-4.0 checkpoints they were converted
from, which is where their provenance belongs.

Any static host works just as well — S3, Cloudflare R2, a plain nginx — as long
as it serves `<base>/<pair>/<file>`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

MODEL_CARD = """---
license: apache-2.0
tags:
  - translation
  - onnx
  - opus-mt
  - offline_translate
---

# offline_translate model bundles

int8 ONNX conversions of [OPUS-MT](https://github.com/Helsinki-NLP/Opus-MT)
(MarianMT) checkpoints, packaged for the
[`offline_translate`](https://pub.dev/packages/offline_translate) Flutter
package. Each direction lives in its own directory:

```text
<pair>/manifest.json   metadata, architecture constants, SHA-256 per file
<pair>/encoder.onnx    graph
<pair>/encoder.data    int8 encoder weights (memory-mapped at runtime)
<pair>/decoder.onnx    graph, with a grafted greedy `next_token` output
<pair>/decoder.data    int8 decoder weights
<pair>/source.spm      upstream SentencePiece model, unmodified
<pair>/vocab.json      upstream shared vocabulary, unmodified
```

Use them with:

```dart
final translator = await OfflineTranslator.initialize(
  modelSource: HttpModelSource(
    baseUrl: Uri.parse('https://huggingface.co/{repo}/resolve/main'),
  ),
);
await translator.installModel(from: Language.en, to: Language.fr);
```

Every file is verified against the SHA-256 in its manifest before the model is
moved into place, so the transport does not have to be trusted beyond TLS.

## Directions

{table}

## Licensing

These are **modified** weights: converted to ONNX, dynamically quantised to
int8, and extended with a greedy arg-max output node.

{licences}

Citation:

> Jörg Tiedemann and Santhosh Thottingal. *OPUS-MT — Building open translation
> services for the World.* EAMT 2020, Lisbon, Portugal.
"""


def build_card(models_dir: str, repo: str) -> str:
    rows = []
    licences = {}
    for pair in sorted(os.listdir(models_dir)):
        manifest_path = os.path.join(models_dir, pair, "manifest.json")
        if not os.path.exists(manifest_path):
            continue
        with open(manifest_path) as fh:
            manifest = json.load(fh)
        size = sum(f["size"] for f in manifest["files"])
        rows.append(f"| `{pair}` | {size / 1048576:.1f} MB | "
                    f"{manifest['architecture']['vocab_size']} | "
                    f"[{manifest['base_model']}]"
                    f"(https://huggingface.co/{manifest['base_model']}) | "
                    f"{manifest['license']} |")
        licences.setdefault(manifest["license"], []).append(pair)

    table = ("| Direction | Size | Vocab | Upstream checkpoint | License |\n"
             "|---|---:|---:|---|---|\n" + "\n".join(rows))

    notes = []
    for licence, pairs in sorted(licences.items()):
        joined = ", ".join(f"`{p}`" for p in pairs)
        if licence.lower().startswith("cc-by"):
            notes.append(f"* **{licence}** — {joined}. Commercial use is "
                         f"permitted **with attribution**: credit the creator, "
                         f"link the licence, and state that changes were made.")
        else:
            notes.append(f"* **{licence}** — {joined}. Keep the licence and a "
                         f"notice of modifications.")
    return MODEL_CARD.format(repo=repo, table=table,
                             licences="\n".join(notes))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("models", help="directory holding <pair>/ bundles")
    ap.add_argument("--repo", required=True, help="e.g. yourname/offline-translate-models")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be uploaded and the resulting URLs")
    args = ap.parse_args()

    pairs = sorted(p for p in os.listdir(args.models)
                   if os.path.exists(os.path.join(args.models, p,
                                                  "manifest.json")))
    if not pairs:
        raise SystemExit(f"no bundles found in {args.models}")
    total = sum(
        sum(f["size"] for f in json.load(
            open(os.path.join(args.models, pair, "manifest.json")))["files"])
        for pair in pairs)

    base = f"https://huggingface.co/{args.repo}/resolve/main"
    print(f"{len(pairs)} directions, {total / 1073741824:.2f} GB")
    print(f"baseUrl: {base}")
    for pair in pairs:
        print(f"  {base}/{pair}/manifest.json")

    if args.dry_run:
        print("\n(dry run, nothing uploaded)")
        return

    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise SystemExit("pip install huggingface_hub, then huggingface-cli login")

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=args.private,
                    exist_ok=True)

    card_path = os.path.join(args.models, "README.md")
    with open(card_path, "w") as fh:
        fh.write(build_card(args.models, args.repo))

    print("\nuploading ...", file=sys.stderr)
    api.upload_folder(
        folder_path=args.models,
        repo_id=args.repo,
        repo_type="model",
        commit_message=f"offline_translate bundles: {', '.join(pairs)}",
        # Only the files a bundle actually needs; anything else in the
        # directory (work files, vectors) stays local.
        allow_patterns=["*/manifest.json", "*/*.onnx", "*/*.data",
                        "*/source.spm", "*/vocab.json", "README.md"],
    )
    print(f"\ndone: https://huggingface.co/{args.repo}")
    print(f"use  HttpModelSource(baseUrl: Uri.parse('{base}'))")


if __name__ == "__main__":
    main()
