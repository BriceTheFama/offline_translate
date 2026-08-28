#!/usr/bin/env python3
"""Uploads built model bundles to a Hugging Face repository.

    pip install huggingface_hub
    huggingface-cli login
    python3 tool/upload_models.py ~/ot-models-tiny

The resulting URLs match `HttpModelSource` exactly, with no rewriting:

    https://huggingface.co/<user>/<repo>/resolve/main/en-fr/manifest.json
                          └──────── baseUrl ────────┘ └pair┘ └── file ──┘

```dart
HttpModelSource.official()                                  // the published repo
HttpModelSource(baseUrl: Uri.parse('https://your.cdn/...'))  // or your own
```

Hugging Face is the default for three reasons: the path layout is already the
one this package expects, LFS redirects to its CDN are followed transparently by
`package:http` (verified against a real object), and it keeps the models next to
the checkpoints they were converted from, which is where their provenance
belongs.

Any static host works just as well — S3, Cloudflare R2, a plain nginx — as long
as it serves `<base>/<pair>/<file>`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

MODEL_CARD = """---
license: {spdx}
tags:
  - translation
  - onnx
  - offline_translate
---

# offline_translate model bundles

int8 ONNX translation models packaged for the
[`offline_translate`](https://pub.dev/packages/offline_translate) Flutter
package: neural machine translation that runs entirely on a phone, with no
network access of any kind once a model is installed.

Each direction lives in its own directory:

```text
<pair>/manifest.json   metadata, architecture constants, SHA-256 per file
<pair>/encoder.onnx    graph
<pair>/encoder.data    int8 encoder weights (memory-mapped at runtime)
<pair>/decoder.onnx    graph, with a greedy `next_token` output built in
<pair>/decoder.data    int8 decoder weights
<pair>/embedding.data  the tied embedding, shared by both graphs (tiny-ssru)
<pair>/source.spm      upstream SentencePiece model, unmodified
<pair>/vocab.json      upstream shared vocabulary, unmodified (marian only)
```

Use them with:

```dart
final translator = await OfflineTranslator.initialize(
  languages: {{Language.en, Language.fr}},
  defaultLanguage: Language.fr,
  modelSource: HttpModelSource.official(),
);
await translator.installModel(from: Language.en, to: Language.fr);

final result = translator.translate('Hello, how are you?');
print(result.translatedText); // Bonjour, comment allez-vous ?
```

Every file is verified against the SHA-256 in its manifest before the model is
moved into place, so the transport does not have to be trusted beyond TLS.

## Directions

{table}

## Licensing

These are **modified** weights: converted to ONNX, dynamically quantised to
int8, and extended with a greedy arg-max output node. The conversion pipeline is
`tool/build_tiny_model.py` (and `tool/build_model.py` for the `marian` family) in
the package repository — that, plus the upstream checkpoint, is the source form.

{licences}

Upstream projects:

> Mozilla. *Firefox Translations models.*
> <https://github.com/mozilla/firefox-translations-models> — MPL-2.0.

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
        family = manifest["architecture"].get("family", "marian")
        rows.append(f"| `{pair}` | {size / 1048576:.1f} MB | "
                    f"{family} | "
                    f"{manifest['architecture']['vocab_size']} | "
                    f"{manifest['base_model']} | "
                    f"{manifest['license']} |")
        licences.setdefault(manifest["license"], []).append(pair)

    table = ("| Direction | Size | Family | Vocab | Upstream | License |\n"
             "|---|---:|---|---:|---|---|\n" + "\n".join(rows))

    notes = []
    for licence, pairs in sorted(licences.items()):
        joined = ", ".join(f"`{p}`" for p in pairs)
        if licence.lower().startswith("mpl"):
            notes.append(f"* **{licence}** — {joined}. File-level copyleft: "
                         f"applications that merely *use* these files are "
                         f"unaffected and keep their own licence. Anyone "
                         f"**redistributing** them must keep them under "
                         f"{licence} and make the source form available.")
        elif licence.lower().startswith("cc-by"):
            notes.append(f"* **{licence}** — {joined}. Commercial use is "
                         f"permitted **with attribution**: credit the creator, "
                         f"link the licence, and state that changes were made.")
        else:
            notes.append(f"* **{licence}** — {joined}. Keep the licence and a "
                         f"notice of modifications.")
    # The card's SPDX header has to be a single licence; when a repository
    # mixes them, the strictest one is the honest choice for the header and the
    # per-direction table carries the detail.
    order = ["mpl-2.0", "cc-by-4.0", "apache-2.0"]
    spdx = next((l for l in order
                 if any(k.lower() == l for k in licences)), "other")
    return MODEL_CARD.format(repo=repo, table=table, spdx=spdx,
                             licences="\n".join(notes))


# The canonical text, so a published repository carries the licence it claims
# rather than a link to an empty file.
MPL_TEXT_URL = "https://www.mozilla.org/media/MPL/2.0/index.txt"


def write_licence(models_dir: str, pairs: list[str]) -> None:
    """Ensures `<models>/LICENSE` holds the text the model card points at.

    The card's front matter links to `LICENSE`, so shipping an empty or absent
    one is worse than shipping none: it claims a licence the repository does not
    actually carry. MPL-2.0 is the only licence in play that imposes obligations
    on a redistributor, so when any bundle is MPL-2.0 that is the text written;
    the per-direction table in the card carries the rest.
    """
    import urllib.request

    licences = set()
    for pair in pairs:
        with open(os.path.join(models_dir, pair, "manifest.json")) as fh:
            licences.add(json.load(fh)["license"])

    path = os.path.join(models_dir, "LICENSE")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    if not any(l.lower().startswith("mpl") for l in licences):
        print(f"note: licences are {sorted(licences)}; no LICENSE file was "
              f"generated. Add {path} yourself before uploading.",
              file=sys.stderr)
        return

    print(f"fetching the MPL-2.0 text -> {path}", file=sys.stderr)
    with urllib.request.urlopen(MPL_TEXT_URL) as response:
        text = response.read().decode("utf-8")
    if "Mozilla Public License Version 2.0" not in text:
        raise SystemExit(f"{MPL_TEXT_URL} did not return the MPL-2.0 text")
    with open(path, "w") as fh:
        fh.write(text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("models", help="directory holding <pair>/ bundles")
    ap.add_argument("--repo", default="fama-corp/offline_translate",
                    help="target repository (default: the package's own)")
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
    write_licence(args.models, pairs)

    # One direction per commit, with retries.
    #
    # Uploading the whole catalogue in a single call is 194 MB in one transfer,
    # and a single dropped connection loses all of it — observed here as
    # `Bad file descriptor` and `SSL: UNEXPECTED_EOF` against the LFS S3
    # endpoint after several minutes. Per-direction commits make each transfer
    # ~32 MB, make a retry cheap, and leave the repository usable after a
    # partial run instead of unchanged.
    #
    # If a run fails with `ConnectionError: ... cas-server.xethub.hf.co`, the
    # Hub's Xet backend is having trouble rather than this script or the token:
    # `HF_HUB_DISABLE_XET=1` falls back to the classic LFS path.
    files = ["manifest.json", "*.onnx", "*.data", "source.spm", "vocab.json"]
    uploaded, failed_pairs = [], []
    for index, pair in enumerate(pairs, start=1):
        for attempt in range(1, 4):
            try:
                print(f"\n[{index}/{len(pairs)}] uploading {pair}"
                      f"{'' if attempt == 1 else f' (attempt {attempt})'} ...",
                      file=sys.stderr)
                api.upload_folder(
                    folder_path=os.path.join(args.models, pair),
                    path_in_repo=pair,
                    repo_id=args.repo,
                    repo_type="model",
                    commit_message=f"offline_translate bundle: {pair}",
                    allow_patterns=files,
                )
                uploaded.append(pair)
                break
            except Exception as error:  # noqa: BLE001 - retry any transport failure
                print(f"  {type(error).__name__}: {str(error)[:160]}",
                      file=sys.stderr)
                if attempt == 3:
                    failed_pairs.append(pair)

    print("\nuploading the model card and licence ...", file=sys.stderr)
    api.upload_folder(
        folder_path=args.models,
        repo_id=args.repo,
        repo_type="model",
        commit_message="offline_translate: model card",
        allow_patterns=["README.md", "LICENSE"],
    )

    if failed_pairs:
        print(f"\nuploaded: {', '.join(uploaded) or 'nothing'}", file=sys.stderr)
        raise SystemExit(
            f"failed after 3 attempts: {', '.join(failed_pairs)}. Re-run to "
            "retry — directions already uploaded are skipped by the Hub.")
    print(f"\ndone: https://huggingface.co/{args.repo}")
    if args.repo == ap.get_default("repo"):
        print("use  HttpModelSource.official()")
    else:
        print(f"use  HttpModelSource(baseUrl: Uri.parse('{base}'))")


if __name__ == "__main__":
    main()
