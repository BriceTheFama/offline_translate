#!/usr/bin/env python3
"""Prints the doc/models.md catalogue table from built manifests.

    python3 tool/catalogue_table.py ~/ot-models

Generated rather than hand-written so the documented sizes, vocabularies and
licences cannot drift from what was actually built.
"""
import json
import os
import sys

NAMES = {"en": "English", "fr": "French", "es": "Spanish", "de": "German"}
ORDER = ["en-fr", "fr-en", "en-es", "es-en", "en-de", "de-en",
         "fr-es", "es-fr", "fr-de", "de-fr", "es-de", "de-es"]

root = sys.argv[1] if len(sys.argv) > 1 else "build/models"
rows = []
total = 0
for pair in ORDER:
    path = os.path.join(root, pair, "manifest.json")
    if not os.path.exists(path):
        rows.append((pair, None))
        continue
    with open(path) as fh:
        manifest = json.load(fh)
    size = sum(f["size"] for f in manifest["files"])
    total += size
    rows.append((pair, (manifest, size)))

print("| Direction | Size | Vocab | Layers | Upstream checkpoint | License |")
print("|---|---:|---:|---:|---|---|")
for pair, data in rows:
    src, dst = pair.split("-")
    arrow = f"{src} → {dst}"
    if data is None:
        print(f"| {arrow} | — | — | — | `Helsinki-NLP/opus-mt-{pair}` | not built |")
        continue
    manifest, size = data
    arch = manifest["architecture"]
    licence = manifest["license"]
    mark = "**" if licence != "Apache-2.0" else ""
    print(f"| {arrow} | {size / 1048576:.1f} MB | {arch['vocab_size']} | "
          f"{arch['decoder_layers']} | `{manifest['base_model']}` | "
          f"{mark}{licence}{mark} |")

built = sum(1 for _, d in rows if d)
print(f"\n{built}/12 built, {total / 1048576 / 1024:.2f} GB total "
      f"({total / built / 1048576:.1f} MB average per direction)")
