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

# The six directions of the V1 catalogue. Every Firefox Translations student is
# English-paired — there is no `fr↔es`, `fr↔de` or `es↔de` checkpoint upstream,
# in any tier — so the remaining six directions of a four-language matrix are
# not a build away: they need a pivot through English, two passes and two
# resident models. That is a feature, not a missing file, and it is listed as
# such rather than as an empty row.
ORDER = ["en-fr", "fr-en", "en-es", "es-en", "en-de", "de-en"]

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

print("| Direction | Size | Upstream BLEU | COMET | Upstream checkpoint | Licence |")
print("|---|---:|---:|---:|---|---|")
for pair, data in rows:
    src, dst = pair.split("-")
    arrow = f"{NAMES[src]} → {NAMES[dst]}"
    if data is None:
        print(f"| {arrow} | — | — | — | — | not built |")
        continue
    manifest, size = data
    flores = (manifest.get("upstream") or {}).get("flores") or {}
    licence = manifest["license"]
    mark = "**" if licence != "Apache-2.0" else ""
    print(f"| {arrow} | {size / 1048576:.1f} MB | {flores.get('bleu', '—')} | "
          f"{flores.get('comet', '—')} | `{manifest['base_model']}` | "
          f"{mark}{licence}{mark} |")

built = sum(1 for _, d in rows if d)
print(f"\n{built}/{len(ORDER)} built, {total / 1048576:.0f} MB total "
      f"({total / built / 1048576:.1f} MB per direction). BLEU and COMET are "
      f"Mozilla's published FLORES scores for the checkpoint, copied from its "
      f"metadata into each manifest.")
