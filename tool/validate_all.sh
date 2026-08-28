#!/usr/bin/env bash
# Validates every built bundle: checksums, graph shape, tokenizer, translation.
#
#   tool/validate_all.sh <models-dir> [python] [vectors-dir]
#
# For each direction it runs, in order:
#   1. tool/validate_bundle.py  — checksums, `next_token`, real greedy output
#   2. tool/make_tokenizer_vectors.py + the Dart harness `--tokens` mode —
#      the Dart tokenizer against transformers.MarianTokenizer, on this
#      direction's own SentencePiece model and vocabulary
#
# Step 2 is not optional busywork: every checkpoint ships its own tokenizer
# files, with its own vocabulary size, normalizer settings and — for the Firefox
# students — its own byte-fallback pieces. Being exact on en→fr says nothing
# about de→es. `make_tokenizer_vectors.py` reads the manifest and picks the
# right reference implementation for the family on its own.
set -uo pipefail

MODELS="${1:?usage: validate_all.sh <models-dir> [python] [vectors-dir]}"
PYTHON="${2:-python3}"
VECTORS="${3:-$(mktemp -d)}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
mkdir -p "$VECTORS"

pass=0
fail=0
failed_pairs=()

for dir in "$MODELS"/*/; do
  pair="$(basename "$dir")"
  [ -f "$dir/manifest.json" ] || continue
  printf '\n\033[1m── %s ─────────────────────────────────────\033[0m\n' "$pair"

  ok=1
  if ! "$PYTHON" tool/validate_bundle.py "$dir" 2>&1 | grep -v -i 'warn'; then
    ok=0
  fi

  if [ "$ok" = 1 ]; then
    if "$PYTHON" tool/make_tokenizer_vectors.py "$dir" "$VECTORS/$pair.json" \
        2>&1 | grep -v -i 'warn'; then
      if ! dart run tool/ffi_harness.dart "$dir" --tokens "$VECTORS/$pair.json" \
          2>&1 | sed 's/Running build hooks\.\.\.//g'; then
        ok=0
      fi
    else
      ok=0
    fi
  fi

  if [ "$ok" = 1 ]; then
    pass=$((pass + 1))
  else
    fail=$((fail + 1))
    failed_pairs+=("$pair")
  fi
done

printf '\n\033[1m════ %d passed, %d failed ════\033[0m\n' "$pass" "$fail"
if [ "$fail" -gt 0 ]; then
  printf '\033[31mfailed: %s\033[0m\n' "${failed_pairs[*]}"
  exit 1
fi
printf '\033[32mevery bundle verified\033[0m\n'
