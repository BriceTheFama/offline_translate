#!/usr/bin/env bash
# Runs each runtime configuration in a fresh process and prints one table.
#
# A released ONNX Runtime session does not return its pages to the OS, so
# measuring several configurations inside one process gives meaningless memory
# deltas. One process per configuration is the only way to get real numbers.
#
# Usage: tool/bench_configs.sh <model-dir> <libonnxruntime> [config ...]
set -euo pipefail
DIR="${1:?model dir}"
LIB="${2:?path to libonnxruntime}"
shift 2
CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then
  CONFIGS=(speed lowMemory noprepack opt-none opt-basic noarena nomempattern \
           noarena+noprepack noarena+optnone noarena+nopattern all-off \
           t1 t2 t4 t6 t8 xnnpack nnapi coreml)
fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
# 'Running build hooks...' is printed without a trailing newline, so it has to
# be stripped from the line rather than filtered out as its own line.
strip() { sed -e 's/Running build hooks\.\.\.//g'; }

dart run tool/ffi_harness.dart --header | strip
for config in "${CONFIGS[@]}"; do
  dart run tool/ffi_harness.dart "$DIR" "$LIB" --config "$config" 2>/dev/null | strip
done
