#!/usr/bin/env bash
# Fetches the ONNX Runtime headers the FFI bindings are generated from, and a
# local runtime for tool/ffi_harness.dart to load.
#
# Nothing here is needed to *use* the package: applications get ONNX Runtime
# from the Gradle and CocoaPods dependencies declared in android/build.gradle
# and the podspecs. This is only for regenerating bindings and running the
# pure-Dart harness.
#
# Usage: tool/fetch_onnxruntime.sh [version]
set -euo pipefail

VERSION="${1:-1.29.0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$HERE/third_party/onnxruntime"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

say "1. headers from the onnxruntime-c pod archive ($VERSION)"
mkdir -p "$DEST/include"
curl -fsSL "https://download.onnxruntime.ai/pod-archive-onnxruntime-c-${VERSION}.zip" \
  -o "$WORK/pod.zip"
unzip -o -q "$WORK/pod.zip" -d "$WORK/pod"
cp "$WORK"/pod/Headers/*.h "$DEST/include/"
echo "  -> $DEST/include ($(ls "$DEST/include" | wc -l | tr -d ' ') headers)"
grep -m1 'define ORT_API_VERSION' "$DEST/include/onnxruntime_c_api.h"

case "$(uname -s)-$(uname -m)" in
  Darwin-arm64)  ASSET="onnxruntime-osx-arm64-${VERSION}.tgz" ;;
  Darwin-x86_64) ASSET="onnxruntime-osx-x86_64-${VERSION}.tgz" ;;
  Linux-x86_64)  ASSET="onnxruntime-linux-x64-${VERSION}.tgz" ;;
  Linux-aarch64) ASSET="onnxruntime-linux-aarch64-${VERSION}.tgz" ;;
  *) ASSET="" ;;
esac

if [ -n "$ASSET" ]; then
  say "2. local runtime for the harness ($ASSET)"
  curl -fsSL "https://github.com/microsoft/onnxruntime/releases/download/v${VERSION}/${ASSET}" \
    -o "$WORK/rt.tgz"
  tar xzf "$WORK/rt.tgz" -C "$WORK"
  mkdir -p "$DEST/lib"
  cp "$WORK"/onnxruntime-*/lib/libonnxruntime.* "$DEST/lib/" 2>/dev/null || true
  echo "  -> $DEST/lib"
  ls -la "$DEST/lib"
else
  say "2. no prebuilt runtime published for $(uname -s)-$(uname -m); skipping"
fi

cat <<EOF

Next:
  dart run ffigen --config ffigen.yaml
  dart compile exe tool/ffi_harness.dart -o /tmp/ffi_harness
  /tmp/ffi_harness ~/ot-models/en-fr third_party/onnxruntime/lib/libonnxruntime.dylib
EOF
