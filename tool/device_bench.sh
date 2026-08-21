#!/usr/bin/env bash
# Benchmarks on a real Android device, in profile mode.
#
#   tool/device_bench.sh <serial> [models-dir] [pair]
#
# `flutter test integration_test` only ever builds in **debug**, which inflates
# every Dart-side cost — the tokenizer especially. To measure what a shipped
# application actually does, this drives the demo app itself in profile mode
# with `--dart-define=OT_AUTORUN=bench` and reads the machine-readable
# `OT_AUTORUN` lines back out of logcat.
#
# The model bundles are served from this machine over `adb reverse`, which
# works over USB on a real device exactly as it does on an emulator.
set -euo pipefail

SERIAL="${1:?usage: device_bench.sh <serial> [models-dir] [pair]}"
MODELS_DIR="${2:-$HOME/ot-models}"
PAIR="${3:-en-fr}"
PORT=8099
PKG=com.example.offline_translator_example
ACTIVITY="$PKG/.MainActivity"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXAMPLE="$HERE/example"
ADB="${ADB:-adb} -s $SERIAL"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
fail() { printf '\033[31mFAIL: %s\033[0m\n' "$*" >&2; exit 1; }
cleanup() { [ -n "${SERVER_PID:-}" ] && kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

[ -d "$MODELS_DIR/$PAIR" ] || fail "no bundle at $MODELS_DIR/$PAIR"

say "device"
$ADB shell getprop ro.product.model
$ADB shell getprop ro.product.cpu.abi
printf 'cores: '; $ADB shell 'cat /proc/cpuinfo | grep -c processor'
printf 'memory: '; $ADB shell "grep MemTotal /proc/meminfo"

say "serving $MODELS_DIR on port $PORT"
pkill -f "http.server $PORT" 2>/dev/null || true
python3 -m http.server "$PORT" --bind 127.0.0.1 --directory "$MODELS_DIR" \
  >/dev/null 2>&1 &
SERVER_PID=$!
sleep 1
$ADB reverse "tcp:$PORT" "tcp:$PORT" >/dev/null

say "building and installing in PROFILE mode"
(cd "$EXAMPLE" && flutter build apk --profile \
    --target-platform android-arm64 \
    --dart-define=OT_AUTORUN=bench \
    --dart-define=OT_AUTORUN_PAIR="$PAIR" \
    --dart-define=OT_MODELS_URL="http://127.0.0.1:$PORT" >/dev/null)
$ADB install -r -d "$EXAMPLE/build/app/outputs/flutter-apk/app-profile.apk" >/dev/null
$ADB shell pm clear "$PKG" >/dev/null

say "running"
$ADB logcat -c
$ADB shell am start -n "$ACTIVITY" >/dev/null
for _ in $(seq 1 300); do
  $ADB logcat -d | grep -q 'OT_AUTORUN status=' && break
  sleep 2
done

LOG="$($ADB logcat -d | grep 'OT_AUTORUN' | sed 's/.*OT_AUTORUN //' || true)"
echo "$LOG"
echo "$LOG" | grep -q 'status=OK' || fail "the run did not complete"

say "APK size (arm64, profile)"
ls -la "$EXAMPLE/build/app/outputs/flutter-apk/app-profile.apk" | awk '{printf "  %.1f MB\n", $5/1048576}'

printf '\n\033[32mdone — these are profile-mode numbers, unlike the debug\n'
printf 'figures produced by `flutter test integration_test`.\033[0m\n'
