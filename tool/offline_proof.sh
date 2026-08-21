#!/usr/bin/env bash
# Proves that translation keeps working with the network completely gone.
#
#   1. serve the model bundles over plain HTTP on the host
#   2. install the demo app on an Android device, launch it once: it downloads
#      and verifies the model, then translates
#   3. cut the network — Wi-Fi off, mobile data off, adb tunnel removed, host
#      server stopped
#   4. force-stop the app and launch it again from scratch
#   5. assert that it still translates, and that it reports the network as down
#
# The app is deliberately driven directly (adb install + am start) rather than
# through `flutter test`, because the test harness uninstalls the app between
# runs, which would wipe the very model whose persistence is being checked.
#
# Usage: tool/offline_proof.sh [<serial>] [<models dir>]
set -euo pipefail

SERIAL="${1:-}"
MODELS_DIR="${2:-$HOME/ot-models}"
PORT=8099
PKG=com.example.offline_translator_example
ACTIVITY="$PKG/.MainActivity"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE="$HERE/../example"
ADB="${ADB:-adb}"
[ -n "$SERIAL" ] && ADB="$ADB -s $SERIAL"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
fail() { printf '\033[31mFAIL: %s\033[0m\n' "$*" >&2; exit 1; }

cleanup() {
  [ -n "${SERVER_PID:-}" ] && kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

[ -d "$MODELS_DIR/en-fr" ] || fail "no bundle at $MODELS_DIR/en-fr (run tool/build_model.py first)"

say "1. serving $MODELS_DIR on port $PORT"
pkill -f "http.server $PORT" 2>/dev/null || true
python3 -m http.server "$PORT" --bind 127.0.0.1 --directory "$MODELS_DIR" >/dev/null 2>&1 &
SERVER_PID=$!
sleep 1
curl -sf "http://127.0.0.1:$PORT/en-fr/manifest.json" >/dev/null || fail "host server did not start"
$ADB reverse "tcp:$PORT" "tcp:$PORT" >/dev/null

say "2. building and installing the demo app in autorun mode"
(cd "$EXAMPLE" && flutter build apk --debug \
    --dart-define=OT_AUTORUN=1 \
    --dart-define=OT_MODELS_URL="http://127.0.0.1:$PORT" >/dev/null)
$ADB install -r -d "$EXAMPLE/build/app/outputs/flutter-apk/app-debug.apk" >/dev/null
$ADB shell pm clear "$PKG" >/dev/null

say "3. first launch: download, verify and translate (network up)"
$ADB logcat -c
$ADB shell am start -n "$ACTIVITY" >/dev/null
for _ in $(seq 1 120); do
  if $ADB logcat -d | grep -q 'OT_AUTORUN status='; then break; fi
  sleep 2
done
ONLINE_LOG="$($ADB logcat -d | grep 'OT_AUTORUN' || true)"
echo "$ONLINE_LOG"
echo "$ONLINE_LOG" | grep -q 'status=OK' || fail "first launch did not complete"
echo "$ONLINE_LOG" | grep -q 'installed_before=false' || echo "  (model was already present)"

say "4. cutting the network"
$ADB reverse --remove-all >/dev/null
$ADB shell svc wifi disable
$ADB shell svc data disable
kill "$SERVER_PID" 2>/dev/null || true
pkill -f "http.server $PORT" 2>/dev/null || true
unset SERVER_PID
sleep 3
curl -sf -m 3 "http://127.0.0.1:$PORT/en-fr/manifest.json" >/dev/null \
  && fail "host server is still up" || echo "  host server stopped"

say "5. relaunching the app with no network at all"
$ADB shell am force-stop "$PKG"
$ADB logcat -c
$ADB shell am start -n "$ACTIVITY" >/dev/null
for _ in $(seq 1 120); do
  if $ADB logcat -d | grep -q 'OT_AUTORUN status='; then break; fi
  sleep 2
done
OFFLINE_LOG="$($ADB logcat -d | grep 'OT_AUTORUN' || true)"
echo "$OFFLINE_LOG"

echo "$OFFLINE_LOG" | grep -q 'status=OK' || fail "offline launch failed"
echo "$OFFLINE_LOG" | grep -q 'online=false' || fail "the device still had network"
echo "$OFFLINE_LOG" | grep -q 'installed_before=true' || fail "model did not survive the relaunch"
echo "$OFFLINE_LOG" | grep -q 'Bonjour, comment allez-vous' || fail "unexpected translation"

printf '\n\033[32mPASS: the model survived the relaunch and translated with the network off.\033[0m\n'
echo "Re-enable networking with: $ADB shell svc wifi enable && $ADB shell svc data enable"
