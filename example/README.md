# offline_translate example

A small app exercising the whole package: language pickers with swap, model
installation with progress, all three translation APIs, timings, the installed
model's licence read from its manifest, and a live network indicator so the
offline claim can be checked by hand.

## Run it

```sh
flutter run
```

That is the whole setup. With no `--dart-define`, the app downloads the `en→fr`
bundle (32 MB) from the published repository on first launch and keeps it. Every
later launch loads it from disk and never touches the network.

**On a phone**, plug it in and run the same command. Android needs nothing but
the `INTERNET` permission, already in this app's manifest and needed *only* for
that first download — delete it and side-load the bundle instead if you would
rather ship an app that cannot reach the network at all.

## Check that it really is offline

This is the demo worth doing on a real device, and it takes a minute:

1. `flutter run`, wait for **Model en-fr loaded**, translate something.
2. Put the device in **airplane mode**.
3. Tap the refresh icon next to the model card. The chip turns **OFFLINE**.
4. Translate again — short text and long text. It still works.
5. Kill the app and relaunch it, still in airplane mode. It still works: the
   model is on disk and nothing in the translation path opens a socket.

Step 5 is the one that matters. The model survives the process, and `initialize`
reloads it without a source.

## Point it somewhere else

The default is the published bundles. Anything local wins over them, which is
what makes the airplane-mode demo possible without a server:

| | |
|---|---|
| `--dart-define=OT_MODELS_DIR=/path` | a directory on the host (desktop, simulators) |
| `ot-models/` in the app's documents directory | side-loaded on a device — `adb push`, or the Xcode file browser |
| `--dart-define=OT_MODELS_URL=https://…` | your own static host |
| *(nothing)* | `HttpModelSource.official()` |

To build a bundle yourself:

```sh
cd ..
python3 -m venv .venv && source .venv/bin/activate
pip install torch onnx onnxruntime sentencepiece
python3 tool/build_tiny_model.py --pair en-fr --out ~/ot-models-tiny
```

Serving it to an Android device over adb:

```sh
python3 -m http.server 8099 --bind 127.0.0.1 --directory ~/ot-models-tiny &
adb reverse tcp:8099 tcp:8099
flutter run --dart-define=OT_MODELS_URL=http://127.0.0.1:8099
```

> The macOS debug build turns the App Sandbox off so it can read
> `OT_MODELS_DIR` from an arbitrary path. Release builds keep it on and install
> into the app's own container.

## Tests

The integration suite needs no configuration either — it falls back to the
published bundles and downloads on first run:

```sh
flutter test integration_test/translation_test.dart          # on the attached device
flutter test integration_test/translation_test.dart -d macos

flutter test integration_test/benchmark_test.dart -d macos   # size, load, latency, RSS

../tool/offline_proof.sh <serial>                            # scripted, Android
```

Pass `--dart-define=OT_MODELS_DIR=$HOME/ot-models-tiny` to any of them to test a
locally built bundle instead.

`offline_proof.sh` drives the app through `adb install` + `am start` rather than
`flutter test`, because the test harness uninstalls the app between runs — which
would wipe the very model whose persistence is being checked.

## Autorun mode

`--dart-define=OT_AUTORUN=1` replaces the UI with a headless self-check that
installs the model if needed, translates, and prints machine-readable
`OT_AUTORUN …` lines to the log. That is what `offline_proof.sh` reads.
