# offline_translate example

A small app exercising the whole package: language pickers with swap, model
installation with progress, all three translation APIs, timings, and the
installed model's license read from its manifest.

## 1. Build a model bundle

```sh
cd ..
python3 -m venv .venv && source .venv/bin/activate
pip install optimum-onnx transformers torch onnx onnxruntime accelerate sentencepiece
python3 tool/build_model.py --pair en-fr --out ~/ot-models
```

## 2. Run

**macOS / iOS simulator** — they can read the host filesystem directly:

```sh
flutter run --dart-define=OT_MODELS_DIR=$HOME/ot-models
```

**Android** — serve the bundles and expose them through adb:

```sh
python3 -m http.server 8099 --bind 127.0.0.1 --directory ~/ot-models &
adb reverse tcp:8099 tcp:8099
flutter run --dart-define=OT_MODELS_URL=http://127.0.0.1:8099
```

You can also push a bundle to the app's documents directory as `ot-models/`,
in which case no define is needed.

> The macOS debug build turns the App Sandbox off so it can read
> `OT_MODELS_DIR` from an arbitrary path. Release builds keep it on and install
> into the app's own container.

## 3. Tests

```sh
# functional, on any device
flutter test integration_test/translation_test.dart -d macos \
  --dart-define=OT_MODELS_DIR=$HOME/ot-models

# benchmarks
flutter test integration_test/benchmark_test.dart -d macos \
  --dart-define=OT_MODELS_DIR=$HOME/ot-models

# the network really is off (Android, scripted end to end)
../tool/offline_proof.sh <serial>
```

`offline_proof.sh` drives the app through `adb install` + `am start` rather than
`flutter test`, because the test harness uninstalls the app between runs — which
would wipe the very model whose persistence is being checked.

## Autorun mode

`--dart-define=OT_AUTORUN=1` replaces the UI with a headless self-check that
installs the model if needed, translates, and prints machine-readable
`OT_AUTORUN …` lines to the log. That is what `offline_proof.sh` reads.
