# offline_translate

Neural machine translation that runs entirely on the device. After a model is
installed, translation performs **no network access of any kind** — no HTTP, no
DNS, no telemetry. Text never leaves the phone.

Built on [OPUS-MT](https://github.com/Helsinki-NLP/Opus-MT) (MarianMT) running
under ONNX Runtime 1.29, through FFI bindings written in this package, with a
SentencePiece tokenizer ported to pure Dart.

```dart
final translator = await OfflineTranslator.initialize(
  from: Language.en,
  to: Language.fr,
  modelSource: HttpModelSource(baseUrl: Uri.parse('https://cdn.example.com/models')),
);

final result = translator.translateSync(text: 'Hello, how are you?');
print(result.translatedText); // Bonjour, comment allez-vous ?
```

| | |
|---|---|
| Languages | English, French, Spanish, German — **all 12 directions** |
| Model size | 102-120 MB per direction (int8) |
| Platforms | Android (arm64, armv7, x86, x86_64), iOS (arm64), macOS |
| Short sentence | ~21 ms on a laptop, ~32-260 ms on a phone |
| Memory | 150-200 MB per loaded direction |
| Long documents | constant memory, ~160 tok/s at any length |
| License | MIT (package) · Apache-2.0 / CC-BY-4.0 (models) |

---

## Status

**All twelve directions are built, validated and running** on macOS, an Android
arm64 emulator and an iOS simulator, including a scripted proof that translation
survives switching the network off and relaunching the app.

The inference layer runs on ONNX Runtime 1.29.0 through FFI bindings written in
this package. Compared with the published plugin the POC used, that removed a
~1 KB-per-token native leak, cut resident memory by up to 56 %, and made
`translate()` genuinely non-blocking. See
[doc/onnx-runtime.md](doc/onnx-runtime.md).

All twelve directions are built by the same pipeline
(`tool/build_model.py --pair all`) and each is verified separately — checksums,
the engine's own decoding protocol, and ~1 820 tokenizer vectors against
`transformers.MarianTokenizer` for *that* checkpoint. They are not
interchangeable: vocabulary sizes range from 58 101 to 74 822 and `en→de` is
CC-BY-4.0 rather than Apache-2.0. See [doc/models.md](doc/models.md).

---

## Install

```yaml
dependencies:
  offline_translate: ^0.3.0
```

**Android** — `minSdk 24`, all four ABIs. `INTERNET` is only needed if you
download models at runtime:

```xml
<uses-permission android:name="android.permission.INTERNET"/>
```

This package does **not** depend on the ONNX Runtime AAR, which would add an
`ai.onnxruntime.TelemetryInitializer` `ContentProvider` and two permissions to
your manifest. Gradle extracts `libonnxruntime.so` from it and nothing else —
see [doc/onnx-runtime.md](doc/onnx-runtime.md#21-the-android-aar-ships-telemetry-and-this-package-does-not-use-it).

**iOS** — deployment target 15.1 or later.
**macOS** — deployment target 14.0 or later.

Both minimums come from ONNX Runtime 1.29.0's `onnxruntime-c` pod. Nothing else
to configure on either.

---

## Getting a model onto the device

Models are not bundled in the package — 104 MB per direction does not belong in
an APK you ship to everyone. Build the bundles once, host them anywhere static,
and install per direction.

### 1. Build a bundle

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install optimum-onnx transformers torch onnx onnxruntime accelerate sentencepiece

python3 tool/build_model.py --pair en-fr --out build/models
python3 tool/validate_bundle.py build/models/en-fr
```

### 2. Host it

The quickest route is Hugging Face, whose URL layout is already the one this
package expects:

```sh
pip install huggingface_hub && huggingface-cli login
python3 tool/upload_models.py ~/ot-models --repo <you>/offline-translate-models
```

Any static host works — see [doc/publishing.md](doc/publishing.md) — as long as
it keeps the layout:

```text
https://cdn.example.com/models/en-fr/manifest.json
https://cdn.example.com/models/en-fr/encoder.onnx
https://cdn.example.com/models/en-fr/encoder.data
https://cdn.example.com/models/en-fr/decoder.onnx
https://cdn.example.com/models/en-fr/decoder.data
https://cdn.example.com/models/en-fr/source.spm
https://cdn.example.com/models/en-fr/vocab.json
```

### 3. Install it

```dart
final translator = await OfflineTranslator.initialize(
  modelSource: HttpModelSource(baseUrl: Uri.parse('https://cdn.example.com/models')),
);

await translator.installModel(
  from: Language.en,
  to: Language.fr,
  onProgress: (p) => print('${p.stage.name} ${p.fraction ?? 0}'),
);
```

Files land in a staging directory, every one is checked against the SHA-256 in
the manifest, and only then is the model moved into place. A failed or corrupted
install cannot replace a working model.

**To ship models with your app instead** — no network permission at all — put
the bundles in the app's documents directory and use `DirectoryModelSource`:

```dart
final translator = await OfflineTranslator.initialize(
  modelSource: DirectoryModelSource('/path/to/bundles'),
);
```

---

## Translating

### Short text: `translateSync`

Returns a `TranslationResult`, not a `Future`. Inference runs synchronously on
the calling isolate.

```dart
await translator.preload(from: Language.en, to: Language.fr);

final result = translator.translateSync(
  text: 'Hello, how are you?',
  from: Language.en,
  to: Language.fr,
);
print(result.translatedText);      // Bonjour, comment allez-vous ?
print(result.duration);            // 0:00:00.025000
```

The model must already be in memory — pass `from`/`to` to `initialize`, call
`preload()`, or await one `translate()` first. Otherwise this throws
`ModelNotLoadedException` rather than silently blocking on disk I/O.

**Use it for words, labels, and sentences.** Cost is roughly
`output_tokens × 6-25 ms` depending on the device, so a sentence is tens to a
couple of hundred milliseconds. A whole paragraph would be one to two seconds —
a visible freeze. Use `translate` for that.

### Long text: `translate`

```dart
final result = await translator.translate(
  text: article,
  from: Language.en,
  to: Language.fr,
);
print(result.chunkCount);   // how many pieces it was split into
```

Splits on paragraphs, then sentences, then clauses, then words, packs the pieces
up to the model's input window, and yields to the event loop between chunks so
the UI keeps running. Paragraph breaks and the whitespace between chunks are
preserved, so the output has the same shape as the input.

### Very long text: `translateStream`

```dart
final buffer = StringBuffer();
await for (final chunk in translator.translateStream(text: book)) {
  buffer.write(chunk.translatedText);   // separator already appended
  setState(() {});
}
```

Concatenating the stream yields exactly what `translate` would have returned.

---

## Managing models

```dart
await translator.isModelAvailable(from: Language.en, to: Language.fr);
await translator.installedModels();     // List<ModelInfo>
await translator.preload(from: Language.en, to: Language.fr);
await translator.unload(from: Language.en, to: Language.fr);   // free RAM, keep files
await translator.deleteModel(from: Language.en, to: Language.fr);
await translator.dispose();
```

A model is loaded once and reused for every subsequent translation. At most
`maxLoadedModels` (default 2) stay resident; the least recently used one is
unloaded past that.

```dart
final translator = await OfflineTranslator.initialize(
  modelSource: source,
  maxLoadedModels: 1,   // low-RAM devices
);
```

**Do not load all twelve directions.** Each costs 150-200 MB.

---

## Cache

Off by default. Enable it when the same strings recur — UI labels, list items,
repeated phrases:

```dart
final translator = await OfflineTranslator.initialize(
  modelSource: source,
  cache: TranslationCache(
    maxEntries: 256,
    timeToLive: Duration(hours: 12),   // optional
  ),
);
```

Keyed by (direction, exact source text), least-recently-used eviction, ~47 µs
per hit. `TranslationResult.fromCache` tells you whether a result came from it.

---

## Tuning

```dart
final translator = await OfflineTranslator.initialize(
  modelSource: source,
  generationConfig: GenerationConfig(
    maxInputTokens: 512,      // encoder window; also the chunk budget
    maxNewTokens: 512,        // hard cap per chunk
    lengthRatioLimit: 3.0,    // stop runaway repetition
  ),
  runtimeConfig: RuntimeConfig(
    threads: 0,               // 0 = pick from the core count, capped at 4
    graphOptimization: GraphOptimization.all,
    accelerator: Accelerator.cpu,
  ),
);
```

The runtime defaults are the ones that measured fastest on every device tested.
Before changing them, read [doc/performance.md](doc/performance.md#4-runtime-configuration):
XNNPACK, NNAPI and CoreML were all benchmarked and none of them helps here, and
two of the settings that read like memory/speed trades (`useMemoryArena`,
`prePackWeights`) make **both** axes worse when switched off.

---

## Errors

| Exception | When |
|---|---|
| `ModelNotInstalledException` | translating a direction that was never installed |
| `ModelNotLoadedException` | `translateSync` before the model is in memory |
| `ModelDownloadException` | the source is unreachable or returned an error |
| `ModelCorruptedException` | a file failed its size or SHA-256 check |
| `InsufficientStorageException` | not enough free space to install |
| `TranslationEngineException` | ONNX Runtime failed |
| `TranslatorDisposedException` | using a translator after `dispose()` |

All extend `OfflineTranslatorException`.

---

## Offline guarantee

After installation, the translation path opens no sockets. This is asserted, not
just claimed:

* an integration test translates inside an `HttpOverrides` zone where creating
  an HTTP client throws;
* `tool/offline_proof.sh` installs a model on an Android device, switches Wi-Fi
  and mobile data off, removes the adb tunnel, stops the host server, relaunches
  the app, and checks that it still translates:

```text
== 5. relaunching the app with no network at all
OT_AUTORUN online=false
OT_AUTORUN installed_before=true
OT_AUTORUN loaded=en-fr
OT_AUTORUN sync_ms=153 text=Bonjour, comment allez-vous ?
OT_AUTORUN status=OK

PASS: the model survived the relaunch and translated with the network off.
```

The only code in this package that opens a socket is `HttpModelSource`, reached
only from `installModel()`.

---

## Performance

Measured, debug builds. Full tables in [doc/performance.md](doc/performance.md).

| | macOS (8 cores) | iOS sim (8 cores) | Android emu (4 cores) |
|---|---|---|---|
| cold start | 435 ms | 525 ms | 1 203 ms |
| "Hello world" | 21 ms | 21 ms | 32 ms |
| 20-word sentence | 162 ms | 172 ms | 261 ms |
| 100-word paragraph | 1 056 ms | 1 139 ms | 1 830 ms |
| ms per generated token | ~6 | ~6 | ~10 |
| RSS after load | +184 MB | +151 MB | +182 MB |

Memory stays flat: 2 000 consecutive translations (48 000 generated tokens) end
*below* their starting point, and a 20 000-word document peaks 25 MB above where
it began.

`translate()` does not block the caller. During a 15 000-character document the
calling isolate kept an 8 ms timer running with a worst gap of **40 ms** on
macOS and **116 ms** on a 4-core Android emulator — against **1 073 ms** for the
same class of text through `translateSync()`.

---

## Limitations

* **Quality is sentence-level.** OPUS-MT has no cross-sentence context, so
  pronouns and register can drift across the chunks of a long document.
* **Greedy decoding**, not beam search. Occasionally less fluent; 4× cheaper.
  [Why](doc/technical-decision.md#42-greedy-vs-beam-search).
* **`translateSync` blocks the calling isolate.** That is the point of it, but
  it means the caller has to respect the size guidance above: 100 words costs
  1.1 s on a laptop and 1.5 s on a phone. Use `translate()` for anything longer
  than a sentence or two.
* **Memory is significant** — 150-200 MB per loaded direction.
* **One direction per model file.** Installing all twelve is over 1.3 GB;
  install only what you use, and keep `maxLoadedModels` small.
* **No language auto-detection yet.** The architecture allows it; see
  [doc/architecture.md](doc/architecture.md#extension-points).
* **Deployment minimums are set by ONNX Runtime 1.29.0**: Android API 24,
  iOS 15.1, macOS 14.0.

---

## Example app

```sh
cd example
flutter run --dart-define=OT_MODELS_DIR=$HOME/ot-models
```

Language pickers with swap, model install with progress, all three translation
APIs, timings, and the model's license shown from its manifest.

---

## Documentation

| | |
|---|---|
| [doc/technical-decision.md](doc/technical-decision.md) | why OPUS-MT, why ONNX Runtime, why a Dart tokenizer — with the measurements |
| [doc/onnx-runtime.md](doc/onnx-runtime.md) | the FFI binding: version audit, design, benchmarks, and the two platform traps |
| [doc/architecture.md](doc/architecture.md) | layers, the greedy loop, chunking, model lifecycle |
| [doc/models.md](doc/models.md) | bundle format, the twelve directions, how to build and host |
| [doc/performance.md](doc/performance.md) | full benchmark tables and the memory breakdown |
| [doc/licensing.md](doc/licensing.md) | what you must ship, and why NLLB was rejected |
| [doc/publishing.md](doc/publishing.md) | hosting the bundles, publishing to pub.dev, and testing on a real phone |

---

## License

The package is MIT. **Model weights are not** — each bundle carries its upstream
license in its manifest. Eleven of the twelve directions are Apache-2.0; `en→de`
is CC-BY-4.0 and requires attribution. Read
[doc/licensing.md](doc/licensing.md) before shipping.

Models come from the OPUS-MT project:

> Jörg Tiedemann and Santhosh Thottingal. *OPUS-MT — Building open translation
> services for the World.* EAMT 2020, Lisbon, Portugal.
