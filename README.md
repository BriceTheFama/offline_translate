# offline_translate

Neural machine translation that runs entirely on the device. After a model is
installed, translation performs **no network access of any kind** — no HTTP, no
DNS, no telemetry. Text never leaves the phone.

Built on the [Firefox Translations](https://github.com/mozilla/firefox-translations-models)
student models, converted from Marian to ONNX by this repository's tooling and
run under ONNX Runtime 1.29 through FFI bindings written here, with a
SentencePiece tokenizer ported to pure Dart.

```dart
final translator = await OfflineTranslator.initialize(
  languages: {Language.en, Language.fr},
  defaultLanguage: Language.fr,
);

// Short text — synchronous. No `await`, the result is simply there.
final result = translator.translate('Hello, how are you?');
print(result.translatedText); // Bonjour, comment allez-vous ?

// Long text — asynchronous, chunked, off the UI isolate.
final article = await translator.translateLong(longText);
```

| | |
|---|---|
| Model | Firefox Translations student, 31.3 M parameters, int8 |
| Model size | **32.3 MB per direction** |
| Short sentence | **8 ms** on a laptop |
| 100 words | **76 ms** |
| Memory | **+63 MB** per loaded direction, constant in output length |
| Languages | `en↔fr` shipped; `es` and `de` build from the same pipeline |
| Platforms | Android (arm64-v8a, armeabi-v7a, x86_64), iOS (arm64), macOS |
| License | MIT (package) · MPL-2.0 (models) |

---

## Status

**`en→fr` is built, converted, quantised and running end to end**, and
`flutter test` proves it on the real model:

```console
$ flutter test test/end_to_end_test.dart
en -> fr, on the real engine translate is synchronous and returns the expected French
en -> fr, on the real engine translateLong is asynchronous and chunks a document
en -> fr, on the real engine translates with every HTTP client in the process disarmed
...
All tests passed!
```

The `en-fr` bundle is **published** at
[`fama-corp/offline_translate`](https://huggingface.co/fama-corp/offline_translate)
and the whole path from that URL to a translation is verified by one command:

```console
$ flutter test tool/verify_published_test.dart
installs en-fr from the published repository
every downloaded file matches its manifest checksum
translates, with every HTTP client in the process disarmed
All tests passed!
```

That test is deliberately *not* in `test/` — `flutter test` must never depend on
a network or a third-party host being up, so it has to be asked for by name.

The model is **3.2× smaller and 6-14× faster** than the OPUS-MT bundle this
package shipped before — see [doc/model-decision.md](doc/model-decision.md) for
the comparison and the measurements. The older OPUS-MT pipeline
(`tool/build_model.py`, twelve directions, Apache-2.0) is kept in the repository
and still runs: `ModelFamily` in each manifest tells the engine which decoding
loop a bundle needs, and both are supported.

The inference layer runs on ONNX Runtime 1.29.0 through FFI bindings written in
this package. See [doc/onnx-runtime.md](doc/onnx-runtime.md).

---

## Install

```yaml
dependencies:
  offline_translate: ^0.3.0
```

**Android** — `minSdk 24`. `INTERNET` is only needed if you download models at
runtime:

```xml
<uses-permission android:name="android.permission.INTERNET"/>
```

This package does **not** depend on the ONNX Runtime AAR as a Gradle module,
which would add an `ai.onnxruntime.TelemetryInitializer` `ContentProvider` and
two permissions to your manifest. Gradle extracts `libonnxruntime.so` from it
and nothing else. Checked on a debug APK built by a *fresh* app that depends on
this package and nothing more:

```console
$ unzip -l app-debug.apk | grep onnxruntime
 32120992  lib/arm64-v8a/libonnxruntime.so
 22727676  lib/armeabi-v7a/libonnxruntime.so
 38473056  lib/x86_64/libonnxruntime.so

$ strings AndroidManifest.xml | grep -c 'permission\.'      # 0
$ strings AndroidManifest.xml | grep -c TelemetryInitializer # 0
$ unzip -l app-debug.apk | grep -c ai/onnxruntime            # 0
```

The AAR also ships an `x86` library; Flutter no longer builds that ABI, so it
never reaches an app. See
[doc/onnx-runtime.md](doc/onnx-runtime.md#21-the-android-aar-ships-telemetry-and-this-package-does-not-use-it).

**iOS** — deployment target 15.1 or later.
**macOS** — deployment target 14.0 or later.

Both minimums come from ONNX Runtime 1.29.0's `onnxruntime-c` pod. Nothing else
to configure on either.

---

## Getting a model onto the device

Models are not bundled in the package — even 32 MB per direction does not belong
in an APK you ship to everyone, and most applications need one or two directions
out of twelve. They are downloaded once, per direction, and then live on disk.

### The short way: the published bundles

```dart
final translator = await OfflineTranslator.initialize(
  languages: {Language.en, Language.fr},
  defaultLanguage: Language.fr,
  modelSource: HttpModelSource.official(),
);

if (!await translator.isModelAvailable(from: Language.en, to: Language.fr)) {
  await translator.installModel(
    from: Language.en,
    to: Language.fr,
    onProgress: (p) => print('${p.stage.name} ${p.receivedBytes}/${p.totalBytes}'),
  );
}
```

`HttpModelSource.official()` points at
[`fama-corp/offline_translate`](https://huggingface.co/fama-corp/offline_translate)
on Hugging Face. Subsequent launches find the model already installed and
`initialize` loads it, so the `if` runs exactly once in the life of the app.

Note that this is **opt-in**: omitting `modelSource` leaves the translator with
no network path at all, which is the right default for an application that ships
its own models. Naming the source is how you ask for downloads.

Everything below is for hosting the bundles yourself.

### 1. Build a bundle

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install torch onnx onnxruntime sentencepiece

python3 tool/build_tiny_model.py --pair en-fr --out ~/ot-models-tiny
```

The script downloads the Marian checkpoint from
`mozilla/firefox-translations-models`, rebuilds it in PyTorch, exports and
quantises two ONNX graphs, shares the tied embedding between them, and then
checks the finished bundle against the float32 rebuild before it is written.

To build an Apache-2.0 OPUS-MT bundle instead — bigger and slower, but with a
permissive licence and more directions — use `tool/build_model.py --pair en-fr`,
which needs `optimum-onnx transformers accelerate` as well.

### 2. Host it

The quickest route is Hugging Face, whose URL layout is already the one this
package expects. `tool/upload_models.py` uploads the bundles and generates the
model card — the directions table, the sizes and the licence notes all come from
the manifests, so a published repository cannot drift from what it contains:

```sh
pip install huggingface_hub && hf auth login
python3 tool/upload_models.py ~/ot-models-tiny --repo <you>/<repo> --dry-run
python3 tool/upload_models.py ~/ot-models-tiny --repo <you>/<repo>
```

Any static host works — see [doc/publishing.md](doc/publishing.md) — as long as
it keeps the layout:

```text
https://cdn.example.com/models/en-fr/manifest.json
https://cdn.example.com/models/en-fr/encoder.onnx
https://cdn.example.com/models/en-fr/encoder.data
https://cdn.example.com/models/en-fr/decoder.onnx
https://cdn.example.com/models/en-fr/decoder.data
https://cdn.example.com/models/en-fr/embedding.data
https://cdn.example.com/models/en-fr/source.spm
```

### 3. Install it

```dart
final translator = await OfflineTranslator.initialize(
  languages: {Language.en, Language.fr},
  modelSource: HttpModelSource(baseUrl: Uri.parse('https://cdn.example.com/models')),
);

await translator.installModel(
  from: Language.en,
  to: Language.fr,
  onProgress: (p) => print('${p.stage.name} ${p.receivedBytes}/${p.totalBytes}'),
);
```

Files land in a staging directory, every one is checked against the SHA-256 in
the manifest, and only then is the model moved into place. A failed or corrupted
install cannot replace a working model.

**To ship models with your app instead** — no network permission at all — put
the bundles in the app's documents directory and use `DirectoryModelSource`:

```dart
final translator = await OfflineTranslator.initialize(
  languages: {Language.en, Language.fr},
  modelSource: DirectoryModelSource('/path/to/bundles'),
);
```

Or omit `modelSource` entirely when the bundles are already in the app's model
directory: without one there is no download path in the process at all.

---

## Translating

### Short text: `translate` — synchronous

Returns a `TranslationResult`, **not** a `Future`:

```dart
final result = translator.translate(
  'Hello, how are you?',
  from: Language.en,
  to: Language.fr,
);
print(result.translatedText);      // Bonjour, comment allez-vous ?
print(result.duration);            // 0:00:00.008000
```

`from` and `to` can be omitted when `initialize` was given two `languages` and a
`defaultLanguage`, which is the common case:

```dart
translator.translate('Hello world').translatedText;   // Bonjour monde
```

The model must already be in memory. `initialize` loads every installed
direction it was told about, so normally it is — otherwise this throws
`ModelNotLoadedException` rather than silently blocking on disk I/O. `preload()`
and `translateLong()` both load on demand.

**Use it for words, labels, and sentences.** Cost is roughly
`output_tokens × 1-4 ms` depending on the device, so a sentence is a few dozen
milliseconds. A 100-word paragraph is ~76 ms on a laptop, which is still several
dropped frames — use `translateLong` past a sentence or two.

### Long text: `translateLong` — asynchronous

```dart
final result = await translator.translateLong(
  article,
  from: Language.en,
  to: Language.fr,
);
print(result.chunkCount);   // how many pieces it was split into
```

Splits on paragraphs, then sentences, then clauses, then words, packs the pieces
up to the model's input window, and runs inference on a worker isolate attached
to the *same* native sessions — the model is loaded once and shared, never
reloaded per chunk. Paragraph breaks and the whitespace between chunks are
preserved, so the output has the same shape as the input.

### Very long text: `translateStream`

```dart
final buffer = StringBuffer();
await for (final chunk in translator.translateStream(book)) {
  buffer.write(chunk.translatedText);   // separator already appended
  setState(() {});
}
```

Concatenating the stream yields exactly what `translateLong` would have
returned.

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
| `ModelNotLoadedException` | synchronous `translate()` before the model is in memory |
| `UnsupportedLanguageException` | a language that was not declared at `initialize` |
| `UnsupportedLanguagePairException` | a direction with no model in the catalogue |
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

* `test/end_to_end_test.dart` translates inside an `HttpOverrides` zone where
  creating an HTTP client throws — and its translator has no model source at
  all, so there is no download path in the process to begin with;
* `tool/verify_published_test.dart` does the same immediately after downloading
  a model from the published repository, which is the case that matters: the
  network was available a second ago and is not used again;
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

`dart run tool/ffi_harness.dart <bundle> <runtime> --report`, macOS, 8 cores:

| | **Firefox student** | OPUS-MT (previous default) |
|---|---:|---:|
| bundle on disk | **32.3 MB** | 104.2 MB |
| cold start | **338 ms** | 549 ms |
| `Hello world` | **2 ms** | 25 ms |
| `Hello, how are you?` | **8 ms** | 53 ms |
| 100 words | **76 ms** | 1 046 ms |
| ms per generated token | **~0.7-1.1** | ~6.5-12.5 |
| RSS after load | **+63 MB** | +201 MB |
| RSS peak over 100 words | **+62 MB** | +283 MB |

The last row is the architecture showing through. The student decoder's
self-attention is an SSRU, so its whole history is one `[1, 384]` state per
layer and **decoding memory does not grow with the output**; OPUS-MT's key/value
cache adds another 82 MB while generating 160 tokens.

The Android and iOS figures in [doc/performance.md](doc/performance.md) were
measured on the OPUS-MT bundle and have not yet been re-run for this model.

---

## Limitations

* **`en→fr` only, for now.** The pipeline builds `fr→en`, `en→es`, `es→en`,
  `en→de` and `de→en` from the same code path — they are entries in
  `CATALOGUE` — but only `en→fr` has been built and validated.
* **Quality is sentence-level.** The model has no cross-sentence context, so
  pronouns and register can drift across the chunks of a long document.
* **Greedy decoding**, not beam search. Occasionally less fluent; 4× cheaper.
  [Why](doc/technical-decision.md#42-greedy-vs-beam-search).
* **`translate()` blocks the calling isolate.** That is the point of it, but the
  caller has to respect the size guidance above. Use `translateLong()` for
  anything longer than a sentence or two.
* **Memory is ~63 MB per loaded direction**, on top of ONNX Runtime itself.
* **One direction per model file.** Install only what you use, declare it in
  `initialize(languages: ...)`, and keep `maxLoadedModels` small.
* **No language auto-detection yet.** The architecture allows it; see
  [doc/architecture.md](doc/architecture.md#extension-points).
* **Deployment minimums are set by ONNX Runtime 1.29.0**: Android API 24,
  iOS 15.1, macOS 14.0.

---

## Example app

```sh
cd example
flutter run
```

No configuration: the app downloads the `en→fr` bundle from the published
repository on first launch and keeps it. Language pickers with swap, install
progress, all three translation APIs, timings, the model's licence read from its
manifest, and a live network indicator — put the device in airplane mode, tap
refresh, and translate anyway. See [example/README.md](example/README.md).

---

## Documentation

| | |
|---|---|
| [doc/model-decision.md](doc/model-decision.md) | **which model and why** — the candidates, the licences, the measurements, and what the conversion cost |
| [doc/technical-decision.md](doc/technical-decision.md) | why ONNX Runtime, why a Dart tokenizer, why greedy — with the measurements |
| [doc/onnx-runtime.md](doc/onnx-runtime.md) | the FFI binding: version audit, design, benchmarks, and the two platform traps |
| [doc/architecture.md](doc/architecture.md) | layers, the greedy loop, chunking, model lifecycle |
| [doc/models.md](doc/models.md) | bundle format, the twelve directions, how to build and host |
| [doc/performance.md](doc/performance.md) | full benchmark tables and the memory breakdown |
| [doc/licensing.md](doc/licensing.md) | what you must ship, and why NLLB was rejected |
| [doc/publishing.md](doc/publishing.md) | hosting the bundles, publishing to pub.dev, and testing on a real phone |
| [doc/model-comparison.md](doc/model-comparison.md) | the size study: where the megabytes go, what compresses, and what does not |
| [doc/model-distribution.md](doc/model-distribution.md) | how many models EN/FR/ES/DE needs, and what an app actually installs |
| [doc/mobile-size-benchmark.md](doc/mobile-size-benchmark.md) | every size, quality and memory measurement behind those two |

---

## License

The package is MIT. **Model weights are not** — each bundle carries its upstream
license in its manifest.

The default models are **MPL-2.0**, from
[mozilla/firefox-translations-models](https://github.com/mozilla/firefox-translations-models).
MPL-2.0 is file-level copyleft: **an application that merely uses these models is
unaffected**, but anyone redistributing the converted bundles must keep them
under MPL-2.0 and make the source form available (the upstream checkpoint and
`tool/build_tiny_model.py`).

The alternative OPUS-MT bundles are Apache-2.0, except `en→de` which is
CC-BY-4.0 and requires attribution:

> Jörg Tiedemann and Santhosh Thottingal. *OPUS-MT — Building open translation
> services for the World.* EAMT 2020, Lisbon, Portugal.

Read [doc/licensing.md](doc/licensing.md) and
[doc/model-decision.md](doc/model-decision.md#7-licence) before shipping.
