# Changelog

## 0.5.0

The rest of the V1 catalogue, and a supply chain that verifies itself.

### Six directions

- `fr→en`, `en→es`, `es→en`, `en→de` and `de→en` join `en→fr`. **32.3 MB each**,
  194 MB for the whole catalogue, and each verified on its own terms:
  Mozilla's published SHA-256 for the checkpoint, Mozilla's published parameter
  count for the rebuild, its own fixtures in its own source language, and its
  own tokenizer vectors. **10 933 tokenizer vectors across the six, all exact.**
- Those six are the whole of what exists directly: **every Firefox Translations
  student is English-paired**, so `fr↔es`, `fr↔de` and `es↔de` would need a
  two-pass pivot through English rather than a download. Documented in
  [doc/models.md](doc/models.md#catalogue) rather than left as empty rows.
- `--pair all` builds the catalogue, `--skip-existing` leaves what is already
  built alone, and one failing direction no longer stops the rest.

### The upstream checkpoints moved, and the pipeline now proves what it fetched

- **The Git LFS objects have been deleted from GitHub.** The batch API answers
  `410 Object does not exist on the server`, `raw` serves a 130-byte pointer and
  `media` answers 404. The `en-fr` model had come from a cache, so the download
  path had never actually run.
- `metadata.json` is still served, and it publishes the **SHA-256 of every
  checkpoint**. Bytes now come from a mirror, metadata always from Mozilla, and
  a mirror serving anything else fails the build and deletes the file. The
  `en-fr` checkpoint we had been using was confirmed byte-identical to
  Mozilla's `base-memory/enfr` this way.
- The rebuilt model's parameter count is checked against Mozilla's published
  count. They differ by exactly the 70 `_QuantMultA` scalars, which is now
  subtracted explicitly — a layer read with the wrong shape or a misread depth
  fails here instead of surfacing as slightly worse translations.

### Corrections

- **The published BLEU was attributed to the wrong model.** 48.5 belongs to
  Mozilla's `tiny` tier (256-dim, 2 decoder layers, 17.1 MB); this package ships
  `base-memory` (384-dim, 4 layers), which scores **49.6 BLEU / 0.870 COMET**.
  `doc/model-decision.md` now carries both tiers and `--tier tiny` is supported
  for anyone whose size constraint is tighter than the brief's.
- Manifests carry an `upstream` block — tier, checkpoint size, SHA-256 and
  FLORES scores — so a bundle records the quality it was published with instead
  of a number copied by hand.
- `tool/catalogue_table.py` generates the catalogue from the manifests, and
  `tool/make_tokenizer_vectors.py` picks its reference implementation from the
  manifest's family, so `tool/validate_all.sh` validates either family without
  being told which.

## 0.4.0

A smaller model and the API shape that goes with it. **Breaking.**

### A 32 MB model, 3.2x smaller and 6-14x faster

- The default model is now a **Firefox Translations student**, converted from
  Marian to ONNX by `tool/build_tiny_model.py`. `en-fr` is **32.3 MB** against
  OPUS-MT's 104.2 MB, a short sentence takes **8 ms** against 53 ms, and 100
  words take **76 ms** against 1 046 ms. Resident memory per direction drops
  from +201 MB to **+63 MB**. See [doc/model-decision.md](doc/model-decision.md)
  for the candidates, the licences and every measurement.
- Its decoder replaces self-attention with an **SSRU**, so the whole decoding
  history is one `[1, 384]` state per layer. Decoding memory is now *constant in
  the output length*: generating 109 tokens costs nothing, where OPUS-MT's
  key/value cache grew by 82 MB over 160 tokens. It also removes the need for a
  `use_cache_branch` `If` node — one graph serves every decoding step.
- The tied embedding is stored transposed and read by `Gather(axis=1)` in both
  graphs, so the 11.7 MB matrix is quantised once and shared as a single
  `embedding.data`. Storing it per graph would have made the bundle 43 MB.
- Cross-attention keys and values are computed in the **encoder** graph, so the
  encoder's hidden states never cross the FFI boundary.
- Models are **MPL-2.0**, not Apache-2.0. File-level copyleft: applications that
  use the models are unaffected, redistributors of the converted bundles are.
- The OPUS-MT pipeline is unchanged and still supported. `manifest.architecture.family`
  (`marian` or `tiny-ssru`) tells the engine which decoding loop a bundle needs,
  and `createRunner` picks between `MarianRunner` and the new `SsruRunner`.

### API

- **`translateSync(text: ...)` is now `translate(text)`** — synchronous, text
  positional, still returning `TranslationResult` rather than a `Future`.
- **`translate(text: ...)` is now `translateLong(text)`** — asynchronous, for
  documents. `translateStream` takes its text positionally too.
- **`initialize({languages, defaultLanguage})`** replaces `initialize({from, to})`.
  `languages` declares the set an application needs; every installed direction
  inside it is loaded up front, which is what makes `translate()` synchronous,
  and anything outside it raises `UnsupportedLanguageException` instead of
  asking for a download. `defaultLanguage` is the default target, and with
  exactly two languages the other one becomes the default source, so
  `translate('Hello world')` needs no direction at all.
- `modelSource` is now optional. Without one there is no download path in the
  process at all — `installModel` throws and nothing else changes.
- `UnsupportedDirectionException` is renamed `UnsupportedLanguagePairException`,
  and `UnsupportedLanguageException` is new.
- `ModelFamily` is exported; `ModelArchitecture.family` and `modelDimension` are
  new.

### Tokenizer

- **Byte fallback.** A character outside the vocabulary now becomes one `<0xNN>`
  piece per UTF-8 byte instead of a single `<unk>`, matching
  `SentencePieceProcessor`. The Firefox vocabulary enables it and the OPUS-MT one
  does not, so both paths are tested.
- **`vocab.json` is optional.** Marian models whose ids *are* SentencePiece ids
  build the identity vocabulary from the `.spm` alone, which removes 629 KB from
  every bundle.
- Byte pieces are excluded from the segmentation trie and unused pieces are
  included, matching `unigram::Model::Model`.
- `MarianTokenizer.tokenize` now returns pieces after byte expansion, so it
  agrees with `sentencepiece` exactly. **1 822 vectors** — including a 1 500-case
  Unicode fuzz corpus — match on both pieces and ids for both vocabularies.

### Packaging

- **`OfflineTranslator.onnxRuntimeLibraryPath`** and
  **`OfflineTranslator.onnxRuntimeVersion`** are public. Applications never need
  the first — Gradle and CocoaPods supply the runtime — but a host test running
  against the real engine had no way to name a local build without importing
  `src/`, which a package should not require of its users.
- The documented Android ABI list is corrected to **arm64-v8a, armeabi-v7a,
  x86_64**. The AAR does ship an `x86` library, but Flutter no longer builds
  that ABI, so claiming it was misleading.
- Verified from a *fresh* application depending only on this package: the debug
  APK carries `libonnxruntime.so` for those three ABIs, **zero permissions**,
  no telemetry `ContentProvider` and no `ai.onnxruntime` Java classes.

### Tests

- `test/end_to_end_test.dart` runs the real model through the real ONNX Runtime
  and the real public API on the host, asserting the expected French, the
  sync/async split, byte fallback, chunking, and that translation completes with
  every `HttpClient` in the process disarmed. `flutter test` is 91 tests.
- `tool/ffi_harness.dart --report` prints the size/load/latency/RSS table.
- `tool/verify_published_test.dart` downloads the published bundle from
  `HttpModelSource.official()`, verifies its checksums, translates, and proves
  the result still holds with every `HttpClient` in the process disarmed. It
  lives outside `test/` so `flutter test` never depends on a network.

### Model hosting

- **`HttpModelSource.official()`** points at the published bundles on Hugging
  Face. It is *not* the default for `initialize`: omitting `modelSource` leaves
  a translator with no network path at all, which is the right default for an
  application that ships its own models.
- `tool/upload_models.py` generates the model card from the manifests — the
  directions table, sizes and licence notes cannot drift from the contents — and
  writes the canonical MPL-2.0 text to `LICENSE` when any bundle needs it.

## 0.3.0

Renames the package and completes the language catalogue.

### Renamed

- **`offline_translator` → `offline_translate`.** The original name is taken on
  pub.dev by an unrelated package. The Dart class `OfflineTranslator` and the
  `OfflineTranslatorException` hierarchy keep their names; only the package
  identifier, the import path (`package:offline_translate/offline_translate.dart`)
  and the platform artefacts changed.

### All twelve directions

- `fr→en`, `en→es`, `es→en`, `en→de`, `de→en`, `fr→es`, `es→fr`, `fr→de`,
  `de→fr`, `es→de`, `de→es` join `en→fr`. 1.27 GB for the catalogue,
  102-120 MB per direction.
- Each is verified independently, because the checkpoints are not
  interchangeable: vocabularies range from 58 101 to 74 822 entries, the pad and
  decoder-start ids move with them, and `en→de` is CC-BY-4.0 while the other
  eleven are Apache-2.0. All of that is read from the checkpoint and frozen into
  the manifest; nothing is hard-coded.
- **21 878 tokenizer vectors matched exactly** across the twelve directions —
  the pure-Dart SentencePiece implementation needed no per-model special case.

### Tooling

- `tool/build_model.py --pair all` builds the catalogue, skips what is already
  built, keeps going past a failure, and deletes each direction's ~1.1 GB of
  intermediates as soon as its bundle is written.
- `tool/make_tokenizer_vectors.py` emits reference vectors from
  `transformers.MarianTokenizer` for any bundle; the Dart harness gained a
  `--tokens` mode that checks them without needing ONNX Runtime at all.
- `tool/validate_all.sh` gates the whole catalogue: checksums, graph shape, real
  decoding, and the tokenizer, per direction.
- `tool/catalogue_table.py` generates the documentation table from the built
  manifests, so documented sizes and licences cannot drift.
- New `example/integration_test/all_directions_test.dart` translates through
  every installed direction on device and asserts the result carries the
  expected content word.

## 0.2.0

Replaces the third-party ONNX Runtime plugin with in-package `dart:ffi`
bindings against ONNX Runtime 1.29.0, and moves asynchronous translation onto a
worker isolate. The public API is unchanged.

### Runtime

- **ONNX Runtime 1.15.1 → 1.29.0**, through bindings generated in this package
  with `ffigen` and a thin hand-written wrapper layer
  (`lib/src/engine/native/`).
- **The per-token native leak is gone.** The previous binding allocated a C
  string for each of the 42 input and output names on every `Run` and never
  freed them — about 1 KB per generated token. Names are now encoded once per
  session in an `OrtRunPlan`. Measured flat over 48 000 generated tokens.
- **A decoding step now allocates nothing.** Input tensors, the
  `use_cache_branch` flags and the first-step placeholders are created at load
  and rewritten in place; encoder buffers are allocated once at full width and
  re-viewed per input length.
- **Long documents no longer spike memory.** A 20 000-word document peaks
  25 MB above its starting point and finishes below it; throughput is constant
  at ~160 tok/s from 100 to 20 000 words.
- **Android gains `x86` and `x86_64`**, so Intel emulators work.
- New `RuntimeConfig` exposes threads, graph optimisation, pre-packing, the
  memory arena, the memory pattern and the execution provider, with
  `RuntimeConfig.speed` and `RuntimeConfig.lowMemory` presets.

### Privacy

- The official ONNX Runtime Android AAR contributes an
  `ai.onnxruntime.TelemetryInitializer` `ContentProvider` — which builds an HTTP
  client at application start — plus `INTERNET` and `ACCESS_NETWORK_STATE`
  permissions, to every app that depends on it. This package no longer uses the
  AAR as a dependency: Gradle extracts `libonnxruntime.so` from it and nothing
  else, so none of that reaches the host application. Verified on the built APK.
- `OrtEnv` additionally calls `DisableTelemetryEvents`.

### Threading

- `translate()` and `translateStream()` now run inference on a worker isolate
  that attaches to the **same** native sessions, so the model is still loaded
  once. Measured on a 15 000-character document: the calling isolate ticked
  every 8 ms with a worst gap of 40 ms.
- `translateSync()` is unchanged and still genuinely synchronous on the
  caller's isolate — and still blocks it, by design. The same class of text
  through `translateSync()` blocks for 1073 ms.
- `TranslationEngine` gains `generateChunks`, the off-isolate entry point.

### Fixed

- `TranslationEngineException.toString()` now includes the underlying cause,
  which previously hid ONNX Runtime's own error text.
- `RuntimeConfig.lowMemory` was defined as pre-packing off plus arena off plus
  no graph optimisation. Measured, that combination is **worse on both axes**;
  it is now graph optimisation off alone.

### Requirements

- Android `minSdk` 21 → **24**, iOS 13 → **15.1**, macOS → **14.0**, all
  inherited from ONNX Runtime 1.29.0's own minimums.

## 0.1.0

First release. Offline neural machine translation for Flutter, English to
French, with the architecture in place for the remaining eleven directions.

### Translation

- `translateSync()` — synchronous, returns a `TranslationResult` rather than a
  `Future`, for words and sentences.
- `translate()` — asynchronous, chunks long text along paragraph, sentence,
  clause and word boundaries, and yields to the event loop between chunks.
- `translateStream()` — one event per chunk; concatenating the stream reproduces
  what `translate()` returns.
- Paragraph structure and inter-chunk whitespace survive reassembly.

### Models

- `ModelManager` with install, verify, delete, list, and per-file SHA-256
  checks. Installs are atomic: a failed or corrupted download cannot replace a
  working model.
- `HttpModelSource` for a static host, `DirectoryModelSource` for bundles that
  ship with the app or are side-loaded.
- Models load once and are reused; at most `maxLoadedModels` (default 2) stay
  resident, least-recently-used first.
- ~104 MB per direction: int8 weights in memory-mapped external data files, plus
  the unmodified upstream `source.spm` and `vocab.json`.

### Engine

- `TranslationEngine` abstraction; `OnnxMarianEngine` is the only file that
  touches ONNX Runtime.
- Greedy decoding with a KV cache that never round-trips through Dart — ONNX
  Runtime tensor handles are passed straight from one step to the next.
- The greedy arg-max runs inside the ONNX graph, so one `int64` crosses the FFI
  boundary per token instead of a 59 514-wide logits tensor.

### Tokenizer

- SentencePiece Unigram segmentation, `nmt_nfkc` normalisation (including the
  `precompiled_charsmap` trie) and Marian vocabulary mapping, all in pure Dart.
- Validated byte-for-byte against `transformers.MarianTokenizer` and
  `sentencepiece` on 1 822 cases, including 300 random Unicode strings.

### Other

- Optional LRU translation cache with an optional time to live.
- Typed exceptions for every failure mode.
- `tool/build_model.py` builds a bundle from an upstream checkpoint;
  `tool/validate_bundle.py` checks it against the reference implementation;
  `tool/offline_proof.sh` proves translation survives losing the network.
- Example app, unit tests, integration tests and benchmarks.

### Known limitations

- Only `en → fr` is built and validated; the other directions need a build pass.
- No beam search, no language auto-detection.
- 180-350 MB resident per loaded direction.
- The `onnxruntime` plugin leaks ~1 KB of native memory per generated token, and
  does not expose `x86_64` Android binaries.
