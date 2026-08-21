# Changelog

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
