# Technical decision record

This document records the Phase 1 audit: which model, which runtime, which
tokenizer, and why. Every number in it was measured on this project, not quoted
from a datasheet. The reference machine is an Apple M-series laptop (8 cores,
macOS 26.5); device numbers are in [performance.md](performance.md).

---

## 1. Model family

### 1.1 Candidates considered

| | **OPUS-MT / MarianMT** | **NLLB-200 distilled 600M** | **M2M-100 418M** | **mBART-50** | **Small LLM (Gemma 3 270M / Qwen3 0.6B)** |
|---|---|---|---|---|---|
| Architecture | 6+6 encoder-decoder, d_model 512 | 12+12, d_model 1024 | 12+12, d_model 1024 | 12+12, d_model 1024 | decoder-only |
| Params (en→fr) | **74 M** | 615 M | 484 M | 611 M | 270 M – 600 M |
| Size int8 on disk | **104 MB** | ~620 MB | ~490 MB | ~610 MB | 200 MB – 700 MB |
| Directions per file | 1 | 200 × 200 | 100 × 100 | 50 × 50 | any (prompted) |
| Quality (en→fr, published BLEU) | 50.5 (Tatoeba) | ~44 (FLORES) | ~42 (FLORES) | ~40 | varies, unreliable |
| Tokenizer | SentencePiece Unigram | SentencePiece BPE | SentencePiece | SentencePiece | BPE |
| ONNX export | optimum, works | optimum, works | optimum, works | optimum, works | partial |
| **License** | **Apache-2.0** (most pairs) | **CC-BY-NC-4.0** | CC-BY-NC-4.0 | MIT (model), CC-BY-NC for data | Gemma Terms / Apache-2.0 |
| **Commercial use** | **yes** | **no** | **no** | ambiguous | Gemma: restricted; Qwen: yes |
| Android / iOS viable | yes | RAM-marginal | RAM-marginal | RAM-marginal | RAM-marginal |

### 1.2 Decision: OPUS-MT (MarianMT), one model per direction

**Licensing settles it before performance does.** NLLB-200 and M2M-100 are both
`CC-BY-NC-4.0` — *non-commercial*. A package whose stated goal is to be usable
inside commercial applications cannot ship them, and cannot recommend them.
This is verifiable from the model cards:

```console
$ curl -s https://huggingface.co/api/models/facebook/nllb-200-distilled-600M | jq .cardData.license
"cc-by-nc-4.0"
$ curl -s https://huggingface.co/api/models/Helsinki-NLP/opus-mt-en-fr | jq .cardData.license
"apache-2.0"
```

Among the remaining candidates OPUS-MT also wins on the mobile criteria that
matter: it is **5-6× smaller**, it is the only one that fits comfortably in a
phone's memory budget, and for the four European languages in scope its
published quality is *higher* than the massively multilingual models, which pay
for their breadth with per-pair quality.

General-purpose small LLMs were rejected for a different reason: translation
quality is unpredictable, output is unbounded (they chat, they refuse, they
continue the text), and per-token cost is far higher for the same output length.
A 74 M-parameter dedicated seq2seq model beats a 270 M-parameter generalist at
this one job, by a wide margin, on every axis we care about.

### 1.3 Specialised models vs one multilingual model

This is the choice between "12 files" and "1 file", and it was decided by
measuring what a user actually pays for.

| | **A: 12 specialised OPUS-MT** | **B: one multilingual model** | **C: hybrid (English pivot)** |
|---|---|---|---|
| Total size, all 12 directions | 12 × 104 MB = 1.25 GB | 620 MB (NLLB int8) | 6 × 104 MB = 626 MB |
| Size for the *typical* user (1-2 directions) | **104-208 MB** | 620 MB | 104-208 MB |
| RAM with one direction loaded | **≈ 340 MB** | ≈ 1.4 GB | ≈ 340 MB |
| Quality en↔fr | **best** | lower | best |
| Quality fr↔de | best | lower | degraded (double translation) |
| Latency fr↔de | 1 pass | 1 pass | **2 passes** |
| Commercial license | **yes** | no (NLLB/M2M) | yes |

**Chosen: A, specialised models, downloaded per direction.**

The "1.25 GB total" figure for option A is misleading: nobody installs all
twelve. The package downloads exactly the directions an application asks for,
so the number that matters is the 104 MB per direction the user actually pays.
Option B loses on every count *and* cannot be licensed commercially.

Option C (pivot through English) is kept as a documented fallback for the six
non-English directions, not as the default: it halves storage for a user who
needs many directions, at the cost of doubled latency and compounded errors.
`OfflineTranslator` resolves a direction through `ModelManager`, so adding a
pivot strategy later is a change in one class and no change to the public API.

### 1.4 Direction availability

All twelve directions exist as separate Helsinki-NLP checkpoints, and the
licenses were checked individually — see [models.md](models.md) for the table.

---

## 2. Runtime

### 2.1 Candidates

| | **ONNX Runtime** | **TFLite / LiteRT** | **executorch** | **llama.cpp / GGUF** | **Custom native (ggml, ctranslate2)** |
|---|---|---|---|---|---|
| Flutter binding on pub.dev | `onnxruntime`, `flutter_onnxruntime` | `tflite_flutter` | none | via `fllama` forks | none |
| Encoder-decoder + KV cache | native (optimum export) | painful (no dynamic shapes) | immature | LLM-shaped, not seq2seq | CTranslate2 is *ideal* but has no mobile build |
| Dynamic sequence length | yes | limited | yes | yes | yes |
| int8 dynamic quantisation | yes | yes | yes | yes | yes |
| Android arm64 / armv7 | yes | yes | yes | yes | build it yourself |
| iOS arm64 | yes | yes | yes | yes | build it yourself |
| Accelerators | XNNPACK, NNAPI, CoreML | NNAPI, GPU, CoreML | XNNPACK, CoreML | Metal | — |
| Runtime size added to app | ~46 MB (full build) | ~3 MB | ~5 MB | ~3 MB | — |
| **Synchronous inference from Dart** | **yes (FFI)** | yes (FFI) | — | — | — |

### 2.2 Decision: ONNX Runtime, through in-package FFI bindings

ONNX Runtime is the only option that handles a **dynamic-shape encoder-decoder
with a KV cache** out of the box, which is exactly what MarianMT is. TFLite
would require rewriting generation around fixed-size buffers; executorch is not
ready; llama.cpp is built for decoder-only models.

CTranslate2 would technically be the best engine for Marian specifically — it is
purpose-built for this architecture and is roughly 2× faster than ORT here — but
it publishes no Android or iOS binaries, so shipping it would mean maintaining
cross-compiled builds for four ABIs inside this package. That is a large,
ongoing cost for a 2× win, and it is recorded here as the most promising future
optimisation rather than the current choice.

**The binding is written in this package**, with `ffigen`, against ONNX Runtime
1.29.0. The POC used the published `onnxruntime` 1.4.1 plugin, which proved the
approach and then hit four limits that could not be fixed from outside it: it
pinned ONNX Runtime 1.15.1, it leaked about 1 KB of native memory per generated
token, it exposed neither `AddSessionConfigEntry` nor the allocator controls,
and it shipped no `x86_64` Android library.

| | published plugin (POC) | in-package bindings |
|---|---|---|
| ONNX Runtime | 1.15.1 (2023-06) | **1.29.0 (2026-08)** |
| Native leak per token | ~1 KB | **none** |
| Session configuration | threads + graph level | **threads, graph level, arena, memory pattern, pre-packing, execution providers** |
| Android ABIs | arm64-v8a, armeabi-v7a | **+ x86, x86_64** |
| Allocations per decoding step | 1 tensor + 42 C strings | **none** |
| `translateSync` off the UI isolate | not possible | **`translate()` runs on a worker sharing the same sessions** |

`doc/onnx-runtime.md` records the version audit, the generated-bindings design,
the measurements, and the two platform problems that had to be solved on the
way: the Android AAR's telemetry `ContentProvider`, and dead-stripping of the
Apple static archive.

Everything above the `TranslationEngine` interface was untouched by the
replacement, which is what that interface exists for.

---

## 3. Tokenizer

### 3.1 The problem

An ONNX file is not a translator. The full chain has to work:

```text
String → normalize → segment → ids → encoder → decoder loop → ids → String
```

MarianMT's tokenizer is a SentencePiece **Unigram** model (`source.spm`,
32 000 pieces) whose output pieces are mapped through a **separate shared
vocabulary** (`vocab.json`, 59 514 entries covering both languages). Getting
either half wrong produces text that looks plausible and translates badly.

### 3.2 Candidates

| | **Port to pure Dart** | **FFI to libsentencepiece** | **FFI to HF `tokenizers` (Rust)** | **Platform channels to native tokenizers** |
|---|---|---|---|---|
| Extra binary per platform | **none** | ~1.5 MB × 4 ABIs | ~3 MB × 4 ABIs | ~2 MB × 2 |
| Works in `flutter test` (host VM) | **yes** | no | no | no |
| Synchronous | **yes** | yes | yes | no |
| Build complexity | **none** | cross-compile + podspec | cross-compile + podspec | two native codebases |
| Risk | divergence from the reference | low | low | low |

### 3.3 Decision: pure Dart, validated against the reference implementation

A pure-Dart tokenizer adds nothing to the app binary, runs on every platform
including tests, and is synchronous — which `translateSync()` needs. The one
real risk is *divergence*: a Dart reimplementation that is subtly different from
`sentencepiece` produces different ids and therefore different translations.

That risk is retired by testing rather than by hoping. Three pieces were ported
faithfully from the C++ source:

* **`SpmNormalizer`** — `nmt_nfkc` normalisation. This is not optional: the
  model's `normalizer_spec` carries a 237 KB `precompiled_charsmap`, a
  darts-clone double-array trie that maps `½ → 1⁄2`, `Ⅻ → XII`, `ﬁ → fi`,
  `ｱ → ア`, collapses whitespace, and so on. The trie is read straight out of
  the `.spm` file and traversed in Dart, byte by byte, exactly as
  `Normalizer::Normalize` does.
* **`UnigramSegmenter`** — a port of `Model::EncodeOptimized`, including the
  unknown-node penalty (`min_score - 10`), the score-reset threshold, and the
  **32-bit** score accumulation. That last detail matters: accumulating in
  `double` instead of `float` changes which of two near-tied segmentations wins,
  and produced a different (still valid, but non-reference) split of long
  repeated strings until it was fixed.
* **`MarianTokenizer`** — piece→id mapping through `vocab.json`, the `</s>`
  suffix, the `>>xx<<` language-prefix convention, and the merging of
  consecutive unknown pieces that `SentencePieceProcessor` performs after
  Viterbi (this is what turns `👨‍👩‍👧‍👦` into a single `<unk>` rather than seven).

**Validation.** `test/tokenizer_test.dart` compares the Dart output against ids
produced by `transformers.MarianTokenizer` and pieces produced by
`sentencepiece` itself, over:

* 29 hand-picked edge cases (empty, whitespace-only, tabs and newlines, combining
  accents, full-width forms, Roman numerals, ligatures, emoji, ZWJ sequences,
  regional indicators, Arabic-Indic digits, CJK, a 300-character run of one
  letter, URLs, mixed scripts);
* a 1 793-case fuzz corpus including 300 strings of uniformly random code points
  across the whole Unicode range.

All 1 822 cases match exactly — pieces and ids. The fixtures and the generator
(`tool/gen_vectors.py` equivalent, see `doc/models.md`) are in the repository so
the comparison can be re-run against any new model.

Because the tokenizer consumes the **original, unmodified `source.spm` and
`vocab.json`**, users can verify a shipped bundle against the upstream
Hugging Face checksums. No derived tokenizer artefact is introduced.

---

## 4. Generation strategy and graph shape

### 4.1 Merged decoder, and the two traps in it

`optimum` exports three decoder graphs. The **merged** one contains both a
"first step" branch and a "cached step" branch behind an `If` node driven by the
`use_cache_branch` input, which is what lets one session serve the whole loop.
Two behaviours had to be discovered by measurement:

1. **Cross-attention KV is produced once.** On the first step the
   `present.*.encoder.*` outputs hold the real cross-attention keys and values.
   On every cached step they are **empty `(0, 8, 1, 64)` placeholders**. Feeding
   those back produces
   `Reshape … The dimension with value zero exceeds the dimension size`. The
   engine captures the encoder KV on step 0 and reuses it unchanged.
2. **Quantising the merged graph is a no-op for its `If` branches.**
   `quantize_dynamic` on `decoder_model_merged.onnx` leaves the weights inside
   the branches at fp32: 224 MB in, 214 MB out. Quantising
   `decoder_model.onnx` and `decoder_with_past_model.onnx` *separately* and
   merging afterwards keeps the int8 weights and deduplicates the shared
   initialisers: **54 MB**. `tool/build_model.py` does it in that order.

### 4.2 Greedy vs beam search

`generation_config.json` asks for `num_beams: 4`. V1 uses **greedy** decoding:

| | greedy | beam 4 |
|---|---|---|
| Decoder passes per token | 1 | 4 |
| Latency (measured, 25-token output) | **148 ms** | ~590 ms (projected 4×) |
| KV cache memory | 1× | 4× |
| Implementation | 40 lines | reordering the cache every step |

Sampled quality on the fixture sentences is already good — *"I would like to
book a table for two people at eight o'clock tonight."* → *"Je voudrais réserver
une table pour deux personnes à huit heures ce soir."* A 4× latency and 4×
memory cost is not justified by what greedy is losing on sentence-level input.
Beam search is left as a `GenerationConfig` extension point; the honest position
is that it should be added only once a BLEU comparison on a held-out set shows a
gap worth paying for.

### 4.3 Arg-max inside the graph

The decoder emits logits of shape `[1, 1, 59514]` — 238 KB of float per token.
Reading that through the plugin's `OrtValue.value` getter builds a
`List<num>` element by element: 59 514 boxed doubles per step.

`tool/build_model.py` therefore grafts two nodes onto the decoder:

```text
logits ──▶ Add(bad_words_mask) ──▶ ArgMax(axis=-1) ──▶ next_token : int64[1, 1]
```

The engine requests only `next_token` and the KV outputs, so **one 8-byte
integer** crosses the FFI boundary per generated token instead of 238 KB. The
mask reproduces `bad_words_ids: [[59513]]` from the generation config, which
forbids the pad token that Marian also uses as `decoder_start_token_id`.

The KV tensors never enter Dart at all: the `OrtValue` pointers returned by one
`run()` are passed straight into the next.

---

## 5. Quantisation and memory

int8 dynamic quantisation is not in question: it is a 4× size reduction (fp32
encoder 199 MB → 47 MB) with output that is, on the fixture sentences,
occasionally *better* than the Xenova-published int8 conversion and
indistinguishable from fp32 in fluency.

Two model-packaging decisions follow from measurement:

**Weights ship as external data.** Moving them into a companion `.data` file
lets ONNX Runtime memory-map them instead of copying them out of the protobuf.
Measured at 466 MB → 344 MB resident for no change in inference speed — the
clearest single win, and what `tool/build_model.py` emits.

**Quantise before merging the decoder.** Running `quantize_dynamic` on
`decoder_model_merged.onnx` leaves the weights inside its `If` branches at
fp32: 224 MB in, 214 MB out. Quantising the two unmerged decoders separately
and merging afterwards keeps int8 and deduplicates the shared initialisers:
54 MB.

Runtime tuning turned out to matter far less than the engine itself. The full
matrix is in [onnx-runtime.md](onnx-runtime.md) §4; the short version, on
ONNX Runtime 1.29 with the current engine:

| | RSS | ms/token |
|---|---|---|
| **default** (opt=all, pre-packing on, 4 threads) | **336 MB** | **6.00** |
| graph optimisation off | 316 MB | 7.70 |
| pre-packing off | 465 MB | 6.79 |

Note the middle and last rows. Disabling graph optimisation is the only knob
that still buys memory, and it buys 20 MB for 28 % throughput. Disabling
pre-packing — which an earlier measurement through Python on ONNX Runtime 1.23
suggested would save ~180 MB — actually costs **129 MB more**. That earlier
number is corrected rather than carried forward; `RuntimeConfig.prePackWeights`
defaults to `true` and its documentation says why.

## 6. Accelerators

The engine can append XNNPACK, NNAPI and CoreML. None is enabled by default,
deliberately: the brief is explicit that an accelerator should be switched on
because a benchmark says so, not because it exists. Now there are benchmarks.

* **CoreML** works and is **8 % slower** (156 vs 144 ms per sentence) for
  **+46 MB** and a 2.5× longer load. A decoder whose sequence length grows every
  step is the wrong shape for it, as expected.
* **XNNPACK** is not compiled into Apple builds of ONNX Runtime. The Android
  build has it; see [performance.md](performance.md).
* **NNAPI** is deprecated as of Android 15 and falls back to CPU for most
  dynamic-shape graphs. It is reachable only through a dedicated Android-only
  entry point, which the engine looks up at runtime.

The default stays CPU with four intra-op threads, which measured fastest on
both an 8-core and a 4-core machine.
