# Mobile size benchmark

The measurements behind [model-comparison.md](model-comparison.md). Reference
machine: Apple M-series, 8 cores, macOS 26.5, ONNX Runtime 1.29.0. Quality is
FLORES-200 devtest (CC-BY-SA-4.0), scored with sacreBLEU through the **engine's
own greedy decoding loop**, so the numbers reflect what a device produces.

Reproduce with:

```sh
python3 tool/precision_sweep.py --pair en-fr --out build/sweep
python3 tool/eval_quality.py    build/sweep/en-fr-int8 --limit 200
python3 tool/dedup_embedding.py ~/ot-models/en-fr build/dedup/en-fr
```

---

## 1. Disk

### 1.1 `en-fr`, by precision

| precision | encoder | decoder | total | vs int8 |
|---|---:|---:|---:|---:|
| fp32 | 199 MB | 207 MB | **405.9 MB** | 3.9× |
| fp16 | 100 MB | 104 MB | **204.4 MB** | 2.0× |
| **int8** | 47.5 MB | 54.0 MB | **104.2 MB** | 1.0× |
| **int8, shared embedding** | 18.5 MB + 29.1 shared | 25.1 MB | **75.5 MB** | **0.72×** |
| int4 (`MatMulNBits`) | — | — | **264.2 MB** | 2.5× |
| int4 then int8 | — | — | **176.4 MB** | 1.7× |

fp16 converts but the resulting graph does not load — `convert_float_to_float16`
with `keep_io_types` leaves a mixed-precision `Sub`. It was not chased: fp16 is
twice int8 on disk and ONNX Runtime has no fp16 CPU kernels, so it is only
interesting for a future GPU or NPU backend.

The two int4 rows are the surprise and are explained in
[model-comparison.md §2.2](model-comparison.md#22-precision--int8-is-already-the-sweet-spot):
`MatMulNBits` only rewrites `MatMul` nodes with a constant operand, so the
embedding (`Gather`, 58 % of the bundle) and everything unmatched stay fp32.

### 1.2 Composition, int8

| | encoder | decoder | total | share |
|---|---:|---:|---:|---:|
| token embedding | 29.3 MB | 29.3 MB | 58.6 MB | **58 %** |
| feed-forward and attention | 18.0 MB | 24.5 MB | 42.5 MB | 42 % |
| layer norms, biases | 0.1 MB | 0.1 MB | 0.2 MB | 0 % |

The two embedding copies are bit-identical. Sharing one blob between the graphs
removes 29 MB.

### 1.3 Architectures compared

| | OPUS-MT base | Firefox tiny |
|---|---:|---:|
| parameters | 74.5 M | **31.3 M** |
| embedding | 30.5 M (41 %) | 12.3 M (39 %) |
| `d_model` | 512 | 384 |
| vocabulary | 59 514 | 32 000 |
| encoder layers | 6 | 6 |
| decoder layers | 6 | **4** |
| decoder self-attention | yes | **SSRU, no attention** |
| int8 on disk | 104 MB / 75.5 MB shared | **30.1 MB** |

### 1.4 The OPUS-MT family is flat

| model | vocab | params | int8 (dedup) | languages |
|---|---:|---:|---:|---|
| `opus-mt-en-fr` | 59 514 | 74.5 M | ~71 MB | 1 direction |
| `opus-mt-en-roa` | 56 671 | 73.1 M | ~70 MB | en → 15 Romance |
| `opus-mt-en-gmw` | 55 477 | 72.4 M | ~69 MB | en → West Germanic |
| `opus-mt-gmw-en` | 55 485 | 72.4 M | ~69 MB | West Germanic → en |
| `opus-mt-mul-en` | 64 172 | 76.9 M | ~73 MB | 100+ → en |
| `opus-mt-en-mul` | 64 110 | 76.9 M | ~73 MB | en → 100+ |

Covering a hundred languages costs 3 % more than covering one.

### 1.5 Vocabulary coverage

Token coverage over 2 024 FLORES sentences (English and French), against the
59 514-entry joint vocabulary. 8 139 distinct ids are used — 13.7 % of it.

| keep top-K | coverage | embedding | saving |
|---:|---:|---:|---:|
| 8 000 | 84.7 % | 3.9 MB | 25.2 MB |
| 16 000 | 92.9 % | 7.8 MB | 21.2 MB |
| 24 000 | 96.1 % | 11.7 MB | 17.3 MB |
| 32 000 | 98.0 % | 15.6 MB | 13.4 MB |
| 40 000 | 99.2 % | 19.5 MB | 9.5 MB |
| 48 000 | 99.9 % | 23.4 MB | 5.6 MB |

Not exploitable without retraining — the embedding is tied to the output
projection, so removing rows removes the ability to emit those subwords.

---

## 2. Quality

FLORES-200 devtest, greedy, sacreBLEU (BLEU + chrF++).

| model | size | sentences | BLEU | chrF++ | tok/s |
|---|---:|---:|---:|---:|---:|
| OPUS-MT `en-fr` fp32 | 405.9 MB | 200 | 45.11 | 66.24 | 222 |
| OPUS-MT `en-fr` int8 | 104.2 MB | 200 | 44.84 | 65.94 | 149 |
| **OPUS-MT `en-fr` int8** | **104.2 MB** | **1 012** | **46.18** | **66.82** | **139** |
| OPUS-MT `en-fr` int8, shared embedding | 75.5 MB | — | identical output | | |
| Firefox tiny `en-fr` (published) | 30.1 MB | 1 012 | **48.5** | — | — |

**int8 costs 0.27 BLEU** against fp32. That is the whole quantisation question
answered.

The Firefox row is Mozilla's own published number, not ours. It very likely uses
beam search where ours is greedy — typically worth 0.5-1.5 BLEU — and
detokenisation conventions differ. Treat it as "at least as good at 3.5× smaller",
not as a precise 2.3-point lead. Confirming it requires the conversion.

---

## 3. Memory and speed

Unchanged by this study; see [performance.md](performance.md) for the full
tables. The relevant figures for sizing:

| | |
|---|---|
| RSS per loaded direction | 150-200 MB (macOS 184, iOS 151, Android 182) |
| Warm sentence | 21 ms (laptop) to 32 ms (4-core phone-class) |
| ms per generated token | ~6 (8 cores) to ~10 (4 cores) |
| Memory over 2 000 translations | flat — ends below where it started |
| ONNX Runtime in the app | ~30 MB per Android ABI, ~32 MB arm64 APK |

**Disk size and RAM are not the same lever**, and our own optimisation proves
it. Sharing the embedding cuts the bundle by 28 % and leaves resident memory
untouched:

| bundle | disk | RSS (fresh process) | ms/token |
|---|---:|---:|---:|
| duplicated embedding | 104.2 MB | 339 MB / 350 MB | 6.07 / 6.40 |
| **shared embedding** | **75.5 MB** | 349 MB / 350 MB | 6.47 / 6.03 |

Two runs each, alternating. ONNX Runtime maps the shared blob once, then
pre-packs the weights into its own buffers on first inference; those copies, not
the mapped pages, are the working set.

Quantisation *does* cut both, because it shrinks what gets pre-packed. So the
ladder for RAM is quantisation and architecture — not file layout.

---

## 4. Where this leaves the size target

| | per direction | status |
|---|---:|---|
| shipped today | 104 MB | 12/12 built |
| shared embedding | **75.5 MB** | measured, lossless, ready |
| Firefox student, converted | ~30 MB | needs SSRU conversion |
| + int4 on its layers | ~22 MB | plausible, unmeasured |
| the < 20 MB target | — | needs training a student, not converting one |

For a two-language application — the common case — that is 208 MB today,
151 MB with the free change, and 60 MB with the Firefox student models.
