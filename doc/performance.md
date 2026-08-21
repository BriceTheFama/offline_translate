# Performance

Every number here was measured on this project. Reproduce with:

```sh
cd example
flutter test integration_test/benchmark_test.dart      -d <device> --dart-define=OT_MODELS_DIR=$HOME/ot-models
flutter test integration_test/stability_test.dart      -d <device> --dart-define=OT_MODELS_DIR=$HOME/ot-models
flutter test integration_test/runtime_config_test.dart -d <device> --dart-define=OT_MODELS_DIR=$HOME/ot-models
```

**All runs are debug builds.** Inference is native and barely affected, but the
Dart-side tokenizer and orchestration are not JIT-optimised, so a release build
is faster than what is shown. Where a number needs to be free of that overhead —
the memory work in particular — it was taken with an AOT-compiled harness
(`dart compile exe tool/ffi_harness.dart`), and says so.

---

## 1. Before and after the engine rewrite

Same model, same bundle, same devices. The only change is the inference layer:
ONNX Runtime 1.15.1 through the published `onnxruntime` plugin, versus 1.29.0
through this package's own FFI bindings.

### macOS — Apple silicon, 8 cores

| | before | after | |
|---|---|---|---|
| cold start (load model) | 569 ms | **435 ms** | −24 % |
| RSS after load | +417 MB | **+184 MB** | **−56 %** |
| first translation | 65 ms | 72 ms | — |
| warm "Hello world" | 23.4 ms | **20.6 ms** | −12 % |
| warm 20-word sentence | 193.6 ms | **161.9 ms** | −16 % |
| warm 100-word paragraph | 1 080 ms | 1 056 ms | −2 % |
| warm 500-word text | 2 476 ms | 2 549 ms | +3 % |
| 20-paragraph doc, async | 20.4 s | **18.1 s** | −11 % |
| cache hit | 48 µs | 54 µs | — |
| native leak | ~1 KB / token | **none** | |

### iOS 26.3 simulator — iPhone 17, 8 cores

| | before | after | |
|---|---|---|---|
| cold start | 677 ms | **525 ms** | −22 % |
| RSS after load | +181 MB | **+151 MB** | −17 % |
| warm "Hello world" | 25.6 ms | **20.6 ms** | −20 % |
| warm 20-word sentence | 224.0 ms | **171.7 ms** | −23 % |
| warm 100-word paragraph | 1 457 ms | **1 139 ms** | −22 % |
| warm 500-word text | 2 812 ms | **2 586 ms** | −8 % |
| 20-paragraph doc, async | — | 18.2 s | |

### Android 16 emulator — arm64, 4 cores, 4 GB

| | before | after | |
|---|---|---|---|
| cold start | 1 136 ms | 1 203 ms | +6 % |
| RSS after load | +193 MB | **+182 MB** | −6 % |
| warm "Hello world" | 30.8 ms | 32.3 ms | — |
| warm 20-word sentence | 264.2 ms | 261.4 ms | — |
| warm 100-word paragraph | 2 434 ms | **1 830 ms** | **−25 %** |
| warm 500-word text | 7 611 ms | **3 654 ms** | **−52 %** |
| 20-paragraph doc, async | 27.3 s | **25.5 s** | −7 % |

The 500-word row is the interesting one: it is two chunks, so it exercises the
per-call setup twice. That is exactly what the old binding made expensive —
42 native string allocations per generated token — and it is where removing
that shows up most.

The simulator and macOS run arm64 natively on the host CPU, so those are
**laptop-class numbers, not phone numbers**. The 4-core Android emulator is the
closest thing here to a real mid-range phone; use it as the planning number.

---

## 2. Throughput

| device | ms per generated token |
|---|---|
| macOS / iOS simulator (8 cores) | ~6 ms |
| Android emulator (4 cores) | ~10 ms |
| expect on a mid-range phone | 12-25 ms |

A translation costs roughly `output_tokens × ms_per_token`. A sentence produces
15-35 tokens; a 100-word paragraph produces about 130. The encoder is
negligible (2-8 ms); everything else is the decoder loop.

**This is why `translateSync` is scoped to short text**, and the stability suite
measures exactly that:

| | `translateSync` | `translate` |
|---|---|---|
| where inference runs | the calling isolate | a worker isolate |
| 100 words, macOS | **blocks 1 073 ms** | — |
| 100 words, Android | **blocks 1 508 ms** | — |
| 15 026-char document, macOS | — | 30.6 s, worst caller stall **40 ms** |
| 15 026-char document, iOS | — | 29.7 s, worst caller stall **91 ms** |
| 15 026-char document, Android | — | 38.9 s, worst caller stall **116 ms** |

During those documents the calling isolate kept its 8 ms timer running with a
**median gap of 8 ms** — that is, it was never meaningfully blocked. The rule
the package documents:

| input | API |
|---|---|
| a word, a label, a sentence (≲ 30 words) | `translateSync` |
| a paragraph or more | `translate` |
| a document you want to show progressively | `translateStream` |

---

## 3. Memory

### 3.1 Cost of a loaded model

| device | RSS delta after loading en→fr |
|---|---|
| macOS | +184 MB |
| iOS | +151 MB |
| Android | +182 MB |

It varies with the direction, because the vocabulary does — the embedding
matrix is the largest single tensor in the model. AOT harness, macOS, same
configuration:

| direction | vocabulary | disk | RSS | ms/token |
|---|---:|---:|---:|---:|
| en → fr | 59 514 | 104 MB | 350 MB | 6.78 |
| en → de | 58 101 | 103 MB | 343 MB | 7.21 |
| fr → es | 74 822 | 120 MB | 427 MB | 8.08 |

Plan for **150-200 MB per loaded direction** on device (the harness figures
above include the process baseline), plus 102-120 MB on disk.
`OfflineTranslator` keeps at most `maxLoadedModels` (default 2) resident and
evicts the least recently used. On a low-RAM device set it to `1` and call
`unload()` when you are done. **Do not load all twelve directions.**

Absolute RSS figures in the benchmark output are much larger — 450-800 MB —
because a debug Flutter application is already 300-750 MB before any model is
loaded. The delta is what this package costs.

### 3.2 It no longer grows

Repeated translation, measured on device (`stability_test.dart`):

| | Android | iOS |
|---|---|---|
| 100 short translations | +18 MB | **−64 MB** |
| 10 × 2 208-char documents | **−4 MB** | **−67 MB** |

And with the AOT harness, which has no Dart debug overhead to hide behind —
2 000 translations, 48 000 generated tokens:

| after | RSS | delta |
|---|---|---|
| warm-up | 520 MB | — |
| 100 translations | 356 MB | −164 MB |
| 1 000 translations | 331 MB | −189 MB |
| **2 000 translations** | **320 MB** | **−200 MB** |

Flat from iteration 50, and below the warm-up peak because the allocator returns
pages. The old binding would have accumulated roughly 48 MB of unfreed name
strings over the same tokens.

### 3.3 Long documents

AOT harness, `--long`. The previous engine's 852 MB spike is gone:

| words | chars | chunks | tokens | before | peak | after | tok/s |
|---|---|---|---|---|---|---|---|
| 100 | 882 | 2 | 246 | 421 | 538 | 538 | 154 |
| 1 000 | 7 512 | 17 | 2 091 | 510 | 527 | 407 | 165 |
| 5 000 | 37 126 | 84 | 10 332 | 407 | 407 | 330 | 159 |
| 10 000 | 73 812 | 167 | 20 541 | 330 | 358 | 333 | 164 |
| **20 000** | **147 626** | **334** | **41 082** | 335 | **360** | **291** | 158 |

Peak memory does not grow with document length, and throughput is constant from
100 to 20 000 words. Chunking already released everything per chunk; what was
missing was an engine that did not leak inside each one.

---

## 4. Runtime configuration

On device, one process, timings only (`runtime_config_test.dart`). Per
25-token sentence.

### Android — 4 cores

| config | load | first | warm | ms/tok |
|---|---|---|---|---|
| **default (cpu, 4 threads)** | 1 328 ms | 533 ms | **272.6 ms** | 10.90 |
| 4 threads (explicit) | 566 ms | 300 ms | **236.5 ms** | 9.46 |
| 2 threads | 484 ms | 296 ms | 239.6 ms | 9.58 |
| 1 thread | 725 ms | 545 ms | 323.2 ms | 12.93 |
| no pre-packing | 432 ms | 283 ms | 239.4 ms | 9.57 |
| **no arena** | 552 ms | 849 ms | **903.2 ms** | 36.13 |
| `lowMemory` (opt=none) | 555 ms | 410 ms | 378.9 ms | 15.15 |
| **xnnpack** | 638 ms | 312 ms | **235.5 ms** | 9.42 |
| **nnapi** | 706 ms | **1 787 ms** | 312.1 ms | 12.48 |
| coreml | — | — | not in the Android build | |

### iOS — 8 cores

| config | load | first | warm | ms/tok |
|---|---|---|---|---|
| **default (cpu, 4 threads)** | 605 ms | 259 ms | 169.4 ms | 6.77 |
| 4 threads (explicit) | 352 ms | 216 ms | **160.9 ms** | 6.44 |
| 2 threads | 348 ms | 252 ms | 186.2 ms | 7.45 |
| 1 thread | 363 ms | 285 ms | 261.0 ms | 10.44 |
| no pre-packing | 312 ms | 287 ms | 190.5 ms | 7.62 |
| **no arena** | 382 ms | 655 ms | **511.3 ms** | 20.45 |
| `lowMemory` (opt=none) | 326 ms | 289 ms | 249.3 ms | 9.97 |
| **xnnpack** | 438 ms | 212 ms | 162.8 ms | 6.51 |
| **coreml** | 678 ms | 221 ms | 179.3 ms | 7.17 |
| nnapi | — | — | not in the Apple build | |

Two rows repeat the same configuration ("default" and "4 threads (explicit)")
and differ by ~15 %; that is the run-to-run noise on these devices, and no
conclusion below rests on a smaller difference than that.

### What the tables say

* **Four intra-op threads** is best on both a 4-core and an 8-core device, and
  is what `threads: 0` picks. One thread is 35-60 % slower; on the laptop, eight
  threads is 40 % slower than four.
* **The memory arena must stay on.** Turning it off is **3-4× slower**
  (236 → 903 ms on Android, 161 → 511 ms on iOS). This is the single most
  damaging setting available, and the one whose name most invites switching off.
* **Pre-packing must stay on.** On the laptop, turning it off costs 129 MB
  *more* and 13 % throughput; on mobile it is throughput-neutral and still buys
  nothing. The POC's "pre-packing costs 180 MB" measurement came from ONNX
  Runtime 1.23 through Python and does not hold here; it is corrected rather
  than carried forward.
* **`lowMemory` is a thin trade**: ~20 MB for 28 % (laptop), 55 % (iOS) or 60 %
  (Android) less throughput. It exists for genuinely constrained devices, not as
  a general recommendation.

### Accelerators

Still off by default — now because they were measured, not predicted.

| | Android | iOS | macOS |
|---|---|---|---|
| **cpu** | **236 ms** | **161 ms** | **144 ms** |
| xnnpack | 236 ms | 163 ms | not in the build |
| nnapi | 312 ms + 1.8 s first run | not in the build | not in the build |
| coreml | not in the build | 179 ms | 156 ms, +46 MB |

XNNPACK is indistinguishable from plain CPU — unsurprising, since ONNX Runtime
already dispatches this graph to the same quantised GEMM kernels. NNAPI is 32 %
slower and spends 1.8 s partitioning the graph on the first inference. CoreML is
8-11 % slower. A decoder whose sequence length grows every step is the wrong
shape for all of them.

---

## 5. What still costs

* **150-200 MB resident per loaded direction.** That is the model, and the only
  ways down from here are a smaller model or a different quantisation.
* **`translateSync` blocks the calling isolate** — by design, and measured at
  1.1-1.5 s for 100 words. Respect the size guidance.
* **Cold start is 0.4-1.2 s.** Call `preload()` off the critical path.
* **First inference after load is 2-3× a warm one** (ONNX Runtime's one-off
  arena and pre-pack work). The benchmark reports it separately for that reason.
