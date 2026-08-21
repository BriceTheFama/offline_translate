# ONNX Runtime binding

This records the second engineering phase: replacing the third-party
`onnxruntime` Flutter plugin with in-package `dart:ffi` bindings against a
current ONNX Runtime, and what that changed.

---

## 1. Why the binding was replaced

The POC used `onnxruntime` 1.4.1 (gtbluesky). It proved the approach, and it
had four problems that could not be fixed from outside it:

| | |
|---|---|
| ONNX Runtime version | 1.15.1 (June 2023); the plugin was last published March 2024 |
| Native leak | `OrtSession.run()` allocated a C string per input and output name on **every call** and never freed them — 42 strings, ~1 KB, per generated token |
| Session configuration | no `AddSessionConfigEntry`, no arena or memory-pattern control, so half the tuning surface was unreachable |
| Android ABIs | `arm64-v8a` and `armeabi-v7a` only — no Intel emulator |

Everything above the `TranslationEngine` interface was unaffected by the
replacement, which is what that interface exists for.

---

## 2. Version audit

| | 1.29.0 | 1.28.x | 1.23.0 | 1.15.1 (old) |
|---|---|---|---|---|
| Released | 2026-08-12 | 2026-07/08 | 2026 | 2023-06 |
| Maven `onnxruntime-android` | yes | yes | yes | yes |
| CocoaPods `onnxruntime-c` | yes | yes | yes | via `-objc` |
| iOS minimum | 15.1 | 15.1 | 15.1 | 11.0 |
| macOS minimum | 14.0 | 14.0 | 13.4 | — |
| Android minimum | API 24 | API 24 | API 24 | API 21 |
| `ORT_API_VERSION` | 29 | 28 | 23 | 15 |

**Chosen: 1.29.0**, the newest stable release at the time of writing.

Availability per platform, and how this package gets it:

| Platform | Artefact | Size | How it is linked |
|---|---|---|---|
| Android arm64-v8a | `libonnxruntime.so` from the official AAR | 30.6 MB | extracted into `jniLibs`, opened with `DynamicLibrary.open` |
| Android armeabi-v7a | same | 21.7 MB | same |
| Android x86 / x86_64 | same | 36.7 MB | same — **Intel emulators now work** |
| iOS arm64 (+ simulator) | `onnxruntime-c` pod, static xcframework | 46 MB device slice | linked into the app, `DynamicLibrary.process()` |
| macOS arm64 + x86_64 | same pod | 101 MB universal archive | same |

The static archives are large; what actually ships is only the parts the
linker keeps. Licensing is unchanged: ONNX Runtime is MIT on every platform.

### 2.1 The Android AAR ships telemetry, and this package does not use it

The official `com.microsoft.onnxruntime:onnxruntime-android` AAR contributes
this to the manifest of every application that depends on it:

```xml
<uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />
<uses-permission android:name="android.permission.INTERNET" />
<provider android:name="ai.onnxruntime.TelemetryInitializer"
          android:authorities="${applicationId}.onnxruntime_telemetry_initializer"
          android:initOrder="100" />
```

`TelemetryInitializer` is a `ContentProvider`, so Android instantiates it at
application start, before any of your code runs. Decompiled, its `onCreate`
constructs an `ai.onnxruntime.telemetry.HttpClient`.

A package whose entire promise is "no network access after installation" cannot
ship that, so the AAR is **not** used as a Gradle dependency. Because this
plugin talks to ONNX Runtime purely through `dart:ffi`, none of the AAR's Java
layer is needed:

```groovy
configurations { onnxRuntimeAar }
dependencies {
    onnxRuntimeAar "com.microsoft.onnxruntime:onnxruntime-android:1.29.0@aar"
}
tasks.register('extractOnnxRuntimeLibraries', Copy) {
    from { configurations.onnxRuntimeAar.collect { zipTree(it) } }
    include 'jni/**/libonnxruntime.so'
    eachFile { it.path = it.path.replaceFirst('^jni/', '') }
    into layout.buildDirectory.dir('onnxruntime-jni')
}
```

Only `libonnxruntime.so` is extracted. No manifest merge, no permissions, no
telemetry provider, no Java classes. Verified on the built APK:

```console
$ unzip -l app-debug.apk | grep onnxruntime
 32120992  lib/arm64-v8a/libonnxruntime.so
 22727676  lib/armeabi-v7a/libonnxruntime.so
 38473056  lib/x86_64/libonnxruntime.so

$ aapt2 dump permissions app-debug.apk
uses-permission: name='android.permission.INTERNET'     # the demo app's own
                                                        # (no ACCESS_NETWORK_STATE)

$ aapt2 dump xmltree --file AndroidManifest.xml app-debug.apk | grep provider
  A: android:name="androidx.startup.InitializationProvider"   # Flutter's own
```

No `TelemetryInitializer`. For belt and braces, `OrtEnv` also calls
`DisableTelemetryEvents` when it is created.

### 2.2 Keeping the Apple static archive linked

`onnxruntime-c` vendors a **static** xcframework, and every call into it comes
from Dart at runtime. Nothing in Objective-C or Swift references it, so the
linker discarded the entire archive and `DynamicLibrary.process()` found no
symbols:

```text
Invalid argument(s): Failed to lookup symbol 'OrtGetApiBase':
dlsym(RTLD_DEFAULT, OrtGetApiBase): symbol not found
```

A C file holding a reference was not enough — the linker dropped that object
too, for the same reason. What works is that CocoaPods already puts `-ObjC` in
the application's `OTHER_LDFLAGS`, which forces the linker to load any archive
member defining an Objective-C class. So `src/ort_shim.m` defines one:

```objc
@interface OfflineTranslatorOrtKeepAlive : NSObject @end
@implementation OfflineTranslatorOrtKeepAlive
+ (const void *)apiBase { return (const void *)OrtGetApiBase(); }
@end
```

Nothing calls it. The reference is the point: `-ObjC` pulls this object in, its
reference to `OrtGetApiBase` is resolved, and ONNX Runtime comes with it.
`-force_load` on the pod also works but collides with `-ObjC` over CocoaPods'
generated dummy class, producing duplicate-symbol errors.

The pod must also be `s.static_framework = true`, or `pod install` refuses:
*"transitive dependencies that include statically linked binaries"*.

---

## 3. The bindings

Generated with `ffigen` from the pinned header, into
`lib/src/engine/native/onnx_runtime_bindings.dart`:

```sh
tool/fetch_onnxruntime.sh          # puts headers in third_party/onnxruntime
dart run ffigen --config ffigen.yaml
```

`ffigen.yaml` includes exactly two free functions (`OrtGetApiBase`,
`OrtSessionOptionsAppendExecutionProvider_CPU`) and the structs and enums they
need. `OrtApi` itself comes out in full — 425 members — and that is not
negotiable: it is a table of function pointers, and the offset of every member
is part of the ABI. A truncated struct would silently call the wrong function.
Of those 425, this package uses 38.

On top of the generated file sit four hand-written wrappers, which are the only
thing the rest of the package sees:

| file | what it owns |
|---|---|
| `onnx_runtime.dart` | library resolution per platform, API-version negotiation, `OrtEnv`, error checking |
| `onnx_runtime_allocator.dart` | the default CPU allocator and one shared `OrtMemoryInfo` |
| `onnx_runtime_tensor.dart` | owned, borrowed and *view* tensors; typed views over their buffers |
| `onnx_runtime_session.dart` | session creation and configuration, and `OrtRunPlan` |

API version negotiation walks down from 29 to 16 until `GetApi` answers, so a
host application that happens to link an older ONNX Runtime still works — the
C API table is append-only, so every offset this package uses stays valid.

### 3.1 What removed the leak

`OrtRunPlan` is the design change. It pre-encodes one call signature — input
names, input values, output names, output values — as native arrays owned by
the session:

```dart
final plan = decoder.plan(inputNames, outputNames);   // once, at load
plan.setInput('input_ids', tensor);                   // per step, a pointer store
plan.run(outputs);                                    // per step
```

A decoding step therefore performs **no allocation at all**, on either side of
the boundary:

* the 42 name strings are encoded once per session instead of once per token;
* `input_ids`, the two `use_cache_branch` flags and the 24 first-step
  placeholders are allocated in `load()` and rewritten in place;
* the encoder's `input_ids` and `attention_mask` buffers are allocated at their
  maximum width once, and a shorter input gets another `OrtValue` over the same
  memory — a pointer wrap, not a copy;
* KV tensors returned by one step are bound straight into the next, and every
  one is released before the step after that.

---

## 4. Measurements

Reference machine: Apple M-series, 8 cores, macOS 26.5, ONNX Runtime 1.29.0,
en→fr int8 bundle. Run through `tool/ffi_harness.dart`, **AOT-compiled**
(`dart compile exe`) so JIT warm-up does not pollute the memory deltas.

### 4.1 The leak is gone

`ffi_harness --leak 2000`, one 25-token sentence per iteration:

| after | RSS | delta from baseline | avg |
|---|---|---|---|
| warm-up | 520 MB | — | — |
| 10 translations (240 tokens) | 522 MB | +2 MB | 157 ms |
| 50 (1 200 tokens) | 329 MB | −191 MB | 160 ms |
| 100 (2 400 tokens) | 356 MB | −164 MB | 156 ms |
| 500 (12 000 tokens) | 324 MB | −196 MB | 154 ms |
| 1 000 (24 000 tokens) | 331 MB | −189 MB | 155 ms |
| **2 000 (48 000 tokens)** | **320 MB** | **−200 MB** | 155 ms |

Memory is flat from iteration 50 and *below* the warm-up peak, because the
allocator hands pages back. The old binding would have accumulated roughly
48 MB of unfreed name strings over the same 48 000 tokens.

### 4.2 Long documents no longer spike

`ffi_harness --long`. The previous engine peaked at 852 MB on a 20-paragraph
document on Android; this is the whole range:

| words | chars | chunks | tokens | RSS before | peak | after | seconds | tok/s |
|---|---|---|---|---|---|---|---|---|
| 100 | 882 | 2 | 246 | 421 | 538 | 538 | 1.6 | 154 |
| 500 | 3 976 | 9 | 1 107 | 538 | 542 | 510 | 6.6 | 167 |
| 1 000 | 7 512 | 17 | 2 091 | 510 | 527 | 407 | 12.7 | 165 |
| 5 000 | 37 126 | 84 | 10 332 | 407 | 407 | 330 | 64.9 | 159 |
| 10 000 | 73 812 | 167 | 20 541 | 330 | 358 | 333 | 125.5 | 164 |
| **20 000** | **147 626** | **334** | **41 082** | 335 | **360** | **291** | 260.5 | 158 |

Peak memory does not grow with document length, and throughput is constant at
~160 tok/s from 100 words to 20 000. Chunking already released everything per
chunk; what was missing was an engine that did not leak inside each one.

### 4.3 Configuration matrix

One process per configuration (`tool/bench_configs.sh`), 15 samples, median.
`rss` is the delta from process start after the first inference.

| config | load | rss | first | warm (25 tok) | ms/tok |
|---|---|---|---|---|---|
| **speed (default)** | 538 ms | **336 MB** | 199 ms | **143.9 ms** | **6.00** |
| noprepack | 275 ms | 465 MB | 215 ms | 163.0 ms | 6.79 |
| opt-none | 248 ms | 316 MB | 200 ms | 184.8 ms | 7.70 |
| opt-basic | 326 ms | 350 MB | 182 ms | 156.5 ms | 6.52 |
| noarena | 302 ms | 344 MB | 181 ms | 146.0 ms | 6.08 |
| nomempattern | 301 ms | 349 MB | 163 ms | 151.0 ms | 6.29 |
| noarena + noprepack | 270 ms | 461 MB | 226 ms | 168.6 ms | 7.02 |
| noarena + opt-none | 250 ms | 315 MB | 200 ms | 190.4 ms | 7.94 |
| everything off | 215 ms | 315 MB | 216 ms | 198.0 ms | 8.25 |
| 1 thread | 296 ms | 349 MB | 260 ms | 226.0 ms | 9.42 |
| 2 threads | 291 ms | 349 MB | 199 ms | 167.9 ms | 7.00 |
| **4 threads** | 299 ms | 350 MB | 164 ms | **146.0 ms** | **6.08** |
| 6 threads | 300 ms | 350 MB | 191 ms | 170.9 ms | 7.12 |
| 8 threads | 309 ms | 350 MB | 315 ms | 205.9 ms | 8.58 |
| CoreML | 764 ms | 382 MB | 248 ms | 156.1 ms | 6.50 |
| XNNPACK | — | — | — | — | not in the Apple build |

### 4.4 Pre-packing: the previous conclusion was wrong

The POC measured pre-packing through Python on ONNX Runtime 1.23 and concluded
it cost ~180 MB. Measured properly, on 1.29, through the actual engine:

| | RSS | ms/token |
|---|---|---|
| pre-packing **on** (default) | **336 MB** | **6.00** |
| pre-packing off | 465 MB | 6.79 |

Turning it off costs **129 MB more and 13 % throughput**. It is not a
memory/speed trade at all. The earlier number came from a different runtime
version and a different allocation pattern, and it is corrected here rather
than carried forward. `RuntimeConfig.prePackWeights` stays `true`, and its
documentation says so.

### 4.5 What is actually left to tune

Very little, which is the point: after the engine rewrite the runtime no longer
wastes memory, so the knobs have almost nothing to reclaim.

* **Graph optimisation off** is the only setting that still lowers memory —
  by about 20 MB (336 → 316), for 28 % less throughput. That is what
  `RuntimeConfig.lowMemory` now is; it was previously defined as
  `opt-none + noprepack + noarena`, which measured *worse on both axes*.
* **Arena and memory-pattern** make no difference outside noise.
* **Four intra-op threads** is the best setting on an 8-core machine, and is
  what the default picks. One thread and eight threads are both ~40 % slower.

### 4.6 Accelerators

Still off by default, now with numbers rather than expectations.

* **CoreML** works and is **8 % slower** (156 vs 144 ms) for **+46 MB** and a
  2.5× longer load. As predicted: a decoder whose sequence length grows every
  step is the wrong shape for it.
* **XNNPACK** is not compiled into Apple builds of ONNX Runtime; the Android
  build has it. Benchmarked separately in `doc/performance.md`.
* **NNAPI** is not reachable through the generic provider entry point — it has
  a dedicated `OrtSessionOptionsAppendExecutionProvider_Nnapi` that exists only
  in the Android build. The engine looks that symbol up at runtime and reports
  a clear error when it is absent.

---

## 5. `translateSync` and `translate`, after the rewrite

The synchronous API is unchanged and still genuinely synchronous: `generate()`
is a straight call chain from Dart into ONNX Runtime on the caller's isolate,
with no `Future` anywhere in it.

What changed is the asynchronous side. ONNX Runtime sessions are safe to run
from multiple threads, so `translate()` and `translateStream()` now hand their
chunks to a **worker isolate that attaches to the same native sessions**:

```text
UI isolate                          worker isolate
──────────                          ──────────────
sessions (owned) ───── addresses ──▶ sessions (adopted, non-owning)
runner  ── plans, tensors            runner ── its own plans, tensors
tokenizer                            tokenizer (its own copy)
   │                                    │
translateSync() ────────────────────────┘ (not used)
translate()    ── chunks ──────────────▶ generate
               ◀── translated chunks ───
```

The model is loaded **once**. The worker adds only its own run plans, scratch
tensors and a tokenizer copy; the 100 MB of weights are shared. If the isolate
cannot be spawned, the engine falls back to translating on the calling isolate,
which is slower to the eye but never wrong.

---

## 6. Reproducing

```sh
tool/fetch_onnxruntime.sh                     # headers + a local runtime
dart run ffigen --config ffigen.yaml          # regenerate the bindings

dart compile exe tool/ffi_harness.dart -o /tmp/ffi_harness
/tmp/ffi_harness ~/ot-models/en-fr <lib> --leak 2000
/tmp/ffi_harness ~/ot-models/en-fr <lib> --long
tool/bench_configs.sh ~/ot-models/en-fr <lib>
```

On device, through Flutter:

```sh
cd example
flutter test integration_test/benchmark_test.dart -d <device> --dart-define=OT_MODELS_DIR=...
flutter test integration_test/stability_test.dart -d <device> --dart-define=OT_MODELS_DIR=...
```

---

## 7. Remaining limitations

* **iOS 15.1 and macOS 14.0 minimums**, inherited from the `onnxruntime-c` pod.
  Older targets need an older pod (1.23.0 goes down to macOS 13.4) and a
  regenerated binding.
* **Android minimum rises to API 24** (was 21), inherited from the AAR.
* **The bindings are pinned to a header.** Moving to a newer ONNX Runtime means
  refreshing `third_party/onnxruntime/include` and re-running `ffigen`; the
  version negotiation only protects against *older* runtimes.
* **Apple builds carry no XNNPACK.** Nothing to do about it short of building
  ONNX Runtime from source.
