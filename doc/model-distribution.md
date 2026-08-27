# Model distribution — how many models, and how they get there

Two questions the size study forces: **how many model files does covering
EN/FR/ES/DE actually require**, and **what does an application end up
installing**. They have different answers, and the second one is the one that
matters.

---

## 1. The number that matters is not the catalogue

The current catalogue is 1.27 GB for twelve directions, which sounds
disqualifying. But nobody installs twelve directions. An application declares
the languages it supports, and that determines everything:

| the application supports | directions it needs | OPUS-MT today | OPUS-MT shared-embedding | Firefox student |
|---|---:|---:|---:|---:|
| one way (en → fr) | 1 | 104 MB | 76 MB | **30 MB** |
| two languages (en, fr) | 2 | 208 MB | 151 MB | **60 MB** |
| three languages (en, fr, es) | 6 | 625 MB | 453 MB | **180 MB** |
| four languages | 12 | 1 270 MB | 906 MB | 361 MB |

These are **download and disk** figures. Resident memory does not follow them:
a loaded direction costs 150-200 MB of RSS regardless of whether its bundle is
104 MB or 75.5 MB, because ONNX Runtime pre-packs the weights into its own
buffers. Only architecture and quantisation move RAM — see
[mobile-size-benchmark.md §3](mobile-size-benchmark.md#3-memory-and-speed).

The second column is the honest headline: **a two-language application is the
common case**, and the difference there is 208 MB versus 60 MB.

---

## 2. Options for covering four languages

### Option A — twelve specialised models

```text
en→fr  fr→en  en→es  es→en  en→de  de→en
fr→es  es→fr  fr→de  de→fr  es→de  de→es
```

| | |
|---|---|
| Size (shared embedding) | **906 MB** |
| Quality | best available — every pair is direct |
| Latency | one pass |
| Status | 12/12 built and validated |

### Option B — six English-centric models, pivot the rest

```text
                    en
                 ╱  │  ╲
               fr   es   de          fr→de  =  fr→en→de
```

| | |
|---|---|
| Size (shared embedding) | **453 MB** — half of A |
| Quality | direct for the six English pairs; the other six lose to error compounding |
| Latency | **two passes** for non-English pairs |
| Precedent | this is what **Firefox ships**: 84 directions, every one of them English-centric |

That precedent is worth weight. Mozilla runs this in production for a browser
and did not ship a single non-English pair.

### Option C — one multilingual pair (`en-mul` + `mul-en`)

| | |
|---|---|
| Size | **146 MB** for *all* twelve directions, and 100+ languages besides |
| Quality | lower per pair than a bilingual model; non-English pairs still pivot |
| Status | not measured — worth a benchmark if breadth ever becomes a requirement |

This option exists only because of the most surprising finding in the size
study: **every OPUS-MT model is the same size regardless of how many languages
it covers.** `opus-mt-mul-en` (100+ languages) is 73 MB; `opus-mt-en-fr` (one
direction) is 71 MB. The cost is the architecture — `d_model` 512, 6 + 6 layers,
~60 k vocabulary — not the language count.

So "one multilingual model" is not a compression technique either. It is a
*breadth* technique that happens to amortise a fixed cost over more pairs.

### Option D — six Firefox student models, pivot the rest

| | |
|---|---|
| Size | **180 MB** for all twelve |
| Quality | at least matching OPUS-MT on the pair measured, per §3.3 of the comparison |
| Latency | two passes for non-English pairs, but each pass is on a 4-layer decoder |
| Status | **requires the SSRU conversion** |

### Comparison

| | models | all 12 directions | typical 2-language app | direct pairs |
|---|---:|---:|---:|---:|
| A — specialised | 12 | 906 MB | 151 MB | 12 |
| B — English-centric | 6 | 453 MB | 151 MB | 6 |
| C — multilingual | 2 | **146 MB** | 146 MB | 6 |
| **D — Firefox student** | 6 | **180 MB** | **60 MB** | 6 |

**D wins on the case that matters** and is second on the catalogue total. C wins
the catalogue but charges every application 146 MB even if it only wants
English and French.

---

## 3. Embedded or downloaded

| | **A — shipped in the app** | **B — downloaded on demand** |
|---|---|---|
| Works offline immediately | yes | after one download |
| App store download | +30 to +150 MB | +32 MB (the runtime only) |
| iOS App Store cellular limit | a 4-language app would exceed it | never an issue |
| Network permission needed | **none at all** | once |
| Model updates | requires an app release | independent of releases |
| Complexity | none | already implemented and tested |

**Recommendation: B by default, A as a documented option.** `DirectoryModelSource`
already supports A for applications that must never touch the network — that is
how the iOS demo loads its models today. The download path is what
`HttpModelSource` and `ModelManager` already do, with checksum verification and
atomic installation.

The runtime itself is fixed cost either way: ONNX Runtime adds ~30 MB per
Android ABI and ~32 MB to an arm64 APK. That is the floor before any model.

---

## 4. The API this implies

The developer should declare languages, not files. Today's API asks for a
direction per call and leaves the developer to reason about which bundles exist.
The size study makes the case for inverting that.

```dart
final translator = await OfflineTranslator.initialize(
  defaultLanguage: Language.fr,
  supportedLanguages: {Language.fr, Language.en, Language.es},
);

final result = await translator.translate(
  text: 'Hello, how are you?',
  to: Language.fr,          // `from` is optional
);
```

What the package would work out on its own:

**Which models are needed.** From `supportedLanguages` and the routing strategy:

```text
{fr, en}          -> en→fr, fr→en                        2 models
{fr, en, es}      -> the four English pairs              4 models, es↔fr pivots
{fr, en, es, de}  -> the six English pairs               6 models, the rest pivot
```

**Which to install first.** `defaultLanguage` is the ranking signal: a French
application translates *into* French far more than out of it, so `en→fr` and
`es→fr` install before `fr→en` and `fr→es`. The rest can download lazily.

**How to route a pair with no direct model.** `es→fr` becomes `es→en→fr`, which
the engine can run as two chunked passes without the caller knowing. This
belongs behind `TranslationEngine`, next to the chunking that is already there.

**Which language the text is in.** With `from` omitted and a bounded
`supportedLanguages` set, a character n-gram detector over three or four
languages is a few hundred kilobytes and is reliable on anything longer than a
few words. It should stay optional and explicit — guessing wrong is worse than
asking.

None of this changes `translateSync` / `translate` / `translateStream`, the
worker isolate, the cache or the model manager. It is a resolution layer above
`ModelManager` plus a pivot strategy inside the engine.

**It should be designed after the model decision, not before.** Whether the
catalogue is English-centric (6 models) or multilingual (2) changes what
"determine the needed models" means, and building the resolver against the wrong
topology would mean writing it twice.
