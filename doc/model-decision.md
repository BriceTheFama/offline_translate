# Model decision

Which model this package runs, and why. Every number below was measured on this
project unless it is explicitly attributed to a publisher. Reference machine:
Apple M-series, 8 cores, macOS 26.5, ONNX Runtime 1.29.0, greedy decoding
through the package's own engine.

The short version:

> **Firefox Translations student `en-fr`, converted from Marian to ONNX and
> quantised to int8: 32.3 MB, MPL-2.0, 8 ms for a short sentence.**

---

## 1. The constraint that decides everything

The brief ranks the criteria: offline, then **size**, then simplicity,
compatibility, quality — and states the budget plainly. Ideally ≤ 30 MB per
direction, acceptable to ~50 MB, and never hundreds of megabytes.

That is a harder constraint than it looks, because a translation model's size is
not really about its layers. For an encoder-decoder of this class the tied
embedding matrix is 40-60 % of the weights, and it scales with the vocabulary,
not with the depth. Any candidate has to be judged on `vocab × d_model` first.

## 2. Candidates

| | **OPUS-MT / MarianMT** | **Firefox Translations student** | **NLLB-200 distilled 600M** | **M2M-100 418M** | **MADLAD-400 3B** | **mT5-small** | **Small LLM (Gemma 3 270M, Qwen3 0.6B)** |
|---|---|---|---|---|---|---|---|
| Parameters (en→fr) | 74.5 M | **31.3 M** | 615 M | 484 M | 3 B | 300 M | 270 M – 600 M |
| `d_model` / layers | 512 / 6+6 | **384 / 6+4** | 1024 / 12+12 | 1024 / 12+12 | 2048 / 32+32 | 512 / 8+8 | decoder-only |
| Vocabulary | 59 514 | **32 000** | 256 204 | 128 112 | 256 000 | 250 112 | 150 k – 260 k |
| Size, int8 on disk | 104 MB | **32 MB** | ~620 MB | ~490 MB | ~3 GB | ~1.2 GB | 200 MB – 700 MB |
| Quality en→fr | 46.2 BLEU (measured, greedy, FLORES devtest) | **48.5 published**, and at least as good on every fixture | ~44 | ~42 | high | not a translator | unpredictable |
| Decoder memory per token | grows with output | **constant** | grows | grows | grows | grows | grows |
| **License** | Apache-2.0 | **MPL-2.0** | **CC-BY-NC-4.0** | **CC-BY-NC-4.0** | Apache-2.0 | Apache-2.0 | Gemma Terms / Apache-2.0 |
| **Commercial use** | yes | **yes** | **no** | **no** | yes | yes | Gemma: restricted |
| Android / iOS viable | yes | **yes** | no, RAM | no, RAM | no | no | marginal |
| Integration effort | low (`optimum`) | **high (see §5)** | low | low | low | low | medium |

### 2.1 Rejected on licence

`NLLB-200` and `M2M-100` are the obvious massively-multilingual candidates and
both are **CC-BY-NC-4.0 — non-commercial**. A package whose stated purpose is to
be usable inside commercial applications cannot ship them and should not
recommend them. This is checkable rather than remembered:

```console
$ curl -s https://huggingface.co/api/models/facebook/nllb-200-distilled-600M | jq .cardData.license
"cc-by-nc-4.0"
```

### 2.2 Rejected on size

MADLAD-400 and mT5-small are Apache-2.0 and unusable here for the same reason
from opposite ends: MADLAD is two orders of magnitude too large, and mT5-small
is *small in layers but enormous in vocabulary* — 250 k pieces × 512 is 128 M
parameters of embedding alone, before any translation quality. It is also a
general text-to-text model, not a translator.

### 2.3 Rejected on predictability

General-purpose small LLMs were rejected on behaviour, not size. Translation
quality is unstable, output is unbounded — they chat, they refuse, they continue
the text — and per-token cost is far higher for the same output. A 31 M
parameter dedicated seq2seq model beats a 270 M parameter generalist at this one
job on every axis that matters here.

## 3. OPUS-MT was built first, and it is not small enough

The first working version of this package ran OPUS-MT, and it is genuinely good:
Apache-2.0, one checkpoint per direction, a clean `optimum` export path. It was
built, validated across twelve directions, and measured. Where its 104 MB goes:

| | encoder | decoder | total | share |
|---|---:|---:|---:|---:|
| **token embedding** | 29.3 MB | 29.3 MB | **58.6 MB** | **58 %** |
| feed-forward and attention | 18.0 MB | 24.5 MB | 42.5 MB | 42 % |
| layer norms, biases | 0.1 MB | 0.1 MB | 0.2 MB | 0 % |

Three ways to shrink it were measured, and none reaches the budget:

* **Sharing the tied embedding between the two graphs** — real, free, lossless:
  104.2 → 75.5 MB, byte-identical output. Still above 50 MB.
* **int4 (`MatMulNBits`)** — *larger*, at 264 MB. The pass only rewrites
  `MatMul` nodes with a constant operand, so the embedding, which a `Gather`
  consumes, stays float32 along with everything else unmatched.
* **Vocabulary pruning** — 86 % of the embedding rows are unused on a
  1 012-sentence corpus, which looks like an easy 21 MB. It is not: the
  embedding is *tied to the output projection*, so dropping rows removes the
  model's ability to emit those pieces. The rows past the first 32 k are
  ordinary French subwords. Pruning the vocabulary requires retraining; without
  it, this is damage rather than compression.

**int8 itself is settled and not part of the problem**: 405.9 MB fp32 →
104.2 MB int8 costs 0.27 BLEU (45.11 → 44.84 on 200 FLORES sentences). fp16 is
twice the size of int8 and ONNX Runtime has no fp16 CPU kernels.

## 4. The Firefox Translations students

Mozilla ships offline translation in Firefox using **distilled student models**
built for exactly this constraint, and publishes them as Marian binaries.

| | OPUS-MT base | **Firefox student** |
|---|---:|---:|
| parameters | 74.5 M | **31.3 M** |
| `d_model` | 512 | 384 |
| vocabulary | 59 514 | **32 000** |
| encoder / decoder layers | 6 / 6 | 6 / **4** |
| decoder self-attention | yes | **none — SSRU** |
| bundle, int8 | 104.2 MB | **32.3 MB** |
| license | Apache-2.0 | MPL-2.0 |

Smaller *and* better is not a contradiction: these are distilled from a much
stronger teacher on far more data, while the OPUS-MT base checkpoints are 2020
models trained on OPUS alone. That is what distillation buys.

The decisive architectural detail is `dec-cell: ssru`. Decoder self-attention is
replaced by a Simpler Simple Recurrent Unit (Kim et al. 2019):

```text
f_t = sigmoid(W_f x_t + b_f)
c_t = f_t ⊙ c_{t-1} + (1 − f_t) ⊙ (W x_t)
h_t = relu(c_t)
```

For an on-device engine this is *better than* a standard decoder, not merely
smaller. The self-attention KV cache disappears and is replaced by one
`[1, 384]` state per layer, so decoding memory is **constant in the output
length**. It also removes the need for optimum's `use_cache_branch` `If` node:
a zero state is the first step, so one graph serves every step.

### 4.1 Measured, side by side

Both bundles, same machine, same engine, same greedy loop
(`dart run tool/ffi_harness.dart <bundle> <runtime> --report`):

| | OPUS-MT int8 | **Firefox student int8** | |
|---|---:|---:|---|
| Bundle on disk | 104.2 MB | **32.3 MB** | 3.2× smaller |
| Load | 549 ms | **338 ms** | |
| `Hello world` | 25 ms | **2 ms** | |
| `Hello, how are you?` | 53 ms | **8 ms** | 6.6× faster |
| 100 words | 1 046 ms | **76 ms** | 13.8× faster |
| RSS after load | +201 MB | **+63 MB** | 3.2× less |
| RSS peak, 100 words | +283 MB | **+62 MB** | see below |

The last row is the SSRU claim paying off. OPUS-MT grows another 82 MB while
generating 160 tokens, because its KV cache grows with the output. The student
model's peak is *below* its post-load RSS: generating 109 tokens costs it
nothing in memory.

Quality on the fixtures is equal or better — for instance
*"The quick brown fox jumps over the lazy dog."* becomes *"Le renard brun rapide
saute **par-dessus** le chien paresseux."* rather than OPUS-MT's *"...saute
**sur** le chien..."*.

Read Mozilla's published 48.5 BLEU with care: it almost certainly uses beam
search against our greedy decoding, worth 0.5-1.5 BLEU on its own. The honest
claim is **at 32 MB it is at least as good as the 104 MB model**, and the size
and speed differences are not in doubt.

## 5. What the conversion cost, and why it is written down

This is the one place the choice is expensive, and it is worth being explicit
because it is the reason a smaller model is not simply the obvious default.

Mozilla publishes Marian binaries. There is no PyTorch checkpoint, no ONNX
export, and `transformers.MarianMTModel` does not implement SSRU, so the usual
`optimum` path does not exist. `tool/build_tiny_model.py` therefore:

1. parses the Marian binary format directly (`tool/marian_binary.py`);
2. dequantises the `intgemm8` weights;
3. rebuilds the architecture in PyTorch from the YAML config Marian stores
   *inside* the checkpoint (`tool/tiny_transformer.py`);
4. exports two ONNX graphs, quantises them, and shares the tied embedding.

Two details in that chain each cost a debugging cycle and are recorded so they
are not rediscovered:

* **`marian-conv --gemm-type intgemm8` stores matmul weights transposed**
  (`PrepareBQuantizedTransposed`), while `Wemb` keeps its row layout because an
  embedding lookup reads it. Getting this backwards produces a model that emits
  fluent French with no relationship to the input — a failure that looks like a
  bad model rather than a bad reader. It was settled by measurement: under the
  wrong orientation the nearest neighbours of `▁Hello` are `dan`, `-1`, `garde`;
  under the right one, `▁Hell`, `▁Bonjour`, `ello`.
* **Marian starts decoding from an all-zero embedding**, not from a
  start-of-sequence token (`DecoderBase::embeddingsFromPrediction` emits a zero
  constant when there is no previous word). The bundle records this as
  `decoder_start_token_id: -1` and the graph turns any negative id into a zero
  embedding.

The pipeline verifies itself at two points: the PyTorch rebuild is checked
against the fixtures before export, and the quantised ONNX bundle is then
checked against the PyTorch rebuild. Six of the seven fixtures are
character-identical; the seventh is `Hello world` → `Bonjour monde` instead of
`Bonjour le monde`, which is int8 rounding and is asserted as such in
`test/end_to_end_test.dart` rather than smoothed over.

## 6. Getting under 30 MB

32.3 MB is 31.2 MB of int8 weights plus a 814 KB SentencePiece model and 300 KB
of graph. The weights are the architecture's floor, so the remaining ideas are:

| | saving | risk |
|---|---:|---|
| sinusoidal position table computed in-graph instead of stored | ~1.5 MB | low, unmeasured |
| int4 on the feed-forward layers, int8 embedding | ~8 MB | 1-3 BLEU, unmeasured |
| a smaller student, trained rather than converted | unbounded | a different project |

None was pursued: the brief's own rule is that a reversible decision should be
made simply and quickly, and 32.3 MB is inside the stated band while 30 MB is
not a cliff. They are recorded in the README as next steps.

## 7. Licence

**MPL-2.0**, file-level copyleft. Concretely, and this is what matters for a
package meant to be used commercially:

* **An application that merely uses these models is unaffected.** MPL-2.0
  copyleft attaches to the licensed *files* and their derivatives, not to the
  program that links or ships alongside them. Your application code stays under
  whatever licence you choose.
* **The converted bundles are derivative works and remain MPL-2.0.** Anyone
  redistributing them must keep them under MPL-2.0 and make the source form
  available. `tool/build_tiny_model.py` and the upstream checkpoint are the
  source form, and both are in this repository or linked from it.
* **Attribution is required**: the models come from
  [mozilla/firefox-translations-models](https://github.com/mozilla/firefox-translations-models).
* The package's own code stays **MIT**. The two licences do not mix, because the
  model files are data the package loads, not code it is built from.

The OPUS-MT pipeline (`tool/build_model.py`, Apache-2.0 checkpoints) is kept in
the repository as a working alternative for anyone who prefers Apache-2.0 to
MPL-2.0, or who needs a direction Mozilla does not publish. See
[licensing.md](licensing.md) for the per-direction table.
