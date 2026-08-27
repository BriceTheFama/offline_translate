# Model comparison — the size study

The POC proved the chain works. It did not prove that OPUS-MT is the right
model. This is the re-evaluation, under a much harder constraint: **the smallest
footprint that still translates EN/FR/ES/DE well.**

Everything below was measured on this project unless a number is explicitly
attributed to a publisher.

---

## 1. Where the 104 MB actually goes

Before shopping for a smaller model, it is worth knowing what the current one
is made of. `en-fr`, int8, per-tensor:

| | encoder.onnx | decoder.onnx | total | share |
|---|---:|---:|---:|---:|
| **token embedding** | 29.3 MB | 29.3 MB | **58.6 MB** | **58 %** |
| feed-forward, attention, projections | 18.0 MB | 24.5 MB | 42.5 MB | 42 % |
| layer norms, biases | 0.1 MB | 0.1 MB | 0.2 MB | 0 % |
| | 47.5 MB | 54.0 MB | 101.4 MB | |

Two things follow immediately.

**The embedding is the model.** 59 514 × 512 int8 = 29.1 MB, and it is 58 % of
the bundle. No amount of tuning the transformer layers matters next to it.

**It is stored twice.** Marian ties the source embedding, the target embedding
and the output projection to one matrix. `optimum` exports encoder and decoder
as separate ONNX files, so the same matrix is written into both — verified
bit-identical:

```console
encoder embedding: (59514, 512) uint8
decoder embedding: (59514, 512) uint8
bit-identical: True
```

That is 29 MB of pure duplication, and removing it is free.

---

## 2. What can be squeezed out of OPUS-MT

### 2.1 Sharing the tied embedding — 28 % off, lossless

ONNX external data lets tensors in different files point at the same blob. Both
graphs now reference one `embedding.data`.

```console
104.2 MB -> 75.5 MB (28% smaller)
8/8 identical token sequences
cold start 475 ms -> 289 ms
```

**It saves disk and download size, not RAM.** That was worth checking rather
than assuming: measured in fresh processes, resident memory is identical with
and without sharing — 339-350 MB either way. ONNX Runtime memory-maps the
shared blob once, but then pre-packs the weights into its own buffers on first
inference, and those copies dominate the working set. §1.5 of the brief warns
about exactly this confusion, and here it applies to our own optimisation.

`tool/dedup_embedding.py`. One subtlety cost a debugging cycle and is worth
recording: `onnx.load()` populates `raw_data` but leaves small tensors marked
`EXTERNAL` with the *source* file's offsets. Saving that silently reads the
quantisation scales from the wrong place in the new blob — the model still
runs, and quietly produces different text. The fix is to reset
`data_location` explicitly for every inlined tensor.

### 2.2 Precision — int8 is already the sweet spot

FLORES-200 devtest, greedy, the engine's own decoding loop:

| precision | size | BLEU (200 sent.) | notes |
|---|---:|---:|---|
| fp32 | 405.9 MB | 45.11 | reference |
| fp16 | 204.4 MB | — | graph is invalid as produced; fp16 also has no CPU kernels in ONNX Runtime |
| **int8** | **104.2 MB** | **44.84** | **−0.27 BLEU for 4× smaller** |
| int8 + shared embedding | **75.5 MB** | 44.84 | identical output |
| int4 (`MatMulNBits`) | 264.2 MB | — | **larger than int8** |
| int4 + int8 hybrid | 176.4 MB | — | still larger than int8 |

**int8 costs 0.27 BLEU and is essentially free.** That question is settled.

**int4 is not a size win here, for structural reasons.** ONNX Runtime's 4-bit
path is `MatMulNBits`, which only replaces `MatMul` with a constant operand.
The embedding — 58 % of the bundle — is consumed by `Gather` and is left
untouched, along with everything else the pass does not match, which stays
fp32. Hence 264 MB. Quantising the leftovers to int8 afterwards gives 176 MB,
still worse, because the merged decoder resists quantisation inside its `If`
branches (the same limitation `tool/build_model.py` works around by quantising
before merging).

A correct int4 build — int4 layers, int8 embedding, quantised before merging —
would land near **54 MB** by arithmetic (29 MB embedding + ~25 MB layers). That
is worth roughly 21 MB against the deduplicated int8 build, at a quality cost
that for a 74 M-parameter model is typically 1-3 BLEU. It was not pursued
because §3 shows a better answer for less risk.

### 2.3 Vocabulary pruning — rejected

The embedding is 58 % of the model and 86 % of its rows are never used on a
1 012-sentence corpus, which looks like an obvious win. It is not.

| keep top-K | token coverage | embedding | saving |
|---:|---:|---:|---:|
| 16 000 | 92.9 % | 7.8 MB | 21.2 MB |
| 32 000 | 98.0 % | 15.6 MB | 13.4 MB |
| 48 000 | 99.9 % | 23.4 MB | 5.6 MB |
| 59 514 | 100 % | 29.1 MB | — |

The embedding is **tied to the output projection**. Dropping rows does not just
make the tokenizer re-segment — SentencePiece would handle that fine — it
removes the model's ability to *emit* those pieces. The 2 % beyond 32 k are
ordinary French subwords (`érie`, `endant`, `uelle`, `▁envoy`), and nothing has
been trained to compose them from what remains.

**Pruning the vocabulary requires retraining.** Without it, this is not a
compression technique, it is damage.

---

## 3. Other models

### 3.1 The OPUS-MT family is one size

Every Helsinki-NLP checkpoint in scope has the same shape — `d_model` 512,
6 + 6 layers — so they all cost the same regardless of how many languages they
cover:

| model | vocab | params | int8 (dedup) | covers |
|---|---:|---:|---:|---|
| `opus-mt-en-fr` | 59 514 | 74.5 M | ~71 MB | 1 direction |
| `opus-mt-en-roa` | 56 671 | 73.1 M | ~70 MB | en → 15 Romance |
| `opus-mt-en-gmw` | 55 477 | 72.4 M | ~69 MB | en → West Germanic |
| `opus-mt-mul-en` | 64 172 | 76.9 M | ~73 MB | 100+ → en |
| `opus-mt-en-mul` | 64 110 | 76.9 M | ~73 MB | en → 100+ |

This is the single most useful fact for the multilingual question: **a model
covering 100 languages costs the same as a model covering one.** Breadth is
free; the floor is the architecture. See
[model-distribution.md](model-distribution.md) for what that implies.

### 3.2 Rejected on licence

Unchanged from the first study, and still the constraint that removes the
obvious candidates:

| model | params | license | verdict |
|---|---:|---|---|
| NLLB-200 distilled 600M | 615 M | **CC-BY-NC-4.0** | rejected — non-commercial |
| NLLB-200 distilled 1.3B | 1.3 B | **CC-BY-NC-4.0** | rejected |
| M2M-100 418M | 484 M | **CC-BY-NC-4.0** | rejected |
| MADLAD-400 3B | 3 B | Apache-2.0 | too large by two orders of magnitude |
| mT5-small | 300 M | Apache-2.0 | not a translator; 1.2 GB int8 |

### 3.3 Firefox Translations / Bergamot — the real candidate

Mozilla ships offline translation in Firefox using **distilled student models**
built for exactly this constraint. The published `en-fr` model was downloaded
and its Marian binary parsed directly:

| | OPUS-MT base | **Firefox tiny** |
|---|---:|---:|
| parameters | 74.5 M | **31.3 M** |
| `d_model` | 512 | 384 |
| vocabulary | 59 514 | **32 000** |
| encoder / decoder layers | 6 / 6 | 6 / **4** |
| decoder self-attention | yes | **none — SSRU** |
| int8 on disk | 104 MB (75.5 dedup) | **30.1 MB** |
| BLEU, FLORES devtest | **46.18** (measured, greedy) | **48.5** (published) |
| license | Apache-2.0 | **MPL-2.0** |

Its size breakdown:

| | | |
|---|---:|---:|
| embedding (tied src/tgt/output) | 11.72 MB | 39 % |
| feed-forward | 11.36 MB | 38 % |
| encoder self-attention | 3.43 MB | 11 % |
| decoder cross-attention | 2.29 MB | 8 % |
| decoder SSRU | 1.14 MB | 4 % |
| | **30.08 MB** | |

**2.5× smaller than the deduplicated OPUS-MT, and reportedly 2.3 BLEU better.**
Smaller *and* better is not a contradiction: it is distilled from a much
stronger teacher on far more data, whereas the OPUS-MT base models are 2020
checkpoints trained on OPUS alone. That is what distillation buys.

Read the 2.3 BLEU with care. Mozilla's figure almost certainly uses beam search
(Marian's default) against our greedy decoding, which is typically worth
0.5-1.5 BLEU on its own, and detokenisation conventions differ. The honest
statement is: **at 30 MB it is at least as good as our 104 MB model**, and the
size difference is not in doubt.

The decoder is the interesting part. `dec-cell: ssru` replaces decoder
self-attention with a Simpler Simple Recurrent Unit:

```text
f_t = sigmoid(W_f x_t + b_f)
c_t = f_t ⊙ c_{t-1} + (1 − f_t) ⊙ (W x_t)
h_t = relu(c_t)
```

For our engine this is *better than* a standard decoder, not merely smaller:
the self-attention KV cache disappears entirely and is replaced by one
`[1, 384]` state per layer. Memory per decoding step becomes constant instead of
growing with the output.

### 3.4 What it would cost to use

This is the catch, and it is real.

| | |
|---|---|
| Format | Marian native binary, **not** ONNX and not PyTorch |
| Weights | int8 quantised for `intgemm` with per-matrix `QuantMultA` scales |
| Architecture support | `transformers.MarianMTModel` does **not** implement SSRU, so the usual `optimum` export path cannot be used |
| Conversion route | parse the Marian binary → dequantise to fp32 → reimplement the architecture in PyTorch → `torch.onnx.export` → requantise to int8 |
| Effort | the parser already works (§3.3 numbers came from it); the PyTorch model is ~200 lines; the risk is in matching Marian's exact layer-norm placement, positional encoding and SSRU formulation |
| Licence obligation | MPL-2.0 is file-level copyleft: the converted bundles must themselves be MPL-2.0, which does **not** affect an application that merely uses them |

---

## 4. Recommendation

**Take the free 28 % now, and pursue the Firefox student models as the
production architecture.**

1. **Ship the shared embedding in the build pipeline.** Done — 104 → 75.5 MB
   per direction, bit-identical output, faster cold start, no risk and no
   licence change. It reduces download and disk only; RSS is unchanged.

2. **Convert one Firefox student model as the next POC.** `en-fr` first, so it
   can be scored against the 46.18 BLEU baseline on the same FLORES set with
   the same greedy loop. If it reproduces its published quality at 30 MB, it
   becomes the production model and OPUS-MT stays as the reference.

3. **Do not build the remaining OPUS-MT directions.** Six of the twelve are
   already built and validated; they are enough to keep as a comparison
   baseline. Building the rest would be 700 MB of work invalidated by step 2.

On the stated targets: **< 20 MB is not reachable with any published model
today.** The realistic ladder is

| | per direction | how |
|---|---:|---|
| today | 104 MB | shipped |
| free, now | **75.5 MB** | shared embedding |
| Firefox student, converted | **~30 MB** | this study's recommendation |
| + int4 on its layers | **~22 MB** | plausible, unmeasured |
| < 20 MB | — | needs training a smaller student ourselves |

30 MB per direction sits inside the "very good" band the brief defines, and 22 MB
would sit just outside "ideal". Getting under 20 MB means training, not
converting — a distillation run with its own data pipeline and GPU budget, which
is a different kind of project from this one.
