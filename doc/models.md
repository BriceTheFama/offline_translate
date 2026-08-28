# Models

## What a model bundle is

One bundle serves one direction, and there are two families of them. Which one a
bundle is, is recorded in its manifest as `architecture.family`; the engine reads
that and picks its decoding loop from it, so both run side by side.

### `tiny-ssru` — the default, from Firefox Translations

```text
en-fr/
├── manifest.json     3 KB    metadata, architecture constants, SHA-256 per file
├── encoder.onnx    210 KB    graph only
├── encoder.data     12 MB    int8 encoder weights, memory-mapped by ONNX Runtime
├── decoder.onnx     84 KB    graph only — one graph for every decoding step
├── decoder.data    7.2 MB    int8 decoder weights
├── embedding.data   12 MB    the tied embedding, shared by both graphs
└── source.spm      814 KB    upstream SentencePiece model, unmodified
```

**32.3 MB per direction.** Three things about that list are deliberate:

* **`embedding.data` is referenced by both graphs.** The tied matrix is 39 % of
  the model, and it is stored transposed so that the source lookup, the target
  lookup and the output projection are all served by one quantised copy. Storing
  it per graph would make the bundle 43 MB.
* **There is no `vocab.json`.** Marian uses the SentencePiece ids directly, so
  the tokenizer builds the identity mapping from the `.spm` alone — 629 KB
  saved, and one less derived artefact to audit.
* **`decoder.onnx` is 84 KB and has no `If` node.** The decoder is an SSRU, so a
  zero state is the first step; one graph serves the whole loop.

The `.spm` is byte-identical to the file Mozilla publishes, so a bundle can be
audited against the upstream checksum. The whole pipeline is reproducible: two
clean runs of `tool/build_tiny_model.py` produce byte-identical `.onnx` and
`.data` files.

```json
{
  "from": "en",
  "to": "fr",
  "version": "1.0.0",
  "base_model": "mozilla/firefox-translations-models enfr",
  "license": "MPL-2.0",
  "quantization": "int8",
  "architecture": {
    "family": "tiny-ssru",
    "decoder_layers": 4,
    "encoder_layers": 6,
    "decoder_attention_heads": 8,
    "head_dimension": 48,
    "model_dimension": 384,
    "decoder_start_token_id": -1,
    "eos_token_id": 0,
    "pad_token_id": 0,
    "max_position_embeddings": 512,
    "vocab_size": 32000
  },
  "files": [{ "name": "encoder.onnx", "size": 215197, "sha256": "…" }],
  "marian_config": "…the YAML Marian stores inside the checkpoint…"
}
```

`decoder_start_token_id: -1` is not a token. Marian starts decoding from an
all-zero embedding rather than a start-of-sequence symbol, and the exported
graph turns any negative id into one — so the engine can drive the first step
and every later step through the same input.

`marian_config` is the architecture description copied verbatim out of the
checkpoint. It is what the PyTorch rebuild was derived from, and keeping it in
the manifest means a bundle can be traced back to the shape it was built from.

### `marian` — OPUS-MT, the alternative

```text
en-fr/
├── manifest.json     1 KB
├── encoder.onnx    147 KB    graph only
├── encoder.data     47 MB    int8 encoder weights
├── decoder.onnx    518 KB    graph only (merged: first step + cached steps)
├── decoder.data     54 MB    int8 decoder weights
├── source.spm      760 KB    upstream SentencePiece model, unmodified
└── vocab.json      1.4 MB    upstream shared vocabulary, unmodified
```

**≈ 104 MB per direction**, Apache-2.0, twelve directions available. Built by
`tool/build_model.py`; its manifest omits `family`, which parses as `marian`.

In both families the architecture block is read from the checkpoint at build
time, so the engine never hard-codes a shape or a special token.

---

## Catalogue

**Every Firefox Translations student is English-paired.** There is no `fr↔es`,
`fr↔de` or `es↔de` checkpoint upstream, in any tier — Mozilla's 42 `base-memory`
directions are all `xx→en` or `en→xx`, because Firefox pivots through English
for the rest. So the six directions below are not a subset of a twelve-direction
catalogue that is half-built: they are the whole of what exists directly, and
they are exactly the six the brief names for V1.

| Direction | Size | Upstream BLEU | COMET | Upstream checkpoint | Licence |
|---|---:|---:|---:|---|---|
| English → French | 32.3 MB | 49.6 | 0.8697 | `mozilla/firefox-translations-models base-memory/enfr` | **MPL-2.0** |
| French → English | 32.3 MB | 44.3 | 0.8859 | `mozilla/firefox-translations-models base-memory/fren` | **MPL-2.0** |
| English → Spanish | 32.3 MB | 27.7 | 0.8527 | `mozilla/firefox-translations-models base-memory/enes` | **MPL-2.0** |
| Spanish → English | 32.3 MB | 27.5 | 0.8568 | `mozilla/firefox-translations-models base-memory/esen` | **MPL-2.0** |
| English → German | 32.3 MB | 40.0 | 0.8651 | `mozilla/firefox-translations-models base-memory/ende` | **MPL-2.0** |
| German → English | 32.3 MB | 41.4 | 0.8829 | `mozilla/firefox-translations-models base-memory/deen` | **MPL-2.0** |

6/6 built, 194 MB total (32.3 MB per direction). BLEU and COMET are Mozilla's published FLORES scores for the checkpoint, copied from its metadata into each manifest.

The remaining six directions of a four-language matrix (`fr↔es`, `fr↔de`,
`es↔de`) need a **pivot through English**: two passes, two resident models,
compounded errors. `OfflineTranslator` resolves a direction through
`ModelManager`, so adding a pivot strategy is a change in one class and none to
the public API — but it is a feature to design, not a file to download, and it
is not implemented.

This table is generated from the built manifests by `tool/catalogue_table.py`,
so the documented sizes, scores and licences cannot drift from what was shipped.

BLEU and COMET are **Mozilla's own FLORES scores for the checkpoint**, not
measurements of this package. They are copied out of each checkpoint's
`metadata.json` into the bundle's manifest, so an application can read them at
runtime. Read them comparatively rather than absolutely: they were almost
certainly produced with beam search against this package's greedy decoding, and
BLEU is not comparable across language pairs — `en→es` at 27.7 is not "worse"
than `en→fr` at 49.6, it is a different reference set.

### Provenance

Every checkpoint is verified against **Mozilla's published SHA-256** before it is
converted, and the rebuilt model's parameter count is checked against Mozilla's
published count. That matters more than usual here, because the upstream Git LFS
objects have been deleted from GitHub — the LFS batch API answers
`410 Object does not exist on the server` — so the bytes come from a mirror while
the metadata, and therefore the hash, still come from Mozilla. A mirror serving
anything other than the published checkpoint fails the build instead of
producing a model that translates slightly wrong. See `CHECKPOINT_MIRRORS` in
`tool/build_tiny_model.py`.

### The alternative: OPUS-MT

`tool/build_model.py` still builds Apache-2.0 OPUS-MT bundles for all twelve
directions, including the six non-English ones, at 102-120 MB each. It is the
answer for anyone who needs `fr→de` in one pass today, or who cannot accept
MPL-2.0. `en→de` is CC-BY-4.0 there and requires attribution; the rest are
Apache-2.0.

## Building a bundle

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install "optimum-onnx" transformers torch onnx onnxruntime accelerate sentencepiece

python3 tool/build_model.py --pair en-fr --out build/models   # one direction
python3 tool/build_model.py --pair all    --out build/models   # the catalogue
```

`--pair all` skips directions that are already built, keeps going when one
fails, and deletes each direction's intermediates as soon as its bundle is
written — the fp32 export alone is about 1.1 GB per direction.

What the tool does, and why each step is there:

1. **Export** with `optimum`, task `text2text-generation-with-past`. Produces
   `encoder_model.onnx`, `decoder_model.onnx`, `decoder_with_past_model.onnx`
   and `decoder_model_merged.onnx`.
   *`accelerate` must be installed* — without it the exporter skips weight
   deduplication and the merged decoder comes out at 346 MB instead of 224 MB.

2. **Quantise to int8, then merge.** Quantising `decoder_model_merged.onnx`
   directly does nothing for the weights inside its `If` branches: 224 MB in,
   214 MB out. Quantising the two unmerged decoders separately and merging
   afterwards gives **54 MB**. The order matters.

3. **Graft a `next_token` output** onto the decoder:
   `ArgMax(logits + bad_words_mask)`. The greedy pick happens inside the graph,
   so one `int64` crosses into Dart per token instead of a 59 514-wide float
   tensor. The mask reproduces `bad_words_ids: [[59513]]` from the upstream
   generation config.

4. **Save with external data.** Weights go into a companion `.data` file that
   ONNX Runtime memory-maps. Measured effect: 466 MB → 344 MB resident, with no
   change in inference speed.

5. **Copy `source.spm` and `vocab.json` unmodified**, and write
   `manifest.json` with a SHA-256 per file.

Verify the result against the reference implementation before shipping it:

```sh
python3 tool/validate_bundle.py build/models/en-fr   # one bundle
tool/validate_all.sh build/models                    # all of them, + tokenizers
```

```text
bundle en-fr v1.0.0 load=282ms
  EN: Hello, how are you?
  FR: Bonjour, comment allez-vous ?
  EN: I would like to book a table for two people at eight o'clock tonight.
  FR: Je voudrais réserver une table pour deux personnes à huit heures ce soir.
```

---

## Hosting bundles

`HttpModelSource` expects one directory per direction under a base URL:

```text
https://cdn.example.com/models/en-fr/manifest.json
https://cdn.example.com/models/en-fr/encoder.onnx
https://cdn.example.com/models/en-fr/encoder.data
…
```

Any static host works — S3, R2, GitHub Releases, a CDN. The client verifies
every file against the manifest's SHA-256 before the model is moved into place,
so the transport does not have to be trusted beyond TLS.

To ship models *inside* the app instead, put the bundles in the app's documents
directory (or an asset you unpack once) and use `DirectoryModelSource`. The
application then never needs the network at all, and on Android you can drop the
`INTERNET` permission entirely.

---

## Quality

Greedy decoding, int8 weights, one sentence through every direction — the same
source sentence per language, so the twelve outputs are comparable:

| Direction | Output |
|---|---|
| en → fr | La réunion a été reportée à mardi prochain. |
| en → es | La reunión se ha aplazado hasta el próximo martes. |
| en → de | Die Sitzung wurde auf den nächsten Dienstag verschoben. |
| fr → en | The meeting was postponed until next Tuesday. |
| fr → es | La reunión se ha aplazado hasta el martes próximo. |
| fr → de | Die Sitzung wurde auf den kommenden Dienstag verschoben. |
| es → en | The meeting has been postponed until next Tuesday. |
| es → fr | La réunion a été reportée à mardi prochain. |
| es → de | Das Treffen wurde auf den nächsten Dienstag verschoben. |
| de → en | The meeting was postponed until next Tuesday. |
| de → fr | La séance a été reportée à mardi prochain. |
| de → es | La sesión se aplazó hasta el martes siguiente. |

Sources: *"The meeting has been postponed until next Tuesday."*,
*"La réunion a été reportée à mardi prochain."*,
*"La reunión se ha aplazado hasta el próximo martes."*,
*"Die Sitzung wurde auf nächsten Dienstag verschoben."*

And a longer look at en → fr:

| English | French |
|---|---|
| Hello, how are you? | Bonjour, comment allez-vous ? |
| The quick brown fox jumps over the lazy dog. | Le renard brun rapide saute sur le chien paresseux. |
| I would like to book a table for two people at eight o'clock tonight. | Je voudrais réserver une table pour deux personnes à huit heures ce soir. |
| Artificial intelligence is transforming the way we build mobile applications. | L'intelligence artificielle transforme la façon dont nous construisons des applications mobiles. |
| She said that the meeting had been postponed until next Tuesday because of the weather. | Elle a dit que la réunion avait été reportée à mardi prochain en raison du temps. |
| The network is switched off. This sentence is translated entirely on the device. | Le réseau est éteint. Cette phrase est traduite entièrement sur l'appareil. |

Known limits of the model itself, independent of this package:

* **Sentence-level.** OPUS-MT is trained on sentence pairs and has no
  cross-sentence context, so pronouns and register can drift between the chunks
  of a long document.
* **512 positions.** Anything longer is chunked; see `TextSegmenter`.
* **No domain adaptation.** Technical jargon, code and proper nouns are handled
  as generic text.
* **Greedy decoding** occasionally picks a less fluent phrasing than beam search
  would. See [technical-decision.md](technical-decision.md) §4.2.
