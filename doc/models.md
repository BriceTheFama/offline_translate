# Models

## What a model bundle is

One bundle serves one direction. It is a directory of six files:

```text
en-fr/
├── manifest.json     1 KB    metadata, architecture constants, SHA-256 per file
├── encoder.onnx    147 KB    graph only
├── encoder.data     47 MB    int8 encoder weights, memory-mapped by ONNX Runtime
├── decoder.onnx    518 KB    graph only (merged: first step + cached steps)
├── decoder.data     54 MB    int8 decoder weights
├── source.spm      760 KB    upstream SentencePiece model, unmodified
└── vocab.json      1.4 MB    upstream shared vocabulary, unmodified
```

**≈ 104 MB per direction.** The `.spm` and `.json` files are byte-identical to
the ones published by Helsinki-NLP, so a bundle can be audited against the
upstream checksums.

`manifest.json` looks like this:

```json
{
  "from": "en",
  "to": "fr",
  "version": "1.0.0",
  "checksum": "05c90e08…",
  "base_model": "Helsinki-NLP/opus-mt-en-fr",
  "license": "Apache-2.0",
  "quantization": "int8",
  "architecture": {
    "decoder_layers": 6,
    "decoder_attention_heads": 8,
    "head_dimension": 64,
    "decoder_start_token_id": 59513,
    "eos_token_id": 0,
    "pad_token_id": 59513,
    "max_position_embeddings": 512,
    "vocab_size": 59514
  },
  "files": [{ "name": "encoder.onnx", "size": 150236, "sha256": "e23400b8…" }]
}
```

The architecture block is read from the upstream `config.json` at build time, so
the engine never hard-codes a shape or a special token. Pointing the build tool
at a different Marian checkpoint is enough to support it.

---

## Catalogue

All twelve V1 directions exist upstream. Licenses were checked **individually**,
because they are not uniform:

| Direction | Size | Vocab | Upstream checkpoint | License | Commercial use |
|---|---:|---:|---|---|---|
| en → fr | 104.2 MB | 59 514 | `Helsinki-NLP/opus-mt-en-fr` | Apache-2.0 | yes |
| fr → en | 104.3 MB | 59 514 | `Helsinki-NLP/opus-mt-fr-en` | Apache-2.0 | yes |
| en → es | 109.9 MB | 65 001 | `Helsinki-NLP/opus-mt-en-es` | Apache-2.0 | yes |
| es → en | 109.9 MB | 65 001 | `Helsinki-NLP/opus-mt-es-en` | Apache-2.0 | yes |
| **en → de** | 102.8 MB | 58 101 | `Helsinki-NLP/opus-mt-en-de` | **CC-BY-4.0** | yes, **with attribution** |
| de → en | 102.8 MB | 58 101 | `Helsinki-NLP/opus-mt-de-en` | Apache-2.0 | yes |
| fr → es | 120.3 MB | 74 822 | `Helsinki-NLP/opus-mt-fr-es` | Apache-2.0 | yes |
| es → fr | 120.3 MB | 74 822 | `Helsinki-NLP/opus-mt-es-fr` | Apache-2.0 | yes |
| fr → de | 106.0 MB | 61 153 | `Helsinki-NLP/opus-mt-fr-de` | Apache-2.0 | yes |
| de → fr | 106.0 MB | 61 153 | `Helsinki-NLP/opus-mt-de-fr` | Apache-2.0 | yes |
| es → de | 106.2 MB | 61 301 | `Helsinki-NLP/opus-mt-es-de` | Apache-2.0 | yes |
| de → es | 106.2 MB | 61 301 | `Helsinki-NLP/opus-mt-de-es` | Apache-2.0 | yes |

**All twelve are built and verified.** 1.27 GB for the whole catalogue,
108 MB average per direction — but the number that matters is the 104-120 MB a
user pays for the one or two directions their application actually installs.

This table is generated from the built manifests by
`tool/catalogue_table.py`, so the documented sizes, vocabularies and licences
cannot drift from what was shipped.

`en-de` is the odd one out: CC-BY-4.0 permits commercial use but **requires
attribution**. Its `manifest.json` records that, so an application can surface
the right notice per installed model. See [licensing.md](licensing.md).

### Why every direction is verified separately

It would be easy to assume that a pipeline proven on `en→fr` is proven
everywhere. It is not. Each OPUS-MT checkpoint ships **its own** SentencePiece
model and shared vocabulary, and they differ:

| | en↔fr | en↔de | en↔es | fr↔de, de↔fr | es↔de, de↔es | fr↔es |
|---|---|---|---|---|---|---|
| vocabulary size | 59 514 | 58 101 | 65 001 | 61 153 | 61 301 | 74 822 |
| pad / decoder-start id | 59 513 | 58 100 | 65 000 | 61 152 | 61 300 | 74 821 |
| bundle size | 104 MB | 103 MB | 110 MB | 106 MB | 106 MB | 120 MB |

Nothing in the engine hard-codes any of that — the architecture constants are
read from the checkpoint's `config.json` at build time and frozen into the
manifest — but "nothing hard-codes it" is a claim that has to be checked, not
asserted. So every direction goes through the same gate:

```sh
tool/validate_all.sh ~/ot-models
```

which, per direction:

1. re-hashes every file against the manifest (the same check the on-device
   `ModelManager` performs);
2. loads both graphs, confirms the grafted `next_token` output is present, and
   runs the **engine's own decoding protocol** — not `transformers.generate` —
   against real source-language sentences;
3. generates ~1 820 tokenizer vectors from `transformers.MarianTokenizer` for
   *that* checkpoint and checks the Dart tokenizer against every one.

Result across the catalogue: **12 of 12 bundles verified, 21 878 tokenizer
vectors matched exactly**, across five distinct vocabularies. The pure-Dart
tokenizer generalises to every checkpoint without a special case.

---

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
