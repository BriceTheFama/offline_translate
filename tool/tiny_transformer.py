"""The Firefox Translations student architecture, in PyTorch, for ONNX export.

Everything here is dictated by the `special:model.yml` Marian writes into the
checkpoint:

    type: transformer          enc-depth: 6        dec-depth: 4
    dim-emb: 384               transformer-dim-ffn: 1536
    transformer-heads: 8       transformer-ffn-activation: relu
    dec-cell: ssru             transformer-decoder-autoreg: rnn
    tied-embeddings-all: true  transformer-train-position-embeddings: false
    transformer-preprocess: "" transformer-postprocess: dan
    transformer-postprocess-emb: d   transformer-postprocess-top: ""

Three of those are the ones that make this model different from OPUS-MT, and
each has a consequence for the exported graph:

* **`dec-cell: ssru`** replaces decoder self-attention with a Simpler Simple
  Recurrent Unit. There is no self-attention KV cache: the decoder's entire
  history is one `[1, 1, 384]` state per layer. Decoding memory is therefore
  constant in the output length, and the graph needs no `If` node to separate a
  "first step" from a "cached step" — a zero state *is* the first step.
* **`transformer-postprocess: dan`** is post-norm: `LayerNorm(x + sublayer(x))`.
* **`transformer-train-position-embeddings: false`** means sinusoidal position
  signals, which Marian computes with sines in the first half of the dimension
  and cosines in the second (not interleaved), and with the timescale increment
  divided by `dim/2 - 1` rather than `dim/2`.

The first decoder step consumes an **all-zero embedding** rather than a
start-of-sequence token (`DecoderBase::embeddingsFromPrediction` produces a zero
constant when there is no previous word). That is expressed here as the token id
`-1`, so the engine can drive both steps through one input.
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

EPS = 1e-9  # marian's layerNorm epsilon
START_TOKEN = -1  # our encoding of "no previous word" -> zero embedding


def sinusoidal_table(length: int, dim: int) -> torch.Tensor:
    """A port of marian `inits::sinusoidalPositionEmbeddings`."""
    half = dim // 2
    increment = math.log(10000.0) / (half - 1.0)
    position = np.arange(length, dtype=np.float64)[:, None]
    index = np.arange(half, dtype=np.float64)[None, :]
    angle = position * np.exp(index * -increment)
    table = np.concatenate([np.sin(angle), np.cos(angle)], axis=1)
    return torch.tensor(table, dtype=torch.float32)


class LayerNorm(nn.Module):
    def __init__(self, scale: np.ndarray, bias: np.ndarray):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale).reshape(-1))
        self.bias = nn.Parameter(torch.tensor(bias).reshape(-1))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        variance = ((x - mean) ** 2).mean(-1, keepdim=True)
        return (x - mean) / torch.sqrt(variance + EPS) * self.scale + self.bias


class Affine(nn.Module):
    """marian `affine(x, W, b)`: W is `[in, out]` and is not transposed."""

    def __init__(self, weight: np.ndarray, bias: np.ndarray | None = None):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(weight))
        self.bias = None if bias is None else nn.Parameter(torch.tensor(bias).reshape(-1))

    def forward(self, x):
        y = x @ self.weight
        return y if self.bias is None else y + self.bias


class Heads(nn.Module):
    def __init__(self, heads: int):
        super().__init__()
        self.heads = heads

    def split(self, x):
        batch, steps, dim = x.shape
        return x.reshape(batch, steps, self.heads, dim // self.heads).transpose(1, 2)

    def merge(self, x):
        batch, heads, steps, depth = x.shape
        return x.transpose(1, 2).reshape(batch, steps, heads * depth)


class Attention(Heads):
    def __init__(self, p: Dict[str, np.ndarray], prefix: str, heads: int):
        super().__init__(heads)
        self.q = Affine(p[f"{prefix}_Wq"], p[f"{prefix}_bq"])
        self.k = Affine(p[f"{prefix}_Wk"], p[f"{prefix}_bk"])
        self.v = Affine(p[f"{prefix}_Wv"], p[f"{prefix}_bv"])
        self.o = Affine(p[f"{prefix}_Wo"], p[f"{prefix}_bo"])

    def project_kv(self, source):
        return self.split(self.k(source)), self.split(self.v(source))

    def attend(self, query, key, value, log_mask):
        q = self.split(self.q(query))
        scale = 1.0 / math.sqrt(q.shape[-1])
        scores = (q @ key.transpose(-1, -2)) * scale + log_mask
        return self.o(self.merge(torch.softmax(scores, dim=-1) @ value))


class FeedForward(nn.Module):
    def __init__(self, p: Dict[str, np.ndarray], prefix: str):
        super().__init__()
        self.w1 = Affine(p[f"{prefix}_W1"], p[f"{prefix}_b1"])
        self.w2 = Affine(p[f"{prefix}_W2"], p[f"{prefix}_b2"])

    def forward(self, x):
        return self.w2(torch.relu(self.w1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, p, index: int, heads: int):
        super().__init__()
        self.attn = Attention(p, f"encoder_l{index}_self", heads)
        self.attn_norm = LayerNorm(
            p[f"encoder_l{index}_self_Wo_ln_scale"],
            p[f"encoder_l{index}_self_Wo_ln_bias"])
        self.ffn = FeedForward(p, f"encoder_l{index}_ffn")
        self.ffn_norm = LayerNorm(
            p[f"encoder_l{index}_ffn_ffn_ln_scale"],
            p[f"encoder_l{index}_ffn_ffn_ln_bias"])

    def forward(self, x, log_mask):
        key, value = self.attn.project_kv(x)
        x = self.attn_norm(x + self.attn.attend(x, key, value, log_mask))
        return self.ffn_norm(x + self.ffn(x))


class SSRU(nn.Module):
    """`f = sigmoid(x Wf + bf)`, `c = f*c' + (1-f)*(x W)`, `h = relu(c)`.

    Kim et al. 2019, "From Research to Production and Back". `W` has no bias.
    """

    def __init__(self, p, prefix: str):
        super().__init__()
        self.w = Affine(p[f"{prefix}_W"])
        self.wf = Affine(p[f"{prefix}_Wf"], p[f"{prefix}_bf"])

    def forward(self, x, state):
        forget = torch.sigmoid(self.wf(x))
        cell = forget * state + (1.0 - forget) * self.w(x)
        return torch.relu(cell), cell


class DecoderLayer(nn.Module):
    def __init__(self, p, index: int, heads: int):
        super().__init__()
        self.rnn = SSRU(p, f"decoder_l{index}_rnn")
        self.rnn_norm = LayerNorm(
            p[f"decoder_l{index}_rnn_ffn_ln_scale"],
            p[f"decoder_l{index}_rnn_ffn_ln_bias"])
        self.cross = Attention(p, f"decoder_l{index}_context", heads)
        self.cross_norm = LayerNorm(
            p[f"decoder_l{index}_context_Wo_ln_scale"],
            p[f"decoder_l{index}_context_Wo_ln_bias"])
        self.ffn = FeedForward(p, f"decoder_l{index}_ffn")
        self.ffn_norm = LayerNorm(
            p[f"decoder_l{index}_ffn_ffn_ln_scale"],
            p[f"decoder_l{index}_ffn_ffn_ln_bias"])

    def forward(self, x, state, cross_key, cross_value, log_mask):
        recurrent, new_state = self.rnn(x, state)
        x = self.rnn_norm(x + recurrent)
        x = self.cross_norm(
            x + self.cross.attend(x, cross_key, cross_value, log_mask))
        return self.ffn_norm(x + self.ffn(x)), new_state


class TinyModel(nn.Module):
    """Both halves in one module, so encoder and decoder share `Wemb`."""

    def __init__(self, p: Dict[str, np.ndarray], max_positions: int = 512):
        super().__init__()
        self.vocab, self.dim = p["Wemb"].shape
        self.heads = 8
        self.encoder_layers = _depth(p, "encoder_l")
        self.decoder_layers = _depth(p, "decoder_l")
        self.scale = math.sqrt(self.dim)
        self.max_positions = max_positions

        # Stored transposed, as `[dim, vocab]`, and used for three things:
        # the source embedding lookup, the target embedding lookup, and the
        # output projection (`tied-embeddings-all: true`). Keeping the single
        # transposed copy — rather than a `[vocab, dim]` table plus its
        # transpose — is what lets the whole tied matrix be quantised once and
        # then shared between the two ONNX files as one blob. It is 39 % of the
        # model, so storing it twice would put the bundle at 43 MB instead of
        # 31 MB. The lookups become `Gather(axis=1)`, a column read.
        self.embedding_t = nn.Parameter(torch.tensor(p["Wemb"]).t().contiguous())
        self.output_bias = nn.Parameter(
            torch.tensor(p["decoder_ff_logit_out_b"]).reshape(-1))
        self.register_buffer(
            "positions", sinusoidal_table(max_positions, self.dim))
        self.encoder = nn.ModuleList(
            [EncoderLayer(p, i + 1, self.heads) for i in range(self.encoder_layers)])
        self.decoder = nn.ModuleList(
            [DecoderLayer(p, i + 1, self.heads) for i in range(self.decoder_layers)])

    # -- shared ---------------------------------------------------------------
    @staticmethod
    def log_mask(attention_mask):
        """0 where a source token is real, a large negative where it is padding."""
        return (1.0 - attention_mask.to(torch.float32))[:, None, None, :] * -1.0e9

    def lookup(self, input_ids):
        """Embedding rows for `[batch, steps]` ids, read as columns."""
        batch, steps = input_ids.shape
        columns = self.embedding_t.index_select(1, input_ids.reshape(-1))
        return columns.t().reshape(batch, steps, self.dim)

    # -- encoder --------------------------------------------------------------
    def encode(self, input_ids, attention_mask):
        length = input_ids.shape[1]
        x = self.lookup(input_ids) * self.scale + self.positions[:length]
        log_mask = self.log_mask(attention_mask)
        for layer in self.encoder:
            x = layer(x, log_mask)
        return x

    def cross_kv(self, hidden) -> List[torch.Tensor]:
        out = []
        for layer in self.decoder:
            key, value = layer.cross.project_kv(hidden)
            out += [key, value]
        return out

    # -- decoder --------------------------------------------------------------
    def step(self, input_ids, position, attention_mask, cross_kv, states):
        # `input_ids == -1` means "no previous word": Marian starts decoding
        # from a zero embedding, not from a start-of-sequence token.
        present = (input_ids >= 0).to(torch.float32)[..., None]
        x = self.lookup(input_ids.clamp(min=0)) * self.scale * present
        x = x + self.positions.index_select(0, position)[None, :, :]

        log_mask = self.log_mask(attention_mask)
        new_states = []
        for i, layer in enumerate(self.decoder):
            x, state = layer(
                x, states[i], cross_kv[2 * i], cross_kv[2 * i + 1], log_mask)
            new_states.append(state)
        logits = x @ self.embedding_t + self.output_bias
        return logits, new_states


def _depth(p: Dict[str, np.ndarray], prefix: str) -> int:
    return max(int(n.split("_")[1][1:]) for n in p if n.startswith(prefix))


# -- the two exported graphs --------------------------------------------------


class EncoderGraph(nn.Module):
    """`input_ids, attention_mask` -> the decoder's cross-attention K and V.

    The encoder's hidden states are never needed outside this graph: the only
    thing the decoder does with them is project them into cross-attention keys
    and values, once. Folding those projections in here means the decoder loop
    reads them straight from an input instead of recomputing them, and the
    hidden states never cross the FFI boundary.
    """

    def __init__(self, model: TinyModel):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return tuple(self.model.cross_kv(self.model.encode(input_ids, attention_mask)))


class DecoderGraph(nn.Module):
    """One decoding step. No `If` branch: step 0 is a zero state and id -1."""

    def __init__(self, model: TinyModel):
        super().__init__()
        self.model = model

    def forward(self, input_ids, position, attention_mask, *rest):
        layers = self.model.decoder_layers
        cross_kv = list(rest[: 2 * layers])
        states = list(rest[2 * layers:])
        logits, new_states = self.model.step(
            input_ids, position, attention_mask, cross_kv, states)
        next_token = torch.argmax(logits, dim=-1)
        return (next_token, *new_states)
