"""
models.py – NumPy implementations of Seq2Seq attention and Transformer
             (forward-pass only; no autograd needed for analysis report).
"""

import numpy as np
import math
from typing import List, Tuple, Dict


# ═══════════════════════════════════════════════════════
# HELPER: ACTIVATION / MATRIX OPS
# ═══════════════════════════════════════════════════════

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def layer_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    sigma = x.std(axis=-1, keepdims=True)
    return (x - mu) / (sigma + eps)


def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    PE = np.zeros((seq_len, d_model))
    pos = np.arange(seq_len)[:, None]
    div = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    PE[:, 0::2] = np.sin(pos * div)
    PE[:, 1::2] = np.cos(pos * div)
    return PE


# ═══════════════════════════════════════════════════════
# SEQ2SEQ + BAHDANAU ATTENTION (demonstration forward pass)
# ═══════════════════════════════════════════════════════

class BahdanauAttention:
    """
    Additive / MLP attention (Bahdanau et al. 2015).
      score(h_t, s_{t'}) = v^T tanh(W_1 h_t + W_2 s_{t'})
    """
    def __init__(self, enc_dim: int, dec_dim: int, att_dim: int, rng):
        self.W1 = rng.standard_normal((enc_dim, att_dim)).astype(np.float32) * 0.1
        self.W2 = rng.standard_normal((dec_dim, att_dim)).astype(np.float32) * 0.1
        self.v  = rng.standard_normal((att_dim,)).astype(np.float32) * 0.1

    def forward(self, encoder_outputs: np.ndarray,
                decoder_hidden: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        encoder_outputs : (src_len, enc_dim)
        decoder_hidden  : (dec_dim,)
        Returns context (enc_dim,), attention weights (src_len,)
        """
        energy = tanh(encoder_outputs @ self.W1 + decoder_hidden @ self.W2)  # (src_len, att_dim)
        scores = energy @ self.v                                               # (src_len,)
        weights = softmax(scores)                                              # (src_len,)
        context = (weights[:, None] * encoder_outputs).sum(axis=0)            # (enc_dim,)
        return context, weights


class GRUCell:
    """Single GRU step (demo)."""
    def __init__(self, input_dim: int, hidden_dim: int, rng):
        scale = 0.1
        self.Wz = rng.standard_normal((input_dim + hidden_dim, hidden_dim)).astype(np.float32) * scale
        self.Wr = rng.standard_normal((input_dim + hidden_dim, hidden_dim)).astype(np.float32) * scale
        self.Wh = rng.standard_normal((input_dim + hidden_dim, hidden_dim)).astype(np.float32) * scale

    def step(self, x: np.ndarray, h: np.ndarray) -> np.ndarray:
        xh = np.concatenate([x, h])
        z = softmax(xh @ self.Wz)           # update gate
        r = softmax(xh @ self.Wr)           # reset gate
        h_tilde = tanh(np.concatenate([x, r * h]) @ self.Wh)
        return (1 - z) * h + z * h_tilde


class Seq2SeqAttention:
    """
    Encoder-Decoder with Bahdanau Attention.
    Architecture:
      Encoder : Bidirectional GRU (2 × hidden_dim → enc_dim via projection)
      Decoder : GRU with attention context concatenated at each step
      Output  : Linear(dec_dim + enc_dim, vocab_size)
    """
    def __init__(self, src_vocab: int, tgt_vocab: int,
                 embed_dim: int = 256, hidden_dim: int = 512,
                 att_dim: int = 256, seed: int = 0):
        rng = np.random.default_rng(seed)
        enc_dim = hidden_dim * 2  # bidirectional
        self.embed_src = rng.standard_normal((src_vocab, embed_dim)).astype(np.float32) * 0.1
        self.embed_tgt = rng.standard_normal((tgt_vocab, embed_dim)).astype(np.float32) * 0.1
        self.enc_fwd = GRUCell(embed_dim, hidden_dim, rng)
        self.enc_bwd = GRUCell(embed_dim, hidden_dim, rng)
        self.dec_rnn = GRUCell(embed_dim + enc_dim, hidden_dim, rng)
        self.attention = BahdanauAttention(enc_dim, hidden_dim, att_dim, rng)
        self.output_proj = rng.standard_normal((hidden_dim + enc_dim, tgt_vocab)).astype(np.float32) * 0.1

    def encode(self, src_ids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        embeds = self.embed_src[src_ids]  # (src_len, embed_dim)
        hidden_dim = self.enc_fwd.Wz.shape[1]
        h_fwd = np.zeros(hidden_dim, dtype=np.float32)
        h_bwd = np.zeros(hidden_dim, dtype=np.float32)
        fwd_states, bwd_states = [], []
        for e in embeds:
            h_fwd = self.enc_fwd.step(e, h_fwd)
            fwd_states.append(h_fwd.copy())
        for e in reversed(embeds):
            h_bwd = self.enc_bwd.step(e, h_bwd)
            bwd_states.append(h_bwd.copy())
        bwd_states = list(reversed(bwd_states))
        enc_out = np.concatenate(
            [np.stack(fwd_states), np.stack(bwd_states)], axis=1
        )  # (src_len, 2*hidden_dim)
        init_dec = h_fwd  # simplified; usually a learned projection
        return enc_out, init_dec

    def decode_step(self, token_id: int, dec_hidden: np.ndarray,
                    enc_out: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        emb = self.embed_tgt[token_id]
        context, att_weights = self.attention.forward(enc_out, dec_hidden)
        inp = np.concatenate([emb, context])
        dec_hidden = self.dec_rnn.step(inp, dec_hidden)
        logits = np.concatenate([dec_hidden, context]) @ self.output_proj
        return logits, dec_hidden, att_weights

    def get_attention_map(self, src_ids: List[int],
                          tgt_ids: List[int]) -> np.ndarray:
        """Return attention weight matrix (tgt_len, src_len)."""
        enc_out, dec_h = self.encode(src_ids)
        att_maps = []
        for tid in tgt_ids[:-1]:
            _, dec_h, weights = self.decode_step(tid, dec_h, enc_out)
            att_maps.append(weights)
        return np.stack(att_maps)  # (tgt_len-1, src_len)


# ═══════════════════════════════════════════════════════
# TRANSFORMER (demo forward pass)
# ═══════════════════════════════════════════════════════

class MultiHeadAttention:
    """
    Scaled dot-product multi-head attention.
      Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
    """
    def __init__(self, d_model: int, n_heads: int, rng):
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        scale = 0.1
        self.Wq = rng.standard_normal((d_model, d_model)).astype(np.float32) * scale
        self.Wk = rng.standard_normal((d_model, d_model)).astype(np.float32) * scale
        self.Wv = rng.standard_normal((d_model, d_model)).astype(np.float32) * scale
        self.Wo = rng.standard_normal((d_model, d_model)).astype(np.float32) * scale

    def forward(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Q, K, V : (seq_len, d_model)
        Returns  : (seq_len, d_model), att_weights (n_heads, seq_len, seq_len)
        """
        seq_len = Q.shape[0]
        Qs = (Q @ self.Wq).reshape(seq_len, self.n_heads, self.d_k).transpose(1, 0, 2)
        Ks = (K @ self.Wk).reshape(seq_len, self.n_heads, self.d_k).transpose(1, 0, 2)
        Vs = (V @ self.Wv).reshape(seq_len, self.n_heads, self.d_k).transpose(1, 0, 2)

        scores = Qs @ Ks.transpose(0, 2, 1) / math.sqrt(self.d_k)  # (H, T, T)
        if mask is not None:
            scores = np.where(mask, scores, -1e9)
        weights = softmax(scores, axis=-1)
        out = weights @ Vs                                            # (H, T, d_k)
        out = out.transpose(1, 0, 2).reshape(seq_len, -1) @ self.Wo  # (T, d_model)
        return out, weights


class FeedForward:
    def __init__(self, d_model: int, d_ff: int, rng):
        self.W1 = rng.standard_normal((d_model, d_ff)).astype(np.float32) * 0.1
        self.W2 = rng.standard_normal((d_ff, d_model)).astype(np.float32) * 0.1

    def forward(self, x: np.ndarray) -> np.ndarray:
        return relu(x @ self.W1) @ self.W2


class TransformerEncoderLayer:
    def __init__(self, d_model: int, n_heads: int, d_ff: int, rng):
        self.self_att = MultiHeadAttention(d_model, n_heads, rng)
        self.ff = FeedForward(d_model, d_ff, rng)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        att_out, weights = self.self_att.forward(x, x, x)
        x = layer_norm(x + att_out)
        x = layer_norm(x + self.ff.forward(x))
        return x, weights


class TransformerModel:
    """
    Vanilla Transformer (Vaswani et al. 2017) for seq2seq.
    d_model=256, n_heads=8, n_layers=3, d_ff=512 (small config for demo).
    """
    def __init__(self, src_vocab: int, tgt_vocab: int,
                 d_model: int = 256, n_heads: int = 8,
                 n_layers: int = 3, d_ff: int = 512, seed: int = 1):
        rng = np.random.default_rng(seed)
        self.d_model = d_model
        self.embed_src = rng.standard_normal((src_vocab, d_model)).astype(np.float32) * 0.1
        self.embed_tgt = rng.standard_normal((tgt_vocab, d_model)).astype(np.float32) * 0.1
        self.enc_layers = [TransformerEncoderLayer(d_model, n_heads, d_ff, rng)
                           for _ in range(n_layers)]
        self.output_proj = rng.standard_normal((d_model, tgt_vocab)).astype(np.float32) * 0.1

    def encode(self, src_ids: List[int]) -> Tuple[np.ndarray, List[np.ndarray]]:
        src_len = len(src_ids)
        x = self.embed_src[src_ids] + positional_encoding(src_len, self.d_model)
        all_weights = []
        for layer in self.enc_layers:
            x, w = layer.forward(x)
            all_weights.append(w)
        return x, all_weights

    def get_self_attention_map(self, src_ids: List[int], layer: int = 0,
                               head: int = 0) -> np.ndarray:
        """Return self-attention weights for a given layer and head."""
        _, all_weights = self.encode(src_ids)
        return all_weights[layer][head]  # (src_len, src_len)


# ═══════════════════════════════════════════════════════
# ARCHITECTURE SUMMARY
# ═══════════════════════════════════════════════════════

def architecture_params(src_vocab: int = 3000, tgt_vocab: int = 4000) -> Dict:
    """Compute approximate parameter counts for both models."""

    def seq2seq_count(embed=256, hidden=512, att=256, src_v=src_vocab, tgt_v=tgt_vocab):
        enc_dim = hidden * 2
        # Embeddings
        emb = (src_v + tgt_v) * embed
        # Encoder GRU (2 directions, 3 gates each)
        enc = 2 * 3 * (embed + hidden) * hidden
        # Decoder GRU
        dec = 3 * (embed + enc_dim + hidden) * hidden
        # Attention
        att_p = enc_dim * att + hidden * att + att
        # Output projection
        out = (hidden + enc_dim) * tgt_v
        return emb + enc + dec + att_p + out

    def transformer_count(d=256, h=8, n=3, ff=512, src_v=src_vocab, tgt_v=tgt_vocab):
        emb = (src_v + tgt_v) * d
        per_layer = (4 * d * d) + (2 * d * ff)   # MHA + FFN
        proj = d * tgt_v
        return emb + n * per_layer + proj

    return {
        "seq2seq_params": seq2seq_count(),
        "transformer_params": transformer_count(),
        "seq2seq_config": "Bi-GRU encoder (512), GRU decoder (512), Bahdanau attention (256), embed=256",
        "transformer_config": "d_model=256, n_heads=8, n_layers=3, d_ff=512, PE=sinusoidal",
    }