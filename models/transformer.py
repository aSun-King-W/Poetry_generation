"""
Transformer core components built from scratch.

Implements: LayerNorm, PositionalEncoding, MultiHeadAttention,
FeedForward, TransformerBlock (Decoder), EncoderBlock.

Uses only basic PyTorch tensor ops — no torch.nn.Transformer or
torch.nn.MultiheadAttention.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def create_causal_mask(seq_len, device=None):
    """Lower-triangular causal mask: (1, 1, seq_len, seq_len)."""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).view(
        1, 1, seq_len, seq_len
    )
    return mask


# ---------------------------------------------------------------------------
# 3.4 LayerNorm
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    """Layer normalisation with learnable scale (γ) and shift (β)."""

    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        out = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * out + self.beta


# ---------------------------------------------------------------------------
# 3.2 PositionalEncoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (no learned parameters)."""

    def __init__(self, d_model, max_len=256, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# 3.1 MultiHeadAttention
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """
    Scaled dot-product multi-head attention.

    Supports:
    - Causal masking (self-attention in decoder)
    - Cross-attention (K / V from encoder output)
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch, q_len, d_model)
            key:   (batch, kv_len, d_model)
            value: (batch, kv_len, d_model)
            mask:  broadcastable to (batch, n_heads, q_len, kv_len)
                   0 → mask out, 1 → keep
        Returns:
            (batch, q_len, d_model)
        """
        batch_size = query.size(0)

        # Linear projections → (batch, n_heads, seq_len, d_k)
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = self.dropout(torch.softmax(scores, dim=-1))
        attn_output = torch.matmul(attn_weights, V)

        # Concatenate heads: (batch, q_len, d_model)
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.d_model)
        )

        return self.W_o(attn_output)


# ---------------------------------------------------------------------------
# 3.3 FeedForward
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """Two-layer FFN with GELU: d_model → d_ff → d_model."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


# ---------------------------------------------------------------------------
# 3.5 TransformerBlock (Decoder block)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    Decoder block with Pre-Norm architecture.

    Sublayers:
      1. Masked multi-head self-attention
      2. Cross-attention over encoder output (optional)
      3. FeedForward

    Each sublayer is wrapped in Residual → Dropout → LayerNorm (Pre-Norm).
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output=None, self_mask=None, cross_mask=None):
        # ----- Sublayer 1: masked self-attention -----
        norm_x = self.norm1(x)
        attn_out = self.self_attn(norm_x, norm_x, norm_x, mask=self_mask)
        x = x + self.dropout(attn_out)

        # ----- Sublayer 2: cross-attention (optional) -----
        if encoder_output is not None:
            norm_x = self.norm2(x)
            cross_out = self.cross_attn(
                norm_x, encoder_output, encoder_output, mask=cross_mask
            )
            x = x + self.dropout(cross_out)
            norm_ffn = self.norm3
        else:
            norm_ffn = self.norm2

        # ----- Sublayer 3: feed-forward -----
        norm_x = norm_ffn(x)
        ffn_out = self.ffn(norm_x)
        x = x + self.dropout(ffn_out)
        return x


# ---------------------------------------------------------------------------
# 3.6 EncoderBlock
# ---------------------------------------------------------------------------

class EncoderBlock(nn.Module):
    """
    Encoder block with Pre-Norm architecture.

    Sublayers:
      1. Multi-head self-attention (bidirectional, no causal mask)
      2. FeedForward
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # ----- Sublayer 1: self-attention -----
        norm_x = self.norm1(x)
        attn_out = self.self_attn(norm_x, norm_x, norm_x, mask=mask)
        x = x + self.dropout(attn_out)

        # ----- Sublayer 2: feed-forward -----
        norm_x = self.norm2(x)
        ffn_out = self.ffn(norm_x)
        x = x + self.dropout(ffn_out)
        return x
