"""Decoder-Only Transformer for Chinese poetry generation."""

import torch
import torch.nn as nn
from models.transformer import (
    PositionalEncoding,
    TransformerBlock,
    LayerNorm,
    create_causal_mask,
)


class DecoderOnlyModel(nn.Module):
    """
    Decoder-Only architecture.

    TokenEmbedding → PositionalEncoding → TransformerBlock×N
        → LayerNorm → Linear(d_model → vocab_size)
    """

    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=6,
                 d_ff=1024, max_len=64, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.ln_head = LayerNorm(d_model)
        self.linear = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: (batch, seq_len) token indices.
            attention_mask: (batch, seq_len) — 1 for real tokens, 0 for padding.
        Returns:
            logits: (batch, seq_len, vocab_size).
        """
        batch, seq_len = input_ids.shape
        device = input_ids.device

        # Embedding
        x = self.token_embedding(input_ids) * (self.d_model ** 0.5)
        x = self.pos_encoding(x)

        # Combine causal mask + padding mask
        causal_mask = create_causal_mask(seq_len, device=device)

        if attention_mask is not None:
            # (batch, 1, 1, seq_len) for broadcasting against (1, 1, seq_len, seq_len)
            pad_mask = attention_mask.view(batch, 1, 1, seq_len)
            full_mask = causal_mask * pad_mask
        else:
            full_mask = causal_mask

        for block in self.blocks:
            x = block(x, self_mask=full_mask)

        x = self.ln_head(x)
        logits = self.linear(x)
        return logits

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=20, temperature=1.0,
                 do_sample=False, top_k=None, eos_id=3):
        """
        Autoregressive generation.

        Args:
            input_ids: (batch, seq_len) or (seq_len,) — prefix tokens.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (>0). 1.0 = no scaling.
            do_sample: If True, sample from distribution; otherwise greedy.
            top_k: If set, sample only from top-k tokens.
            eos_id: Stop token ID.

        Returns:
            (batch, full_seq_len) including input prefix.
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        batch = input_ids.size(0)
        device = input_ids.device
        generated = input_ids

        for _ in range(max_new_tokens):
            # Crop to max_len to avoid exceeding PE capacity
            if generated.size(1) > self.max_len:
                generated = generated[:, -self.max_len:]

            logits = self.forward(generated)  # (batch, seq_len, vocab_size)
            next_logits = logits[:, -1, :]    # (batch, vocab_size)

            # Apply temperature
            next_logits = next_logits / temperature

            # Top-k filtering
            if top_k is not None:
                top_k_vals, _ = torch.topk(next_logits, top_k, dim=-1)
                threshold = top_k_vals[:, -1].unsqueeze(-1)
                next_logits[next_logits < threshold] = float('-inf')

            # Sample or greedy
            if do_sample:
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)

            # Check if all sequences have hit EOS
            if (next_token == eos_id).all():
                break

        return generated

    @torch.no_grad()
    def generate_beam(self, input_ids, beam_size=5, max_new_tokens=20,
                      eos_id=3):
        """
        Beam search decoding.

        Args:
            input_ids: (seq_len,) prefix tokens.
            beam_size: Number of parallel candidates.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            (full_seq_len,) best candidate sequence.
        """
        if input_ids.dim() == 2:
            input_ids = input_ids.squeeze(0)

        device = input_ids.device
        seq = input_ids.unsqueeze(0)  # (1, seq_len)
        scores = torch.zeros(1, device=device)

        for _ in range(max_new_tokens):
            if seq.size(1) > self.max_len:
                seq = seq[:, -self.max_len:]

            logits = self.forward(seq)  # (beam, seq_len, vocab_size)
            next_logits = logits[:, -1, :]  # (beam, vocab_size)
            log_probs = torch.log_softmax(next_logits, dim=-1)

            # Accumulate scores and expand
            total_scores = scores.unsqueeze(-1) + log_probs  # (beam, vocab_size)
            total_scores = total_scores.view(-1)

            # Select top beam_size candidates
            top_scores, top_indices = total_scores.topk(beam_size)
            beam_indices = top_indices // log_probs.size(-1)
            token_indices = top_indices % log_probs.size(-1)

            # Build new sequences
            new_seqs = []
            for i in range(beam_size):
                beam_idx = beam_indices[i].item()
                token_idx = token_indices[i].item()
                candidate = torch.cat([seq[beam_idx], torch.tensor([token_idx], device=device)])
                new_seqs.append(candidate)

            seq = torch.stack(new_seqs)  # (beam, new_seq_len)
            scores = top_scores

            # Stop if all beams hit EOS
            if (seq[:, -1] == eos_id).all():
                break

        # Return the best candidate (highest score)
        best_idx = scores.argmax()
        return seq[best_idx]
