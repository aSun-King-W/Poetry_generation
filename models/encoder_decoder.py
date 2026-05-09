"""Encoder-Decoder Transformer for Chinese poetry generation."""

import torch
import torch.nn as nn
from models.transformer import (
    PositionalEncoding,
    EncoderBlock,
    TransformerBlock,
    LayerNorm,
    create_causal_mask,
)


class EncoderDecoderModel(nn.Module):
    """
    Encoder-Decoder architecture.

    Encoder: TokenEmbedding → PositionalEncoding → EncoderBlock×N → encoded vec
    Decoder: TokenEmbedding → PositionalEncoding → TransformerBlock×N
             → LayerNorm → Linear(d_model → vocab_size)
    """

    def __init__(self, vocab_size, d_model=256, n_heads=8,
                 enc_n_layers=4, dec_n_layers=4,
                 d_ff=1024, max_len=64, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Shared embedding & positional encoding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # Encoder
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(enc_n_layers)
        ])

        # Decoder
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(dec_n_layers)
        ])

        self.ln_head = LayerNorm(d_model)
        self.linear = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _create_pad_mask(self, input_ids):
        """Padding mask: 1 for real tokens, 0 for padding.

        Returns shape (batch, 1, 1, seq_len) — broadcastable to attention scores.
        """
        return (input_ids != 0).unsqueeze(1).unsqueeze(2).float()

    # ------------------------------------------------------------------
    # Encoder
    # ------------------------------------------------------------------

    def encode(self, encoder_input_ids):
        """
        Args:
            encoder_input_ids: (batch, enc_seq_len)
        Returns:
            encoder_output: (batch, enc_seq_len, d_model)
            enc_pad_mask:   (batch, 1, 1, enc_seq_len)
        """
        # Padding mask for encoder self-attention & decoder cross-attention
        enc_pad_mask = self._create_pad_mask(encoder_input_ids)

        x = self.token_embedding(encoder_input_ids) * (self.d_model ** 0.5)
        x = self.pos_encoding(x)

        for block in self.encoder_blocks:
            x = block(x, mask=enc_pad_mask)

        return x, enc_pad_mask

    # ------------------------------------------------------------------
    # Decoder
    # ------------------------------------------------------------------

    def decode(self, decoder_input_ids, encoder_output, enc_pad_mask):
        """
        Args:
            decoder_input_ids: (batch, dec_seq_len)
            encoder_output:    (batch, enc_seq_len, d_model)
            enc_pad_mask:      (batch, 1, 1, enc_seq_len)
        Returns:
            logits: (batch, dec_seq_len, vocab_size)
        """
        batch, dec_seq_len = decoder_input_ids.shape
        device = decoder_input_ids.device

        # Combined causal + padding mask for decoder self-attention
        causal_mask = create_causal_mask(dec_seq_len, device=device)
        dec_pad_mask = self._create_pad_mask(decoder_input_ids)
        self_mask = causal_mask * dec_pad_mask

        x = self.token_embedding(decoder_input_ids) * (self.d_model ** 0.5)
        x = self.pos_encoding(x)

        for block in self.decoder_blocks:
            x = block(x, encoder_output=encoder_output,
                      self_mask=self_mask, cross_mask=enc_pad_mask)

        x = self.ln_head(x)
        logits = self.linear(x)
        return logits

    # ------------------------------------------------------------------
    # Full forward
    # ------------------------------------------------------------------

    def forward(self, encoder_input_ids, decoder_input_ids):
        """Encode then decode."""
        encoder_output, enc_pad_mask = self.encode(encoder_input_ids)
        logits = self.decode(decoder_input_ids, encoder_output, enc_pad_mask)
        return logits

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, encoder_input_ids, max_new_tokens=20, temperature=1.0,
                 do_sample=False, top_k=None, eos_id=3):
        """
        Autoregressive generation.

        Args:
            encoder_input_ids: (batch, seq_len) or (seq_len,) — upper verse.
            max_new_tokens: Max tokens to generate.
            temperature: Sampling temperature.
            do_sample: Sample (True) or greedy (False).
            top_k: Top-k filtering (None = disabled).
            eos_id: Stop token ID.

        Returns:
            (batch, full_seq_len) including BOS and generated tokens.
        """
        if encoder_input_ids.dim() == 1:
            encoder_input_ids = encoder_input_ids.unsqueeze(0)

        batch = encoder_input_ids.size(0)
        device = encoder_input_ids.device

        # Encode once — shared across all decoding steps
        encoder_output, enc_pad_mask = self.encode(encoder_input_ids)

        # Start with <BOS>
        bos_id = 2
        decoder_seq = torch.full((batch, 1), bos_id, dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            if decoder_seq.size(1) > self.max_len:
                decoder_seq = decoder_seq[:, -self.max_len:]

            logits = self.decode(decoder_seq, encoder_output, enc_pad_mask)
            next_logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k is not None:
                top_k_vals, _ = torch.topk(next_logits, top_k, dim=-1)
                threshold = top_k_vals[:, -1].unsqueeze(-1)
                next_logits[next_logits < threshold] = float('-inf')

            if do_sample:
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            decoder_seq = torch.cat([decoder_seq, next_token], dim=-1)

            if (next_token == eos_id).all():
                break

        return decoder_seq

    @torch.no_grad()
    def generate_beam(self, encoder_input_ids, beam_size=5, max_new_tokens=20,
                      eos_id=3):
        """
        Beam search decoding.

        Args:
            encoder_input_ids: (seq_len,) — upper verse.
            beam_size: Number of parallel candidates.
            max_new_tokens: Max tokens to generate.

        Returns:
            (full_seq_len,) best candidate sequence including BOS.
        """
        if encoder_input_ids.dim() == 2:
            encoder_input_ids = encoder_input_ids.squeeze(0)

        device = encoder_input_ids.device

        # Encode once, then expand for beams
        enc_out, enc_mask = self.encode(encoder_input_ids.unsqueeze(0))
        enc_out = enc_out.repeat(beam_size, 1, 1)
        enc_mask = enc_mask.repeat(beam_size, 1, 1, 1)

        bos_id = 2
        seq = torch.full((beam_size, 1), bos_id, dtype=torch.long, device=device)
        scores = torch.zeros(beam_size, device=device)

        for _ in range(max_new_tokens):
            if seq.size(1) > self.max_len:
                seq = seq[:, -self.max_len:]

            logits = self.decode(seq, enc_out, enc_mask)
            log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)

            # Accumulate scores and select top beam_size
            total_scores = scores.unsqueeze(-1) + log_probs
            total_scores = total_scores.view(-1)

            top_scores, top_indices = total_scores.topk(beam_size)
            beam_indices = top_indices // log_probs.size(-1)
            token_indices = top_indices % log_probs.size(-1)

            new_seqs = []
            for i in range(beam_size):
                beam_idx = beam_indices[i].item()
                token_idx = token_indices[i].item()
                candidate = torch.cat(
                    [seq[beam_idx], torch.tensor([token_idx], device=device)]
                )
                new_seqs.append(candidate)

            seq = torch.stack(new_seqs)
            scores = top_scores

            if (seq[:, -1] == eos_id).all():
                break

        best_idx = scores.argmax()
        return seq[best_idx]
