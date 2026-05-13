"""Pretrained GPT-2 Chinese Poem fine-tuning for poetry generation (方法三).

Loads uer/gpt2-chinese-poem (GPT-2 pre-trained on 800K classical poems)
and fine-tunes on the couplet-completion task.

Data format: "[CLS] 上 句 字 间 空 格 [SEP] 下 句 字 间 空 格"
"""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, BertTokenizer

MODEL_NAME = "uer/gpt2-chinese-poem"


class PretrainedPoetryModel(nn.Module):
    """Wrapper around uer/gpt2-chinese-poem for poetry line completion."""

    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, local_files_only=False):
        """Load pre-trained model and tokenizer from HuggingFace."""
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, local_files_only=local_files_only)
        model = GPT2LMHeadModel.from_pretrained(MODEL_NAME, local_files_only=local_files_only)
        return cls(model=model, tokenizer=tokenizer)

    @property
    def config(self):
        return self.model.config

    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass delegating to GPT2LMHeadModel.

        When labels is provided (with -100 for masked positions), the
        HuggingFace model internally computes CrossEntropyLoss ignoring -100.
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    # ------------------------------------------------------------------
    # Generation utilities
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, max_new_tokens=20,
                 temperature=1.0, do_sample=False, top_k=None, eos_id=None,
                 pad_id=0):
        """Greedy / top-k / temperature sampling via HuggingFace generate."""
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_id or self.tokenizer.sep_token_id,
            pad_token_id=pad_id,
            output_scores=False,
            return_dict_in_generate=False,
        )

        if do_sample:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            if top_k is not None:
                gen_kwargs["top_k"] = top_k
        else:
            gen_kwargs["do_sample"] = False
            gen_kwargs["num_beams"] = 1

        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )

    @torch.no_grad()
    def generate_beam(self, input_ids, attention_mask=None, beam_size=5,
                      max_new_tokens=20, eos_id=None, pad_id=0):
        """Beam search decoding."""
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=beam_size,
            eos_token_id=eos_id or self.tokenizer.sep_token_id,
            pad_token_id=pad_id,
            early_stopping=True,
            num_return_sequences=1,
        )[0]

    # ------------------------------------------------------------------
    # Tokenizer utilities
    # ------------------------------------------------------------------

    @staticmethod
    def format_upper(upper_text):
        """Convert upper verse to pretrained input format: '[CLS] chars [SEP]'."""
        return f"[CLS] {' '.join(list(upper_text))} [SEP]"

    @staticmethod
    def format_pair(upper_text, lower_text):
        """Format a full (upper, lower) pair for training."""
        return f"[CLS] {' '.join(list(upper_text))} [SEP] {' '.join(list(lower_text))}"
