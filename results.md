# Evaluation Results

Generated on 2026-05-12 during Phase 7 evaluation.

## Test Set Perplexity & BLEU-4

| Model | PPL ↓ | BLEU-4 ↑ |
|------|:-----:|:---------:|
| Decoder-Only | 79.15 | 0.0010 |
| Encoder-Decoder | 102.44 | 0.0010 |
| Pretrained (uer/gpt2-chinese-poem) | 1.49 | 0.0136 |

**Decoding strategy:** Greedy (for BLEU generation)
**BLEU samples:** 500 per model
**Notes:** Low BLEU scores are expected for poetry generation — one upper verse can have multiple valid lower verses, and BLEU only measures n-gram overlap with a single reference.

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total (upper, lower) pairs | ~130K |
| Vocabulary size | ~6500 |
| Train/Valid/Test split | 80%/10%/10% |
| Verse lengths | 5 or 7 characters |
