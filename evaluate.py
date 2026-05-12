"""Evaluate poetry generation models: Perplexity and BLEU-4.

Usage:
    python evaluate.py                          # evaluate all three models
    python evaluate.py --models decoder-only    # evaluate one model
    python evaluate.py --bleu-samples 200       # use 200 samples for BLEU
    python evaluate.py --no-bleu                # skip BLEU (faster)
"""

import os
import json
import math
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from utils.tokenizer import load_vocab, SPECIAL_TOKENS
from generate import (
    load_decoder_only, load_encoder_decoder, load_pretrained,
    encode_decoder_only, encode_encoder_decoder,
    decode_output, decode_pretrained,
    generate_greedy, generate_beam,
)
from train import (
    PreTokenizedDataset,
    collate_decoder_only, collate_encoder_decoder, collate_pretrained,
)

try:
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Decode pre-tokenized data back to text for BLEU evaluation
# ---------------------------------------------------------------------------

def extract_pairs_decoder_only(data, id2char):
    """Extract (upper, lower) text pairs from decoder-only test data."""
    pairs = []
    for item in data:
        ids = item["input_ids"].tolist()
        upper_chars, lower_chars = [], []
        seen_sep = False
        for tid in ids:
            if tid == 0:   # PAD
                continue
            if tid == 3:   # EOS
                break
            if tid == 4:   # SEP
                seen_sep = True
                continue
            # Character token (id >= len(SPECIAL_TOKENS) = 5)
            ch = id2char.get(tid, "")
            if not ch:
                continue
            if not seen_sep:
                upper_chars.append(ch)
            else:
                lower_chars.append(ch)
        upper = "".join(upper_chars)
        lower = "".join(lower_chars)
        if upper and lower:
            pairs.append((upper, lower))
    return pairs


def extract_pairs_encdec(data, id2char):
    """Extract (upper, lower) text pairs from encoder-decoder test data."""
    pairs = []
    for item in data:
        enc_ids = item["encoder_input_ids"].tolist()
        labels = item["labels"].tolist()

        upper_chars = []
        for tid in enc_ids:
            if tid == 0:
                break   # first PAD means end of valid sequence
            if tid < len(SPECIAL_TOKENS):
                continue
            upper_chars.append(id2char.get(tid, ""))

        lower_chars = []
        for tid in labels:
            if tid == -100 or tid == 0:
                continue
            if tid == 3:   # EOS
                break
            if tid < len(SPECIAL_TOKENS):
                continue
            lower_chars.append(id2char.get(tid, ""))

        upper = "".join(upper_chars)
        lower = "".join(lower_chars)
        if upper and lower:
            pairs.append((upper, lower))
    return pairs


def extract_pairs_pretrained(data, tokenizer):
    """Extract (upper, lower) text pairs from pretrained test data."""
    pairs = []
    sep_id = tokenizer.sep_token_id
    for item in data:
        ids = item["input_ids"].tolist()
        if sep_id not in ids:
            continue
        first_sep = ids.index(sep_id)
        prefix_ids = ids[:first_sep + 1]
        lower_ids = ids[first_sep + 1:]

        upper = tokenizer.decode(prefix_ids, skip_special_tokens=True).replace(" ", "")
        lower = tokenizer.decode(lower_ids, skip_special_tokens=True).replace(" ", "")
        if upper and lower:
            pairs.append((upper, lower))
    return pairs


# ---------------------------------------------------------------------------
# Perplexity computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_ppl_decoder_only(model, test_loader, device):
    """Compute perplexity for decoder-only model on test set."""
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)  # no label smoothing for eval
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        n_tokens = (shift_labels != -100).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens
    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(avg_loss) if avg_loss < 100 else float("inf")


@torch.no_grad()
def compute_ppl_encoder_decoder(model, test_loader, device):
    """Compute perplexity for encoder-decoder model on test set."""
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in test_loader:
        enc_ids = batch["encoder_input_ids"].to(device)
        dec_ids = batch["decoder_input_ids"].to(device)
        labels = batch["labels"].to(device)
        logits = model(enc_ids, dec_ids)
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        n_tokens = (labels != -100).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens
    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(avg_loss) if avg_loss < 100 else float("inf")


@torch.no_grad()
def compute_ppl_pretrained(model, test_loader, device):
    """Compute perplexity for pretrained model on test set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss.mean()
        n_tokens = (labels != -100).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens
    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(avg_loss) if avg_loss < 100 else float("inf")


# ---------------------------------------------------------------------------
# BLEU-4 computation
# ---------------------------------------------------------------------------

def compute_bleu(model, test_pairs, model_type, id2char, char2id,
                 tokenizer, device, max_samples, decode_strategy="greedy"):
    """Generate lower verses and compute BLEU-4 against references.

    Args:
        test_pairs: List of (upper_text, lower_text) tuples.
        model_type: 'decoder-only', 'encoder-decoder', or 'pretrained'.
        max_samples: Maximum number of samples to evaluate.
        decode_strategy: 'greedy' or 'beam'.

    Returns:
        bleu_score (float) or None if nltk unavailable.
        Also prints progress.
    """
    if not NLTK_AVAILABLE:
        return None

    if max_samples is not None and max_samples <= 0:
        return None

    hypotheses = []
    references = []
    n_samples = min(len(test_pairs), max_samples) if max_samples else len(test_pairs)

    print(f"  Generating {n_samples} samples for BLEU...")
    for i in range(n_samples):
        upper, ref_lower = test_pairs[i]

        if model_type == "pretrained":
            input_ids, attn_mask = encode_pretrained(upper, tokenizer, device)
            if decode_strategy == "beam":
                seq = model.generate_beam(
                    input_ids, attention_mask=attn_mask,
                    beam_size=5, max_new_tokens=20,
                    eos_id=tokenizer.sep_token_id, pad_id=tokenizer.pad_token_id,
                )
            else:
                seq = model.generate(
                    input_ids, attention_mask=attn_mask,
                    max_new_tokens=20,
                )
                seq = seq[0] if seq.dim() > 1 else seq
            gen_lower = decode_pretrained(seq, tokenizer)
        elif model_type == "decoder-only":
            gen_input = encode_decoder_only(upper, char2id)
            if decode_strategy == "beam":
                seq = generate_beam(model, gen_input, beam_size=5, max_new_tokens=20,
                                    device=device, model_type="decoder-only")
                in_len = len(gen_input)
            else:
                seq, in_len = generate_greedy(model, gen_input, max_new_tokens=20,
                                              device=device, model_type="decoder-only")
            gen_lower = decode_output(seq, id2char, skip_first_n=in_len)
        else:  # encoder-decoder
            gen_input = encode_encoder_decoder(upper, char2id)
            if decode_strategy == "beam":
                seq = generate_beam(model, gen_input, beam_size=5, max_new_tokens=20,
                                    device=device, model_type="encoder-decoder")
                in_len = 0
            else:
                seq, in_len = generate_greedy(model, gen_input, max_new_tokens=20,
                                              device=device, model_type="encoder-decoder")
            gen_lower = decode_output(seq, id2char, skip_first_n=in_len)

        if gen_lower and ref_lower:
            hypotheses.append(list(gen_lower))
            references.append([list(ref_lower)])

        if (i + 1) % 100 == 0:
            print(f"    [{i+1}/{n_samples}]")

    if not hypotheses:
        print("  Warning: no valid generations for BLEU")
        return 0.0

    smoothie = SmoothingFunction().method1
    bleu = corpus_bleu(references, hypotheses, smoothing_function=smoothie)
    return bleu


def encode_pretrained(upper_text, tokenizer, device, max_len=64):
    """Encode upper verse for pretrained model generation."""
    text = f"[CLS] {' '.join(list(upper_text))} [SEP]"
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_tensors="pt",
    )
    return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------

def evaluate_model(model_type, args, test_data_map):
    """Evaluate a single model: PPL + BLEU."""
    print(f"\n{'='*50}")
    print(f"  Evaluating: {model_type}")
    print(f"{'='*50}")

    test_pairs = test_data_map.get(model_type, [])
    result = {"model": model_type, "ppl": float("inf"), "bleu": None}

    # ---- Load model ----
    device = torch.device(DEVICE)
    if model_type == "decoder-only":
        ckpt_path = args.decoder_ckpt or "checkpoints/decoder_only_best.pt"
        if not os.path.exists(ckpt_path):
            print(f"  Checkpoint not found: {ckpt_path}, skipping")
            return result
        model, char2id, id2char = load_decoder_only(ckpt_path, args.vocab, device)
        tokenizer = None

        # Load test data
        test_data = torch.load(
            os.path.join(args.data_dir, "test_decoder_only.pt"),
            weights_only=False,
        )
        dataset = PreTokenizedDataset(
            os.path.join(args.data_dir, "test_decoder_only.pt")
        )
        test_loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_decoder_only,
        )

        # PPL
        print("  Computing Perplexity...")
        ppl = compute_ppl_decoder_only(model, test_loader, device)
        result["ppl"] = ppl
        print(f"  Perplexity: {ppl:.2f}")

        # BLEU
        if args.compute_bleu and test_pairs:
            print("  Computing BLEU-4...")
            bleu = compute_bleu(
                model, test_pairs, "decoder-only",
                id2char, char2id, None, device,
                args.bleu_samples, args.decode,
            )
            result["bleu"] = bleu
            if bleu is not None:
                print(f"  BLEU-4: {bleu:.4f}")

    elif model_type == "encoder-decoder":
        ckpt_path = args.encdec_ckpt or "checkpoints/encoder_decoder_best.pt"
        if not os.path.exists(ckpt_path):
            print(f"  Checkpoint not found: {ckpt_path}, skipping")
            return result
        model, char2id, id2char = load_encoder_decoder(ckpt_path, args.vocab, device)
        tokenizer = None

        test_loader = DataLoader(
            PreTokenizedDataset(
                os.path.join(args.data_dir, "test_encoder_decoder.pt")
            ),
            batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_encoder_decoder,
        )

        # PPL
        print("  Computing Perplexity...")
        ppl = compute_ppl_encoder_decoder(model, test_loader, device)
        result["ppl"] = ppl
        print(f"  Perplexity: {ppl:.2f}")

        # BLEU
        if args.compute_bleu and test_pairs:
            print("  Computing BLEU-4...")
            bleu = compute_bleu(
                model, test_pairs, "encoder-decoder",
                id2char, char2id, None, device,
                args.bleu_samples, args.decode,
            )
            result["bleu"] = bleu
            if bleu is not None:
                print(f"  BLEU-4: {bleu:.4f}")

    else:  # pretrained
        ckpt_path = args.pretrained_ckpt or "checkpoints/pretrained_best.pt"
        if not os.path.exists(ckpt_path):
            print(f"  Checkpoint not found: {ckpt_path}, skipping")
            return result
        model = load_pretrained(ckpt_path, device)
        tokenizer = model.tokenizer
        char2id, id2char = None, None

        test_loader = DataLoader(
            PreTokenizedDataset(
                os.path.join(args.data_dir, "test_pretrained.pt")
            ),
            batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_pretrained,
        )

        # PPL
        print("  Computing Perplexity...")
        ppl = compute_ppl_pretrained(model, test_loader, device)
        result["ppl"] = ppl
        print(f"  Perplexity: {ppl:.2f}")

        # BLEU
        if args.compute_bleu and test_pairs:
            print("  Computing BLEU-4...")
            bleu = compute_bleu(
                model, test_pairs, "pretrained",
                None, None, tokenizer, device,
                args.bleu_samples, args.decode,
            )
            result["bleu"] = bleu
            if bleu is not None:
                print(f"  BLEU-4: {bleu:.4f}")

    return result


def print_results(results):
    """Print evaluation results table."""
    print("\n\n" + "=" * 48)
    print("           Evaluation Results")
    print("=" * 48)
    print(f"{'Model':<20} {'PPL':<12} {'BLEU-4':<12}")
    print("-" * 48)
    for r in results:
        ppl_str = f"{r['ppl']:.2f}" if r['ppl'] != float("inf") else "N/A"
        bleu_str = f"{r['bleu']:.4f}" if r['bleu'] is not None else "N/A"
        print(f"{r['model']:<20} {ppl_str:<12} {bleu_str:<12}")
    print("=" * 48)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate poetry generation models")
    parser.add_argument("--models", nargs="+",
                        choices=["decoder-only", "encoder-decoder", "pretrained"],
                        default=["decoder-only", "encoder-decoder", "pretrained"],
                        help="Models to evaluate (default: all three)")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--vocab", default="data/vocab.json")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size for PPL computation")
    parser.add_argument("--decoder-ckpt", default=None,
                        help="Decoder-Only checkpoint path (default: checkpoints/decoder_only_best.pt)")
    parser.add_argument("--encdec-ckpt", default=None,
                        help="Encoder-Decoder checkpoint path (default: checkpoints/encoder_decoder_best.pt)")
    parser.add_argument("--pretrained-ckpt", default=None,
                        help="Pretrained checkpoint path (default: checkpoints/pretrained_best.pt)")
    parser.add_argument("--compute-bleu", action="store_true", default=True,
                        help="Compute BLEU-4 (default: True)")
    parser.add_argument("--no-bleu", action="store_false", dest="compute_bleu",
                        help="Skip BLEU-4 computation")
    parser.add_argument("--bleu-samples", type=int, default=500,
                        help="Number of test samples for BLEU (default: 500)")
    parser.add_argument("--decode", choices=["greedy", "beam"], default="greedy",
                        help="Decoding strategy for BLEU generation (default: greedy)")

    args = parser.parse_args()

    print(f"Device: {DEVICE}")

    # Check that at least one checkpoint exists
    all_ckpts_exist = True
    for model_type in args.models:
        ckpt_map = {
            "decoder-only": args.decoder_ckpt or "checkpoints/decoder_only_best.pt",
            "encoder-decoder": args.encdec_ckpt or "checkpoints/encoder_decoder_best.pt",
            "pretrained": args.pretrained_ckpt or "checkpoints/pretrained_best.pt",
        }
        path = ckpt_map[model_type]
        if not os.path.exists(path):
            print(f"Warning: checkpoint not found: {path}")
            all_ckpts_exist = False

    if not all_ckpts_exist:
        print("\nSome checkpoints are missing. Train models first:")
        print("  python train.py decoder-only")
        print("  python train.py encoder-decoder")
        print("  python train.py pretrained")

    # ---- Pre-extract test pairs for BLEU (avoids re-extracting per model) ----
    test_data_map = {}
    if args.compute_bleu and NLTK_AVAILABLE:
        print("\nPreparing test pairs for BLEU evaluation...")
        for model_type in args.models:
            if model_type == "pretrained":
                pt_data = torch.load(
                    os.path.join(args.data_dir, "test_pretrained.pt"),
                    weights_only=False,
                )
                # Only need tokenizer for extraction, not the full model
                tokenizer = BertTokenizer.from_pretrained(
                    "uer/gpt2-chinese-poem", local_files_only=True,
                )
                pairs = extract_pairs_pretrained(pt_data, tokenizer)
            elif model_type == "decoder-only":
                data = torch.load(
                    os.path.join(args.data_dir, "test_decoder_only.pt"),
                    weights_only=False,
                )
                _, id2char = load_vocab(args.vocab)
                pairs = extract_pairs_decoder_only(data, id2char)
            else:  # encoder-decoder
                data = torch.load(
                    os.path.join(args.data_dir, "test_encoder_decoder.pt"),
                    weights_only=False,
                )
                _, id2char = load_vocab(args.vocab)
                pairs = extract_pairs_encdec(data, id2char)
            test_data_map[model_type] = pairs
            print(f"  {model_type}: {len(pairs)} test pairs")

    # ---- Evaluate each model ----
    results = []
    for model_type in args.models:
        result = evaluate_model(model_type, args, test_data_map)
        results.append(result)

    # ---- Print summary ----
    print_results(results)


if __name__ == "__main__":
    main()
