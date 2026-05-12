"""Inference: generate lower verse lines from upper lines."""

import os
import json
import argparse
import torch

from models.decoder_only import DecoderOnlyModel
from models.encoder_decoder import EncoderDecoderModel
from models.pretrained import PretrainedPoetryModel
from utils.tokenizer import SPECIAL_TOKENS


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_decoder_only(checkpoint_path, vocab_path, device):
    """Load a trained DecoderOnlyModel from checkpoint."""
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
    char2id = vocab_data["char2id"]
    id2char = {int(k): v for k, v in vocab_data["id2char"].items()}
    vocab_size = len(char2id)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args_dict = checkpoint.get("args", {})

    model = DecoderOnlyModel(
        vocab_size=vocab_size,
        d_model=args_dict.get("d_model", 256),
        n_heads=args_dict.get("n_heads", 8),
        n_layers=args_dict.get("n_layers", 6),
        d_ff=args_dict.get("d_ff", 1024),
        max_len=args_dict.get("max_len", 32),
        dropout=args_dict.get("dropout", 0.1),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, char2id, id2char


def load_encoder_decoder(checkpoint_path, vocab_path, device):
    """Load a trained EncoderDecoderModel from checkpoint."""
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
    char2id = vocab_data["char2id"]
    id2char = {int(k): v for k, v in vocab_data["id2char"].items()}
    vocab_size = len(char2id)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args_dict = checkpoint.get("args", {})

    model = EncoderDecoderModel(
        vocab_size=vocab_size,
        d_model=args_dict.get("d_model", 256),
        n_heads=args_dict.get("n_heads", 8),
        enc_n_layers=args_dict.get("enc_n_layers", 4),
        dec_n_layers=args_dict.get("dec_n_layers", 4),
        d_ff=args_dict.get("d_ff", 1024),
        max_len=args_dict.get("max_len", 32),
        dropout=args_dict.get("dropout", 0.1),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, char2id, id2char


def load_pretrained(checkpoint_path, device):
    """Load a fine-tuned PretrainedPoetryModel from checkpoint."""
    pretrained = PretrainedPoetryModel.from_pretrained(local_files_only=True)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Clean state dict keys (strip 'module.' prefix if saved from DataParallel)
    state_dict = checkpoint["model_state_dict"]
    from collections import OrderedDict
    cleaned = OrderedDict()
    for k, v in state_dict.items():
        cleaned[k.replace("module.", "")] = v

    # strict=False ignores attn.bias/masked_bias buffers that differ across
    # PyTorch/Transformers versions — these are causal-mask constants, not learned
    pretrained.model.load_state_dict(cleaned, strict=False)
    pretrained.to(device)
    pretrained.eval()
    return pretrained


# ---------------------------------------------------------------------------
# Input encoding
# ---------------------------------------------------------------------------

def encode_decoder_only(upper_text, char2id, sep_id=4, max_len=32):
    """Decoder-only: [上句..., SEP]."""
    ids = [char2id.get(c, 1) for c in upper_text]
    ids = ids[:max_len - 1]
    ids.append(sep_id)
    return torch.tensor(ids, dtype=torch.long)


def encode_encoder_decoder(upper_text, char2id, max_len=32):
    """Encoder-Decoder: [上句...] (no SEP)."""
    ids = [char2id.get(c, 1) for c in upper_text]
    ids = ids[:max_len]
    return torch.tensor(ids, dtype=torch.long)


def encode_pretrained(upper_text, tokenizer, device, max_len=64):
    """Pretrained: '[CLS] u p p e r [SEP]' → input_ids, attention_mask.

    Adds spaces between characters as required by the BertTokenizer.
    No padding — the unpadded input keeps position encoding consistent
    with training (where right-padding was used).
    """
    text = f"[CLS] {' '.join(list(upper_text))} [SEP]"
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_tensors="pt",
    )
    return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)


def decode_pretrained(token_ids, tokenizer):
    """Decode pretrained model output, extracting only the generated part.

    Takes tokens after the last [SEP] (the lower verse the model generated).
    Strips inter-character spaces added by the BertTokenizer.
    """
    ids = token_ids.tolist() if torch.is_tensor(token_ids) else list(token_ids)
    # Find [SEP] position and take only the generated tokens after it
    sep_id = tokenizer.sep_token_id
    if sep_id in ids:
        last_sep = len(ids) - 1 - ids[::-1].index(sep_id)
        ids = ids[last_sep + 1:]
    text = tokenizer.decode(ids, skip_special_tokens=True)
    return text.replace(" ", "")


# ---------------------------------------------------------------------------
# Output decoding
# ---------------------------------------------------------------------------

def decode_output(token_ids, id2char, eos_id=3, pad_id=0, sep_id=4,
                  skip_first_n=0):
    """Decode token IDs to text, stopping at EOS."""
    chars = []
    for i, tid in enumerate(token_ids):
        if i < skip_first_n:
            continue
        tid = tid.item() if torch.is_tensor(tid) else tid
        if tid == eos_id:
            break
        if tid in (pad_id, sep_id):
            continue
        if tid < len(SPECIAL_TOKENS):
            continue
        chars.append(id2char.get(tid, "�"))
    return "".join(chars)


# ---------------------------------------------------------------------------
# Generation wrappers
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_greedy(model, input_ids, max_new_tokens=20, eos_id=3, device="cpu",
                    model_type="decoder-only"):
    """Greedy decoding."""
    input_ids = input_ids.to(device)
    if model_type == "decoder-only":
        seq = model.generate(input_ids, max_new_tokens=max_new_tokens,
                             do_sample=False, eos_id=eos_id)
        return seq[0], len(input_ids)
    else:
        seq = model.generate(input_ids, max_new_tokens=max_new_tokens,
                             do_sample=False, eos_id=eos_id)
        return seq[0], 0  # BOS handled internally, no skip


@torch.no_grad()
def generate_beam(model, input_ids, beam_size=5, max_new_tokens=20,
                  eos_id=3, device="cpu", model_type="decoder-only"):
    """Beam search decoding."""
    input_ids = input_ids.to(device)
    if model_type == "decoder-only":
        seq = model.generate_beam(input_ids, beam_size=beam_size,
                                  max_new_tokens=max_new_tokens, eos_id=eos_id)
        return seq, len(input_ids)
    else:
        seq = model.generate_beam(input_ids, beam_size=beam_size,
                                  max_new_tokens=max_new_tokens, eos_id=eos_id)
        return seq, 0


@torch.no_grad()
def generate_sample(model, input_ids, temperature=0.8, top_k=None,
                    max_new_tokens=20, eos_id=3, device="cpu",
                    model_type="decoder-only"):
    """Temperature sampling."""
    input_ids = input_ids.to(device)
    if model_type == "decoder-only":
        seq = model.generate(input_ids, max_new_tokens=max_new_tokens,
                             temperature=temperature, do_sample=True,
                             top_k=top_k, eos_id=eos_id)
        return seq[0], len(input_ids)
    else:
        seq = model.generate(input_ids, max_new_tokens=max_new_tokens,
                             temperature=temperature, do_sample=True,
                             top_k=top_k, eos_id=eos_id)
        return seq[0], 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate poetry with trained model")
    parser.add_argument("--model", choices=["decoder-only", "encoder-decoder", "pretrained"],
                        default="decoder-only", help="Model architecture")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path (auto-determined by model type if not set)")
    parser.add_argument("--vocab", default="data/vocab.json")
    parser.add_argument("--upper", type=str, default=None,
                        help="Upper verse line (上句). If not given, runs demo.")
    parser.add_argument("--method", choices=["greedy", "beam", "sample"],
                        default="greedy", help="Decoding strategy")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-new", type=int, default=20)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Determine checkpoint path
    if args.checkpoint is None:
        ckpt_map = {
            "decoder-only": "decoder_only_best.pt",
            "encoder-decoder": "encoder_decoder_best.pt",
            "pretrained": "pretrained_best.pt",
        }
        args.checkpoint = os.path.join("checkpoints", ckpt_map[args.model])

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        print(f"Train a model first: python train.py {args.model}")
        return

    # Load model
    if args.model == "pretrained":
        model = load_pretrained(args.checkpoint, device)
        tokenizer = model.tokenizer
    elif args.model == "decoder-only":
        model, char2id, id2char = load_decoder_only(args.checkpoint, args.vocab, device)
    else:
        model, char2id, id2char = load_encoder_decoder(args.checkpoint, args.vocab, device)
    print(f"Model loaded from {args.checkpoint}")

    # Demo inputs if no upper given
    demo_uppers = [
        "床前明月光",
        "举头望明月",
        "白日依山尽",
        "春眠不觉晓",
        "海内存知己",
    ]

    uppers = [args.upper] if args.upper else demo_uppers

    for upper in uppers:
        if args.model == "pretrained":
            # Pretrained uses HuggingFace generate API with its own tokenizer
            input_ids, attention_mask = encode_pretrained(upper, tokenizer, device)

            # Build kwargs based on decoding strategy
            if args.method == "beam":
                output_ids = model.generate_beam(
                    input_ids, attention_mask=attention_mask,
                    beam_size=args.beam_size, max_new_tokens=args.max_new,
                    eos_id=tokenizer.sep_token_id, pad_id=tokenizer.pad_token_id,
                )
            else:
                output_ids = model.generate(
                    input_ids, attention_mask=attention_mask,
                    max_new_tokens=args.max_new,
                    temperature=args.temperature,
                    do_sample=(args.method == "sample"),
                    top_k=args.top_k,
                    eos_id=tokenizer.sep_token_id,
                    pad_id=tokenizer.pad_token_id,
                )
                output_ids = output_ids[0]  # (batch, seq) → (seq,)

            lower = decode_pretrained(output_ids, tokenizer)
        else:
            # Custom model path
            if args.model == "decoder-only":
                input_ids = encode_decoder_only(upper, char2id)
            else:
                input_ids = encode_encoder_decoder(upper, char2id)

            print(f"\n上句: {upper}")

            if args.method == "greedy":
                seq, in_len = generate_greedy(model, input_ids, max_new_tokens=args.max_new,
                                              device=device, model_type=args.model)
            elif args.method == "beam":
                seq, in_len = generate_beam(model, input_ids, beam_size=args.beam_size,
                                            max_new_tokens=args.max_new, device=device,
                                            model_type=args.model)
            else:
                seq, in_len = generate_sample(model, input_ids, temperature=args.temperature,
                                              top_k=args.top_k, max_new_tokens=args.max_new,
                                              device=device, model_type=args.model)

            lower = decode_output(seq, id2char, skip_first_n=in_len)

        print(f"\n上句: {upper}")
        print(f"下句: {lower}")


if __name__ == "__main__":
    main()
