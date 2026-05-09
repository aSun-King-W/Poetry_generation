"""Inference: generate lower verse lines from upper lines."""

import os
import json
import argparse
import torch

from models.decoder_only import DecoderOnlyModel


def load_model(checkpoint_path, vocab_path, device):
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


def encode_input(upper_text, char2id, sep_id=4, max_len=32):
    """Convert upper sentence to token IDs: [上句..., SEP]."""
    ids = [char2id.get(c, 1) for c in upper_text]  # 1 = <UNK>
    ids = ids[:max_len - 1]  # leave room for SEP
    ids.append(sep_id)
    return torch.tensor(ids, dtype=torch.long)


def decode_output(token_ids, id2char, eos_id=3, pad_id=0, sep_id=4,
                  skip_first_n=0):
    """将 token ID 序列解码为文本，遇到 EOS 停止。

    Args:
        skip_first_n: 跳过前 N 个 token（输入部分），只解码模型新生成的内容。
    """
    chars = []
    for i, tid in enumerate(token_ids):
        if i < skip_first_n:
            continue
        tid = tid.item() if torch.is_tensor(tid) else tid
        if tid == eos_id:
            break
        if tid in (pad_id, sep_id):
            continue
        chars.append(id2char.get(tid, "�"))
    return "".join(chars)


@torch.no_grad()
def generate_greedy(model, input_ids, max_new_tokens=20, eos_id=3, device="cpu"):
    """贪心解码：每步选概率最高的 token"""
    input_ids = input_ids.to(device)
    seq = model.generate(
        input_ids, max_new_tokens=max_new_tokens,
        temperature=1.0, do_sample=False, eos_id=eos_id,
    )
    return seq[0], len(input_ids)


@torch.no_grad()
def generate_beam(model, input_ids, beam_size=5, max_new_tokens=20,
                  eos_id=3, device="cpu"):
    """集束搜索：维护 top-k 条候选路径"""
    input_ids = input_ids.to(device)
    seq = model.generate_beam(
        input_ids, beam_size=beam_size,
        max_new_tokens=max_new_tokens, eos_id=eos_id,
    )
    return seq, len(input_ids)


@torch.no_grad()
def generate_sample(model, input_ids, temperature=0.8, top_k=None,
                    max_new_tokens=20, eos_id=3, device="cpu"):
    """温度采样：从概率分布中随机抽取"""
    input_ids = input_ids.to(device)
    seq = model.generate(
        input_ids, max_new_tokens=max_new_tokens,
        temperature=temperature, do_sample=True,
        top_k=top_k, eos_id=eos_id,
    )
    return seq[0], len(input_ids)


def main():
    parser = argparse.ArgumentParser(description="Generate poetry with trained model")
    parser.add_argument("--checkpoint", default="checkpoints/decoder_only_best.pt")
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

    # Load model
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Train a model first: python train.py decoder-only")
        return
    model, char2id, id2char = load_model(args.checkpoint, args.vocab, device)
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
        input_ids = encode_input(upper, char2id)
        print(f"\n上句: {upper}")

        if args.method == "greedy":
            seq, in_len = generate_greedy(model, input_ids, max_new_tokens=args.max_new,
                                          device=device)
        elif args.method == "beam":
            seq, in_len = generate_beam(model, input_ids, beam_size=args.beam_size,
                                        max_new_tokens=args.max_new, device=device)
        else:  # sample
            seq, in_len = generate_sample(model, input_ids, temperature=args.temperature,
                                          top_k=args.top_k, max_new_tokens=args.max_new,
                                          device=device)

        lower = decode_output(seq, id2char, skip_first_n=in_len)
        print(f"下句: {lower}")


if __name__ == "__main__":
    main()
