"""PyTorch Datasets for Chinese poetry generation."""

import json
import os
import torch
from torch.utils.data import Dataset, random_split
import random


class PoetryDataset(Dataset):
    """Unified dataset for three model architectures.

    Modes:
        - 'decoder_only': Input = "上句 SEP 下句 EOS", causal LM format
        - 'encoder_decoder': Encoder input = "上句", Decoder input = "BOS 下句", labels = "下句 EOS"
        - 'pretrained': Uses pretrained tokenizer (space-separated chars + [CLS] prefix)

    For 'decoder_only' and 'encoder_decoder', the dataset pre-tokenizes all data
    on initialization for fast access during training.
    """

    def __init__(self, pairs, char2id, mode="decoder_only", max_len=32):
        """Initialize dataset.

        Args:
            pairs: List of (upper, lower) tuples.
            char2id: Character to token ID mapping.
            mode: One of 'decoder_only', 'encoder_decoder', 'pretrained'.
            max_len: Maximum sequence length for padding/truncation.
        """
        super().__init__()
        self.mode = mode
        self.max_len = max_len
        self.char2id = char2id
        self.pairs = pairs

        self.pad_id = 0   # <PAD>
        self.bos_id = 2   # <BOS>
        self.eos_id = 3   # <EOS>
        self.sep_id = 4   # <SEP>

        # Pre-tokenize all data for fast access
        print(f"Pre-tokenizing {len(pairs)} pairs for '{mode}' mode...")
        self.data = []
        for upper, lower in pairs:
            item = self._tokenize_pair(upper, lower)
            if item is not None:
                self.data.append(item)
        print(f"  Ready: {len(self.data)} samples")

    def _encode(self, text):
        """Convert text to token IDs, mapping unknown chars to <UNK>."""
        return [self.char2id.get(c, 1) for c in text]   # 1 = <UNK>

    def _tokenize_pair(self, upper, lower):
        """Tokenize a single pair for the configured mode."""
        upper_ids = self._encode(upper)
        lower_ids = self._encode(lower)

        if self.mode == "decoder_only":
            # input_ids = [上句..., SEP, 下句..., EOS, PAD...]
            # labels    = [上句..., SEP, 下句..., EOS, PAD...]
            # Loss ignores PAD; model internally shifts.
            seq = upper_ids + [self.sep_id] + lower_ids + [self.eos_id]
            if len(seq) > self.max_len:
                seq = seq[:self.max_len]
            length = len(seq)
            seq = seq + [self.pad_id] * (self.max_len - length)
            input_ids = torch.tensor(seq, dtype=torch.long)
            labels = input_ids.clone()
            # Set PAD positions in labels to -100 (ignored by CrossEntropyLoss)
            labels[labels == self.pad_id] = -100
            return {
                "input_ids": input_ids,
                "labels": labels,
                "length": length,
            }

        elif self.mode == "encoder_decoder":
            # encoder_input_ids = [上句..., PAD...]
            # decoder_input_ids = [BOS, 下句..., PAD...]
            # labels            = [下句..., EOS, PAD...]
            enc = upper_ids[:self.max_len]
            enc_len = len(enc)
            enc = enc + [self.pad_id] * (self.max_len - enc_len)

            dec = [self.bos_id] + lower_ids
            dec_len = len(dec)
            if dec_len > self.max_len:
                dec = dec[:self.max_len]
                dec_len = self.max_len
            dec = dec + [self.pad_id] * (self.max_len - dec_len)

            lbl = lower_ids + [self.eos_id]
            if len(lbl) > self.max_len:
                lbl = lbl[:self.max_len]
            lbl_len = len(lbl)
            lbl = lbl + [self.pad_id] * (self.max_len - lbl_len)

            labels = torch.tensor(lbl, dtype=torch.long)
            labels[labels == self.pad_id] = -100

            return {
                "encoder_input_ids": torch.tensor(enc, dtype=torch.long),
                "decoder_input_ids": torch.tensor(dec, dtype=torch.long),
                "labels": labels,
                "encoder_length": enc_len,
                "decoder_length": dec_len,
            }

        elif self.mode == "pretrained":
            # For pretrained, we store raw text for the tokenizer to process on-the-fly
            return {
                "upper": upper,
                "lower": lower,
            }

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_pretrained_item(upper, lower, tokenizer, max_len=64):
    """Convert a pair to pretrained model format on the fly.

    Format: "[CLS] 上 句 字 间 空 格 [SEP] 下 句 字 间 空 格"
    Characters must be space-separated for BertTokenizer.
    """
    upper_spaced = " ".join(list(upper))
    lower_spaced = " ".join(list(lower))
    text = f"[CLS] {upper_spaced} [SEP] {lower_spaced}"

    encoded = tokenizer(
        text,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    return {
        "input_ids": encoded["input_ids"].squeeze(0),
        "attention_mask": encoded["attention_mask"].squeeze(0),
        "upper": upper,
        "lower": lower,
    }


def create_dataloaders(pairs, char2id, mode="decoder_only", max_len=32,
                       batch_size=64, train_ratio=0.8, val_ratio=0.1,
                       seed=42, num_workers=0):
    """Create train/valid/test dataloaders.

    Args:
        pairs: List of (upper, lower) tuples.
        char2id: Character to token ID mapping.
        mode: Model mode ('decoder_only', 'encoder_decoder', 'pretrained').
        max_len: Maximum sequence length.
        batch_size: Batch size for dataloaders.
        train_ratio: Proportion of data for training.
        val_ratio: Proportion of data for validation.
        seed: Random seed for reproducibility.
        num_workers: Number of data loading workers.

    Returns:
        Tuple of (train_loader, valid_loader, test_loader).
    """
    random.seed(seed)
    random.shuffle(pairs)

    total = len(pairs)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_pairs = pairs[:train_end]
    val_pairs = pairs[train_end:val_end]
    test_pairs = pairs[val_end:]

    print(f"Split: train={len(train_pairs)}, valid={len(val_pairs)}, test={len(test_pairs)}")

    train_dataset = PoetryDataset(train_pairs, char2id, mode=mode, max_len=max_len)
    val_dataset = PoetryDataset(val_pairs, char2id, mode=mode, max_len=max_len)
    test_dataset = PoetryDataset(test_pairs, char2id, mode=mode, max_len=max_len)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def save_processed_data(pairs_path, output_dir, char2id,
                        max_len_decoder=32, max_len_encdec=32,
                        train_ratio=0.8, val_ratio=0.1, seed=42):
    """Pre-tokenize and save processed datasets for fast loading during training."""
    from utils.tokenizer import load_pairs

    pairs = load_pairs(pairs_path)

    random.seed(seed)
    random.shuffle(pairs)

    total = len(pairs)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        "train": pairs[:train_end],
        "valid": pairs[train_end:val_end],
        "test": pairs[val_end:],
    }

    os.makedirs(output_dir, exist_ok=True)

    for split_name, split_pairs in splits.items():
        for mode_name, ml in [("decoder_only", max_len_decoder),
                               ("encoder_decoder", max_len_encdec)]:
            ds = PoetryDataset(split_pairs, char2id, mode=mode_name, max_len=ml)
            save_path = os.path.join(output_dir, f"{split_name}_{mode_name}.pt")
            torch.save(ds.data, save_path)
            print(f"Saved {save_path} ({len(ds.data)} samples)")

    print("All processed data saved.")


def save_pretrained_processed_data(pairs_path, output_dir, max_len=64,
                                    train_ratio=0.8, val_ratio=0.1, seed=42):
    """Pre-tokenize pairs using uer/gpt2-chinese-poem tokenizer and save as .pt.

    Format: "[CLS] 上 句 字 [SEP] 下 句 字"

    Labels mask out the prefix (up to and including [SEP]) so the model
    only learns to predict the lower verse.
    """
    from models.pretrained import PretrainedPoetryModel
    from utils.tokenizer import load_pairs

    pairs = load_pairs(pairs_path)
    print(f"Loaded {len(pairs)} pairs")

    # Load pretrained tokenizer only (no model needed for preprocessing)
    tokenizer = PretrainedPoetryModel.from_pretrained().tokenizer

    random.seed(seed)
    random.shuffle(pairs)

    total = len(pairs)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        "train": pairs[:train_end],
        "valid": pairs[train_end:val_end],
        "test": pairs[val_end:],
    }

    os.makedirs(output_dir, exist_ok=True)

    for split_name, split_pairs in splits.items():
        data = []
        for upper, lower in split_pairs:
            # Build prefix: "[CLS] u p p e r [SEP]"
            upper_spaced = " ".join(list(upper))
            prefix_text = f"[CLS] {upper_spaced} [SEP]"

            # Build full text: prefix + " l o w e r"
            lower_spaced = " ".join(list(lower))
            full_text = f"{prefix_text} {lower_spaced}"

            # Tokenize full sequence with padding/truncation
            encoded = tokenizer(
                full_text,
                max_length=max_len,
                padding="max_length",
                truncation=True,
                add_special_tokens=False,
                return_tensors="pt",
            )

            # Tokenize prefix alone (no padding) to get exact prefix length
            prefix_enc = tokenizer(
                prefix_text,
                add_special_tokens=False,
            )

            input_ids = encoded["input_ids"].squeeze(0)       # (max_len,)
            attention_mask = encoded["attention_mask"].squeeze(0)
            labels = input_ids.clone()

            # Mask prefix positions (including [CLS], upper, [SEP])
            prefix_len = len(prefix_enc["input_ids"])
            labels[:prefix_len] = -100

            data.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })

        save_path = os.path.join(output_dir, f"{split_name}_pretrained.pt")
        torch.save(data, save_path)
        print(f"Saved {save_path} ({len(data)} samples)")

    print("All pretrained processed data saved.")


if __name__ == "__main__":
    import argparse
    from utils.tokenizer import load_vocab, load_pairs, load_and_pair_data

    parser = argparse.ArgumentParser(description="Preprocess and save dataset")
    parser.add_argument("--mode", choices=["custom", "pretrained"], default="custom",
                        help="custom=build vocab+preproc (default), pretrained=pretokenize for method-3")
    parser.add_argument("--pairs-path", default="data/processed/all_pairs.json")
    parser.add_argument("--vocab-path", default="data/vocab.json")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-len-decoder", type=int, default=32)
    parser.add_argument("--max-len-encdec", type=int, default=32)
    parser.add_argument("--max-len-pretrained", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.mode == "pretrained":
        save_pretrained_processed_data(
            pairs_path=args.pairs_path,
            output_dir=args.output_dir,
            max_len=args.max_len_pretrained,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
    else:
        print(f"Loading vocabulary from {args.vocab_path}...")
        char2id, id2char = load_vocab(args.vocab_path)
        print(f"Vocab size: {len(char2id)}")

        save_processed_data(
            pairs_path=args.pairs_path,
            output_dir=args.output_dir,
            char2id=char2id,
            max_len_decoder=args.max_len_decoder,
            max_len_encdec=args.max_len_encdec,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
