"""Tokenization and vocabulary building for Chinese poetry generation."""

import json
import os
import glob
import re
from collections import Counter

# Special tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
SEP_TOKEN = "<SEP>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, SEP_TOKEN]
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
SEP_ID = 4

# Chinese punctuation for splitting lines
CHINESE_PUNCTUATION = set("，。！？、；：""''（）《》【】「」『』—…·")


def is_chinese_char(c):
    """Check if a character is a Chinese character (CJK Unified Ideographs)."""
    return '\u4e00' <= c <= '\u9fff' or '\u3400' <= c <= '\u4dbf'


def split_paragraph(para):
    """Split a paragraph into individual lines by Chinese punctuation."""
    # Split on Chinese punctuation (keep empty parts)
    parts = re.split(r'[，。！？、；：]', para)
    return [p.strip() for p in parts if p.strip()]


def extract_poem_lines(poem):
    """Extract individual 5/7-character verse lines from a poem record."""
    paragraphs = poem.get("paragraphs", [])
    all_lines = []
    for para in paragraphs:
        # First try splitting by punctuation
        split_lines = split_paragraph(para)
        if not split_lines:
            continue
        # Check if each split line is a valid verse line (5 or 7 Chinese chars)
        for line in split_lines:
            chars = [c for c in line if is_chinese_char(c)]
            if len(chars) in (5, 7):
                all_lines.append("".join(chars))
    return all_lines


def pair_adjacent_lines(lines):
    """Pair adjacent lines into (上句, 下句) tuples."""
    pairs = []
    for i in range(0, len(lines) - 1, 2):
        upper, lower = lines[i], lines[i + 1]
        # A couplet should have the same length
        if len(upper) == len(lower):
            pairs.append((upper, lower))
    return pairs


def load_and_pair_data(data_dirs, max_pairs=None):
    """Load poem data, extract lines, and create (上句, 下句) pairs.

    Args:
        data_dirs: List of directories containing JSON files.
        max_pairs: Optional limit on number of pairs to extract.

    Returns:
        List of (upper_line, lower_line) tuples.
    """
    all_pairs = []

    for data_dir in data_dirs:
        json_files = glob.glob(os.path.join(data_dir, "poet.song.*.json"))
        json_files.sort()

        for fpath in json_files:
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    poems = json.load(f)
            except Exception:
                continue

            for poem in poems:
                lines = extract_poem_lines(poem)
                pairs = pair_adjacent_lines(lines)
                all_pairs.extend(pairs)

                if max_pairs is not None and len(all_pairs) >= max_pairs:
                    return all_pairs[:max_pairs]

    return all_pairs


def build_vocab(all_pairs, min_freq=3):
    """Build character vocabulary from all poem pairs.

    Args:
        all_pairs: List of (upper, lower) tuples.
        min_freq: Minimum character frequency to include in vocab.

    Returns:
        Tuple of (char2id, id2char) dictionaries.
    """
    counter = Counter()
    for upper, lower in all_pairs:
        counter.update(upper)
        counter.update(lower)

    # Filter by frequency, sort by frequency descending
    vocab_chars = [c for c, freq in counter.items() if freq >= min_freq]

    # Sort by frequency descending, then alphabetically for stability
    vocab_chars.sort(key=lambda c: (-counter[c], c))

    char2id = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    id2char = {i: tok for i, tok in enumerate(SPECIAL_TOKENS)}

    for i, char in enumerate(vocab_chars):
        idx = len(SPECIAL_TOKENS) + i
        char2id[char] = idx
        id2char[idx] = char

    print(f"Vocabulary size: {len(char2id)} (special: {len(SPECIAL_TOKENS)}, chars: {len(vocab_chars)})")
    print(f"Character frequency threshold: >= {min_freq}")
    print(f"Total character types before filtering: {len(counter)}")

    return char2id, id2char


def save_vocab(char2id, id2char, save_path):
    """Save vocabulary to JSON file."""
    vocab_data = {
        "char2id": char2id,
        "id2char": {str(k): v for k, v in id2char.items()},
        "special_tokens": SPECIAL_TOKENS,
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)
    print(f"Vocabulary saved to {save_path}")


def load_vocab(vocab_path):
    """Load vocabulary from JSON file."""
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)

    char2id = vocab_data["char2id"]
    id2char = {int(k): v for k, v in vocab_data["id2char"].items()}
    return char2id, id2char


class PoetryTokenizer:
    """Tokenizer for Chinese poetry using built character vocabulary."""

    def __init__(self, char2id):
        self.char2id = char2id
        self.id2char = {v: k for k, v in char2id.items()}

    def encode(self, text, add_special=False):
        """Convert text to token IDs.

        Args:
            text: Input string.
            add_special: If True, wrap with BOS/EOS.

        Returns:
            List of token IDs.
        """
        ids = []
        if add_special:
            ids.append(BOS_ID)
        for char in text:
            ids.append(self.char2id.get(char, UNK_ID))
        if add_special:
            ids.append(EOS_ID)
        return ids

    def decode(self, ids, skip_special=False):
        """Convert token IDs back to text."""
        chars = []
        for i in ids:
            if skip_special and i < len(SPECIAL_TOKENS):
                continue
            if i in self.id2char:
                chars.append(self.id2char[i])
        return "".join(chars)


def save_pairs(all_pairs, save_path):
    """Save poem pairs to JSON file."""
    data = [{"upper": u, "lower": l} for u, l in all_pairs]
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(all_pairs)} pairs to {save_path}")


def load_pairs(pairs_path):
    """Load poem pairs from JSON file."""
    with open(pairs_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [(item["upper"], item["lower"]) for item in data]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess Chinese poetry data")
    parser.add_argument("--data-dirs", nargs="+", default=[
        "data/raw/chinese-poetry/全唐诗",
    ], help="Directories containing poem JSON files")
    parser.add_argument("--output-dir", default="data/processed",
                        help="Output directory for processed data")
    parser.add_argument("--vocab-path", default="data/vocab.json",
                        help="Path to save vocabulary")
    parser.add_argument("--min-freq", type=int, default=3,
                        help="Minimum character frequency")
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="Maximum number of pairs to extract")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading and pairing poem data...")
    all_pairs = load_and_pair_data(args.data_dirs, max_pairs=args.max_pairs)
    print(f"Total pairs extracted: {len(all_pairs)}")

    # Save raw pairs
    pairs_path = os.path.join(args.output_dir, "all_pairs.json")
    save_pairs(all_pairs, pairs_path)

    # Build and save vocabulary
    print("Building vocabulary...")
    char2id, id2char = build_vocab(all_pairs, min_freq=args.min_freq)
    save_vocab(char2id, id2char, args.vocab_path)
