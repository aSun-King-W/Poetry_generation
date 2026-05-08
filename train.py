"""Unified training script for poetry generation models."""

import os
import math
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

try:
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False


# ---------------------------------------------------------------------------
# Dataset loader for pre-tokenized .pt files
# ---------------------------------------------------------------------------

class PreTokenizedDataset(Dataset):
    """Wrapper over pre-tokenized data saved as .pt."""

    def __init__(self, data_path):
        self.data = torch.load(data_path, weights_only=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_decoder_only(batch):
    """Collate function for decoder-only pre-tokenized data."""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    # attention_mask: 1 for non-pad positions
    attention_mask = (input_ids != 0).long()
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


# ---------------------------------------------------------------------------
# LR scheduler: warmup + cosine decay
# ---------------------------------------------------------------------------

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Linear warmup followed by cosine decay to 0."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1.0, warmup_steps)
        progress = float(step - warmup_steps) / max(1.0, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device):
    """Compute validation loss."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in dataloader:
        with autocast(enabled=(device.type == "cuda" and AMP_AVAILABLE)):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        total_loss += loss.item() * shift_labels.numel()
        total_tokens += (shift_labels != -100).sum().item()

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")
    return avg_loss, perplexity


def train_decoder_only(args):
    """Train the Decoder-Only model."""
    from models.decoder_only import DecoderOnlyModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    print("Loading data...")
    train_dataset = PreTokenizedDataset(os.path.join(args.data_dir, "train_decoder_only.pt"))
    valid_dataset = PreTokenizedDataset(os.path.join(args.data_dir, "valid_decoder_only.pt"))

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_decoder_only,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_decoder_only,
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Valid samples: {len(valid_dataset)}")

    # Vocab size
    import json
    with open(args.vocab_path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
    vocab_size = len(vocab_data["char2id"])
    print(f"  Vocab size: {vocab_size}")

    # Model
    model = DecoderOnlyModel(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_len=args.max_len,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Optimizer & scheduler
    optimizer = Adam(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )

    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps=args.warmup_steps, total_steps=total_steps
    )

    # Loss
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=-100, label_smoothing=args.label_smoothing
    )

    # AMP scaler
    scaler = GradScaler(enabled=(device.type == "cuda" and AMP_AVAILABLE)) if AMP_AVAILABLE else None

    # Training loop
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_valid_loss = float("inf")
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            with autocast(enabled=(device.type == "cuda" and AMP_AVAILABLE)):
                logits = model(input_ids)  # (batch, seq_len, vocab_size)

                # Shift so that position t predicts token at t+1
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()

                loss = loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
                optimizer.step()
            scheduler.step()

            global_step += 1
            n_tokens = (shift_labels != -100).sum().item()
            epoch_loss += loss.item() * n_tokens
            epoch_tokens += n_tokens

            if global_step % args.log_interval == 0:
                print(f"  Step {global_step}/{total_steps} | loss: {loss.item():.4f}")

        # Epoch end: evaluate
        train_ppl = math.exp(epoch_loss / max(epoch_tokens, 1)) if epoch_tokens > 0 else float("inf")
        valid_loss, valid_ppl = evaluate(model, valid_loader, loss_fn, device)
        lr_now = scheduler.get_last_lr()[0]

        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train PPL: {train_ppl:.2f} | Valid loss: {valid_loss:.4f} | Valid PPL: {valid_ppl:.2f} | LR: {lr_now:.2e}")

        # Save checkpoint (best by valid loss)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            ckpt_path = os.path.join(args.checkpoint_dir, "decoder_only_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "valid_loss": valid_loss,
                "valid_ppl": valid_ppl,
                "args": vars(args),
            }, ckpt_path)
            print(f"  → Best model saved to {ckpt_path}")

        # Save epoch checkpoint
        ckpt_path = os.path.join(args.checkpoint_dir, f"decoder_only_epoch{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "valid_loss": valid_loss,
        }, ckpt_path)

        print()

    print("Training complete!")
    print(f"Best valid loss: {best_valid_loss:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train poetry generation model")
    subparsers = parser.add_subparsers(dest="model", required=True)

    # Decoder-Only subcommand
    dec = subparsers.add_parser("decoder-only")
    dec.add_argument("--data-dir", default="data/processed")
    dec.add_argument("--vocab-path", default="data/vocab.json")
    dec.add_argument("--checkpoint-dir", default="checkpoints")
    dec.add_argument("--d-model", type=int, default=256)
    dec.add_argument("--n-heads", type=int, default=8)
    dec.add_argument("--n-layers", type=int, default=6)
    dec.add_argument("--d-ff", type=int, default=1024)
    dec.add_argument("--max-len", type=int, default=32)
    dec.add_argument("--dropout", type=float, default=0.1)
    dec.add_argument("--lr", type=float, default=1e-4)
    dec.add_argument("--beta1", type=float, default=0.9)
    dec.add_argument("--beta2", type=float, default=0.999)
    dec.add_argument("--eps", type=float, default=1e-8)
    dec.add_argument("--batch-size", type=int, default=256)
    dec.add_argument("--epochs", type=int, default=30)
    dec.add_argument("--warmup-steps", type=int, default=2000)
    dec.add_argument("--max-norm", type=float, default=1.0)
    dec.add_argument("--label-smoothing", type=float, default=0.1)
    dec.add_argument("--num-workers", type=int, default=0)
    dec.add_argument("--log-interval", type=int, default=100)

    args = parser.parse_args()

    if args.model == "decoder-only":
        train_decoder_only(args)


if __name__ == "__main__":
    main()
