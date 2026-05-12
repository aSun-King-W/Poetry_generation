"""Unified training script for poetry generation models."""

import os
import math
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR

try:
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False


# ---------------------------------------------------------------------------
# Multi-GPU helpers
# ---------------------------------------------------------------------------

def setup_multi_gpu(model, args):
    """Wrap model with DataParallel if --multi-gpu is set and multiple GPUs exist."""
    if args.multi_gpu:
        n_gpu = torch.cuda.device_count()
        if n_gpu > 1:
            model = nn.DataParallel(model)
            print(f"  Using {n_gpu} GPUs (DataParallel)")
        elif n_gpu == 1:
            print("  Only 1 GPU found, --multi-gpu has no effect")
        else:
            print("  No GPU found, --multi-gpu has no effect")
    return model


def unwrap_model(model):
    """Get the underlying model (unwraps DataParallel)."""
    return model.module if hasattr(model, "module") else model


def get_model_state(model):
    """Get state_dict, unwrapping DataParallel if needed."""
    return unwrap_model(model).state_dict()


def load_state_into_model(model, state_dict):
    """Load state_dict, handling DataParallel wrapper mismatch."""
    try:
        model.load_state_dict(state_dict)
    except KeyError:
        # Keys are prefixed with 'module.' from DataParallel save
        from collections import OrderedDict
        new_state = OrderedDict()
        for k, v in state_dict.items():
            new_state[k.replace("module.", "")] = v
        model.load_state_dict(new_state)


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


def collate_encoder_decoder(batch):
    """Collate function for encoder-decoder pre-tokenized data."""
    encoder_input_ids = torch.stack([item["encoder_input_ids"] for item in batch])
    decoder_input_ids = torch.stack([item["decoder_input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {
        "encoder_input_ids": encoder_input_ids,
        "decoder_input_ids": decoder_input_ids,
        "labels": labels,
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
        n_tokens = (shift_labels != -100).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

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
    start_epoch = 1
    global_step = 0

    # 断点续训：从 checkpoint 恢复模型、优化器、调度器状态
    if args.resume:
        print(f"Resuming from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_valid_loss = ckpt.get("valid_loss", float("inf"))
        print(f"  Restored epoch={ckpt['epoch']}, valid_loss={best_valid_loss:.4f}")
        # 恢复 AMP scaler 状态（如果保存了）
        if "scaler_state_dict" in ckpt and scaler is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])

    for epoch in range(start_epoch, args.epochs + 1):
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
            save_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "valid_loss": valid_loss,
                "valid_ppl": valid_ppl,
                "args": vars(args),
            }
            if scaler is not None:
                save_dict["scaler_state_dict"] = scaler.state_dict()
            torch.save(save_dict, ckpt_path)
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


def train_encoder_decoder(args):
    """Train the Encoder-Decoder model."""
    from models.encoder_decoder import EncoderDecoderModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    print("Loading data...")
    train_dataset = PreTokenizedDataset(
        os.path.join(args.data_dir, "train_encoder_decoder.pt")
    )
    valid_dataset = PreTokenizedDataset(
        os.path.join(args.data_dir, "valid_encoder_decoder.pt")
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_encoder_decoder,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_encoder_decoder,
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
    model = EncoderDecoderModel(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        enc_n_layers=args.enc_n_layers,
        dec_n_layers=args.dec_n_layers,
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

    # Loss (ignore PAD positions)
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=-100, label_smoothing=args.label_smoothing
    )

    # AMP scaler
    scaler = GradScaler(enabled=(device.type == "cuda" and AMP_AVAILABLE)) if AMP_AVAILABLE else None

    # Training loop
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_valid_loss = float("inf")
    start_epoch = 1
    global_step = 0

    # Resume from checkpoint (BEFORE DataParallel wrapping, to avoid key mismatch)
    if args.resume:
        print(f"Resuming from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_valid_loss = ckpt.get("valid_loss", float("inf"))
        print(f"  Restored epoch={ckpt['epoch']}, valid_loss={best_valid_loss:.4f}")
        if "scaler_state_dict" in ckpt and scaler is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])

    # Wrap with DataParallel AFTER loading checkpoint
    model = setup_multi_gpu(model, args)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0

        for batch in train_loader:
            enc_ids = batch["encoder_input_ids"].to(device)
            dec_ids = batch["decoder_input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            with autocast(enabled=(device.type == "cuda" and AMP_AVAILABLE)):
                logits = model(enc_ids, dec_ids)  # (batch, seq_len, vocab_size)
                loss = loss_fn(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
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
            n_tokens = (labels != -100).sum().item()
            epoch_loss += loss.item() * n_tokens
            epoch_tokens += n_tokens

            if global_step % args.log_interval == 0:
                print(f"  Step {global_step}/{total_steps} | loss: {loss.item():.4f}")

        # Epoch end: evaluate
        train_ppl = math.exp(epoch_loss / max(epoch_tokens, 1)) if epoch_tokens > 0 else float("inf")
        valid_loss, valid_ppl = evaluate_encoder_decoder(model, valid_loader, loss_fn, device)
        lr_now = scheduler.get_last_lr()[0]

        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train PPL: {train_ppl:.2f} | Valid loss: {valid_loss:.4f} | Valid PPL: {valid_ppl:.2f} | LR: {lr_now:.2e}")

        # Save checkpoint (best by valid loss)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            ckpt_path = os.path.join(args.checkpoint_dir, "encoder_decoder_best.pt")
            save_dict = {
                "epoch": epoch,
                "model_state_dict": get_model_state(model),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "valid_loss": valid_loss,
                "valid_ppl": valid_ppl,
                "args": vars(args),
            }
            if scaler is not None:
                save_dict["scaler_state_dict"] = scaler.state_dict()
            torch.save(save_dict, ckpt_path)
            print(f"  → Best model saved to {ckpt_path}")

        # Save epoch checkpoint
        ckpt_path = os.path.join(args.checkpoint_dir, f"encoder_decoder_epoch{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": get_model_state(model),
            "valid_loss": valid_loss,
        }, ckpt_path)

        print()

    print("Training complete!")
    print(f"Best valid loss: {best_valid_loss:.4f}")


@torch.no_grad()
def evaluate_encoder_decoder(model, dataloader, loss_fn, device):
    """Compute validation loss for encoder-decoder."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in dataloader:
        with autocast(enabled=(device.type == "cuda" and AMP_AVAILABLE)):
            enc_ids = batch["encoder_input_ids"].to(device)
            dec_ids = batch["decoder_input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits = model(enc_ids, dec_ids)
            loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )
        n_tokens = (labels != -100).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")
    return avg_loss, perplexity


# ---------------------------------------------------------------------------
# Pretrained dataset
# ---------------------------------------------------------------------------

def collate_pretrained(batch):
    """Collate function for pre-tokenized pretrained data."""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


@torch.no_grad()
def evaluate_pretrained(model, dataloader, device):
    """Compute validation loss for pretrained model."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in dataloader:
        with autocast(enabled=(device.type == "cuda" and AMP_AVAILABLE)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss.mean()

        n_tokens = (labels != -100).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")
    return avg_loss, perplexity


def train_pretrained(args):
    """Fine-tune the uer/gpt2-chinese-poem pretrained model."""
    from models.pretrained import PretrainedPoetryModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    print("Loading data...")
    train_dataset = PreTokenizedDataset(os.path.join(args.data_dir, "train_pretrained.pt"))
    valid_dataset = PreTokenizedDataset(os.path.join(args.data_dir, "valid_pretrained.pt"))

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_pretrained,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_pretrained,
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Valid samples: {len(valid_dataset)}")

    # Model
    print(f"Loading pretrained model: uer/gpt2-chinese-poem...")
    pretrained = PretrainedPoetryModel.from_pretrained()

    # Multi-GPU: wrap the inner GPT2LMHeadModel
    if args.multi_gpu:
        n_gpu = torch.cuda.device_count()
        if n_gpu > 1:
            pretrained.model = nn.DataParallel(pretrained.model)
            print(f"  Using {n_gpu} GPUs (DataParallel on inner model)")
        elif n_gpu == 1:
            print("  Only 1 GPU found, --multi-gpu has no effect")
        else:
            print("  No GPU found, --multi-gpu has no effect")

    pretrained.to(device)

    n_params = sum(p.numel() for p in pretrained.model.parameters())
    print(f"Model parameters: {n_params:,}")

    def get_inner_model():
        """Get the underlying GPT2LMHeadModel (unwrapping DataParallel)."""
        m = pretrained.model
        return m.module if hasattr(m, 'module') else m

    # Optimizer: AdamW with weight decay (standard for fine-tuning)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in get_inner_model().named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in get_inner_model().named_parameters()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.eps)

    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps=args.warmup_steps, total_steps=total_steps
    )

    # AMP scaler
    scaler = GradScaler(enabled=(device.type == "cuda" and AMP_AVAILABLE)) if AMP_AVAILABLE else None

    # Training loop
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_valid_loss = float("inf")
    start_epoch = 1
    global_step = 0

    # Resume from checkpoint
    if args.resume:
        print(f"Resuming from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        load_state_into_model(get_inner_model(), ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_valid_loss = ckpt.get("valid_loss", float("inf"))
        print(f"  Restored epoch={ckpt['epoch']}, valid_loss={best_valid_loss:.4f}")
        if "scaler_state_dict" in ckpt and scaler is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])

    for epoch in range(start_epoch, args.epochs + 1):
        pretrained.train()
        epoch_loss = 0.0
        epoch_tokens = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            with autocast(enabled=(device.type == "cuda" and AMP_AVAILABLE)):
                outputs = pretrained(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss.mean()  # .mean() 处理 DataParallel 下 loss 变向量的情况

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(pretrained.parameters(), args.max_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(pretrained.parameters(), args.max_norm)
                optimizer.step()
            scheduler.step()

            global_step += 1
            n_tokens = (labels != -100).sum().item()
            epoch_loss += loss.item() * n_tokens
            epoch_tokens += n_tokens

            if global_step % args.log_interval == 0:
                print(f"  Step {global_step}/{total_steps} | loss: {loss.item():.4f}")

        # Epoch end: evaluate
        train_ppl = math.exp(epoch_loss / max(epoch_tokens, 1)) if epoch_tokens > 0 else float("inf")
        valid_loss, valid_ppl = evaluate_pretrained(pretrained, valid_loader, device)
        lr_now = scheduler.get_last_lr()[0]

        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train PPL: {train_ppl:.2f} | Valid loss: {valid_loss:.4f} | Valid PPL: {valid_ppl:.2f} | LR: {lr_now:.2e}")

        # Save checkpoint (best by valid loss)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            ckpt_path = os.path.join(args.checkpoint_dir, "pretrained_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": get_inner_model().state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "valid_loss": valid_loss,
                "valid_ppl": valid_ppl,
                "args": vars(args),
            }, ckpt_path)
            print(f"  → Best model saved to {ckpt_path}")

        # Save epoch checkpoint
        ckpt_path = os.path.join(args.checkpoint_dir, f"pretrained_epoch{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": get_inner_model().state_dict(),
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
    dec.add_argument("--d-model", type=int, default=512)
    dec.add_argument("--n-heads", type=int, default=8)
    dec.add_argument("--n-layers", type=int, default=8)
    dec.add_argument("--d-ff", type=int, default=2048)
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
    dec.add_argument("--resume", type=str, default=None,
                      help="从 checkpoint 文件恢复训练（如 checkpoints/decoder_only_best.pt）")
    dec.add_argument("--num-workers", type=int, default=0)
    dec.add_argument("--log-interval", type=int, default=100)
    dec.add_argument("--multi-gpu", action="store_true",
                      help="使用所有可用 GPU 训练")

    # Encoder-Decoder subcommand
    encdec = subparsers.add_parser("encoder-decoder")
    encdec.add_argument("--data-dir", default="data/processed")
    encdec.add_argument("--vocab-path", default="data/vocab.json")
    encdec.add_argument("--checkpoint-dir", default="checkpoints")
    encdec.add_argument("--d-model", type=int, default=512)
    encdec.add_argument("--n-heads", type=int, default=8)
    encdec.add_argument("--enc-n-layers", type=int, default=6)
    encdec.add_argument("--dec-n-layers", type=int, default=6)
    encdec.add_argument("--d-ff", type=int, default=2048)
    encdec.add_argument("--max-len", type=int, default=32)
    encdec.add_argument("--dropout", type=float, default=0.1)
    encdec.add_argument("--lr", type=float, default=1e-4)
    encdec.add_argument("--beta1", type=float, default=0.9)
    encdec.add_argument("--beta2", type=float, default=0.999)
    encdec.add_argument("--eps", type=float, default=1e-8)
    encdec.add_argument("--batch-size", type=int, default=256)
    encdec.add_argument("--epochs", type=int, default=30)
    encdec.add_argument("--warmup-steps", type=int, default=2000)
    encdec.add_argument("--max-norm", type=float, default=1.0)
    encdec.add_argument("--label-smoothing", type=float, default=0.1)
    encdec.add_argument("--resume", type=str, default=None,
                        help="从 checkpoint 恢复训练（如 checkpoints/encoder_decoder_best.pt）")
    encdec.add_argument("--num-workers", type=int, default=0)
    encdec.add_argument("--log-interval", type=int, default=100)
    encdec.add_argument("--multi-gpu", action="store_true",
                        help="使用所有可用 GPU 训练")

    # Pretrained subcommand
    pt = subparsers.add_parser("pretrained")
    pt.add_argument("--data-dir", default="data/processed")
    pt.add_argument("--checkpoint-dir", default="checkpoints")
    pt.add_argument("--lr", type=float, default=2e-5)
    pt.add_argument("--batch-size", type=int, default=64)
    pt.add_argument("--epochs", type=int, default=10)
    pt.add_argument("--warmup-steps", type=int, default=500)
    pt.add_argument("--max-norm", type=float, default=1.0)
    pt.add_argument("--weight-decay", type=float, default=0.01)
    pt.add_argument("--eps", type=float, default=1e-8)
    pt.add_argument("--resume", type=str, default=None,
                    help="从 checkpoint 恢复训练（如 checkpoints/pretrained_best.pt）")
    pt.add_argument("--num-workers", type=int, default=0)
    pt.add_argument("--log-interval", type=int, default=100)
    pt.add_argument("--multi-gpu", action="store_true",
                    help="使用所有可用 GPU 训练")

    args = parser.parse_args()

    if args.model == "decoder-only":
        train_decoder_only(args)
    elif args.model == "encoder-decoder":
        train_encoder_decoder(args)
    elif args.model == "pretrained":
        train_pretrained(args)


if __name__ == "__main__":
    main()
