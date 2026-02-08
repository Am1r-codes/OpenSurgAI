#!/usr/bin/env python3
"""Train phase recognition model on Cholec80.

Simple training script:
- ResNet50 backbone (ImageNet pretrained)
- 7-class head (Cholec80 phases)
- Cross-entropy loss with class weights
- 5-10 epochs, early stopping if val accuracy plateaus

Usage:
    python scripts/train_phase.py

    # Custom split
    python scripts/train_phase.py --train-videos 1-30 --val-videos 31-35

    # Quick test
    python scripts/train_phase.py --train-videos 1-5 --val-videos 6-7 --epochs 2
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.phase.dataset import Cholec80PhaseDataset
from src.phase.pipeline import build_phase_model, NUM_PHASES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_video_range(s: str) -> list[str]:
    """Parse '1-30' or '1,2,5' into ['video01', 'video02', ...]."""
    ids = []
    for part in s.split(","):
        if "-" in part:
            start, end = part.split("-")
            ids.extend(range(int(start), int(end) + 1))
        else:
            ids.append(int(part))
    return [f"video{i:02d}" for i in ids]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> tuple[float, float]:
    """Train for one epoch. Returns (loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 50 == 0:
            log.info("  batch %d/%d, loss=%.4f", batch_idx + 1, len(loader), loss.item())

    return total_loss / total, correct / total


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    """Validate. Returns (loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Train Cholec80 phase recognition")
    parser.add_argument("--data-dir", type=Path, default=_PROJECT_ROOT / "data" / "cholec80")
    parser.add_argument("--train-videos", type=str, default="1-30", help="e.g. '1-30' or '1,2,5'")
    parser.add_argument("--val-videos", type=str, default="31-35", help="e.g. '31-35'")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=_PROJECT_ROOT / "weights" / "phase_resnet50.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    # Parse video IDs
    train_ids = parse_video_range(args.train_videos)
    val_ids = parse_video_range(args.val_videos)
    log.info("Train videos: %s", train_ids)
    log.info("Val videos: %s", val_ids)

    # Datasets
    train_dataset = Cholec80PhaseDataset(args.data_dir, train_ids, frame_stride=25)
    val_dataset = Cholec80PhaseDataset(args.data_dir, val_ids, frame_stride=25)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    # Model
    model = build_phase_model(num_classes=NUM_PHASES, weights_path=None)
    model = model.to(device)

    # Loss with class weights
    class_weights = train_dataset.get_class_weights().to(device)
    log.info("Class weights: %s", class_weights.tolist())
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    best_val_acc = 0.0
    patience = 3
    no_improve = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        log.info("=" * 50)
        log.info("Epoch %d/%d", epoch, args.epochs)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        log.info("Train loss=%.4f, acc=%.1f%%", train_loss, train_acc * 100)

        val_loss, val_acc = validate(model, val_loader, criterion, device)
        log.info("Val   loss=%.4f, acc=%.1f%%", val_loss, val_acc * 100)

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_accuracy": val_acc,
            }, args.output)
            log.info("Saved best model (%.1f%%) -> %s", val_acc * 100, args.output)
        else:
            no_improve += 1
            log.info("No improvement (%d/%d)", no_improve, patience)

        # Early stopping
        if no_improve >= patience:
            log.info("Early stopping at epoch %d", epoch)
            break

    log.info("=" * 50)
    log.info("Training complete. Best val accuracy: %.1f%%", best_val_acc * 100)
    log.info("Weights saved to: %s", args.output)


if __name__ == "__main__":
    main()
