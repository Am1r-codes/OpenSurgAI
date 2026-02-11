#!/usr/bin/env python3
"""Train multi-label surgical tool presence classifier on Cholec80.

ResNet50 backbone (ImageNet pretrained) with a 7-output sigmoid head.
Uses BCEWithLogitsLoss with positive class weighting for imbalanced tools.

Usage:
    python scripts/train_tool.py

    # Custom split
    python scripts/train_tool.py --train-videos 1-32 --val-videos 33-40

    # Quick test
    python scripts/train_tool.py --train-videos 1-5 --val-videos 6-7 --epochs 2
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet50_Weights

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.detection.tool_dataset import Cholec80ToolDataset, NUM_TOOLS, CHOLEC80_INSTRUMENTS

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


def build_tool_model(num_tools: int = NUM_TOOLS, weights_path: str | Path | None = None) -> nn.Module:
    """Build a ResNet50 with a multi-label sigmoid head."""
    backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    in_features = backbone.fc.in_features
    backbone.fc = nn.Linear(in_features, num_tools)

    if weights_path is not None:
        state = torch.load(str(weights_path), map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        backbone.load_state_dict(state)
        log.info("Loaded tool-classifier weights from %s", weights_path)

    return backbone


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> tuple[float, float]:
    """Train for one epoch. Returns (loss, mean_ap)."""
    model.train()
    total_loss = 0.0
    correct_preds = 0
    total_preds = 0

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        # Per-label accuracy (threshold 0.5)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct_preds += (preds == targets).sum().item()
        total_preds += targets.numel()

        if (batch_idx + 1) % 50 == 0:
            log.info("  batch %d/%d, loss=%.4f", batch_idx + 1, len(loader), loss.item())

    acc = correct_preds / total_preds if total_preds > 0 else 0
    return total_loss / len(loader.dataset), acc


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float, dict[str, float]]:
    """Validate. Returns (loss, mean_label_accuracy, per_tool_accuracy)."""
    model.eval()
    total_loss = 0.0
    per_tool_correct = [0] * NUM_TOOLS
    per_tool_total = [0] * NUM_TOOLS

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        loss = criterion(logits, targets)
        total_loss += loss.item() * images.size(0)

        preds = (torch.sigmoid(logits) > 0.5).float()
        for t in range(NUM_TOOLS):
            per_tool_correct[t] += (preds[:, t] == targets[:, t]).sum().item()
            per_tool_total[t] += targets.size(0)

    per_tool_acc = {}
    for t in range(NUM_TOOLS):
        acc = per_tool_correct[t] / per_tool_total[t] if per_tool_total[t] > 0 else 0
        per_tool_acc[CHOLEC80_INSTRUMENTS[t]] = round(acc, 4)

    mean_acc = sum(per_tool_acc.values()) / NUM_TOOLS
    return total_loss / len(loader.dataset), mean_acc, per_tool_acc


def main():
    parser = argparse.ArgumentParser(description="Train Cholec80 tool presence classifier")
    parser.add_argument("--data-dir", type=Path, default=_PROJECT_ROOT / "data" / "cholec80")
    parser.add_argument("--train-videos", type=str, default="1-32", help="e.g. '1-32'")
    parser.add_argument("--val-videos", type=str, default="33-40", help="e.g. '33-40'")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=_PROJECT_ROOT / "weights" / "tool_resnet50.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    train_ids = parse_video_range(args.train_videos)
    val_ids = parse_video_range(args.val_videos)
    log.info("Train videos: %d (%s .. %s)", len(train_ids), train_ids[0], train_ids[-1])
    log.info("Val videos: %d (%s .. %s)", len(val_ids), val_ids[0], val_ids[-1])

    # Datasets
    train_dataset = Cholec80ToolDataset(args.data_dir, train_ids, frame_stride=1)
    val_dataset = Cholec80ToolDataset(args.data_dir, val_ids, frame_stride=1)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    # Model
    model = build_tool_model(num_tools=NUM_TOOLS, weights_path=None)
    model = model.to(device)

    # Loss with positive class weights
    pos_weights = train_dataset.get_pos_weights().to(device)
    log.info("Positive weights: %s", {CHOLEC80_INSTRUMENTS[i]: round(pos_weights[i].item(), 2) for i in range(NUM_TOOLS)})
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    # Training loop
    best_val_acc = 0.0
    patience = 4
    no_improve = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        log.info("=" * 60)
        log.info("Epoch %d/%d", epoch, args.epochs)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        log.info("Train  loss=%.4f, label_acc=%.1f%%", train_loss, train_acc * 100)

        val_loss, val_acc, per_tool = validate(model, val_loader, criterion, device)
        log.info("Val    loss=%.4f, mean_label_acc=%.1f%%", val_loss, val_acc * 100)
        for tool, acc in per_tool.items():
            log.info("  %-20s %.1f%%", tool, acc * 100)

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_accuracy": val_acc,
                "per_tool_accuracy": per_tool,
                "num_tools": NUM_TOOLS,
                "instruments": CHOLEC80_INSTRUMENTS,
            }, args.output)
            log.info("Saved best model (%.1f%%) -> %s", val_acc * 100, args.output)
        else:
            no_improve += 1
            log.info("No improvement (%d/%d)", no_improve, patience)

        if no_improve >= patience:
            log.info("Early stopping at epoch %d", epoch)
            break

    log.info("=" * 60)
    log.info("Training complete. Best mean label accuracy: %.1f%%", best_val_acc * 100)
    log.info("Weights saved to: %s", args.output)


if __name__ == "__main__":
    main()
