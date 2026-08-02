"""Train the cat-face bounding-box detector on CatFLW.

Local:
    python -m ml.train_bbox --data-dir "ml/data/CatFLW dataset" --output ml/checkpoints/bbox_net.pt

Colab: see ml/README.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

from ml.bbox_dataset import CatFLWBBoxDataset
from ml.bbox_model import BBoxNet
from ml.dataset import subject_disjoint_split

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_dataloaders(
    data_dir, image_size, batch_size, val_split, seed, num_workers, group_by_subject=True
):
    normalize = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )
    train_dataset = CatFLWBBoxDataset(data_dir, image_size=image_size, transform=normalize, augment=True)
    val_dataset = CatFLWBBoxDataset(data_dir, image_size=image_size, transform=normalize, augment=False)

    n = len(train_dataset)
    if group_by_subject:
        # Same cat-level leak the landmark trainer had -- see ml/dataset.py.
        train_indices, val_indices = subject_disjoint_split(train_dataset.samples, val_split, seed)
    else:
        # Legacy index-level split, kept only to reproduce pre-fix checkpoints.
        val_size = max(1, int(n * val_split))
        indices = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
        train_indices, val_indices = indices[: n - val_size], indices[n - val_size :]

    train_set = Subset(train_dataset, train_indices)
    val_set = Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def box_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean IoU between two (B, 4) tensors of [x_min, y_min, x_max, y_max] boxes."""
    x_min = torch.maximum(pred[:, 0], target[:, 0])
    y_min = torch.maximum(pred[:, 1], target[:, 1])
    x_max = torch.minimum(pred[:, 2], target[:, 2])
    y_max = torch.minimum(pred[:, 3], target[:, 3])
    inter = (x_max - x_min).clamp(min=0) * (y_max - y_min).clamp(min=0)
    pred_area = (pred[:, 2] - pred[:, 0]).clamp(min=0) * (pred[:, 3] - pred[:, 1]).clamp(min=0)
    target_area = (target[:, 2] - target[:, 0]).clamp(min=0) * (target[:, 3] - target[:, 1]).clamp(min=0)
    union = pred_area + target_area - inter
    return (inter / union.clamp(min=1e-8)).mean().item()


def run_epoch(model, loader, device, criterion, optimizer=None):
    """Runs one train or eval pass. Returns (avg_loss, avg_iou).

    avg_iou (mean IoU between predicted and ground-truth boxes) is only
    computed during eval -- it's a much more interpretable accuracy signal
    than the raw SmoothL1 loss value for a bounding box.
    """
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_iou = 0.0
    with torch.set_grad_enabled(is_train):
        for images, targets in tqdm(loader, leave=False):
            images, targets = images.to(device), targets.to(device)
            preds = model(images)
            loss = criterion(preds, targets)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                total_iou += box_iou(preds.detach(), targets) * images.size(0)
            total_loss += loss.item() * images.size(0)
    avg_loss = total_loss / len(loader.dataset)
    avg_iou = None if is_train else total_iou / len(loader.dataset)
    return avg_loss, avg_iou


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", required=True, help="CatFLW dataset root (contains images/ and labels/)"
    )
    parser.add_argument("--output", default="ml/checkpoints/bbox_net.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--legacy-split",
        action="store_true",
        help="use the old index-level split (leaks cats across train/val -- only for "
             "reproducing pre-fix checkpoints)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    train_loader, val_loader = build_dataloaders(
        args.data_dir, args.image_size, args.batch_size, args.val_split, args.seed,
        args.num_workers, group_by_subject=not args.legacy_split,
    )
    split_kind = "index-level (LEAKY)" if args.legacy_split else "subject-disjoint"
    print(
        f"train samples={len(train_loader.dataset)}  val samples={len(val_loader.dataset)}  "
        f"split={split_kind}  device={device}"
    )

    model = BBoxNet(pretrained=True).to(device)
    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss, _ = run_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_iou = run_epoch(model, val_loader, device, criterion)
        print(
            f"epoch {epoch}/{args.epochs}  train_loss={train_loss:.5f}  "
            f"val_loss={val_loss:.5f}  val_iou={val_iou:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "image_size": args.image_size,
                    # See the matching comment in ml/train.py -- a checkpoint should
                    # always carry the split it was trained under.
                    "split": {
                        "val_split": args.val_split,
                        "seed": args.seed,
                        "group_by_subject": not args.legacy_split,
                    },
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "best_val_loss": val_loss,
                    "val_iou": val_iou,
                },
                output_path,
            )
            print(f"  saved new best checkpoint to {output_path}")


if __name__ == "__main__":
    main()
