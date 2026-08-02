"""Train the cat facial-landmark regressor on the CatFLW dataset.

Local:
    python -m ml.train --data-dir "ml/data/CatFLW dataset" --output ml/checkpoints/landmark_net.pt

Colab: see ml/README.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

from ml.dataset import NUM_LANDMARKS, CatFLWDataset, subject_of
from ml.model import LandmarkNet, decode_heatmaps

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Ear-region landmark indices (see ml/diagnose.py): these 10 points came out of
# training with near-zero prediction/ground-truth correlation despite plenty of
# prediction variance -- not the mean-collapse the heatmap switch fixed for the
# other 38 points, but ears being harder to localize (they rotate independently
# of head pose, thin/low-contrast cartilage edges). Upweighting their loss
# pushes the model to spend more of its capacity getting these right.
EAR_LANDMARK_INDICES = list(range(22, 32))
EAR_LOSS_WEIGHT = 3.0


def make_weighted_heatmap_loss(device):
    weights = torch.ones(NUM_LANDMARKS, device=device)
    weights[EAR_LANDMARK_INDICES] = EAR_LOSS_WEIGHT
    weights = weights.view(1, NUM_LANDMARKS, 1, 1)

    def loss_fn(preds, targets):
        return (weights * (preds - targets) ** 2).mean()

    return loss_fn


def _subject_disjoint_split(samples, val_split, seed):
    """Splits whole cats into train/val, never the same cat's photos across both.

    CatFLW is ~339 cats with a median of 6 photos each, so an index-level random
    split leaks: measured on the default 0.15/seed-42 split, 310 of 311 val images
    had another photo of the same cat in training. That makes val a memorization
    check rather than a generalization one and inflates every held-out metric.
    Whole subjects are assigned to val until the image-count target is reached.
    """
    by_subject: dict[str, list[int]] = {}
    for index, stem in enumerate(samples):
        by_subject.setdefault(subject_of(stem), []).append(index)

    subjects = sorted(by_subject)
    order = torch.randperm(len(subjects), generator=torch.Generator().manual_seed(seed)).tolist()

    target = max(1, int(len(samples) * val_split))
    val_indices: list[int] = []
    for position in order:
        if len(val_indices) >= target:
            break
        val_indices.extend(by_subject[subjects[position]])

    val_lookup = set(val_indices)
    train_indices = [i for i in range(len(samples)) if i not in val_lookup]
    return train_indices, sorted(val_indices)


def build_dataloaders(
    data_dir, image_size, batch_size, val_split, seed, num_workers, group_by_subject=True
):
    normalize = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )
    # Two dataset instances over the same files: augmentation only ever applies to the
    # training split, so validation loss stays a clean, comparable signal across runs.
    train_dataset = CatFLWDataset(data_dir, image_size=image_size, transform=normalize, augment=True)
    val_dataset = CatFLWDataset(data_dir, image_size=image_size, transform=normalize, augment=False)

    n = len(train_dataset)
    if group_by_subject:
        train_indices, val_indices = _subject_disjoint_split(
            train_dataset.samples, val_split, seed
        )
    else:
        # Legacy index-level split. Leaks cats across the split -- kept only to
        # reproduce checkpoints trained before the subject-disjoint split existed.
        val_size = max(1, int(n * val_split))
        indices = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
        train_indices, val_indices = indices[: n - val_size], indices[n - val_size :]

    train_set = Subset(train_dataset, train_indices)
    val_set = Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def run_epoch(model, loader, device, criterion, optimizer=None):
    """Runs one train or eval pass. Returns (avg_loss, avg_landmark_error).

    avg_landmark_error is the mean absolute error, in normalized [0, 1] crop
    units, between argmax-decoded predicted points and ground truth -- only
    computed during eval (optimizer=None), since it's just for logging: a
    heatmap-MSE number alone isn't comparable to the coordinate-MSE numbers
    from before the heatmap switch, but this decoded error is.
    """
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_landmark_error = 0.0
    with torch.set_grad_enabled(is_train):
        for images, heatmaps, points in tqdm(loader, leave=False):
            images, heatmaps = images.to(device), heatmaps.to(device)
            preds = model(images)
            loss = criterion(preds, heatmaps)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                decoded = decode_heatmaps(preds.detach()).cpu()
                total_landmark_error += (decoded - points).abs().mean().item() * images.size(0)
            total_loss += loss.item() * images.size(0)
    avg_loss = total_loss / len(loader.dataset)
    avg_landmark_error = None if is_train else total_landmark_error / len(loader.dataset)
    return avg_loss, avg_landmark_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", required=True, help="CatFLW dataset root (contains images/ and labels/)"
    )
    parser.add_argument("--output", default="ml/checkpoints/landmark_net.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    train_loader, val_loader = build_dataloaders(
        args.data_dir, args.image_size, args.batch_size, args.val_split, args.seed, args.num_workers
    )
    print(f"train samples={len(train_loader.dataset)}  val samples={len(val_loader.dataset)}  device={device}")

    model = LandmarkNet(pretrained=True).to(device)
    criterion = make_weighted_heatmap_loss(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss, _ = run_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_landmark_error = run_epoch(model, val_loader, device, criterion)
        print(
            f"epoch {epoch}/{args.epochs}  train_loss={train_loss:.5f}  "
            f"val_loss={val_loss:.5f}  val_landmark_error={val_landmark_error:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {"model_state": model.state_dict(), "image_size": args.image_size}, output_path
            )
            print(f"  saved new best checkpoint to {output_path}")


if __name__ == "__main__":
    main()
