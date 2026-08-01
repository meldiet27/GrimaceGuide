"""Check whether the landmark model discriminates individual points or collapses.

For each of the 48 landmark indices, compares the model's prediction variance
across the validation set against the ground-truth variance. A landmark the
model has collapsed on (always predicting close to the same spot regardless of
the actual face) will have near-zero prediction variance even though the
ground-truth variance is real -- a low pred/gt variance ratio is the signature,
independent of how it looks at a glance in ml/visualize.py's 256px thumbnails.

Variance alone isn't sufficient, though: a model can vary its prediction across
samples without that variation actually tracking the true position (noise with
the right marginal spread, uncorrelated with ground truth per sample). So this
also reports the per-landmark Pearson correlation between predicted and true
coordinates -- the real signature of "does it track the right position," where
the variance ratio only rules out the flat-collapse failure mode specifically.

Usage:
    python -m ml.diagnose \
        --data-dir "ml/data/CatFLW dataset" \
        --checkpoint ml/checkpoints/landmark_net.pt
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from ml.model import LandmarkNet, decode_heatmaps
from ml.train import build_dataloaders


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--collapse-threshold", type=float, default=0.3,
                         help="pred/gt variance ratio below this is flagged as collapsed")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    image_size = checkpoint.get("image_size", 256)

    _, val_loader = build_dataloaders(
        args.data_dir, image_size, args.batch_size, args.val_split, args.seed, args.num_workers
    )

    model = LandmarkNet(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, _, points in val_loader:
            preds = decode_heatmaps(model(images.to(device))).cpu()
            all_preds.append(preds.numpy())
            all_targets.append(points.numpy())

    preds = np.concatenate(all_preds, axis=0).reshape(-1, 48, 2)
    targets = np.concatenate(all_targets, axis=0).reshape(-1, 48, 2)
    print(f"validation samples: {len(preds)}")

    pred_spread = preds.var(axis=0).sum(axis=-1)      # (48,) var(x)+var(y) per landmark
    gt_spread = targets.var(axis=0).sum(axis=-1)
    ratio = pred_spread / np.maximum(gt_spread, 1e-8)

    corr_x = np.array([np.corrcoef(preds[:, i, 0], targets[:, i, 0])[0, 1] for i in range(48)])
    corr_y = np.array([np.corrcoef(preds[:, i, 1], targets[:, i, 1])[0, 1] for i in range(48)])
    corr = (corr_x + corr_y) / 2

    order = np.argsort(corr)
    flagged_collapsed = 0
    flagged_uncorrelated = 0
    print(f"{'idx':>4}  {'pred_var':>10}  {'gt_var':>10}  {'var_ratio':>9}  {'corr':>6}")
    for i in order:
        flags = []
        if ratio[i] < args.collapse_threshold:
            flags.append("collapsed")
            flagged_collapsed += 1
        if corr[i] < 0.5:
            flags.append("uncorrelated")
            flagged_uncorrelated += 1
        flag_str = f" <-- {', '.join(flags)}" if flags else ""
        print(
            f"{i:>4}  {pred_spread[i]:>10.6f}  {gt_spread[i]:>10.6f}  "
            f"{ratio[i]:>9.3f}  {corr[i]:>6.3f}{flag_str}"
        )

    print(
        f"\n{flagged_collapsed}/48 landmarks collapsed (var_ratio < {args.collapse_threshold}); "
        f"{flagged_uncorrelated}/48 uncorrelated (corr < 0.5) despite non-collapsed variance\n"
        f"mean var_ratio={ratio.mean():.3f}  mean corr={corr.mean():.3f}  median corr={np.median(corr):.3f}"
    )


if __name__ == "__main__":
    main()
