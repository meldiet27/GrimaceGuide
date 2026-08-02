"""Check whether the AU thresholds calibrated on ground-truth landmarks still hold on predicted ones.

docs/au_threshold_calibration.md set every per-AU threshold from CatFLW *ground-truth* landmarks.
Production scores *predicted* landmarks instead, so a systematic bias in predicted point positions
would shift the geometry distributions out from under those thresholds -- a cutoff sitting at the
90th percentile of ground-truth geometry could sit at the 70th or the 99th in production. This
compares the two directly.

Three landmark sources are scored through the identical production formulas:

  gt        -- CatFLW ground-truth landmarks (what the thresholds were calibrated on)
  pred_gtbox-- LandmarkNet run on a ground-truth-bbox crop (isolates landmark-model error)
  pred_prod -- the real LocalLandmarkModel path: BBoxNet finds the face, then LandmarkNet
               (adds bbox-detector error on top, and is what the app actually runs)

Comparing pred_gtbox against pred_prod separates "the landmark model is biased" from "the face
detector crops differently than CatFLW's ground-truth boxes do".

All three are converted back to original-image pixel coordinates before any geometry is computed.
That matters: the model emits points normalized to a non-square crop that was resized to a square,
so computing ratios/angles in normalized-crop space would apply an anisotropic distortion that
pixel-space calibration never saw. (Angles in particular are not invariant under anisotropic
scaling.) Mapping back through crop_w/crop_h undoes it, and mirrors what
LocalLandmarkModel.detect_landmarks already does before handing points to scoring.

Usage:
    python -m ml.compare_predicted_geometry \
        --data-dir "ml/data/CatFLW dataset" \
        --checkpoint ml/checkpoints/landmark_net.pt \
        --bbox-checkpoint ml/checkpoints/bbox_net.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from grimaceguide.core.landmarks import landmarks_from_catflw_array
from grimaceguide.core.models import ActionUnitScore
from grimaceguide.core.scoring import (
    PAIN_THRESHOLD,
    _ear_geometry,
    _estimate_face_width,
    _head_geometry,
    _label_landmarks,
    _muzzle_geometry,
    _whisker_geometry,
    compute_grimace_score,
)
from ml.dataset import CatFLWDataset
from ml.model import LandmarkNet, decode_heatmaps

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
BOX_PADDING = 0.15  # matches ml/dataset.py's training-time crop convention

SOURCES = ("gt", "pred_gtbox", "pred_prod")
AU_NAMES = ("ears", "eyes", "muzzle", "whiskers", "head")

# The thresholds currently shipping in core/scoring.py, as (metric, comparison, cutoff)
# tuples -- used to report what fraction of the population each source's geometry puts
# on each side of the line. Kept in sync manually with _score_*; the point is to detect
# a distribution shift, so hardcoding what the code says today is intentional.
SHIPPING_THRESHOLDS = {
    "ear_angle": [("OBVIOUS", "gt", 55.0), ("MODERATE", "gt", 50.0)],
    "ear_tip_base_ratio": [("OBVIOUS", "gt", 1.9), ("MODERATE", "gt", 1.7)],
    "muzzle_ratio": [("OBVIOUS", "gt", 1.5), ("MODERATE", "gt", 1.1)],
    "whisker_spread": [("ABSENT-branch", "gt", 0.24)],
    "head_diff": [("OBVIOUS", "gt", 1.32), ("MODERATE", "gt", 0.86)],
    "eye_aspect_ratio": [("OBVIOUS", "lt", 0.30), ("MODERATE", "lt", 0.5)],
}


def val_stems(data_dir: Path, val_split: float, seed: int) -> list[str]:
    """Replicates ml.train.build_dataloaders' split so this evaluates held-out images only."""
    stems = sorted(p.stem for p in (data_dir / "images").glob("*.png"))
    n = len(stems)
    val_size = max(1, int(n * val_split))
    train_size = n - val_size
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    return [stems[i] for i in indices[train_size:]]


def padded_box(box, image_w: int, image_h: int) -> tuple[float, float, float, float]:
    x_min, y_min, x_max, y_max = box
    pad_x, pad_y = (x_max - x_min) * BOX_PADDING, (y_max - y_min) * BOX_PADDING
    return (
        max(0.0, x_min - pad_x),
        max(0.0, y_min - pad_y),
        min(float(image_w), x_max + pad_x),
        min(float(image_h), y_max + pad_y),
    )


def predict_on_box(model, image: Image.Image, box, image_size: int, device, normalize) -> list[float]:
    """Runs LandmarkNet on the given (already padded) crop box; returns flat pixel-space points."""
    x_min, y_min, x_max, y_max = box
    crop = image.crop((x_min, y_min, x_max, y_max))
    crop_w, crop_h = crop.size
    tensor = normalize(crop.resize((image_size, image_size), Image.BILINEAR)).unsqueeze(0).to(device)
    with torch.no_grad():
        points = decode_heatmaps(model(tensor)).cpu().numpy()[0].reshape(-1, 2)
    pixel = points.astype(np.float64)
    pixel[:, 0] = x_min + pixel[:, 0] * crop_w
    pixel[:, 1] = y_min + pixel[:, 1] * crop_h
    return pixel.reshape(-1).tolist()


def geometry_of(flat_points: list[float], width: int, height: int) -> dict[str, float]:
    """Computes every per-AU raw geometric value via the production helpers."""
    landmarks = landmarks_from_catflw_array(flat_points, image_width=width, image_height=height)
    labeled = _label_landmarks(landmarks)
    face_width = _estimate_face_width(labeled)

    row: dict[str, float] = {}
    ear = _ear_geometry(labeled)
    if ear is not None:
        row["ear_angle"], row["ear_tip_base_ratio"] = ear
    muzzle = _muzzle_geometry(labeled)
    if muzzle is not None:
        row["muzzle_ratio"] = muzzle
    whisker = _whisker_geometry(labeled, face_width)
    if whisker is not None and whisker[0] is not None:
        row["whisker_spread"] = whisker[0]
    head = _head_geometry(labeled, face_width)
    if head is not None:
        row["head_diff"] = head

    # eye_aspect_ratio has no extracted helper (it's computed inline in _score_eyes);
    # recompute it here from the same landmarks the production formula reads.
    l_outer, l_inner, l_top, l_bottom = (labeled.get(f"left_eye_{i + 1}") for i in range(4))
    r_outer, r_inner, r_top, r_bottom = (labeled.get(f"right_eye_{i + 1}") for i in range(4))
    if all([l_outer, l_inner, l_top, l_bottom, r_outer, r_inner, r_top, r_bottom]):
        avg_v = (np.hypot(l_top.x - l_bottom.x, l_top.y - l_bottom.y)
                 + np.hypot(r_top.x - r_bottom.x, r_top.y - r_bottom.y)) / 2
        avg_h = (np.hypot(l_outer.x - l_inner.x, l_outer.y - l_inner.y)
                 + np.hypot(r_outer.x - r_inner.x, r_outer.y - r_inner.y)) / 2
        if avg_h > 0:
            row["eye_aspect_ratio"] = float(avg_v / avg_h)

    result = compute_grimace_score(None, landmarks)
    for au, value in result.breakdown.as_dict().items():
        row[f"au_{au}"] = value
    row["total"] = result.breakdown.total
    row["pain"] = float(result.pain_likely)
    return row


def _pct(values: np.ndarray, p: float) -> float:
    return float(np.percentile(values, p))


def report_distributions(rows: dict[str, list[dict]], metric: str) -> None:
    print(f"\n### {metric}")
    header = f"{'source':<12}{'n':>6}{'mean':>9}{'median':>9}{'p10':>9}{'p90':>9}{'min':>9}{'max':>9}"
    print(header)
    for source in SOURCES:
        values = np.array([r[metric] for r in rows[source] if metric in r], dtype=np.float64)
        if len(values) == 0:
            print(f"{source:<12}{'--':>6}")
            continue
        print(
            f"{source:<12}{len(values):>6}{values.mean():>9.3f}{np.median(values):>9.3f}"
            f"{_pct(values, 10):>9.3f}{_pct(values, 90):>9.3f}{values.min():>9.3f}{values.max():>9.3f}"
        )

    for label, comparison, cutoff in SHIPPING_THRESHOLDS.get(metric, []):
        line = f"  hit rate {label} ({'<' if comparison == 'lt' else '>'} {cutoff}):"
        parts = []
        for source in SOURCES:
            values = np.array([r[metric] for r in rows[source] if metric in r], dtype=np.float64)
            if len(values) == 0:
                parts.append(f"{source}=--")
                continue
            hit = (values < cutoff) if comparison == "lt" else (values > cutoff)
            parts.append(f"{source}={hit.mean() * 100:.1f}%")
        print(f"{line:<44}{'  '.join(parts)}")


def report_agreement(rows: dict[str, list[dict]], pred_source: str) -> None:
    print(f"\n=== AU score agreement: gt vs {pred_source} ===")
    gt_rows, pred_rows = rows["gt"], rows[pred_source]
    for au in AU_NAMES:
        gt_scores = np.array([r[f"au_{au}"] for r in gt_rows])
        pred_scores = np.array([r[f"au_{au}"] for r in pred_rows])
        agree = (gt_scores == pred_scores).mean() * 100
        confusion = np.zeros((3, 3), dtype=int)
        for g, p in zip(gt_scores, pred_scores):
            confusion[int(g), int(p)] += 1
        rows_str = " | ".join(
            f"gt={i}: " + " ".join(f"{confusion[i, j]:>4}" for j in range(3)) for i in range(3)
        )
        print(f"  {au:<9} exact agreement {agree:>5.1f}%   [pred=0 1 2]  {rows_str}")

    gt_total = np.array([r["total"] for r in gt_rows], dtype=np.float64)
    pred_total = np.array([r["total"] for r in pred_rows], dtype=np.float64)
    gt_pain = np.array([r["pain"] for r in gt_rows])
    pred_pain = np.array([r["pain"] for r in pred_rows])
    correlation = float(np.corrcoef(gt_total, pred_total)[0, 1]) if gt_total.std() > 0 else float("nan")
    print(f"\n  total score: MAE={np.abs(gt_total - pred_total).mean():.3f}  corr={correlation:.3f}")
    print(
        f"  pain flag (normalized >= {PAIN_THRESHOLD}): "
        f"gt={gt_pain.mean() * 100:.1f}% flagged, {pred_source}={pred_pain.mean() * 100:.1f}% flagged, "
        f"agreement={(gt_pain == pred_pain).mean() * 100:.1f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bbox-checkpoint", required=True)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None, help="only evaluate the first N images")
    parser.add_argument("--dump-json", default=None, help="write per-image rows here for follow-up analysis")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    device = torch.device(args.device)
    normalize = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    image_size = checkpoint.get("image_size", 256)
    model = LandmarkNet(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # The production path, exactly as container.py wires it for the app.
    from grimaceguide.infrastructure.local_landmark_model import LocalLandmarkModel

    production = LocalLandmarkModel(
        checkpoint_path=args.checkpoint,
        bbox_checkpoint_path=args.bbox_checkpoint,
        device=str(device),
    )

    stems = val_stems(data_dir, args.val_split, args.seed)
    if args.limit:
        stems = stems[: args.limit]
    print(f"Evaluating {len(stems)} held-out val images on {device} (image_size={image_size})")

    rows: dict[str, list[dict]] = {source: [] for source in SOURCES}
    for stem in tqdm(stems):
        image = Image.open(data_dir / "images" / f"{stem}.png").convert("RGB")
        label = json.loads((data_dir / "labels" / f"{stem}.json").read_text())
        width, height = image.size

        gt_flat = [coord for point in label["labels"] for coord in point]
        rows["gt"].append(geometry_of(gt_flat, width, height))

        box = padded_box(label["bounding_boxes"], width, height)
        rows["pred_gtbox"].append(
            geometry_of(predict_on_box(model, image, box, image_size, device, normalize), width, height)
        )

        bgr = np.array(image)[:, :, ::-1].copy()  # LocalLandmarkModel expects an OpenCV-style BGR array
        landmark_set = production.detect_landmarks(bgr, name=f"{stem}.png")
        prod_flat = [
            coordinate
            for point in landmark_set.points
            for coordinate in (point.x, point.y)
        ]
        # detect_landmarks already applied the CatFLW->LANDMARK_LABELS remap; undo it so
        # geometry_of can apply the same remap to all three sources uniformly.
        from grimaceguide.core.landmarks import CATFLW_INDEX_FOR_LABEL

        raw = [0.0] * (len(CATFLW_INDEX_FOR_LABEL) * 2)
        for slot, catflw_index in enumerate(CATFLW_INDEX_FOR_LABEL):
            raw[2 * catflw_index] = prod_flat[2 * slot]
            raw[2 * catflw_index + 1] = prod_flat[2 * slot + 1]
        rows["pred_prod"].append(geometry_of(raw, width, height))

    if args.dump_json:
        Path(args.dump_json).write_text(json.dumps({"stems": stems, "rows": rows}))
        print(f"wrote per-image rows to {args.dump_json}")

    print("\n" + "=" * 78)
    print("GEOMETRY DISTRIBUTIONS (all in original-image pixel coordinates)")
    print("=" * 78)
    for metric in SHIPPING_THRESHOLDS:
        report_distributions(rows, metric)

    print("\n" + "=" * 78)
    print("SCORE AGREEMENT")
    print("=" * 78)
    report_agreement(rows, "pred_gtbox")
    report_agreement(rows, "pred_prod")


if __name__ == "__main__":
    main()
