"""PyTorch dataset for the CatFLW facial-landmark dataset.

Expects the on-disk layout the CatFLW download extracts to:
    <root>/images/*.png
    <root>/labels/*.json   (one per image, same stem)

Each label JSON looks like:
    {"labels": [[x, y], ...],  # 48 points, pixel coordinates
     "bounding_boxes": [xmin, ymin, xmax, ymax]}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

NUM_LANDMARKS = 48


class CatFLWDataset(Dataset):
    """Crops each image to its face bounding box and yields (image, flattened landmarks).

    Landmarks are returned as a (NUM_LANDMARKS * 2,) float32 array normalized to [0, 1]
    relative to the (padded) crop, so they're independent of the original image size.
    """

    def __init__(self, root, image_size: int = 256, transform=None, box_padding: float = 0.15):
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.labels_dir = self.root / "labels"
        self.image_size = image_size
        self.transform = transform
        self.box_padding = box_padding
        self.samples = sorted(p.stem for p in self.images_dir.glob("*.png"))
        if not self.samples:
            raise FileNotFoundError(f"No images found under {self.images_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        stem = self.samples[idx]
        image = Image.open(self.images_dir / f"{stem}.png").convert("RGB")
        label = json.loads((self.labels_dir / f"{stem}.json").read_text())

        x_min, y_min, x_max, y_max = label["bounding_boxes"]
        pad_x = (x_max - x_min) * self.box_padding
        pad_y = (y_max - y_min) * self.box_padding
        x_min = max(0.0, x_min - pad_x)
        y_min = max(0.0, y_min - pad_y)
        x_max = min(float(image.width), x_max + pad_x)
        y_max = min(float(image.height), y_max + pad_y)

        crop = image.crop((x_min, y_min, x_max, y_max)).resize(
            (self.image_size, self.image_size), Image.BILINEAR
        )

        points = np.array(label["labels"], dtype=np.float32)
        points[:, 0] = (points[:, 0] - x_min) / (x_max - x_min)
        points[:, 1] = (points[:, 1] - y_min) / (y_max - y_min)
        points = np.clip(points, 0.0, 1.0)

        if self.transform:
            crop = self.transform(crop)

        return crop, points.reshape(-1)
