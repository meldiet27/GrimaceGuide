"""PyTorch dataset for CatFLW face-bounding-box regression.

Reuses the same on-disk layout as CatFLWDataset (dataset.py) but targets the
raw ground-truth bounding_boxes label instead of cropping to it -- this is the
detector that produces that crop at inference time for photos that aren't
already face-centered (see grimaceguide/infrastructure/local_landmark_model.py).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter


class CatFLWBBoxDataset(Dataset):
    """Yields (image, bbox) where bbox = [x_min, y_min, x_max, y_max] normalized to
    [0, 1] relative to the *whole* image (not a crop, unlike CatFLWDataset).

    augment=True applies a random horizontal flip -- safe here, unlike for
    landmarks, since a bounding box has no left/right semantic identity to
    mislabel -- plus color jitter.
    """

    def __init__(self, root, image_size: int = 224, transform=None, augment: bool = False):
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.labels_dir = self.root / "labels"
        self.image_size = image_size
        self.transform = transform
        self.augment = augment
        self.color_jitter = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)
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
        w, h = image.size

        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            x_min, x_max = w - x_max, w - x_min

        if self.augment:
            image = self.color_jitter(image)

        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        bbox = np.array([x_min / w, y_min / h, x_max / w, y_max / h], dtype=np.float32)
        bbox = np.clip(bbox, 0.0, 1.0)

        if self.transform:
            image = self.transform(image)

        return image, bbox
