"""Visualize a trained LandmarkNet's predictions against ground truth.

Picks a handful of images from the dataset, runs inference, and draws both the
predicted landmarks (red) and ground-truth landmarks (green) on the face crop so you
can eyeball prediction quality before wiring the model into the app.

Usage:
    python -m ml.visualize \
        --data-dir "ml/data/CatFLW dataset/CatFLW dataset" \
        --checkpoint ml/checkpoints/landmark_net.pt \
        --output-dir ml/predictions \
        --num-samples 8
"""
from __future__ import annotations

import argparse
import random

import torch
from PIL import ImageDraw
from torchvision import transforms
from pathlib import Path

from ml.dataset import CatFLWDataset
from ml.model import LandmarkNet

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def denormalize(tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def draw_points(image, points_xy, color, radius=3):
    draw = ImageDraw.Draw(image)
    w, h = image.size
    for x, y in points_xy:
        px, py = x * w, y * h
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="ml/predictions")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    image_size = checkpoint.get("image_size", 256)

    normalize = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )
    dataset = CatFLWDataset(args.data_dir, image_size=image_size, transform=normalize)

    model = LandmarkNet(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    random.seed(args.seed)
    indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for i in indices:
            image_tensor, target = dataset[i]
            pred = model(image_tensor.unsqueeze(0).to(device)).cpu().squeeze(0).numpy()

            display_image = transforms.ToPILImage()(denormalize(image_tensor))
            draw_points(display_image, target.reshape(-1, 2), color="lime")
            draw_points(display_image, pred.reshape(-1, 2), color="red")

            out_path = output_dir / f"sample_{i}.png"
            display_image.save(out_path)
            print(f"wrote {out_path}  (green=ground truth, red=predicted)")


if __name__ == "__main__":
    main()
