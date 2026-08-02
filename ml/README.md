# Local landmark model training

Trains two CNNs used together by `grimaceguide/infrastructure/local_landmark_model.py`
(`LocalLandmarkModel`, wired in via `container.py`'s `landmark_source="local"` / `GG_LANDMARK_SOURCE=local`):

1. `bbox_model.py` / `train_bbox.py` — finds the cat's face in an arbitrary photo.
2. `model.py` / `train.py` — localizes 48 facial landmarks within that face crop.

Both train from the same CatFLW dataset, just against different labels (`bounding_boxes` vs.
`labels`). Neither depends on `grimaceguide/`; `grimaceguide/` depends on their checkpoints.

Predicts a spatial heatmap per landmark rather than directly regressing (x, y) coordinates: an
earlier direct-regression version collapsed on the tightly-clustered points (eyes/nose/muzzle),
predicting their mean position instead of discriminating individual points, since plain MSE over a
flat coordinate vector lets it get away with that when those points have low positional variance
across aligned crops. Heatmaps avoid this since each landmark gets its own spatial output. See
`ml/model.py` for the architecture notes.

## Layout

- `dataset.py` — `CatFLWDataset`, a `torch.utils.data.Dataset` over the CatFLW `images/` + `labels/`
  layout. Crops to the face bounding box (with padding), normalizes landmark coordinates to
  `[0, 1]` relative to the crop, and renders each sample's training target as a per-landmark
  Gaussian heatmap (`HEATMAP_STRIDE`/`HEATMAP_SIGMA` control resolution/spread).
- `model.py` — `LandmarkNet`, a ResNet18 encoder + 3-stage deconv head predicting a
  `(NUM_LANDMARKS, H, W)` heatmap stack; `decode_heatmaps` argmax-decodes those back to normalized
  `(x, y)` points.
- `train.py` — CLI training loop. Loss is MSE between predicted and target heatmaps; also logs a
  decoded pixel-error metric (mean absolute error in normalized crop units) each epoch, since raw
  heatmap-MSE isn't a human-interpretable accuracy signal on its own. Saves the best
  (lowest val loss) checkpoint.
- `bbox_dataset.py` — `CatFLWBBoxDataset`, yields the *whole* (uncropped) image plus its
  ground-truth `bounding_boxes` label normalized `[0, 1]`. Unlike landmarks, a bounding box has no
  left/right semantic identity, so random horizontal flip is safe here as augmentation.
- `bbox_model.py` — `BBoxNet`, a ResNet18 with a 4-output sigmoid-activated regression head
  (`x_min, y_min, x_max, y_max`, normalized).
- `train_bbox.py` — CLI training loop. Loss is `SmoothL1Loss`; logs mean IoU between predicted and
  ground-truth boxes each epoch as the interpretable accuracy signal.
- `data/` — gitignored. Put the extracted CatFLW dataset here as `ml/data/CatFLW dataset/`
  (containing `images/` and `labels/`), or point `--data-dir` anywhere else.
- `checkpoints/` — gitignored. Trained weights land here by default.

  Two generations live side by side. **`*_subjsplit.pt` are the ones to use** — trained under
  `dataset.py::subject_disjoint_split` and wired as the defaults in `grimaceguide/container.py`.
  The plain `landmark_net.pt` / `bbox_net.pt` are older checkpoints trained with an index-level
  split that leaked cats across train/val (310 of 311 val images had a same-cat photo in
  training), so their held-out metrics were measured on cats they had already seen. They are
  equivalent in accuracy — the leak inflated the *metrics*, not the model — but nothing measured
  on them can be trusted. Kept only for comparison; see `docs/heatmap_decoding.md`.

  Checkpoints written since that fix embed their own `split` config, which
  `compare_predicted_geometry.py` reads back so evaluation can't silently score training images.
  A quick way to tell them apart:

  ```bash
  python -c "import torch; print(torch.load('ml/checkpoints/landmark_net_subjsplit.pt', map_location='cpu')['split'])"
  ```

## Running locally

```bash
pip install -r ml/requirements.txt
python -m ml.train_bbox --data-dir "ml/data/CatFLW dataset" --output ml/checkpoints/bbox_net.pt
python -m ml.train --data-dir "ml/data/CatFLW dataset" --output ml/checkpoints/landmark_net.pt
```

Add `--device cpu` if you don't have a CUDA GPU locally (slow, but fine for a smoke test with
`--epochs 1`).

## Running in Google Colab

The dataset (~1.4GB) and the code are handled separately: code comes from the repo, data comes
from Drive, so you're not re-uploading either one from scratch each session.

**One-time setup:**
1. Zip your local `ml/data/CatFLW dataset/` folder and upload the zip to the top level of your
   Google Drive (`MyDrive/CatFLW dataset.zip`). Not a subfolder — `!ls /content/drive/MyDrive/` to
   confirm the exact filename before running the unzip step below, since a mismatch fails silently
   with a "cannot find" error rather than an obvious one.

**Each Colab session:**
```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/meldiet27/GrimaceGuide.git
%cd GrimaceGuide
!git log -1 --oneline   # sanity check: confirms which commit you're training against
!pip install -r ml/requirements.txt

!mkdir -p "ml/data/CatFLW dataset"
!unzip -q "/content/drive/MyDrive/CatFLW dataset.zip" -d "ml/data/CatFLW dataset"

!python -m ml.train_bbox \
    --data-dir "ml/data/CatFLW dataset" \
    --epochs 30 \
    --output ml/checkpoints/bbox_net.pt

!python -m ml.train \
    --data-dir "ml/data/CatFLW dataset" \
    --epochs 30 \
    --output ml/checkpoints/landmark_net.pt
```

`/content` is ephemeral — it's wiped if the runtime disconnects or restarts. As soon as training
finishes, immediately copy both checkpoints back to Drive in the same runtime, before doing
anything else:
```python
!cp ml/checkpoints/bbox_net.pt "/content/drive/MyDrive/bbox_net.pt"
!cp ml/checkpoints/landmark_net.pt "/content/drive/MyDrive/landmark_net.pt"
```

If you push training-code changes from Colab, do it as normal commits from your own machine —
don't commit from the Colab environment. Treat Colab as compute only; the repo is still the
source of truth for code.
