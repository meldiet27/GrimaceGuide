# Local landmark model training

Trains a CNN (ResNet18 backbone + heatmap head) to localize the 48 CatFLW facial landmarks from a
cat face crop. This is training/experimentation code — it does not depend on and is not depended
on by `grimaceguide/`. Once a checkpoint is good enough, a future `LocalLandmarkModel` adapter
under `grimaceguide/infrastructure/` can load it and be wired in via `container.py`, matching the
existing `LandmarkAPIClient` interface. That wiring is not done yet.

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
- `data/` — gitignored. Put the extracted CatFLW dataset here as `ml/data/CatFLW dataset/`
  (containing `images/` and `labels/`), or point `--data-dir` anywhere else.
- `checkpoints/` — gitignored. Trained weights land here by default.

## Running locally

```bash
pip install -r ml/requirements.txt
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

!python -m ml.train \
    --data-dir "ml/data/CatFLW dataset" \
    --epochs 30 \
    --output ml/checkpoints/landmark_net.pt
```

`/content` is ephemeral — it's wiped if the runtime disconnects or restarts. As soon as training
finishes, immediately copy the checkpoint back to Drive in the same runtime, before doing anything
else:
```python
!cp ml/checkpoints/landmark_net.pt "/content/drive/MyDrive/landmark_net.pt"
```

If you push training-code changes from Colab, do it as normal commits from your own machine —
don't commit from the Colab environment. Treat Colab as compute only; the repo is still the
source of truth for code.
