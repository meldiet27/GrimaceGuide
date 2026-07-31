# Local landmark model training

Trains a CNN (ResNet18 backbone) to regress the 48 CatFLW facial landmarks from a cat face crop.
This is training/experimentation code — it does not depend on and is not depended on by
`grimaceguide/`. Once a checkpoint is good enough, a future `LocalLandmarkModel` adapter under
`grimaceguide/infrastructure/` can load it and be wired in via `container.py`, matching the
existing `LandmarkAPIClient` interface. That wiring is not done yet.

## Layout

- `dataset.py` — `CatFLWDataset`, a `torch.utils.data.Dataset` over the CatFLW `images/` + `labels/`
  layout. Crops to the face bounding box (with padding) and normalizes landmark coordinates to
  `[0, 1]` relative to the crop.
- `model.py` — `LandmarkNet`, a ResNet18 with the final layer replaced by a `96`-unit regression
  head (48 landmarks x, y).
- `train.py` — CLI training loop. Saves the best (lowest val loss) checkpoint.
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
1. Zip your local `ml/data/CatFLW dataset/` folder and upload the zip to Google Drive (e.g.
   `MyDrive/grimaceguide/CatFLW_dataset.zip`).

**Each Colab session:**
```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/<your-username>/GrimaceGuide.git
%cd GrimaceGuide
!pip install -r ml/requirements.txt

!mkdir -p "ml/data/CatFLW dataset"
!unzip -q "/content/drive/MyDrive/grimaceguide/CatFLW_dataset.zip" -d "ml/data/CatFLW dataset"

!python -m ml.train \
    --data-dir "ml/data/CatFLW dataset" \
    --epochs 30 \
    --output ml/checkpoints/landmark_net.pt
```

Then copy the checkpoint back to Drive so it survives the session ending:
```python
!cp ml/checkpoints/landmark_net.pt /content/drive/MyDrive/grimaceguide/landmark_net.pt
```

If you push training-code changes from Colab, do it as normal commits from your own machine —
don't commit from the Colab environment. Treat Colab as compute only; the repo is still the
source of truth for code.
