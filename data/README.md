# Data

This directory contains data for experiments.

## Structure

```
data/
├── download_scripts/    # Scripts to download datasets
├── preprocessing/       # Preprocessing utilities
├── dataloaders/        # PyTorch dataloaders
└── raw/                # Raw data (not tracked by git)
```

## Supported Datasets

### FER2013

Facial Expression Recognition 2013 dataset.
- 35,887 grayscale images (48x48)
- 7 emotion categories
- Download from Kaggle

### AffectNet

Large-scale facial expression dataset.
- 400,000+ images
- 8 emotion categories + valence/arousal
- Request access from authors

## Usage

```python
from data.dataloaders.fer2013 import FER2013DataLoader

loader = FER2013DataLoader(
    data_dir="data/raw/fer2013",
    batch_size=64,
    split="train"
)

for images, labels in loader:
    # images: [B, 1, 48, 48]
    # labels: [B] integers 0-6
    pass
```

## Download

```bash
# FER2013 (requires Kaggle account)
python data/download_scripts/download_fer2013.py

# AffectNet (requires manual download)
# Follow instructions at https://mohammadmahoor.com/affectnet/
```
