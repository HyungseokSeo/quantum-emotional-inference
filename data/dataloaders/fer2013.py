"""FER2013 dataloader."""
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple


class FER2013Dataset(Dataset):
    """FER2013 Facial Expression Recognition dataset."""
    
    EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform = None
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.samples = []
        
        # Load data (placeholder - actual implementation would load CSV)
        self._load_data()
    
    def _load_data(self):
        """Load dataset from disk."""
        # Placeholder - implement actual loading
        pass
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, label = self.samples[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def FER2013DataLoader(
    data_dir: str,
    batch_size: int = 64,
    split: str = "train",
    num_workers: int = 4,
    transform = None
) -> DataLoader:
    """Create FER2013 DataLoader."""
    dataset = FER2013Dataset(data_dir, split, transform)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True
    )
