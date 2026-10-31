import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional


class FraudTabularDataset(Dataset):
    """
    PyTorch Dataset for tabular fraud detection.
    Separates categorical attributes (for Entity Embeddings) and continuous numerical features.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        cat_cols: List[str],
        num_cols: List[str],
        target_col: Optional[str] = None
    ):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.target_col = target_col

        # Convert to contiguous numpy arrays for fast PyTorch tensor creation
        self.cat_data = df[cat_cols].to_numpy(dtype=np.int64)
        self.num_data = df[num_cols].to_numpy(dtype=np.float32)

        if target_col is not None and target_col in df.columns:
            self.targets = df[target_col].to_numpy(dtype=np.float32)
        else:
            self.targets = None

    def __len__(self) -> int:
        return len(self.cat_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        cat_x = torch.from_numpy(self.cat_data[idx])
        num_x = torch.from_numpy(self.num_data[idx])

        if self.targets is not None:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            return cat_x, num_x, target
        return cat_x, num_x, torch.tensor(-1.0, dtype=torch.float32)


def create_dataloader(
    df: pd.DataFrame,
    cat_cols: List[str],
    num_cols: List[str],
    target_col: Optional[str] = None,
    batch_size: int = 1024,
    shuffle: bool = False,
    num_workers: int = 0
) -> DataLoader:
    dataset = FraudTabularDataset(df, cat_cols, num_cols, target_col)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
