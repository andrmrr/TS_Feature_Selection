import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple

class TimeSeriesDataset(Dataset):
    def __init__(self, 
                 data: np.ndarray,
                 time_data: np.ndarray,
                 seq_length: int = 24):
        # No feature mask!
        self.time_feature = torch.tensor(data[:, 0], dtype=torch.float32).reshape(-1, 1)
        self.other_features = torch.tensor(data[:, 1:], dtype=torch.float32)
        self.time_data = torch.tensor(time_data, dtype=torch.float32)
        self.seq_length = seq_length

    def __len__(self) -> int:
        return len(self.other_features) - self.seq_length

    def __getitem__(self, idx: int):
        time_seq = self.time_feature[idx:idx + self.seq_length]
        other_seq = self.other_features[idx:idx + self.seq_length]
        seq_data = torch.cat([time_seq, other_seq], dim=1)
        static_data = torch.zeros(0, dtype=torch.float32)  # Empty tensor with 0 features
        target = self.other_features[idx + self.seq_length, 0].reshape(1)
        return (seq_data, static_data), target
