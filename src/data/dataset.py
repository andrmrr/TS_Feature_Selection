import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple, Optional

class TimeSeriesDataset(Dataset):
    def __init__(self, 
                 data: np.ndarray,
                 time_data: np.ndarray,
                 seq_length: int = 24,
                 feature_mask: Optional[np.ndarray] = None):
        """
        Initialize the dataset with both sequence and static data.
        
        Args:
            data: Input data array where first column is the target
            time_data: Time features array
            seq_length: Length of sequence to use for prediction
            feature_mask: Optional mask to select specific features
        """
        # Process sequence data
        self.time_feature = torch.tensor(data[:, 0], dtype=torch.float32).reshape(-1, 1)
        self.other_features = torch.tensor(data[:, 1:], dtype=torch.float32)
        if feature_mask is not None:
            self.other_features = self.other_features[:, feature_mask]
        
        # Process time data as static features
        self.time_data = torch.tensor(time_data, dtype=torch.float32)
        
        # Create static features by combining time data with current features
        self.static_data = torch.cat([
            self.time_data,
            self.other_features
        ], dim=1)
        
        self.seq_length = seq_length

    def __len__(self) -> int:
        return len(self.other_features) - self.seq_length

    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        # Get sequence data
        time_seq = self.time_feature[idx:idx + self.seq_length]
        other_seq = self.other_features[idx:idx + self.seq_length]
        seq_data = torch.cat([time_seq, other_seq], dim=1)
        
        # Get static data (current time and features)
        static_data = torch.cat([
            self.time_data[idx].reshape(1, -1),
            self.other_features[idx].reshape(1, -1)
        ], dim=1)
        
        # Get target
        target = self.time_feature[idx + self.seq_length]
        
        return (seq_data, static_data), target
