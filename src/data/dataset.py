import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple

class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time series data.
    Handles both sequence and static features.
    """
    
    def __init__(self, 
                 data: np.ndarray,
                 time_data: np.ndarray,
                 seq_length: int = 24,
                 feature_mask: torch.Tensor = None):
        """
        Initialize the dataset.
        
        Args:
            data: Main data array (n_samples, n_features)
            time_data: Time data array (n_samples, n_time_features)
            seq_length: Length of sequence to use for prediction
            feature_mask: Binary mask for feature selection
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.time_data = torch.tensor(time_data, dtype=torch.float32)
        self.seq_length = seq_length
        
        # Apply feature mask if provided
        if feature_mask is not None:
            self.data = self.data[:, feature_mask]
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of ((sequence_data, static_data), target)
            sequence_data: Shape (seq_length, n_features)
            static_data: Shape (n_time_features,)
            target: Shape (1,)
        """
        # Get sequence data
        seq_data = self.data[idx:idx + self.seq_length]
        
        # Get static data (time features)
        static_data = self.time_data[idx + self.seq_length]
        
        # Get target (next value after sequence)
        target = self.data[idx + self.seq_length, 0].reshape(1)  # Assuming first column is target
        
        return (seq_data, static_data), target 