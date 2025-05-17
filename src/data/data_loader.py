import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from dataclasses import dataclass
from pathlib import Path
from hydra.utils import to_absolute_path




@dataclass
class DataPartition:
    """Represents a single partition of the time series data."""
    train_data: np.ndarray
    train_time: np.ndarray
    val_data: np.ndarray
    val_time: np.ndarray
    
    @property
    def train_size(self) -> int:
        return len(self.train_data)
    
    @property
    def val_size(self) -> int:
        return len(self.val_data)

class TimeSeriesDataLoader:
    """Handles loading and partitioning of time series data."""
    
    def __init__(self, data_path: str, time_data_path: str):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the main data file (numpy array)
            time_data_path: Path to the time data file (numpy array)
        """
        data_path = to_absolute_path('data/data.npy')
        time_data_path = to_absolute_path('data/time_data.npy')
        self.data_path = Path(data_path)
        self.time_data_path = Path(time_data_path)
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load the data and time arrays."""
        data = np.load(self.data_path, allow_pickle=True)
        time_data = np.load(self.time_data_path, allow_pickle=True)
        return data, time_data
    
    def create_partitions(self, 
                         data: np.ndarray, 
                         time_data: np.ndarray, 
                         n_partitions: int,
                         train_val_split: float = 0.8) -> List[DataPartition]:
        """
        Create chronological partitions of the data.
        
        Args:
            data: Main data array
            time_data: Time data array
            n_partitions: Number of partitions to create
            train_val_split: Proportion of data to use for training (default: 0.8)
            
        Returns:
            List of DataPartition objects
        """
        assert len(data) == len(time_data), "Data and time_data must have the same length"
        
        # Calculate partition sizes
        total_samples = len(data)
        partition_size = total_samples // n_partitions
        
        partitions = []
        
        for i in range(n_partitions):
            start_idx = i * partition_size
            end_idx = start_idx + partition_size if i < n_partitions - 1 else total_samples
            
            # Get partition data
            partition_data = data[start_idx:end_idx]
            partition_time = time_data[start_idx:end_idx]
            
            # Split into train/val
            train_size = int(len(partition_data) * train_val_split)
            
            train_data = partition_data[:train_size]
            train_time = partition_time[:train_size]
            val_data = partition_data[train_size:]
            val_time = partition_time[train_size:]
            
            partitions.append(DataPartition(
                train_data=train_data,
                train_time=train_time,
                val_data=val_data,
                val_time=val_time
            ))
        
        return partitions
    
    def get_feature_names(self) -> List[str]:
        """Get the names of features in the dataset."""
        # This is a placeholder - implement based on your actual data structure
        # You might need to load this from a separate file or infer from the data
        return [f"feature_{i}" for i in range(31)]  # Assuming 31 features as mentioned in the code
    
    def get_feature_importance(self, feature_masks: List[np.ndarray]) -> Dict[str, float]:
        """
        Calculate feature importance based on frequency of selection in Pareto-optimal solutions.
        
        Args:
            feature_masks: List of binary feature masks from Pareto-optimal solutions
            
        Returns:
            Dictionary mapping feature names to their importance scores
        """
        feature_names = self.get_feature_names()
        n_solutions = len(feature_masks)
        
        # Calculate frequency of selection for each feature
        importance_scores = {}
        for i, name in enumerate(feature_names):
            frequency = sum(mask[i] for mask in feature_masks) / n_solutions
            importance_scores[name] = frequency
            
        return importance_scores 