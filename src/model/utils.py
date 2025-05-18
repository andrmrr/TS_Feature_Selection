import numpy as np
from torch.utils.data import DataLoader
from lstm import TimeSeriesDataset

def partition_time_series(data, n_partitions):
    """
    Split data chronologically into n_partitions (non-overlapping).
    Returns a list of arrays, one per partition.
    """
    N = len(data)
    partition_size = N // n_partitions
    partitions = []
    for i in range(n_partitions):
        start = i * partition_size
        end = (i + 1) * partition_size if i < n_partitions - 1 else N
        partitions.append(data[start:end])
    return partitions

def split_train_val(partition, train_ratio=0.8):
    """
    Split a partition into train/val sets chronologically.
    """
    N = len(partition)
    train_end = int(N * train_ratio)
    train_data = partition[:train_end]
    val_data = partition[train_end:]
    return train_data, val_data

def create_dataloaders(train_data, val_data, seq_length, batch_size=32, feature_mask=None, num_workers=4):
    train_dataset = TimeSeriesDataset(train_data, seq_length, feature_mask=feature_mask)
    val_dataset = TimeSeriesDataset(val_data, seq_length, feature_mask=feature_mask)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader 