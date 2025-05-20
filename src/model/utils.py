import numpy as np
from torch.utils.data import DataLoader
from lstm import TimeSeriesDataset
from sklearn.preprocessing import MinMaxScaler

def load_dataset_2(path, time_data_path):
    data = np.load(path, allow_pickle=True)
    time_data = np.load(time_data_path, allow_pickle=True)
    N = len(data)

    train_end = int(N * 0.8)
    train_data, train_time_data = data[:train_end], time_data[:train_end]
    val_data, val_time_data = data[train_end:], time_data[train_end:]
    train_dataset = TimeSeriesDataset(train_data, train_time_data, seq_length=24)
    val_dataset = TimeSeriesDataset(val_data, val_time_data, seq_length=24)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    return train_loader, val_loader

def normalize_independently(train_data, test_data):
    """
    Normalize train and test independently (like in the paper).
    Each gets its own MinMaxScaler fitted to its data.
    """
    scaler_train = MinMaxScaler()
    scaler_test = MinMaxScaler()
    train_data[:, 1:] = scaler_train.fit_transform(train_data[:, 1:])
    test_data[:, 1:] = scaler_test.fit_transform(test_data[:, 1:])
    return train_data, test_data, scaler_train, scaler_test

def load_dataset(path, time_data_path):
    data = np.load(path, allow_pickle=True)
    time_data = np.load(time_data_path, allow_pickle=True)
    return data, time_data

def partition_time_series(data, n_partitions):
    N = len(data)
    partition_size = N // n_partitions
    partitions = []
    for i in range(n_partitions):
        start = i * partition_size
        end = (i + 1) * partition_size if i < n_partitions - 1 else N
        part = data[start:end]
        scaler = MinMaxScaler()
        part = scaler.fit_transform(part)
        partitions.append(part)
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

def create_dataloaders(
    train_sequence_data, train_static_data,
    val_sequence_data, val_static_data,
    seq_length, batch_size=32, feature_mask=None, static_mask=None, num_workers=2
):
    # Create empty static features if not provided
    if train_static_data is None:
        train_static_data = np.zeros((len(train_sequence_data), 1))  # Single dummy static feature
    if val_static_data is None:
        val_static_data = np.zeros((len(val_sequence_data), 1))  # Single dummy static feature

    train_dataset = TimeSeriesDataset(
        train_sequence_data, train_static_data, seq_length,
        feature_mask=feature_mask, static_mask=static_mask
    )
    val_dataset = TimeSeriesDataset(
        val_sequence_data, val_static_data, seq_length,
        feature_mask=feature_mask, static_mask=static_mask
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        persistent_workers=True
    )
    return train_loader, val_loader