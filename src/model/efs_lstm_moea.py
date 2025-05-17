from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn as nn
from platypus import NSGAII, Problem, Binary
import random
from torch.utils.data import DataLoader
from lstm import LSTMModel, TimeSeriesDataset, GradHistLogger



cfg=DictConfig({
    "data": {
        "path": "data/data.npy"},
    "time_data": {
            "path": "data/time_data.npy"
        }
    })
cfg.data.path = "data/data.npy"
cfg.time_data.path = "data/time_data.npy"
#Partition Data into N partitions chronologically
def load_dataset(path, time_data_path):
    data = np.load(path, allow_pickle=True)
    time_data = np.load(time_data_path, allow_pickle=True)
    N = len(data)

    train_end = int(N * 0.8)
    train_data, train_time_data = data[:train_end], time_data[:train_end]
    val_data, val_time_data = data[train_end:], time_data[train_end:]
    train_dataset = TimeSeriesDataset(train_data, train_time_data, seq_length=24)
    train_dataset[0]
    val_dataset = TimeSeriesDataset(val_data, val_time_data, seq_length=24)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    return train_loader, val_loader

def split_data_chronologically(train_data, val_data, n_partitions):
    """
    Split data and time_data into N chronological partitions.
    
    Parameters:
    data (numpy.ndarray): The main data array.
    time_data (numpy.ndarray): The corresponding time data array.
    n_partitions (int): Number of partitions to create.
    
    Returns:
    tuple: Two lists containing the partitioned data and time_data.
    """
    # Ensure data and time_data have the same length
    assert len(train_data) == len(val_data), "Data and time_data must have the same length."
    
    # Calculate the size of each partition
    partition_size = len(val_data) // n_partitions
    
    # Initialize lists to store partitions
    train_data_partitions = []
    val_data_partitions = []
    
    # Split the data into N partitions
    for i in range(n_partitions):
        start_idx = i * partition_size
        end_idx = start_idx + partition_size if i < n_partitions - 1 else len(train_data)
        
        train_data_partitions.append(train_data[start_idx:end_idx])
        val_data_partitions.append(val_data[start_idx:end_idx])
    
    return train_data_partitions, val_data_partitions

# Example usage:
cfg.data.path = "data/data.npy"
cfg.time_data.path = "data/time_data.npy"

# Load your data
data = np.load(cfg.data.path)
time_data = np.load(cfg.time_data.path)

# Specify the number of partitions
n_partitions = 5

# Split the data
data_partitions, time_data_partitions = split_data_chronologically(data, time_data, n_partitions)

# Now you can use these partitions as needed
for i, (data_part, time_part) in enumerate(zip(data_partitions, time_data_partitions)):
    print(f"Partition {i+1}:")
    print(f"  Data shape: {data_part.shape}")
    print(f"  Time data shape: {time_part.shape}")
    print(f"  Time range: {time_part[0]} to {time_part[-1]}")

#we have 31 features

# =============== Simple LSTM Model Definition ===============
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take last output
        out = self.fc(out)
        return out

# ========== Function to Train/Evaluate LSTM on One Partition ==========
def train_and_evaluate_lstm(X_train, y_train, X_val, y_val, num_epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, y_train = torch.tensor(X_train, dtype=torch.float32).to(device), torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val, y_val = torch.tensor(X_val, dtype=torch.float32).to(device), torch.tensor(y_val, dtype=torch.float32).to(device)
    model = SimpleLSTM(input_size=X_train.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    # Simple training loop (for demo)
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = loss_fn(output, y_train)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        pred = model(X_val)
        mse = loss_fn(pred, y_val).item()
        rmse = np.sqrt(mse)
    return rmse

# ============= Platypus Problem Class ==================
class LSTMFeatureSelectionProblem(Problem):
    def __init__(self, num_features, partitions):
        # Each variable is a binary: use feature or not
        # Each objective: RMSE on one partition
        super().__init__(num_features, len(partitions))
        self.types[:] = [Binary(1)] * num_features
        self.directions[:] = [Problem.MINIMIZE] * len(partitions)
        self.partitions = partitions  # list of (X_train, y_train, X_val, y_val) tuples

    def evaluate(self, solution):
        mask = np.array([int(b[0]) for b in solution.variables])
        if mask.sum() == 0:
            solution.objectives[:] = [9999.0] * len(self.partitions)
            return
        rmses = []
        for (X_train, y_train, X_val, y_val) in self.partitions:
            # Apply mask to last dimension (features)
            X_train_masked = X_train[:, :, mask == 1]
            X_val_masked = X_val[:, :, mask == 1]
            try:
                rmse = train_and_evaluate_lstm(X_train_masked, y_train, X_val_masked, y_val, num_epochs=3)
            except Exception as e:
                # fallback for training errors (e.g. mask removes all features)
                rmse = 9999.0
            rmses.append(rmse)
        solution.objectives[:] = rmses

# ============= Example: Dummy Data Preparation =============
def create_dummy_data(num_samples=100, seq_len=10, num_features=8):
    X = np.random.randn(num_samples, seq_len, num_features)
    y = np.random.randn(num_samples, 1)
    return X, y

def load_dataset(path, time_data_path, n_partitions=3, seq_length=24, batch_size=32):
    data = np.load(path, allow_pickle=True)
    time_data = np.load(time_data_path, allow_pickle=True)
    N = len(data)

    indices = np.arange(N)
    partitions = np.array_split(indices, n_partitions)
    partition_loaders = []

    for part_indices in partitions:
        part_data = data[part_indices]
        part_time_data = time_data[part_indices]
        n_samples = len(part_indices)
        n_train = int(n_samples * 0.8)
        train_data, train_time_data = part_data[:n_train], part_time_data[:n_train]
        val_data, val_time_data = part_data[n_train:], part_time_data[n_train:]

        train_dataset = TimeSeriesDataset(train_data, train_time_data, seq_length=seq_length)
        val_dataset = TimeSeriesDataset(val_data, val_time_data, seq_length=seq_length)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        partition_loaders.append((train_loader, val_loader))

    return partition_loaders

partition_loaders = load_dataset(
    cfg.data.path, 
    cfg.time_data.path, 
    n_partitions=3,        # Or set as needed
    seq_length=24,         # Or use cfg.model.seq_length if you have it
    batch_size=32          # Or use cfg.trainer.batch_size if you have it
)

# ============= Main Platypus Optimization Loop ==============
def dataloader_to_arrays(loader):
    # Flatten DataLoader batches to single (X, y) arrays
    X_list, y_list = [], []
    for batch in loader:
        X, y = batch
        if isinstance(X, tuple) or isinstance(X, list):
            X = X[0]
        X_list.append(X)
        y_list.append(y)
    X_all = torch.cat(X_list, dim=0).cpu().numpy()
    y_all = torch.cat(y_list, dim=0).cpu().numpy()
    return X_all, y_all

def main():
    num_features = 32
    n_partitions = 3

    # Get DataLoaders for each partition
    partition_loaders = load_dataset(
        cfg.data.path,
        cfg.time_data.path,
        n_partitions=n_partitions,
        seq_length=24,
        batch_size=32
    )

    # Convert DataLoaders to arrays per partition
    partitions = []
    for train_loader, val_loader in partition_loaders:
        X_train, y_train = dataloader_to_arrays(train_loader)
        X_val, y_val = dataloader_to_arrays(val_loader)
        partitions.append((X_train, y_train, X_val, y_val))

    # Now run NSGA-II feature selection as before
    problem = LSTMFeatureSelectionProblem(num_features, partitions)
    algorithm = NSGAII(problem)
    algorithm.run(10)  # Try more generations for real runs!

    print("\nPareto-optimal solutions (masks and RMSEs):")
    for solution in algorithm.result:
        mask = [int(b[0]) for b in solution.variables]
        print(f"Mask: {mask}, Objectives (RMSEs): {solution.objectives}")

if __name__ == "__main__":
    main()