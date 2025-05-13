import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
import numpy as np
import pytorch_lightning as pl

from lstm import LSTMModel, TimeSeriesDataset

def load_dataset(path):
    data = np.load(path, allow_pickle=True)
    N = len(data)

    train_end = int(N * 0.8)
    train_data = data[:train_end]
    val_data = data[train_end:]
    train_dataset = TimeSeriesDataset(train_data, seq_length=24)
    val_dataset = TimeSeriesDataset(val_data, seq_length=24)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    return train_loader, val_loader


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)
    
    train_loader, val_loader = load_dataset(cfg.data.path)

    model = LSTMModel(
        input_size=cfg.model.input_size,
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
    )

    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()


    