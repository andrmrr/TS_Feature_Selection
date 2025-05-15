import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from lstm import LSTMModel, TimeSeriesDataset, GradHistLogger

def load_dataset(path):
    data = np.load(path, allow_pickle=True)
    _, counts = np.unique(data[:, 0], return_counts=True)
    pos_weights = counts[0] / counts[1]
    N = len(data)

    train_end = int(N * 0.8)
    train_data = data[:train_end]
    val_data = data[train_end:]
    train_dataset = TimeSeriesDataset(train_data, seq_length=24)
    val_dataset = TimeSeriesDataset(val_data, seq_length=24)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    return train_loader, val_loader, pos_weights


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig):
    tb_logger = TensorBoardLogger(
        save_dir=cfg.logger.save_dir,
        name=cfg.logger.name,
        version=cfg.logger.version,
    )
    pl.seed_everything(cfg.seed)
    
    train_loader, val_loader, pos_weights = load_dataset(cfg.data.path)

    model = LSTMModel(
        input_size=cfg.model.input_size,
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        pos_weight=pos_weights,
    )

    trainer = pl.Trainer(
        callbacks=[GradHistLogger()],
        logger=tb_logger,
        max_epochs=cfg.trainer.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
    )

    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()


    