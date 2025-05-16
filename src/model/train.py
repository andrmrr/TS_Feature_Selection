import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from lstm import LSTMModel, TimeSeriesDataset, GradHistLogger

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


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig):
    tb_logger = TensorBoardLogger(
        save_dir=cfg.logger.save_dir,
        name=cfg.logger.name,
        version=cfg.logger.version,
    )
    pl.seed_everything(cfg.seed)
    
    train_loader, val_loader = load_dataset(cfg.data.path, cfg.time_data.path)

    model = LSTMModel(
        lstm_input_size=cfg.model.lstm_input_size,
        lstm_hidden_size=cfg.model.lstm_hidden_size,
        lstm_num_layers=cfg.model.lstm_num_layers,
        static_input_size=cfg.model.static_input_size,
        static_hidden_size=cfg.model.static_hidden_size,
        merged_hidden_size=cfg.model.merged_hidden_size,
        output_size=cfg.model.output_size
    )

    trainer = pl.Trainer(
        callbacks=[GradHistLogger()],
        logger=tb_logger,
        max_epochs=cfg.trainer.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else 1,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
    )

    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()


    