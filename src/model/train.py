import hydra
from omegaconf import DictConfig
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from utils import load_dataset, normalize_independently
from torch.utils.data import DataLoader
from lstm import LSTMModel, TimeSeriesDataset


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig):
    tb_logger = TensorBoardLogger(
        save_dir=cfg.logger.save_dir,
        name=cfg.logger.name,
        version=cfg.logger.version,
    )
    pl.seed_everything(cfg.seed)

    data, time_data = load_dataset(cfg.data.path, cfg.time_data.path)
    N = len(data)
    test_start = int(N * 0.8)
    norm_data, norm_time_data, _, _ = normalize_independently(data, time_data)
    import pdb; pdb.set_trace()
    train_norm_data, valid_norm_data = norm_data[:test_start], norm_data[test_start:]
    train_time_data, valid_time_data = norm_time_data[:test_start], norm_time_data[test_start:]
    train_dataset = TimeSeriesDataset(train_norm_data, train_time_data, seq_length=cfg.model.seq_length)
    val_dataset = TimeSeriesDataset(valid_norm_data, valid_time_data, seq_length=cfg.model.seq_length)
    train_loader = DataLoader(train_dataset, batch_size=cfg.trainer.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.trainer.batch_size, shuffle=False)

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
        logger=tb_logger,
        max_epochs=cfg.trainer.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else 1,
    )
    
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
