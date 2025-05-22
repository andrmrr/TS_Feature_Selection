import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
import numpy as np
import pytorch_lightning as pl
from lstm import LSTMModel, TimeSeriesDataset
from feature_selection import run_nsga2_feature_selection
from ensemble import retrain_and_predict, train_stacked_ensemble, create_datasets_stacked
import pandas as pd
from utils import normalize_independently, load_dataset

def load_dataset(path, time_data_path):
    data = np.load(path, allow_pickle=True)
    time_data = np.load(time_data_path, allow_pickle=True)
    return data, time_data

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)
    data, time_data = load_dataset(cfg.data.path, cfg.time_data.path)
    n_features = data.shape[1] - 1
    # Split off a held-out test set (e.g., last 10%)
    N = len(data)
    test_start = int(N * 0.8)
    norm_data, norm_time_data, _, _ = normalize_independently(data, time_data)
    train_norm_data, valid_norm_data = norm_data[:test_start], norm_data[test_start:]
    train_time_data, valid_time_data = norm_time_data[:test_start], norm_time_data[test_start:]

    # train an initial model
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
        max_epochs=cfg.trainer.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else 1,
    )
    trainer.fit(model, train_loader, val_loader)

    # # 1. Run NSGA-II feature selection
    masks, objectives = run_nsga2_feature_selection(
        train_norm_data=train_norm_data,
        valid_norm_data=valid_norm_data,
        train_time_data=train_time_data,
        valid_time_data=valid_time_data,
        input_size=n_features,
        seq_length=cfg.model.seq_length,
        n_partitions=cfg.feature_selection.n_partitions,
        model=model,
        population_size=cfg.feature_selection.population_size,
        n_generations=cfg.feature_selection.n_generations,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    print(f"Found {len(masks)} Pareto-optimal feature masks.")

    print("masks:")
    for i, mask in enumerate(masks):
        print(f"Mask {i}: {mask}")
    print("Objectives:")

    # masks = [
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
    #     [1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1],
    #     [1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1],
    #     [1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1],
    #     [1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
    #     [1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1],
    #     [0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1],
    #     [1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1],
    #     [1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1],
    #     [1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1],
    #     [1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1],
    #     [1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1],
    #     [1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1],
    #     [1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1],
    #     [1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1],
    #     [1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1],
    #     [1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1],
    #     [1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1],
    #     [1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1],
    #     [1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1],
    #     [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1],
    #     [1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1],
    #     [1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1],
    #     [1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
    #     [1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1],
    #     [1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1],
    #     [1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1],
    #     [1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1],
    #     [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1],
    #     [1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1],
    #     [1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1],
    #     [1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1],
    #     [1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1],
    #     [1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1],
    #     [1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1],
    #     [1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1],
    #     [1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1],
    #     [1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1]]
    
    masks = [np.array(mask) for mask in masks]
    # 2. Retrain Pareto-optimal models and stack predictions
    lstm_models = retrain_and_predict(
        train_norm_data=train_norm_data,
        valid_norm_data=valid_norm_data,
        train_time_data=train_time_data,
        valid_time_data=valid_time_data,
        masks=masks,
        n_time_features=cfg.model.time_features_size,
        static_hidden_size=cfg.model.static_hidden_size,
        seq_length=cfg.model.seq_length,
        hidden_size=cfg.model.lstm_hidden_size,
        num_layers=cfg.model.lstm_num_layers,
        max_epochs=cfg.trainer.max_epochs,
        batch_size=256,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )

    # 3. Train stacked ensemble model

    train_preds, train_targets, valid_preds, valid_targets = create_datasets_stacked(lstm_models, train_norm_data, 
                                                                                     valid_norm_data, train_time_data, 
                                                                                     valid_time_data, masks, cfg.model.seq_length)
    
    # Save datasets to CSV
    train_preds_df = pd.DataFrame(train_preds)
    train_targets_df = pd.DataFrame(train_targets)
    valid_preds_df = pd.DataFrame(valid_preds)
    valid_targets_df = pd.DataFrame(valid_targets)

    train_preds_df.to_csv("stacked_train_preds.csv", index=False)
    train_targets_df.to_csv("stacked_train_targets.csv", index=False)
    valid_preds_df.to_csv("stacked_valid_preds.csv", index=False)
    valid_targets_df.to_csv("stacked_valid_targets.csv", index=False)

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')  # or
    main()