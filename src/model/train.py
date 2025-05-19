import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
import numpy as np
import pytorch_lightning as pl
from lstm import LSTMModel, TimeSeriesDataset
from feature_selection import run_nsga2_feature_selection
from ensemble import retrain_and_predict, train_meta_learner, evaluate_ensemble, estimate_feature_importance
import pandas as pd

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
    test_start = int(N * 0.9)
    data_main = data[:test_start]
    test_data = data[test_start:]
    time_data_main = time_data[:test_start]
    time_test_data = time_data[test_start:]

    # 1. Run NSGA-II feature selection
    masks, objectives = run_nsga2_feature_selection(
        data_main,
        n_partitions=cfg.feature_selection.n_partitions,
        seq_length=cfg.model.seq_length,
        input_size=n_features,
        hidden_size=cfg.model.lstm_hidden_size,
        num_layers=cfg.model.lstm_num_layers,
        max_epochs=cfg.trainer.max_epochs,
        batch_size=cfg.trainer.batch_size,
        population_size=cfg.feature_selection.population_size,
        n_generations=cfg.feature_selection.n_generations,
        device='gpu',
    )
    print(f"Found {len(masks)} Pareto-optimal feature masks.")

    # 2. Retrain Pareto-optimal models and stack predictions
    predictions, stack_targets = retrain_and_predict(
        data_main, masks,
        seq_length=cfg.model.seq_length,
        hidden_size=cfg.model.lstm_hidden_size,
        num_layers=cfg.model.lstm_num_layers,
        max_epochs=cfg.trainer.max_epochs,
        batch_size=cfg.trainer.batch_size,
        device='gpu',
        time_data=time_data_main  # Pass static data
    )

    # 3. Train meta-learner
    meta = train_meta_learner(predictions, stack_targets)
    
    # Save meta-learner feature importances
    meta_importance_df = pd.DataFrame({
        'lstm_model_index': range(len(meta.feature_importances_)),
        'meta_importance': meta.feature_importances_
    })
    meta_importance_df.to_csv('meta_learner_importances.csv', index=False)
    print(f"\nMeta-learner feature importances saved to 'meta_learner_importances.csv'")

    # 4. Estimate feature importance
    importance = estimate_feature_importance(masks)
    print("Feature importance (frequency of selection):")
    for i, imp in enumerate(importance):
        print(f"Feature {i}: {imp:.2f}")
    
    # Save feature importances to CSV
    importance_df = pd.DataFrame({
        'feature_index': range(len(importance)),
        'importance': importance
    })
    importance_df.to_csv('feature_importances.csv', index=False)
    print(f"\nFeature importances saved to 'feature_importances.csv'")

    # 5. Evaluate ensemble on held-out test set
    # Prepare test set predictions from each Pareto model
    from utils import create_dataloaders
    N_test = len(test_data)
    test_preds_list = []
    for mask in masks:
        # Use all data_main for training, test_data for testing
        train_loader, test_loader = create_dataloaders(
            data_main, time_data_main,  # Use actual static data instead of dummy
            test_data, time_test_data,  # Use actual static data instead of dummy
            cfg.model.seq_length, cfg.trainer.batch_size, 
            feature_mask=mask.astype(bool), num_workers=2
        )
        model = LSTMModel(
            lstm_input_size=mask.sum() + 1,  # +1 for target that's concatenated in __getitem__
            lstm_hidden_size=cfg.model.lstm_hidden_size,
            lstm_num_layers=cfg.model.lstm_num_layers,
            static_input_size=time_data.shape[1],  # Use actual static feature size
            static_hidden_size=cfg.model.lstm_hidden_size,
            merged_hidden_size=cfg.model.lstm_hidden_size,
            output_size=cfg.model.output_size
        )
        trainer = pl.Trainer(
            max_epochs=cfg.trainer.max_epochs,
            enable_checkpointing=False,
            logger=False,
            enable_model_summary=False,
            accelerator='gpu',
            devices=1,
            enable_progress_bar=False,
            callbacks=[],
            strategy='auto'
        )
        trainer.fit(model, train_loader, test_loader)
        model.eval()
        preds = []
        with torch.no_grad():
            for x, y in test_loader:
                out = model(x)
                preds.append(out.cpu().numpy())
        preds = np.concatenate(preds).flatten()
        test_preds_list.append(preds)
    test_predictions = np.stack(test_preds_list, axis=1)
    # Get test targets
    _, test_loader = create_dataloaders(
        data_main, time_data_main,  # Use actual static data
        test_data, time_test_data,  # Use actual static data
        cfg.model.seq_length, cfg.trainer.batch_size, 
        feature_mask=np.ones(n_features, dtype=bool), num_workers=2
    )
    test_targets = []
    for _, y in test_loader:
        test_targets.append(y.cpu().numpy())
    test_targets = np.concatenate(test_targets).flatten()
    # Evaluate ensemble
    test_rmse = evaluate_ensemble(meta, test_predictions, test_targets)
    print(f"Stacked ensemble test RMSE: {test_rmse:.4f}")

    # Save RMSE to the same CSV file
    rmse_df = pd.DataFrame({
        'metric': ['test_rmse'],
        'value': [test_rmse]
    })
    rmse_df.to_csv('feature_importances.csv', mode='a', header=False, index=False)
    print(f"Test RMSE saved to 'feature_importances.csv'")

if __name__ == "__main__":
    main()


    