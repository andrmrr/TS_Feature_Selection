import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
import numpy as np
import pytorch_lightning as pl
from lstm import LSTMModel, TimeSeriesDataset
from feature_selection import run_nsga2_feature_selection, simple_forward_lstm
from ensemble import get_predictions_from_evolved_models, train_meta_learner, evaluate_ensemble, estimate_feature_importance
import pandas as pd
import copy
from sklearn.preprocessing import MinMaxScaler



def normalize_independently(train_data, test_data):
    """
    Normalize train and test independently (like in the paper).
    Each gets its own MinMaxScaler fitted to its data.
    """
    scaler_train = MinMaxScaler()
    scaler_test = MinMaxScaler()
    train_data_norm = scaler_train.fit_transform(train_data)
    test_data_norm = scaler_test.fit_transform(test_data)
    return train_data_norm, test_data_norm, scaler_train, scaler_test


def recursive_multistep_forecast(test_data, pareto_models, meta, h, hidden_size):
    """
    Recursive multi-step ahead forecasting for each sample in the test set.
    Args:
        test_data: np.array, shape (N_test, n_features + 1)
        pareto_models: list of (mask, w, _)
        meta: trained meta-learner
        h: number of steps ahead to forecast
        hidden_size: hidden size for LSTM
        
    Returns:
        preds: array, shape (N_test - h, h)
        targets: array, shape (N_test - h, h)
    """
    X_full = test_data[:, 1:]
    y_full = test_data[:, 0]
    N = len(test_data)
    preds = []
    targets = []

    for i in range(N - h):   # <---- ONLY GO TO N - h
        X_input = X_full[i, :].reshape(1, -1)
        step_preds = []
        step_targets = y_full[i+1 : i+h+1]  # always length h

        for t in range(h):
            pareto_preds = []
            for mask, w, _ in pareto_models:
                X_masked = X_input[:, mask.astype(bool)]
                pred = simple_forward_lstm(X_masked, w, mask.sum(), hidden_size)
                pareto_preds.append(pred[0])
            pareto_preds = np.array(pareto_preds).reshape(1, -1)
            ensemble_pred = meta.predict(pareto_preds)[0]
            step_preds.append(ensemble_pred)

            # Prepare input for next step (simplest: use next row of test data)
            if i + t + 1 < N:
                X_input = X_full[i + t + 1, :].reshape(1, -1)
            else:
                break

        preds.append(step_preds)
        targets.append(step_targets)

    preds = np.array(preds)
    targets = np.array(targets)
    return preds, targets


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
    
    data_main, test_data, scaler_train, scaler_test = normalize_independently(data_main, test_data)
    
    time_data_main = time_data[:test_start]
    time_test_data = time_data[test_start:]

    # 1. Run NSGA-II feature selection
    pareto_models = run_nsga2_feature_selection(
        data_main,
        n_partitions=cfg.feature_selection.n_partitions,
        seq_length=cfg.model.seq_length,
        input_size=n_features,
        hidden_size=cfg.model.lstm_hidden_size,
        #max_epochs=cfg.trainer.max_epochs,
        #batch_size=cfg.trainer.batch_size,
        population_size=cfg.feature_selection.population_size,
        n_generations=cfg.feature_selection.n_generations,
        #device='gpu',
    )
    print(f"Found {len(pareto_models)} Pareto-optimal feature masks.")

    # 2. Retrain Pareto-optimal models and stack predictions
    stack_start = int(len(data_main) * 0.8)
    stacking_range = (stack_start, len(data_main))
    predictions, stack_targets = get_predictions_from_evolved_models(
        data_main, pareto_models, stacking_range, cfg.model.lstm_hidden_size
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
    importance = estimate_feature_importance(pareto_models)
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
    X_test = test_data[:, 1:]
    y_test = test_data[:, 0]
    test_preds_list = []
    for mask, w, _ in pareto_models:
        X_test_masked = X_test[:, mask.astype(bool)]
        preds = simple_forward_lstm(X_test_masked, w, mask.sum(), hidden_size=cfg.model.lstm_hidden_size)
        test_preds_list.append(preds)
    test_predictions = np.stack(test_preds_list, axis=1)
    test_targets = y_test
    
    # Evaluate ensemble
    test_rmse = evaluate_ensemble(meta, test_predictions, test_targets)
    print(f"Stacked ensemble test RMSE: {test_rmse:.4f}")
    
        # ---- MULTI-STEP RECURSIVE FORECAST ----
    h = 3  # Number of steps ahead to forecast, match the paper
    preds, targets = recursive_multistep_forecast(
        test_data, pareto_models, meta, h, hidden_size=cfg.model.lstm_hidden_size
    )

    # Compute RMSE for each step
    ms_rmses = []
    for step in range(h):
        step_preds = preds[:, step]
        step_targets = targets[:, step]
        rmse = np.sqrt(np.mean((step_preds - step_targets) ** 2))
        ms_rmses.append(rmse)
        print(f"Multi-step RMSE at step {step+1}: {rmse:.4f}")

    # Optionally save results
    ms_rmse_df = pd.DataFrame({
        'step': list(range(1, h+1)),
        'multi_step_rmse': ms_rmses
    })
    ms_rmse_df.to_csv('multi_step_rmse.csv', index=False)
    print(f"Multi-step RMSEs saved to 'multi_step_rmse.csv'")


    # Save RMSE to the same CSV file
    rmse_df = pd.DataFrame({
        'metric': ['test_rmse'],
        'value': [test_rmse]
    })
    rmse_df.to_csv('feature_importances.csv', mode='a', header=False, index=False)
    print(f"Test RMSE saved to 'feature_importances.csv'")

if __name__ == "__main__":
    main()


    