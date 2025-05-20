import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
import torch
import pytorch_lightning as pl
from lstm import LSTMModel
from utils import create_dataloaders
import feature_selection as fs
from feature_selection import simple_forward_lstm

def build_stacking_dataset(data, pareto_models, split_range, hidden_size):
    X = data[split_range[0]:split_range[1], 1:]
    y = data[split_range[0]:split_range[1], 0]
    pred_matrix = []
    for mask, w, _ in pareto_models:
        preds = simple_forward_lstm(X[:, mask.astype(bool)], w, mask.sum(), hidden_size)
        pred_matrix.append(preds)
    return np.column_stack(pred_matrix), y


def get_predictions_from_evolved_models(data, model_tuples, stacking_range,hidden_size):
    """
    For each (mask, w) tuple, get predictions on stacking set.
    model_tuples: list of (mask, w) from NSGA-II
    stacking_range: tuple (start_idx, end_idx) for stacking/validation split
    Returns: predictions (n_samples, n_models), targets (n_samples,)
    """
    preds_list = []
    stacking_data = data[stacking_range[0]:stacking_range[1]]
    X_stack = stacking_data[:, 1:]
    y_stack = stacking_data[:, 0]

    for mask, w, _ in model_tuples:
        X_stack_masked = X_stack[:, mask.astype(bool)]
        preds = simple_forward_lstm(X_stack_masked, w, mask.sum(), hidden_size)
        preds_list.append(preds)
    predictions = np.stack(preds_list, axis=1)
    targets = y_stack
    return predictions, targets

def train_meta_learner(predictions, targets):
    """
    Train a Random Forest meta-learner to combine predictions from multiple LSTM models.
    Returns the trained model and feature importances for each LSTM model.
    """
    meta = ExtraTreesRegressor(n_estimators=100, random_state=42)
    meta.fit(predictions, targets)
    return meta

def evaluate_ensemble(meta, predictions, targets):
    ensemble_preds = meta.predict(predictions)
    rmse = np.sqrt(mean_squared_error(targets, ensemble_preds))
    return rmse

def estimate_feature_importance(pareto_models):
    """
    Estimate feature importance by how often each feature is selected in the Pareto front.
    """
    masks = [mask for mask, _, _ in pareto_models]  # Only take the first element of each tuple
    masks = np.array(masks)
    importance = masks.sum(axis=0) / len(masks)
    return importance