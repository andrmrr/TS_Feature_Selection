import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import torch
import pytorch_lightning as pl
from lstm import LSTMModel
from utils import create_dataloaders

def retrain_and_predict(data, masks, seq_length, hidden_size, num_layers, max_epochs, batch_size=32, device='gpu'):
    """
    For each mask, retrain LSTM on the full training set and generate predictions on the stacking/validation set.
    Returns: predictions (n_samples, n_models), targets (n_samples,)
    """
    # Split data into train/stack (e.g., 80/20)
    N = len(data)
    train_end = int(N * 0.8)
    train_data = data[:train_end]
    stack_data = data[train_end:]
    preds_list = []
    for mask in masks:
        train_loader, stack_loader = create_dataloaders(
            train_data, stack_data, seq_length, batch_size, feature_mask=mask.astype(bool)
        )
        model = LSTMModel(
            input_size=mask.sum(),
            hidden_size=hidden_size,
            num_layers=num_layers
        )
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            enable_checkpointing=False,
            logger=False,
            enable_model_summary=False,
            accelerator='gpu',
            devices=1,
            enable_progress_bar=True,
            strategy='auto'
        )
        trainer.fit(model, train_loader, stack_loader)
        model.eval()
        preds = []
        with torch.no_grad():
            for x, y in stack_loader:
                out = model(x)
                preds.append(out.cpu().numpy())
        preds = np.concatenate(preds).flatten()
        preds_list.append(preds)
    # All models predict on the same stacking set
    predictions = np.stack(preds_list, axis=1)
    # Get targets from stacking set
    _, stack_loader = create_dataloaders(
        train_data, stack_data, seq_length, batch_size, feature_mask=np.ones(data.shape[1]-1, dtype=bool)
    )
    targets = []
    for _, y in stack_loader:
        targets.append(y.cpu().numpy())
    targets = np.concatenate(targets).flatten()
    return predictions, targets

def train_meta_learner(predictions, targets):
    meta = LinearRegression()
    meta.fit(predictions, targets)
    return meta

def evaluate_ensemble(meta, predictions, targets):
    ensemble_preds = meta.predict(predictions)
    rmse = np.sqrt(mean_squared_error(targets, ensemble_preds))
    return rmse

def estimate_feature_importance(masks):
    """
    Estimate feature importance by frequency of selection in Pareto-optimal masks.
    Returns: array of shape (n_features,)
    """
    masks = np.array(masks)
    importance = masks.sum(axis=0) / masks.shape[0]
    return importance 