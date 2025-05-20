import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
import torch
import pytorch_lightning as pl
from lstm import LSTMModel
from utils import create_dataloaders

def retrain_and_predict(data, masks, seq_length, hidden_size, num_layers, max_epochs, batch_size=32, device='gpu', time_data=None):
    """
    For each mask, retrain LSTM on the full training set and generate predictions on the stacking/validation set.
    Returns: predictions (n_samples, n_models), targets (n_samples,)
    """
    # Split data into train/stack (e.g., 80/20)
    N = len(data)
    train_end = int(N * 0.8)
    train_data = data[:train_end]
    stack_data = data[train_end:]
    
    # Split static data if provided
    if time_data is not None:
        train_static_data = time_data[:train_end]
        stack_static_data = time_data[train_end:]
    else:
        train_static_data = np.zeros((len(train_data), 1))
        stack_static_data = np.zeros((len(stack_data), 1))
    
    preds_list = []
    for mask in masks:
        train_loader, stack_loader = create_dataloaders(
            train_data, train_static_data,
            stack_data, stack_static_data,
            seq_length, batch_size, feature_mask=mask.astype(bool)
        )
        model = LSTMModel(
            lstm_input_size=mask.sum() + 1,  # +1 for target that's concatenated in __getitem__
            lstm_hidden_size=hidden_size,
            lstm_num_layers=num_layers,
            static_input_size=train_static_data.shape[1],  # Use actual static feature size
            static_hidden_size=hidden_size,
            merged_hidden_size=hidden_size,
            output_size=1
        )
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            enable_checkpointing=False,
            logger=False,
            enable_model_summary=False,
            accelerator='gpu',
            devices=1,
            enable_progress_bar=False,
            callbacks=[],
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
        train_data, train_static_data,
        stack_data, stack_static_data, 
        seq_length, batch_size, 
        feature_mask=np.ones(data.shape[1]-1, dtype=bool)
    )
    targets = []
    for _, y in stack_loader:
        targets.append(y.cpu().numpy())
    targets = np.concatenate(targets).flatten()
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

def estimate_feature_importance(masks):
    """
    Estimate feature importance by frequency of selection in Pareto-optimal masks.
    Returns: array of shape (n_features,)
    """
    masks = np.array(masks)
    importance = masks.sum(axis=0) / masks.shape[0]
    return importance 