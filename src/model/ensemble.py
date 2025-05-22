import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
import pytorch_lightning as pl
import torch
from lstm import LSTMModel, TimeSeriesDataset
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger

def retrain_and_predict(train_norm_data, valid_norm_data, train_time_data, valid_time_data,
                        masks, n_time_features, static_hidden_size, seq_length, hidden_size, num_layers, 
                        max_epochs, batch_size=32, device='gpu', time_data=None):
    """
    For each mask, retrain LSTM on the full training set and generate predictions on the stacking/validation set.
    Returns: predictions (n_samples, n_models), targets (n_samples,)
    """

    
    lstm_models = []
    for i, mask in enumerate(masks):
        train_dataset = TimeSeriesDataset(
            train_norm_data, train_time_data, seq_length=seq_length,
            feature_mask=mask, static_mask=None
        )
        
        valid_dataset = TimeSeriesDataset(
            valid_norm_data, valid_time_data, seq_length=seq_length,
            feature_mask=mask, static_mask=None
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=8, persistent_workers=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)
        
        model = LSTMModel(
            lstm_input_size=mask.sum() + 1,  # +1 for target that's concatenated in __getitem__
            lstm_hidden_size=hidden_size,
            lstm_num_layers=num_layers,
            static_input_size=mask.sum() + n_time_features,  # Use actual static feature size
            static_hidden_size=static_hidden_size,
            merged_hidden_size=hidden_size,
            output_size=1
        )

        logger = TensorBoardLogger(
            save_dir='logs/model',
            name=f'masked_model_{i}',
            version=0,
        )

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            enable_checkpointing=False,
            logger=False,
            enable_model_summary=False,
            accelerator=device,
            devices=1,
            enable_progress_bar=True,
            callbacks=[],
            strategy='auto'
        )
        trainer.fit(model, train_loader, valid_loader)
        model.eval()
        lstm_models.append(model)
    return lstm_models

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

def create_datasets_stacked(lstm_models, train_norm_data, valid_norm_data, train_time_data, valid_time_data, masks, seq_length):
    # Prepare datasets and loaders
    # For train set
    train_preds = []
    train_targets = []
    valid_preds = []
    valid_targets = []
    with torch.no_grad():
        for i, model in enumerate(lstm_models):
            train_dataset = TimeSeriesDataset(train_norm_data, train_time_data, seq_length=seq_length, feature_mask=masks[i])
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=4)
            preds = []
            targets = []
            for x, y in train_loader:
                model.eval()
                out = model(x)
                preds.append(out.cpu().numpy().squeeze())
                targets.append(y.cpu().numpy().squeeze())
            train_preds.append(np.concatenate(preds).flatten())
            train_targets.append(np.concatenate(targets).flatten())
        train_preds = np.stack(train_preds, axis=1)  # shape: (n_samples, n_models)
        train_targets = np.array(train_targets[0])

    
    with torch.no_grad():
        for i, model in enumerate(lstm_models):
            valid_dataset = TimeSeriesDataset(valid_norm_data, valid_time_data, seq_length=seq_length, feature_mask=masks[i])
            valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, num_workers=4)
            preds, targets = [], []
            for x, y in valid_loader:
                model.eval()
                out = model(x)
                preds.append(out.cpu().numpy().squeeze())
                targets.append(y.cpu().numpy().squeeze())
            valid_preds.append(np.concatenate(preds).flatten())
            valid_targets.append(np.concatenate(targets).flatten())    
        valid_preds = np.stack(valid_preds, axis=1)  # shape: (n_samples, n_models)
        valid_targets = np.array(valid_targets[0])

    return train_preds, train_targets, valid_preds, valid_targets

def train_stacked_ensemble(lstm_models, train_norm_data, valid_norm_data, train_time_data,
                           valid_time_data, masks, max_epochs, batch_size, seq_length):
    train_dataset = TimeSeriesDataset(
        train_norm_data, train_time_data, seq_length=seq_length,
        feature_mask=None, static_mask=None
    )
    valid_dataset = TimeSeriesDataset(
        valid_norm_data, valid_time_data, seq_length=seq_length,
        feature_mask=None, static_mask=None
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    stacked_model = StackedLSTMModel(
        lstm_models=lstm_models,
        masks=masks,
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

    trainer.fit(stacked_model, train_loader, valid_loader)
    stacked_model.eval()
    return stacked_model


