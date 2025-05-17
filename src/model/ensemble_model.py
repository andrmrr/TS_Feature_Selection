from typing import List, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from omegaconf import DictConfig
from .feature_selection_lstm import FeatureSelectionLSTM
import pytorch_lightning as pl
import logging

class EnsembleModel(pl.LightningModule):
    """
    Ensemble model that combines predictions from multiple LSTM models
    with different feature masks.
    """
    
    def __init__(self, config, base_model: FeatureSelectionLSTM, pareto_front: List[torch.Tensor]):
        """
        Initialize the ensemble model.
        
        Args:
            config: Configuration dictionary
            base_model: Base LSTM model to use as template
            pareto_front: List of feature masks from Pareto front
        """
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Create base models with different feature masks
        self.base_models = nn.ModuleList()
        for mask in pareto_front:
            model = FeatureSelectionLSTM(config)
            model.load_state_dict(base_model.state_dict())
            model.set_feature_mask(torch.tensor(mask, dtype=torch.bool))
            self.base_models.append(model)
        
        # Meta-learner to combine predictions
        self.meta_learner = nn.Sequential(
            nn.Linear(len(pareto_front), config.model.ensemble.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.model.ensemble.dropout),
            nn.Linear(config.model.ensemble.hidden_size, 1)
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
    
    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the ensemble.
        
        Args:
            x: Tuple of (sequence_data, static_data)
               - sequence_data: Tensor of shape (batch_size, seq_length, n_features)
               - static_data: Tensor of shape (batch_size, n_static_features)
        
        Returns:
            Tensor of shape (batch_size, 1) containing ensemble predictions
        """
        # Get predictions from each base model
        base_predictions = []
        for model in self.base_models:
            pred = model(x)
            base_predictions.append(pred)
        
        # Stack predictions
        stacked_preds = torch.stack(base_predictions, dim=1)
        
        # Combine predictions using meta-learner
        return self.meta_learner(stacked_preds)
    
    def training_step(self, batch: Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step.
        
        Args:
            batch: Tuple of (x, y) where x is a tuple of (sequence_data, static_data)
            batch_idx: Index of the current batch
        
        Returns:
            Loss tensor
        """
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch: Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], batch_idx: int) -> None:
        """
        Validation step.
        
        Args:
            batch: Tuple of (x, y) where x is a tuple of (sequence_data, static_data)
            batch_idx: Index of the current batch
        """
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)
        
        # Calculate RMSE
        rmse = torch.sqrt(torch.mean((y_hat - y) ** 2))
        self.log('val_rmse', rmse)
    
    def test_step(self, batch: Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], batch_idx: int) -> None:
        """
        Test step.
        
        Args:
            batch: Tuple of (x, y) where x is a tuple of (sequence_data, static_data)
            batch_idx: Index of the current batch
        """
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss)
        
        # Calculate RMSE
        rmse = torch.sqrt(torch.mean((y_hat - y) ** 2))
        self.log('test_rmse', rmse)
    
    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.
        
        Returns:
            Dictionary containing optimizer and scheduler configuration
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.training.learning_rate
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=self.config.training.lr_patience,
            min_lr=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }
    
    def train(self, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Train the ensemble model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
        
        Returns:
            Dictionary containing training metrics
        """
        trainer = pl.Trainer(
            max_epochs=self.config.training.max_epochs,
            callbacks=[
                pl.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.config.training.early_stopping_patience
                )
            ]
        )
        
        trainer.fit(self, train_loader, val_loader)
        
        return trainer.callback_metrics
    
    def evaluate(self, test_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Evaluate the ensemble model.
        
        Args:
            test_loader: DataLoader for test data
        
        Returns:
            Dictionary containing test metrics
        """
        trainer = pl.Trainer()
        results = trainer.test(self, test_loader)
        return results[0]
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Calculate feature importance based on frequency in base models.
        
        Returns:
            Dictionary mapping feature indices to importance scores
        """
        n_models = len(self.base_models)
        n_features = len(self.base_models[0].feature_mask)
        
        # Calculate frequency of selection for each feature
        importance_scores = {}
        for i in range(n_features):
            frequency = sum(mask[i].item() for mask in self.base_models[0].feature_mask) / n_models
            importance_scores[f"feature_{i}"] = frequency
        
        return importance_scores 