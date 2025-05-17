import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Optional, Tuple, Dict, Any
import numpy as np
from omegaconf import DictConfig

class FeatureSelectionLSTM(pl.LightningModule):
    """
    LSTM model for time series prediction with feature selection capability.
    """
    
    def __init__(self, config):
        """
        Initialize the LSTM model.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Initialize feature mask (all features selected by default)
        self.feature_mask = torch.ones(config.model.lstm.input_size, dtype=torch.bool)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.model.lstm.input_size,
            hidden_size=config.model.lstm.hidden_size,
            num_layers=config.model.lstm.num_layers,
            batch_first=True,
            dropout=config.model.lstm.dropout
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(config.model.lstm.hidden_size, config.model.lstm.fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(config.model.lstm.dropout),
            nn.Linear(config.model.lstm.fc_hidden_size, 1)
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
    def set_feature_mask(self, mask: torch.Tensor):
        """
        Set the feature mask for feature selection.
        
        Args:
            mask: Boolean tensor indicating which features to use
        """
        self.feature_mask = mask
        
    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Tuple of (sequence_data, static_data)
               - sequence_data: Tensor of shape (batch_size, seq_length, n_features)
               - static_data: Tensor of shape (batch_size, n_static_features)
        
        Returns:
            Tensor of shape (batch_size, 1) containing predictions
        """
        sequence_data, static_data = x
        
        # Apply feature mask to sequence data
        masked_data = sequence_data[:, :, self.feature_mask]
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(masked_data)
        
        # Use only the last time step's output
        last_hidden = lstm_out[:, -1, :]
        
        # Combine with static features if available
        if static_data is not None:
            combined = torch.cat([last_hidden, static_data], dim=1)
        else:
            combined = last_hidden
        
        # Final prediction
        return self.fc_layers(combined)
    
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
    
    def get_feature_importance(self) -> torch.Tensor:
        """Get the current feature mask."""
        return self.feature_mask
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get a summary of the model architecture and parameters."""
        return {
            'lstm_params': sum(p.numel() for p in self.lstm.parameters()),
            'fc_params': sum(p.numel() for p in self.fc_layers.parameters()),
            'total_params': sum(p.numel() for p in self.parameters()),
            'active_features': self.feature_mask.sum().item(),
            'total_features': len(self.feature_mask)
        } 