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
        # First feature (time) is always selected
        self.feature_mask = torch.ones(config.model.lstm.input_size - 1, dtype=torch.bool)  # Exclude time feature
        
        # LSTM layers - input size will be 1 (time) + number of selected features
        self.lstm = nn.LSTM(
            input_size=1 + self.feature_mask.sum().item(),  # Time feature + selected features
            hidden_size=config.model.lstm.hidden_size,
            num_layers=config.model.lstm.num_layers,
            batch_first=True,
            dropout=config.model.lstm.dropout
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(config.model.lstm.hidden_size, config.model.lstm.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.model.lstm.dropout),
            nn.Linear(config.model.lstm.hidden_size, 1)
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
    def set_feature_mask(self, mask: torch.Tensor):
        """
        Set the feature mask for feature selection.
        
        Args:
            mask: Boolean tensor indicating which features to use (excluding time feature)
        """
        if len(mask) != self.config.model.lstm.input_size - 1:
            raise ValueError(f"Feature mask length ({len(mask)}) does not match number of features ({self.config.model.lstm.input_size - 1})")
        self.feature_mask = mask
        
        # Recreate LSTM with new input size
        if mask.sum().item() == 0:
            print("No features selected")
        self.lstm = nn.LSTM(
            input_size=1 + mask.sum().item(),  # Time feature + selected features
            hidden_size=self.config.model.lstm.hidden_size,
            num_layers=self.config.model.lstm.num_layers,
            batch_first=True,
            dropout=self.config.model.lstm.dropout
        ).to(self.device)
        
        # Recreate fc_layers with correct input size
        self.fc_layers = nn.Sequential(
            nn.Linear(self.config.model.lstm.hidden_size, self.config.model.lstm.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.config.model.lstm.dropout),
            nn.Linear(self.config.model.lstm.hidden_size, 1)
        ).to(self.device)
        
    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the model.
        Args:
            x: Tuple of (sequence_data, static_data)
            - sequence_data: Tensor of shape (batch_size, seq_length, n_features)
            - static_data: Tensor of shape (batch_size, n_static_features) or None
        Returns:
            Tensor of shape (batch_size, 1) containing predictions
        """
        sequence_data, static_data = x

        # 1. Split time and other features
        time_feature = sequence_data[:, :, 0:1]   # shape: [batch, seq, 1]
        other_features = sequence_data[:, :, 1:]  # shape: [batch, seq, n_other_features]

        # 2. Mask application safety
        if other_features.size(-1) != len(self.feature_mask):
            raise ValueError(
                f"Number of features in input ({other_features.size(-1)}) does not match mask length ({len(self.feature_mask)})"
            )

        masked_features = other_features[:, :, self.feature_mask]  # Apply mask, shape: [batch, seq, n_masked_features]
        masked_data = torch.cat([time_feature, masked_features], dim=2)  # Combine time and masked, shape: [batch, seq, 1 + n_masked_features]

        # 3. LSTM forward pass
        lstm_out, _ = self.lstm(masked_data)
        last_hidden = lstm_out[:, -1, :]  # shape: [batch, hidden_size]

        # 4. Use only LSTM output since static features are disabled
        combined = last_hidden  # shape: [batch, hidden_size]

        # 5. Final prediction
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