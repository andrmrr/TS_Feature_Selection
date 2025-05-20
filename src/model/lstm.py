import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, sequence_data, static_data, seq_length, feature_mask=None, static_mask=None):
        # Feature masking for sequence features
        features = sequence_data[:, 1:]
        if feature_mask is not None:
            features = features[:, feature_mask]
        self.data = features

        self.target = sequence_data[:, 0]
        self.target = self.target.reshape(-1, 1)
        
        # Feature masking for static features (optional, if needed)
        if static_mask is not None:
            static_data = static_data[:, static_mask]
        self.static_data = static_data
        
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.static_data = torch.tensor(self.static_data, dtype=torch.float32)
        self.target = torch.tensor(self.target, dtype=torch.float32)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        current_features = self.data[idx]
        current_static_features = self.static_data[idx]

        old_features = self.data[idx: idx + self.seq_length]
        old_targets = self.target[idx: idx + self.seq_length]
        seq_x = torch.cat((old_features, old_targets), dim=1)
        static_x = torch.cat((current_features, current_static_features))
        y = self.target[idx + self.seq_length]
        return (seq_x.detach().clone(), static_x.detach().clone()), y.detach().clone()
    
class LSTMModel(pl.LightningModule):
    def __init__(self,
                 lstm_input_size, lstm_hidden_size, lstm_num_layers,
                 static_input_size, static_hidden_size,
                 merged_hidden_size, output_size):
        super().__init__()
        self.save_hyperparameters()

        # SEQUENTIAL BRANCH

        self.sequential_fc = nn.Sequential(
            nn.LSTM(lstm_input_size, lstm_hidden_size, lstm_num_layers, batch_first=True),
            nn.ReLU()
        )

        # STATIC BRANCH

        self.static_fc = nn.Sequential(
            nn.Linear(static_input_size, static_hidden_size),
            nn.ReLU()
        )

        # COMBINED BRANCH
        self.combined_fc = nn.Sequential(
            nn.Linear(lstm_hidden_size + static_hidden_size, merged_hidden_size),
            nn.ReLU(),
            nn.Linear(merged_hidden_size, output_size),
            nn.Softplus()
        )
        
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        seq_x, static_x = x
        # SEQUENTIAL BRANCH
        lstm_out, _ = self.sequential_fc[0](seq_x)
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.sequential_fc[1:](lstm_out)
        # STATIC BRANCH
        static_out = self.static_fc(static_x)
        # COMBINED BRANCH
        combined_out = torch.cat((lstm_out, static_out), dim=1)
        combined_out = self.combined_fc(combined_out)
        return combined_out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss, prog_bar=True, reduce_fx="mean")

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            },
        }
