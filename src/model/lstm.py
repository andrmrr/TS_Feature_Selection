import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length, feature_mask=None):
        features = data[:, 1:]
        if feature_mask is not None:
            features = features[:, feature_mask]
        self.data = features
        self.target = data[:, 0]
        self.target = self.target.reshape(-1, 1)
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.target = torch.tensor(self.target, dtype=torch.float32)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.target[idx + self.seq_length]
        return x.detach().clone(), y.detach().clone() #torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
class LSTMModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.save_hyperparameters()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
