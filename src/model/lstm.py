import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torch.utils.data import Dataset

class GradHistLogger(Callback):
    def on_after_backward(self, trainer, pl_module):
        logger = pl_module.logger.experiment  # this is a TB SummaryWriter
        global_step = trainer.global_step
        
        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                # log a histogram of the gradients
                logger.add_histogram(
                    tag=f"grads/{name.replace('.', '/')}",
                    values=param.grad,
                    global_step=global_step
                )

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data[:, 1:]
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
        return x.detach().clone(), y.detach().clone()
    
class LSTMModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, pos_weight=0.5):
        super().__init__()
        self.save_hyperparameters()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
        pw = torch.tensor([pos_weight], dtype=torch.float32)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        preds = torch.sigmoid(y_hat) > 0.5
        acc = (preds == y).float().mean()
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

        preds = torch.sigmoid(y_hat) > 0.5
        acc = (preds == y).float().mean()
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            },
        }
