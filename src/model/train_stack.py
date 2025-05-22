import hydra
from omegaconf import DictConfig
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

class StackedEnsemble(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size):
        super(StackedEnsemble, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig):

    train_data = pd.read_csv("stacked_train_preds.csv").values
    train_targets = pd.read_csv("stacked_train_targets.csv").values.ravel()
    valid_data = pd.read_csv("stacked_valid_preds.csv").values
    valid_targets = pd.read_csv("stacked_valid_targets.csv").values.ravel()

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(train_data, train_targets)

    train_preds = rf.predict(train_data)
    valid_preds = rf.predict(valid_data)

    # Plot first 500 predictions
    plt.figure(figsize=(10, 5))
    plt.plot(train_targets[:500], label='True Train Targets', alpha=0.5)
    plt.plot(train_preds[:500], label='Predicted Train Targets', alpha=0.5)
    plt.title('Train Predictions vs True Targets (First 500 Samples)')
    plt.legend()

    plt.savefig('plots/rf_train_predictions.png')

    # Plot validation predictions
    plt.figure(figsize=(10, 5))
    plt.plot(valid_targets[:500], label='True Validation Targets', alpha=0.5)
    plt.plot(valid_preds[:500], label='Predicted Validation Targets', alpha=0.5)
    plt.title('Validation Predictions vs True Targets (First 500 Samples)')
    plt.legend()
    plt.savefig('plots/rf_valid_predictions.png')
    

    train_rmse = np.sqrt(mean_squared_error(train_targets[:500], train_preds[:500]))
    valid_rmse = np.sqrt(mean_squared_error(valid_targets, valid_preds))

    print(f"Train RMSE: {train_rmse}")
    print(f"Validation RMSE: {valid_rmse}")

    # Prepare data for meta-learner
    train_data = torch.tensor(train_data, dtype=torch.float32)
    train_targets = torch.tensor(train_targets, dtype=torch.float32)
    valid_data = torch.tensor(valid_data, dtype=torch.float32)
    valid_targets = torch.tensor(valid_targets, dtype=torch.float32)

    train_dataset = TensorDataset(train_data, train_targets)
    valid_dataset = TensorDataset(valid_data, valid_targets)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False)

    # Create a simple neural network model for the meta-learner
    input_size = train_data.shape[1]
    hidden_size = 64
    output_size = 1
    meta_model = StackedEnsemble(input_size, hidden_size, output_size)
    trainer = pl.Trainer(max_epochs=50, accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1 if torch.cuda.is_available() else 1)
    trainer.fit(meta_model, train_loader, valid_loader)
    meta_train_preds = meta_model(torch.tensor(train_data, dtype=torch.float32))
    meta_valid_preds = meta_model(torch.tensor(valid_data, dtype=torch.float32))
    meta_train_rmse = np.sqrt(mean_squared_error(train_targets, meta_train_preds.detach().numpy()))
    meta_valid_rmse = np.sqrt(mean_squared_error(valid_targets, meta_valid_preds.detach().numpy()))
    print(f"Meta-learner Train RMSE: {meta_train_rmse}")
    print(f"Meta-learner Validation RMSE: {meta_valid_rmse}")

    # Plot train predictions
    plt.figure(figsize=(10, 5))
    plt.plot(train_targets[:500], label='True Train Targets', alpha=0.5)
    plt.plot(meta_train_preds.detach().numpy()[:500], label='Meta-learner Train Predictions', alpha=0.5)
    plt.title('Meta-learner Train Predictions vs True Targets (First 500 Samples)')
    plt.legend()
    plt.savefig('plots/meta_learner_train_predictions.png')

    # Plot meta-learner predictions
    plt.figure(figsize=(10, 5))
    plt.plot(valid_targets[:500], label='True Validation Targets', alpha=0.5)
    plt.plot(meta_valid_preds.detach().numpy()[:500], label='Meta-learner Validation Predictions', alpha=0.5)
    plt.title('Meta-learner Validation Predictions vs True Targets (First 500 Samples)')
    plt.legend()
    plt.savefig('plots/meta_learner_valid_predictions.png')

if __name__ == "__main__":
    main()
