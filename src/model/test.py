import matplotlib.pyplot as plt
from utils import load_dataset_2
import hydra
from omegaconf import DictConfig
from lstm import LSTMModel
import torch

def plot_real_target(train_loader, val_loader, model):
    device = next(model.parameters()).device

    # Collect all batches from the validation loader
    all_y = []
    all_y_hat = []

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            if isinstance(x, (tuple, list)):
                x = tuple(xx.to(device) for xx in x)
            else:
                x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            all_y.append(y.cpu())
            all_y_hat.append(y_hat.cpu())
    all_y = torch.cat(all_y, dim=0).numpy()
    all_y_hat = torch.cat(all_y_hat, dim=0).numpy()

    MSE_loss = ((all_y - all_y_hat) ** 2)
    # 

    # Plot the whole dataset
    plt.figure(figsize=(10, 5))
    plt.plot(all_y, label='Real')
    plt.plot(all_y_hat, label='Predicted')
    plt.title('Real vs Predicted')
    plt.show()


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig):
    data_path = "data/data.npy"
    time_data_path = "data/time_data.npy"

    train_loader, val_loader = load_dataset_2(data_path, time_data_path)

    checkpoints = [
        ("logs/lstm_1_layer/version_1/checkpoints/epoch=199-step=87600.ckpt", 1, "1_layer_lstm"),
        ("logs/lstm_2_layers/version_1/checkpoints/epoch=199-step=87600.ckpt", 2, "2_layers_lstm"),
        ("logs/model/version_1/checkpoints/epoch=199-step=87600.ckpt", 3, "3_layers_lstm"),
    ]

    for checkpoint, layer_size, name in checkpoints:
        model = LSTMModel.load_from_checkpoint(
            checkpoint,
            lstm_input_size=cfg.model.lstm_input_size,
            lstm_hidden_size=cfg.model.lstm_hidden_size,
            lstm_num_layers=layer_size,
            static_input_size=cfg.model.static_input_size,
            static_hidden_size=cfg.model.static_hidden_size,
            merged_hidden_size=cfg.model.merged_hidden_size,
            output_size=cfg.model.output_size
        )

        device = next(model.parameters()).device

        # Collect all batches from the validation loader
        all_y = []
        all_y_hat = []

        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                if isinstance(x, (tuple, list)):
                    x = tuple(xx.to(device) for xx in x)
                else:
                    x = x.to(device)
                y = y.to(device)
                y_hat = model(x)
                all_y.append(y.cpu())
                all_y_hat.append(y_hat.cpu())
        all_y = torch.cat(all_y, dim=0).numpy()
        all_y_hat = torch.cat(all_y_hat, dim=0).numpy()

        MSE_loss = ((all_y[:500] - all_y_hat[:500]) ** 2)
        print(f"Mean Squared Error for {name}: {MSE_loss.mean()}")

        # Plot the first 500 predictions
        plt.figure(figsize=(10, 5))
        plt.plot(all_y[:500], label='True Valid Targets', alpha=0.5)
        plt.plot(all_y_hat[:500], label='Predictions', alpha=0.5)
        plt.title(f'model {name} Validation Predictions vs True Targets (First 500 Samples)')
        plt.legend()
        plt.savefig(f'plots/{name}_valid_predictions.png')

        # train
        all_y_train = []
        all_y_hat_train = []
        with torch.no_grad():
            for batch in train_loader:
                x, y = batch
                if isinstance(x, (tuple, list)):
                    x = tuple(xx.to(device) for xx in x)
                else:
                    x = x.to(device)
                y = y.to(device)
                y_hat = model(x)
                all_y_train.append(y.cpu())
                all_y_hat_train.append(y_hat.cpu())
        all_y_train = torch.cat(all_y_train, dim=0).numpy()
        all_y_hat_train = torch.cat(all_y_hat_train, dim=0).numpy()
        MSE_loss = ((all_y_train[:500] - all_y_hat_train[:500]) ** 2)
        print(f"Mean Squared Error for {name} train: {MSE_loss.mean()}")
        # Plot the first 500 predictions
        plt.figure(figsize=(10, 5))
        plt.plot(all_y_train[:500], label='True Train Targets', alpha=0.5)
        plt.plot(all_y_hat_train[:500], label='Predictions', alpha=0.5)
        plt.title(f'model {name} Train Predictions vs True Targets (First 500 Samples)')
        plt.legend()
        plt.savefig(f'plots/{name}_train_predictions.png')



if __name__ == "__main__":
    main()