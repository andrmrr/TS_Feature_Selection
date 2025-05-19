import matplotlib.pyplot as plt
from train import load_dataset
import hydra
from omegaconf import DictConfig
from lstm import LSTMModel
import torch

def plot_real_target(train_loader, val_loader, model):
    # Get the first batch from the validation loader
    for batch in train_loader:
        x, y = batch
        break

    device = next(model.parameters()).device

    # Move input to the same device as model
    if isinstance(x, (tuple, list)):
        x = tuple(xx.to(device) for xx in x)
    else:
        x = x.to(device)
    y = y.to(device)

    model.eval()
    with torch.no_grad():
        y_hat = model(x)


    # Plot the real vs predicted values
    plt.figure(figsize=(10, 5))
    plt.plot(y.cpu().numpy(), label='Real')
    plt.plot(y_hat.cpu().numpy(), label='Predicted')
    plt.legend()
    plt.title('Real vs Predicted Values')
    plt.show()

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig):
    data_path = "data/data.npy"
    time_data_path = "data/time_data.npy"

    train_loader, val_loader = load_dataset(data_path, time_data_path)

    model = LSTMModel.load_from_checkpoint(
        "logs/model/version_3/checkpoints/epoch=499-step=219000.ckpt",
        lstm_input_size=cfg.model.lstm_input_size,
        lstm_hidden_size=cfg.model.lstm_hidden_size,
        lstm_num_layers=cfg.model.lstm_num_layers,
        static_input_size=cfg.model.static_input_size,
        static_hidden_size=cfg.model.static_hidden_size,
        merged_hidden_size=cfg.model.merged_hidden_size,
        output_size=cfg.model.output_size
    )

    # Set the model to evaluation mode
    plot_real_target(train_loader=train_loader, val_loader=val_loader, model=model)


if __name__ == "__main__":
    main()