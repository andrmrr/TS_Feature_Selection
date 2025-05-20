import matplotlib.pyplot as plt
from train import load_dataset
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
    import pdb; pdb.set_trace()
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