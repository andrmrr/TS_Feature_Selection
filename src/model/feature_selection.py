import numpy as np
from platypus import NSGAII, Problem, Binary, nondominated
from sklearn.metrics import mean_squared_error
import torch
import pytorch_lightning as pl
from lstm import LSTMModel
from utils import partition_time_series, split_train_val, create_dataloaders

class FeatureSelectionProblem(Problem):
    def __init__(self, data, n_partitions, seq_length, input_size, hidden_size, num_layers, max_epochs, batch_size=32, device='cpu'):
        super().__init__(input_size, n_partitions)
        self.types[:] = [Binary(1) for _ in range(input_size)]
        self.directions[:] = [self.MINIMIZE] * n_partitions
        self.data = data
        self.n_partitions = n_partitions
        self.seq_length = seq_length
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.device = device
        self.partitions = partition_time_series(data, n_partitions)
        self.train_val = [split_train_val(p) for p in self.partitions]

    def evaluate(self, solution):
        mask = np.array([int(bit[0]) for bit in solution.variables])
        if mask.sum() == 0:
            # Penalize empty feature set
            solution.objectives[:] = [1e6] * self.n_partitions
            return
        rmses = []
        for (train_data, val_data) in self.train_val:
            train_loader, val_loader = create_dataloaders(
                train_data, val_data, self.seq_length, self.batch_size, feature_mask=mask.astype(bool)
            )
            model = LSTMModel(
                input_size=mask.sum(),
                hidden_size=self.hidden_size,
                num_layers=self.num_layers
            )
            trainer = pl.Trainer(
                max_epochs=self.max_epochs,
                enable_checkpointing=False,
                logger=False,
                enable_model_summary=False,
                accelerator='gpu',
                devices=1,
                enable_progress_bar=True,
                strategy='auto'
            )
            trainer.fit(model, train_loader, val_loader)
            preds, targets = [], []
            model.eval()
            with torch.no_grad():
                for x, y in val_loader:
                    out = model(x)
                    preds.append(out.cpu().numpy())
                    targets.append(y.cpu().numpy())
            preds = np.concatenate(preds).flatten()
            targets = np.concatenate(targets).flatten()
            rmse = np.sqrt(mean_squared_error(targets, preds))
            rmses.append(rmse)
        solution.objectives[:] = rmses

def run_nsga2_feature_selection(data, n_partitions, seq_length, input_size, hidden_size, num_layers, max_epochs, batch_size=32, population_size=20, n_generations=10, device='cpu'):
    problem = FeatureSelectionProblem(
        data, n_partitions, seq_length, input_size, hidden_size, num_layers, max_epochs, batch_size, device
    )
    algorithm = NSGAII(problem, population_size)
    algorithm.run(n_generations)
    pareto_solutions = nondominated(algorithm.result)
    masks = [np.array([int(bit[0]) for bit in s.variables]) for s in pareto_solutions]
    objectives = [s.objectives for s in pareto_solutions]
    return masks, objectives 