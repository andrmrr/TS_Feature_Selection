import numpy as np
from platypus import NSGAII, Problem, Binary, nondominated
from sklearn.metrics import mean_squared_error
import torch
import pytorch_lightning as pl
from lstm import LSTMModel
from lstm import TimeSeriesDataset
from torch.utils.data import DataLoader

class FeatureSelectionProblem(Problem):
    def __init__(self, train_norm_data, valid_norm_data, train_time_data, valid_time_data,
                 n_partitions, input_size, device='cpu', model=None,  seq_length=10):
        super().__init__(input_size, n_partitions)
        self.types[:] = [Binary(1) for _ in range(input_size)]
        self.directions[:] = [self.MINIMIZE] * n_partitions
        self.train_norm_data=train_norm_data
        self.valid_norm_data=valid_norm_data
        self.train_time_data=train_time_data
        self.valid_time_data=valid_time_data
        self.n_partitions = n_partitions
        self.input_size = input_size
        self.seq_length = seq_length
        self.device = device
        self.model = model.eval().to(device)

    def evaluate(self, solution):
        mask = np.array([int(bit[0]) for bit in solution.variables])
        print(mask)
        if mask.sum() == 0:
            # Penalize empty feature set
            solution.objectives[:] = [1e6] * self.n_partitions
            return
        rmses = []
        train_dataset = TimeSeriesDataset(
            self.train_norm_data, self.train_time_data, seq_length=self.seq_length, feature_mask=None, static_mask=None
        )
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False, num_workers=4, persistent_workers=True)
        preds, targets = [], []
        with torch.no_grad():
            for x, y in train_loader:
                if isinstance(x, (tuple, list)):
                    x = tuple(xx.to(self.device) for xx in x)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                out = self.model(x, seq_mask=mask)
                preds.append(out.cpu().numpy())
                targets.append(y.cpu().numpy())
        preds = np.concatenate(preds).flatten()
        targets = np.concatenate(targets).flatten()
        partition_size = len(preds) // self.n_partitions
        for i in range(self.n_partitions):
            rmse = np.sqrt(mean_squared_error(targets[i*partition_size:(i+1)*partition_size], preds[i*partition_size:(i+1)*partition_size]))
            rmses.append(rmse)
        solution.objectives[:] = rmses

def run_nsga2_feature_selection(train_norm_data, valid_norm_data, train_time_data, valid_time_data, input_size,
                                n_partitions, model, population_size=20, n_generations=10, device='cpu', seq_length=10):
    problem = FeatureSelectionProblem(
        train_norm_data, valid_norm_data, train_time_data, valid_time_data,
        n_partitions, input_size, device, model=model, seq_length=seq_length
    )
    algorithm = NSGAII(problem, population_size)
    algorithm.run(n_generations)
    pareto_solutions = nondominated(algorithm.result)
    masks = [np.array([int(bit[0]) for bit in s.variables]) for s in pareto_solutions]
    objectives = [s.objectives for s in pareto_solutions]
    return masks, objectives 