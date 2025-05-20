import numpy as np
from platypus import NSGAII, Problem, Binary, nondominated
from sklearn.metrics import mean_squared_error
import torch
import pytorch_lightning as pl
from lstm import LSTMModel
from sklearn.metrics import mean_squared_error
from platypus import HUX, SBX, PM, CompoundOperator

import numpy as np
from platypus import NSGAII, Problem, Binary, Real, nondominated
from sklearn.metrics import mean_squared_error
import torch

from utils import partition_time_series, split_train_val


def compute_num_weights(input_size, hidden_size):
    # 4 gates × (input to hidden + hidden to hidden + bias per gate)
    num = 4 * (input_size * hidden_size + hidden_size * hidden_size + hidden_size)
    # Output layer: hidden_size → 1
    num += hidden_size + 1
    return num

def extract_lstm_weights(w, input_size, hidden_size):
    # w is a flat vector, follow the order: input→hidden weights, hidden→hidden weights, biases, output weights/bias
    idx = 0
    W_ih, W_hh, b_ih, b_hh = [], [], [], []
    for _ in range(4):  # Four gates
        W_ih.append(w[idx: idx + hidden_size * input_size].reshape(hidden_size, input_size))
        idx += hidden_size * input_size
    for _ in range(4):
        W_hh.append(w[idx: idx + hidden_size * hidden_size].reshape(hidden_size, hidden_size))
        idx += hidden_size * hidden_size
    for _ in range(4):
        b_ih.append(w[idx: idx + hidden_size])
        idx += hidden_size
    for _ in range(4):
        b_hh.append(w[idx: idx + hidden_size])
        idx += hidden_size
    # Output layer
    W_out = w[idx:idx+hidden_size].reshape(1, hidden_size)
    idx += hidden_size
    b_out = w[idx:idx+1]
    return W_ih, W_hh, b_ih, b_hh, W_out, b_out

def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1 / (1 + np.exp(-x))

def simple_forward_lstm(X, w, input_size, hidden_size):
    # X: (n_samples, input_size), one-step prediction
    # w: weights vector
    # Returns: predictions (n_samples,)
    W_ih, W_hh, b_ih, b_hh, W_out, b_out = extract_lstm_weights(w, input_size, hidden_size)
    n_samples = X.shape[0]
    h = np.zeros((hidden_size,))
    c = np.zeros((hidden_size,))
    preds = []
    # For each sample (no temporal dependency for simplest forward)
    for i in range(n_samples):
        x_t = X[i]
        gates = []
        for gate in range(4):
            gates.append(
                np.dot(W_ih[gate], x_t) + b_ih[gate] + np.dot(W_hh[gate], h) + b_hh[gate]
            )
        i_t = sigmoid(gates[0])
        f_t = sigmoid(gates[1])
        g_t = np.tanh(gates[2])
        o_t = sigmoid(gates[3])
        c = f_t * c + i_t * g_t
        h = o_t * np.tanh(c)
        y = np.dot(W_out, h) + b_out
        preds.append(y.item())
    return np.array(preds)


class FeatureSelectionProblem(Problem):
    def __init__(self, data, n_partitions, seq_length, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.n_partitions = n_partitions
        self.num_weights = compute_num_weights(input_size, hidden_size)
        # Add one more objective for feature count
        super().__init__(input_size + self.num_weights, n_partitions + 1)
        self.types[:] = [Binary(1) for _ in range(input_size)] + [Real(-1, 1) for _ in range(self.num_weights)]
        # Minimize RMSE for each partition, minimize number of features
        self.directions[:] = [self.MINIMIZE] * (n_partitions + 1)
        self.data = data
        self.partitions = partition_time_series(data, n_partitions)
        self.train_val = [split_train_val(p) for p in self.partitions]

    def evaluate(self, solution):
        mask = np.array([int(bit[0]) for bit in solution.variables[:self.input_size]])
        w = np.array([float(x) for x in solution.variables[self.input_size:]])
        
        if mask.sum() == 0:
            solution.objectives[:] = [1e6] * (self.n_partitions + 1)
            return
            
        rmses = []
        for (train_data, val_data) in self.train_val:
            # Prepare masked inputs
            X_val = val_data[:, 1:][:, mask.astype(bool)]
            y_val = val_data[:, 0]
            # Single-step forecasting, no static features for simplicity
            preds = simple_forward_lstm(X_val, w, mask.sum(), self.hidden_size)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            rmses.append(rmse)
            
        # Add feature count as an additional objective
        solution.objectives[:] = rmses + [mask.sum()]


def run_nsga2_feature_selection(data, n_partitions, seq_length, input_size, hidden_size, population_size, n_generations):
    problem = FeatureSelectionProblem(data, n_partitions, seq_length, input_size, hidden_size)
    
    # Create a more diverse set of genetic operators
    variator = CompoundOperator(
        HUX(probability=0.9),  # High probability for binary crossover
        SBX(probability=0.9),  # High probability for real-valued crossover
        PM(probability=0.1)    # Low probability for mutation to maintain diversity
    )
    
    # Use a larger population and more generations
    algorithm = NSGAII(
        problem,
        population_size=population_size,
        variator=variator,
        archive_size=population_size  # Keep all non-dominated solutions
    )
    
    algorithm.run(n_generations)
    pareto_solutions = nondominated(algorithm.result)
    
    # For each solution, save (mask, weights, objectives)
    results = []
    for s in pareto_solutions:
        mask = np.array([int(bit[0]) for bit in s.variables[:input_size]])
        w = np.array([float(x) for x in s.variables[input_size:]])
        results.append((mask, w, s.objectives))
    
    print(f"Found {len(results)} Pareto-optimal solutions")
    return results
