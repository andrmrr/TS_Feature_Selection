from typing import List, Tuple, Dict, Any
import numpy as np
import torch
from platypus import NSGAII, Problem, Binary
from omegaconf import DictConfig
from .feature_selection_lstm import FeatureSelectionLSTM
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
import logging
from torch.utils.data import DataLoader
from ..data.dataset import TimeSeriesDataset

class FeatureSelectionProblem(Problem):
    """
    Multi-objective optimization problem for feature selection.
    Each objective corresponds to the RMSE on a different data partition.
    """
    
    def __init__(self, 
                 model: FeatureSelectionLSTM,
                 partitions: List[Any],  # List of DataPartition objects
                 config: DictConfig):
        """
        Initialize the feature selection problem.
        
        Args:
            model: The LSTM model to use for evaluation
            partitions: List of DataPartition objects
            config: Configuration dictionary
        """
        self.model = model
        self.partitions = partitions
        self.config = config
        
        # Number of features to select from
        n_features = config.model.lstm.input_size
        
        # Number of objectives (one per partition)
        n_objectives = len(partitions)
        
        # Initialize problem with binary variables and minimization objectives
        super().__init__(n_features, n_objectives)
        self.types[:] = [Binary(1)] * n_features
        self.directions[:] = [Problem.MINIMIZE] * n_objectives
        
    def evaluate(self, solution):
        """
        Evaluate a feature mask solution on all partitions.
        
        Args:
            solution: Platypus solution object containing the feature mask
        """
        # Convert solution to binary mask
        mask = np.array([int(b[0]) for b in solution.variables])
        
        # Skip if no features are selected
        if mask.sum() == 0:
            solution.objectives[:] = [9999.0] * len(self.partitions)
            return
        
        # Convert mask to torch tensor
        mask_tensor = torch.tensor(mask, dtype=torch.bool)
        
        # Set feature mask in model
        self.model.set_feature_mask(mask_tensor)
        
        # Evaluate on each partition
        rmses = []
        for partition in self.partitions:
            try:
                # Create datasets
                train_dataset = TimeSeriesDataset(
                    data=partition.train_data,
                    time_data=partition.train_time,
                    seq_length=self.config.data.sequence_length,
                    feature_mask=mask_tensor
                )
                val_dataset = TimeSeriesDataset(
                    data=partition.val_data,
                    time_data=partition.val_time,
                    seq_length=self.config.data.sequence_length,
                    feature_mask=mask_tensor
                )
                
                # Create dataloaders
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=self.config.data.batch_size,
                    shuffle=True
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=self.config.data.batch_size,
                    shuffle=False
                )
                
                # Train model
                trainer = Trainer(
                    max_epochs=self.config.training.max_epochs,
                    callbacks=[
                        EarlyStopping(
                            monitor='val_loss',
                            patience=self.config.training.early_stopping_patience
                        )
                    ],
                    enable_progress_bar=False
                )
                trainer.fit(self.model, train_loader, val_loader)
                
                # Evaluate on validation set
                results = trainer.test(self.model, val_loader)
                rmse = results[0]['test_rmse']
                rmses.append(rmse)
                
            except Exception as e:
                logging.error(f"Error during evaluation: {str(e)}")
                rmses.append(9999.0)
        
        solution.objectives[:] = rmses

class MOEAOptimizer:
    """
    Multi-Objective Evolutionary Algorithm optimizer for feature selection.
    Uses NSGA-II to find Pareto-optimal feature subsets.
    """
    
    def __init__(self, config: DictConfig):
        """
        Initialize the MOEA optimizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.moea_config = config.moea
        
    def optimize(self, 
                model: FeatureSelectionLSTM,
                partitions: List[Any]) -> List[np.ndarray]:  # List of DataPartition objects
        """
        Run the MOEA optimization process.
        
        Args:
            model: The LSTM model to use for evaluation
            partitions: List of DataPartition objects
            
        Returns:
            List of Pareto-optimal feature masks
        """
        # Create optimization problem
        problem = FeatureSelectionProblem(model, partitions, self.config)
        
        # Initialize NSGA-II algorithm
        algorithm = NSGAII(
            problem,
            population_size=self.moea_config.population_size,
            tournament_size=self.moea_config.tournament_size
        )
        
        # Run optimization
        algorithm.run(self.moea_config.generations)
        
        # Extract Pareto-optimal solutions
        pareto_front = []
        for solution in algorithm.result:
            mask = np.array([int(b[0]) for b in solution.variables])
            pareto_front.append(mask)
        
        return pareto_front
    
    def get_feature_importance(self, pareto_front: List[np.ndarray]) -> Dict[str, float]:
        """
        Calculate feature importance based on frequency in Pareto-optimal solutions.
        
        Args:
            pareto_front: List of Pareto-optimal feature masks
            
        Returns:
            Dictionary mapping feature indices to importance scores
        """
        n_solutions = len(pareto_front)
        n_features = len(pareto_front[0])
        
        # Calculate frequency of selection for each feature
        importance_scores = {}
        for i in range(n_features):
            frequency = sum(mask[i] for mask in pareto_front) / n_solutions
            importance_scores[f"feature_{i}"] = frequency
        
        return importance_scores 