import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
import logging
from typing import List, Tuple, Dict, Any
import numpy as np
import json

from src.data.data_loader import TimeSeriesDataLoader
from src.data.dataset import TimeSeriesDataset
from src.model.feature_selection_lstm import FeatureSelectionLSTM
from src.model.moea_optimizer import MOEAOptimizer
from src.model.ensemble_model import EnsembleModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(config: DictConfig):
    """
    Main function that orchestrates the feature selection and training process.
    
    Args:
        config: Configuration dictionary
    """
    # Set random seeds for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Initialize data loader
    data_loader = TimeSeriesDataLoader(
        data_path=config.data.path,
        time_data_path=config.data.time_data_path
    )
    
    # Load and partition data
    logger.info("Loading and partitioning data...")
    data, time_data = data_loader.load_data()
    partitions = data_loader.create_partitions(
        data=data,
        time_data=time_data,
        n_partitions=config.data.n_partitions,
        train_val_split=config.data.train_val_split
    )
    
    # Initialize model
    logger.info("Initializing LSTM model...")
    model = FeatureSelectionLSTM(config)
    
    # Initialize MOEA optimizer
    logger.info("Initializing MOEA optimizer...")
    optimizer = MOEAOptimizer(config)
    
    # Run feature selection
    logger.info("Running feature selection...")
    pareto_front = optimizer.optimize(model, partitions)
    
    # Calculate feature importance
    logger.info("Calculating feature importance...")
    feature_importance = optimizer.get_feature_importance(pareto_front)
    
    # Print feature importance
    logger.info("Feature importance scores:")
    for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"{feature}: {importance:.4f}")
    
    # Create ensemble model
    logger.info("Creating ensemble model...")
    ensemble = EnsembleModel(config, model, pareto_front)
    
    # Train ensemble on first partition
    logger.info("Training ensemble model...")
    train_dataset = TimeSeriesDataset(
        data=partitions[0].train_data,
        time_data=partitions[0].train_time,
        seq_length=config.data.sequence_length
    )
    val_dataset = TimeSeriesDataset(
        data=partitions[0].val_data,
        time_data=partitions[0].val_time,
        seq_length=config.data.sequence_length
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False
    )
    
    # Create PyTorch Lightning trainer
    trainer = Trainer(
        max_epochs=config.training.max_epochs,
        callbacks=[
            EarlyStopping(
                monitor='val_loss',
                patience=config.training.early_stopping_patience
            )
        ]
    )
    
    # Train the ensemble model using PyTorch Lightning
    trainer.fit(ensemble, train_loader, val_loader)
    
    # Evaluate on last partition
    logger.info("Evaluating ensemble model...")
    test_dataset = TimeSeriesDataset(
        data=partitions[-1].val_data,
        time_data=partitions[-1].val_time,
        seq_length=config.data.sequence_length
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False
    )
    
    metrics = ensemble.evaluate(test_loader)
    logger.info(f"Test metrics: {metrics}")

if __name__ == "__main__":
    main() 