import os
import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
import logging
from data.data_loader import TimeSeriesDataLoader
from data.dataset import TimeSeriesDataset
from model.feature_selection_lstm import FeatureSelectionLSTM
from model.moea_optimizer import MOEAOptimizer
from model.ensemble_model import EnsembleModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(config: DictConfig):
    """
    Main function to run the feature selection and model training pipeline.
    
    Args:
        config: Hydra configuration object
    """
    # Set random seed for reproducibility
    torch.manual_seed(config.seed)
    
    # Initialize data loader
    data_loader = TimeSeriesDataLoader(config)
    
    # Load and partition data
    logger.info("Loading and partitioning data...")
    data, time_data = data_loader.load_data()
    partitions = data_loader.create_partitions(data, time_data)
    
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
    
    ensemble.train(train_loader, val_loader)
    
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