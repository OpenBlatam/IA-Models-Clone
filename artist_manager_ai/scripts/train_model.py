#!/usr/bin/env python3
"""
Train Model Script
==================

Script para entrenar modelos PyTorch.
Sigue mejores prácticas de PyTorch/Transformers.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import logging
import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.models import EventDurationPredictor, RoutineCompletionPredictor, OptimalTimePredictor
from ml.data import EventDataset, RoutineDataset, create_dataloaders, FeatureExtractor
from ml.training import Trainer
from ml.config import load_config
from experiments import ExperimentTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_event_model(config_path: str, data_path: str):
    """Train event duration prediction model."""
    logger.info("Training event duration model")
    
    # Load config
    config = load_config(config_path)
    model_config = config["model"]["event_duration"]
    training_config = config["training"]
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load data (placeholder - implement actual data loading)
    # events = load_events(data_path)
    events = []  # TODO: Implement data loading
    
    if not events:
        logger.warning("No events found. Using dummy data for demonstration.")
        # Create dummy data
        from datetime import datetime, timedelta
        events = [
            {
                "type": "concert",
                "start_time": (datetime.now() + timedelta(days=i)).isoformat(),
                "end_time": (datetime.now() + timedelta(days=i, hours=3)).isoformat(),
                "location": "Venue A"
            }
            for i in range(100)
        ]
    
    # Create dataset
    feature_extractor = FeatureExtractor()
    dataset = EventDataset(events, feature_extractor)
    
    if len(dataset) == 0:
        logger.error("Dataset is empty. Cannot train.")
        return
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset,
        batch_size=training_config["batch_size"],
        train_ratio=config["data"]["train_ratio"],
        val_ratio=config["data"]["val_ratio"],
        test_ratio=config["data"]["test_ratio"]
    )
    
    # Create model
    model = EventDurationPredictor(**model_config)
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config.get("weight_decay", 0.0001)
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=nn.MSELoss(),
        optimizer=optimizer,
        device=device,
        config=training_config
    )
    
    # Experiment tracking
    if config["experiment"]["tracking"]:
        tracker = ExperimentTracker(config["experiment"]["experiment_dir"])
        exp_id = tracker.start_experiment("event_duration", config)
        trainer.experiment_tracker = tracker
    
    # Train
    history = trainer.train(num_epochs=training_config["max_epochs"])
    
    # Save model
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "event_duration_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": model_config,
        "history": history
    }, model_path)
    
    logger.info(f"Model saved to {model_path}")
    
    if config["experiment"]["tracking"]:
        tracker.finish_experiment({"final_val_loss": history["val_loss"][-1]})


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train ML models")
    parser.add_argument(
        "--model",
        type=str,
        choices=["event", "routine", "time"],
        default="event",
        help="Model to train"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="ml/config/training_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Path to data file"
    )
    
    args = parser.parse_args()
    
    if args.model == "event":
        train_event_model(args.config, args.data)
    else:
        logger.warning(f"Training for {args.model} not yet implemented")


if __name__ == "__main__":
    main()




