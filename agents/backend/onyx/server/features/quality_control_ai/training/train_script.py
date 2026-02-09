"""
Training Script for Quality Control AI Models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import logging
from pathlib import Path

from ..core.models.autoencoder import create_autoencoder
from ..core.models.defect_classifier import create_defect_classifier
from ..config.training_config import Config, create_default_config_file
from .trainer import ModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dummy_dataset(num_samples: int = 1000, image_size: tuple = (224, 224)):
    """Create dummy dataset for testing"""
    from torch.utils.data import Dataset
    
    class DummyDataset(Dataset):
        def __init__(self, num_samples, image_size):
            self.num_samples = num_samples
            self.image_size = image_size
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # Random image
            image = torch.randn(3, *self.image_size)
            return image
    
    return DummyDataset(num_samples, image_size)


def train_autoencoder(config: Config):
    """Train autoencoder model"""
    logger.info("Training autoencoder...")
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_autoencoder(
        input_channels=config.model.input_channels,
        latent_dim=config.model.latent_dim,
        input_size=config.model.input_size,
        device=device
    )
    
    # Create dummy dataset (replace with real dataset)
    train_dataset = create_dummy_dataset(1000, config.model.input_size)
    val_dataset = create_dummy_dataset(200, config.model.input_size)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory
    )
    
    # Create trainer
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        use_mixed_precision=config.training.use_mixed_precision,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        log_dir=config.experiment.log_dir,
        use_wandb=config.experiment.use_wandb,
        wandb_project=config.experiment.wandb_project
    )
    
    # Optimizer
    if config.optimizer.optimizer_type == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.optimizer.learning_rate,
            weight_decay=config.optimizer.weight_decay
        )
    elif config.optimizer.optimizer_type == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.optimizer.learning_rate,
            weight_decay=config.optimizer.weight_decay
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.optimizer.learning_rate,
            momentum=config.optimizer.momentum,
            weight_decay=config.optimizer.weight_decay
        )
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Scheduler
    scheduler = None
    if config.scheduler.scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.scheduler.T_max,
            eta_min=config.scheduler.eta_min
        )
    elif config.scheduler.scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.scheduler.step_size,
            gamma=config.scheduler.gamma
        )
    elif config.scheduler.scheduler_type == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.scheduler.gamma,
            patience=config.scheduler.step_size
        )
    
    # Train
    trainer.train(
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=config.training.num_epochs,
        scheduler=scheduler,
        clip_grad_norm=config.training.clip_grad_norm,
        save_dir=config.experiment.save_dir,
        save_best=True
    )
    
    logger.info("Training completed!")


def train_classifier(config: Config):
    """Train defect classifier"""
    logger.info("Training defect classifier...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_defect_classifier(
        num_classes=config.model.num_classes,
        model_name=config.model.model_name,
        pretrained=config.model.pretrained,
        device=device
    )
    
    # Create dummy dataset (replace with real dataset)
    train_dataset = create_dummy_dataset(1000, config.model.input_size)
    val_dataset = create_dummy_dataset(200, config.model.input_size)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers
    )
    
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        use_mixed_precision=config.training.use_mixed_precision,
        log_dir=config.experiment.log_dir
    )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay
    )
    
    criterion = nn.CrossEntropyLoss()
    
    trainer.train(
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=config.training.num_epochs,
        save_dir=config.experiment.save_dir
    )
    
    logger.info("Training completed!")


def main():
    parser = argparse.ArgumentParser(description="Train Quality Control AI Models")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["autoencoder", "classifier"],
        default="autoencoder",
        help="Model to train"
    )
    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Create default config file and exit"
    )
    
    args = parser.parse_args()
    
    if args.create_config:
        create_default_config_file(args.config)
        logger.info(f"Default config created at {args.config}")
        return
    
    # Load config
    if not Path(args.config).exists():
        logger.warning(f"Config file {args.config} not found. Creating default...")
        create_default_config_file(args.config)
    
    config = Config.from_yaml(args.config)
    
    # Train
    if args.model == "autoencoder":
        train_autoencoder(config)
    elif args.model == "classifier":
        train_classifier(config)


if __name__ == "__main__":
    main()

