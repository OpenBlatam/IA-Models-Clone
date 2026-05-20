"""
Training Pipeline - Complete Training Workflow
==============================================

High-level pipeline that orchestrates the entire training process:
- Data loading
- Model initialization
- Training loop
- Validation
- Checkpointing
- Experiment tracking
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..core.base import BaseComponent
from ..models import BaseModel, create_model
from ..data import create_dataloader, train_val_test_split
from ..training import (
    Trainer, TrainingConfig, EarlyStopping,
    create_optimizer, create_scheduler
)
from ..evaluation import evaluate_model
from ..config import ConfigManager
from ..utils import get_device, set_seed, ExperimentTracker

logger = logging.getLogger(__name__)


class TrainingPipeline(BaseComponent):
    """
    Complete training pipeline.
    
    Orchestrates the entire training workflow from data to trained model.
    """
    
    def _initialize(self) -> None:
        """Initialize pipeline."""
        self.device = get_device()
        self.config_manager = ConfigManager()
        self.tracker = None
        self.model = None
        self.trainer = None
    
    def setup(
        self,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
        experiment_name: str = "training",
        log_dir: Optional[Path] = None
    ) -> 'TrainingPipeline':
        """
        Setup pipeline with configurations.
        
        Args:
            model_config: Model configuration
            training_config: Training configuration
            experiment_name: Experiment name
            log_dir: Logging directory
            
        Returns:
            Self for method chaining
        """
        # Set seed for reproducibility
        seed = training_config.get('seed', 42)
        set_seed(seed)
        
        # Setup experiment tracking
        if log_dir is None:
            log_dir = Path("logs") / experiment_name
        
        self.tracker = ExperimentTracker(
            experiment_name=experiment_name,
            log_dir=log_dir,
            use_tensorboard=training_config.get('use_tensorboard', True),
            use_wandb=training_config.get('use_wandb', False),
            wandb_project=training_config.get('wandb_project'),
            wandb_config=training_config
        )
        
        # Create model
        model_type = model_config.get('type', 'transformer')
        self.model = create_model(model_type, model_config)
        self.model = self.model.to(self.device)
        
        logger.info(f"Model created: {self.model.get_num_parameters():,} parameters")
        
        # Create trainer config
        train_config = TrainingConfig(
            num_epochs=training_config.get('num_epochs', 10),
            batch_size=training_config.get('batch_size', 32),
            learning_rate=training_config.get('learning_rate', 1e-4),
            weight_decay=training_config.get('weight_decay', 0.01),
            gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 1),
            use_mixed_precision=training_config.get('use_mixed_precision', True),
            device=self.device,
            save_dir=Path(training_config.get('save_dir', 'checkpoints')),
            early_stopping=EarlyStopping(
                patience=training_config.get('early_stopping_patience', 5),
                mode='min'
            ) if training_config.get('use_early_stopping', True) else None
        )
        
        # Create optimizer and scheduler
        optimizer = create_optimizer(
            self.model,
            optimizer_type=training_config.get('optimizer', 'adamw'),
            learning_rate=training_config.get('learning_rate', 1e-4),
            weight_decay=training_config.get('weight_decay', 0.01)
        )
        
        scheduler = create_scheduler(
            optimizer,
            scheduler_type=training_config.get('scheduler', 'cosine'),
            num_epochs=training_config.get('num_epochs', 10)
        )
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            config=train_config,
            optimizer=optimizer,
            scheduler=scheduler
        )
        
        return self
    
    def train(
        self,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: Optional[torch.utils.data.Dataset] = None,
        test_dataset: Optional[torch.utils.data.Dataset] = None
    ) -> Dict[str, Any]:
        """
        Run training pipeline.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            test_dataset: Test dataset (optional)
            
        Returns:
            Dictionary with training results
        """
        if self.trainer is None:
            raise RuntimeError("Pipeline not setup. Call setup() first.")
        
        # Create data loaders
        train_loader = create_dataloader(
            train_dataset,
            batch_size=self.trainer.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = None
        if val_dataset is not None:
            val_loader = create_dataloader(
                val_dataset,
                batch_size=self.trainer.config.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        
        # Train
        logger.info("Starting training...")
        history = self.trainer.train(train_loader, val_loader)
        
        # Evaluate on test set if provided
        test_metrics = None
        if test_dataset is not None:
            test_loader = create_dataloader(
                test_dataset,
                batch_size=self.trainer.config.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
            
            logger.info("Evaluating on test set...")
            test_metrics, test_info = evaluate_model(
                self.model,
                test_loader,
                self.device,
                task_type=self.config.get('task_type', 'classification')
            )
        
        # Close tracking
        if self.tracker:
            self.tracker.close()
        
        return {
            'history': history,
            'test_metrics': test_metrics.to_dict() if test_metrics else None,
            'model': self.model
        }
    
    def train_from_config(
        self,
        config_path: Path,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: Optional[torch.utils.data.Dataset] = None
    ) -> Dict[str, Any]:
        """
        Train from configuration file.
        
        Args:
            config_path: Path to configuration file
            train_dataset: Training dataset
            val_dataset: Validation dataset
            
        Returns:
            Training results
        """
        config = self.config_manager.load(config_path)
        
        model_config = config.get('model', {})
        training_config = config.get('training', {})
        
        self.setup(model_config, training_config)
        return self.train(train_dataset, val_dataset)



