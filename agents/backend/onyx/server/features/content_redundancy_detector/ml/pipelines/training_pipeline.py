"""
Training Pipeline
High-level training pipeline builder
"""

import torch
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from ..models.mobilenet.factory import MobileNetFactory
from ..models.mobilenet.config import MobileNetConfig, TrainingConfig
from ..training import (
    MobileNetTrainer,
    OptimizerFactory,
    SchedulerFactory,
    LossFactory,
    CallbackList,
    EarlyStoppingCallback,
    ModelCheckpointCallback,
    ExperimentTrackingCallback,
    CheckpointManager,
)
from ..utils import ConfigLoader

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    High-level training pipeline
    Orchestrates the entire training process
    """
    
    def __init__(
        self,
        config_path: Optional[Path] = None,
        config_dict: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize training pipeline
        
        Args:
            config_path: Path to YAML config file
            config_dict: Configuration dictionary
        """
        if config_path:
            self.config = ConfigLoader.load_yaml(config_path)
        elif config_dict:
            self.config = config_dict
        else:
            raise ValueError("Either config_path or config_dict must be provided")
        
        self.model = None
        self.trainer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def setup(self) -> 'TrainingPipeline':
        """Setup model and trainer"""
        # Create model
        model_config = MobileNetConfig.from_dict(self.config['model'])
        self.model = MobileNetFactory.create_model(model_config, device=self.device)
        
        # Create trainer
        training_config = TrainingConfig(**self.config['training'])
        self.trainer = MobileNetTrainer(self.model, self.device, training_config)
        
        # Setup callbacks
        self._setup_callbacks()
        
        logger.info("Training pipeline setup complete")
        return self
    
    def _setup_callbacks(self) -> None:
        """Setup training callbacks"""
        callbacks = CallbackList()
        
        # Early stopping
        if self.config.get('callbacks', {}).get('early_stopping', {}).get('enabled', False):
            early_stop_config = self.config['callbacks']['early_stopping']
            callbacks.append(EarlyStoppingCallback(
                monitor=early_stop_config.get('monitor', 'val_loss'),
                patience=early_stop_config.get('patience', 5),
            ))
        
        # Checkpoint
        if self.config.get('callbacks', {}).get('checkpoint', {}).get('enabled', False):
            checkpoint_config = self.config['callbacks']['checkpoint']
            callbacks.append(ModelCheckpointCallback(
                checkpoint_dir=Path(checkpoint_config.get('checkpoint_dir', './checkpoints')),
                monitor=checkpoint_config.get('monitor', 'val_loss'),
            ))
        
        # Experiment tracking
        if self.config.get('callbacks', {}).get('experiment_tracking', {}).get('enabled', False):
            tracking_config = self.config['callbacks']['experiment_tracking']
            callbacks.append(ExperimentTrackingCallback(
                tracker_type=tracking_config.get('tracker_type', 'wandb'),
                project_name=tracking_config.get('project_name'),
                run_name=tracking_config.get('run_name'),
            ))
        
        self.trainer.add_callback(callbacks)
    
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
    ) -> Dict[str, Any]:
        """
        Run training
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Training history
        """
        if self.trainer is None:
            self.setup()
        
        return self.trainer.train(train_loader, val_loader)

