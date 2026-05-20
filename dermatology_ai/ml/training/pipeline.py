"""
Training Pipeline
High-level pipeline for complete training workflow
"""

from typing import Dict, Any, Optional, List
import logging
from pathlib import Path

from .trainer_refactored import RefactoredTrainer
from .callbacks import (
    EarlyStoppingCallback,
    ModelCheckpointCallback,
    LearningRateSchedulerCallback,
    MetricsLoggingCallback
)
from .losses import MultiTaskLoss
from .optimizers import get_optimizer, get_scheduler
from .metrics import MetricCalculator
from data.datasets import SkinDataset
from training.trainer import create_data_loaders

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    Complete training pipeline
    Handles entire training workflow from data to model
    """
    
    def __init__(
        self,
        model,
        train_dataset: SkinDataset,
        val_dataset: Optional[SkinDataset] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config or {}
        
        # Create data loaders
        self.loaders = create_data_loaders(
            train_dataset,
            val_dataset,
            batch_size=self.config.get('batch_size', 32),
            num_workers=self.config.get('num_workers', 4),
            pin_memory=self.config.get('pin_memory', True),
            use_distributed=self.config.get('use_distributed', False)
        )
        
        # Create trainer
        self.trainer = RefactoredTrainer(
            model=model,
            train_loader=self.loaders['train'],
            val_loader=self.loaders.get('val'),
            device=self.config.get('device', 'cuda'),
            use_mixed_precision=self.config.get('use_mixed_precision', True),
            gradient_clip_val=self.config.get('gradient_clip_val', 1.0),
            gradient_accumulation_steps=self.config.get('gradient_accumulation_steps', 1),
            use_ddp=self.config.get('use_ddp', False),
            enable_profiling=self.config.get('enable_profiling', False),
            enable_anomaly_detection=self.config.get('enable_anomaly_detection', False)
        )
        
        # Setup callbacks
        self._setup_callbacks()
    
    def _setup_callbacks(self):
        """Setup training callbacks"""
        callbacks = []
        
        # Early stopping
        if self.config.get('early_stopping', {}).get('enabled', True):
            callbacks.append(EarlyStoppingCallback(
                monitor=self.config.get('early_stopping', {}).get('monitor', 'val_loss'),
                patience=self.config.get('early_stopping', {}).get('patience', 10),
                mode=self.config.get('early_stopping', {}).get('mode', 'min')
            ))
        
        # Model checkpointing
        checkpoint_dir = self.config.get('checkpointing', {}).get('save_dir', './checkpoints')
        if checkpoint_dir:
            callbacks.append(ModelCheckpointCallback(
                checkpoint_dir=checkpoint_dir,
                save_best=self.config.get('checkpointing', {}).get('save_best', True),
                save_frequency=self.config.get('checkpointing', {}).get('save_frequency', 10),
                monitor=self.config.get('early_stopping', {}).get('monitor', 'val_loss'),
                mode=self.config.get('early_stopping', {}).get('mode', 'min')
            ))
        
        # Metrics logging
        callbacks.append(MetricsLoggingCallback(
            log_frequency=self.config.get('log_frequency', 1)
        ))
        
        # Add callbacks to trainer
        for callback in callbacks:
            self.trainer.add_callback(callback)
    
    def train(self) -> Dict[str, Any]:
        """
        Run complete training pipeline
        
        Returns:
            Dictionary with training results
        """
        # Create loss function
        loss_config = self.config.get('loss', {})
        loss_fn = MultiTaskLoss(
            condition_weight=loss_config.get('condition_weight', 1.0),
            metric_weight=loss_config.get('metric_weight', 1.0),
            condition_loss_type=loss_config.get('condition_loss_type', 'bce'),
            metric_loss_type=loss_config.get('metric_loss_type', 'mse')
        )
        
        # Create optimizer
        optimizer_config = self.config.get('optimizer', {})
        optimizer = get_optimizer(
            self.model,
            optimizer_name=optimizer_config.get('name', 'adamw'),
            learning_rate=optimizer_config.get('learning_rate', 1e-4),
            weight_decay=optimizer_config.get('weight_decay', 1e-4),
            **optimizer_config.get('extra_params', {})
        )
        
        # Create scheduler
        scheduler = None
        scheduler_config = self.config.get('scheduler')
        if scheduler_config:
            scheduler = get_scheduler(
                optimizer,
                scheduler_name=scheduler_config.get('name', 'cosine'),
                num_epochs=self.config.get('num_epochs', 100),
                **scheduler_config.get('extra_params', {})
            )
        
        # Train
        self.trainer.fit(
            optimizer=optimizer,
            num_epochs=self.config.get('num_epochs', 100),
            scheduler=scheduler,
            criterion=loss_fn
        )
        
        # Return results
        return {
            'training_history': self.trainer.training_history,
            'best_epoch': self.trainer.current_epoch,
            'final_metrics': self.trainer.training_history['val_metrics'][-1] if self.trainer.training_history['val_metrics'] else {}
        }
    
    @classmethod
    def from_config(
        cls,
        model,
        config: Dict[str, Any],
        train_images: List,
        val_images: List,
        train_labels: Optional[Dict] = None,
        val_labels: Optional[Dict] = None
    ) -> 'TrainingPipeline':
        """
        Create training pipeline from configuration
        
        Args:
            model: PyTorch model
            config: Configuration dictionary
            train_images: Training images
            val_images: Validation images
            train_labels: Training labels
            val_labels: Validation labels
            
        Returns:
            TrainingPipeline instance
        """
        from ml.data.dataset_factory import DatasetFactory
        
        # Create datasets
        datasets = DatasetFactory.create_datasets_from_config(
            config=config,
            train_images=train_images,
            val_images=val_images,
            train_labels=train_labels,
            val_labels=val_labels
        )
        
        return cls(
            model=model,
            train_dataset=datasets['train'],
            val_dataset=datasets.get('val'),
            config=config
        )













