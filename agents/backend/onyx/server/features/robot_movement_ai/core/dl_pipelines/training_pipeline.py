"""
Training Pipeline - Modular Training Workflow
=============================================

Pipeline modular para el flujo completo de entrenamiento.
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..dl_models.factories import ModelFactory, ModelType
from ..dl_data import TrajectoryDataset, create_dataloader
from ..dl_data.transforms import create_training_transforms, create_validation_transforms
from ..dl_training.builders import TrainerBuilder
from ..dl_training.trainer import Trainer
from ..dl_utils import DeviceManager, TrajectoryLoss, get_loss_function
from ..config import ExperimentConfig, load_yaml_config

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    Pipeline completo de entrenamiento.
    
    Orquesta todo el flujo desde carga de datos hasta
    entrenamiento y evaluación.
    """
    
    def __init__(
        self,
        config: Optional[ExperimentConfig] = None,
        config_path: Optional[str] = None
    ):
        """
        Inicializar pipeline.
        
        Args:
            config: Configuración del experimento
            config_path: Ruta a archivo de configuración YAML
        """
        if config is None and config_path:
            self.config = load_yaml_config(config_path)
        elif config:
            self.config = config
        else:
            raise ValueError("Either config or config_path must be provided")
        
        self.model: Optional[nn.Module] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.trainer: Optional[Trainer] = None
        self.device_manager: Optional[DeviceManager] = None
        
        logger.info("Training Pipeline initialized")
    
    def setup_data(self):
        """Configurar datasets y data loaders."""
        logger.info("Setting up data...")
        
        # Crear datasets
        train_dataset = TrajectoryDataset(
            data_path=self.config.data.data_path,
            trajectory_length=self.config.data.trajectory_length,
            trajectory_dim=self.config.data.trajectory_dim,
            normalize=self.config.data.normalize,
            augment=self.config.data.augment
        )
        
        val_dataset = TrajectoryDataset(
            data_path=self.config.data.data_path.replace('train', 'val'),
            trajectory_length=self.config.data.trajectory_length,
            trajectory_dim=self.config.data.trajectory_dim,
            normalize=self.config.data.normalize,
            augment=False
        )
        
        # Crear data loaders
        self.train_loader = create_dataloader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory
        )
        
        self.val_loader = create_dataloader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory
        )
        
        logger.info("Data setup complete")
    
    def setup_model(self):
        """Configurar modelo."""
        logger.info("Setting up model...")
        
        # Determinar tipo de modelo desde config
        model_type_map = {
            'transformer': ModelType.TRANSFORMER,
            'diffusion': ModelType.DIFFUSION,
            'mlp': ModelType.MLP
        }
        
        model_type = model_type_map.get(
            self.config.model.type.lower(),
            ModelType.TRANSFORMER
        )
        
        # Crear modelo usando factory
        self.model = ModelFactory.create(
            model_type,
            config={
                'input_size': self.config.model.input_size,
                'output_size': self.config.model.output_size,
                'hidden_dim': self.config.model.hidden_dim,
                'num_layers': self.config.model.num_layers,
                'dropout': self.config.model.dropout
            }
        )
        
        logger.info(f"Model created: {self.config.model.name}")
    
    def setup_device(self):
        """Configurar dispositivo."""
        logger.info("Setting up device...")
        
        from ..dl_utils import DeviceManager
        
        self.device_manager = DeviceManager(
            use_mixed_precision=self.config.training.use_amp
        )
        
        # Mover modelo a dispositivo
        if self.model:
            self.model = self.device_manager.move_to_device(self.model)
        
        logger.info(f"Device setup complete: {self.device_manager.device}")
    
    def setup_trainer(self):
        """Configurar trainer."""
        logger.info("Setting up trainer...")
        
        if self.model is None:
            raise ValueError("Model must be set up before trainer")
        if self.train_loader is None:
            raise ValueError("Data loaders must be set up before trainer")
        
        # Crear función de pérdida
        loss_fn = get_loss_function(
            'trajectory',
            position_weight=1.0,
            velocity_weight=0.5,
            smoothness_weight=0.3
        )
        
        # Construir trainer usando builder
        self.trainer = (TrainerBuilder()
            .with_model(self.model)
            .with_data_loaders(self.train_loader, self.val_loader)
            .with_optimizer(
                optimizer_type=self.config.training.optimizer,
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
            .with_scheduler(
                scheduler_type=self.config.training.scheduler,
                T_max=self.config.training.num_epochs
            )
            .with_loss_function(loss_fn)
            .with_device(self.device_manager.device)
            .with_mixed_precision(self.config.training.use_amp)
            .with_gradient_accumulation(self.config.training.gradient_accumulation_steps)
            .with_gradient_clipping(self.config.training.max_grad_norm)
            .with_early_stopping(
                patience=self.config.training.early_stopping_patience,
                monitor='val_loss'
            )
            .with_model_checkpoint(
                checkpoint_dir=self.config.training.checkpoint_dir,
                save_best=self.config.training.save_best
            )
            .with_experiment_name(self.config.experiment_name)
            .build())
        
        logger.info("Trainer setup complete")
    
    def train(self):
        """Ejecutar entrenamiento."""
        if self.trainer is None:
            raise ValueError("Trainer must be set up before training")
        
        logger.info("Starting training...")
        self.trainer.train(
            num_epochs=self.config.training.num_epochs,
            save_best=self.config.training.save_best,
            save_last=True
        )
        logger.info("Training complete")
    
    def run(self):
        """Ejecutar pipeline completo."""
        logger.info("=" * 60)
        logger.info("Starting Training Pipeline")
        logger.info("=" * 60)
        
        # Setup en orden
        self.setup_data()
        self.setup_model()
        self.setup_device()
        self.setup_trainer()
        
        # Entrenar
        self.train()
        
        logger.info("=" * 60)
        logger.info("Training Pipeline Complete")
        logger.info("=" * 60)
    
    def get_results(self) -> Dict[str, Any]:
        """Obtener resultados del entrenamiento."""
        if self.trainer is None:
            return {}
        
        return {
            'train_losses': self.trainer.train_losses,
            'val_losses': self.trainer.val_losses,
            'best_val_loss': self.trainer.best_val_loss,
            'experiment_name': self.config.experiment_name
        }









