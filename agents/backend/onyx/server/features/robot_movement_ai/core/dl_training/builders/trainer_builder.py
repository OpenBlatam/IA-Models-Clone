"""
Trainer Builder - Modular Trainer Construction
==============================================

Builder pattern para construir trainers de manera modular y flexible.
"""

import logging
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ...dl_training.trainer import Trainer
from ...dl_training.callbacks import Callback, EarlyStopping, ModelCheckpoint, WandBCallback, TensorBoardCallback
from ...dl_training.optimizers import get_optimizer
from ...dl_training.schedulers import get_scheduler

logger = logging.getLogger(__name__)


class TrainerBuilder:
    """
    Builder para crear trainers de manera modular.
    
    Permite construir trainers paso a paso con diferentes
    configuraciones y componentes.
    """
    
    def __init__(self):
        """Inicializar builder."""
        self._model: Optional[nn.Module] = None
        self._train_loader: Optional[DataLoader] = None
        self._val_loader: Optional[DataLoader] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._scheduler: Optional[Any] = None
        self._loss_fn: Optional[Any] = None
        self._device: Optional[torch.device] = None
        self._use_amp: bool = True
        self._gradient_accumulation_steps: int = 1
        self._max_grad_norm: float = 1.0
        self._callbacks: List[Callback] = []
        self._experiment_name: Optional[str] = None
        self._checkpoint_dir: Optional[str] = None
    
    def with_model(self, model: nn.Module) -> 'TrainerBuilder':
        """Agregar modelo."""
        self._model = model
        return self
    
    def with_data_loaders(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> 'TrainerBuilder':
        """Agregar data loaders."""
        self._train_loader = train_loader
        self._val_loader = val_loader
        return self
    
    def with_optimizer(
        self,
        optimizer_type: str = 'adamw',
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        **kwargs
    ) -> 'TrainerBuilder':
        """Configurar optimizador."""
        if self._model is None:
            raise ValueError("Model must be set before configuring optimizer")
        
        self._optimizer = get_optimizer(
            self._model,
            optimizer_type=optimizer_type,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        return self
    
    def with_scheduler(
        self,
        scheduler_type: str = 'cosine',
        **kwargs
    ) -> 'TrainerBuilder':
        """Configurar scheduler."""
        if self._optimizer is None:
            raise ValueError("Optimizer must be set before configuring scheduler")
        
        self._scheduler = get_scheduler(
            self._optimizer,
            scheduler_type=scheduler_type,
            **kwargs
        )
        return self
    
    def with_loss_function(self, loss_fn: Any) -> 'TrainerBuilder':
        """Agregar función de pérdida."""
        self._loss_fn = loss_fn
        return self
    
    def with_device(self, device: torch.device) -> 'TrainerBuilder':
        """Configurar dispositivo."""
        self._device = device
        return self
    
    def with_mixed_precision(self, use_amp: bool = True) -> 'TrainerBuilder':
        """Configurar mixed precision."""
        self._use_amp = use_amp
        return self
    
    def with_gradient_accumulation(self, steps: int = 1) -> 'TrainerBuilder':
        """Configurar acumulación de gradientes."""
        self._gradient_accumulation_steps = steps
        return self
    
    def with_gradient_clipping(self, max_norm: float = 1.0) -> 'TrainerBuilder':
        """Configurar clipping de gradientes."""
        self._max_grad_norm = max_norm
        return self
    
    def with_early_stopping(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        monitor: str = 'val_loss'
    ) -> 'TrainerBuilder':
        """Agregar early stopping."""
        self._callbacks.append(
            EarlyStopping(patience=patience, min_delta=min_delta, monitor=monitor)
        )
        return self
    
    def with_model_checkpoint(
        self,
        checkpoint_dir: str,
        save_best: bool = True,
        save_last: bool = True,
        monitor: str = 'val_loss'
    ) -> 'TrainerBuilder':
        """Agregar model checkpointing."""
        self._checkpoint_dir = checkpoint_dir
        self._callbacks.append(
            ModelCheckpoint(
                checkpoint_dir=checkpoint_dir,
                save_best=save_best,
                save_last=save_last,
                monitor=monitor
            )
        )
        return self
    
    def with_wandb(
        self,
        project_name: str,
        experiment_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> 'TrainerBuilder':
        """Agregar Weights & Biases tracking."""
        try:
            self._callbacks.append(
                WandBCallback(
                    project_name=project_name,
                    experiment_name=experiment_name,
                    config=config
                )
            )
        except ImportError:
            logger.warning("wandb not available, skipping WandB callback")
        return self
    
    def with_tensorboard(
        self,
        log_dir: str = "runs",
        experiment_name: Optional[str] = None
    ) -> 'TrainerBuilder':
        """Agregar TensorBoard tracking."""
        try:
            self._callbacks.append(
                TensorBoardCallback(
                    log_dir=log_dir,
                    experiment_name=experiment_name
                )
            )
        except ImportError:
            logger.warning("tensorboard not available, skipping TensorBoard callback")
        return self
    
    def with_callback(self, callback: Callback) -> 'TrainerBuilder':
        """Agregar callback personalizado."""
        self._callbacks.append(callback)
        return self
    
    def with_experiment_name(self, name: str) -> 'TrainerBuilder':
        """Configurar nombre del experimento."""
        self._experiment_name = name
        return self
    
    def build(self) -> Trainer:
        """
        Construir trainer.
        
        Returns:
            Trainer configurado
        """
        if self._model is None:
            raise ValueError("Model is required")
        if self._train_loader is None:
            raise ValueError("Train loader is required")
        
        trainer = Trainer(
            model=self._model,
            train_loader=self._train_loader,
            val_loader=self._val_loader,
            optimizer=self._optimizer,
            scheduler=self._scheduler,
            loss_fn=self._loss_fn,
            device=self._device,
            use_amp=self._use_amp,
            gradient_accumulation_steps=self._gradient_accumulation_steps,
            max_grad_norm=self._max_grad_norm,
            callbacks=self._callbacks if self._callbacks else None,
            experiment_name=self._experiment_name,
            checkpoint_dir=self._checkpoint_dir
        )
        
        logger.info("Trainer built successfully")
        return trainer

