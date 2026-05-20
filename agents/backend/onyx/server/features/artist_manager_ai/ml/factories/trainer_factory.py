"""
Trainer Factory
===============

Factory for creating trainer instances.
"""

import torch
import torch.optim as optim
import logging
from typing import Dict, Any, Optional

from ..base import BaseTrainer, TrainerConfig
from ..training import Trainer, DistributedTrainer

logger = logging.getLogger(__name__)


class TrainerFactory:
    """
    Factory for creating trainer instances.
    
    Supports:
    - Standard trainer
    - Distributed trainer
    - Custom optimizers
    - Custom schedulers
    """
    
    def __init__(self):
        """Initialize factory."""
        self._logger = logger
    
    def create(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        config: Optional[Dict[str, Any]] = None,
        use_distributed: bool = False,
        optimizer_type: str = "adam",
        scheduler_type: Optional[str] = None
    ) -> BaseTrainer:
        """
        Create trainer instance.
        
        Args:
            model: PyTorch model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            criterion: Loss function
            config: Trainer configuration
            use_distributed: Whether to use distributed training
            optimizer_type: Optimizer type (adam, sgd, adamw)
            scheduler_type: Scheduler type (step, cosine, None)
        
        Returns:
            Trainer instance
        """
        # Create config
        if config is None:
            config = self._get_default_config()
        
        trainer_config = TrainerConfig(**config)
        
        # Create optimizer
        optimizer = self._create_optimizer(model, optimizer_type, config)
        
        # Device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create trainer
        if use_distributed:
            trainer = DistributedTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                config=config,
                use_ddp=True
            )
        else:
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                config=config
            )
        
        return trainer
    
    def _create_optimizer(
        self,
        model: torch.nn.Module,
        optimizer_type: str,
        config: Dict[str, Any]
    ) -> optim.Optimizer:
        """
        Create optimizer.
        
        Args:
            model: Model
            optimizer_type: Optimizer type
            config: Configuration
        
        Returns:
            Optimizer
        """
        lr = config.get("learning_rate", 0.001)
        weight_decay = config.get("weight_decay", 0.0001)
        
        optimizers = {
            "adam": optim.Adam,
            "adamw": optim.AdamW,
            "sgd": optim.SGD
        }
        
        optimizer_class = optimizers.get(optimizer_type.lower(), optim.Adam)
        
        if optimizer_type.lower() == "sgd":
            return optimizer_class(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            return optimizer_class(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default trainer configuration."""
        return {
            "batch_size": 32,
            "learning_rate": 0.001,
            "num_epochs": 100,
            "use_mixed_precision": True,
            "grad_clip": 1.0,
            "early_stopping_patience": 10
        }




