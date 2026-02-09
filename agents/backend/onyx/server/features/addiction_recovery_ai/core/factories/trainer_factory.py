"""
Trainer Factory
Factory pattern for creating trainers
"""

from typing import Optional, Dict, Any, Type
import torch.nn as nn
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)


class TrainerFactory:
    """
    Factory for creating trainers
    """
    
    _registry: Dict[str, Type] = {}
    
    @classmethod
    def register(cls, name: str, trainer_class: Type):
        """
        Register a trainer class
        
        Args:
            name: Trainer name
            trainer_class: Trainer class
        """
        cls._registry[name] = trainer_class
        logger.info(f"Registered trainer: {name}")
    
    @classmethod
    def create(
        cls,
        trainer_type: str,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Create trainer instance
        
        Args:
            trainer_type: Type of trainer
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Trainer configuration
            
        Returns:
            Trainer instance
        """
        if trainer_type not in cls._registry:
            raise ValueError(f"Unknown trainer type: {trainer_type}. Available: {list(cls._registry.keys())}")
        
        trainer_class = cls._registry[trainer_type]
        config = config or {}
        
        try:
            trainer = trainer_class(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                **config
            )
            return trainer
        except Exception as e:
            logger.error(f"Failed to create trainer {trainer_type}: {e}")
            raise
    
    @classmethod
    def list_trainers(cls) -> list:
        """List all registered trainers"""
        return list(cls._registry.keys())













