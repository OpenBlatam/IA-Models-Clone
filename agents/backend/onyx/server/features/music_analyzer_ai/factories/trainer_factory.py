"""
Trainer Factory - Create trainer instances
"""

from typing import Dict, Any, Optional, Type
import logging
import torch
import torch.nn as nn

from ..interfaces.trainer_interface import ITrainer

# Lazy imports to avoid circular dependencies
try:
    from ..models.music_transformer import MusicModelTrainer
except ImportError:
    MusicModelTrainer = None

try:
    from ..training.fast_trainer import FastMusicTrainer
except ImportError:
    FastMusicTrainer = None

logger = logging.getLogger(__name__)


class TrainerFactory:
    """
    Factory for creating trainer instances
    """
    
    _trainer_registry: Dict[str, Type[ITrainer]] = {}
    
    @classmethod
    def register(cls, name: str, trainer_class: Type[ITrainer]):
        """Register a trainer class"""
        cls._trainer_registry[name] = trainer_class
        logger.info(f"Registered trainer: {name}")
    
    @classmethod
    def create(
        cls,
        trainer_type: str,
        model: nn.Module,
        config: Optional[Dict[str, Any]] = None
    ) -> ITrainer:
        """
        Create trainer instance
        
        Args:
            trainer_type: Type of trainer
            model: Model to train
            config: Trainer configuration
        
        Returns:
            Trainer instance
        """
        if trainer_type not in cls._trainer_registry:
            raise ValueError(f"Unknown trainer type: {trainer_type}")
        
        if config is None:
            config = {}
        
        trainer_class = cls._trainer_registry[trainer_type]
        trainer = trainer_class(model=model, **config)
        
        logger.info(f"Created {trainer_type} trainer")
        return trainer
    
    @classmethod
    def list_available(cls) -> list:
        """List available trainer types"""
        return list(cls._trainer_registry.keys())


# Register default trainers (if available)
if MusicModelTrainer is not None:
    TrainerFactory.register("standard", MusicModelTrainer)
if FastMusicTrainer is not None:
    TrainerFactory.register("fast", FastMusicTrainer)


def create_trainer(
    trainer_type: str = "standard",
    model: Optional[nn.Module] = None,
    config: Optional[Dict[str, Any]] = None
) -> ITrainer:
    """
    Convenience function to create a trainer
    
    Args:
        trainer_type: Type of trainer
        model: Model to train
        config: Trainer configuration
    
    Returns:
        Trainer instance
    """
    if model is None:
        raise ValueError("Model is required")
    
    return TrainerFactory.create(trainer_type, model, config)

