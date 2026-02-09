"""
Trainer Builder - Build trainers with fluent interface
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import logging

from ..interfaces.trainer_interface import ITrainer

logger = logging.getLogger(__name__)


class TrainerBuilder:
    """
    Builder for creating trainers
    """
    
    def __init__(self):
        self._model: Optional[nn.Module] = None
        self._trainer_type: str = "standard"
        self._config: Dict[str, Any] = {}
        self._device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def with_model(self, model: nn.Module) -> 'TrainerBuilder':
        """Set model"""
        self._model = model
        return self
    
    def with_type(self, trainer_type: str) -> 'TrainerBuilder':
        """Set trainer type"""
        self._trainer_type = trainer_type
        return self
    
    def with_config(self, config: Dict[str, Any]) -> 'TrainerBuilder':
        """Set trainer configuration"""
        self._config.update(config)
        return self
    
    def with_device(self, device: str) -> 'TrainerBuilder':
        """Set device"""
        self._device = device
        return self
    
    def with_learning_rate(self, lr: float) -> 'TrainerBuilder':
        """Set learning rate"""
        self._config["learning_rate"] = lr
        return self
    
    def with_batch_size(self, batch_size: int) -> 'TrainerBuilder':
        """Set batch size"""
        self._config["batch_size"] = batch_size
        return self
    
    def with_mixed_precision(self, enabled: bool = True) -> 'TrainerBuilder':
        """Enable/disable mixed precision"""
        self._config["use_mixed_precision"] = enabled
        return self
    
    def build(self) -> ITrainer:
        """Build trainer"""
        if self._model is None:
            raise ValueError("Model not set")
        
        from ..factories.trainer_factory import TrainerFactory
        
        trainer = TrainerFactory.create(
            self._trainer_type,
            self._model,
            self._config
        )
        
        return trainer


# Convenience function
def build_trainer(model: nn.Module) -> TrainerBuilder:
    """Start building a trainer"""
    return TrainerBuilder().with_model(model)








