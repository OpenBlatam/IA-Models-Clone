"""
Base Trainer Interface
=======================

Abstract base class for trainers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Base trainer configuration."""
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    device: str = "auto"
    use_mixed_precision: bool = True
    grad_clip: float = 1.0
    early_stopping_patience: int = 10


class BaseTrainer(ABC):
    """
    Abstract base class for trainers.
    
    All trainers should implement:
    - train(): Training loop
    - validate(): Validation
    - save_checkpoint(): Save model
    - load_checkpoint(): Load model
    """
    
    def __init__(self, config: TrainerConfig):
        """
        Initialize base trainer.
        
        Args:
            config: Trainer configuration
        """
        self.config = config
        self._logger = logger
        self.history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": []
        }
    
    @abstractmethod
    def train(self, num_epochs: Optional[int] = None) -> Dict[str, Any]:
        """
        Train model.
        
        Args:
            num_epochs: Number of epochs (overrides config)
        
        Returns:
            Training history
        """
        pass
    
    @abstractmethod
    def validate(self) -> Dict[str, float]:
        """
        Validate model.
        
        Returns:
            Validation metrics
        """
        pass
    
    @abstractmethod
    def save_checkpoint(self, path: str, is_best: bool = False):
        """
        Save checkpoint.
        
        Args:
            path: Checkpoint path
            is_best: Whether this is the best model
        """
        pass
    
    @abstractmethod
    def load_checkpoint(self, path: str):
        """
        Load checkpoint.
        
        Args:
            path: Checkpoint path
        """
        pass
    
    def get_history(self) -> Dict[str, list]:
        """Get training history."""
        return self.history




