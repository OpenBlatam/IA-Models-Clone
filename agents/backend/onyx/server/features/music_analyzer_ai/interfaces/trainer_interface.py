"""
Trainer Interfaces - Define contracts for training
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader


class ITrainer(ABC):
    """
    Base interface for all trainers
    """
    
    @abstractmethod
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train one epoch"""
        pass
    
    @abstractmethod
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model"""
        pass
    
    @abstractmethod
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict[str, float]) -> None:
        """Save training checkpoint"""
        pass
    
    @abstractmethod
    def load_checkpoint(self, path: str) -> tuple:
        """Load training checkpoint"""
        pass


class ITrainingCallback(ABC):
    """
    Interface for training callbacks
    """
    
    @abstractmethod
    def on_epoch_start(self, epoch: int) -> None:
        """Called at start of epoch"""
        pass
    
    @abstractmethod
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Called at end of epoch"""
        pass
    
    @abstractmethod
    def on_batch_end(self, batch_idx: int, loss: float) -> None:
        """Called after each batch"""
        pass


class IOptimizer(ABC):
    """
    Interface for optimizers
    """
    
    @abstractmethod
    def step(self) -> None:
        """Perform optimization step"""
        pass
    
    @abstractmethod
    def zero_grad(self) -> None:
        """Zero gradients"""
        pass
    
    @abstractmethod
    def get_lr(self) -> float:
        """Get current learning rate"""
        pass













