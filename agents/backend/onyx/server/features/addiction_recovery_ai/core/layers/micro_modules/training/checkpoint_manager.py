"""
Checkpoint Manager - Ultra-Specific Checkpoint Management
Separated into its own file for maximum modularity
"""

import torch
import torch.nn as nn
import os
import logging
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)


class CheckpointManagerBase(ABC):
    """Base class for checkpoint management"""
    
    def __init__(self, name: str = "CheckpointManager"):
        self.name = name
    
    @abstractmethod
    def save(self, model: nn.Module, path: str, **kwargs) -> Dict[str, Any]:
        """Save checkpoint"""
        pass
    
    @abstractmethod
    def load(self, model: nn.Module, path: str, **kwargs) -> Dict[str, Any]:
        """Load checkpoint"""
        pass


class FullCheckpointManager(CheckpointManagerBase):
    """Save full model checkpoint"""
    
    def __init__(self):
        super().__init__("FullCheckpointManager")
    
    def save(self, model: nn.Module, path: str, optimizer: Optional[Any] = None, epoch: Optional[int] = None, loss: Optional[float] = None, **kwargs) -> Dict[str, Any]:
        """Save full checkpoint"""
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_class': model.__class__.__name__,
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if loss is not None:
            checkpoint['loss'] = loss
        
        checkpoint.update(kwargs)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        torch.save(checkpoint, path)
        logger.info(f"Full checkpoint saved to {path}")
        
        return {
            'path': path,
            'size': os.path.getsize(path),
            'components': list(checkpoint.keys())
        }
    
    def load(self, model: nn.Module, path: str, optimizer: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        """Load full checkpoint"""
        checkpoint = torch.load(path, map_location=kwargs.get('map_location', 'cpu'))
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        result = {
            'model_loaded': True,
            'epoch': checkpoint.get('epoch', None),
            'loss': checkpoint.get('loss', None),
        }
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            result['optimizer_loaded'] = True
        
        logger.info(f"Full checkpoint loaded from {path}")
        
        return result


class StateDictCheckpointManager(CheckpointManagerBase):
    """Save only model state dict"""
    
    def __init__(self):
        super().__init__("StateDictCheckpointManager")
    
    def save(self, model: nn.Module, path: str, **kwargs) -> Dict[str, Any]:
        """Save state dict only"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save(model.state_dict(), path)
        logger.info(f"State dict saved to {path}")
        
        return {
            'path': path,
            'size': os.path.getsize(path),
            'type': 'state_dict'
        }
    
    def load(self, model: nn.Module, path: str, **kwargs) -> Dict[str, Any]:
        """Load state dict only"""
        state_dict = torch.load(path, map_location=kwargs.get('map_location', 'cpu'))
        model.load_state_dict(state_dict)
        logger.info(f"State dict loaded from {path}")
        
        return {
            'model_loaded': True,
            'type': 'state_dict'
        }


class BestModelCheckpointManager(CheckpointManagerBase):
    """Save best model based on metric"""
    
    def __init__(self, metric_name: str = 'loss', mode: str = 'min'):
        super().__init__("BestModelCheckpointManager")
        self.metric_name = metric_name
        self.mode = mode
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.best_path = None
    
    def save(self, model: nn.Module, path: str, metric_value: float, **kwargs) -> Dict[str, Any]:
        """Save if best model"""
        is_best = False
        
        if self.mode == 'min':
            if metric_value < self.best_value:
                is_best = True
                self.best_value = metric_value
        else:
            if metric_value > self.best_value:
                is_best = True
                self.best_value = metric_value
        
        if is_best:
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
            torch.save(model.state_dict(), path)
            self.best_path = path
            logger.info(f"Best model saved to {path} ({self.metric_name}={metric_value:.4f})")
        
        return {
            'is_best': is_best,
            'best_value': self.best_value,
            'current_value': metric_value,
            'path': path if is_best else None
        }
    
    def load(self, model: nn.Module, path: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Load best model"""
        load_path = path or self.best_path
        if load_path is None:
            raise ValueError("No checkpoint path provided")
        
        state_dict = torch.load(load_path, map_location=kwargs.get('map_location', 'cpu'))
        model.load_state_dict(state_dict)
        logger.info(f"Best model loaded from {load_path}")
        
        return {
            'model_loaded': True,
            'best_value': self.best_value,
            'path': load_path
        }


class PeriodicCheckpointManager(CheckpointManagerBase):
    """Save checkpoints periodically"""
    
    def __init__(self, save_interval: int = 10):
        super().__init__("PeriodicCheckpointManager")
        self.save_interval = save_interval
        self.checkpoint_count = 0
    
    def save(self, model: nn.Module, path: str, epoch: int, **kwargs) -> Dict[str, Any]:
        """Save periodically"""
        should_save = (epoch % self.save_interval == 0) or (epoch == 0)
        
        if should_save:
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch
            }, path)
            self.checkpoint_count += 1
            logger.info(f"Periodic checkpoint saved to {path} (epoch {epoch})")
        
        return {
            'saved': should_save,
            'epoch': epoch,
            'checkpoint_count': self.checkpoint_count,
            'path': path if should_save else None
        }
    
    def load(self, model: nn.Module, path: str, **kwargs) -> Dict[str, Any]:
        """Load checkpoint"""
        checkpoint = torch.load(path, map_location=kwargs.get('map_location', 'cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Periodic checkpoint loaded from {path}")
        
        return {
            'model_loaded': True,
            'epoch': checkpoint.get('epoch', None)
        }


# Factory for checkpoint managers
class CheckpointManagerFactory:
    """Factory for creating checkpoint managers"""
    
    _registry = {
        'full': FullCheckpointManager,
        'state_dict': StateDictCheckpointManager,
        'best': BestModelCheckpointManager,
        'periodic': PeriodicCheckpointManager,
    }
    
    @classmethod
    def create(cls, manager_type: str, **kwargs) -> CheckpointManagerBase:
        """Create checkpoint manager"""
        manager_type = manager_type.lower()
        if manager_type not in cls._registry:
            raise ValueError(f"Unknown checkpoint manager type: {manager_type}")
        return cls._registry[manager_type](**kwargs)
    
    @classmethod
    def register(cls, name: str, manager_class: type):
        """Register custom checkpoint manager"""
        cls._registry[name.lower()] = manager_class


__all__ = [
    "CheckpointManagerBase",
    "FullCheckpointManager",
    "StateDictCheckpointManager",
    "BestModelCheckpointManager",
    "PeriodicCheckpointManager",
    "CheckpointManagerFactory",
]



