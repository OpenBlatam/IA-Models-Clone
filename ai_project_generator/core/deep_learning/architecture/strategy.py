"""
Strategy Pattern - Interchangeable Algorithms
=============================================

Provides strategy pattern for interchangeable training and data strategies.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class TrainingStrategy(ABC):
    """Abstract base class for training strategies."""
    
    @abstractmethod
    def train(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute training strategy.
        
        Args:
            model: PyTorch model
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            **kwargs: Additional arguments
            
        Returns:
            Training results
        """
        pass


class StandardTrainingStrategy(TrainingStrategy):
    """Standard training strategy with validation."""
    
    def train(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute standard training."""
        from ..training import Trainer, TrainingConfig, create_optimizer, create_scheduler
        
        # Create trainer
        config = TrainingConfig(**kwargs.get('config', {}))
        optimizer = create_optimizer(model, **kwargs.get('optimizer_config', {}))
        scheduler = create_scheduler(optimizer, **kwargs.get('scheduler_config', {}))
        
        trainer = Trainer(model, config, optimizer, scheduler)
        history = trainer.train(train_loader, val_loader)
        
        return {'history': history}


class FastTrainingStrategy(TrainingStrategy):
    """Fast training strategy (fewer epochs, larger batches)."""
    
    def train(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute fast training."""
        from ..training import Trainer, TrainingConfig, create_optimizer
        
        config = TrainingConfig(
            num_epochs=kwargs.get('num_epochs', 3),
            batch_size=kwargs.get('batch_size', 64),
            use_mixed_precision=False
        )
        optimizer = create_optimizer(model, optimizer_type='adam', learning_rate=1e-3)
        
        trainer = Trainer(model, config, optimizer, None)
        history = trainer.train(train_loader, val_loader)
        
        return {'history': history}


class DataStrategy(ABC):
    """Abstract base class for data strategies."""
    
    @abstractmethod
    def prepare_data(
        self,
        dataset: Dataset,
        **kwargs
    ) -> Dict[str, DataLoader]:
        """
        Prepare data according to strategy.
        
        Args:
            dataset: Dataset
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with data loaders
        """
        pass


class StandardDataStrategy(DataStrategy):
    """Standard data strategy (train/val/test split)."""
    
    def prepare_data(
        self,
        dataset: Dataset,
        **kwargs
    ) -> Dict[str, DataLoader]:
        """Prepare standard data split."""
        from ..data import train_val_test_split, create_dataloader
        
        train_ds, val_ds, test_ds = train_val_test_split(
            dataset,
            train_ratio=kwargs.get('train_ratio', 0.7),
            val_ratio=kwargs.get('val_ratio', 0.15),
            test_ratio=kwargs.get('test_ratio', 0.15)
        )
        
        batch_size = kwargs.get('batch_size', 32)
        
        return {
            'train': create_dataloader(train_ds, batch_size=batch_size, shuffle=True),
            'val': create_dataloader(val_ds, batch_size=batch_size, shuffle=False),
            'test': create_dataloader(test_ds, batch_size=batch_size, shuffle=False)
        }


class CrossValidationDataStrategy(DataStrategy):
    """Cross-validation data strategy."""
    
    def prepare_data(
        self,
        dataset: Dataset,
        **kwargs
    ) -> Dict[str, DataLoader]:
        """Prepare cross-validation splits."""
        from sklearn.model_selection import KFold
        from ..data import create_dataloader
        import numpy as np
        
        k_folds = kwargs.get('k_folds', 5)
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        indices = np.arange(len(dataset))
        folds = {}
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(indices)):
            train_subset = torch.utils.data.Subset(dataset, train_idx)
            val_subset = torch.utils.data.Subset(dataset, val_idx)
            
            batch_size = kwargs.get('batch_size', 32)
            folds[f'fold_{fold_idx}'] = {
                'train': create_dataloader(train_subset, batch_size=batch_size, shuffle=True),
                'val': create_dataloader(val_subset, batch_size=batch_size, shuffle=False)
            }
        
        return folds



