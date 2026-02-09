"""
Cross Validation
================

Cross-validation utilities following best practices.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Callable
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
import logging

logger = logging.getLogger(__name__)


class CrossValidator:
    """
    Cross-validation utility.
    
    Features:
    - K-Fold CV
    - Stratified K-Fold
    - Time Series Split
    - Custom splits
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: Optional[int] = None
    ):
        """
        Initialize cross-validator.
        
        Args:
            n_splits: Number of folds
            shuffle: Whether to shuffle
            random_state: Random state
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self._logger = logger
    
    def k_fold_cv(
        self,
        dataset: torch.utils.data.Dataset,
        trainer_factory: Callable,
        metrics_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Perform K-Fold cross-validation.
        
        Args:
            dataset: PyTorch dataset
            trainer_factory: Function to create trainer
            metrics_callback: Optional callback for metrics
        
        Returns:
            CV results
        """
        kfold = KFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state
        )
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(dataset)))):
            self._logger.info(f"Fold {fold + 1}/{self.n_splits}")
            
            # Create splits
            train_subset = torch.utils.data.Subset(dataset, train_idx)
            val_subset = torch.utils.data.Subset(dataset, val_idx)
            
            # Create dataloaders
            train_loader = torch.utils.data.DataLoader(train_subset, batch_size=32)
            val_loader = torch.utils.data.DataLoader(val_subset, batch_size=32)
            
            # Train
            trainer = trainer_factory(train_loader, val_loader)
            history = trainer.train(num_epochs=50)
            
            # Evaluate
            if metrics_callback:
                metrics = metrics_callback(trainer, val_loader)
            else:
                metrics = trainer.validate()
            
            fold_results.append({
                "fold": fold + 1,
                "train_loss": history["train_loss"][-1],
                "val_loss": history["val_loss"][-1],
                "metrics": metrics
            })
        
        # Aggregate results
        avg_train_loss = np.mean([r["train_loss"] for r in fold_results])
        avg_val_loss = np.mean([r["val_loss"] for r in fold_results])
        std_val_loss = np.std([r["val_loss"] for r in fold_results])
        
        return {
            "fold_results": fold_results,
            "mean_train_loss": float(avg_train_loss),
            "mean_val_loss": float(avg_val_loss),
            "std_val_loss": float(std_val_loss)
        }
    
    def stratified_k_fold_cv(
        self,
        dataset: torch.utils.data.Dataset,
        labels: List[int],
        trainer_factory: Callable
    ) -> Dict[str, Any]:
        """
        Perform Stratified K-Fold CV.
        
        Args:
            dataset: PyTorch dataset
            labels: Labels for stratification
            trainer_factory: Function to create trainer
        
        Returns:
            CV results
        """
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state
        )
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(dataset)), labels)):
            self._logger.info(f"Stratified Fold {fold + 1}/{self.n_splits}")
            
            # Similar to k_fold_cv but with stratified splits
            train_subset = torch.utils.data.Subset(dataset, train_idx)
            val_subset = torch.utils.data.Subset(dataset, val_idx)
            
            train_loader = torch.utils.data.DataLoader(train_subset, batch_size=32)
            val_loader = torch.utils.data.DataLoader(val_subset, batch_size=32)
            
            trainer = trainer_factory(train_loader, val_loader)
            history = trainer.train(num_epochs=50)
            
            fold_results.append({
                "fold": fold + 1,
                "train_loss": history["train_loss"][-1],
                "val_loss": history["val_loss"][-1]
            })
        
        return {
            "fold_results": fold_results,
            "mean_val_loss": float(np.mean([r["val_loss"] for r in fold_results]))
        }




