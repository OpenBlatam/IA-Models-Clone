"""
Hyperparameter Optimization using Optuna and Grid Search
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Callable
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available. Install: pip install optuna")


class HyperparameterOptimizer:
    """Hyperparameter optimization using Optuna"""
    
    def __init__(
        self,
        model_factory: Callable,
        train_loader,
        val_loader,
        device: Optional[torch.device] = None,
        n_trials: int = 50,
        direction: str = "minimize"
    ):
        """
        Initialize hyperparameter optimizer
        
        Args:
            model_factory: Function to create model with hyperparameters
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to use
            n_trials: Number of optimization trials
            direction: Optimization direction (minimize/maximize)
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for hyperparameter optimization")
        
        self.model_factory = model_factory
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_trials = n_trials
        self.direction = direction
        
        self.study = None
        self.best_params = None
        self.best_value = None
        
        logger.info("HyperparameterOptimizer initialized")
    
    def objective(self, trial) -> float:
        """Objective function for optimization"""
        # Suggest hyperparameters
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128, 256])
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        
        # Create model with hyperparameters
        model = self.model_factory(hidden_size=hidden_size, dropout=dropout)
        model = model.to(self.device)
        
        # Create optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        criterion = nn.BCELoss()
        
        # Train for a few epochs
        model.train()
        for epoch in range(5):  # Quick training
            for batch in self.train_loader:
                inputs = batch[0].to(self.device)
                targets = batch[1].to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
        
        # Validate
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch[0].to(self.device)
                targets = batch[1].to(self.device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run hyperparameter optimization
        
        Returns:
            Dictionary with best parameters and value
        """
        self.study = optuna.create_study(direction=self.direction)
        self.study.optimize(self.objective, n_trials=self.n_trials)
        
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value
        
        logger.info(f"Optimization complete. Best value: {self.best_value:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return {
            "best_params": self.best_params,
            "best_value": self.best_value,
            "n_trials": self.n_trials
        }
    
    def get_trial_history(self) -> List[Dict[str, Any]]:
        """Get trial history"""
        if self.study is None:
            return []
        
        history = []
        for trial in self.study.trials:
            history.append({
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": trial.state.name
            })
        
        return history


class GridSearchOptimizer:
    """Grid search hyperparameter optimization"""
    
    def __init__(
        self,
        model_factory: Callable,
        train_loader,
        val_loader,
        param_grid: Dict[str, List[Any]],
        device: Optional[torch.device] = None
    ):
        """
        Initialize grid search optimizer
        
        Args:
            model_factory: Function to create model
            train_loader: Training data loader
            val_loader: Validation data loader
            param_grid: Parameter grid dictionary
            device: Device to use
        """
        self.model_factory = model_factory
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.param_grid = param_grid
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.best_params = None
        self.best_score = None
        self.results = []
        
        logger.info("GridSearchOptimizer initialized")
    
    def _evaluate_params(self, params: Dict[str, Any]) -> float:
        """Evaluate parameters"""
        model = self.model_factory(**params)
        model = model.to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=params.get("lr", 1e-3))
        criterion = nn.BCELoss()
        
        # Quick training
        model.train()
        for epoch in range(3):
            for batch in self.train_loader:
                inputs = batch[0].to(self.device)
                targets = batch[1].to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
        
        # Validate
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch[0].to(self.device)
                targets = batch[1].to(self.device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def search(self) -> Dict[str, Any]:
        """
        Run grid search
        
        Returns:
            Dictionary with best parameters and score
        """
        from itertools import product
        
        # Generate all parameter combinations
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        
        best_score = float("inf")
        best_params = None
        
        for combination in product(*values):
            params = dict(zip(keys, combination))
            score = self._evaluate_params(params)
            
            self.results.append({
                "params": params,
                "score": score
            })
            
            if score < best_score:
                best_score = score
                best_params = params
        
        self.best_params = best_params
        self.best_score = best_score
        
        logger.info(f"Grid search complete. Best score: {self.best_score:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "n_combinations": len(self.results)
        }

