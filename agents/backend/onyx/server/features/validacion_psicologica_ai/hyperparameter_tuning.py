"""
Hyperparameter Tuning
=====================
Automated hyperparameter optimization
"""

from typing import Dict, Any, List, Optional, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import structlog
import numpy as np
from dataclasses import dataclass
from enum import Enum

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger = structlog.get_logger()
    logger.warning("optuna not available, using basic grid search")

logger = structlog.get_logger()


class SearchStrategy(str, Enum):
    """Hyperparameter search strategies"""
    GRID = "grid"
    RANDOM = "random"
    BAYESIAN = "bayesian"  # Requires optuna


@dataclass
class HyperparameterSpace:
    """Hyperparameter search space"""
    learning_rate: tuple = (1e-6, 1e-3)  # (min, max) for continuous
    batch_size: List[int] = None  # List for discrete
    num_epochs: List[int] = None
    dropout: tuple = (0.0, 0.5)
    weight_decay: tuple = (1e-6, 1e-2)
    
    def __post_init__(self):
        if self.batch_size is None:
            self.batch_size = [8, 16, 32, 64]
        if self.num_epochs is None:
            self.num_epochs = [3, 5, 10]


class HyperparameterTuner:
    """
    Hyperparameter tuning with multiple strategies
    """
    
    def __init__(
        self,
        strategy: SearchStrategy = SearchStrategy.RANDOM,
        n_trials: int = 20,
        direction: str = "minimize"  # "minimize" or "maximize"
    ):
        """
        Initialize tuner
        
        Args:
            strategy: Search strategy
            n_trials: Number of trials
            direction: Optimization direction
        """
        self.strategy = strategy
        self.n_trials = n_trials
        self.direction = direction
        self.best_params = None
        self.best_score = float('inf') if direction == "minimize" else float('-inf')
        self.trial_results = []
        
        logger.info(
            "HyperparameterTuner initialized",
            strategy=strategy.value,
            n_trials=n_trials
        )
    
    def optimize(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        search_space: HyperparameterSpace
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters
        
        Args:
            objective_fn: Objective function that takes params and returns score
            search_space: Search space definition
            
        Returns:
            Best hyperparameters
        """
        if self.strategy == SearchStrategy.BAYESIAN and OPTUNA_AVAILABLE:
            return self._bayesian_optimization(objective_fn, search_space)
        elif self.strategy == SearchStrategy.RANDOM:
            return self._random_search(objective_fn, search_space)
        else:
            return self._grid_search(objective_fn, search_space)
    
    def _random_search(
        self,
        objective_fn: Callable,
        search_space: HyperparameterSpace
    ) -> Dict[str, Any]:
        """Random search optimization"""
        for trial in range(self.n_trials):
            # Sample hyperparameters
            params = {
                "learning_rate": np.random.uniform(*search_space.learning_rate),
                "batch_size": np.random.choice(search_space.batch_size),
                "num_epochs": np.random.choice(search_space.num_epochs),
                "dropout": np.random.uniform(*search_space.dropout),
                "weight_decay": np.random.uniform(*search_space.weight_decay)
            }
            
            # Evaluate
            try:
                score = objective_fn(params)
                self.trial_results.append({"params": params, "score": score})
                
                # Update best
                if (self.direction == "minimize" and score < self.best_score) or \
                   (self.direction == "maximize" and score > self.best_score):
                    self.best_score = score
                    self.best_params = params
                    logger.info(f"Trial {trial+1}: New best score = {score:.4f}")
                
            except Exception as e:
                logger.error(f"Error in trial {trial+1}", error=str(e))
                continue
        
        logger.info("Random search completed", best_score=self.best_score)
        return self.best_params
    
    def _grid_search(
        self,
        objective_fn: Callable,
        search_space: HyperparameterSpace
    ) -> Dict[str, Any]:
        """Grid search optimization"""
        # Create grid
        learning_rates = np.logspace(
            np.log10(search_space.learning_rate[0]),
            np.log10(search_space.learning_rate[1]),
            num=5
        )
        
        trial = 0
        for lr in learning_rates:
            for bs in search_space.batch_size:
                for epochs in search_space.num_epochs:
                    for dropout in np.linspace(*search_space.dropout, num=3):
                        params = {
                            "learning_rate": float(lr),
                            "batch_size": bs,
                            "num_epochs": epochs,
                            "dropout": float(dropout),
                            "weight_decay": np.random.uniform(*search_space.weight_decay)
                        }
                        
                        try:
                            score = objective_fn(params)
                            self.trial_results.append({"params": params, "score": score})
                            
                            if (self.direction == "minimize" and score < self.best_score) or \
                               (self.direction == "maximize" and score > self.best_score):
                                self.best_score = score
                                self.best_params = params
                            
                            trial += 1
                            if trial >= self.n_trials:
                                break
                        except Exception as e:
                            logger.error(f"Error in grid search trial", error=str(e))
                            continue
        
        return self.best_params
    
    def _bayesian_optimization(
        self,
        objective_fn: Callable,
        search_space: HyperparameterSpace
    ) -> Dict[str, Any]:
        """Bayesian optimization using Optuna"""
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available, falling back to random search")
            return self._random_search(objective_fn, search_space)
        
        def optuna_objective(trial):
            params = {
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    search_space.learning_rate[0],
                    search_space.learning_rate[1],
                    log=True
                ),
                "batch_size": trial.suggest_categorical("batch_size", search_space.batch_size),
                "num_epochs": trial.suggest_categorical("num_epochs", search_space.num_epochs),
                "dropout": trial.suggest_float("dropout", *search_space.dropout),
                "weight_decay": trial.suggest_float(
                    "weight_decay",
                    search_space.weight_decay[0],
                    search_space.weight_decay[1],
                    log=True
                )
            }
            
            return objective_fn(params)
        
        study = optuna.create_study(direction=self.direction)
        study.optimize(optuna_objective, n_trials=self.n_trials)
        
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        logger.info("Bayesian optimization completed", best_score=self.best_score)
        
        return self.best_params


class LearningRateFinder:
    """Learning rate finder for optimal learning rate"""
    
    @staticmethod
    def find_learning_rate(
        model: nn.Module,
        train_loader: DataLoader,
        loss_fn: nn.Module,
        min_lr: float = 1e-8,
        max_lr: float = 1.0,
        num_iterations: int = 100
    ) -> float:
        """
        Find optimal learning rate using LR range test
        
        Args:
            model: Model
            train_loader: Training data loader
            loss_fn: Loss function
            min_lr: Minimum learning rate
            max_lr: Maximum learning rate
            num_iterations: Number of iterations
            
        Returns:
            Optimal learning rate
        """
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=min_lr)
        
        lrs = np.logspace(np.log10(min_lr), np.log10(max_lr), num_iterations)
        losses = []
        
        for i, (batch, lr) in enumerate(zip(train_loader, lrs)):
            if i >= num_iterations:
                break
            
            # Update learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            # Forward pass
            outputs = model(**batch)
            loss = loss_fn(outputs, batch.get("labels"))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        # Find steepest descent point
        # Simplified: return learning rate with minimum loss
        best_idx = np.argmin(losses)
        optimal_lr = lrs[best_idx]
        
        logger.info("Learning rate found", optimal_lr=optimal_lr)
        
        return optimal_lr


# Global tuner instance
hyperparameter_tuner = HyperparameterTuner()




