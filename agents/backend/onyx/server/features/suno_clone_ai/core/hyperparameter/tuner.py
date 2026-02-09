"""
Hyperparameter Tuning

Utilities for hyperparameter optimization.
"""

import logging
import itertools
import random
from typing import Dict, List, Any, Callable, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Try to import optuna for Bayesian optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available for Bayesian optimization")


class HyperparameterTuner:
    """Tune hyperparameters."""
    
    def __init__(self, objective_fn: Callable):
        """
        Initialize hyperparameter tuner.
        
        Args:
            objective_fn: Objective function to optimize
        """
        self.objective_fn = objective_fn
        self.best_params = None
        self.best_score = float('-inf')
        self.results = []
    
    def grid_search(
        self,
        search_space: Dict[str, List[Any]],
        max_combinations: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform grid search.
        
        Args:
            search_space: Dictionary of parameter names and value lists
            max_combinations: Maximum combinations to try
            
        Returns:
            Best parameters
        """
        # Generate all combinations
        keys = list(search_space.keys())
        values = list(search_space.values())
        combinations = list(itertools.product(*values))
        
        if max_combinations:
            combinations = combinations[:max_combinations]
        
        logger.info(f"Grid search: {len(combinations)} combinations")
        
        for combo in combinations:
            params = dict(zip(keys, combo))
            score = self.objective_fn(params)
            
            self.results.append({
                'params': params,
                'score': score
            })
            
            if score > self.best_score:
                self.best_score = score
                self.best_params = params
        
        logger.info(f"Best score: {self.best_score}, Best params: {self.best_params}")
        
        return self.best_params
    
    def random_search(
        self,
        search_space: Dict[str, List[Any]],
        n_trials: int = 100
    ) -> Dict[str, Any]:
        """
        Perform random search.
        
        Args:
            search_space: Dictionary of parameter names and value lists
            n_trials: Number of trials
            
        Returns:
            Best parameters
        """
        logger.info(f"Random search: {n_trials} trials")
        
        for _ in range(n_trials):
            params = {}
            for key, values in search_space.items():
                params[key] = random.choice(values)
            
            score = self.objective_fn(params)
            
            self.results.append({
                'params': params,
                'score': score
            })
            
            if score > self.best_score:
                self.best_score = score
                self.best_params = params
        
        logger.info(f"Best score: {self.best_score}, Best params: {self.best_params}")
        
        return self.best_params
    
    def bayesian_optimization(
        self,
        search_space: Dict[str, List[Any]],
        n_trials: int = 100
    ) -> Dict[str, Any]:
        """
        Perform Bayesian optimization.
        
        Args:
            search_space: Dictionary of parameter names and value lists
            n_trials: Number of trials
            
        Returns:
            Best parameters
        """
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available, falling back to random search")
            return self.random_search(search_space, n_trials)
        
        def objective(trial):
            params = {}
            for key, values in search_space.items():
                if isinstance(values[0], (int, float)):
                    if isinstance(values[0], int):
                        params[key] = trial.suggest_int(key, min(values), max(values))
                    else:
                        params[key] = trial.suggest_float(key, min(values), max(values))
                else:
                    params[key] = trial.suggest_categorical(key, values)
            
            return self.objective_fn(params)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        logger.info(f"Best score: {self.best_score}, Best params: {self.best_params}")
        
        return self.best_params


def grid_search(
    objective_fn: Callable,
    search_space: Dict[str, List[Any]],
    **kwargs
) -> Dict[str, Any]:
    """Perform grid search."""
    tuner = HyperparameterTuner(objective_fn)
    return tuner.grid_search(search_space, **kwargs)


def random_search(
    objective_fn: Callable,
    search_space: Dict[str, List[Any]],
    **kwargs
) -> Dict[str, Any]:
    """Perform random search."""
    tuner = HyperparameterTuner(objective_fn)
    return tuner.random_search(search_space, **kwargs)


def bayesian_optimization(
    objective_fn: Callable,
    search_space: Dict[str, List[Any]],
    **kwargs
) -> Dict[str, Any]:
    """Perform Bayesian optimization."""
    tuner = HyperparameterTuner(objective_fn)
    return tuner.bayesian_optimization(search_space, **kwargs)



