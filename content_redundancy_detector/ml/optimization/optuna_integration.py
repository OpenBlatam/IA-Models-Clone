"""
Optuna Integration
Optuna-based hyperparameter optimization
"""

import torch
from typing import Dict, Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available. Install with: pip install optuna")


class OptunaOptimizer:
    """
    Optuna-based hyperparameter optimization
    """
    
    def __init__(
        self,
        objective_function: Callable,
        study_name: Optional[str] = None,
        direction: str = 'maximize',
    ):
        """
        Initialize Optuna optimizer
        
        Args:
            objective_function: Function to optimize
            study_name: Study name
            direction: 'maximize' or 'minimize'
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is not installed. Install with: pip install optuna")
        
        self.objective_function = objective_function
        self.study = optuna.create_study(
            study_name=study_name,
            direction=direction,
        )
    
    def suggest_hyperparameters(self, trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters using Optuna
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of suggested hyperparameters
        """
        params = {
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'weight_decay': trial.suggest_loguniform('weight_decay', 1e-5, 1e-2),
            'dropout': trial.suggest_uniform('dropout', 0.0, 0.5),
            'optimizer': trial.suggest_categorical('optimizer', ['adam', 'sgd', 'adamw']),
        }
        
        return params
    
    def optimize(self, n_trials: int = 100) -> Dict[str, Any]:
        """
        Run optimization
        
        Args:
            n_trials: Number of trials
            
        Returns:
            Best hyperparameters and study results
        """
        def objective(trial):
            params = self.suggest_hyperparameters(trial)
            score = self.objective_function(params)
            return score
        
        self.study.optimize(objective, n_trials=n_trials)
        
        return {
            'best_params': self.study.best_params,
            'best_value': self.study.best_value,
            'n_trials': len(self.study.trials),
            'study': self.study,
        }



