"""
Hyperparameter Tuning
Grid search and random search for hyperparameters
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
import logging
import itertools

logger = logging.getLogger(__name__)


@dataclass
class SearchSpace:
    """Hyperparameter search space"""
    learning_rate: List[float] = None
    batch_size: List[int] = None
    weight_decay: List[float] = None
    dropout: List[float] = None
    optimizer: List[str] = None
    
    def __post_init__(self):
        """Set defaults if None"""
        if self.learning_rate is None:
            self.learning_rate = [0.001, 0.0001, 0.00001]
        if self.batch_size is None:
            self.batch_size = [16, 32, 64]
        if self.weight_decay is None:
            self.weight_decay = [0.0, 0.0001, 0.001]
        if self.dropout is None:
            self.dropout = [0.0, 0.1, 0.2]
        if self.optimizer is None:
            self.optimizer = ['adam', 'sgd']


class HyperparameterTuner:
    """
    Hyperparameter tuning with grid search and random search
    """
    
    def __init__(
        self,
        search_space: SearchSpace,
        objective_function: Callable,
        search_method: str = 'grid',
        n_trials: int = 10,
    ):
        """
        Initialize tuner
        
        Args:
            search_space: Search space configuration
            objective_function: Function to optimize (returns score)
            search_method: 'grid' or 'random'
            n_trials: Number of trials for random search
        """
        self.search_space = search_space
        self.objective_function = objective_function
        self.search_method = search_method
        self.n_trials = n_trials
        self.results = []
    
    def grid_search(self) -> Dict[str, Any]:
        """
        Perform grid search
        
        Returns:
            Best hyperparameters and results
        """
        # Generate all combinations
        param_names = []
        param_values = []
        
        for field_name, field_value in self.search_space.__dict__.items():
            if field_value is not None:
                param_names.append(field_name)
                param_values.append(field_value)
        
        best_score = float('-inf')
        best_params = None
        
        # Iterate over all combinations
        for combination in itertools.product(*param_values):
            params = dict(zip(param_names, combination))
            
            try:
                score = self.objective_function(params)
                self.results.append({
                    'params': params,
                    'score': score,
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params
                
                logger.info(f"Params: {params}, Score: {score:.4f}")
            
            except Exception as e:
                logger.error(f"Error with params {params}: {e}")
                continue
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': self.results,
        }
    
    def random_search(self) -> Dict[str, Any]:
        """
        Perform random search
        
        Returns:
            Best hyperparameters and results
        """
        best_score = float('-inf')
        best_params = None
        
        for trial in range(self.n_trials):
            # Sample random parameters
            params = {}
            
            for field_name, field_value in self.search_space.__dict__.items():
                if field_value is not None:
                    params[field_name] = np.random.choice(field_value)
            
            try:
                score = self.objective_function(params)
                self.results.append({
                    'params': params,
                    'score': score,
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params
                
                logger.info(f"Trial {trial+1}/{self.n_trials}: {params}, Score: {score:.4f}")
            
            except Exception as e:
                logger.error(f"Error with params {params}: {e}")
                continue
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': self.results,
        }
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run optimization
        
        Returns:
            Best hyperparameters
        """
        if self.search_method == 'grid':
            return self.grid_search()
        elif self.search_method == 'random':
            return self.random_search()
        else:
            raise ValueError(f"Unknown search method: {self.search_method}")



