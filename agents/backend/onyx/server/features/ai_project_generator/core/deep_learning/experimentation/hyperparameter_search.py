"""
Hyperparameter Search Utilities
================================

Advanced hyperparameter search and optimization.
"""

import logging
from typing import Optional, Dict, Any, List, Callable
import numpy as np
from itertools import product
import random

logger = logging.getLogger(__name__)

# Try to import optuna
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available, using basic search")


class HyperparameterSearch:
    """
    Hyperparameter search with multiple strategies.
    """
    
    def __init__(
        self,
        search_space: Dict[str, List[Any]],
        objective_fn: Callable,
        search_method: str = 'grid',
        n_trials: int = 10
    ):
        """
        Initialize hyperparameter search.
        
        Args:
            search_space: Dictionary with parameter names and possible values
            objective_fn: Objective function to minimize
            search_method: Search method ('grid', 'random', 'bayesian')
            n_trials: Number of trials
        """
        self.search_space = search_space
        self.objective_fn = objective_fn
        self.search_method = search_method
        self.n_trials = n_trials
        self.results = []
    
    def grid_search(self) -> Dict[str, Any]:
        """
        Perform grid search.
        
        Returns:
            Best hyperparameters
        """
        param_names = list(self.search_space.keys())
        param_values = list(self.search_space.values())
        
        best_score = float('inf')
        best_params = None
        
        for combination in product(*param_values):
            params = dict(zip(param_names, combination))
            score = self.objective_fn(params)
            
            self.results.append({
                'params': params,
                'score': score
            })
            
            if score < best_score:
                best_score = score
                best_params = params
        
        logger.info(f"Grid search completed. Best score: {best_score:.4f}")
        return best_params
    
    def random_search(self) -> Dict[str, Any]:
        """
        Perform random search.
        
        Returns:
            Best hyperparameters
        """
        best_score = float('inf')
        best_params = None
        
        for _ in range(self.n_trials):
            params = {}
            for key, values in self.search_space.items():
                params[key] = random.choice(values)
            
            score = self.objective_fn(params)
            
            self.results.append({
                'params': params,
                'score': score
            })
            
            if score < best_score:
                best_score = score
                best_params = params
        
        logger.info(f"Random search completed. Best score: {best_score:.4f}")
        return best_params
    
    def bayesian_search(self) -> Dict[str, Any]:
        """
        Perform Bayesian optimization (using Optuna if available).
        
        Returns:
            Best hyperparameters
        """
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available, falling back to random search")
            return self.random_search()
        
        def objective(trial):
            params = {}
            for key, values in self.search_space.items():
                if isinstance(values[0], (int, float)):
                    if isinstance(values[0], int):
                        params[key] = trial.suggest_int(key, min(values), max(values))
                    else:
                        params[key] = trial.suggest_float(key, min(values), max(values))
                else:
                    params[key] = trial.suggest_categorical(key, values)
            
            score = self.objective_fn(params)
            return score
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)
        
        best_params = study.best_params
        best_score = study.best_value
        
        logger.info(f"Bayesian search completed. Best score: {best_score:.4f}")
        return best_params
    
    def search(self) -> Dict[str, Any]:
        """
        Perform search based on method.
        
        Returns:
            Best hyperparameters
        """
        if self.search_method == 'grid':
            return self.grid_search()
        elif self.search_method == 'random':
            return self.random_search()
        elif self.search_method == 'bayesian':
            return self.bayesian_search()
        else:
            raise ValueError(f"Unknown search method: {self.search_method}")


class ExperimentComparator:
    """
    Compare multiple experiments.
    """
    
    def __init__(self):
        """Initialize comparator."""
        self.experiments = []
    
    def add_experiment(
        self,
        name: str,
        metrics: Dict[str, float],
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add experiment to comparison.
        
        Args:
            name: Experiment name
            metrics: Experiment metrics
            config: Experiment configuration
        """
        self.experiments.append({
            'name': name,
            'metrics': metrics,
            'config': config or {}
        })
    
    def compare(self, metric: str = 'val_loss') -> Dict[str, Any]:
        """
        Compare experiments by metric.
        
        Args:
            metric: Metric to compare
            
        Returns:
            Comparison results
        """
        results = []
        for exp in self.experiments:
            if metric in exp['metrics']:
                results.append({
                    'name': exp['name'],
                    'value': exp['metrics'][metric],
                    'config': exp['config']
                })
        
        results.sort(key=lambda x: x['value'])
        
        return {
            'best': results[0] if results else None,
            'worst': results[-1] if results else None,
            'all': results
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all experiments.
        
        Returns:
            Summary dictionary
        """
        summary = {}
        
        # Collect all metric names
        all_metrics = set()
        for exp in self.experiments:
            all_metrics.update(exp['metrics'].keys())
        
        # Compute statistics for each metric
        for metric in all_metrics:
            values = [exp['metrics'][metric] for exp in self.experiments if metric in exp['metrics']]
            if values:
                summary[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        return summary


class ABTester:
    """
    A/B testing for models.
    """
    
    def __init__(self, model_a: Any, model_b: Any):
        """
        Initialize A/B tester.
        
        Args:
            model_a: Model A
            model_b: Model B
        """
        self.model_a = model_a
        self.model_b = model_b
        self.results_a = []
        self.results_b = []
    
    def test(
        self,
        test_loader: Any,
        metric_fn: Callable,
        num_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run A/B test.
        
        Args:
            test_loader: Test data loader
            metric_fn: Metric function
            num_samples: Number of samples to test
            
        Returns:
            Test results
        """
        import torch
        
        self.model_a.eval()
        self.model_b.eval()
        
        count = 0
        for batch in test_loader:
            if num_samples and count >= num_samples:
                break
            
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0], batch[1]
            elif isinstance(batch, dict):
                inputs = batch.get('input_ids', batch.get('inputs', None))
                targets = batch.get('labels', None)
            else:
                inputs = batch
                targets = None
            
            with torch.no_grad():
                pred_a = self.model_a(inputs)
                pred_b = self.model_b(inputs)
            
            if targets is not None:
                metric_a = metric_fn(pred_a, targets)
                metric_b = metric_fn(pred_b, targets)
                
                self.results_a.append(metric_a)
                self.results_b.append(metric_b)
            
            count += 1
        
        # Statistical test
        from scipy import stats
        
        if len(self.results_a) > 1 and len(self.results_b) > 1:
            t_stat, p_value = stats.ttest_ind(self.results_a, self.results_b)
            
            return {
                'model_a_mean': np.mean(self.results_a),
                'model_b_mean': np.mean(self.results_b),
                'model_a_std': np.std(self.results_a),
                'model_b_std': np.std(self.results_b),
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        return {
            'model_a_mean': np.mean(self.results_a) if self.results_a else 0,
            'model_b_mean': np.mean(self.results_b) if self.results_b else 0
        }


def create_experiment_config(
    base_config: Dict[str, Any],
    variations: Dict[str, List[Any]]
) -> List[Dict[str, Any]]:
    """
    Create multiple experiment configurations from base and variations.
    
    Args:
        base_config: Base configuration
        variations: Dictionary with parameter variations
        
    Returns:
        List of experiment configurations
    """
    configs = []
    param_names = list(variations.keys())
    param_values = list(variations.values())
    
    for combination in product(*param_values):
        config = base_config.copy()
        for name, value in zip(param_names, combination):
            config[name] = value
        configs.append(config)
    
    return configs



