#!/usr/bin/env python3
"""
Advanced Hyperparameter Optimization System for Frontier Model Training
Provides comprehensive hyperparameter tuning, optimization strategies, and automated ML pipeline optimization.
"""

import os
import json
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
import sqlite3
from contextlib import contextmanager
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, TimeSeriesSplit
from sklearn.model_selection import validation_curve, learning_curve
import optuna
from optuna import create_study, Trial, Study
from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner, HyperbandPruner
import hyperopt
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL
import scipy.optimize
from scipy.optimize import differential_evolution, dual_annealing
import skopt
from skopt import gp_minimize, forest_minimize, gbrt_minimize
from skopt.space import Real, Integer, Categorical
from skopt.acquisition import gaussian_ei, gaussian_pi, gaussian_lcb
import joblib
import pickle
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

console = Console()

class OptimizationStrategy(Enum):
    """Optimization strategies."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    TREE_PARZEN_ESTIMATOR = "tree_parzen_estimator"
    CMA_ES = "cma_es"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    SIMULATED_ANNEALING = "simulated_annealing"
    GENETIC_ALGORITHM = "genetic_algorithm"
    PARTICLE_SWARM = "particle_swarm"
    GRADIENT_BASED = "gradient_based"
    MULTI_OBJECTIVE = "multi_objective"
    ADAPTIVE_OPTIMIZATION = "adaptive_optimization"

class PruningStrategy(Enum):
    """Pruning strategies."""
    NO_PRUNING = "no_pruning"
    MEDIAN_PRUNER = "median_pruner"
    SUCCESSIVE_HALVING = "successive_halving"
    HYPERBAND = "hyperband"
    ASHA = "asha"
    ADAPTIVE_PRUNING = "adaptive_pruning"

class ObjectiveFunction(Enum):
    """Objective functions."""
    ACCURACY = "accuracy"
    F1_SCORE = "f1_score"
    PRECISION = "precision"
    RECALL = "recall"
    ROC_AUC = "roc_auc"
    LOG_LOSS = "log_loss"
    CUSTOM = "custom"
    MULTI_OBJECTIVE = "multi_objective"

class SearchSpace(Enum):
    """Search space types."""
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"
    MIXED = "mixed"
    HIERARCHICAL = "hierarchical"

@dataclass
class HyperparameterConfig:
    """Hyperparameter optimization configuration."""
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.BAYESIAN_OPTIMIZATION
    pruning_strategy: PruningStrategy = PruningStrategy.MEDIAN_PRUNER
    objective_function: ObjectiveFunction = ObjectiveFunction.ACCURACY
    search_space: SearchSpace = SearchSpace.MIXED
    n_trials: int = 100
    n_jobs: int = -1
    timeout: int = 3600
    enable_early_stopping: bool = True
    enable_parallel_optimization: bool = True
    enable_multi_objective: bool = True
    enable_adaptive_sampling: bool = True
    enable_warm_start: bool = True
    enable_pruning: bool = True
    enable_visualization: bool = True
    enable_export: bool = True
    device: str = "auto"

@dataclass
class HyperparameterTrial:
    """Hyperparameter trial result."""
    trial_id: int
    parameters: Dict[str, Any]
    objective_value: float
    objective_values: Dict[str, float]
    status: str
    duration: float
    created_at: datetime

@dataclass
class OptimizationResult:
    """Optimization result."""
    result_id: str
    strategy: OptimizationStrategy
    best_parameters: Dict[str, Any]
    best_objective: float
    best_objectives: Dict[str, float]
    trials: List[HyperparameterTrial]
    optimization_history: List[Dict[str, Any]]
    search_space: Dict[str, Any]
    created_at: datetime

class SearchSpaceBuilder:
    """Search space builder."""
    
    def __init__(self, config: HyperparameterConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def build_search_space(self, model_type: str) -> Dict[str, Any]:
        """Build search space for specific model type."""
        console.print(f"[blue]Building search space for {model_type}...[/blue]")
        
        if model_type == 'random_forest':
            return self._build_random_forest_space()
        elif model_type == 'gradient_boosting':
            return self._build_gradient_boosting_space()
        elif model_type == 'svm':
            return self._build_svm_space()
        elif model_type == 'neural_network':
            return self._build_neural_network_space()
        elif model_type == 'xgboost':
            return self._build_xgboost_space()
        elif model_type == 'lightgbm':
            return self._build_lightgbm_space()
        else:
            return self._build_default_space()
    
    def _build_random_forest_space(self) -> Dict[str, Any]:
        """Build Random Forest search space."""
        return {
            'n_estimators': (10, 1000),
            'max_depth': (1, 50),
            'min_samples_split': (2, 20),
            'min_samples_leaf': (1, 10),
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False],
            'criterion': ['gini', 'entropy']
        }
    
    def _build_gradient_boosting_space(self) -> Dict[str, Any]:
        """Build Gradient Boosting search space."""
        return {
            'n_estimators': (10, 500),
            'learning_rate': (0.01, 1.0),
            'max_depth': (1, 20),
            'min_samples_split': (2, 20),
            'min_samples_leaf': (1, 10),
            'subsample': (0.5, 1.0),
            'max_features': ['sqrt', 'log2', None]
        }
    
    def _build_svm_space(self) -> Dict[str, Any]:
        """Build SVM search space."""
        return {
            'C': (0.001, 1000),
            'gamma': (0.0001, 1.0),
            'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
            'degree': (2, 5),
            'coef0': (0.0, 1.0)
        }
    
    def _build_neural_network_space(self) -> Dict[str, Any]:
        """Build Neural Network search space."""
        return {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
            'activation': ['relu', 'tanh', 'logistic'],
            'solver': ['adam', 'lbfgs', 'sgd'],
            'alpha': (0.0001, 1.0),
            'learning_rate': ['constant', 'adaptive', 'invscaling'],
            'max_iter': (100, 1000)
        }
    
    def _build_xgboost_space(self) -> Dict[str, Any]:
        """Build XGBoost search space."""
        return {
            'n_estimators': (10, 1000),
            'learning_rate': (0.01, 1.0),
            'max_depth': (1, 20),
            'min_child_weight': (1, 10),
            'subsample': (0.5, 1.0),
            'colsample_bytree': (0.5, 1.0),
            'reg_alpha': (0.0, 1.0),
            'reg_lambda': (0.0, 1.0)
        }
    
    def _build_lightgbm_space(self) -> Dict[str, Any]:
        """Build LightGBM search space."""
        return {
            'n_estimators': (10, 1000),
            'learning_rate': (0.01, 1.0),
            'max_depth': (1, 20),
            'num_leaves': (10, 100),
            'min_child_samples': (1, 100),
            'subsample': (0.5, 1.0),
            'colsample_bytree': (0.5, 1.0),
            'reg_alpha': (0.0, 1.0),
            'reg_lambda': (0.0, 1.0)
        }
    
    def _build_default_space(self) -> Dict[str, Any]:
        """Build default search space."""
        return {
            'param1': (0.1, 10.0),
            'param2': (1, 100),
            'param3': ['option1', 'option2', 'option3']
        }

class ObjectiveFunctionBuilder:
    """Objective function builder."""
    
    def __init__(self, config: HyperparameterConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def build_objective_function(self, model_class: Any, X: np.ndarray, y: np.ndarray, 
                               search_space: Dict[str, Any]) -> Callable:
        """Build objective function for optimization."""
        console.print(f"[blue]Building objective function for {self.config.objective_function.value}...[/blue]")
        
        if self.config.objective_function == ObjectiveFunction.ACCURACY:
            return self._build_accuracy_objective(model_class, X, y, search_space)
        elif self.config.objective_function == ObjectiveFunction.F1_SCORE:
            return self._build_f1_objective(model_class, X, y, search_space)
        elif self.config.objective_function == ObjectiveFunction.MULTI_OBJECTIVE:
            return self._build_multi_objective(model_class, X, y, search_space)
        else:
            return self._build_accuracy_objective(model_class, X, y, search_space)
    
    def _build_accuracy_objective(self, model_class: Any, X: np.ndarray, y: np.ndarray, 
                                search_space: Dict[str, Any]) -> Callable:
        """Build accuracy-based objective function."""
        def objective(trial):
            try:
                # Sample parameters
                params = {}
                for param_name, param_range in search_space.items():
                    if isinstance(param_range, tuple):
                        if isinstance(param_range[0], int) and isinstance(param_range[1], int):
                            params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                        else:
                            params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
                    elif isinstance(param_range, list):
                        params[param_name] = trial.suggest_categorical(param_name, param_range)
                
                # Create and train model
                model = model_class(**params)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
                
                return cv_scores.mean()
                
            except Exception as e:
                self.logger.error(f"Objective function failed: {e}")
                return 0.0
        
        return objective
    
    def _build_f1_objective(self, model_class: Any, X: np.ndarray, y: np.ndarray, 
                          search_space: Dict[str, Any]) -> Callable:
        """Build F1-score-based objective function."""
        def objective(trial):
            try:
                # Sample parameters
                params = {}
                for param_name, param_range in search_space.items():
                    if isinstance(param_range, tuple):
                        if isinstance(param_range[0], int) and isinstance(param_range[1], int):
                            params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                        else:
                            params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
                    elif isinstance(param_range, list):
                        params[param_name] = trial.suggest_categorical(param_name, param_range)
                
                # Create and train model
                model = model_class(**params)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')
                
                return cv_scores.mean()
                
            except Exception as e:
                self.logger.error(f"Objective function failed: {e}")
                return 0.0
        
        return objective
    
    def _build_multi_objective(self, model_class: Any, X: np.ndarray, y: np.ndarray, 
                            search_space: Dict[str, Any]) -> Callable:
        """Build multi-objective function."""
        def objective(trial):
            try:
                # Sample parameters
                params = {}
                for param_name, param_range in search_space.items():
                    if isinstance(param_range, tuple):
                        if isinstance(param_range[0], int) and isinstance(param_range[1], int):
                            params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                        else:
                            params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
                    elif isinstance(param_range, list):
                        params[param_name] = trial.suggest_categorical(param_name, param_range)
                
                # Create and train model
                model = model_class(**params)
                
                # Cross-validation for multiple metrics
                accuracy_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
                f1_scores = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')
                precision_scores = cross_val_score(model, X, y, cv=5, scoring='precision_weighted')
                recall_scores = cross_val_score(model, X, y, cv=5, scoring='recall_weighted')
                
                # Return weighted combination
                return (accuracy_scores.mean() + f1_scores.mean() + 
                       precision_scores.mean() + recall_scores.mean()) / 4
                
            except Exception as e:
                self.logger.error(f"Multi-objective function failed: {e}")
                return 0.0
        
        return objective

class OptimizationEngine:
    """Main optimization engine."""
    
    def __init__(self, config: HyperparameterConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.search_space_builder = SearchSpaceBuilder(config)
        self.objective_builder = ObjectiveFunctionBuilder(config)
    
    def optimize_hyperparameters(self, model_class: Any, X: np.ndarray, y: np.ndarray, 
                               model_type: str = "default") -> OptimizationResult:
        """Optimize hyperparameters for given model."""
        console.print(f"[blue]Starting hyperparameter optimization using {self.config.optimization_strategy.value}...[/blue]")
        
        start_time = time.time()
        result_id = f"opt_{int(time.time())}"
        
        # Build search space
        search_space = self.search_space_builder.build_search_space(model_type)
        
        # Build objective function
        objective = self.objective_builder.build_objective_function(model_class, X, y, search_space)
        
        # Run optimization
        if self.config.optimization_strategy == OptimizationStrategy.BAYESIAN_OPTIMIZATION:
            result = self._bayesian_optimization(objective, search_space)
        elif self.config.optimization_strategy == OptimizationStrategy.TREE_PARZEN_ESTIMATOR:
            result = self._tpe_optimization(objective, search_space)
        elif self.config.optimization_strategy == OptimizationStrategy.RANDOM_SEARCH:
            result = self._random_search(objective, search_space)
        elif self.config.optimization_strategy == OptimizationStrategy.GRID_SEARCH:
            result = self._grid_search(objective, search_space)
        else:
            result = self._bayesian_optimization(objective, search_space)
        
        optimization_time = time.time() - start_time
        
        # Create optimization result
        optimization_result = OptimizationResult(
            result_id=result_id,
            strategy=self.config.optimization_strategy,
            best_parameters=result['best_parameters'],
            best_objective=result['best_objective'],
            best_objectives=result.get('best_objectives', {}),
            trials=result['trials'],
            optimization_history=result.get('optimization_history', []),
            search_space=search_space,
            created_at=datetime.now()
        )
        
        console.print(f"[green]Hyperparameter optimization completed in {optimization_time:.2f} seconds[/green]")
        console.print(f"[blue]Best objective: {result['best_objective']:.4f}[/blue]")
        console.print(f"[blue]Best parameters: {result['best_parameters']}[/blue]")
        
        return optimization_result
    
    def _bayesian_optimization(self, objective: Callable, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Bayesian optimization using Optuna."""
        # Create study
        sampler = TPESampler(seed=42)
        pruner = MedianPruner() if self.config.enable_pruning else None
        
        study = create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner
        )
        
        # Optimize
        study.optimize(objective, n_trials=self.config.n_trials, timeout=self.config.timeout)
        
        # Extract results
        trials = []
        for trial in study.trials:
            trial_result = HyperparameterTrial(
                trial_id=trial.number,
                parameters=trial.params,
                objective_value=trial.value if trial.value is not None else 0.0,
                objective_values={'objective': trial.value if trial.value is not None else 0.0},
                status=trial.state.name,
                duration=trial.duration.total_seconds() if trial.duration else 0.0,
                created_at=datetime.now()
            )
            trials.append(trial_result)
        
        return {
            'best_parameters': study.best_params,
            'best_objective': study.best_value,
            'best_objectives': {'objective': study.best_value},
            'trials': trials,
            'optimization_history': [{'trial': t.number, 'value': t.value} for t in study.trials if t.value is not None]
        }
    
    def _tpe_optimization(self, objective: Callable, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Tree-structured Parzen Estimator optimization."""
        # Use Optuna with TPE sampler
        sampler = TPESampler(seed=42)
        study = create_study(direction='maximize', sampler=sampler)
        
        study.optimize(objective, n_trials=self.config.n_trials, timeout=self.config.timeout)
        
        # Extract results
        trials = []
        for trial in study.trials:
            trial_result = HyperparameterTrial(
                trial_id=trial.number,
                parameters=trial.params,
                objective_value=trial.value if trial.value is not None else 0.0,
                objective_values={'objective': trial.value if trial.value is not None else 0.0},
                status=trial.state.name,
                duration=trial.duration.total_seconds() if trial.duration else 0.0,
                created_at=datetime.now()
            )
            trials.append(trial_result)
        
        return {
            'best_parameters': study.best_params,
            'best_objective': study.best_value,
            'best_objectives': {'objective': study.best_value},
            'trials': trials,
            'optimization_history': [{'trial': t.number, 'value': t.value} for t in study.trials if t.value is not None]
        }
    
    def _random_search(self, objective: Callable, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Random search optimization."""
        sampler = RandomSampler(seed=42)
        study = create_study(direction='maximize', sampler=sampler)
        
        study.optimize(objective, n_trials=self.config.n_trials, timeout=self.config.timeout)
        
        # Extract results
        trials = []
        for trial in study.trials:
            trial_result = HyperparameterTrial(
                trial_id=trial.number,
                parameters=trial.params,
                objective_value=trial.value if trial.value is not None else 0.0,
                objective_values={'objective': trial.value if trial.value is not None else 0.0},
                status=trial.state.name,
                duration=trial.duration.total_seconds() if trial.duration else 0.0,
                created_at=datetime.now()
            )
            trials.append(trial_result)
        
        return {
            'best_parameters': study.best_params,
            'best_objective': study.best_value,
            'best_objectives': {'objective': study.best_value},
            'trials': trials,
            'optimization_history': [{'trial': t.number, 'value': t.value} for t in study.trials if t.value is not None]
        }
    
    def _grid_search(self, objective: Callable, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Grid search optimization."""
        # Simplified grid search
        best_params = None
        best_value = float('-inf')
        trials = []
        
        # Generate grid points (simplified)
        param_combinations = self._generate_grid_combinations(search_space)
        
        for i, params in enumerate(param_combinations[:self.config.n_trials]):
            try:
                # Create trial object
                trial = type('Trial', (), {
                    'params': params,
                    'suggest_int': lambda name, low, high: params[name],
                    'suggest_float': lambda name, low, high: params[name],
                    'suggest_categorical': lambda name, choices: params[name]
                })()
                
                value = objective(trial)
                
                trial_result = HyperparameterTrial(
                    trial_id=i,
                    parameters=params,
                    objective_value=value,
                    objective_values={'objective': value},
                    status='COMPLETE',
                    duration=0.0,
                    created_at=datetime.now()
                )
                trials.append(trial_result)
                
                if value > best_value:
                    best_value = value
                    best_params = params
                    
            except Exception as e:
                self.logger.error(f"Grid search trial failed: {e}")
                continue
        
        return {
            'best_parameters': best_params or {},
            'best_objective': best_value,
            'best_objectives': {'objective': best_value},
            'trials': trials,
            'optimization_history': [{'trial': i, 'value': t.objective_value} for i, t in enumerate(trials)]
        }
    
    def _generate_grid_combinations(self, search_space: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate grid combinations."""
        import itertools
        
        # Convert search space to lists
        param_lists = {}
        for param_name, param_range in search_space.items():
            if isinstance(param_range, tuple):
                if isinstance(param_range[0], int) and isinstance(param_range[1], int):
                    param_lists[param_name] = list(range(param_range[0], param_range[1] + 1, max(1, (param_range[1] - param_range[0]) // 5)))
                else:
                    param_lists[param_name] = np.linspace(param_range[0], param_range[1], 5).tolist()
            elif isinstance(param_range, list):
                param_lists[param_name] = param_range
        
        # Generate combinations
        combinations = []
        for combination in itertools.product(*param_lists.values()):
            param_dict = dict(zip(param_lists.keys(), combination))
            combinations.append(param_dict)
        
        return combinations

class HyperparameterOptimizationSystem:
    """Main hyperparameter optimization system."""
    
    def __init__(self, config: HyperparameterConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.optimization_engine = OptimizationEngine(config)
        
        # Initialize database
        self.db_path = self._init_database()
        
        # Results storage
        self.optimization_results: Dict[str, OptimizationResult] = {}
    
    def _init_database(self) -> str:
        """Initialize hyperparameter optimization database."""
        db_path = Path("./hyperparameter_optimization.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS optimization_trials (
                    trial_id INTEGER PRIMARY KEY,
                    result_id TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    objective_value REAL NOT NULL,
                    objective_values TEXT NOT NULL,
                    status TEXT NOT NULL,
                    duration REAL NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS optimization_results (
                    result_id TEXT PRIMARY KEY,
                    strategy TEXT NOT NULL,
                    best_parameters TEXT NOT NULL,
                    best_objective REAL NOT NULL,
                    best_objectives TEXT NOT NULL,
                    search_space TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
        
        return str(db_path)
    
    def run_optimization_experiment(self, model_class: Any, X: np.ndarray, y: np.ndarray, 
                                  model_type: str = "default") -> OptimizationResult:
        """Run complete hyperparameter optimization experiment."""
        console.print("[blue]Starting hyperparameter optimization experiment...[/blue]")
        
        # Run optimization
        result = self.optimization_engine.optimize_hyperparameters(model_class, X, y, model_type)
        
        # Store result
        self.optimization_results[result.result_id] = result
        
        # Save to database
        self._save_optimization_result(result)
        
        console.print(f"[green]Optimization experiment completed[/green]")
        console.print(f"[blue]Best objective: {result.best_objective:.4f}[/blue]")
        console.print(f"[blue]Number of trials: {len(result.trials)}[/blue]")
        
        return result
    
    def _save_optimization_result(self, result: OptimizationResult):
        """Save optimization result to database."""
        with sqlite3.connect(self.db_path) as conn:
            # Save trials
            for trial in result.trials:
                conn.execute("""
                    INSERT OR REPLACE INTO optimization_trials 
                    (trial_id, result_id, parameters, objective_value, objective_values,
                     status, duration, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trial.trial_id,
                    result.result_id,
                    json.dumps(trial.parameters),
                    trial.objective_value,
                    json.dumps(trial.objective_values),
                    trial.status,
                    trial.duration,
                    trial.created_at.isoformat()
                ))
            
            # Save result
            conn.execute("""
                INSERT OR REPLACE INTO optimization_results 
                (result_id, strategy, best_parameters, best_objective,
                 best_objectives, search_space, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                result.result_id,
                result.strategy.value,
                json.dumps(result.best_parameters),
                result.best_objective,
                json.dumps(result.best_objectives),
                json.dumps(result.search_space),
                result.created_at.isoformat()
            ))
    
    def visualize_optimization_results(self, result: OptimizationResult, 
                                    output_path: str = None) -> str:
        """Visualize optimization results."""
        if output_path is None:
            output_path = f"hyperparameter_optimization_{result.result_id}.png"
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Optimization history
        if result.optimization_history:
            trials = [h['trial'] for h in result.optimization_history]
            values = [h['value'] for h in result.optimization_history]
            
            axes[0, 0].plot(trials, values, 'b-', alpha=0.7)
            axes[0, 0].set_title('Optimization History')
            axes[0, 0].set_xlabel('Trial')
            axes[0, 0].set_ylabel('Objective Value')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Parameter importance (simplified)
        if result.best_parameters:
            param_names = list(result.best_parameters.keys())
            param_values = list(result.best_parameters.values())
            
            # Normalize values for visualization
            normalized_values = []
            for val in param_values:
                if isinstance(val, (int, float)):
                    normalized_values.append(abs(val))
                else:
                    normalized_values.append(1.0)
            
            axes[0, 1].bar(param_names, normalized_values)
            axes[0, 1].set_title('Best Parameters')
            axes[0, 1].set_ylabel('Normalized Value')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Trial distribution
        if result.trials:
            trial_values = [t.objective_value for t in result.trials if t.objective_value is not None]
            
            axes[1, 0].hist(trial_values, bins=20, alpha=0.7)
            axes[1, 0].set_title('Trial Value Distribution')
            axes[1, 0].set_xlabel('Objective Value')
            axes[1, 0].set_ylabel('Frequency')
        
        # Performance metrics
        metrics = {
            'Best Objective': result.best_objective,
            'Number of Trials': len(result.trials),
            'Strategy': len(result.strategy.value),
            'Search Space Size': len(result.search_space)
        }
        
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        axes[1, 1].bar(metric_names, metric_values)
        axes[1, 1].set_title('Optimization Summary')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        console.print(f"[green]Optimization visualization saved: {output_path}[/green]")
        return output_path
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get optimization summary."""
        if not self.optimization_results:
            return {'total_experiments': 0}
        
        total_experiments = len(self.optimization_results)
        
        # Calculate average metrics
        best_objectives = [result.best_objective for result in self.optimization_results.values()]
        trial_counts = [len(result.trials) for result in self.optimization_results.values()]
        
        avg_objective = np.mean(best_objectives)
        avg_trials = np.mean(trial_counts)
        
        # Best performing experiment
        best_result = max(self.optimization_results.values(), 
                         key=lambda x: x.best_objective)
        
        return {
            'total_experiments': total_experiments,
            'average_best_objective': avg_objective,
            'average_trials': avg_trials,
            'best_objective': best_result.best_objective,
            'best_experiment_id': best_result.result_id,
            'strategies_used': list(set(result.strategy.value for result in self.optimization_results.values())),
            'total_trials': sum(trial_counts)
        }

def main():
    """Main function for hyperparameter optimization CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization System")
    parser.add_argument("--optimization-strategy", type=str,
                       choices=["bayesian_optimization", "tree_parzen_estimator", "random_search", "grid_search"],
                       default="bayesian_optimization", help="Optimization strategy")
    parser.add_argument("--pruning-strategy", type=str,
                       choices=["no_pruning", "median_pruner", "successive_halving"],
                       default="median_pruner", help="Pruning strategy")
    parser.add_argument("--objective-function", type=str,
                       choices=["accuracy", "f1_score", "precision", "recall"],
                       default="accuracy", help="Objective function")
    parser.add_argument("--n-trials", type=int, default=100,
                       help="Number of trials")
    parser.add_argument("--timeout", type=int, default=3600,
                       help="Timeout in seconds")
    parser.add_argument("--model-type", type=str,
                       choices=["random_forest", "gradient_boosting", "svm", "neural_network"],
                       default="random_forest", help="Model type")
    parser.add_argument("--num-samples", type=int, default=1000,
                       help="Number of samples")
    parser.add_argument("--num-classes", type=int, default=3,
                       help="Number of classes")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use")
    
    args = parser.parse_args()
    
    # Create hyperparameter configuration
    config = HyperparameterConfig(
        optimization_strategy=OptimizationStrategy(args.optimization_strategy),
        pruning_strategy=PruningStrategy(args.pruning_strategy),
        objective_function=ObjectiveFunction(args.objective_function),
        n_trials=args.n_trials,
        timeout=args.timeout,
        device=args.device
    )
    
    # Create hyperparameter optimization system
    opt_system = HyperparameterOptimizationSystem(config)
    
    # Create sample dataset
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    
    X, y = make_classification(
        n_samples=args.num_samples,
        n_features=20,
        n_classes=args.num_classes,
        n_redundant=0,
        n_informative=15,
        random_state=42
    )
    
    # Select model class
    if args.model_type == 'random_forest':
        model_class = RandomForestClassifier
    elif args.model_type == 'gradient_boosting':
        model_class = GradientBoostingClassifier
    elif args.model_type == 'svm':
        model_class = SVC
    elif args.model_type == 'neural_network':
        model_class = MLPClassifier
    else:
        model_class = RandomForestClassifier
    
    # Run optimization experiment
    result = opt_system.run_optimization_experiment(model_class, X, y, args.model_type)
    
    # Show results
    console.print(f"[green]Hyperparameter optimization completed[/green]")
    console.print(f"[blue]Strategy: {result.strategy.value}[/blue]")
    console.print(f"[blue]Best objective: {result.best_objective:.4f}[/blue]")
    console.print(f"[blue]Number of trials: {len(result.trials)}[/blue]")
    console.print(f"[blue]Best parameters: {result.best_parameters}[/blue]")
    
    # Create visualization
    opt_system.visualize_optimization_results(result)
    
    # Show summary
    summary = opt_system.get_optimization_summary()
    console.print(f"[blue]Summary: {summary}[/blue]")

if __name__ == "__main__":
    main()
