"""
Advanced Optimization Engine for Export IA
State-of-the-art optimization techniques for model training and inference
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, CosineAnnealingWarmRestarts, 
    ReduceLROnPlateau, OneCycleLR, CyclicLR
)
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
import math
import random
from collections import defaultdict, deque
import optuna
from optuna.samplers import TPESampler, CmaEsSampler
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler, HyperBandScheduler
from ray.tune.search.optuna import OptunaSearch
import wandb
from wandb.sweep import Sweep

logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    """Configuration for advanced optimization"""
    # Hyperparameter optimization
    optimization_method: str = "optuna"  # optuna, ray_tune, wandb_sweep
    n_trials: int = 100
    timeout: int = 3600  # seconds
    direction: str = "minimize"  # minimize, maximize
    
    # Search space
    search_space: Dict[str, Any] = None
    
    # Pruning
    enable_pruning: bool = True
    pruning_method: str = "median"  # median, successive_halving
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    
    # Multi-objective optimization
    multi_objective: bool = False
    objectives: List[str] = None
    
    # Resource allocation
    max_parallel_trials: int = 4
    gpu_per_trial: float = 0.25
    
    # Advanced techniques
    use_ensemble: bool = False
    use_meta_learning: bool = False
    use_neural_architecture_search: bool = False

class AdvancedOptimizer:
    """Advanced optimization engine with multiple techniques"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.study = None
        self.best_params = None
        self.optimization_history = []
        self.performance_metrics = defaultdict(list)
        
    def optimize_hyperparameters(self, objective_function: Callable, 
                                search_space: Dict[str, Any] = None) -> Dict[str, Any]:
        """Optimize hyperparameters using specified method"""
        
        if search_space is None:
            search_space = self.config.search_space
            
        if self.config.optimization_method == "optuna":
            return self._optimize_with_optuna(objective_function, search_space)
        elif self.config.optimization_method == "ray_tune":
            return self._optimize_with_ray_tune(objective_function, search_space)
        elif self.config.optimization_method == "wandb_sweep":
            return self._optimize_with_wandb_sweep(objective_function, search_space)
        else:
            raise ValueError(f"Unknown optimization method: {self.config.optimization_method}")
            
    def _optimize_with_optuna(self, objective_function: Callable, 
                             search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize using Optuna"""
        
        # Create study
        sampler = TPESampler(seed=42)
        pruner = MedianPruner() if self.config.enable_pruning else None
        
        self.study = optuna.create_study(
            direction=self.config.direction,
            sampler=sampler,
            pruner=pruner
        )
        
        # Define objective wrapper
        def objective(trial):
            # Sample hyperparameters
            params = {}
            for param_name, param_config in search_space.items():
                if param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name, param_config['low'], param_config['high'], 
                        log=param_config.get('log', False)
                    )
                elif param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name, param_config['low'], param_config['high'],
                        log=param_config.get('log', False)
                    )
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config['choices']
                    )
                    
            # Evaluate objective
            try:
                result = objective_function(params)
                
                # Handle multi-objective
                if self.config.multi_objective:
                    return result
                else:
                    return result['loss'] if isinstance(result, dict) else result
                    
            except Exception as e:
                logger.error(f"Trial failed: {e}")
                return float('inf') if self.config.direction == 'minimize' else float('-inf')
                
        # Optimize
        self.study.optimize(
            objective, 
            n_trials=self.config.n_trials,
            timeout=self.config.timeout
        )
        
        # Get best parameters
        self.best_params = self.study.best_params
        best_value = self.study.best_value
        
        return {
            'best_params': self.best_params,
            'best_value': best_value,
            'n_trials': len(self.study.trials),
            'study': self.study
        }
        
    def _optimize_with_ray_tune(self, objective_function: Callable, 
                               search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize using Ray Tune"""
        
        # Initialize Ray
        if not ray.is_initialized():
            ray.init()
            
        # Convert search space to Ray Tune format
        ray_search_space = {}
        for param_name, param_config in search_space.items():
            if param_config['type'] == 'float':
                ray_search_space[param_name] = tune.uniform(
                    param_config['low'], param_config['high']
                )
            elif param_config['type'] == 'int':
                ray_search_space[param_name] = tune.randint(
                    param_config['low'], param_config['high']
                )
            elif param_config['type'] == 'categorical':
                ray_search_space[param_name] = tune.choice(param_config['choices'])
                
        # Create scheduler
        scheduler = ASHAScheduler(
            metric="loss",
            mode="min" if self.config.direction == "minimize" else "max",
            max_t=100,
            grace_period=10
        )
        
        # Define trainable function
        def trainable(config):
            result = objective_function(config)
            tune.report(**result)
            
        # Run optimization
        analysis = tune.run(
            trainable,
            config=ray_search_space,
            num_samples=self.config.n_trials,
            scheduler=scheduler,
            resources_per_trial={"cpu": 2, "gpu": self.config.gpu_per_trial},
            local_dir="./ray_results"
        )
        
        # Get best result
        best_config = analysis.best_config
        best_result = analysis.best_result
        
        return {
            'best_params': best_config,
            'best_value': best_result['loss'],
            'analysis': analysis
        }
        
    def _optimize_with_wandb_sweep(self, objective_function: Callable, 
                                  search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize using Weights & Biases Sweep"""
        
        # Convert search space to WandB format
        wandb_config = {}
        for param_name, param_config in search_space.items():
            if param_config['type'] == 'float':
                wandb_config[param_name] = {
                    'distribution': 'uniform',
                    'min': param_config['low'],
                    'max': param_config['high']
                }
            elif param_config['type'] == 'int':
                wandb_config[param_name] = {
                    'distribution': 'int_uniform',
                    'min': param_config['low'],
                    'max': param_config['high']
                }
            elif param_config['type'] == 'categorical':
                wandb_config[param_name] = {
                    'values': param_config['choices']
                }
                
        # Create sweep configuration
        sweep_config = {
            'method': 'bayes',
            'metric': {'name': 'loss', 'goal': 'minimize'},
            'parameters': wandb_config
        }
        
        # Define sweep function
        def sweep_function():
            with wandb.init() as run:
                config = wandb.config
                result = objective_function(dict(config))
                wandb.log(result)
                
        # Create and run sweep
        sweep_id = wandb.sweep(sweep_config, project="export-ia-optimization")
        wandb.agent(sweep_id, sweep_function, count=self.config.n_trials)
        
        # Get best run
        api = wandb.Api()
        sweep = api.sweep(f"export-ia-optimization/{sweep_id}")
        best_run = sweep.best_run()
        
        return {
            'best_params': best_run.config,
            'best_value': best_run.summary['loss'],
            'sweep_id': sweep_id
        }

class LearningRateScheduler:
    """Advanced learning rate scheduling with multiple strategies"""
    
    def __init__(self, optimizer: optim.Optimizer, config: Dict[str, Any]):
        self.optimizer = optimizer
        self.config = config
        self.scheduler = self._create_scheduler()
        self.current_lr = self.optimizer.param_groups[0]['lr']
        
    def _create_scheduler(self):
        """Create learning rate scheduler based on config"""
        scheduler_type = self.config.get('type', 'cosine')
        
        if scheduler_type == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('T_max', 100),
                eta_min=self.config.get('eta_min', 0)
            )
        elif scheduler_type == 'cosine_warm_restarts':
            return CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.get('T_0', 10),
                T_mult=self.config.get('T_mult', 2),
                eta_min=self.config.get('eta_min', 0)
            )
        elif scheduler_type == 'plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode=self.config.get('mode', 'min'),
                factor=self.config.get('factor', 0.5),
                patience=self.config.get('patience', 10),
                min_lr=self.config.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'one_cycle':
            return OneCycleLR(
                self.optimizer,
                max_lr=self.config.get('max_lr', 1e-3),
                total_steps=self.config.get('total_steps', 1000),
                pct_start=self.config.get('pct_start', 0.3)
            )
        elif scheduler_type == 'cyclic':
            return CyclicLR(
                self.optimizer,
                base_lr=self.config.get('base_lr', 1e-5),
                max_lr=self.config.get('max_lr', 1e-3),
                step_size_up=self.config.get('step_size_up', 2000)
            )
        else:
            return None
            
    def step(self, metric: float = None):
        """Step the scheduler"""
        if self.scheduler is not None:
            if isinstance(self.scheduler, ReduceLROnPlateau):
                if metric is not None:
                    self.scheduler.step(metric)
            else:
                self.scheduler.step()
                
        self.current_lr = self.optimizer.param_groups[0]['lr']
        
    def get_lr(self) -> float:
        """Get current learning rate"""
        return self.current_lr

class GradientOptimizer:
    """Advanced gradient optimization techniques"""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.gradient_history = deque(maxlen=1000)
        self.parameter_history = defaultdict(list)
        
    def apply_gradient_clipping(self, gradients: List[torch.Tensor], 
                               method: str = "norm") -> List[torch.Tensor]:
        """Apply gradient clipping with various methods"""
        
        if method == "norm":
            # Global norm clipping
            total_norm = torch.norm(torch.stack([torch.norm(g) for g in gradients]))
            max_norm = self.config.get('max_norm', 1.0)
            clip_coef = min(1.0, max_norm / (total_norm + 1e-6))
            
            clipped_gradients = [g * clip_coef for g in gradients]
            
        elif method == "value":
            # Value clipping
            max_value = self.config.get('max_value', 1.0)
            clipped_gradients = [torch.clamp(g, -max_value, max_value) for g in gradients]
            
        elif method == "adaptive":
            # Adaptive clipping based on gradient history
            if len(self.gradient_history) > 0:
                avg_norm = np.mean([torch.norm(g).item() for g in self.gradient_history])
                adaptive_norm = avg_norm * self.config.get('adaptive_factor', 1.5)
            else:
                adaptive_norm = self.config.get('max_norm', 1.0)
                
            total_norm = torch.norm(torch.stack([torch.norm(g) for g in gradients]))
            clip_coef = min(1.0, adaptive_norm / (total_norm + 1e-6))
            clipped_gradients = [g * clip_coef for g in gradients]
            
        else:
            clipped_gradients = gradients
            
        # Store gradient history
        self.gradient_history.extend(clipped_gradients)
        
        return clipped_gradients
        
    def apply_gradient_accumulation(self, gradients: List[torch.Tensor], 
                                   accumulation_steps: int) -> List[torch.Tensor]:
        """Apply gradient accumulation"""
        if accumulation_steps <= 1:
            return gradients
            
        # Average gradients over accumulation steps
        accumulated_gradients = [g / accumulation_steps for g in gradients]
        return accumulated_gradients
        
    def apply_gradient_noise(self, gradients: List[torch.Tensor], 
                            noise_scale: float = 0.01) -> List[torch.Tensor]:
        """Add noise to gradients for regularization"""
        noisy_gradients = []
        for g in gradients:
            noise = torch.randn_like(g) * noise_scale
            noisy_gradients.append(g + noise)
        return noisy_gradients

class ModelEnsemble:
    """Advanced model ensemble techniques"""
    
    def __init__(self, models: List[nn.Module], config: Dict[str, Any]):
        self.models = models
        self.config = config
        self.weights = self._initialize_weights()
        
    def _initialize_weights(self) -> List[float]:
        """Initialize ensemble weights"""
        if self.config.get('weight_method') == 'uniform':
            return [1.0 / len(self.models)] * len(self.models)
        elif self.config.get('weight_method') == 'performance':
            # Initialize with equal weights, will be updated based on performance
            return [1.0 / len(self.models)] * len(self.models)
        else:
            return [1.0 / len(self.models)] * len(self.models)
            
    def predict(self, inputs: torch.Tensor, method: str = "weighted_average") -> torch.Tensor:
        """Make ensemble prediction"""
        predictions = []
        
        for model in self.models:
            with torch.no_grad():
                pred = model(inputs)
                predictions.append(pred)
                
        if method == "weighted_average":
            # Weighted average of predictions
            weighted_pred = torch.zeros_like(predictions[0])
            for pred, weight in zip(predictions, self.weights):
                weighted_pred += weight * pred
            return weighted_pred
            
        elif method == "majority_voting":
            # Majority voting for classification
            if predictions[0].dim() > 1:
                # Soft voting
                avg_pred = torch.mean(torch.stack(predictions), dim=0)
                return avg_pred
            else:
                # Hard voting
                votes = torch.stack(predictions)
                return torch.mode(votes, dim=0)[0]
                
        elif method == "stacking":
            # Stacking ensemble (requires meta-learner)
            stacked_pred = torch.cat(predictions, dim=-1)
            # Apply meta-learner (simplified)
            return torch.mean(stacked_pred, dim=-1, keepdim=True)
            
        else:
            return torch.mean(torch.stack(predictions), dim=0)
            
    def update_weights(self, performance_scores: List[float]):
        """Update ensemble weights based on performance"""
        if self.config.get('weight_method') == 'performance':
            # Softmax normalization
            scores = torch.tensor(performance_scores)
            self.weights = torch.softmax(scores, dim=0).tolist()

class NeuralArchitectureSearch:
    """Neural Architecture Search (NAS) implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.architecture_space = self._define_architecture_space()
        self.performance_history = []
        
    def _define_architecture_space(self) -> Dict[str, Any]:
        """Define searchable architecture space"""
        return {
            'num_layers': {'type': 'int', 'low': 2, 'high': 12},
            'hidden_dim': {'type': 'categorical', 'choices': [256, 512, 768, 1024]},
            'num_heads': {'type': 'categorical', 'choices': [4, 8, 12, 16]},
            'dropout': {'type': 'float', 'low': 0.0, 'high': 0.5},
            'activation': {'type': 'categorical', 'choices': ['relu', 'gelu', 'swish', 'mish']},
            'use_residual': {'type': 'categorical', 'choices': [True, False]},
            'use_layer_norm': {'type': 'categorical', 'choices': [True, False]}
        }
        
    def search_architecture(self, train_function: Callable, 
                           validation_function: Callable) -> Dict[str, Any]:
        """Search for optimal architecture"""
        
        best_architecture = None
        best_performance = float('-inf')
        
        for trial in range(self.config.get('max_trials', 50)):
            # Sample architecture
            architecture = self._sample_architecture()
            
            # Train and evaluate
            try:
                model = self._build_model(architecture)
                train_function(model, architecture)
                performance = validation_function(model)
                
                self.performance_history.append({
                    'architecture': architecture,
                    'performance': performance
                })
                
                if performance > best_performance:
                    best_performance = performance
                    best_architecture = architecture
                    
            except Exception as e:
                logger.error(f"Architecture search trial {trial} failed: {e}")
                continue
                
        return {
            'best_architecture': best_architecture,
            'best_performance': best_performance,
            'search_history': self.performance_history
        }
        
    def _sample_architecture(self) -> Dict[str, Any]:
        """Sample random architecture from search space"""
        architecture = {}
        for param_name, param_config in self.architecture_space.items():
            if param_config['type'] == 'int':
                architecture[param_name] = random.randint(
                    param_config['low'], param_config['high']
                )
            elif param_config['type'] == 'float':
                architecture[param_name] = random.uniform(
                    param_config['low'], param_config['high']
                )
            elif param_config['type'] == 'categorical':
                architecture[param_name] = random.choice(param_config['choices'])
                
        return architecture
        
    def _build_model(self, architecture: Dict[str, Any]) -> nn.Module:
        """Build model from architecture specification"""
        # This would be implemented based on your specific model architecture
        # For now, return a placeholder
        return nn.Module()

class AdvancedOptimizationEngine:
    """Main optimization engine combining all techniques"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.optimizer = AdvancedOptimizer(config)
        self.ensemble = None
        self.nas = NeuralArchitectureSearch(config.__dict__) if config.use_neural_architecture_search else None
        
    def comprehensive_optimization(self, model: nn.Module, 
                                  train_function: Callable,
                                  validation_function: Callable) -> Dict[str, Any]:
        """Comprehensive optimization using multiple techniques"""
        
        results = {}
        
        # 1. Hyperparameter optimization
        if self.config.optimization_method:
            logger.info("Starting hyperparameter optimization...")
            hp_results = self.optimizer.optimize_hyperparameters(
                lambda params: self._evaluate_hyperparameters(
                    model, params, train_function, validation_function
                )
            )
            results['hyperparameter_optimization'] = hp_results
            
        # 2. Neural Architecture Search
        if self.config.use_neural_architecture_search and self.nas:
            logger.info("Starting Neural Architecture Search...")
            nas_results = self.nas.search_architecture(train_function, validation_function)
            results['neural_architecture_search'] = nas_results
            
        # 3. Model Ensemble
        if self.config.use_ensemble:
            logger.info("Creating model ensemble...")
            ensemble_results = self._create_ensemble(model, train_function, validation_function)
            results['model_ensemble'] = ensemble_results
            
        return results
        
    def _evaluate_hyperparameters(self, model: nn.Module, params: Dict[str, Any],
                                 train_function: Callable, 
                                 validation_function: Callable) -> float:
        """Evaluate hyperparameters"""
        try:
            # Update model with new hyperparameters
            self._update_model_hyperparameters(model, params)
            
            # Train model
            train_function(model, params)
            
            # Validate model
            performance = validation_function(model)
            
            return performance
            
        except Exception as e:
            logger.error(f"Hyperparameter evaluation failed: {e}")
            return float('-inf')
            
    def _update_model_hyperparameters(self, model: nn.Module, params: Dict[str, Any]):
        """Update model hyperparameters"""
        # This would update model architecture based on parameters
        # Implementation depends on your specific model structure
        pass
        
    def _create_ensemble(self, base_model: nn.Module, train_function: Callable,
                        validation_function: Callable) -> Dict[str, Any]:
        """Create model ensemble"""
        models = []
        performances = []
        
        # Create multiple models with different initializations
        for i in range(self.config.get('ensemble_size', 5)):
            # Clone model with different initialization
            model = self._clone_model(base_model)
            
            # Train model
            train_function(model, {})
            performance = validation_function(model)
            
            models.append(model)
            performances.append(performance)
            
        # Create ensemble
        ensemble_config = {'weight_method': 'performance'}
        self.ensemble = ModelEnsemble(models, ensemble_config)
        self.ensemble.update_weights(performances)
        
        return {
            'ensemble_size': len(models),
            'individual_performances': performances,
            'ensemble_weights': self.ensemble.weights
        }
        
    def _clone_model(self, model: nn.Module) -> nn.Module:
        """Clone model with different initialization"""
        # Create a copy of the model
        cloned_model = type(model)(**model.config.__dict__)
        
        # Reinitialize weights
        cloned_model._initialize_weights()
        
        return cloned_model

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test advanced optimization
    print("Testing Advanced Optimization Engine...")
    
    # Create optimization config
    config = OptimizationConfig(
        optimization_method="optuna",
        n_trials=10,
        search_space={
            'learning_rate': {'type': 'float', 'low': 1e-5, 'high': 1e-2, 'log': True},
            'batch_size': {'type': 'categorical', 'choices': [16, 32, 64]},
            'dropout': {'type': 'float', 'low': 0.0, 'high': 0.5}
        }
    )
    
    # Create optimization engine
    engine = AdvancedOptimizationEngine(config)
    
    # Define dummy objective function
    def dummy_objective(params):
        # Simulate model training and evaluation
        import time
        time.sleep(0.1)  # Simulate training time
        
        # Return random performance metric
        performance = np.random.random()
        return {'loss': 1.0 - performance, 'accuracy': performance}
    
    # Run optimization
    print("Running hyperparameter optimization...")
    results = engine.optimizer.optimize_hyperparameters(dummy_objective)
    
    print(f"Best parameters: {results['best_params']}")
    print(f"Best value: {results['best_value']:.4f}")
    print(f"Number of trials: {results['n_trials']}")
    
    # Test learning rate scheduler
    print("\nTesting Learning Rate Scheduler...")
    dummy_model = nn.Linear(10, 1)
    dummy_optimizer = optim.Adam(dummy_model.parameters(), lr=1e-3)
    
    scheduler_config = {
        'type': 'cosine',
        'T_max': 10,
        'eta_min': 1e-6
    }
    
    scheduler = LearningRateScheduler(dummy_optimizer, scheduler_config)
    
    for epoch in range(10):
        scheduler.step()
        print(f"Epoch {epoch}: LR = {scheduler.get_lr():.6f}")
        
    # Test gradient optimizer
    print("\nTesting Gradient Optimizer...")
    gradient_config = {'max_norm': 1.0, 'adaptive_factor': 1.5}
    grad_optimizer = GradientOptimizer(dummy_model, gradient_config)
    
    # Simulate gradients
    dummy_gradients = [torch.randn(10, 1) for _ in range(3)]
    clipped_gradients = grad_optimizer.apply_gradient_clipping(dummy_gradients, method="norm")
    
    print(f"Original gradient norm: {torch.norm(torch.stack([torch.norm(g) for g in dummy_gradients])):.4f}")
    print(f"Clipped gradient norm: {torch.norm(torch.stack([torch.norm(g) for g in clipped_gradients])):.4f}")
    
    print("\nAdvanced optimization engine initialized successfully!")
























