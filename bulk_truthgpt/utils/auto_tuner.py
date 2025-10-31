"""
Auto Tuner
==========

Advanced auto-tuning system for optimal performance.
"""

import asyncio
import logging
import time
import numpy as np
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import weakref
from collections import defaultdict, deque
import json
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

logger = logging.getLogger(__name__)

class TuningStrategy(str, Enum):
    """Tuning strategies."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    GENETIC = "genetic"
    NEURAL = "neural"

class TuningObjective(str, Enum):
    """Tuning objectives."""
    PERFORMANCE = "performance"
    MEMORY = "memory"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ACCURACY = "accuracy"
    EFFICIENCY = "efficiency"

@dataclass
class TuningParameter:
    """Tuning parameter definition."""
    name: str
    min_value: float
    max_value: float
    step: float
    parameter_type: str = "float"  # float, int, bool, categorical
    categories: List[str] = field(default_factory=list)

@dataclass
class TuningResult:
    """Tuning result."""
    parameters: Dict[str, Any]
    score: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TuningConfig:
    """Tuning configuration."""
    strategy: TuningStrategy = TuningStrategy.BAYESIAN
    objective: TuningObjective = TuningObjective.PERFORMANCE
    max_iterations: int = 100
    max_time: int = 3600
    early_stopping: bool = True
    patience: int = 10
    enable_parallel: bool = True
    max_workers: int = 4
    enable_ml_model: bool = True
    model_retrain_interval: int = 50

class AutoTuner:
    """
    Advanced auto-tuning system.
    
    Features:
    - Multiple tuning strategies
    - Machine learning-based optimization
    - Parallel parameter exploration
    - Early stopping
    - Performance prediction
    """
    
    def __init__(self, config: Optional[TuningConfig] = None):
        self.config = config or TuningConfig()
        self.parameters = {}
        self.results = []
        self.best_result = None
        self.ml_model = None
        self.tuning_history = deque(maxlen=1000)
        self.stats = {
            'total_iterations': 0,
            'best_score': float('-inf'),
            'improvement_count': 0,
            'convergence_iterations': 0
        }
        self.running = False
        self.lock = asyncio.Lock()
        
    async def initialize(self):
        """Initialize auto tuner."""
        logger.info("Initializing Auto Tuner...")
        
        try:
            # Initialize ML model if enabled
            if self.config.enable_ml_model:
                await self._initialize_ml_model()
            
            logger.info("Auto Tuner initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Auto Tuner: {str(e)}")
            raise
    
    async def _initialize_ml_model(self):
        """Initialize machine learning model."""
        try:
            # Initialize Random Forest model
            self.ml_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            logger.info("ML model initialized for performance prediction")
            
        except Exception as e:
            logger.error(f"Failed to initialize ML model: {str(e)}")
            self.ml_model = None
    
    def register_parameter(self, param: TuningParameter):
        """Register tuning parameter."""
        self.parameters[param.name] = param
        logger.debug(f"Registered parameter: {param.name}")
    
    async def tune(self, objective_func: Callable, *args, **kwargs) -> TuningResult:
        """Execute auto-tuning."""
        logger.info("Starting auto-tuning...")
        start_time = time.time()
        
        try:
            self.running = True
            self.stats['total_iterations'] = 0
            
            # Generate initial parameter sets
            if self.config.strategy == TuningStrategy.GRID_SEARCH:
                param_sets = await self._generate_grid_search_params()
            elif self.config.strategy == TuningStrategy.RANDOM_SEARCH:
                param_sets = await self._generate_random_search_params()
            elif self.config.strategy == TuningStrategy.BAYESIAN:
                param_sets = await self._generate_bayesian_params()
            elif self.config.strategy == TuningStrategy.GENETIC:
                param_sets = await self._generate_genetic_params()
            elif self.config.strategy == TuningStrategy.NEURAL:
                param_sets = await self._generate_neural_params()
            else:
                param_sets = await self._generate_random_search_params()
            
            # Execute tuning iterations
            for i, param_set in enumerate(param_sets):
                if not self.running:
                    break
                
                # Check time limit
                if time.time() - start_time > self.config.max_time:
                    logger.info("Time limit reached, stopping tuning")
                    break
                
                # Check iteration limit
                if i >= self.config.max_iterations:
                    logger.info("Iteration limit reached, stopping tuning")
                    break
                
                # Execute parameter set
                result = await self._evaluate_parameters(param_set, objective_func, *args, **kwargs)
                
                # Update best result
                if self.best_result is None or result.score > self.best_result.score:
                    self.best_result = result
                    self.stats['best_score'] = result.score
                    self.stats['improvement_count'] += 1
                
                # Store result
                self.results.append(result)
                self.tuning_history.append({
                    'iteration': i,
                    'parameters': param_set,
                    'score': result.score,
                    'timestamp': datetime.utcnow()
                })
                
                self.stats['total_iterations'] += 1
                
                # Check early stopping
                if self.config.early_stopping and self._should_stop_early():
                    logger.info("Early stopping triggered")
                    break
                
                # Retrain ML model
                if self.config.enable_ml_model and i % self.config.model_retrain_interval == 0:
                    await self._retrain_ml_model()
                
                logger.debug(f"Tuning iteration {i+1}/{self.config.max_iterations} completed")
            
            # Finalize tuning
            self.running = False
            
            logger.info(f"Auto-tuning completed. Best score: {self.best_result.score:.4f}")
            return self.best_result
            
        except Exception as e:
            logger.error(f"Auto-tuning failed: {str(e)}")
            raise
        finally:
            self.running = False
    
    async def _generate_grid_search_params(self) -> List[Dict[str, Any]]:
        """Generate grid search parameters."""
        param_sets = []
        
        # Generate all combinations
        param_names = list(self.parameters.keys())
        param_values = []
        
        for name in param_names:
            param = self.parameters[name]
            if param.parameter_type == "float":
                values = np.arange(param.min_value, param.max_value + param.step, param.step)
            elif param.parameter_type == "int":
                values = np.arange(int(param.min_value), int(param.max_value) + 1, int(param.step))
            elif param.parameter_type == "bool":
                values = [True, False]
            elif param.parameter_type == "categorical":
                values = param.categories
            else:
                values = [param.min_value]
            
            param_values.append(values)
        
        # Generate combinations
        from itertools import product
        for combination in product(*param_values):
            param_set = dict(zip(param_names, combination))
            param_sets.append(param_set)
        
        return param_sets
    
    async def _generate_random_search_params(self) -> List[Dict[str, Any]]:
        """Generate random search parameters."""
        param_sets = []
        
        for _ in range(self.config.max_iterations):
            param_set = {}
            
            for name, param in self.parameters.items():
                if param.parameter_type == "float":
                    value = np.random.uniform(param.min_value, param.max_value)
                elif param.parameter_type == "int":
                    value = np.random.randint(int(param.min_value), int(param.max_value) + 1)
                elif param.parameter_type == "bool":
                    value = np.random.choice([True, False])
                elif param.parameter_type == "categorical":
                    value = np.random.choice(param.categories)
                else:
                    value = param.min_value
                
                param_set[name] = value
            
            param_sets.append(param_set)
        
        return param_sets
    
    async def _generate_bayesian_params(self) -> List[Dict[str, Any]]:
        """Generate Bayesian optimization parameters."""
        param_sets = []
        
        # Simple Bayesian optimization implementation
        for i in range(self.config.max_iterations):
            if i == 0:
                # Random initial point
                param_set = await self._generate_random_param_set()
            else:
                # Use ML model to predict best parameters
                param_set = await self._predict_best_parameters()
            
            param_sets.append(param_set)
        
        return param_sets
    
    async def _generate_genetic_params(self) -> List[Dict[str, Any]]:
        """Generate genetic algorithm parameters."""
        param_sets = []
        
        # Initialize population
        population_size = min(20, self.config.max_iterations)
        population = []
        
        for _ in range(population_size):
            param_set = await self._generate_random_param_set()
            population.append(param_set)
        
        # Evolve population
        for generation in range(self.config.max_iterations // population_size):
            # Evaluate population
            evaluated_population = []
            for param_set in population:
                # This would be evaluated in the main loop
                evaluated_population.append(param_set)
            
            # Select best individuals
            best_individuals = sorted(evaluated_population, key=lambda x: x.get('score', 0), reverse=True)[:population_size//2]
            
            # Generate new population
            new_population = []
            for _ in range(population_size):
                if len(best_individuals) > 0:
                    parent1 = np.random.choice(best_individuals)
                    parent2 = np.random.choice(best_individuals)
                    child = await self._crossover(parent1, parent2)
                    child = await self._mutate(child)
                    new_population.append(child)
                else:
                    new_population.append(await self._generate_random_param_set())
            
            population = new_population
            param_sets.extend(population)
        
        return param_sets
    
    async def _generate_neural_params(self) -> List[Dict[str, Any]]:
        """Generate neural network-based parameters."""
        param_sets = []
        
        # Use neural network to predict parameters
        for i in range(self.config.max_iterations):
            if i == 0:
                param_set = await self._generate_random_param_set()
            else:
                param_set = await self._neural_predict_parameters()
            
            param_sets.append(param_set)
        
        return param_sets
    
    async def _generate_random_param_set(self) -> Dict[str, Any]:
        """Generate random parameter set."""
        param_set = {}
        
        for name, param in self.parameters.items():
            if param.parameter_type == "float":
                value = np.random.uniform(param.min_value, param.max_value)
            elif param.parameter_type == "int":
                value = np.random.randint(int(param.min_value), int(param.max_value) + 1)
            elif param.parameter_type == "bool":
                value = np.random.choice([True, False])
            elif param.parameter_type == "categorical":
                value = np.random.choice(param.categories)
            else:
                value = param.min_value
            
            param_set[name] = value
        
        return param_set
    
    async def _predict_best_parameters(self) -> Dict[str, Any]:
        """Predict best parameters using ML model."""
        if self.ml_model is None or len(self.results) < 10:
            return await self._generate_random_param_set()
        
        try:
            # Prepare training data
            X = []
            y = []
            
            for result in self.results:
                param_vector = []
                for name in self.parameters.keys():
                    param_vector.append(result.parameters.get(name, 0))
                X.append(param_vector)
                y.append(result.score)
            
            # Train model
            X = np.array(X)
            y = np.array(y)
            
            if len(X) > 5:
                self.ml_model.fit(X, y)
                
                # Generate candidate parameters
                candidates = []
                for _ in range(100):
                    candidate = await self._generate_random_param_set()
                    candidate_vector = [candidate.get(name, 0) for name in self.parameters.keys()]
                    candidates.append(candidate_vector)
                
                # Predict scores
                candidate_scores = self.ml_model.predict(candidates)
                
                # Select best candidate
                best_idx = np.argmax(candidate_scores)
                best_candidate = candidates[best_idx]
                
                # Convert back to parameter dict
                param_set = {}
                for i, name in enumerate(self.parameters.keys()):
                    param_set[name] = best_candidate[i]
                
                return param_set
            
        except Exception as e:
            logger.error(f"Failed to predict best parameters: {str(e)}")
        
        return await self._generate_random_param_set()
    
    async def _evaluate_parameters(self, param_set: Dict[str, Any], objective_func: Callable, *args, **kwargs) -> TuningResult:
        """Evaluate parameter set."""
        try:
            start_time = time.time()
            
            # Execute objective function
            score = await objective_func(param_set, *args, **kwargs)
            
            evaluation_time = time.time() - start_time
            
            # Create result
            result = TuningResult(
                parameters=param_set,
                score=score,
                timestamp=datetime.utcnow(),
                metadata={
                    'evaluation_time': evaluation_time,
                    'iteration': self.stats['total_iterations']
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to evaluate parameters: {str(e)}")
            return TuningResult(
                parameters=param_set,
                score=float('-inf'),
                timestamp=datetime.utcnow(),
                metadata={'error': str(e)}
            )
    
    def _should_stop_early(self) -> bool:
        """Check if early stopping should be triggered."""
        if len(self.results) < self.config.patience:
            return False
        
        # Check if no improvement in last patience iterations
        recent_scores = [result.score for result in self.results[-self.config.patience:]]
        if len(recent_scores) < self.config.patience:
            return False
        
        # Check if scores are not improving
        if max(recent_scores) <= max([result.score for result in self.results[:-self.config.patience]]):
            return True
        
        return False
    
    async def _retrain_ml_model(self):
        """Retrain ML model with new data."""
        if self.ml_model is None or len(self.results) < 10:
            return
        
        try:
            # Prepare training data
            X = []
            y = []
            
            for result in self.results:
                param_vector = []
                for name in self.parameters.keys():
                    param_vector.append(result.parameters.get(name, 0))
                X.append(param_vector)
                y.append(result.score)
            
            # Train model
            X = np.array(X)
            y = np.array(y)
            
            if len(X) > 5:
                self.ml_model.fit(X, y)
                logger.debug("ML model retrained with new data")
            
        except Exception as e:
            logger.error(f"Failed to retrain ML model: {str(e)}")
    
    def get_tuning_stats(self) -> Dict[str, Any]:
        """Get tuning statistics."""
        return {
            'total_iterations': self.stats['total_iterations'],
            'best_score': self.stats['best_score'],
            'improvement_count': self.stats['improvement_count'],
            'convergence_iterations': self.stats['convergence_iterations'],
            'best_parameters': self.best_result.parameters if self.best_result else {},
            'config': {
                'strategy': self.config.strategy.value,
                'objective': self.config.objective.value,
                'max_iterations': self.config.max_iterations,
                'max_time': self.config.max_time,
                'early_stopping': self.config.early_stopping,
                'patience': self.config.patience,
                'ml_model_enabled': self.config.enable_ml_model
            }
        }
    
    async def cleanup(self):
        """Cleanup auto tuner."""
        try:
            self.running = False
            self.results.clear()
            self.tuning_history.clear()
            self.ml_model = None
            
            logger.info("Auto Tuner cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Failed to cleanup Auto Tuner: {str(e)}")

# Global auto tuner
auto_tuner = AutoTuner()

# Decorators for auto-tuning
def auto_tune(objective: TuningObjective = TuningObjective.PERFORMANCE, max_iterations: int = 100):
    """Decorator for auto-tuning functions."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Register parameters
            for name, param in func.__annotations__.items():
                if isinstance(param, TuningParameter):
                    auto_tuner.register_parameter(param)
            
            # Execute tuning
            result = await auto_tuner.tune(func, *args, **kwargs)
            
            return result
        
        return wrapper
    return decorator

def tuning_parameter(name: str, min_value: float, max_value: float, step: float = 1.0, parameter_type: str = "float"):
    """Decorator for tuning parameters."""
    def decorator(func):
        param = TuningParameter(
            name=name,
            min_value=min_value,
            max_value=max_value,
            step=step,
            parameter_type=parameter_type
        )
        
        # Store parameter in function metadata
        if not hasattr(func, '_tuning_parameters'):
            func._tuning_parameters = []
        func._tuning_parameters.append(param)
        
        return func
    return decorator











