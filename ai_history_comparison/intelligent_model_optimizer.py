"""
Intelligent Model Optimizer
===========================

Advanced AI model optimization system with intelligent parameter tuning,
performance enhancement, and automated model improvement.
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import hashlib
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import optuna
import joblib
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class OptimizationStrategy(str, Enum):
    """Optimization strategies"""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    GENETIC_ALGORITHM = "genetic_algorithm"
    PARTICLE_SWARM = "particle_swarm"
    GRADIENT_BASED = "gradient_based"
    EVOLUTIONARY = "evolutionary"
    HYBRID = "hybrid"


class OptimizationGoal(str, Enum):
    """Optimization goals"""
    MAXIMIZE_ACCURACY = "maximize_accuracy"
    MINIMIZE_ERROR = "minimize_error"
    MAXIMIZE_EFFICIENCY = "maximize_efficiency"
    MINIMIZE_LATENCY = "minimize_latency"
    MAXIMIZE_THROUGHPUT = "maximize_throughput"
    MINIMIZE_COST = "minimize_cost"
    BALANCE_PERFORMANCE = "balance_performance"
    CUSTOM = "custom"


class ModelType(str, Enum):
    """Model types for optimization"""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    TIME_SERIES = "time_series"
    DEEP_LEARNING = "deep_learning"
    ENSEMBLE = "ensemble"
    CUSTOM = "custom"


@dataclass
class OptimizationResult:
    """Result of model optimization"""
    optimization_id: str
    model_name: str
    strategy: OptimizationStrategy
    goal: OptimizationGoal
    best_parameters: Dict[str, Any]
    best_score: float
    improvement_percentage: float
    optimization_time: float
    iterations_completed: int
    convergence_achieved: bool
    optimization_history: List[Dict[str, Any]]
    created_at: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ModelConfiguration:
    """Model configuration for optimization"""
    model_type: ModelType
    base_algorithm: str
    parameter_space: Dict[str, Any]
    constraints: Dict[str, Any]
    optimization_budget: int
    cross_validation_folds: int
    scoring_metric: str
    early_stopping: bool
    parallel_processing: bool


@dataclass
class PerformanceMetrics:
    """Performance metrics for model evaluation"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    mse: float
    rmse: float
    mae: float
    r2_score: float
    training_time: float
    prediction_time: float
    memory_usage: float
    cpu_usage: float


class IntelligentModelOptimizer:
    """Intelligent model optimizer with advanced optimization strategies"""
    
    def __init__(self, max_optimization_time: int = 3600):
        self.max_optimization_time = max_optimization_time
        self.optimization_results: Dict[str, OptimizationResult] = {}
        self.model_configurations: Dict[str, ModelConfiguration] = {}
        self.performance_history: Dict[str, List[PerformanceMetrics]] = defaultdict(list)
        
        # Optimization tracking
        self.optimization_stats = {
            "total_optimizations": 0,
            "successful_optimizations": 0,
            "failed_optimizations": 0,
            "average_improvement": 0.0,
            "best_improvement": 0.0
        }
        
        # Cache for optimization results
        self.optimization_cache = {}
        self.cache_ttl = 7200  # 2 hours
    
    async def optimize_model(self, 
                           model_name: str,
                           X_train: np.ndarray,
                           y_train: np.ndarray,
                           X_test: np.ndarray,
                           y_test: np.ndarray,
                           configuration: ModelConfiguration,
                           strategy: OptimizationStrategy = OptimizationStrategy.BAYESIAN_OPTIMIZATION,
                           goal: OptimizationGoal = OptimizationGoal.MAXIMIZE_ACCURACY) -> OptimizationResult:
        """Optimize model using specified strategy and goal"""
        try:
            start_time = datetime.now()
            optimization_id = hashlib.md5(f"{model_name}_{strategy}_{goal}_{datetime.now()}".encode()).hexdigest()
            
            logger.info(f"Starting optimization for {model_name} using {strategy.value}")
            
            # Prepare optimization
            optimization_history = []
            best_score = -float('inf')
            best_parameters = {}
            iterations_completed = 0
            convergence_achieved = False
            
            # Execute optimization based on strategy
            if strategy == OptimizationStrategy.GRID_SEARCH:
                best_score, best_parameters, optimization_history = await self._grid_search_optimization(
                    X_train, y_train, X_test, y_test, configuration, goal
                )
            elif strategy == OptimizationStrategy.RANDOM_SEARCH:
                best_score, best_parameters, optimization_history = await self._random_search_optimization(
                    X_train, y_train, X_test, y_test, configuration, goal
                )
            elif strategy == OptimizationStrategy.BAYESIAN_OPTIMIZATION:
                best_score, best_parameters, optimization_history = await self._bayesian_optimization(
                    X_train, y_train, X_test, y_test, configuration, goal
                )
            elif strategy == OptimizationStrategy.GENETIC_ALGORITHM:
                best_score, best_parameters, optimization_history = await self._genetic_algorithm_optimization(
                    X_train, y_train, X_test, y_test, configuration, goal
                )
            elif strategy == OptimizationStrategy.PARTICLE_SWARM:
                best_score, best_parameters, optimization_history = await self._particle_swarm_optimization(
                    X_train, y_train, X_test, y_test, configuration, goal
                )
            elif strategy == OptimizationStrategy.HYBRID:
                best_score, best_parameters, optimization_history = await self._hybrid_optimization(
                    X_train, y_train, X_test, y_test, configuration, goal
                )
            else:
                raise ValueError(f"Unsupported optimization strategy: {strategy}")
            
            end_time = datetime.now()
            optimization_time = (end_time - start_time).total_seconds()
            
            # Calculate improvement
            baseline_score = await self._calculate_baseline_score(X_train, y_train, X_test, y_test, configuration)
            improvement_percentage = ((best_score - baseline_score) / baseline_score * 100) if baseline_score > 0 else 0
            
            # Create optimization result
            result = OptimizationResult(
                optimization_id=optimization_id,
                model_name=model_name,
                strategy=strategy,
                goal=goal,
                best_parameters=best_parameters,
                best_score=best_score,
                improvement_percentage=improvement_percentage,
                optimization_time=optimization_time,
                iterations_completed=iterations_completed,
                convergence_achieved=convergence_achieved,
                optimization_history=optimization_history,
                created_at=start_time,
                metadata={
                    "baseline_score": baseline_score,
                    "configuration": asdict(configuration),
                    "data_shape": {
                        "X_train": X_train.shape,
                        "y_train": y_train.shape,
                        "X_test": X_test.shape,
                        "y_test": y_test.shape
                    }
                }
            )
            
            # Store result
            self.optimization_results[optimization_id] = result
            
            # Update stats
            self.optimization_stats["total_optimizations"] += 1
            if improvement_percentage > 0:
                self.optimization_stats["successful_optimizations"] += 1
                self.optimization_stats["average_improvement"] = (
                    (self.optimization_stats["average_improvement"] * (self.optimization_stats["successful_optimizations"] - 1) + 
                     improvement_percentage) / self.optimization_stats["successful_optimizations"]
                )
                if improvement_percentage > self.optimization_stats["best_improvement"]:
                    self.optimization_stats["best_improvement"] = improvement_percentage
            else:
                self.optimization_stats["failed_optimizations"] += 1
            
            logger.info(f"Optimization completed for {model_name}: {improvement_percentage:.2f}% improvement")
            
            return result
            
        except Exception as e:
            logger.error(f"Error optimizing model {model_name}: {str(e)}")
            self.optimization_stats["failed_optimizations"] += 1
            raise e
    
    async def auto_optimize_model(self, 
                                model_name: str,
                                X_train: np.ndarray,
                                y_train: np.ndarray,
                                X_test: np.ndarray,
                                y_test: np.ndarray,
                                model_type: ModelType = ModelType.REGRESSION) -> OptimizationResult:
        """Automatically optimize model using best strategy"""
        try:
            # Determine best strategy based on data characteristics
            strategy = await self._determine_best_strategy(X_train, y_train, model_type)
            
            # Create default configuration
            configuration = await self._create_default_configuration(model_type)
            
            # Determine optimization goal
            goal = await self._determine_optimization_goal(model_type, X_train.shape[0])
            
            # Perform optimization
            result = await self.optimize_model(
                model_name, X_train, y_train, X_test, y_test, 
                configuration, strategy, goal
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in auto optimization for {model_name}: {str(e)}")
            raise e
    
    async def batch_optimize_models(self, 
                                  models_data: Dict[str, Dict[str, Any]],
                                  strategy: OptimizationStrategy = OptimizationStrategy.BAYESIAN_OPTIMIZATION) -> Dict[str, OptimizationResult]:
        """Optimize multiple models in batch"""
        try:
            results = {}
            
            # Create optimization tasks
            tasks = []
            for model_name, data in models_data.items():
                task = asyncio.create_task(
                    self.optimize_model(
                        model_name=model_name,
                        X_train=data["X_train"],
                        y_train=data["y_train"],
                        X_test=data["X_test"],
                        y_test=data["y_test"],
                        configuration=data["configuration"],
                        strategy=strategy,
                        goal=data.get("goal", OptimizationGoal.MAXIMIZE_ACCURACY)
                    )
                )
                tasks.append((model_name, task))
            
            # Execute optimizations
            for model_name, task in tasks:
                try:
                    result = await task
                    results[model_name] = result
                except Exception as e:
                    logger.error(f"Error optimizing {model_name}: {str(e)}")
                    results[model_name] = None
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch optimization: {str(e)}")
            return {}
    
    async def get_optimization_summary(self, model_name: str = None) -> Dict[str, Any]:
        """Get optimization summary"""
        try:
            if model_name:
                # Get summary for specific model
                model_results = [r for r in self.optimization_results.values() if r.model_name == model_name]
                if not model_results:
                    return {"error": f"No optimization results found for {model_name}"}
                
                summary = {
                    "model_name": model_name,
                    "total_optimizations": len(model_results),
                    "best_improvement": max(r.improvement_percentage for r in model_results),
                    "average_improvement": np.mean([r.improvement_percentage for r in model_results]),
                    "best_strategy": max(model_results, key=lambda x: x.improvement_percentage).strategy.value,
                    "optimization_history": [
                        {
                            "optimization_id": r.optimization_id,
                            "strategy": r.strategy.value,
                            "improvement": r.improvement_percentage,
                            "optimization_time": r.optimization_time,
                            "created_at": r.created_at.isoformat()
                        }
                        for r in model_results
                    ]
                }
            else:
                # Get global summary
                summary = {
                    "total_models": len(set(r.model_name for r in self.optimization_results.values())),
                    "total_optimizations": self.optimization_stats["total_optimizations"],
                    "successful_optimizations": self.optimization_stats["successful_optimizations"],
                    "failed_optimizations": self.optimization_stats["failed_optimizations"],
                    "success_rate": (
                        self.optimization_stats["successful_optimizations"] / 
                        self.optimization_stats["total_optimizations"] * 100
                        if self.optimization_stats["total_optimizations"] > 0 else 0
                    ),
                    "average_improvement": self.optimization_stats["average_improvement"],
                    "best_improvement": self.optimization_stats["best_improvement"],
                    "strategy_performance": await self._analyze_strategy_performance()
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting optimization summary: {str(e)}")
            return {"error": str(e)}
    
    # Private optimization methods
    async def _grid_search_optimization(self, 
                                      X_train: np.ndarray,
                                      y_train: np.ndarray,
                                      X_test: np.ndarray,
                                      y_test: np.ndarray,
                                      configuration: ModelConfiguration,
                                      goal: OptimizationGoal) -> Tuple[float, Dict[str, Any], List[Dict[str, Any]]]:
        """Grid search optimization"""
        try:
            # Create model
            model = await self._create_model(configuration.base_algorithm)
            
            # Prepare parameter grid
            param_grid = configuration.parameter_space
            
            # Create cross-validation
            cv = TimeSeriesSplit(n_splits=configuration.cross_validation_folds)
            
            # Grid search
            grid_search = GridSearchCV(
                model, param_grid, cv=cv, scoring=configuration.scoring_metric,
                n_jobs=-1 if configuration.parallel_processing else 1,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            
            # Get best parameters and score
            best_parameters = grid_search.best_params_
            best_score = grid_search.best_score_
            
            # Create optimization history
            optimization_history = []
            for i, (params, score) in enumerate(zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score'])):
                optimization_history.append({
                    "iteration": i + 1,
                    "parameters": params,
                    "score": score,
                    "timestamp": datetime.now().isoformat()
                })
            
            return best_score, best_parameters, optimization_history
            
        except Exception as e:
            logger.error(f"Error in grid search optimization: {str(e)}")
            return -float('inf'), {}, []
    
    async def _random_search_optimization(self, 
                                        X_train: np.ndarray,
                                        y_train: np.ndarray,
                                        X_test: np.ndarray,
                                        y_test: np.ndarray,
                                        configuration: ModelConfiguration,
                                        goal: OptimizationGoal) -> Tuple[float, Dict[str, Any], List[Dict[str, Any]]]:
        """Random search optimization"""
        try:
            # Create model
            model = await self._create_model(configuration.base_algorithm)
            
            # Prepare parameter distribution
            param_distributions = configuration.parameter_space
            
            # Create cross-validation
            cv = TimeSeriesSplit(n_splits=configuration.cross_validation_folds)
            
            # Random search
            random_search = RandomizedSearchCV(
                model, param_distributions, n_iter=configuration.optimization_budget,
                cv=cv, scoring=configuration.scoring_metric,
                n_jobs=-1 if configuration.parallel_processing else 1,
                random_state=42, verbose=0
            )
            
            random_search.fit(X_train, y_train)
            
            # Get best parameters and score
            best_parameters = random_search.best_params_
            best_score = random_search.best_score_
            
            # Create optimization history
            optimization_history = []
            for i, (params, score) in enumerate(zip(random_search.cv_results_['params'], random_search.cv_results_['mean_test_score'])):
                optimization_history.append({
                    "iteration": i + 1,
                    "parameters": params,
                    "score": score,
                    "timestamp": datetime.now().isoformat()
                })
            
            return best_score, best_parameters, optimization_history
            
        except Exception as e:
            logger.error(f"Error in random search optimization: {str(e)}")
            return -float('inf'), {}, []
    
    async def _bayesian_optimization(self, 
                                   X_train: np.ndarray,
                                   y_train: np.ndarray,
                                   X_test: np.ndarray,
                                   y_test: np.ndarray,
                                   configuration: ModelConfiguration,
                                   goal: OptimizationGoal) -> Tuple[float, Dict[str, Any], List[Dict[str, Any]]]:
        """Bayesian optimization using Optuna"""
        try:
            def objective(trial):
                # Suggest parameters
                params = {}
                for param_name, param_config in configuration.parameter_space.items():
                    if isinstance(param_config, list):
                        params[param_name] = trial.suggest_categorical(param_name, param_config)
                    elif isinstance(param_config, tuple) and len(param_config) == 2:
                        if isinstance(param_config[0], int) and isinstance(param_config[1], int):
                            params[param_name] = trial.suggest_int(param_name, param_config[0], param_config[1])
                        else:
                            params[param_name] = trial.suggest_float(param_name, param_config[0], param_config[1])
                    else:
                        params[param_name] = trial.suggest_float(param_name, 0.0, 1.0)
                
                # Create and train model
                model = await self._create_model(configuration.base_algorithm, params)
                
                # Cross-validation
                cv_scores = cross_val_score(
                    model, X_train, y_train, 
                    cv=configuration.cross_validation_folds,
                    scoring=configuration.scoring_metric
                )
                
                return cv_scores.mean()
            
            # Create study
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=configuration.optimization_budget)
            
            # Get best parameters and score
            best_parameters = study.best_params
            best_score = study.best_value
            
            # Create optimization history
            optimization_history = []
            for trial in study.trials:
                optimization_history.append({
                    "iteration": trial.number + 1,
                    "parameters": trial.params,
                    "score": trial.value,
                    "timestamp": datetime.now().isoformat()
                })
            
            return best_score, best_parameters, optimization_history
            
        except Exception as e:
            logger.error(f"Error in Bayesian optimization: {str(e)}")
            return -float('inf'), {}, []
    
    async def _genetic_algorithm_optimization(self, 
                                            X_train: np.ndarray,
                                            y_train: np.ndarray,
                                            X_test: np.ndarray,
                                            y_test: np.ndarray,
                                            configuration: ModelConfiguration,
                                            goal: OptimizationGoal) -> Tuple[float, Dict[str, Any], List[Dict[str, Any]]]:
        """Genetic algorithm optimization"""
        try:
            # This is a simplified genetic algorithm implementation
            # In practice, you might want to use a library like DEAP
            
            population_size = 50
            generations = configuration.optimization_budget // population_size
            mutation_rate = 0.1
            crossover_rate = 0.8
            
            # Initialize population
            population = await self._initialize_population(population_size, configuration.parameter_space)
            optimization_history = []
            
            best_score = -float('inf')
            best_parameters = {}
            
            for generation in range(generations):
                # Evaluate population
                fitness_scores = []
                for individual in population:
                    score = await self._evaluate_individual(individual, X_train, y_train, configuration)
                    fitness_scores.append(score)
                    
                    if score > best_score:
                        best_score = score
                        best_parameters = individual.copy()
                
                # Record generation
                optimization_history.append({
                    "generation": generation + 1,
                    "best_score": best_score,
                    "average_score": np.mean(fitness_scores),
                    "timestamp": datetime.now().isoformat()
                })
                
                # Selection, crossover, and mutation
                new_population = []
                for _ in range(population_size):
                    # Selection (tournament selection)
                    parent1 = await self._tournament_selection(population, fitness_scores)
                    parent2 = await self._tournament_selection(population, fitness_scores)
                    
                    # Crossover
                    if np.random.random() < crossover_rate:
                        child = await self._crossover(parent1, parent2)
                    else:
                        child = parent1.copy()
                    
                    # Mutation
                    if np.random.random() < mutation_rate:
                        child = await self._mutate(child, configuration.parameter_space)
                    
                    new_population.append(child)
                
                population = new_population
            
            return best_score, best_parameters, optimization_history
            
        except Exception as e:
            logger.error(f"Error in genetic algorithm optimization: {str(e)}")
            return -float('inf'), {}, []
    
    async def _particle_swarm_optimization(self, 
                                         X_train: np.ndarray,
                                         y_train: np.ndarray,
                                         X_test: np.ndarray,
                                         y_test: np.ndarray,
                                         configuration: ModelConfiguration,
                                         goal: OptimizationGoal) -> Tuple[float, Dict[str, Any], List[Dict[str, Any]]]:
        """Particle swarm optimization"""
        try:
            # Simplified PSO implementation
            n_particles = 30
            n_iterations = configuration.optimization_budget // n_particles
            w = 0.9  # inertia weight
            c1 = 2.0  # cognitive parameter
            c2 = 2.0  # social parameter
            
            # Initialize particles
            particles = await self._initialize_particles(n_particles, configuration.parameter_space)
            velocities = [np.random.random(len(particles[0])) * 0.1 for _ in range(n_particles)]
            
            # Initialize best positions
            personal_best = [p.copy() for p in particles]
            personal_best_scores = [-float('inf')] * n_particles
            
            global_best = particles[0].copy()
            global_best_score = -float('inf')
            
            optimization_history = []
            
            for iteration in range(n_iterations):
                for i, particle in enumerate(particles):
                    # Evaluate particle
                    score = await self._evaluate_individual(particle, X_train, y_train, configuration)
                    
                    # Update personal best
                    if score > personal_best_scores[i]:
                        personal_best_scores[i] = score
                        personal_best[i] = particle.copy()
                    
                    # Update global best
                    if score > global_best_score:
                        global_best_score = score
                        global_best = particle.copy()
                
                # Update velocities and positions
                for i in range(n_particles):
                    for j in range(len(particles[i])):
                        r1, r2 = np.random.random(2)
                        
                        # Update velocity
                        velocities[i][j] = (w * velocities[i][j] + 
                                          c1 * r1 * (personal_best[i][j] - particles[i][j]) +
                                          c2 * r2 * (global_best[j] - particles[i][j]))
                        
                        # Update position
                        particles[i][j] += velocities[i][j]
                
                # Record iteration
                optimization_history.append({
                    "iteration": iteration + 1,
                    "best_score": global_best_score,
                    "average_score": np.mean(personal_best_scores),
                    "timestamp": datetime.now().isoformat()
                })
            
            # Convert global best to parameter dictionary
            param_names = list(configuration.parameter_space.keys())
            best_parameters = {name: global_best[i] for i, name in enumerate(param_names)}
            
            return global_best_score, best_parameters, optimization_history
            
        except Exception as e:
            logger.error(f"Error in particle swarm optimization: {str(e)}")
            return -float('inf'), {}, []
    
    async def _hybrid_optimization(self, 
                                 X_train: np.ndarray,
                                 y_train: np.ndarray,
                                 X_test: np.ndarray,
                                 y_test: np.ndarray,
                                 configuration: ModelConfiguration,
                                 goal: OptimizationGoal) -> Tuple[float, Dict[str, Any], List[Dict[str, Any]]]:
        """Hybrid optimization combining multiple strategies"""
        try:
            # Phase 1: Random search for exploration
            logger.info("Phase 1: Random search exploration")
            random_score, random_params, random_history = await self._random_search_optimization(
                X_train, y_train, X_test, y_test, configuration, goal
            )
            
            # Phase 2: Bayesian optimization for exploitation
            logger.info("Phase 2: Bayesian optimization exploitation")
            # Update configuration with reduced search space around best random result
            refined_config = await self._refine_configuration(configuration, random_params)
            bayesian_score, bayesian_params, bayesian_history = await self._bayesian_optimization(
                X_train, y_train, X_test, y_test, refined_config, goal
            )
            
            # Choose best result
            if bayesian_score > random_score:
                best_score = bayesian_score
                best_parameters = bayesian_params
                optimization_history = random_history + bayesian_history
            else:
                best_score = random_score
                best_parameters = random_params
                optimization_history = random_history
            
            return best_score, best_parameters, optimization_history
            
        except Exception as e:
            logger.error(f"Error in hybrid optimization: {str(e)}")
            return -float('inf'), {}, []
    
    # Helper methods
    async def _create_model(self, algorithm: str, params: Dict[str, Any] = None) -> Any:
        """Create model instance"""
        try:
            if params is None:
                params = {}
            
            if algorithm == "RandomForestRegressor":
                return RandomForestRegressor(**params, random_state=42)
            elif algorithm == "GradientBoostingRegressor":
                return GradientBoostingRegressor(**params, random_state=42)
            elif algorithm == "LinearRegression":
                return LinearRegression(**params)
            elif algorithm == "Ridge":
                return Ridge(**params, random_state=42)
            elif algorithm == "Lasso":
                return Lasso(**params, random_state=42)
            elif algorithm == "ElasticNet":
                return ElasticNet(**params, random_state=42)
            elif algorithm == "SVR":
                return SVR(**params)
            elif algorithm == "MLPRegressor":
                return MLPRegressor(**params, random_state=42)
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
                
        except Exception as e:
            logger.error(f"Error creating model: {str(e)}")
            raise e
    
    async def _calculate_baseline_score(self, 
                                      X_train: np.ndarray,
                                      y_train: np.ndarray,
                                      X_test: np.ndarray,
                                      y_test: np.ndarray,
                                      configuration: ModelConfiguration) -> float:
        """Calculate baseline score with default parameters"""
        try:
            model = await self._create_model(configuration.base_algorithm)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            if configuration.scoring_metric == 'r2':
                return r2_score(y_test, y_pred)
            elif configuration.scoring_metric == 'neg_mean_squared_error':
                return -mean_squared_error(y_test, y_pred)
            elif configuration.scoring_metric == 'neg_mean_absolute_error':
                return -mean_absolute_error(y_test, y_pred)
            else:
                return r2_score(y_test, y_pred)  # Default
                
        except Exception as e:
            logger.error(f"Error calculating baseline score: {str(e)}")
            return 0.0
    
    async def _determine_best_strategy(self, 
                                     X_train: np.ndarray,
                                     y_train: np.ndarray,
                                     model_type: ModelType) -> OptimizationStrategy:
        """Determine best optimization strategy based on data characteristics"""
        try:
            n_samples, n_features = X_train.shape
            
            # Simple heuristics for strategy selection
            if n_samples < 1000:
                return OptimizationStrategy.GRID_SEARCH
            elif n_features > 100:
                return OptimizationStrategy.RANDOM_SEARCH
            elif n_samples > 10000:
                return OptimizationStrategy.BAYESIAN_OPTIMIZATION
            else:
                return OptimizationStrategy.HYBRID
                
        except Exception as e:
            logger.error(f"Error determining best strategy: {str(e)}")
            return OptimizationStrategy.BAYESIAN_OPTIMIZATION
    
    async def _create_default_configuration(self, model_type: ModelType) -> ModelConfiguration:
        """Create default configuration for model type"""
        try:
            if model_type == ModelType.REGRESSION:
                return ModelConfiguration(
                    model_type=model_type,
                    base_algorithm="RandomForestRegressor",
                    parameter_space={
                        "n_estimators": [50, 100, 200, 300],
                        "max_depth": [None, 10, 20, 30],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 2, 4]
                    },
                    constraints={},
                    optimization_budget=100,
                    cross_validation_folds=5,
                    scoring_metric="r2",
                    early_stopping=True,
                    parallel_processing=True
                )
            else:
                # Default configuration
                return ModelConfiguration(
                    model_type=model_type,
                    base_algorithm="RandomForestRegressor",
                    parameter_space={
                        "n_estimators": [50, 100, 200],
                        "max_depth": [None, 10, 20]
                    },
                    constraints={},
                    optimization_budget=50,
                    cross_validation_folds=3,
                    scoring_metric="r2",
                    early_stopping=True,
                    parallel_processing=True
                )
                
        except Exception as e:
            logger.error(f"Error creating default configuration: {str(e)}")
            raise e
    
    async def _determine_optimization_goal(self, 
                                         model_type: ModelType,
                                         n_samples: int) -> OptimizationGoal:
        """Determine optimization goal based on model type and data size"""
        try:
            if model_type == ModelType.REGRESSION:
                if n_samples > 10000:
                    return OptimizationGoal.MAXIMIZE_EFFICIENCY
                else:
                    return OptimizationGoal.MAXIMIZE_ACCURACY
            else:
                return OptimizationGoal.BALANCE_PERFORMANCE
                
        except Exception as e:
            logger.error(f"Error determining optimization goal: {str(e)}")
            return OptimizationGoal.MAXIMIZE_ACCURACY
    
    async def _analyze_strategy_performance(self) -> Dict[str, Any]:
        """Analyze performance of different optimization strategies"""
        try:
            strategy_performance = defaultdict(list)
            
            for result in self.optimization_results.values():
                strategy_performance[result.strategy.value].append(result.improvement_percentage)
            
            analysis = {}
            for strategy, improvements in strategy_performance.items():
                analysis[strategy] = {
                    "count": len(improvements),
                    "average_improvement": np.mean(improvements),
                    "best_improvement": np.max(improvements),
                    "std_improvement": np.std(improvements)
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing strategy performance: {str(e)}")
            return {}
    
    # Genetic algorithm helper methods
    async def _initialize_population(self, size: int, parameter_space: Dict[str, Any]) -> List[List[float]]:
        """Initialize population for genetic algorithm"""
        try:
            population = []
            param_names = list(parameter_space.keys())
            
            for _ in range(size):
                individual = []
                for param_name in param_names:
                    param_config = parameter_space[param_name]
                    if isinstance(param_config, list):
                        # Categorical parameter
                        individual.append(np.random.choice(param_config))
                    elif isinstance(param_config, tuple) and len(param_config) == 2:
                        # Numeric parameter
                        individual.append(np.random.uniform(param_config[0], param_config[1]))
                    else:
                        # Default range
                        individual.append(np.random.uniform(0.0, 1.0))
                
                population.append(individual)
            
            return population
            
        except Exception as e:
            logger.error(f"Error initializing population: {str(e)}")
            return []
    
    async def _evaluate_individual(self, 
                                 individual: List[float],
                                 X_train: np.ndarray,
                                 y_train: np.ndarray,
                                 configuration: ModelConfiguration) -> float:
        """Evaluate individual in genetic algorithm"""
        try:
            # Convert individual to parameters
            param_names = list(configuration.parameter_space.keys())
            params = {name: individual[i] for i, name in enumerate(param_names)}
            
            # Create and train model
            model = await self._create_model(configuration.base_algorithm, params)
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=configuration.cross_validation_folds,
                scoring=configuration.scoring_metric
            )
            
            return cv_scores.mean()
            
        except Exception as e:
            logger.error(f"Error evaluating individual: {str(e)}")
            return -float('inf')
    
    async def _tournament_selection(self, 
                                  population: List[List[float]],
                                  fitness_scores: List[float],
                                  tournament_size: int = 3) -> List[float]:
        """Tournament selection for genetic algorithm"""
        try:
            tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmax(tournament_fitness)]
            
            return population[winner_index].copy()
            
        except Exception as e:
            logger.error(f"Error in tournament selection: {str(e)}")
            return population[0].copy()
    
    async def _crossover(self, parent1: List[float], parent2: List[float]) -> List[float]:
        """Crossover operation for genetic algorithm"""
        try:
            child = []
            for i in range(len(parent1)):
                if np.random.random() < 0.5:
                    child.append(parent1[i])
                else:
                    child.append(parent2[i])
            
            return child
            
        except Exception as e:
            logger.error(f"Error in crossover: {str(e)}")
            return parent1.copy()
    
    async def _mutate(self, individual: List[float], parameter_space: Dict[str, Any]) -> List[float]:
        """Mutation operation for genetic algorithm"""
        try:
            mutated = individual.copy()
            param_names = list(parameter_space.keys())
            
            for i, param_name in enumerate(param_names):
                if np.random.random() < 0.1:  # 10% mutation rate per parameter
                    param_config = parameter_space[param_name]
                    if isinstance(param_config, list):
                        # Categorical parameter
                        mutated[i] = np.random.choice(param_config)
                    elif isinstance(param_config, tuple) and len(param_config) == 2:
                        # Numeric parameter
                        mutated[i] = np.random.uniform(param_config[0], param_config[1])
                    else:
                        # Default mutation
                        mutated[i] += np.random.normal(0, 0.1)
            
            return mutated
            
        except Exception as e:
            logger.error(f"Error in mutation: {str(e)}")
            return individual.copy()
    
    # Particle swarm optimization helper methods
    async def _initialize_particles(self, n_particles: int, parameter_space: Dict[str, Any]) -> List[List[float]]:
        """Initialize particles for PSO"""
        try:
            particles = []
            param_names = list(parameter_space.keys())
            
            for _ in range(n_particles):
                particle = []
                for param_name in param_names:
                    param_config = parameter_space[param_name]
                    if isinstance(param_config, list):
                        # Categorical parameter - convert to index
                        particle.append(np.random.randint(0, len(param_config)))
                    elif isinstance(param_config, tuple) and len(param_config) == 2:
                        # Numeric parameter
                        particle.append(np.random.uniform(param_config[0], param_config[1]))
                    else:
                        # Default range
                        particle.append(np.random.uniform(0.0, 1.0))
                
                particles.append(particle)
            
            return particles
            
        except Exception as e:
            logger.error(f"Error initializing particles: {str(e)}")
            return []
    
    async def _refine_configuration(self, 
                                  configuration: ModelConfiguration,
                                  best_params: Dict[str, Any]) -> ModelConfiguration:
        """Refine configuration around best parameters"""
        try:
            refined_space = {}
            
            for param_name, param_value in best_params.items():
                if param_name in configuration.parameter_space:
                    param_config = configuration.parameter_space[param_name]
                    
                    if isinstance(param_config, list):
                        # For categorical, keep the same options
                        refined_space[param_name] = param_config
                    elif isinstance(param_config, tuple) and len(param_config) == 2:
                        # For numeric, create smaller range around best value
                        range_size = (param_config[1] - param_config[0]) * 0.2  # 20% of original range
                        new_min = max(param_config[0], param_value - range_size)
                        new_max = min(param_config[1], param_value + range_size)
                        refined_space[param_name] = (new_min, new_max)
                    else:
                        refined_space[param_name] = param_config
            
            # Create refined configuration
            refined_config = ModelConfiguration(
                model_type=configuration.model_type,
                base_algorithm=configuration.base_algorithm,
                parameter_space=refined_space,
                constraints=configuration.constraints,
                optimization_budget=configuration.optimization_budget // 2,  # Reduce budget
                cross_validation_folds=configuration.cross_validation_folds,
                scoring_metric=configuration.scoring_metric,
                early_stopping=configuration.early_stopping,
                parallel_processing=configuration.parallel_processing
            )
            
            return refined_config
            
        except Exception as e:
            logger.error(f"Error refining configuration: {str(e)}")
            return configuration


# Global optimizer instance
_optimizer: Optional[IntelligentModelOptimizer] = None


def get_intelligent_model_optimizer(max_optimization_time: int = 3600) -> IntelligentModelOptimizer:
    """Get or create global intelligent model optimizer instance"""
    global _optimizer
    if _optimizer is None:
        _optimizer = IntelligentModelOptimizer(max_optimization_time)
    return _optimizer


# Example usage
async def main():
    """Example usage of the intelligent model optimizer"""
    optimizer = get_intelligent_model_optimizer()
    
    # Generate sample data
    np.random.seed(42)
    X_train = np.random.randn(1000, 10)
    y_train = np.random.randn(1000)
    X_test = np.random.randn(200, 10)
    y_test = np.random.randn(200)
    
    # Create configuration
    configuration = ModelConfiguration(
        model_type=ModelType.REGRESSION,
        base_algorithm="RandomForestRegressor",
        parameter_space={
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10]
        },
        constraints={},
        optimization_budget=50,
        cross_validation_folds=5,
        scoring_metric="r2",
        early_stopping=True,
        parallel_processing=True
    )
    
    # Optimize model
    result = await optimizer.optimize_model(
        model_name="sample_model",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        configuration=configuration,
        strategy=OptimizationStrategy.BAYESIAN_OPTIMIZATION,
        goal=OptimizationGoal.MAXIMIZE_ACCURACY
    )
    
    print(f"Optimization completed:")
    print(f"Best score: {result.best_score:.4f}")
    print(f"Improvement: {result.improvement_percentage:.2f}%")
    print(f"Best parameters: {result.best_parameters}")
    print(f"Optimization time: {result.optimization_time:.2f} seconds")
    
    # Get optimization summary
    summary = await optimizer.get_optimization_summary()
    print(f"\nOptimization Summary:")
    print(f"Total optimizations: {summary['total_optimizations']}")
    print(f"Success rate: {summary['success_rate']:.1f}%")
    print(f"Average improvement: {summary['average_improvement']:.2f}%")


if __name__ == "__main__":
    asyncio.run(main())



























