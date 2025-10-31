"""
Evolutionary Optimizer for Export IA
====================================

Advanced evolutionary algorithms for optimizing document processing
using genetic algorithms, particle swarm optimization, and other
bio-inspired optimization techniques.
"""

import asyncio
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import random
import copy
import math
from datetime import datetime
import uuid
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import differential_evolution
from scipy.stats import norm, uniform
import deap
from deap import base, creator, tools, algorithms
import pyswarm
import pyswarms
from pyswarms import single, multi
import optuna
from optuna import Trial, create_study, samplers, pruners
import hyperopt
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL
import skopt
from skopt import gp_minimize, forest_minimize, gbrt_minimize
import bayes_opt
from bayes_opt import BayesianOptimization
import nevergrad
import ax
from ax import optimize
import ray
from ray import tune
import wandb
import tensorboard
from tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

class OptimizationAlgorithm(Enum):
    """Evolutionary optimization algorithms."""
    GENETIC_ALGORITHM = "genetic_algorithm"
    PARTICLE_SWARM = "particle_swarm"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    SIMULATED_ANNEALING = "simulated_annealing"
    ANT_COLONY = "ant_colony"
    BEE_ALGORITHM = "bee_algorithm"
    FIREFLY_ALGORITHM = "firefly_algorithm"
    CUCKOO_SEARCH = "cuckoo_search"
    BAT_ALGORITHM = "bat_algorithm"
    WHALE_OPTIMIZATION = "whale_optimization"
    GREY_WOLF_OPTIMIZER = "grey_wolf_optimizer"
    DRAGONFLY_ALGORITHM = "dragonfly_algorithm"
    MOTH_FLAME_OPTIMIZATION = "moth_flame_optimization"
    SINE_COSINE_ALGORITHM = "sine_cosine_algorithm"
    SALP_SWARM_ALGORITHM = "salp_swarm_algorithm"
    GRASSHOPPER_OPTIMIZATION = "grasshopper_optimization"
    ANT_LION_OPTIMIZER = "ant_lion_optimizer"
    MULTI_VERSE_OPTIMIZER = "multi_verse_optimizer"
    HARMONY_SEARCH = "harmony_search"
    TEACHING_LEARNING_BASED = "teaching_learning_based"
    JAYA_ALGORITHM = "jaya_algorithm"
    SINE_COSINE_ALGORITHM = "sine_cosine_algorithm"
    SALP_SWARM_ALGORITHM = "salp_swarm_algorithm"
    GRASSHOPPER_OPTIMIZATION = "grasshopper_optimization"
    ANT_LION_OPTIMIZER = "ant_lion_optimizer"
    MULTI_VERSE_OPTIMIZER = "multi_verse_optimizer"
    HARMONY_SEARCH = "harmony_search"
    TEACHING_LEARNING_BASED = "teaching_learning_based"
    JAYA_ALGORITHM = "jaya_algorithm"

class SelectionStrategy(Enum):
    """Selection strategies for evolutionary algorithms."""
    ROULETTE_WHEEL = "roulette_wheel"
    TOURNAMENT = "tournament"
    RANK = "rank"
    STOCHASTIC_UNIVERSAL = "stochastic_universal"
    TRUNCATION = "truncation"
    ELITISM = "elitism"
    BOLTZMANN = "boltzmann"
    LINEAR_RANKING = "linear_ranking"
    EXPONENTIAL_RANKING = "exponential_ranking"
    PROPORTIONAL = "proportional"

class CrossoverStrategy(Enum):
    """Crossover strategies for genetic algorithms."""
    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"
    ARITHMETIC = "arithmetic"
    BLEND = "blend"
    SIMULATED_BINARY = "simulated_binary"
    POLYNOMIAL = "polynomial"
    DIFFERENTIAL = "differential"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"

class MutationStrategy(Enum):
    """Mutation strategies for genetic algorithms."""
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    POLYNOMIAL = "polynomial"
    BOUNDARY = "boundary"
    NON_UNIFORM = "non_uniform"
    CREEP = "creep"
    RANDOM = "random"
    ADAPTIVE = "adaptive"
    SELF_ADAPTIVE = "self_adaptive"
    DIFFERENTIAL = "differential"

@dataclass
class OptimizationConfig:
    """Configuration for evolutionary optimization."""
    algorithm: OptimizationAlgorithm = OptimizationAlgorithm.GENETIC_ALGORITHM
    population_size: int = 100
    generations: int = 1000
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    selection_strategy: SelectionStrategy = SelectionStrategy.TOURNAMENT
    crossover_strategy: CrossoverStrategy = CrossoverStrategy.UNIFORM
    mutation_strategy: MutationStrategy = MutationStrategy.GAUSSIAN
    tournament_size: int = 3
    elitism_size: int = 5
    convergence_threshold: float = 1e-6
    max_stagnation: int = 100
    parallel_evaluation: bool = True
    num_workers: int = 4
    seed: Optional[int] = None
    verbose: bool = True
    logging_frequency: int = 10
    save_frequency: int = 100
    checkpoint_frequency: int = 50

@dataclass
class Individual:
    """Individual in evolutionary algorithm."""
    id: str
    genes: np.ndarray
    fitness: float = 0.0
    age: int = 0
    parent_ids: List[str] = field(default_factory=list)
    mutation_count: int = 0
    crossover_count: int = 0
    evaluation_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class OptimizationResult:
    """Result of evolutionary optimization."""
    id: str
    algorithm: OptimizationAlgorithm
    best_individual: Individual
    population_history: List[List[Individual]]
    fitness_history: List[float]
    diversity_history: List[float]
    convergence_generation: int
    total_evaluations: int
    optimization_time: float
    final_fitness: float
    best_fitness: float
    average_fitness: float
    fitness_std: float
    diversity_score: float
    convergence_score: float
    performance_metrics: Dict[str, float]
    quality_scores: Dict[str, float]
    created_at: datetime = field(default_factory=datetime.now)

class EvolutionaryOptimizer:
    """Advanced evolutionary optimizer for document processing."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.population = []
        self.population_history = []
        self.fitness_history = []
        self.diversity_history = []
        self.best_individual = None
        self.generation = 0
        self.stagnation_count = 0
        self.total_evaluations = 0
        
        # Initialize random seed
        if self.config.seed is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)
        
        # Initialize optimization components
        self._initialize_optimization_components()
        
        logger.info(f"Evolutionary optimizer initialized with {config.algorithm.value}")
    
    def _initialize_optimization_components(self):
        """Initialize optimization components."""
        try:
            # Initialize DEAP components for genetic algorithm
            if self.config.algorithm == OptimizationAlgorithm.GENETIC_ALGORITHM:
                self._initialize_deap_components()
            
            # Initialize PySwarms for particle swarm optimization
            if self.config.algorithm == OptimizationAlgorithm.PARTICLE_SWARM:
                self._initialize_pyswarms_components()
            
            # Initialize other optimization libraries
            self._initialize_other_components()
            
            logger.info("Optimization components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize optimization components: {e}")
            raise
    
    def _initialize_deap_components(self):
        """Initialize DEAP components for genetic algorithm."""
        try:
            # Create fitness class
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax)
            
            # Create toolbox
            self.toolbox = base.Toolbox()
            
            # Register functions
            self.toolbox.register("attr_float", random.uniform, -1.0, 1.0)
            self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                                self.toolbox.attr_float, n=10)
            self.toolbox.register("population", tools.initRepeat, list,
                                self.toolbox.individual)
            
            # Register genetic operators
            self.toolbox.register("evaluate", self._evaluate_individual)
            self.toolbox.register("mate", self._crossover_individuals)
            self.toolbox.register("mutate", self._mutate_individual)
            self.toolbox.register("select", self._select_individuals)
            
            logger.info("DEAP components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize DEAP components: {e}")
            raise
    
    def _initialize_pyswarms_components(self):
        """Initialize PySwarms components for particle swarm optimization."""
        try:
            # Initialize PSO optimizer
            self.pso_optimizer = single.GlobalBestPSO(
                n_particles=self.config.population_size,
                dimensions=10,
                options={'c1': 0.5, 'c2': 0.3, 'w': 0.9}
            )
            
            logger.info("PySwarms components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize PySwarms components: {e}")
            raise
    
    def _initialize_other_components(self):
        """Initialize other optimization components."""
        try:
            # Initialize Optuna study
            self.optuna_study = create_study(
                direction='maximize',
                sampler=samplers.TPESampler(),
                pruner=pruners.MedianPruner()
            )
            
            # Initialize Hyperopt trials
            self.hyperopt_trials = Trials()
            
            # Initialize Scikit-optimize
            self.skopt_result = None
            
            # Initialize Bayesian optimization
            self.bayesian_optimizer = None
            
            logger.info("Other optimization components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize other components: {e}")
            raise
    
    async def optimize(
        self,
        objective_function: Callable[[np.ndarray], float],
        bounds: List[Tuple[float, float]],
        max_evaluations: int = 10000
    ) -> OptimizationResult:
        """Run evolutionary optimization."""
        
        start_time = datetime.now()
        result_id = str(uuid.uuid4())
        
        logger.info(f"Starting evolutionary optimization with {self.config.algorithm.value}")
        
        try:
            # Initialize population
            await self._initialize_population(bounds)
            
            # Run optimization based on algorithm
            if self.config.algorithm == OptimizationAlgorithm.GENETIC_ALGORITHM:
                await self._run_genetic_algorithm(objective_function, max_evaluations)
            elif self.config.algorithm == OptimizationAlgorithm.PARTICLE_SWARM:
                await self._run_particle_swarm_optimization(objective_function, bounds)
            elif self.config.algorithm == OptimizationAlgorithm.DIFFERENTIAL_EVOLUTION:
                await self._run_differential_evolution(objective_function, bounds)
            elif self.config.algorithm == OptimizationAlgorithm.SIMULATED_ANNEALING:
                await self._run_simulated_annealing(objective_function, bounds)
            else:
                await self._run_generic_optimization(objective_function, bounds)
            
            # Calculate final metrics
            optimization_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = OptimizationResult(
                id=result_id,
                algorithm=self.config.algorithm,
                best_individual=self.best_individual,
                population_history=self.population_history,
                fitness_history=self.fitness_history,
                diversity_history=self.diversity_history,
                convergence_generation=self.generation,
                total_evaluations=self.total_evaluations,
                optimization_time=optimization_time,
                final_fitness=self.best_individual.fitness if self.best_individual else 0.0,
                best_fitness=max(self.fitness_history) if self.fitness_history else 0.0,
                average_fitness=np.mean(self.fitness_history) if self.fitness_history else 0.0,
                fitness_std=np.std(self.fitness_history) if self.fitness_history else 0.0,
                diversity_score=self._calculate_diversity_score(),
                convergence_score=self._calculate_convergence_score(),
                performance_metrics=self._calculate_performance_metrics(optimization_time),
                quality_scores=self._calculate_quality_scores()
            )
            
            logger.info(f"Optimization completed in {optimization_time:.3f}s")
            logger.info(f"Best fitness: {result.best_fitness:.6f}")
            logger.info(f"Total evaluations: {result.total_evaluations}")
            
            return result
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise
    
    async def _initialize_population(self, bounds: List[Tuple[float, float]]):
        """Initialize population of individuals."""
        
        self.population = []
        
        for i in range(self.config.population_size):
            # Generate random genes within bounds
            genes = np.array([
                random.uniform(bounds[j][0], bounds[j][1])
                for j in range(len(bounds))
            ])
            
            individual = Individual(
                id=str(uuid.uuid4()),
                genes=genes
            )
            
            self.population.append(individual)
        
        logger.info(f"Initialized population of {len(self.population)} individuals")
    
    async def _run_genetic_algorithm(
        self,
        objective_function: Callable[[np.ndarray], float],
        max_evaluations: int
    ):
        """Run genetic algorithm optimization."""
        
        logger.info("Running genetic algorithm optimization")
        
        # Convert population to DEAP format
        deap_population = self.toolbox.population(n=self.config.population_size)
        
        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, deap_population))
        for ind, fit in zip(deap_population, fitnesses):
            ind.fitness.values = fit
        
        # Track best individual
        best_ind = tools.selBest(deap_population, 1)[0]
        self.best_individual = Individual(
            id=str(uuid.uuid4()),
            genes=np.array(best_ind),
            fitness=best_ind.fitness.values[0]
        )
        
        # Evolution loop
        for generation in range(self.config.generations):
            # Select parents
            offspring = self.toolbox.select(deap_population, len(deap_population))
            offspring = list(map(self.toolbox.clone, offspring))
            
            # Apply crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.config.crossover_rate:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # Apply mutation
            for mutant in offspring:
                if random.random() < self.config.mutation_rate:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluate invalid individuals
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Replace population
            deap_population[:] = offspring
            
            # Update best individual
            current_best = tools.selBest(deap_population, 1)[0]
            if current_best.fitness.values[0] > self.best_individual.fitness:
                self.best_individual = Individual(
                    id=str(uuid.uuid4()),
                    genes=np.array(current_best),
                    fitness=current_best.fitness.values[0]
                )
            
            # Track statistics
            self.generation = generation
            self.fitness_history.append(self.best_individual.fitness)
            self.diversity_history.append(self._calculate_population_diversity())
            
            # Check convergence
            if self._check_convergence():
                break
            
            # Log progress
            if generation % self.config.logging_frequency == 0:
                logger.info(f"Generation {generation}: Best fitness = {self.best_individual.fitness:.6f}")
    
    async def _run_particle_swarm_optimization(
        self,
        objective_function: Callable[[np.ndarray], float],
        bounds: List[Tuple[float, float]]
    ):
        """Run particle swarm optimization."""
        
        logger.info("Running particle swarm optimization")
        
        # Define objective function for PySwarms
        def pso_objective(particles):
            return [objective_function(particle) for particle in particles]
        
        # Run optimization
        cost, pos = self.pso_optimizer.optimize(
            pso_objective,
            iters=self.config.generations,
            verbose=self.config.verbose
        )
        
        # Update best individual
        self.best_individual = Individual(
            id=str(uuid.uuid4()),
            genes=pos,
            fitness=-cost  # PySwarms minimizes, we maximize
        )
        
        logger.info(f"PSO completed: Best fitness = {self.best_individual.fitness:.6f}")
    
    async def _run_differential_evolution(
        self,
        objective_function: Callable[[np.ndarray], float],
        bounds: List[Tuple[float, float]]
    ):
        """Run differential evolution optimization."""
        
        logger.info("Running differential evolution optimization")
        
        # Define objective function for scipy
        def de_objective(x):
            return -objective_function(x)  # Minimize negative for maximization
        
        # Run optimization
        result = differential_evolution(
            de_objective,
            bounds,
            maxiter=self.config.generations,
            popsize=self.config.population_size,
            seed=self.config.seed
        )
        
        # Update best individual
        self.best_individual = Individual(
            id=str(uuid.uuid4()),
            genes=result.x,
            fitness=-result.fun  # Convert back to maximization
        )
        
        logger.info(f"DE completed: Best fitness = {self.best_individual.fitness:.6f}")
    
    async def _run_simulated_annealing(
        self,
        objective_function: Callable[[np.ndarray], float],
        bounds: List[Tuple[float, float]]
    ):
        """Run simulated annealing optimization."""
        
        logger.info("Running simulated annealing optimization")
        
        # Initialize solution
        current_solution = np.array([
            random.uniform(bounds[i][0], bounds[i][1])
            for i in range(len(bounds))
        ])
        current_fitness = objective_function(current_solution)
        
        # Simulated annealing parameters
        initial_temp = 100.0
        final_temp = 0.1
        cooling_rate = 0.95
        
        temperature = initial_temp
        best_solution = current_solution.copy()
        best_fitness = current_fitness
        
        for iteration in range(self.config.generations):
            # Generate neighbor
            neighbor = current_solution + np.random.normal(0, 0.1, size=current_solution.shape)
            
            # Ensure bounds
            for i in range(len(bounds)):
                neighbor[i] = max(bounds[i][0], min(bounds[i][1], neighbor[i]))
            
            neighbor_fitness = objective_function(neighbor)
            
            # Accept or reject
            if neighbor_fitness > current_fitness or random.random() < np.exp((neighbor_fitness - current_fitness) / temperature):
                current_solution = neighbor
                current_fitness = neighbor_fitness
                
                if current_fitness > best_fitness:
                    best_solution = current_solution.copy()
                    best_fitness = current_fitness
            
            # Cool down
            temperature *= cooling_rate
            
            # Track statistics
            self.fitness_history.append(best_fitness)
            
            # Log progress
            if iteration % self.config.logging_frequency == 0:
                logger.info(f"Iteration {iteration}: Best fitness = {best_fitness:.6f}")
        
        # Update best individual
        self.best_individual = Individual(
            id=str(uuid.uuid4()),
            genes=best_solution,
            fitness=best_fitness
        )
        
        logger.info(f"SA completed: Best fitness = {self.best_individual.fitness:.6f}")
    
    async def _run_generic_optimization(
        self,
        objective_function: Callable[[np.ndarray], float],
        bounds: List[Tuple[float, float]]
    ):
        """Run generic optimization for unsupported algorithms."""
        
        logger.info("Running generic optimization")
        
        # Simple random search as fallback
        best_fitness = float('-inf')
        best_solution = None
        
        for iteration in range(self.config.generations):
            # Generate random solution
            solution = np.array([
                random.uniform(bounds[i][0], bounds[i][1])
                for i in range(len(bounds))
            ])
            
            fitness = objective_function(solution)
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_solution = solution
            
            # Track statistics
            self.fitness_history.append(best_fitness)
            
            # Log progress
            if iteration % self.config.logging_frequency == 0:
                logger.info(f"Iteration {iteration}: Best fitness = {best_fitness:.6f}")
        
        # Update best individual
        self.best_individual = Individual(
            id=str(uuid.uuid4()),
            genes=best_solution,
            fitness=best_fitness
        )
        
        logger.info(f"Generic optimization completed: Best fitness = {self.best_individual.fitness:.6f}")
    
    def _evaluate_individual(self, individual) -> Tuple[float]:
        """Evaluate individual fitness."""
        # This is a placeholder - should be replaced with actual objective function
        fitness = np.sum(np.array(individual)**2)  # Simple sphere function
        self.total_evaluations += 1
        return (fitness,)
    
    def _crossover_individuals(self, ind1, ind2):
        """Crossover two individuals."""
        if self.config.crossover_strategy == CrossoverStrategy.UNIFORM:
            for i in range(len(ind1)):
                if random.random() < 0.5:
                    ind1[i], ind2[i] = ind2[i], ind1[i]
        elif self.config.crossover_strategy == CrossoverStrategy.SINGLE_POINT:
            point = random.randint(1, len(ind1) - 1)
            ind1[point:], ind2[point:] = ind2[point:], ind1[point:]
        elif self.config.crossover_strategy == CrossoverStrategy.TWO_POINT:
            point1 = random.randint(1, len(ind1) - 1)
            point2 = random.randint(point1, len(ind1) - 1)
            ind1[point1:point2], ind2[point1:point2] = ind2[point1:point2], ind1[point1:point2]
    
    def _mutate_individual(self, individual):
        """Mutate individual."""
        if self.config.mutation_strategy == MutationStrategy.GAUSSIAN:
            for i in range(len(individual)):
                if random.random() < self.config.mutation_rate:
                    individual[i] += random.gauss(0, 0.1)
        elif self.config.mutation_strategy == MutationStrategy.UNIFORM:
            for i in range(len(individual)):
                if random.random() < self.config.mutation_rate:
                    individual[i] = random.uniform(-1, 1)
    
    def _select_individuals(self, population, k):
        """Select individuals from population."""
        if self.config.selection_strategy == SelectionStrategy.TOURNAMENT:
            return tools.selTournament(population, k, tournsize=self.config.tournament_size)
        elif self.config.selection_strategy == SelectionStrategy.ROULETTE_WHEEL:
            return tools.selRoulette(population, k)
        elif self.config.selection_strategy == SelectionStrategy.RANK:
            return tools.selBest(population, k)
        else:
            return tools.selBest(population, k)
    
    def _calculate_population_diversity(self) -> float:
        """Calculate population diversity."""
        if len(self.population) < 2:
            return 0.0
        
        # Calculate average pairwise distance
        total_distance = 0.0
        count = 0
        
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                distance = np.linalg.norm(
                    self.population[i].genes - self.population[j].genes
                )
                total_distance += distance
                count += 1
        
        return total_distance / count if count > 0 else 0.0
    
    def _check_convergence(self) -> bool:
        """Check if optimization has converged."""
        if len(self.fitness_history) < 10:
            return False
        
        # Check if fitness has improved recently
        recent_fitness = self.fitness_history[-10:]
        if max(recent_fitness) - min(recent_fitness) < self.config.convergence_threshold:
            self.stagnation_count += 1
        else:
            self.stagnation_count = 0
        
        return self.stagnation_count >= self.config.max_stagnation
    
    def _calculate_diversity_score(self) -> float:
        """Calculate diversity score."""
        if not self.diversity_history:
            return 0.0
        
        return np.mean(self.diversity_history)
    
    def _calculate_convergence_score(self) -> float:
        """Calculate convergence score."""
        if len(self.fitness_history) < 2:
            return 0.0
        
        # Calculate improvement rate
        initial_fitness = self.fitness_history[0]
        final_fitness = self.fitness_history[-1]
        
        if initial_fitness == 0:
            return 0.0
        
        return (final_fitness - initial_fitness) / abs(initial_fitness)
    
    def _calculate_performance_metrics(self, optimization_time: float) -> Dict[str, float]:
        """Calculate performance metrics."""
        return {
            "optimization_time": optimization_time,
            "evaluations_per_second": self.total_evaluations / optimization_time if optimization_time > 0 else 0.0,
            "convergence_speed": self.generation / optimization_time if optimization_time > 0 else 0.0,
            "efficiency": self.best_individual.fitness / self.total_evaluations if self.total_evaluations > 0 else 0.0,
            "stability": 1.0 - (np.std(self.fitness_history) / np.mean(self.fitness_history)) if self.fitness_history and np.mean(self.fitness_history) > 0 else 0.0,
            "robustness": len(self.fitness_history) / self.config.generations if self.config.generations > 0 else 0.0,
            "scalability": self.total_evaluations / self.config.population_size if self.config.population_size > 0 else 0.0,
            "memory_efficiency": len(self.population) / self.config.population_size if self.config.population_size > 0 else 0.0,
            "cpu_efficiency": self.total_evaluations / optimization_time if optimization_time > 0 else 0.0,
            "gpu_efficiency": 0.0  # Placeholder
        }
    
    def _calculate_quality_scores(self) -> Dict[str, float]:
        """Calculate quality scores."""
        return {
            "solution_quality": self.best_individual.fitness if self.best_individual else 0.0,
            "convergence_quality": self._calculate_convergence_score(),
            "diversity_quality": self._calculate_diversity_score(),
            "robustness_quality": 1.0 - (np.std(self.fitness_history) / np.mean(self.fitness_history)) if self.fitness_history and np.mean(self.fitness_history) > 0 else 0.0,
            "efficiency_quality": self.best_individual.fitness / self.total_evaluations if self.total_evaluations > 0 else 0.0,
            "stability_quality": 1.0 - (np.std(self.fitness_history) / np.mean(self.fitness_history)) if self.fitness_history and np.mean(self.fitness_history) > 0 else 0.0,
            "scalability_quality": self.total_evaluations / self.config.population_size if self.config.population_size > 0 else 0.0,
            "adaptability_quality": len(self.fitness_history) / self.config.generations if self.config.generations > 0 else 0.0,
            "innovation_quality": self._calculate_diversity_score(),
            "creativity_quality": self._calculate_convergence_score()
        }

# Global evolutionary optimizer instance
_global_evolutionary_optimizer: Optional[EvolutionaryOptimizer] = None

def get_global_evolutionary_optimizer() -> EvolutionaryOptimizer:
    """Get the global evolutionary optimizer instance."""
    global _global_evolutionary_optimizer
    if _global_evolutionary_optimizer is None:
        config = OptimizationConfig(algorithm=OptimizationAlgorithm.GENETIC_ALGORITHM)
        _global_evolutionary_optimizer = EvolutionaryOptimizer(config)
    return _global_evolutionary_optimizer



























