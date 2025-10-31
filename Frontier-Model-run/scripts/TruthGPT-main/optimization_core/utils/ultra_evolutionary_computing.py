"""
Ultra-Advanced Evolutionary Computing Module
============================================

This module provides evolutionary computing capabilities for TruthGPT models,
including genetic algorithms, evolutionary strategies, and genetic programming.

Author: TruthGPT Ultra-Advanced Optimization Core Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy
import time
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import pickle
from pathlib import Path
import concurrent.futures
from collections import defaultdict, deque
import math
import statistics
import warnings
import threading
import queue
import asyncio
from abc import ABC, abstractmethod

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

class EvolutionaryAlgorithm(Enum):
    """Evolutionary algorithms."""
    GENETIC_ALGORITHM = "genetic_algorithm"
    EVOLUTIONARY_STRATEGY = "evolutionary_strategy"
    GENETIC_PROGRAMMING = "genetic_programming"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    PARTICLE_SWARM = "particle_swarm"
    EVOLUTIONARY_PROGRAMMING = "evolutionary_programming"
    MEMETIC_ALGORITHM = "memetic_algorithm"
    CO_EVOLUTIONARY = "co_evolutionary"

class SelectionMethod(Enum):
    """Selection methods."""
    ROULETTE_WHEEL = "roulette_wheel"
    TOURNAMENT = "tournament"
    RANK = "rank"
    ELITIST = "elitist"
    STOCHASTIC_UNIVERSAL = "stochastic_universal"
    TRUNCATION = "truncation"
    LINEAR_RANKING = "linear_ranking"
    EXPONENTIAL_RANKING = "exponential_ranking"

class CrossoverMethod(Enum):
    """Crossover methods."""
    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"
    ARITHMETIC = "arithmetic"
    BLEND = "blend"
    SIMULATED_BINARY = "simulated_binary"
    ORDER = "order"
    CYCLE = "cycle"

class MutationMethod(Enum):
    """Mutation methods."""
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    POLYNOMIAL = "polynomial"
    NON_UNIFORM = "non_uniform"
    CREEP = "creep"
    BOUNDARY = "boundary"
    RANDOM_RESET = "random_reset"
    SCRAMBLE = "scramble"

class ReplacementStrategy(Enum):
    """Replacement strategies."""
    GENERATIONAL = "generational"
    STEADY_STATE = "steady_state"
    ELITIST = "elitist"
    CROWDING = "crowding"
    FITNESS_SHARING = "fitness_sharing"
    DETERMINISTIC_CROWDING = "deterministic_crowding"

@dataclass
class EvolutionaryConfig:
    """Configuration for evolutionary computing."""
    algorithm: EvolutionaryAlgorithm = EvolutionaryAlgorithm.GENETIC_ALGORITHM
    population_size: int = 100
    max_generations: int = 1000
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    selection_method: SelectionMethod = SelectionMethod.TOURNAMENT
    crossover_method: CrossoverMethod = CrossoverMethod.SINGLE_POINT
    mutation_method: MutationMethod = MutationMethod.GAUSSIAN
    replacement_strategy: ReplacementStrategy = ReplacementStrategy.GENERATIONAL
    tournament_size: int = 3
    elite_size: int = 5
    convergence_threshold: float = 1e-6
    diversity_threshold: float = 0.1
    device: str = "auto"
    log_level: str = "INFO"
    output_dir: str = "./evolutionary_results"

class Individual:
    """Individual in evolutionary algorithm."""
    
    def __init__(self, chromosome: np.ndarray, fitness: float = None):
        self.chromosome = chromosome.copy()
        self.fitness = fitness if fitness is not None else float('inf')
        self.age = 0
        self.generation = 0
        self.parents = []
        self.children = []
        self.metadata = {}
        
    def copy(self):
        """Create a copy of the individual."""
        new_individual = Individual(self.chromosome, self.fitness)
        new_individual.age = self.age
        new_individual.generation = self.generation
        new_individual.parents = self.parents.copy()
        new_individual.children = self.children.copy()
        new_individual.metadata = self.metadata.copy()
        return new_individual
    
    def mutate(self, mutation_method: MutationMethod, mutation_rate: float, 
               bounds: Tuple[float, float] = None) -> 'Individual':
        """Mutate the individual."""
        mutated = self.copy()
        
        if mutation_method == MutationMethod.GAUSSIAN:
            mutated = self._gaussian_mutation(mutated, mutation_rate, bounds)
        elif mutation_method == MutationMethod.UNIFORM:
            mutated = self._uniform_mutation(mutated, mutation_rate, bounds)
        elif mutation_method == MutationMethod.POLYNOMIAL:
            mutated = self._polynomial_mutation(mutated, mutation_rate, bounds)
        elif mutation_method == MutationMethod.NON_UNIFORM:
            mutated = self._non_uniform_mutation(mutated, mutation_rate, bounds)
        elif mutation_method == MutationMethod.CREEP:
            mutated = self._creep_mutation(mutated, mutation_rate, bounds)
        elif mutation_method == MutationMethod.BOUNDARY:
            mutated = self._boundary_mutation(mutated, mutation_rate, bounds)
        elif mutation_method == MutationMethod.RANDOM_RESET:
            mutated = self._random_reset_mutation(mutated, mutation_rate, bounds)
        else:  # SCRAMBLE
            mutated = self._scramble_mutation(mutated, mutation_rate)
        
        return mutated
    
    def _gaussian_mutation(self, individual: 'Individual', mutation_rate: float, 
                          bounds: Tuple[float, float]) -> 'Individual':
        """Gaussian mutation."""
        mask = np.random.random(len(individual.chromosome)) < mutation_rate
        noise = np.random.normal(0, 0.1, size=len(individual.chromosome))
        individual.chromosome[mask] += noise[mask]
        
        if bounds:
            individual.chromosome = np.clip(individual.chromosome, bounds[0], bounds[1])
        
        return individual
    
    def _uniform_mutation(self, individual: 'Individual', mutation_rate: float,
                         bounds: Tuple[float, float]) -> 'Individual':
        """Uniform mutation."""
        mask = np.random.random(len(individual.chromosome)) < mutation_rate
        
        if bounds:
            individual.chromosome[mask] = np.random.uniform(bounds[0], bounds[1], size=np.sum(mask))
        else:
            individual.chromosome[mask] = np.random.uniform(-10, 10, size=np.sum(mask))
        
        return individual
    
    def _polynomial_mutation(self, individual: 'Individual', mutation_rate: float,
                           bounds: Tuple[float, float]) -> 'Individual':
        """Polynomial mutation."""
        mask = np.random.random(len(individual.chromosome)) < mutation_rate
        
        for i in range(len(individual.chromosome)):
            if mask[i]:
                u = random.random()
                if u < 0.5:
                    delta = (2 * u) ** (1 / 3) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / 3)
                
                if bounds:
                    individual.chromosome[i] += delta * (bounds[1] - bounds[0])
                    individual.chromosome[i] = np.clip(individual.chromosome[i], bounds[0], bounds[1])
                else:
                    individual.chromosome[i] += delta * 10
        
        return individual
    
    def _non_uniform_mutation(self, individual: 'Individual', mutation_rate: float,
                             bounds: Tuple[float, float]) -> 'Individual':
        """Non-uniform mutation."""
        mask = np.random.random(len(individual.chromosome)) < mutation_rate
        
        for i in range(len(individual.chromosome)):
            if mask[i]:
                u = random.random()
                if u < 0.5:
                    delta = (2 * u) ** (1 / (1 + individual.age)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / (1 + individual.age))
                
                if bounds:
                    individual.chromosome[i] += delta * (bounds[1] - bounds[0])
                    individual.chromosome[i] = np.clip(individual.chromosome[i], bounds[0], bounds[1])
                else:
                    individual.chromosome[i] += delta * 10
        
        return individual
    
    def _creep_mutation(self, individual: 'Individual', mutation_rate: float,
                       bounds: Tuple[float, float]) -> 'Individual':
        """Creep mutation."""
        mask = np.random.random(len(individual.chromosome)) < mutation_rate
        creep_amount = 0.01
        
        individual.chromosome[mask] += np.random.normal(0, creep_amount, size=np.sum(mask))
        
        if bounds:
            individual.chromosome = np.clip(individual.chromosome, bounds[0], bounds[1])
        
        return individual
    
    def _boundary_mutation(self, individual: 'Individual', mutation_rate: float,
                          bounds: Tuple[float, float]) -> 'Individual':
        """Boundary mutation."""
        if not bounds:
            return individual
        
        mask = np.random.random(len(individual.chromosome)) < mutation_rate
        
        for i in range(len(individual.chromosome)):
            if mask[i]:
                individual.chromosome[i] = random.choice([bounds[0], bounds[1]])
        
        return individual
    
    def _random_reset_mutation(self, individual: 'Individual', mutation_rate: float,
                              bounds: Tuple[float, float]) -> 'Individual':
        """Random reset mutation."""
        mask = np.random.random(len(individual.chromosome)) < mutation_rate
        
        if bounds:
            individual.chromosome[mask] = np.random.uniform(bounds[0], bounds[1], size=np.sum(mask))
        else:
            individual.chromosome[mask] = np.random.uniform(-10, 10, size=np.sum(mask))
        
        return individual
    
    def _scramble_mutation(self, individual: 'Individual', mutation_rate: float) -> 'Individual':
        """Scramble mutation."""
        if random.random() < mutation_rate:
            start = random.randint(0, len(individual.chromosome) - 1)
            end = random.randint(start, len(individual.chromosome))
            segment = individual.chromosome[start:end]
            np.random.shuffle(segment)
            individual.chromosome[start:end] = segment
        
        return individual

class SelectionOperator:
    """Selection operator for evolutionary algorithms."""
    
    @staticmethod
    def select(population: List[Individual], selection_method: SelectionMethod,
               num_parents: int, tournament_size: int = 3) -> List[Individual]:
        """Select parents from population."""
        if selection_method == SelectionMethod.ROULETTE_WHEEL:
            return SelectionOperator._roulette_wheel_selection(population, num_parents)
        elif selection_method == SelectionMethod.TOURNAMENT:
            return SelectionOperator._tournament_selection(population, num_parents, tournament_size)
        elif selection_method == SelectionMethod.RANK:
            return SelectionOperator._rank_selection(population, num_parents)
        elif selection_method == SelectionMethod.ELITIST:
            return SelectionOperator._elitist_selection(population, num_parents)
        elif selection_method == SelectionMethod.STOCHASTIC_UNIVERSAL:
            return SelectionOperator._stochastic_universal_selection(population, num_parents)
        elif selection_method == SelectionMethod.TRUNCATION:
            return SelectionOperator._truncation_selection(population, num_parents)
        elif selection_method == SelectionMethod.LINEAR_RANKING:
            return SelectionOperator._linear_ranking_selection(population, num_parents)
        else:  # EXPONENTIAL_RANKING
            return SelectionOperator._exponential_ranking_selection(population, num_parents)
    
    @staticmethod
    def _roulette_wheel_selection(population: List[Individual], num_parents: int) -> List[Individual]:
        """Roulette wheel selection."""
        # Convert fitness to selection probabilities
        fitnesses = [ind.fitness for ind in population]
        max_fitness = max(fitnesses)
        min_fitness = min(fitnesses)
        
        # Handle minimization problem
        if max_fitness == min_fitness:
            probabilities = [1.0 / len(population)] * len(population)
        else:
            # Invert fitness for minimization
            inverted_fitnesses = [max_fitness - f + 1e-10 for f in fitnesses]
            total_fitness = sum(inverted_fitnesses)
            probabilities = [f / total_fitness for f in inverted_fitnesses]
        
        # Select parents
        parents = []
        for _ in range(num_parents):
            rand = random.random()
            cumulative_prob = 0.0
            
            for i, prob in enumerate(probabilities):
                cumulative_prob += prob
                if rand <= cumulative_prob:
                    parents.append(population[i])
                    break
        
        return parents
    
    @staticmethod
    def _tournament_selection(population: List[Individual], num_parents: int, 
                            tournament_size: int) -> List[Individual]:
        """Tournament selection."""
        parents = []
        
        for _ in range(num_parents):
            tournament = random.sample(population, min(tournament_size, len(population)))
            winner = min(tournament, key=lambda x: x.fitness)
            parents.append(winner)
        
        return parents
    
    @staticmethod
    def _rank_selection(population: List[Individual], num_parents: int) -> List[Individual]:
        """Rank selection."""
        # Sort by fitness
        sorted_population = sorted(population, key=lambda x: x.fitness)
        
        # Assign ranks
        ranks = list(range(1, len(sorted_population) + 1))
        total_rank = sum(ranks)
        
        # Calculate selection probabilities
        probabilities = [rank / total_rank for rank in ranks]
        
        # Select parents
        parents = []
        for _ in range(num_parents):
            rand = random.random()
            cumulative_prob = 0.0
            
            for i, prob in enumerate(probabilities):
                cumulative_prob += prob
                if rand <= cumulative_prob:
                    parents.append(sorted_population[i])
                    break
        
        return parents
    
    @staticmethod
    def _elitist_selection(population: List[Individual], num_parents: int) -> List[Individual]:
        """Elitist selection."""
        sorted_population = sorted(population, key=lambda x: x.fitness)
        return sorted_population[:num_parents]
    
    @staticmethod
    def _stochastic_universal_selection(population: List[Individual], num_parents: int) -> List[Individual]:
        """Stochastic universal selection."""
        fitnesses = [ind.fitness for ind in population]
        max_fitness = max(fitnesses)
        min_fitness = min(fitnesses)
        
        if max_fitness == min_fitness:
            probabilities = [1.0 / len(population)] * len(population)
        else:
            inverted_fitnesses = [max_fitness - f + 1e-10 for f in fitnesses]
            total_fitness = sum(inverted_fitnesses)
            probabilities = [f / total_fitness for f in inverted_fitnesses]
        
        parents = []
        step = 1.0 / num_parents
        start = random.uniform(0, step)
        
        cumulative_prob = 0.0
        current_parent = 0
        
        for i, prob in enumerate(probabilities):
            cumulative_prob += prob
            
            while current_parent < num_parents and start + current_parent * step <= cumulative_prob:
                parents.append(population[i])
                current_parent += 1
        
        return parents
    
    @staticmethod
    def _truncation_selection(population: List[Individual], num_parents: int) -> List[Individual]:
        """Truncation selection."""
        sorted_population = sorted(population, key=lambda x: x.fitness)
        truncation_point = len(sorted_population) // 2
        
        parents = []
        for _ in range(num_parents):
            parents.append(random.choice(sorted_population[:truncation_point]))
        
        return parents
    
    @staticmethod
    def _linear_ranking_selection(population: List[Individual], num_parents: int) -> List[Individual]:
        """Linear ranking selection."""
        sorted_population = sorted(population, key=lambda x: x.fitness)
        
        # Linear ranking probabilities
        s = 1.5  # Selection pressure
        probabilities = []
        
        for i in range(len(sorted_population)):
            prob = (2 - s) / len(sorted_population) + 2 * (s - 1) * (len(sorted_population) - i - 1) / (len(sorted_population) * (len(sorted_population) - 1))
            probabilities.append(prob)
        
        # Select parents
        parents = []
        for _ in range(num_parents):
            rand = random.random()
            cumulative_prob = 0.0
            
            for i, prob in enumerate(probabilities):
                cumulative_prob += prob
                if rand <= cumulative_prob:
                    parents.append(sorted_population[i])
                    break
        
        return parents
    
    @staticmethod
    def _exponential_ranking_selection(population: List[Individual], num_parents: int) -> List[Individual]:
        """Exponential ranking selection."""
        sorted_population = sorted(population, key=lambda x: x.fitness)
        
        # Exponential ranking probabilities
        c = 0.9  # Exponential factor
        probabilities = []
        
        for i in range(len(sorted_population)):
            prob = (c ** (len(sorted_population) - i - 1)) * (1 - c) / (1 - c ** len(sorted_population))
            probabilities.append(prob)
        
        # Select parents
        parents = []
        for _ in range(num_parents):
            rand = random.random()
            cumulative_prob = 0.0
            
            for i, prob in enumerate(probabilities):
                cumulative_prob += prob
                if rand <= cumulative_prob:
                    parents.append(sorted_population[i])
                    break
        
        return parents

class CrossoverOperator:
    """Crossover operator for evolutionary algorithms."""
    
    @staticmethod
    def crossover(parent1: Individual, parent2: Individual, crossover_method: CrossoverMethod,
                 crossover_rate: float) -> Tuple[Individual, Individual]:
        """Perform crossover between two parents."""
        if random.random() > crossover_rate:
            return parent1.copy(), parent2.copy()
        
        if crossover_method == CrossoverMethod.SINGLE_POINT:
            return CrossoverOperator._single_point_crossover(parent1, parent2)
        elif crossover_method == CrossoverMethod.TWO_POINT:
            return CrossoverOperator._two_point_crossover(parent1, parent2)
        elif crossover_method == CrossoverMethod.UNIFORM:
            return CrossoverOperator._uniform_crossover(parent1, parent2)
        elif crossover_method == CrossoverMethod.ARITHMETIC:
            return CrossoverOperator._arithmetic_crossover(parent1, parent2)
        elif crossover_method == CrossoverMethod.BLEND:
            return CrossoverOperator._blend_crossover(parent1, parent2)
        elif crossover_method == CrossoverMethod.SIMULATED_BINARY:
            return CrossoverOperator._simulated_binary_crossover(parent1, parent2)
        elif crossover_method == CrossoverMethod.ORDER:
            return CrossoverOperator._order_crossover(parent1, parent2)
        else:  # CYCLE
            return CrossoverOperator._cycle_crossover(parent1, parent2)
    
    @staticmethod
    def _single_point_crossover(parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Single point crossover."""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        if len(parent1.chromosome) > 1:
            crossover_point = random.randint(1, len(parent1.chromosome) - 1)
            
            child1.chromosome[crossover_point:] = parent2.chromosome[crossover_point:]
            child2.chromosome[crossover_point:] = parent1.chromosome[crossover_point:]
        
        return child1, child2
    
    @staticmethod
    def _two_point_crossover(parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Two point crossover."""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        if len(parent1.chromosome) > 2:
            point1 = random.randint(0, len(parent1.chromosome) - 2)
            point2 = random.randint(point1 + 1, len(parent1.chromosome) - 1)
            
            child1.chromosome[point1:point2] = parent2.chromosome[point1:point2]
            child2.chromosome[point1:point2] = parent1.chromosome[point1:point2]
        
        return child1, child2
    
    @staticmethod
    def _uniform_crossover(parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Uniform crossover."""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        for i in range(len(parent1.chromosome)):
            if random.random() < 0.5:
                child1.chromosome[i] = parent2.chromosome[i]
                child2.chromosome[i] = parent1.chromosome[i]
        
        return child1, child2
    
    @staticmethod
    def _arithmetic_crossover(parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Arithmetic crossover."""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        alpha = random.random()
        
        child1.chromosome = alpha * parent1.chromosome + (1 - alpha) * parent2.chromosome
        child2.chromosome = (1 - alpha) * parent1.chromosome + alpha * parent2.chromosome
        
        return child1, child2
    
    @staticmethod
    def _blend_crossover(parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Blend crossover."""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        alpha = 0.5
        
        for i in range(len(parent1.chromosome)):
            if parent1.chromosome[i] < parent2.chromosome[i]:
                low, high = parent1.chromosome[i], parent2.chromosome[i]
            else:
                low, high = parent2.chromosome[i], parent1.chromosome[i]
            
            diff = high - low
            
            child1.chromosome[i] = low - alpha * diff + random.random() * (1 + 2 * alpha) * diff
            child2.chromosome[i] = low - alpha * diff + random.random() * (1 + 2 * alpha) * diff
        
        return child1, child2
    
    @staticmethod
    def _simulated_binary_crossover(parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Simulated binary crossover."""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        eta_c = 20  # Distribution index
        
        for i in range(len(parent1.chromosome)):
            u = random.random()
            
            if u <= 0.5:
                beta = (2 * u) ** (1 / (eta_c + 1))
            else:
                beta = (1 / (2 * (1 - u))) ** (1 / (eta_c + 1))
            
            child1.chromosome[i] = 0.5 * ((1 + beta) * parent1.chromosome[i] + (1 - beta) * parent2.chromosome[i])
            child2.chromosome[i] = 0.5 * ((1 - beta) * parent1.chromosome[i] + (1 + beta) * parent2.chromosome[i])
        
        return child1, child2
    
    @staticmethod
    def _order_crossover(parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Order crossover for permutation problems."""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        if len(parent1.chromosome) > 2:
            point1 = random.randint(0, len(parent1.chromosome) - 2)
            point2 = random.randint(point1 + 1, len(parent1.chromosome) - 1)
            
            # Copy segment from parent1 to child1
            child1.chromosome[point1:point2] = parent1.chromosome[point1:point2]
            
            # Fill remaining positions from parent2
            remaining = [x for x in parent2.chromosome if x not in child1.chromosome[point1:point2]]
            remaining_idx = 0
            
            for i in range(len(child1.chromosome)):
                if i < point1 or i >= point2:
                    child1.chromosome[i] = remaining[remaining_idx]
                    remaining_idx += 1
            
            # Similar for child2
            child2.chromosome[point1:point2] = parent2.chromosome[point1:point2]
            
            remaining = [x for x in parent1.chromosome if x not in child2.chromosome[point1:point2]]
            remaining_idx = 0
            
            for i in range(len(child2.chromosome)):
                if i < point1 or i >= point2:
                    child2.chromosome[i] = remaining[remaining_idx]
                    remaining_idx += 1
        
        return child1, child2
    
    @staticmethod
    def _cycle_crossover(parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Cycle crossover for permutation problems."""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        if len(parent1.chromosome) > 1:
            # Find cycles
            visited = [False] * len(parent1.chromosome)
            cycles = []
            
            for i in range(len(parent1.chromosome)):
                if not visited[i]:
                    cycle = []
                    current = i
                    
                    while not visited[current]:
                        visited[current] = True
                        cycle.append(current)
                        current = np.where(parent2.chromosome == parent1.chromosome[current])[0][0]
                    
                    cycles.append(cycle)
            
            # Alternate cycles between children
            for cycle_idx, cycle in enumerate(cycles):
                if cycle_idx % 2 == 0:
                    for pos in cycle:
                        child1.chromosome[pos] = parent1.chromosome[pos]
                        child2.chromosome[pos] = parent2.chromosome[pos]
                else:
                    for pos in cycle:
                        child1.chromosome[pos] = parent2.chromosome[pos]
                        child2.chromosome[pos] = parent1.chromosome[pos]
        
        return child1, child2

class GeneticAlgorithm:
    """Genetic Algorithm implementation."""
    
    def __init__(self, config: EvolutionaryConfig):
        self.config = config
        self.population = []
        self.best_individual = None
        self.generation = 0
        self.fitness_history = deque(maxlen=1000)
        self.diversity_history = deque(maxlen=1000)
        
    def initialize_population(self, problem_dimension: int, bounds: Tuple[float, float] = (-10, 10)):
        """Initialize population."""
        self.population = []
        
        for i in range(self.config.population_size):
            chromosome = np.random.uniform(bounds[0], bounds[1], size=problem_dimension)
            individual = Individual(chromosome)
            self.population.append(individual)
        
        self.generation = 0
    
    def evolve(self, objective_function: Callable, problem_dimension: int,
               bounds: Tuple[float, float] = (-10, 10)) -> Dict[str, Any]:
        """Run genetic algorithm evolution."""
        self.initialize_population(problem_dimension, bounds)
        
        start_time = time.time()
        
        for generation in range(self.config.max_generations):
            self.generation = generation
            
            # Evaluate fitness
            for individual in self.population:
                individual.fitness = objective_function(individual.chromosome)
                individual.generation = generation
            
            # Update best individual
            current_best = min(self.population, key=lambda x: x.fitness)
            if self.best_individual is None or current_best.fitness < self.best_individual.fitness:
                self.best_individual = current_best.copy()
            
            # Record fitness and diversity
            self.fitness_history.append(self.best_individual.fitness)
            self.diversity_history.append(self._calculate_diversity())
            
            # Check convergence
            if len(self.fitness_history) >= 10:
                recent_improvement = abs(self.fitness_history[-1] - self.fitness_history[-10])
                if recent_improvement < self.config.convergence_threshold:
                    logger.info(f"Converged at generation {generation}")
                    break
            
            # Create next generation
            self._create_next_generation(bounds)
        
        evolution_time = time.time() - start_time
        
        return {
            'best_individual': self.best_individual,
            'best_fitness': self.best_individual.fitness,
            'generations': generation + 1,
            'evolution_time': evolution_time,
            'fitness_history': list(self.fitness_history),
            'diversity_history': list(self.diversity_history),
            'converged': recent_improvement < self.config.convergence_threshold
        }
    
    def _create_next_generation(self, bounds: Tuple[float, float]):
        """Create next generation."""
        new_population = []
        
        # Elitism
        if self.config.replacement_strategy == ReplacementStrategy.ELITIST:
            sorted_population = sorted(self.population, key=lambda x: x.fitness)
            elite = sorted_population[:self.config.elite_size]
            new_population.extend(elite)
        
        # Generate offspring
        while len(new_population) < self.config.population_size:
            # Selection
            parents = SelectionOperator.select(
                self.population, 
                self.config.selection_method,
                2, 
                self.config.tournament_size
            )
            
            # Crossover
            offspring = CrossoverOperator.crossover(
                parents[0], 
                parents[1], 
                self.config.crossover_method,
                self.config.crossover_rate
            )
            
            # Mutation
            for child in offspring:
                child = child.mutate(
                    self.config.mutation_method,
                    self.config.mutation_rate,
                    bounds
                )
                child.age = 0
                child.generation = self.generation + 1
                new_population.append(child)
        
        # Replace population
        if self.config.replacement_strategy == ReplacementStrategy.GENERATIONAL:
            self.population = new_population[:self.config.population_size]
        elif self.config.replacement_strategy == ReplacementStrategy.STEADY_STATE:
            self._steady_state_replacement(new_population)
        else:  # ELITIST
            self.population = new_population[:self.config.population_size]
    
    def _steady_state_replacement(self, offspring: List[Individual]):
        """Steady state replacement."""
        # Sort population by fitness
        self.population.sort(key=lambda x: x.fitness)
        
        # Replace worst individuals
        for child in offspring:
            if child.fitness < self.population[-1].fitness:
                self.population[-1] = child
                self.population.sort(key=lambda x: x.fitness)
    
    def _calculate_diversity(self) -> float:
        """Calculate population diversity."""
        if len(self.population) < 2:
            return 0.0
        
        # Calculate average pairwise distance
        total_distance = 0.0
        count = 0
        
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                distance = np.linalg.norm(
                    self.population[i].chromosome - self.population[j].chromosome
                )
                total_distance += distance
                count += 1
        
        return total_distance / count if count > 0 else 0.0

class EvolutionaryStrategy:
    """Evolutionary Strategy implementation."""
    
    def __init__(self, config: EvolutionaryConfig):
        self.config = config
        self.population = []
        self.best_individual = None
        self.generation = 0
        self.fitness_history = deque(maxlen=1000)
        
    def initialize_population(self, problem_dimension: int, bounds: Tuple[float, float] = (-10, 10)):
        """Initialize population."""
        self.population = []
        
        for i in range(self.config.population_size):
            chromosome = np.random.uniform(bounds[0], bounds[1], size=problem_dimension)
            # Add strategy parameters (mutation strengths)
            strategy_params = np.random.uniform(0.1, 1.0, size=problem_dimension)
            chromosome = np.concatenate([chromosome, strategy_params])
            
            individual = Individual(chromosome)
            individual.metadata['strategy_params'] = strategy_params
            self.population.append(individual)
        
        self.generation = 0
    
    def evolve(self, objective_function: Callable, problem_dimension: int,
               bounds: Tuple[float, float] = (-10, 10)) -> Dict[str, Any]:
        """Run evolutionary strategy evolution."""
        self.initialize_population(problem_dimension, bounds)
        
        start_time = time.time()
        
        for generation in range(self.config.max_generations):
            self.generation = generation
            
            # Evaluate fitness
            for individual in self.population:
                # Extract solution and strategy parameters
                solution = individual.chromosome[:problem_dimension]
                individual.fitness = objective_function(solution)
                individual.generation = generation
            
            # Update best individual
            current_best = min(self.population, key=lambda x: x.fitness)
            if self.best_individual is None or current_best.fitness < self.best_individual.fitness:
                self.best_individual = current_best.copy()
            
            # Record fitness
            self.fitness_history.append(self.best_individual.fitness)
            
            # Check convergence
            if len(self.fitness_history) >= 10:
                recent_improvement = abs(self.fitness_history[-1] - self.fitness_history[-10])
                if recent_improvement < self.config.convergence_threshold:
                    logger.info(f"Converged at generation {generation}")
                    break
            
            # Create next generation
            self._create_next_generation(problem_dimension, bounds)
        
        evolution_time = time.time() - start_time
        
        return {
            'best_individual': self.best_individual,
            'best_fitness': self.best_individual.fitness,
            'generations': generation + 1,
            'evolution_time': evolution_time,
            'fitness_history': list(self.fitness_history),
            'converged': recent_improvement < self.config.convergence_threshold
        }
    
    def _create_next_generation(self, problem_dimension: int, bounds: Tuple[float, float]):
        """Create next generation using ES."""
        new_population = []
        
        # Generate offspring
        for _ in range(self.config.population_size):
            # Select parent
            parent = random.choice(self.population)
            
            # Create offspring
            offspring = parent.copy()
            
            # Self-adaptive mutation
            offspring = self._self_adaptive_mutation(offspring, problem_dimension, bounds)
            
            offspring.age = 0
            offspring.generation = self.generation + 1
            new_population.append(offspring)
        
        # Replace population
        self.population = new_population
    
    def _self_adaptive_mutation(self, individual: Individual, problem_dimension: int,
                               bounds: Tuple[float, float]) -> Individual:
        """Self-adaptive mutation."""
        # Extract solution and strategy parameters
        solution = individual.chromosome[:problem_dimension]
        strategy_params = individual.chromosome[problem_dimension:]
        
        # Mutate strategy parameters first
        tau = 1.0 / math.sqrt(2 * problem_dimension)
        tau_prime = 1.0 / math.sqrt(2 * math.sqrt(problem_dimension))
        
        # Global strategy parameter
        global_strategy = strategy_params[0] if len(strategy_params) > 0 else 0.1
        global_strategy *= math.exp(tau_prime * random.gauss(0, 1))
        
        # Individual strategy parameters
        for i in range(len(strategy_params)):
            strategy_params[i] *= math.exp(tau * random.gauss(0, 1))
            strategy_params[i] = max(0.001, strategy_params[i])  # Prevent too small values
        
        # Mutate solution using strategy parameters
        for i in range(problem_dimension):
            sigma = strategy_params[i] if i < len(strategy_params) else global_strategy
            solution[i] += sigma * random.gauss(0, 1)
            solution[i] = np.clip(solution[i], bounds[0], bounds[1])
        
        # Update chromosome
        individual.chromosome = np.concatenate([solution, strategy_params])
        
        return individual

class EvolutionaryComputingManager:
    """Main manager for evolutionary computing."""
    
    def __init__(self, config: EvolutionaryConfig):
        self.config = config
        self.algorithms = {
            EvolutionaryAlgorithm.GENETIC_ALGORITHM: GeneticAlgorithm(config),
            EvolutionaryAlgorithm.EVOLUTIONARY_STRATEGY: EvolutionaryStrategy(config)
        }
        self.evolution_history = deque(maxlen=1000)
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def evolve(self, objective_function: Callable, problem_dimension: int,
               bounds: Tuple[float, float] = (-10, 10)) -> Dict[str, Any]:
        """Run evolutionary algorithm."""
        logger.info(f"Starting {self.config.algorithm.value} evolution")
        
        algorithm = self.algorithms.get(self.config.algorithm)
        if not algorithm:
            raise ValueError(f"Unsupported algorithm: {self.config.algorithm}")
        
        start_time = time.time()
        
        if self.config.algorithm == EvolutionaryAlgorithm.GENETIC_ALGORITHM:
            result = algorithm.evolve(objective_function, problem_dimension, bounds)
        elif self.config.algorithm == EvolutionaryAlgorithm.EVOLUTIONARY_STRATEGY:
            result = algorithm.evolve(objective_function, problem_dimension, bounds)
        else:
            raise ValueError(f"Algorithm {self.config.algorithm} not implemented")
        
        evolution_time = time.time() - start_time
        
        # Record evolution
        evolution_record = {
            'algorithm': self.config.algorithm.value,
            'result': result,
            'evolution_time': evolution_time,
            'timestamp': time.time()
        }
        
        self.evolution_history.append(evolution_record)
        
        logger.info(f"Evolution completed in {evolution_time:.4f}s")
        
        return result
    
    def compare_algorithms(self, objective_function: Callable, problem_dimension: int,
                          bounds: Tuple[float, float] = (-10, 10)) -> Dict[str, Any]:
        """Compare different evolutionary algorithms."""
        results = {}
        
        for algorithm in [EvolutionaryAlgorithm.GENETIC_ALGORITHM, 
                         EvolutionaryAlgorithm.EVOLUTIONARY_STRATEGY]:
            config = copy.deepcopy(self.config)
            config.algorithm = algorithm
            
            algorithm_instance = self.algorithms.get(algorithm)
            if algorithm_instance:
                result = algorithm_instance.evolve(objective_function, problem_dimension, bounds)
                results[algorithm.value] = result
        
        return results
    
    def get_evolution_statistics(self) -> Dict[str, Any]:
        """Get evolution statistics."""
        if not self.evolution_history:
            return {'total_evolutions': 0}
        
        algorithms_used = [record['algorithm'] for record in self.evolution_history]
        evolution_times = [record['evolution_time'] for record in self.evolution_history]
        
        return {
            'total_evolutions': len(self.evolution_history),
            'average_evolution_time': statistics.mean(evolution_times),
            'algorithm_distribution': {alg: algorithms_used.count(alg) for alg in set(algorithms_used)},
            'current_algorithm': self.config.algorithm.value,
            'population_size': self.config.population_size,
            'max_generations': self.config.max_generations
        }

# Factory functions
def create_evolutionary_config(algorithm: EvolutionaryAlgorithm = EvolutionaryAlgorithm.GENETIC_ALGORITHM,
                             population_size: int = 100,
                             **kwargs) -> EvolutionaryConfig:
    """Create evolutionary configuration."""
    return EvolutionaryConfig(
        algorithm=algorithm,
        population_size=population_size,
        **kwargs
    )

def create_genetic_algorithm(config: EvolutionaryConfig) -> GeneticAlgorithm:
    """Create genetic algorithm."""
    return GeneticAlgorithm(config)

def create_evolutionary_strategy(config: EvolutionaryConfig) -> EvolutionaryStrategy:
    """Create evolutionary strategy."""
    return EvolutionaryStrategy(config)

def create_evolutionary_computing_manager(config: Optional[EvolutionaryConfig] = None) -> EvolutionaryComputingManager:
    """Create evolutionary computing manager."""
    if config is None:
        config = create_evolutionary_config()
    return EvolutionaryComputingManager(config)

# Example usage
def example_evolutionary_computing():
    """Example of evolutionary computing."""
    # Test objective function
    def sphere_function(x):
        return np.sum(x**2)
    
    # Create configuration
    config = create_evolutionary_config(
        algorithm=EvolutionaryAlgorithm.GENETIC_ALGORITHM,
        population_size=50,
        max_generations=100
    )
    
    # Create manager
    manager = create_evolutionary_computing_manager(config)
    
    # Run evolution
    result = manager.evolve(sphere_function, problem_dimension=10)
    
    print(f"Best fitness: {result['best_fitness']:.6f}")
    print(f"Generations: {result['generations']}")
    print(f"Evolution time: {result['evolution_time']:.4f}s")
    
    # Compare algorithms
    comparison = manager.compare_algorithms(sphere_function, problem_dimension=10)
    print(f"\nAlgorithm comparison:")
    for alg, res in comparison.items():
        print(f"{alg}: {res['best_fitness']:.6f} in {res['evolution_time']:.4f}s")
    
    # Get statistics
    stats = manager.get_evolution_statistics()
    print(f"\nStatistics: {stats}")
    
    return result

if __name__ == "__main__":
    # Run example
    example_evolutionary_computing()
