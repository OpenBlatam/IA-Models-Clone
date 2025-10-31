"""
Ultra-Advanced Swarm Intelligence Module
========================================

This module provides swarm intelligence capabilities for TruthGPT models,
including particle swarm optimization, ant colony optimization, and bee algorithms.

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

class SwarmAlgorithm(Enum):
    """Swarm intelligence algorithms."""
    PARTICLE_SWARM = "particle_swarm"
    ANT_COLONY = "ant_colony"
    BEE_ALGORITHM = "bee_algorithm"
    FIREFLY = "firefly"
    BAT_ALGORITHM = "bat_algorithm"
    CUCKOO_SEARCH = "cuckoo_search"
    WOLF_PACK = "wolf_pack"
    HYBRID_SWARM = "hybrid_swarm"

class SwarmTopology(Enum):
    """Swarm topologies."""
    RING = "ring"
    STAR = "star"
    MESH = "mesh"
    TREE = "tree"
    RANDOM = "random"
    SCALE_FREE = "scale_free"
    SMALL_WORLD = "small_world"

class CommunicationMode(Enum):
    """Communication modes in swarm."""
    GLOBAL = "global"
    LOCAL = "local"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE = "adaptive"
    STIGMERGY = "stigmergy"

class SwarmBehavior(Enum):
    """Swarm behaviors."""
    EXPLORATION = "exploration"
    EXPLOITATION = "exploitation"
    BALANCED = "balanced"
    ADAPTIVE = "adaptive"
    COOPERATIVE = "cooperative"

@dataclass
class SwarmConfig:
    """Configuration for swarm intelligence."""
    algorithm: SwarmAlgorithm = SwarmAlgorithm.PARTICLE_SWARM
    topology: SwarmTopology = SwarmTopology.RING
    communication_mode: CommunicationMode = CommunicationMode.GLOBAL
    behavior: SwarmBehavior = SwarmBehavior.BALANCED
    population_size: int = 50
    max_iterations: int = 1000
    convergence_threshold: float = 1e-6
    inertia_weight: float = 0.9
    cognitive_weight: float = 2.0
    social_weight: float = 2.0
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    device: str = "auto"
    log_level: str = "INFO"
    output_dir: str = "./swarm_results"

class Particle:
    """Particle in particle swarm optimization."""
    
    def __init__(self, position: np.ndarray, velocity: np.ndarray = None):
        self.position = position.copy()
        self.velocity = velocity if velocity is not None else np.zeros_like(position)
        self.best_position = position.copy()
        self.best_fitness = float('inf')
        self.fitness = float('inf')
        self.age = 0
        self.neighbors = []
        
    def update_velocity(self, global_best: np.ndarray, config: SwarmConfig):
        """Update particle velocity."""
        r1, r2 = random.random(), random.random()
        
        # Inertia component
        inertia = config.inertia_weight * self.velocity
        
        # Cognitive component
        cognitive = config.cognitive_weight * r1 * (self.best_position - self.position)
        
        # Social component
        social = config.social_weight * r2 * (global_best - self.position)
        
        self.velocity = inertia + cognitive + social
        
    def update_position(self):
        """Update particle position."""
        self.position += self.velocity
        self.age += 1
        
    def update_best(self):
        """Update personal best if current fitness is better."""
        if self.fitness < self.best_fitness:
            self.best_fitness = self.fitness
            self.best_position = self.position.copy()

class Ant:
    """Ant in ant colony optimization."""
    
    def __init__(self, start_node: int):
        self.current_node = start_node
        self.visited_nodes = [start_node]
        self.path = [start_node]
        self.path_length = 0.0
        self.pheromone_trail = []
        self.memory = set()
        
    def select_next_node(self, graph: np.ndarray, pheromones: np.ndarray, 
                        alpha: float = 1.0, beta: float = 2.0) -> int:
        """Select next node using probability distribution."""
        available_nodes = []
        probabilities = []
        
        for node in range(len(graph)):
            if node not in self.visited_nodes and graph[self.current_node][node] > 0:
                available_nodes.append(node)
                
                # Calculate probability based on pheromone and distance
                pheromone = pheromones[self.current_node][node] ** alpha
                distance = (1.0 / graph[self.current_node][node]) ** beta
                probabilities.append(pheromone * distance)
        
        if not available_nodes:
            return -1
        
        # Normalize probabilities
        total_prob = sum(probabilities)
        if total_prob > 0:
            probabilities = [p / total_prob for p in probabilities]
        else:
            probabilities = [1.0 / len(available_nodes)] * len(available_nodes)
        
        # Select node based on probabilities
        rand = random.random()
        cumulative_prob = 0.0
        
        for i, prob in enumerate(probabilities):
            cumulative_prob += prob
            if rand <= cumulative_prob:
                return available_nodes[i]
        
        return available_nodes[-1]
    
    def move_to_node(self, node: int, graph: np.ndarray):
        """Move ant to next node."""
        if node != -1 and graph[self.current_node][node] > 0:
            self.path_length += graph[self.current_node][node]
            self.current_node = node
            self.visited_nodes.append(node)
            self.path.append(node)
    
    def deposit_pheromone(self, pheromones: np.ndarray, evaporation_rate: float = 0.1):
        """Deposit pheromone on the path."""
        pheromone_amount = 1.0 / self.path_length if self.path_length > 0 else 0.0
        
        for i in range(len(self.path) - 1):
            from_node, to_node = self.path[i], self.path[i + 1]
            pheromones[from_node][to_node] += pheromone_amount

class Bee:
    """Bee in artificial bee colony algorithm."""
    
    def __init__(self, position: np.ndarray, bee_type: str = "worker"):
        self.position = position.copy()
        self.fitness = float('inf')
        self.bee_type = bee_type  # worker, scout, onlooker
        self.trial_count = 0
        self.max_trials = 10
        self.nectar_amount = 0.0
        
    def search_neighborhood(self, config: SwarmConfig) -> np.ndarray:
        """Search in neighborhood of current position."""
        if self.bee_type == "scout":
            # Scout bees explore randomly
            new_position = np.random.uniform(-10, 10, size=self.position.shape)
        else:
            # Worker/onlooker bees search locally
            perturbation = np.random.normal(0, 0.1, size=self.position.shape)
            new_position = self.position + perturbation
        
        return new_position
    
    def evaluate_fitness(self, objective_function: Callable) -> float:
        """Evaluate fitness using objective function."""
        try:
            self.fitness = objective_function(self.position)
            return self.fitness
        except Exception as e:
            logger.warning(f"Fitness evaluation failed: {e}")
            return float('inf')
    
    def should_abandon(self) -> bool:
        """Check if bee should abandon current position."""
        return self.trial_count >= self.max_trials

class SwarmTopologyManager:
    """Manages swarm topology and communication."""
    
    def __init__(self, config: SwarmConfig):
        self.config = config
        self.topology_graph = None
        self.neighbors = {}
        
    def create_topology(self, population_size: int):
        """Create topology graph."""
        if self.config.topology == SwarmTopology.RING:
            self._create_ring_topology(population_size)
        elif self.config.topology == SwarmTopology.STAR:
            self._create_star_topology(population_size)
        elif self.config.topology == SwarmTopology.MESH:
            self._create_mesh_topology(population_size)
        elif self.config.topology == SwarmTopology.TREE:
            self._create_tree_topology(population_size)
        elif self.config.topology == SwarmTopology.RANDOM:
            self._create_random_topology(population_size)
        elif self.config.topology == SwarmTopology.SCALE_FREE:
            self._create_scale_free_topology(population_size)
        else:  # SMALL_WORLD
            self._create_small_world_topology(population_size)
    
    def _create_ring_topology(self, population_size: int):
        """Create ring topology."""
        self.topology_graph = np.zeros((population_size, population_size))
        for i in range(population_size):
            self.topology_graph[i][(i + 1) % population_size] = 1
            self.topology_graph[i][(i - 1) % population_size] = 1
    
    def _create_star_topology(self, population_size: int):
        """Create star topology."""
        self.topology_graph = np.zeros((population_size, population_size))
        center = population_size // 2
        for i in range(population_size):
            if i != center:
                self.topology_graph[center][i] = 1
                self.topology_graph[i][center] = 1
    
    def _create_mesh_topology(self, population_size: int):
        """Create mesh topology."""
        self.topology_graph = np.zeros((population_size, population_size))
        grid_size = int(math.sqrt(population_size))
        
        for i in range(population_size):
            row, col = i // grid_size, i % grid_size
            
            # Connect to adjacent cells
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < grid_size and 0 <= new_col < grid_size:
                    neighbor = new_row * grid_size + new_col
                    self.topology_graph[i][neighbor] = 1
    
    def _create_tree_topology(self, population_size: int):
        """Create tree topology."""
        self.topology_graph = np.zeros((population_size, population_size))
        
        for i in range(1, population_size):
            parent = (i - 1) // 2
            self.topology_graph[parent][i] = 1
            self.topology_graph[i][parent] = 1
    
    def _create_random_topology(self, population_size: int):
        """Create random topology."""
        self.topology_graph = np.zeros((population_size, population_size))
        avg_degree = 4
        
        for i in range(population_size):
            num_connections = random.randint(1, avg_degree * 2)
            connections = random.sample(range(population_size), min(num_connections, population_size))
            
            for j in connections:
                if i != j:
                    self.topology_graph[i][j] = 1
    
    def _create_scale_free_topology(self, population_size: int):
        """Create scale-free topology."""
        self.topology_graph = np.zeros((population_size, population_size))
        
        # Start with a small connected network
        for i in range(min(3, population_size)):
            for j in range(i + 1, min(3, population_size)):
                self.topology_graph[i][j] = 1
                self.topology_graph[j][i] = 1
        
        # Add nodes with preferential attachment
        for i in range(3, population_size):
            degrees = np.sum(self.topology_graph, axis=1)
            probabilities = degrees / np.sum(degrees) if np.sum(degrees) > 0 else np.ones(population_size) / population_size
            
            # Select nodes to connect to
            num_connections = random.randint(1, 3)
            connections = np.random.choice(population_size, size=min(num_connections, i), 
                                         replace=False, p=probabilities[:i])
            
            for j in connections:
                self.topology_graph[i][j] = 1
                self.topology_graph[j][i] = 1
    
    def _create_small_world_topology(self, population_size: int):
        """Create small-world topology."""
        # Start with regular ring
        self._create_ring_topology(population_size)
        
        # Add random shortcuts
        rewiring_probability = 0.1
        for i in range(population_size):
            for j in range(i + 1, population_size):
                if self.topology_graph[i][j] == 1 and random.random() < rewiring_probability:
                    # Remove existing connection
                    self.topology_graph[i][j] = 0
                    self.topology_graph[j][i] = 0
                    
                    # Add random connection
                    new_j = random.randint(0, population_size - 1)
                    if new_j != i:
                        self.topology_graph[i][new_j] = 1
                        self.topology_graph[new_j][i] = 1
    
    def get_neighbors(self, node_id: int) -> List[int]:
        """Get neighbors of a node."""
        if self.topology_graph is None:
            return []
        
        neighbors = []
        for i in range(len(self.topology_graph)):
            if self.topology_graph[node_id][i] == 1:
                neighbors.append(i)
        
        return neighbors

class ParticleSwarmOptimizer:
    """Particle Swarm Optimization implementation."""
    
    def __init__(self, config: SwarmConfig):
        self.config = config
        self.particles = []
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.topology_manager = SwarmTopologyManager(config)
        self.fitness_history = deque(maxlen=1000)
        
    def initialize_swarm(self, problem_dimension: int, bounds: Tuple[float, float] = (-10, 10)):
        """Initialize particle swarm."""
        self.particles = []
        
        for i in range(self.config.population_size):
            position = np.random.uniform(bounds[0], bounds[1], size=problem_dimension)
            velocity = np.random.uniform(-1, 1, size=problem_dimension)
            particle = Particle(position, velocity)
            self.particles.append(particle)
        
        self.topology_manager.create_topology(self.config.population_size)
        
    def optimize(self, objective_function: Callable, problem_dimension: int, 
                bounds: Tuple[float, float] = (-10, 10)) -> Dict[str, Any]:
        """Run particle swarm optimization."""
        self.initialize_swarm(problem_dimension, bounds)
        
        start_time = time.time()
        
        for iteration in range(self.config.max_iterations):
            # Evaluate fitness for all particles
            for particle in self.particles:
                particle.fitness = objective_function(particle.position)
                particle.update_best()
                
                if particle.fitness < self.global_best_fitness:
                    self.global_best_fitness = particle.fitness
                    self.global_best_position = particle.position.copy()
            
            # Update velocities and positions
            for i, particle in enumerate(self.particles):
                # Get neighborhood best based on topology
                neighborhood_best = self._get_neighborhood_best(i)
                
                particle.update_velocity(neighborhood_best, self.config)
                particle.update_position()
                
                # Apply bounds
                particle.position = np.clip(particle.position, bounds[0], bounds[1])
            
            # Record fitness
            self.fitness_history.append(self.global_best_fitness)
            
            # Check convergence
            if len(self.fitness_history) >= 10:
                recent_improvement = abs(self.fitness_history[-1] - self.fitness_history[-10])
                if recent_improvement < self.config.convergence_threshold:
                    logger.info(f"Converged at iteration {iteration}")
                    break
        
        optimization_time = time.time() - start_time
        
        return {
            'best_position': self.global_best_position,
            'best_fitness': self.global_best_fitness,
            'iterations': iteration + 1,
            'optimization_time': optimization_time,
            'fitness_history': list(self.fitness_history),
            'converged': recent_improvement < self.config.convergence_threshold
        }
    
    def _get_neighborhood_best(self, particle_index: int) -> np.ndarray:
        """Get best position in particle's neighborhood."""
        if self.config.communication_mode == CommunicationMode.GLOBAL:
            return self.global_best_position
        
        neighbors = self.topology_manager.get_neighbors(particle_index)
        if not neighbors:
            return self.global_best_position
        
        best_fitness = float('inf')
        best_position = self.global_best_position
        
        for neighbor_idx in neighbors:
            neighbor = self.particles[neighbor_idx]
            if neighbor.best_fitness < best_fitness:
                best_fitness = neighbor.best_fitness
                best_position = neighbor.best_position
        
        return best_position

class AntColonyOptimizer:
    """Ant Colony Optimization implementation."""
    
    def __init__(self, config: SwarmConfig):
        self.config = config
        self.graph = None
        self.pheromones = None
        self.ants = []
        self.best_path = []
        self.best_path_length = float('inf')
        self.fitness_history = deque(maxlen=1000)
        
    def optimize(self, graph: np.ndarray, start_node: int = 0, 
                end_node: int = None) -> Dict[str, Any]:
        """Run ant colony optimization."""
        self.graph = graph.copy()
        self.pheromones = np.ones_like(graph) * 0.1
        end_node = end_node or len(graph) - 1
        
        start_time = time.time()
        
        for iteration in range(self.config.max_iterations):
            # Create ants
            self.ants = [Ant(start_node) for _ in range(self.config.population_size)]
            
            # Let ants build solutions
            for ant in self.ants:
                while ant.current_node != end_node:
                    next_node = ant.select_next_node(self.graph, self.pheromones)
                    if next_node == -1:
                        break
                    ant.move_to_node(next_node, self.graph)
                
                # Update best solution
                if ant.path_length < self.best_path_length:
                    self.best_path_length = ant.path_length
                    self.best_path = ant.path.copy()
            
            # Update pheromones
            self._update_pheromones()
            
            # Record fitness
            self.fitness_history.append(self.best_path_length)
            
            # Check convergence
            if len(self.fitness_history) >= 10:
                recent_improvement = abs(self.fitness_history[-1] - self.fitness_history[-10])
                if recent_improvement < self.config.convergence_threshold:
                    logger.info(f"Converged at iteration {iteration}")
                    break
        
        optimization_time = time.time() - start_time
        
        return {
            'best_path': self.best_path,
            'best_path_length': self.best_path_length,
            'iterations': iteration + 1,
            'optimization_time': optimization_time,
            'fitness_history': list(self.fitness_history),
            'converged': recent_improvement < self.config.convergence_threshold
        }
    
    def _update_pheromones(self):
        """Update pheromone trails."""
        # Evaporate pheromones
        evaporation_rate = 0.1
        self.pheromones *= (1 - evaporation_rate)
        
        # Deposit pheromones
        for ant in self.ants:
            ant.deposit_pheromone(self.pheromones, evaporation_rate)

class BeeColonyOptimizer:
    """Artificial Bee Colony optimization implementation."""
    
    def __init__(self, config: SwarmConfig):
        self.config = config
        self.bees = []
        self.best_position = None
        self.best_fitness = float('inf')
        self.fitness_history = deque(maxlen=1000)
        
    def optimize(self, objective_function: Callable, problem_dimension: int,
                bounds: Tuple[float, float] = (-10, 10)) -> Dict[str, Any]:
        """Run artificial bee colony optimization."""
        self._initialize_bees(problem_dimension, bounds)
        
        start_time = time.time()
        
        for iteration in range(self.config.max_iterations):
            # Employed bees phase
            self._employed_bees_phase(objective_function, bounds)
            
            # Onlooker bees phase
            self._onlooker_bees_phase(objective_function, bounds)
            
            # Scout bees phase
            self._scout_bees_phase(objective_function, bounds)
            
            # Update best solution
            for bee in self.bees:
                if bee.fitness < self.best_fitness:
                    self.best_fitness = bee.fitness
                    self.best_position = bee.position.copy()
            
            # Record fitness
            self.fitness_history.append(self.best_fitness)
            
            # Check convergence
            if len(self.fitness_history) >= 10:
                recent_improvement = abs(self.fitness_history[-1] - self.fitness_history[-10])
                if recent_improvement < self.config.convergence_threshold:
                    logger.info(f"Converged at iteration {iteration}")
                    break
        
        optimization_time = time.time() - start_time
        
        return {
            'best_position': self.best_position,
            'best_fitness': self.best_fitness,
            'iterations': iteration + 1,
            'optimization_time': optimization_time,
            'fitness_history': list(self.fitness_history),
            'converged': recent_improvement < self.config.convergence_threshold
        }
    
    def _initialize_bees(self, problem_dimension: int, bounds: Tuple[float, float]):
        """Initialize bee colony."""
        self.bees = []
        
        # Create employed bees
        for i in range(self.config.population_size // 2):
            position = np.random.uniform(bounds[0], bounds[1], size=problem_dimension)
            bee = Bee(position, "worker")
            self.bees.append(bee)
        
        # Create onlooker bees
        for i in range(self.config.population_size // 2):
            position = np.random.uniform(bounds[0], bounds[1], size=problem_dimension)
            bee = Bee(position, "onlooker")
            self.bees.append(bee)
    
    def _employed_bees_phase(self, objective_function: Callable, bounds: Tuple[float, float]):
        """Employed bees phase."""
        for bee in self.bees:
            if bee.bee_type == "worker":
                # Search neighborhood
                new_position = bee.search_neighborhood(self.config)
                new_position = np.clip(new_position, bounds[0], bounds[1])
                
                # Evaluate new position
                new_fitness = objective_function(new_position)
                
                # Update if better
                if new_fitness < bee.fitness:
                    bee.position = new_position
                    bee.fitness = new_fitness
                    bee.trial_count = 0
                else:
                    bee.trial_count += 1
    
    def _onlooker_bees_phase(self, objective_function: Callable, bounds: Tuple[float, float]):
        """Onlooker bees phase."""
        # Calculate selection probabilities
        fitnesses = [bee.fitness for bee in self.bees if bee.bee_type == "worker"]
        max_fitness = max(fitnesses) if fitnesses else 1.0
        
        probabilities = []
        for bee in self.bees:
            if bee.bee_type == "worker":
                prob = (max_fitness - bee.fitness + 1e-10) / (max_fitness + 1e-10)
                probabilities.append(prob)
            else:
                probabilities.append(0.0)
        
        # Normalize probabilities
        total_prob = sum(probabilities)
        if total_prob > 0:
            probabilities = [p / total_prob for p in probabilities]
        
        # Select bees for onlooker phase
        for bee in self.bees:
            if bee.bee_type == "onlooker":
                # Select based on probability
                rand = random.random()
                cumulative_prob = 0.0
                selected_bee = None
                
                for i, prob in enumerate(probabilities):
                    cumulative_prob += prob
                    if rand <= cumulative_prob:
                        selected_bee = self.bees[i]
                        break
                
                if selected_bee:
                    # Search around selected bee's position
                    perturbation = np.random.normal(0, 0.1, size=selected_bee.position.shape)
                    new_position = selected_bee.position + perturbation
                    new_position = np.clip(new_position, bounds[0], bounds[1])
                    
                    new_fitness = objective_function(new_position)
                    
                    if new_fitness < bee.fitness:
                        bee.position = new_position
                        bee.fitness = new_fitness
                        bee.trial_count = 0
                    else:
                        bee.trial_count += 1
    
    def _scout_bees_phase(self, objective_function: Callable, bounds: Tuple[float, float]):
        """Scout bees phase."""
        for bee in self.bees:
            if bee.should_abandon():
                # Abandon current position and search randomly
                bee.position = np.random.uniform(bounds[0], bounds[1], size=bee.position.shape)
                bee.fitness = objective_function(bee.position)
                bee.trial_count = 0
                bee.bee_type = "scout"
            else:
                bee.bee_type = "worker" if bee.bee_type == "scout" else bee.bee_type

class SwarmIntelligenceManager:
    """Main manager for swarm intelligence."""
    
    def __init__(self, config: SwarmConfig):
        self.config = config
        self.optimizers = {
            SwarmAlgorithm.PARTICLE_SWARM: ParticleSwarmOptimizer(config),
            SwarmAlgorithm.ANT_COLONY: AntColonyOptimizer(config),
            SwarmAlgorithm.BEE_ALGORITHM: BeeColonyOptimizer(config)
        }
        self.optimization_history = deque(maxlen=1000)
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def optimize(self, objective_function: Callable, problem_dimension: int,
                bounds: Tuple[float, float] = (-10, 10), graph: np.ndarray = None) -> Dict[str, Any]:
        """Run swarm intelligence optimization."""
        logger.info(f"Starting {self.config.algorithm.value} optimization")
        
        optimizer = self.optimizers.get(self.config.algorithm)
        if not optimizer:
            raise ValueError(f"Unsupported algorithm: {self.config.algorithm}")
        
        start_time = time.time()
        
        if self.config.algorithm == SwarmAlgorithm.PARTICLE_SWARM:
            result = optimizer.optimize(objective_function, problem_dimension, bounds)
        elif self.config.algorithm == SwarmAlgorithm.ANT_COLONY:
            if graph is None:
                raise ValueError("Graph required for ant colony optimization")
            result = optimizer.optimize(graph)
        elif self.config.algorithm == SwarmAlgorithm.BEE_ALGORITHM:
            result = optimizer.optimize(objective_function, problem_dimension, bounds)
        else:
            raise ValueError(f"Algorithm {self.config.algorithm} not implemented")
        
        optimization_time = time.time() - start_time
        
        # Record optimization
        optimization_record = {
            'algorithm': self.config.algorithm.value,
            'result': result,
            'optimization_time': optimization_time,
            'timestamp': time.time()
        }
        
        self.optimization_history.append(optimization_record)
        
        logger.info(f"Optimization completed in {optimization_time:.4f}s")
        
        return result
    
    def compare_algorithms(self, objective_function: Callable, problem_dimension: int,
                          bounds: Tuple[float, float] = (-10, 10)) -> Dict[str, Any]:
        """Compare different swarm algorithms."""
        results = {}
        
        for algorithm in [SwarmAlgorithm.PARTICLE_SWARM, SwarmAlgorithm.BEE_ALGORITHM]:
            config = copy.deepcopy(self.config)
            config.algorithm = algorithm
            
            optimizer = self.optimizers.get(algorithm)
            if optimizer:
                result = optimizer.optimize(objective_function, problem_dimension, bounds)
                results[algorithm.value] = result
        
        return results
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        if not self.optimization_history:
            return {'total_optimizations': 0}
        
        algorithms_used = [record['algorithm'] for record in self.optimization_history]
        optimization_times = [record['optimization_time'] for record in self.optimization_history]
        
        return {
            'total_optimizations': len(self.optimization_history),
            'average_optimization_time': statistics.mean(optimization_times),
            'algorithm_distribution': {alg: algorithms_used.count(alg) for alg in set(algorithms_used)},
            'current_algorithm': self.config.algorithm.value,
            'population_size': self.config.population_size,
            'max_iterations': self.config.max_iterations
        }

# Factory functions
def create_swarm_config(algorithm: SwarmAlgorithm = SwarmAlgorithm.PARTICLE_SWARM,
                       population_size: int = 50,
                       **kwargs) -> SwarmConfig:
    """Create swarm configuration."""
    return SwarmConfig(
        algorithm=algorithm,
        population_size=population_size,
        **kwargs
    )

def create_particle_swarm_optimizer(config: SwarmConfig) -> ParticleSwarmOptimizer:
    """Create particle swarm optimizer."""
    return ParticleSwarmOptimizer(config)

def create_ant_colony_optimizer(config: SwarmConfig) -> AntColonyOptimizer:
    """Create ant colony optimizer."""
    return AntColonyOptimizer(config)

def create_bee_colony_optimizer(config: SwarmConfig) -> BeeColonyOptimizer:
    """Create bee colony optimizer."""
    return BeeColonyOptimizer(config)

def create_swarm_intelligence_manager(config: Optional[SwarmConfig] = None) -> SwarmIntelligenceManager:
    """Create swarm intelligence manager."""
    if config is None:
        config = create_swarm_config()
    return SwarmIntelligenceManager(config)

# Example usage
def example_swarm_intelligence():
    """Example of swarm intelligence optimization."""
    # Test objective function
    def sphere_function(x):
        return np.sum(x**2)
    
    # Create configuration
    config = create_swarm_config(
        algorithm=SwarmAlgorithm.PARTICLE_SWARM,
        population_size=30,
        max_iterations=100
    )
    
    # Create manager
    manager = create_swarm_intelligence_manager(config)
    
    # Run optimization
    result = manager.optimize(sphere_function, problem_dimension=10)
    
    print(f"Best position: {result['best_position']}")
    print(f"Best fitness: {result['best_fitness']:.6f}")
    print(f"Iterations: {result['iterations']}")
    print(f"Optimization time: {result['optimization_time']:.4f}s")
    
    # Compare algorithms
    comparison = manager.compare_algorithms(sphere_function, problem_dimension=10)
    print(f"\nAlgorithm comparison:")
    for alg, res in comparison.items():
        print(f"{alg}: {res['best_fitness']:.6f} in {res['optimization_time']:.4f}s")
    
    # Get statistics
    stats = manager.get_optimization_statistics()
    print(f"\nStatistics: {stats}")
    
    return result

if __name__ == "__main__":
    # Run example
    example_swarm_intelligence()
