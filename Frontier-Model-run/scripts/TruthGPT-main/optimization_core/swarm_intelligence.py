"""
Advanced Neural Network Swarm Intelligence System for TruthGPT Optimization Core
Complete swarm intelligence with particle swarm optimization and ant colony optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import logging
from dataclasses import dataclass, field
from enum import Enum
import time
import json
from pathlib import Path
import pickle
from abc import ABC, abstractmethod
import math
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class SwarmAlgorithm(Enum):
    """Swarm intelligence algorithms"""
    PARTICLE_SWARM_OPTIMIZATION = "particle_swarm_optimization"
    ANT_COLONY_OPTIMIZATION = "ant_colony_optimization"
    BEE_ALGORITHM = "bee_algorithm"
    FIREFLY_ALGORITHM = "firefly_algorithm"
    BAT_ALGORITHM = "bat_algorithm"
    CUCKOO_SEARCH = "cuckoo_search"
    GRAY_WOLF_OPTIMIZER = "gray_wolf_optimizer"
    WHALE_OPTIMIZATION = "whale_optimization"

class SwarmBehavior(Enum):
    """Swarm behaviors"""
    EXPLORATION = "exploration"
    EXPLOITATION = "exploitation"
    BALANCED = "balanced"
    ADAPTIVE = "adaptive"
    COLLABORATIVE = "collaborative"
    COMPETITIVE = "competitive"

class SwarmConfig:
    """Configuration for swarm intelligence system"""
    # Basic settings
    algorithm: SwarmAlgorithm = SwarmAlgorithm.PARTICLE_SWARM_OPTIMIZATION
    behavior: SwarmBehavior = SwarmBehavior.BALANCED
    
    # Swarm settings
    population_size: int = 50
    max_iterations: int = 100
    dimension: int = 10
    
    # PSO settings
    inertia_weight: float = 0.9
    cognitive_weight: float = 2.0
    social_weight: float = 2.0
    velocity_limit: float = 1.0
    
    # ACO settings
    num_ants: int = 30
    evaporation_rate: float = 0.1
    pheromone_constant: float = 1.0
    alpha: float = 1.0
    beta: float = 2.0
    
    # Bee algorithm settings
    num_scout_bees: int = 10
    num_employed_bees: int = 20
    num_onlooker_bees: int = 20
    patch_size: float = 0.1
    
    # Firefly algorithm settings
    absorption_coefficient: float = 1.0
    attractiveness: float = 1.0
    randomization_parameter: float = 0.2
    
    # Advanced features
    enable_adaptive_parameters: bool = True
    enable_swarm_diversity: bool = True
    enable_swarm_memory: bool = True
    enable_swarm_communication: bool = True
    
    def __post_init__(self):
        """Validate swarm configuration"""
        if self.population_size <= 0:
            raise ValueError("Population size must be positive")
        if self.max_iterations <= 0:
            raise ValueError("Max iterations must be positive")
        if self.dimension <= 0:
            raise ValueError("Dimension must be positive")
        if not (0 <= self.inertia_weight <= 1):
            raise ValueError("Inertia weight must be between 0 and 1")
        if self.cognitive_weight <= 0:
            raise ValueError("Cognitive weight must be positive")
        if self.social_weight <= 0:
            raise ValueError("Social weight must be positive")
        if self.velocity_limit <= 0:
            raise ValueError("Velocity limit must be positive")
        if self.num_ants <= 0:
            raise ValueError("Number of ants must be positive")
        if not (0 < self.evaporation_rate < 1):
            raise ValueError("Evaporation rate must be between 0 and 1")
        if self.pheromone_constant <= 0:
            raise ValueError("Pheromone constant must be positive")
        if self.alpha <= 0:
            raise ValueError("Alpha must be positive")
        if self.beta <= 0:
            raise ValueError("Beta must be positive")

class Particle:
    """Particle for Particle Swarm Optimization"""
    
    def __init__(self, dimension: int, bounds: Tuple[float, float] = (-5.0, 5.0)):
        self.dimension = dimension
        self.bounds = bounds
        
        # Initialize position and velocity
        self.position = np.random.uniform(bounds[0], bounds[1], dimension)
        self.velocity = np.random.uniform(-1.0, 1.0, dimension)
        
        # Initialize best positions
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')
        
        # Current fitness
        self.fitness = float('inf')
        
        logger.info(f"✅ Particle initialized with dimension {dimension}")
    
    def update_velocity(self, global_best_position: np.ndarray, 
                       inertia_weight: float, cognitive_weight: float, 
                       social_weight: float, velocity_limit: float):
        """Update particle velocity"""
        # Generate random numbers
        r1 = np.random.random(self.dimension)
        r2 = np.random.random(self.dimension)
        
        # Calculate velocity components
        inertia_component = inertia_weight * self.velocity
        cognitive_component = cognitive_weight * r1 * (self.best_position - self.position)
        social_component = social_weight * r2 * (global_best_position - self.position)
        
        # Update velocity
        self.velocity = inertia_component + cognitive_component + social_component
        
        # Apply velocity limit
        velocity_norm = np.linalg.norm(self.velocity)
        if velocity_norm > velocity_limit:
            self.velocity = self.velocity / velocity_norm * velocity_limit
    
    def update_position(self):
        """Update particle position"""
        self.position += self.velocity
        
        # Apply bounds
        self.position = np.clip(self.position, self.bounds[0], self.bounds[1])
    
    def evaluate_fitness(self, objective_function: Callable):
        """Evaluate particle fitness"""
        self.fitness = objective_function(self.position)
        
        # Update best position if current fitness is better
        if self.fitness < self.best_fitness:
            self.best_fitness = self.fitness
            self.best_position = self.position.copy()

class Ant:
    """Ant for Ant Colony Optimization"""
    
    def __init__(self, num_cities: int):
        self.num_cities = num_cities
        self.tour = []
        self.tour_length = 0.0
        self.visited = np.zeros(num_cities, dtype=bool)
        
        logger.info(f"✅ Ant initialized with {num_cities} cities")
    
    def construct_tour(self, pheromone_matrix: np.ndarray, distance_matrix: np.ndarray,
                      alpha: float, beta: float, start_city: int = 0):
        """Construct tour using pheromone and distance information"""
        self.tour = [start_city]
        self.visited = np.zeros(self.num_cities, dtype=bool)
        self.visited[start_city] = True
        current_city = start_city
        
        # Construct tour
        for _ in range(self.num_cities - 1):
            next_city = self._select_next_city(current_city, pheromone_matrix, 
                                             distance_matrix, alpha, beta)
            self.tour.append(next_city)
            self.visited[next_city] = True
            current_city = next_city
        
        # Calculate tour length
        self.tour_length = self._calculate_tour_length(distance_matrix)
    
    def _select_next_city(self, current_city: int, pheromone_matrix: np.ndarray,
                         distance_matrix: np.ndarray, alpha: float, beta: float) -> int:
        """Select next city based on pheromone and distance"""
        probabilities = np.zeros(self.num_cities)
        
        for city in range(self.num_cities):
            if not self.visited[city]:
                pheromone = pheromone_matrix[current_city, city] ** alpha
                distance = (1.0 / (distance_matrix[current_city, city] + 1e-10)) ** beta
                probabilities[city] = pheromone * distance
        
        # Normalize probabilities
        probabilities = probabilities / np.sum(probabilities)
        
        # Select city based on probabilities
        return np.random.choice(self.num_cities, p=probabilities)
    
    def _calculate_tour_length(self, distance_matrix: np.ndarray) -> float:
        """Calculate tour length"""
        total_length = 0.0
        
        for i in range(len(self.tour) - 1):
            total_length += distance_matrix[self.tour[i], self.tour[i + 1]]
        
        # Add distance from last city back to start
        total_length += distance_matrix[self.tour[-1], self.tour[0]]
        
        return total_length

class ParticleSwarmOptimization:
    """Particle Swarm Optimization algorithm"""
    
    def __init__(self, config: SwarmConfig):
        self.config = config
        self.particles = []
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.optimization_history = []
        logger.info("✅ Particle Swarm Optimization initialized")
    
    def initialize_swarm(self, bounds: Tuple[float, float] = (-5.0, 5.0)):
        """Initialize particle swarm"""
        self.particles = []
        
        for i in range(self.config.population_size):
            particle = Particle(self.config.dimension, bounds)
            self.particles.append(particle)
        
        logger.info(f"🧬 Initialized swarm with {len(self.particles)} particles")
    
    def optimize(self, objective_function: Callable, bounds: Tuple[float, float] = (-5.0, 5.0)) -> Dict[str, Any]:
        """Run PSO optimization"""
        logger.info("🔍 Starting PSO optimization")
        
        # Initialize swarm
        self.initialize_swarm(bounds)
        
        # Evaluate initial fitness
        for particle in self.particles:
            particle.evaluate_fitness(objective_function)
            
            if particle.fitness < self.global_best_fitness:
                self.global_best_fitness = particle.fitness
                self.global_best_position = particle.position.copy()
        
        # Optimization loop
        for iteration in range(self.config.max_iterations):
            # Update particles
            for particle in self.particles:
                # Update velocity
                particle.update_velocity(
                    self.global_best_position,
                    self.config.inertia_weight,
                    self.config.cognitive_weight,
                    self.config.social_weight,
                    self.config.velocity_limit
                )
                
                # Update position
                particle.update_position()
                
                # Evaluate fitness
                particle.evaluate_fitness(objective_function)
                
                # Update global best
                if particle.fitness < self.global_best_fitness:
                    self.global_best_fitness = particle.fitness
                    self.global_best_position = particle.position.copy()
            
            # Record history
            self.optimization_history.append({
                'iteration': iteration,
                'best_fitness': self.global_best_fitness,
                'avg_fitness': np.mean([p.fitness for p in self.particles])
            })
            
            # Adaptive parameters
            if self.config.enable_adaptive_parameters:
                self._update_adaptive_parameters(iteration)
        
        optimization_result = {
            'best_fitness': self.global_best_fitness,
            'best_position': self.global_best_position,
            'iterations': self.config.max_iterations,
            'population_size': self.config.population_size,
            'status': 'success'
        }
        
        return optimization_result
    
    def _update_adaptive_parameters(self, iteration: int):
        """Update adaptive parameters"""
        # Linear decrease of inertia weight
        progress = iteration / self.config.max_iterations
        self.config.inertia_weight = 0.9 - 0.5 * progress

class AntColonyOptimization:
    """Ant Colony Optimization algorithm"""
    
    def __init__(self, config: SwarmConfig):
        self.config = config
        self.ants = []
        self.pheromone_matrix = None
        self.distance_matrix = None
        self.best_tour = None
        self.best_tour_length = float('inf')
        self.optimization_history = []
        logger.info("✅ Ant Colony Optimization initialized")
    
    def initialize_ants(self, num_cities: int):
        """Initialize ant colony"""
        self.ants = []
        
        for i in range(self.config.num_ants):
            ant = Ant(num_cities)
            self.ants.append(ant)
        
        logger.info(f"🐜 Initialized colony with {len(self.ants)} ants")
    
    def optimize(self, distance_matrix: np.ndarray) -> Dict[str, Any]:
        """Run ACO optimization"""
        logger.info("🔍 Starting ACO optimization")
        
        num_cities = distance_matrix.shape[0]
        self.distance_matrix = distance_matrix
        
        # Initialize pheromone matrix
        self.pheromone_matrix = np.ones((num_cities, num_cities))
        
        # Initialize ants
        self.initialize_ants(num_cities)
        
        # Optimization loop
        for iteration in range(self.config.max_iterations):
            # Construct tours
            for ant in self.ants:
                start_city = np.random.randint(0, num_cities)
                ant.construct_tour(
                    self.pheromone_matrix,
                    self.distance_matrix,
                    self.config.alpha,
                    self.config.beta,
                    start_city
                )
                
                # Update best tour
                if ant.tour_length < self.best_tour_length:
                    self.best_tour_length = ant.tour_length
                    self.best_tour = ant.tour.copy()
            
            # Update pheromones
            self._update_pheromones()
            
            # Record history
            self.optimization_history.append({
                'iteration': iteration,
                'best_tour_length': self.best_tour_length,
                'avg_tour_length': np.mean([ant.tour_length for ant in self.ants])
            })
        
        optimization_result = {
            'best_tour_length': self.best_tour_length,
            'best_tour': self.best_tour,
            'iterations': self.config.max_iterations,
            'num_ants': self.config.num_ants,
            'status': 'success'
        }
        
        return optimization_result
    
    def _update_pheromones(self):
        """Update pheromone matrix"""
        # Evaporate pheromones
        self.pheromone_matrix *= (1 - self.config.evaporation_rate)
        
        # Add pheromones from ant tours
        for ant in self.ants:
            pheromone_deposit = self.config.pheromone_constant / ant.tour_length
            
            for i in range(len(ant.tour) - 1):
                city1, city2 = ant.tour[i], ant.tour[i + 1]
                self.pheromone_matrix[city1, city2] += pheromone_deposit
                self.pheromone_matrix[city2, city1] += pheromone_deposit
            
            # Add pheromone for return to start
            city1, city2 = ant.tour[-1], ant.tour[0]
            self.pheromone_matrix[city1, city2] += pheromone_deposit
            self.pheromone_matrix[city2, city1] += pheromone_deposit

class SwarmNeuralNetwork:
    """Swarm-based neural network"""
    
    def __init__(self, config: SwarmConfig):
        self.config = config
        self.swarm_layers = []
        self.classical_layers = []
        self.training_history = []
        logger.info("✅ Swarm Neural Network initialized")
    
    def add_swarm_layer(self, input_size: int, output_size: int, swarm_size: int = None):
        """Add swarm-based layer"""
        swarm_size = swarm_size or self.config.population_size
        
        swarm_layer = {
            'input_size': input_size,
            'output_size': output_size,
            'swarm_size': swarm_size,
            'particles': [np.random.normal(0, 0.1, (input_size, output_size)) for _ in range(swarm_size)],
            'velocities': [np.zeros((input_size, output_size)) for _ in range(swarm_size)],
            'best_particles': [np.random.normal(0, 0.1, (input_size, output_size)) for _ in range(swarm_size)],
            'best_fitness': [float('inf')] * swarm_size,
            'layer_id': len(self.swarm_layers)
        }
        
        self.swarm_layers.append(swarm_layer)
        logger.info(f"➕ Added swarm layer: {input_size} -> {output_size} with {swarm_size} particles")
    
    def add_classical_layer(self, input_size: int, output_size: int):
        """Add classical layer"""
        classical_layer = {
            'input_size': input_size,
            'output_size': output_size,
            'weights': np.random.normal(0, 0.1, (input_size, output_size)),
            'bias': np.zeros(output_size),
            'layer_id': len(self.classical_layers)
        }
        
        self.classical_layers.append(classical_layer)
        logger.info(f"➕ Added classical layer: {input_size} -> {output_size}")
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """Forward pass through swarm neural network"""
        current_data = input_data
        
        # Process through swarm layers
        for swarm_layer in self.swarm_layers:
            current_data = self._process_swarm_layer(current_data, swarm_layer)
        
        # Process through classical layers
        for classical_layer in self.classical_layers:
            current_data = self._process_classical_layer(current_data, classical_layer)
        
        return current_data
    
    def _process_swarm_layer(self, input_data: np.ndarray, swarm_layer: Dict[str, Any]) -> np.ndarray:
        """Process swarm layer"""
        # Use best particle for forward pass
        best_particle_idx = np.argmin(swarm_layer['best_fitness'])
        best_particle = swarm_layer['best_particles'][best_particle_idx]
        
        # Linear transformation
        output = np.dot(input_data, best_particle)
        
        # Apply activation function (ReLU)
        output = np.maximum(0, output)
        
        return output
    
    def _process_classical_layer(self, input_data: np.ndarray, classical_layer: Dict[str, Any]) -> np.ndarray:
        """Process classical layer"""
        weights = classical_layer['weights']
        bias = classical_layer['bias']
        
        # Linear transformation
        output = np.dot(input_data, weights) + bias
        
        # Apply activation function (ReLU)
        output = np.maximum(0, output)
        
        return output
    
    def train(self, train_data: np.ndarray, train_labels: np.ndarray, epochs: int = 10):
        """Train swarm neural network"""
        logger.info(f"🏋️ Training swarm neural network for {epochs} epochs")
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for i in range(len(train_data)):
                # Forward pass
                prediction = self.forward(train_data[i])
                
                # Calculate loss
                target = train_labels[i]
                loss = np.mean((prediction - target) ** 2)
                epoch_loss += loss
                
                # Update swarm particles
                self._update_swarm_particles(train_data[i], target, prediction)
            
            avg_loss = epoch_loss / len(train_data)
            
            self.training_history.append({
                'epoch': epoch,
                'loss': avg_loss
            })
            
            logger.info(f"   Epoch {epoch + 1}: Loss = {avg_loss:.4f}")
    
    def _update_swarm_particles(self, input_data: np.ndarray, target: np.ndarray, prediction: np.ndarray):
        """Update swarm particles using PSO"""
        error = target - prediction
        
        for swarm_layer in self.swarm_layers:
            # Find global best particle
            global_best_idx = np.argmin(swarm_layer['best_fitness'])
            global_best_particle = swarm_layer['best_particles'][global_best_idx]
            
            # Update each particle
            for i in range(swarm_layer['swarm_size']):
                particle = swarm_layer['particles'][i]
                velocity = swarm_layer['velocities'][i]
                best_particle = swarm_layer['best_particles'][i]
                
                # Calculate fitness
                fitness = np.mean((np.dot(input_data, particle) - target) ** 2)
                
                # Update best particle
                if fitness < swarm_layer['best_fitness'][i]:
                    swarm_layer['best_fitness'][i] = fitness
                    swarm_layer['best_particles'][i] = particle.copy()
                
                # Update velocity
                r1 = np.random.random(particle.shape)
                r2 = np.random.random(particle.shape)
                
                velocity = (self.config.inertia_weight * velocity +
                          self.config.cognitive_weight * r1 * (best_particle - particle) +
                          self.config.social_weight * r2 * (global_best_particle - particle))
                
                swarm_layer['velocities'][i] = velocity
                
                # Update position
                swarm_layer['particles'][i] = particle + velocity

class SwarmOptimization:
    """Main swarm optimization system"""
    
    def __init__(self, config: SwarmConfig):
        self.config = config
        
        # Components
        self.pso = ParticleSwarmOptimization(config)
        self.aco = AntColonyOptimization(config)
        self.swarm_nn = SwarmNeuralNetwork(config)
        
        # Swarm optimization state
        self.swarm_history = []
        
        logger.info("✅ Swarm Optimization System initialized")
    
    def run_swarm_optimization(self, objective_function: Callable = None, 
                              distance_matrix: np.ndarray = None) -> Dict[str, Any]:
        """Run swarm optimization"""
        logger.info("🚀 Starting swarm optimization")
        
        swarm_results = {
            'start_time': time.time(),
            'config': self.config,
            'stages': {}
        }
        
        # Stage 1: Particle Swarm Optimization
        if self.config.algorithm == SwarmAlgorithm.PARTICLE_SWARM_OPTIMIZATION:
            logger.info("🐝 Stage 1: Particle Swarm Optimization")
            
            if objective_function is None:
                objective_function = lambda x: np.sum(x**2)  # Default sphere function
            
            pso_result = self.pso.optimize(objective_function)
            
            swarm_results['stages']['particle_swarm_optimization'] = pso_result
        
        # Stage 2: Ant Colony Optimization
        elif self.config.algorithm == SwarmAlgorithm.ANT_COLONY_OPTIMIZATION:
            logger.info("🐜 Stage 2: Ant Colony Optimization")
            
            if distance_matrix is None:
                # Create random distance matrix
                num_cities = 10
                distance_matrix = np.random.uniform(1, 10, (num_cities, num_cities))
                distance_matrix = (distance_matrix + distance_matrix.T) / 2
                np.fill_diagonal(distance_matrix, 0)
            
            aco_result = self.aco.optimize(distance_matrix)
            
            swarm_results['stages']['ant_colony_optimization'] = aco_result
        
        # Stage 3: Swarm Neural Network
        logger.info("🧠 Stage 3: Swarm Neural Network")
        
        # Add swarm layers
        self.swarm_nn.add_swarm_layer(4, 8)
        self.swarm_nn.add_classical_layer(8, 2)
        
        # Create dummy training data
        train_data = np.random.randn(100, 4)
        train_labels = np.random.randn(100, 2)
        
        # Train swarm neural network
        self.swarm_nn.train(train_data, train_labels, epochs=5)
        
        swarm_results['stages']['swarm_neural_network'] = {
            'swarm_layers': len(self.swarm_nn.swarm_layers),
            'classical_layers': len(self.swarm_nn.classical_layers),
            'training_epochs': len(self.swarm_nn.training_history),
            'status': 'success'
        }
        
        # Final evaluation
        swarm_results['end_time'] = time.time()
        swarm_results['total_duration'] = swarm_results['end_time'] - swarm_results['start_time']
        
        # Store results
        self.swarm_history.append(swarm_results)
        
        logger.info("✅ Swarm optimization completed")
        return swarm_results
    
    def generate_swarm_report(self, results: Dict[str, Any]) -> str:
        """Generate swarm optimization report"""
        report = []
        report.append("=" * 50)
        report.append("SWARM OPTIMIZATION REPORT")
        report.append("=" * 50)
        
        # Configuration
        report.append("\nSWARM CONFIGURATION:")
        report.append("-" * 22)
        report.append(f"Algorithm: {self.config.algorithm.value}")
        report.append(f"Behavior: {self.config.behavior.value}")
        report.append(f"Population Size: {self.config.population_size}")
        report.append(f"Max Iterations: {self.config.max_iterations}")
        report.append(f"Dimension: {self.config.dimension}")
        report.append(f"Inertia Weight: {self.config.inertia_weight}")
        report.append(f"Cognitive Weight: {self.config.cognitive_weight}")
        report.append(f"Social Weight: {self.config.social_weight}")
        report.append(f"Velocity Limit: {self.config.velocity_limit}")
        report.append(f"Number of Ants: {self.config.num_ants}")
        report.append(f"Evaporation Rate: {self.config.evaporation_rate}")
        report.append(f"Pheromone Constant: {self.config.pheromone_constant}")
        report.append(f"Alpha: {self.config.alpha}")
        report.append(f"Beta: {self.config.beta}")
        report.append(f"Number of Scout Bees: {self.config.num_scout_bees}")
        report.append(f"Number of Employed Bees: {self.config.num_employed_bees}")
        report.append(f"Number of Onlooker Bees: {self.config.num_onlooker_bees}")
        report.append(f"Patch Size: {self.config.patch_size}")
        report.append(f"Absorption Coefficient: {self.config.absorption_coefficient}")
        report.append(f"Attractiveness: {self.config.attractiveness}")
        report.append(f"Randomization Parameter: {self.config.randomization_parameter}")
        report.append(f"Adaptive Parameters: {'Enabled' if self.config.enable_adaptive_parameters else 'Disabled'}")
        report.append(f"Swarm Diversity: {'Enabled' if self.config.enable_swarm_diversity else 'Disabled'}")
        report.append(f"Swarm Memory: {'Enabled' if self.config.enable_swarm_memory else 'Disabled'}")
        report.append(f"Swarm Communication: {'Enabled' if self.config.enable_swarm_communication else 'Disabled'}")
        
        # Results
        report.append("\nSWARM OPTIMIZATION RESULTS:")
        report.append("-" * 30)
        report.append(f"Total Duration: {results.get('total_duration', 0):.2f} seconds")
        report.append(f"Start Time: {results.get('start_time', 'Unknown')}")
        report.append(f"End Time: {results.get('end_time', 'Unknown')}")
        
        # Stage results
        if 'stages' in results:
            for stage_name, stage_data in results['stages'].items():
                report.append(f"\n{stage_name.upper()}:")
                report.append("-" * len(stage_name))
                
                if isinstance(stage_data, dict):
                    for key, value in stage_data.items():
                        report.append(f"  {key}: {value}")
        
        return "\n".join(report)
    
    def visualize_swarm_results(self, save_path: str = None):
        """Visualize swarm optimization results"""
        if not self.swarm_history:
            logger.warning("No swarm optimization history to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: PSO optimization progress
        if self.pso.optimization_history:
            iterations = [h['iteration'] for h in self.pso.optimization_history]
            best_fitness = [h['best_fitness'] for h in self.pso.optimization_history]
            avg_fitness = [h['avg_fitness'] for h in self.pso.optimization_history]
            
            axes[0, 0].plot(iterations, best_fitness, 'b-', linewidth=2, label='Best Fitness')
            axes[0, 0].plot(iterations, avg_fitness, 'r--', linewidth=2, label='Average Fitness')
            axes[0, 0].set_xlabel('Iteration')
            axes[0, 0].set_ylabel('Fitness')
            axes[0, 0].set_title('PSO Optimization Progress')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # Plot 2: ACO optimization progress
        if self.aco.optimization_history:
            iterations = [h['iteration'] for h in self.aco.optimization_history]
            best_tour_length = [h['best_tour_length'] for h in self.aco.optimization_history]
            avg_tour_length = [h['avg_tour_length'] for h in self.aco.optimization_history]
            
            axes[0, 1].plot(iterations, best_tour_length, 'g-', linewidth=2, label='Best Tour Length')
            axes[0, 1].plot(iterations, avg_tour_length, 'orange', linewidth=2, label='Average Tour Length')
            axes[0, 1].set_xlabel('Iteration')
            axes[0, 1].set_ylabel('Tour Length')
            axes[0, 1].set_title('ACO Optimization Progress')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Plot 3: Swarm Neural Network training
        if self.swarm_nn.training_history:
            epochs = [h['epoch'] for h in self.swarm_nn.training_history]
            losses = [h['loss'] for h in self.swarm_nn.training_history]
            
            axes[1, 0].plot(epochs, losses, 'purple', linewidth=2)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].set_title('Swarm Neural Network Training')
            axes[1, 0].grid(True)
        
        # Plot 4: Swarm configuration
        config_values = [
            self.config.population_size,
            self.config.max_iterations,
            self.config.dimension,
            len(self.swarm_nn.swarm_layers)
        ]
        config_labels = ['Population Size', 'Max Iterations', 'Dimension', 'Swarm Layers']
        
        axes[1, 1].bar(config_labels, config_values, color=['blue', 'green', 'orange', 'red'])
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_title('Swarm Configuration')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()

# Factory functions
def create_swarm_config(**kwargs) -> SwarmConfig:
    """Create swarm configuration"""
    return SwarmConfig(**kwargs)

def create_particle(dimension: int, bounds: Tuple[float, float] = (-5.0, 5.0)) -> Particle:
    """Create particle"""
    return Particle(dimension, bounds)

def create_ant(num_cities: int) -> Ant:
    """Create ant"""
    return Ant(num_cities)

def create_particle_swarm_optimization(config: SwarmConfig) -> ParticleSwarmOptimization:
    """Create PSO"""
    return ParticleSwarmOptimization(config)

def create_ant_colony_optimization(config: SwarmConfig) -> AntColonyOptimization:
    """Create ACO"""
    return AntColonyOptimization(config)

def create_swarm_neural_network(config: SwarmConfig) -> SwarmNeuralNetwork:
    """Create swarm neural network"""
    return SwarmNeuralNetwork(config)

def create_swarm_optimization(config: SwarmConfig) -> SwarmOptimization:
    """Create swarm optimization system"""
    return SwarmOptimization(config)

# Example usage
def example_swarm_intelligence():
    """Example of swarm intelligence system"""
    # Create configuration
    config = create_swarm_config(
        algorithm=SwarmAlgorithm.PARTICLE_SWARM_OPTIMIZATION,
        behavior=SwarmBehavior.BALANCED,
        population_size=30,
        max_iterations=50,
        dimension=5,
        inertia_weight=0.9,
        cognitive_weight=2.0,
        social_weight=2.0,
        velocity_limit=1.0,
        num_ants=20,
        evaporation_rate=0.1,
        pheromone_constant=1.0,
        alpha=1.0,
        beta=2.0,
        num_scout_bees=5,
        num_employed_bees=10,
        num_onlooker_bees=10,
        patch_size=0.1,
        absorption_coefficient=1.0,
        attractiveness=1.0,
        randomization_parameter=0.2,
        enable_adaptive_parameters=True,
        enable_swarm_diversity=True,
        enable_swarm_memory=True,
        enable_swarm_communication=True
    )
    
    # Create swarm optimization system
    swarm_optimization = create_swarm_optimization(config)
    
    # Define objective function
    def objective_function(x):
        return np.sum(x**2)  # Sphere function
    
    # Run swarm optimization
    swarm_results = swarm_optimization.run_swarm_optimization(objective_function)
    
    # Generate report
    swarm_report = swarm_optimization.generate_swarm_report(swarm_results)
    
    print(f"✅ Swarm Intelligence Example Complete!")
    print(f"🚀 Swarm Intelligence Statistics:")
    print(f"   Algorithm: {config.algorithm.value}")
    print(f"   Behavior: {config.behavior.value}")
    print(f"   Population Size: {config.population_size}")
    print(f"   Max Iterations: {config.max_iterations}")
    print(f"   Dimension: {config.dimension}")
    print(f"   Inertia Weight: {config.inertia_weight}")
    print(f"   Cognitive Weight: {config.cognitive_weight}")
    print(f"   Social Weight: {config.social_weight}")
    print(f"   Velocity Limit: {config.velocity_limit}")
    print(f"   Number of Ants: {config.num_ants}")
    print(f"   Evaporation Rate: {config.evaporation_rate}")
    print(f"   Pheromone Constant: {config.pheromone_constant}")
    print(f"   Alpha: {config.alpha}")
    print(f"   Beta: {config.beta}")
    print(f"   Number of Scout Bees: {config.num_scout_bees}")
    print(f"   Number of Employed Bees: {config.num_employed_bees}")
    print(f"   Number of Onlooker Bees: {config.num_onlooker_bees}")
    print(f"   Patch Size: {config.patch_size}")
    print(f"   Absorption Coefficient: {config.absorption_coefficient}")
    print(f"   Attractiveness: {config.attractiveness}")
    print(f"   Randomization Parameter: {config.randomization_parameter}")
    print(f"   Adaptive Parameters: {'Enabled' if config.enable_adaptive_parameters else 'Disabled'}")
    print(f"   Swarm Diversity: {'Enabled' if config.enable_swarm_diversity else 'Disabled'}")
    print(f"   Swarm Memory: {'Enabled' if config.enable_swarm_memory else 'Disabled'}")
    print(f"   Swarm Communication: {'Enabled' if config.enable_swarm_communication else 'Disabled'}")
    
    print(f"\n📊 Swarm Optimization Results:")
    print(f"   Swarm History Length: {len(swarm_optimization.swarm_history)}")
    print(f"   Total Duration: {swarm_results.get('total_duration', 0):.2f} seconds")
    
    # Show stage results summary
    if 'stages' in swarm_results:
        for stage_name, stage_data in swarm_results['stages'].items():
            print(f"   {stage_name}: {len(stage_data) if isinstance(stage_data, dict) else 'N/A'} results")
    
    print(f"\n📋 Swarm Optimization Report:")
    print(swarm_report)
    
    return swarm_optimization

# Export utilities
__all__ = [
    'SwarmAlgorithm',
    'SwarmBehavior',
    'SwarmConfig',
    'Particle',
    'Ant',
    'ParticleSwarmOptimization',
    'AntColonyOptimization',
    'SwarmNeuralNetwork',
    'SwarmOptimization',
    'create_swarm_config',
    'create_particle',
    'create_ant',
    'create_particle_swarm_optimization',
    'create_ant_colony_optimization',
    'create_swarm_neural_network',
    'create_swarm_optimization',
    'example_swarm_intelligence'
]

if __name__ == "__main__":
    example_swarm_intelligence()
    print("✅ Swarm intelligence example completed successfully!")