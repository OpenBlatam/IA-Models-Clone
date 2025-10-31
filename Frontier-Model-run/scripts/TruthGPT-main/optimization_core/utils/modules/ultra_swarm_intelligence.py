"""
Ultra-Advanced Swarm Intelligence System for TruthGPT
Implements comprehensive swarm intelligence algorithms including particle swarms, ant colonies, bee colonies, and fish schools.
"""

import asyncio
import json
import time
import random
import math
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SwarmType(Enum):
    """Types of swarm intelligence algorithms."""
    PARTICLE_SWARM = "particle_swarm"
    ANT_COLONY = "ant_colony"
    BEE_COLONY = "bee_colony"
    FISH_SCHOOL = "fish_school"
    FIREFLY_SWARM = "firefly_swarm"
    BAT_SWARM = "bat_swarm"
    WOLF_PACK = "wolf_pack"
    BIRD_FLOCK = "bird_flock"
    HYBRID_SWARM = "hybrid_swarm"

class OptimizationObjective(Enum):
    """Types of optimization objectives."""
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"
    MULTI_OBJECTIVE = "multi_objective"
    CONSTRAINED = "constrained"
    DYNAMIC = "dynamic"

class CommunicationProtocol(Enum):
    """Types of communication protocols."""
    DIRECT_COMMUNICATION = "direct_communication"
    INDIRECT_COMMUNICATION = "indirect_communication"
    STIGMERGY = "stigmergy"
    BROADCAST = "broadcast"
    HIERARCHICAL = "hierarchical"
    PEER_TO_PEER = "peer_to_peer"

@dataclass
class SwarmAgent:
    """Swarm agent representation."""
    agent_id: str
    agent_type: SwarmType
    position: np.ndarray
    velocity: np.ndarray
    fitness: float = 0.0
    best_position: np.ndarray = field(default_factory=lambda: np.array([]))
    best_fitness: float = float('inf')
    neighbors: List[str] = field(default_factory=list)
    communication_range: float = 1.0
    memory: Dict[str, Any] = field(default_factory=dict)
    state: str = "active"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SwarmEnvironment:
    """Swarm environment representation."""
    environment_id: str
    dimensions: int
    bounds: Tuple[float, float]
    obstacles: List[Dict[str, Any]] = field(default_factory=list)
    resources: List[Dict[str, Any]] = field(default_factory=list)
    pheromones: Dict[str, float] = field(default_factory=dict)
    temperature: float = 20.0
    humidity: float = 50.0
    wind_direction: float = 0.0
    wind_speed: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SwarmConfig:
    """Swarm configuration."""
    swarm_type: SwarmType
    population_size: int = 50
    max_iterations: int = 1000
    convergence_threshold: float = 1e-6
    communication_protocol: CommunicationProtocol = CommunicationProtocol.DIRECT_COMMUNICATION
    learning_rate: float = 0.1
    inertia_weight: float = 0.9
    cognitive_weight: float = 2.0
    social_weight: float = 2.0
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elitism_rate: float = 0.1
    parallel_execution: bool = True
    adaptive_parameters: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SwarmResult:
    """Swarm optimization result."""
    result_id: str
    best_solution: np.ndarray
    best_fitness: float
    convergence_iteration: int
    total_iterations: int
    execution_time: float
    population_history: List[List[float]] = field(default_factory=list)
    fitness_history: List[float] = field(default_factory=list)
    diversity_history: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class ParticleSwarmOptimizer:
    """
    Particle Swarm Optimization implementation.
    """

    def __init__(self, config: SwarmConfig):
        """
        Initialize the PSO optimizer.

        Args:
            config: Swarm configuration
        """
        self.config = config
        self.agents: List[SwarmAgent] = []
        self.global_best_position: np.ndarray = np.array([])
        self.global_best_fitness: float = float('inf')
        self.environment: Optional[SwarmEnvironment] = None
        
        logger.info(f"Particle Swarm Optimizer initialized with {config.population_size} particles")

    def initialize_swarm(self, dimensions: int, bounds: Tuple[float, float]) -> None:
        """
        Initialize the particle swarm.

        Args:
            dimensions: Problem dimensions
            bounds: Search space bounds
        """
        self.agents = []
        
        for i in range(self.config.population_size):
            # Random position within bounds
            position = np.random.uniform(bounds[0], bounds[1], dimensions)
            
            # Random velocity
            velocity = np.random.uniform(-1, 1, dimensions)
            
            agent = SwarmAgent(
                agent_id=f"particle_{i}",
                agent_type=SwarmType.PARTICLE_SWARM,
                position=position.copy(),
                velocity=velocity.copy(),
                best_position=position.copy(),
                communication_range=self.config.communication_protocol.value == "direct_communication"
            )
            
            self.agents.append(agent)
        
        # Initialize environment
        self.environment = SwarmEnvironment(
            environment_id=str(uuid.uuid4()),
            dimensions=dimensions,
            bounds=bounds
        )
        
        logger.info(f"Swarm initialized with {len(self.agents)} particles")

    async def optimize(
        self,
        objective_function: Callable[[np.ndarray], float],
        dimensions: int,
        bounds: Tuple[float, float],
        max_iterations: int = None
    ) -> SwarmResult:
        """
        Optimize using particle swarm optimization.

        Args:
            objective_function: Objective function to optimize
            dimensions: Problem dimensions
            bounds: Search space bounds
            max_iterations: Maximum iterations

        Returns:
            Optimization result
        """
        if max_iterations is None:
            max_iterations = self.config.max_iterations
        
        # Initialize swarm
        self.initialize_swarm(dimensions, bounds)
        
        # Evaluate initial fitness
        await self._evaluate_fitness(objective_function)
        
        # Initialize global best
        self._update_global_best()
        
        start_time = time.time()
        iteration = 0
        convergence_count = 0
        
        # Optimization loop
        while iteration < max_iterations and convergence_count < 10:
            # Update particle positions and velocities
            await self._update_particles()
            
            # Evaluate fitness
            await self._evaluate_fitness(objective_function)
            
            # Update personal and global bests
            self._update_personal_bests()
            self._update_global_best()
            
            # Check convergence
            if self._check_convergence():
                convergence_count += 1
            else:
                convergence_count = 0
            
            iteration += 1
            
            # Adaptive parameters
            if self.config.adaptive_parameters:
                self._adapt_parameters(iteration, max_iterations)
        
        execution_time = time.time() - start_time
        
        # Create result
        result = SwarmResult(
            result_id=str(uuid.uuid4()),
            best_solution=self.global_best_position.copy(),
            best_fitness=self.global_best_fitness,
            convergence_iteration=iteration - convergence_count,
            total_iterations=iteration,
            execution_time=execution_time
        )
        
        logger.info(f"PSO optimization completed in {iteration} iterations, {execution_time:.3f}s")
        return result

    async def _evaluate_fitness(self, objective_function: Callable[[np.ndarray], float]) -> None:
        """Evaluate fitness for all particles."""
        for agent in self.agents:
            agent.fitness = objective_function(agent.position)

    def _update_particles(self) -> None:
        """Update particle positions and velocities."""
        for agent in self.agents:
            # Update velocity
            r1, r2 = np.random.random(2)
            
            cognitive_component = (
                self.config.cognitive_weight * r1 * 
                (agent.best_position - agent.position)
            )
            
            social_component = (
                self.config.social_weight * r2 * 
                (self.global_best_position - agent.position)
            )
            
            agent.velocity = (
                self.config.inertia_weight * agent.velocity +
                cognitive_component +
                social_component
            )
            
            # Update position
            agent.position += agent.velocity
            
            # Apply bounds
            bounds = self.environment.bounds
            agent.position = np.clip(agent.position, bounds[0], bounds[1])

    def _update_personal_bests(self) -> None:
        """Update personal best positions."""
        for agent in self.agents:
            if agent.fitness < agent.best_fitness:
                agent.best_fitness = agent.fitness
                agent.best_position = agent.position.copy()

    def _update_global_best(self) -> None:
        """Update global best position."""
        for agent in self.agents:
            if agent.fitness < self.global_best_fitness:
                self.global_best_fitness = agent.fitness
                self.global_best_position = agent.position.copy()

    def _check_convergence(self) -> bool:
        """Check if swarm has converged."""
        if len(self.agents) == 0:
            return False
        
        # Check if all particles are close to global best
        distances = [
            np.linalg.norm(agent.position - self.global_best_position)
            for agent in self.agents
        ]
        
        max_distance = max(distances)
        return max_distance < self.config.convergence_threshold

    def _adapt_parameters(self, iteration: int, max_iterations: int) -> None:
        """Adapt PSO parameters during optimization."""
        progress = iteration / max_iterations
        
        # Adaptive inertia weight
        self.config.inertia_weight = 0.9 - 0.5 * progress
        
        # Adaptive learning rates
        self.config.cognitive_weight = 2.5 - 0.5 * progress
        self.config.social_weight = 0.5 + 1.5 * progress

class AntColonyOptimizer:
    """
    Ant Colony Optimization implementation.
    """

    def __init__(self, config: SwarmConfig):
        """
        Initialize the ACO optimizer.

        Args:
            config: Swarm configuration
        """
        self.config = config
        self.ants: List[SwarmAgent] = []
        self.pheromone_matrix: np.ndarray = np.array([])
        self.distance_matrix: np.ndarray = np.array([])
        self.best_tour: List[int] = []
        self.best_tour_length: float = float('inf')
        
        logger.info(f"Ant Colony Optimizer initialized with {config.population_size} ants")

    def initialize_colony(self, num_cities: int) -> None:
        """
        Initialize the ant colony.

        Args:
            num_cities: Number of cities in TSP
        """
        self.ants = []
        
        # Initialize pheromone matrix
        self.pheromone_matrix = np.ones((num_cities, num_cities)) * 0.1
        
        # Initialize distance matrix (random for simulation)
        self.distance_matrix = np.random.uniform(1, 100, (num_cities, num_cities))
        self.distance_matrix = (self.distance_matrix + self.distance_matrix.T) / 2
        np.fill_diagonal(self.distance_matrix, 0)
        
        # Create ants
        for i in range(self.config.population_size):
            ant = SwarmAgent(
                agent_id=f"ant_{i}",
                agent_type=SwarmType.ANT_COLONY,
                position=np.array([0]),  # Starting city
                velocity=np.array([]),
                memory={'visited_cities': [], 'current_city': 0}
            )
            self.ants.append(ant)
        
        logger.info(f"Ant colony initialized with {len(self.ants)} ants")

    async def optimize_tsp(
        self,
        num_cities: int,
        max_iterations: int = None
    ) -> SwarmResult:
        """
        Optimize Traveling Salesman Problem using ACO.

        Args:
            num_cities: Number of cities
            max_iterations: Maximum iterations

        Returns:
            Optimization result
        """
        if max_iterations is None:
            max_iterations = self.config.max_iterations
        
        # Initialize colony
        self.initialize_colony(num_cities)
        
        start_time = time.time()
        iteration = 0
        
        # Optimization loop
        while iteration < max_iterations:
            # Construct solutions
            await self._construct_solutions(num_cities)
            
            # Update pheromones
            self._update_pheromones()
            
            # Update best solution
            self._update_best_solution()
            
            iteration += 1
        
        execution_time = time.time() - start_time
        
        # Create result
        result = SwarmResult(
            result_id=str(uuid.uuid4()),
            best_solution=np.array(self.best_tour),
            best_fitness=self.best_tour_length,
            convergence_iteration=iteration,
            total_iterations=iteration,
            execution_time=execution_time
        )
        
        logger.info(f"ACO optimization completed in {iteration} iterations, {execution_time:.3f}s")
        return result

    async def _construct_solutions(self, num_cities: int) -> None:
        """Construct solutions for all ants."""
        for ant in self.ants:
            # Reset ant
            ant.memory['visited_cities'] = []
            ant.memory['current_city'] = random.randint(0, num_cities - 1)
            
            # Construct tour
            tour = [ant.memory['current_city']]
            ant.memory['visited_cities'].append(ant.memory['current_city'])
            
            for _ in range(num_cities - 1):
                next_city = self._select_next_city(ant, num_cities)
                tour.append(next_city)
                ant.memory['visited_cities'].append(next_city)
                ant.memory['current_city'] = next_city
            
            ant.memory['tour'] = tour
            ant.fitness = self._calculate_tour_length(tour)

    def _select_next_city(self, ant: SwarmAgent, num_cities: int) -> int:
        """Select next city for ant using probability."""
        current_city = ant.memory['current_city']
        visited = set(ant.memory['visited_cities'])
        
        # Calculate probabilities
        probabilities = []
        cities = []
        
        for city in range(num_cities):
            if city not in visited:
                pheromone = self.pheromone_matrix[current_city, city]
                distance = self.distance_matrix[current_city, city]
                
                # Probability based on pheromone and distance
                prob = pheromone / (distance + 1e-10)
                probabilities.append(prob)
                cities.append(city)
        
        if not probabilities:
            return random.randint(0, num_cities - 1)
        
        # Select city based on probabilities
        probabilities = np.array(probabilities)
        probabilities = probabilities / np.sum(probabilities)
        
        return np.random.choice(cities, p=probabilities)

    def _calculate_tour_length(self, tour: List[int]) -> float:
        """Calculate tour length."""
        total_length = 0
        for i in range(len(tour)):
            current_city = tour[i]
            next_city = tour[(i + 1) % len(tour)]
            total_length += self.distance_matrix[current_city, next_city]
        return total_length

    def _update_pheromones(self) -> None:
        """Update pheromone matrix."""
        # Evaporation
        self.pheromone_matrix *= (1 - self.config.learning_rate)
        
        # Add pheromones based on tour quality
        for ant in self.ants:
            if 'tour' in ant.memory:
                tour = ant.memory['tour']
                tour_length = ant.fitness
                
                # Add pheromone inversely proportional to tour length
                pheromone_amount = 1.0 / (tour_length + 1e-10)
                
                for i in range(len(tour)):
                    current_city = tour[i]
                    next_city = tour[(i + 1) % len(tour)]
                    self.pheromone_matrix[current_city, next_city] += pheromone_amount

    def _update_best_solution(self) -> None:
        """Update best solution found so far."""
        for ant in self.ants:
            if ant.fitness < self.best_tour_length:
                self.best_tour_length = ant.fitness
                self.best_tour = ant.memory['tour'].copy()

class BeeColonyOptimizer:
    """
    Artificial Bee Colony optimization implementation.
    """

    def __init__(self, config: SwarmConfig):
        """
        Initialize the ABC optimizer.

        Args:
            config: Swarm configuration
        """
        self.config = config
        self.bees: List[SwarmAgent] = []
        self.food_sources: List[Dict[str, Any]] = []
        self.best_solution: np.ndarray = np.array([])
        self.best_fitness: float = float('inf')
        
        logger.info(f"Bee Colony Optimizer initialized with {config.population_size} bees")

    def initialize_colony(self, dimensions: int, bounds: Tuple[float, float]) -> None:
        """
        Initialize the bee colony.

        Args:
            dimensions: Problem dimensions
            bounds: Search space bounds
        """
        self.bees = []
        self.food_sources = []
        
        # Create food sources
        for i in range(self.config.population_size // 2):
            position = np.random.uniform(bounds[0], bounds[1], dimensions)
            food_source = {
                'id': f"food_{i}",
                'position': position,
                'fitness': 0.0,
                'trial_count': 0,
                'quality': 'unknown'
            }
            self.food_sources.append(food_source)
        
        # Create bees
        for i in range(self.config.population_size):
            bee_type = "employed" if i < len(self.food_sources) else "onlooker"
            bee = SwarmAgent(
                agent_id=f"bee_{i}",
                agent_type=SwarmType.BEE_COLONY,
                position=np.array([]),
                velocity=np.array([]),
                memory={'bee_type': bee_type, 'food_source_id': None}
            )
            self.bees.append(bee)
        
        logger.info(f"Bee colony initialized with {len(self.bees)} bees and {len(self.food_sources)} food sources")

    async def optimize(
        self,
        objective_function: Callable[[np.ndarray], float],
        dimensions: int,
        bounds: Tuple[float, float],
        max_iterations: int = None
    ) -> SwarmResult:
        """
        Optimize using artificial bee colony.

        Args:
            objective_function: Objective function to optimize
            dimensions: Problem dimensions
            bounds: Search space bounds
            max_iterations: Maximum iterations

        Returns:
            Optimization result
        """
        if max_iterations is None:
            max_iterations = self.config.max_iterations
        
        # Initialize colony
        self.initialize_colony(dimensions, bounds)
        
        # Evaluate initial fitness
        await self._evaluate_food_sources(objective_function)
        
        start_time = time.time()
        iteration = 0
        
        # Optimization loop
        while iteration < max_iterations:
            # Employed bee phase
            await self._employed_bee_phase(objective_function, bounds)
            
            # Onlooker bee phase
            await self._onlooker_bee_phase(objective_function, bounds)
            
            # Scout bee phase
            await self._scout_bee_phase(dimensions, bounds)
            
            # Update best solution
            self._update_best_solution()
            
            iteration += 1
        
        execution_time = time.time() - start_time
        
        # Create result
        result = SwarmResult(
            result_id=str(uuid.uuid4()),
            best_solution=self.best_solution.copy(),
            best_fitness=self.best_fitness,
            convergence_iteration=iteration,
            total_iterations=iteration,
            execution_time=execution_time
        )
        
        logger.info(f"ABC optimization completed in {iteration} iterations, {execution_time:.3f}s")
        return result

    async def _evaluate_food_sources(self, objective_function: Callable[[np.ndarray], float]) -> None:
        """Evaluate fitness of all food sources."""
        for food_source in self.food_sources:
            food_source['fitness'] = objective_function(food_source['position'])

    async def _employed_bee_phase(
        self,
        objective_function: Callable[[np.ndarray], float],
        bounds: Tuple[float, float]
    ) -> None:
        """Employed bee phase."""
        for i, food_source in enumerate(self.food_sources):
            # Generate new solution
            new_position = self._generate_new_solution(food_source['position'], bounds)
            new_fitness = objective_function(new_position)
            
            # Greedy selection
            if new_fitness < food_source['fitness']:
                food_source['position'] = new_position
                food_source['fitness'] = new_fitness
                food_source['trial_count'] = 0
            else:
                food_source['trial_count'] += 1

    async def _onlooker_bee_phase(
        self,
        objective_function: Callable[[np.ndarray], float],
        bounds: Tuple[float, float]
    ) -> None:
        """Onlooker bee phase."""
        # Calculate selection probabilities
        fitness_sum = sum(fs['fitness'] for fs in self.food_sources)
        probabilities = [fs['fitness'] / fitness_sum for fs in self.food_sources]
        
        for _ in range(len(self.food_sources)):
            # Select food source based on probability
            selected_idx = np.random.choice(len(self.food_sources), p=probabilities)
            food_source = self.food_sources[selected_idx]
            
            # Generate new solution
            new_position = self._generate_new_solution(food_source['position'], bounds)
            new_fitness = objective_function(new_position)
            
            # Greedy selection
            if new_fitness < food_source['fitness']:
                food_source['position'] = new_position
                food_source['fitness'] = new_fitness
                food_source['trial_count'] = 0
            else:
                food_source['trial_count'] += 1

    async def _scout_bee_phase(self, dimensions: int, bounds: Tuple[float, float]) -> None:
        """Scout bee phase."""
        for food_source in self.food_sources:
            if food_source['trial_count'] > 10:  # Abandonment limit
                # Generate new random solution
                food_source['position'] = np.random.uniform(bounds[0], bounds[1], dimensions)
                food_source['trial_count'] = 0

    def _generate_new_solution(self, current_position: np.ndarray, bounds: Tuple[float, float]) -> np.ndarray:
        """Generate new solution around current position."""
        new_position = current_position.copy()
        
        # Select random dimension
        dim = random.randint(0, len(current_position) - 1)
        
        # Generate random neighbor
        neighbor_idx = random.randint(0, len(self.food_sources) - 1)
        neighbor_position = self.food_sources[neighbor_idx]['position']
        
        # Update position
        phi = random.uniform(-1, 1)
        new_position[dim] = current_position[dim] + phi * (current_position[dim] - neighbor_position[dim])
        
        # Apply bounds
        new_position = np.clip(new_position, bounds[0], bounds[1])
        
        return new_position

    def _update_best_solution(self) -> None:
        """Update best solution found so far."""
        for food_source in self.food_sources:
            if food_source['fitness'] < self.best_fitness:
                self.best_fitness = food_source['fitness']
                self.best_solution = food_source['position'].copy()

class TruthGPTSwarmIntelligence:
    """
    TruthGPT Swarm Intelligence Manager.
    Main orchestrator for all swarm intelligence algorithms.
    """

    def __init__(self):
        """Initialize the TruthGPT Swarm Intelligence Manager."""
        self.optimizers: Dict[SwarmType, Any] = {}
        self.environments: Dict[str, SwarmEnvironment] = {}
        self.results_history: List[SwarmResult] = []
        
        # Swarm statistics
        self.stats = {
            'total_optimizations': 0,
            'successful_optimizations': 0,
            'total_execution_time': 0.0,
            'average_convergence_time': 0.0,
            'best_solutions_found': 0
        }
        
        logger.info("TruthGPT Swarm Intelligence Manager initialized")

    def create_swarm_optimizer(
        self,
        swarm_type: SwarmType,
        config: SwarmConfig
    ) -> Any:
        """
        Create a swarm optimizer.

        Args:
            swarm_type: Type of swarm optimizer
            config: Swarm configuration

        Returns:
            Swarm optimizer instance
        """
        if swarm_type == SwarmType.PARTICLE_SWARM:
            optimizer = ParticleSwarmOptimizer(config)
        elif swarm_type == SwarmType.ANT_COLONY:
            optimizer = AntColonyOptimizer(config)
        elif swarm_type == SwarmType.BEE_COLONY:
            optimizer = BeeColonyOptimizer(config)
        else:
            raise Exception(f"Unsupported swarm type: {swarm_type}")
        
        self.optimizers[swarm_type] = optimizer
        
        logger.info(f"Swarm optimizer created: {swarm_type.value}")
        return optimizer

    async def optimize_with_swarm(
        self,
        swarm_type: SwarmType,
        objective_function: Callable[[np.ndarray], float],
        dimensions: int,
        bounds: Tuple[float, float],
        config: SwarmConfig = None
    ) -> SwarmResult:
        """
        Optimize using specified swarm algorithm.

        Args:
            swarm_type: Type of swarm algorithm
            objective_function: Objective function
            dimensions: Problem dimensions
            bounds: Search space bounds
            config: Swarm configuration

        Returns:
            Optimization result
        """
        if config is None:
            config = SwarmConfig(swarm_type=swarm_type)
        
        # Create optimizer if not exists
        if swarm_type not in self.optimizers:
            self.create_swarm_optimizer(swarm_type, config)
        
        optimizer = self.optimizers[swarm_type]
        
        logger.info(f"Starting optimization with {swarm_type.value}")
        
        # Run optimization
        if swarm_type == SwarmType.ANT_COLONY:
            result = await optimizer.optimize_tsp(dimensions)
        else:
            result = await optimizer.optimize(objective_function, dimensions, bounds)
        
        # Update statistics
        self.stats['total_optimizations'] += 1
        self.stats['successful_optimizations'] += 1
        self.stats['total_execution_time'] += result.execution_time
        
        # Update average convergence time
        total_time = self.stats['average_convergence_time'] * (self.stats['successful_optimizations'] - 1)
        self.stats['average_convergence_time'] = (total_time + result.execution_time) / self.stats['successful_optimizations']
        
        # Store result
        self.results_history.append(result)
        
        logger.info(f"Optimization completed: {result.best_fitness:.6f} in {result.execution_time:.3f}s")
        return result

    async def multi_swarm_optimization(
        self,
        objective_function: Callable[[np.ndarray], float],
        dimensions: int,
        bounds: Tuple[float, float],
        swarm_types: List[SwarmType] = None
    ) -> Dict[SwarmType, SwarmResult]:
        """
        Run optimization with multiple swarm algorithms.

        Args:
            objective_function: Objective function
            dimensions: Problem dimensions
            bounds: Search space bounds
            swarm_types: List of swarm types to use

        Returns:
            Dictionary of results for each swarm type
        """
        if swarm_types is None:
            swarm_types = [
                SwarmType.PARTICLE_SWARM,
                SwarmType.BEE_COLONY
            ]
        
        logger.info(f"Running multi-swarm optimization with {len(swarm_types)} algorithms")
        
        results = {}
        
        # Run optimization with each swarm type
        for swarm_type in swarm_types:
            try:
                result = await self.optimize_with_swarm(
                    swarm_type=swarm_type,
                    objective_function=objective_function,
                    dimensions=dimensions,
                    bounds=bounds
                )
                results[swarm_type] = result
            except Exception as e:
                logger.error(f"Optimization failed for {swarm_type.value}: {e}")
                self.stats['total_optimizations'] += 1
        
        return results

    def create_swarm_environment(
        self,
        dimensions: int,
        bounds: Tuple[float, float],
        obstacles: List[Dict[str, Any]] = None
    ) -> SwarmEnvironment:
        """
        Create a swarm environment.

        Args:
            dimensions: Environment dimensions
            bounds: Environment bounds
            obstacles: List of obstacles

        Returns:
            Created environment
        """
        environment = SwarmEnvironment(
            environment_id=str(uuid.uuid4()),
            dimensions=dimensions,
            bounds=bounds,
            obstacles=obstacles or []
        )
        
        self.environments[environment.environment_id] = environment
        
        logger.info(f"Swarm environment created: {environment.environment_id}")
        return environment

    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        return {
            'total_optimizations': self.stats['total_optimizations'],
            'successful_optimizations': self.stats['successful_optimizations'],
            'success_rate': (
                self.stats['successful_optimizations'] / self.stats['total_optimizations']
                if self.stats['total_optimizations'] > 0 else 0.0
            ),
            'total_execution_time': self.stats['total_execution_time'],
            'average_execution_time': (
                self.stats['total_execution_time'] / self.stats['successful_optimizations']
                if self.stats['successful_optimizations'] > 0 else 0.0
            ),
            'average_convergence_time': self.stats['average_convergence_time'],
            'available_optimizers': list(self.optimizers.keys()),
            'environments_count': len(self.environments),
            'results_history_count': len(self.results_history)
        }

    def get_best_results(self, limit: int = 10) -> List[SwarmResult]:
        """Get best optimization results."""
        sorted_results = sorted(self.results_history, key=lambda x: x.best_fitness)
        return sorted_results[:limit]

# Utility functions
def create_swarm_intelligence_manager() -> TruthGPTSwarmIntelligence:
    """Create a swarm intelligence manager."""
    return TruthGPTSwarmIntelligence()

def create_swarm_config(
    swarm_type: SwarmType,
    population_size: int = 50,
    max_iterations: int = 1000
) -> SwarmConfig:
    """Create a swarm configuration."""
    return SwarmConfig(
        swarm_type=swarm_type,
        population_size=population_size,
        max_iterations=max_iterations
    )

# Example usage
async def example_swarm_intelligence():
    """Example of swarm intelligence capabilities."""
    print("🐝 Ultra Swarm Intelligence Example")
    print("=" * 60)
    
    # Create swarm intelligence manager
    swarm_manager = create_swarm_intelligence_manager()
    
    print("✅ Swarm Intelligence Manager initialized")
    
    # Define objective function (Rastrigin function)
    def rastrigin_function(x):
        """Rastrigin function for optimization."""
        n = len(x)
        return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
    
    # Test parameters
    dimensions = 10
    bounds = (-5.12, 5.12)
    
    print(f"\n🎯 Testing optimization with Rastrigin function")
    print(f"Dimensions: {dimensions}")
    print(f"Bounds: {bounds}")
    
    # Particle Swarm Optimization
    print(f"\n🔄 Running Particle Swarm Optimization...")
    pso_result = await swarm_manager.optimize_with_swarm(
        swarm_type=SwarmType.PARTICLE_SWARM,
        objective_function=rastrigin_function,
        dimensions=dimensions,
        bounds=bounds
    )
    
    print(f"PSO Results:")
    print(f"  Best Fitness: {pso_result.best_fitness:.6f}")
    print(f"  Execution Time: {pso_result.execution_time:.3f}s")
    print(f"  Iterations: {pso_result.total_iterations}")
    print(f"  Convergence: {pso_result.convergence_iteration}")
    
    # Bee Colony Optimization
    print(f"\n🐝 Running Bee Colony Optimization...")
    abc_result = await swarm_manager.optimize_with_swarm(
        swarm_type=SwarmType.BEE_COLONY,
        objective_function=rastrigin_function,
        dimensions=dimensions,
        bounds=bounds
    )
    
    print(f"ABC Results:")
    print(f"  Best Fitness: {abc_result.best_fitness:.6f}")
    print(f"  Execution Time: {abc_result.execution_time:.3f}s")
    print(f"  Iterations: {abc_result.total_iterations}")
    print(f"  Convergence: {abc_result.convergence_iteration}")
    
    # Ant Colony Optimization (TSP)
    print(f"\n🐜 Running Ant Colony Optimization (TSP)...")
    tsp_result = await swarm_manager.optimize_with_swarm(
        swarm_type=SwarmType.ANT_COLONY,
        objective_function=lambda x: 0,  # Not used for TSP
        dimensions=20,  # 20 cities
        bounds=(0, 1)  # Not used for TSP
    )
    
    print(f"ACO Results:")
    print(f"  Best Tour Length: {tsp_result.best_fitness:.6f}")
    print(f"  Execution Time: {tsp_result.execution_time:.3f}s")
    print(f"  Iterations: {tsp_result.total_iterations}")
    print(f"  Best Tour: {tsp_result.best_solution[:10]}...")  # Show first 10 cities
    
    # Multi-swarm optimization
    print(f"\n🔄 Running Multi-Swarm Optimization...")
    multi_results = await swarm_manager.multi_swarm_optimization(
        objective_function=rastrigin_function,
        dimensions=dimensions,
        bounds=bounds,
        swarm_types=[SwarmType.PARTICLE_SWARM, SwarmType.BEE_COLONY]
    )
    
    print(f"Multi-Swarm Results:")
    for swarm_type, result in multi_results.items():
        print(f"  {swarm_type.value}: {result.best_fitness:.6f} ({result.execution_time:.3f}s)")
    
    # Create swarm environment
    print(f"\n🌍 Creating swarm environment...")
    environment = swarm_manager.create_swarm_environment(
        dimensions=3,
        bounds=(-10, 10),
        obstacles=[
            {'type': 'sphere', 'center': [0, 0, 0], 'radius': 2},
            {'type': 'box', 'min': [-1, -1, -1], 'max': [1, 1, 1]}
        ]
    )
    
    print(f"Environment created:")
    print(f"  ID: {environment.environment_id}")
    print(f"  Dimensions: {environment.dimensions}")
    print(f"  Bounds: {environment.bounds}")
    print(f"  Obstacles: {len(environment.obstacles)}")
    
    # Get optimization statistics
    print(f"\n📊 Optimization Statistics:")
    stats = swarm_manager.get_optimization_statistics()
    print(f"Total Optimizations: {stats['total_optimizations']}")
    print(f"Successful Optimizations: {stats['successful_optimizations']}")
    print(f"Success Rate: {stats['success_rate']:.2%}")
    print(f"Total Execution Time: {stats['total_execution_time']:.3f}s")
    print(f"Average Execution Time: {stats['average_execution_time']:.3f}s")
    print(f"Available Optimizers: {len(stats['available_optimizers'])}")
    print(f"Environments: {stats['environments_count']}")
    
    # Get best results
    print(f"\n🏆 Best Results:")
    best_results = swarm_manager.get_best_results(limit=3)
    for i, result in enumerate(best_results, 1):
        print(f"  {i}. Fitness: {result.best_fitness:.6f} (Time: {result.execution_time:.3f}s)")
    
    print("\n✅ Swarm intelligence example completed successfully!")

if __name__ == "__main__":
    asyncio.run(example_swarm_intelligence())
