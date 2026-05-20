"""
Optimization Module for Blaze AI

Provides various optimization strategies and algorithms as a modular component
that can be used independently or as part of the system.
"""

import asyncio
import logging
import time
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import json

from .base import BaseModule, ModuleConfig, ModuleType, ModulePriority, ModuleStatus, HealthStatus

logger = logging.getLogger(__name__)

# ============================================================================
# OPTIMIZATION-SPECIFIC ENUMS AND CONSTANTS
# ============================================================================

class OptimizationType(Enum):
    """Types of optimization algorithms."""
    GENETIC = auto()
    SIMULATED_ANNEALING = auto()
    PARTICLE_SWARM = auto()
    NEURAL_NETWORK = auto()
    QUANTUM = auto()
    HYBRID = auto()
    CUSTOM = auto()

class OptimizationStatus(Enum):
    """Status of optimization process."""
    IDLE = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    FAILED = auto()

class FitnessFunction(Enum):
    """Types of fitness functions."""
    MINIMIZE = auto()
    MAXIMIZE = auto()
    TARGET = auto()

# Default constants
DEFAULT_MAX_ITERATIONS = 1000
DEFAULT_POPULATION_SIZE = 100
DEFAULT_MUTATION_RATE = 0.1
DEFAULT_CROSSOVER_RATE = 0.8

# ============================================================================
# OPTIMIZATION-SPECIFIC DATACLASSES
# ============================================================================

@dataclass
class OptimizationConfig(ModuleConfig):
    """Configuration for optimization modules."""
    optimization_type: OptimizationType = OptimizationType.GENETIC
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    population_size: int = DEFAULT_POPULATION_SIZE
    mutation_rate: float = DEFAULT_MUTATION_RATE
    crossover_rate: float = DEFAULT_CROSSOVER_RATE
    fitness_function: FitnessFunction = FitnessFunction.MINIMIZE
    target_value: Optional[float] = None
    tolerance: float = 1e-6
    enable_parallel: bool = True
    max_workers: int = 4
    custom_parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set module type automatically."""
        self.module_type = ModuleType.OPTIMIZATION

@dataclass
class OptimizationResult:
    """Result of an optimization process."""
    success: bool
    best_solution: Any
    best_fitness: float
    iterations: int
    execution_time: float
    convergence_history: List[float]
    parameters: Dict[str, Any]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "best_solution": str(self.best_solution),
            "best_fitness": self.best_fitness,
            "iterations": self.iterations,
            "execution_time": self.execution_time,
            "convergence_history": self.convergence_history,
            "parameters": self.parameters,
            "metadata": self.metadata
        }

@dataclass
class OptimizationTask:
    """Definition of an optimization task."""
    id: str
    name: str
    objective_function: Callable
    constraints: List[Callable]
    bounds: Dict[str, Tuple[float, float]]
    initial_population: Optional[List[Any]] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 5
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "constraints": len(self.constraints),
            "bounds": self.bounds,
            "parameters": self.parameters,
            "priority": self.priority,
            "created_at": self.created_at
        }

# ============================================================================
# OPTIMIZATION ALGORITHMS
# ============================================================================

class OptimizationAlgorithm(ABC):
    """Abstract base class for optimization algorithms."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.status = OptimizationStatus.IDLE
        self.current_iteration = 0
        self.best_solution = None
        self.best_fitness = float('inf')
        self.convergence_history = []
    
    @abstractmethod
    async def optimize(self, task: OptimizationTask) -> OptimizationResult:
        """Execute optimization algorithm."""
        pass
    
    def update_status(self, status: OptimizationStatus):
        """Update optimization status."""
        self.status = status
    
    def record_iteration(self, fitness: float):
        """Record fitness value for current iteration."""
        self.convergence_history.append(fitness)
        if self.config.fitness_function == FitnessFunction.MINIMIZE:
            if fitness < self.best_fitness:
                self.best_fitness = fitness
        else:
            if fitness > self.best_fitness:
                self.best_fitness = fitness

class GeneticAlgorithm(OptimizationAlgorithm):
    """Genetic Algorithm implementation."""
    
    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        self.population = []
        self.fitness_scores = []
    
    async def optimize(self, task: OptimizationTask) -> OptimizationResult:
        """Execute genetic algorithm optimization."""
        start_time = time.time()
        self.update_status(OptimizationStatus.RUNNING)
        
        try:
            # Initialize population
            await self._initialize_population(task)
            
            # Main optimization loop
            for iteration in range(self.config.max_iterations):
                self.current_iteration = iteration
                
                # Evaluate fitness
                await self._evaluate_fitness(task)
                
                # Check convergence
                if self._check_convergence():
                    break
                
                # Selection
                parents = await self._selection()
                
                # Crossover
                offspring = await self._crossover(parents)
                
                # Mutation
                await self._mutation(offspring)
                
                # Update population
                self.population = offspring
                
                # Record best fitness
                best_fitness = min(self.fitness_scores) if self.fitness_scores else float('inf')
                self.record_iteration(best_fitness)
                
                # Check target
                if self.config.target_value and abs(best_fitness - self.config.target_value) < self.config.tolerance:
                    break
            
            self.update_status(OptimizationStatus.COMPLETED)
            
            return OptimizationResult(
                success=True,
                best_solution=self.best_solution,
                best_fitness=self.best_fitness,
                iterations=self.current_iteration + 1,
                execution_time=time.time() - start_time,
                convergence_history=self.convergence_history,
                parameters=self.config.custom_parameters,
                metadata={"algorithm": "genetic"}
            )
            
        except Exception as e:
            self.update_status(OptimizationStatus.FAILED)
            logger.error(f"Genetic algorithm failed: {e}")
            raise
    
    async def _initialize_population(self, task: OptimizationTask):
        """Initialize the population."""
        if task.initial_population:
            self.population = task.initial_population.copy()
        else:
            # Generate random population
            self.population = []
            for _ in range(self.config.population_size):
                individual = self._generate_random_individual(task.bounds)
                self.population.append(individual)
    
    async def _evaluate_fitness(self, task: OptimizationTask):
        """Evaluate fitness of all individuals."""
        self.fitness_scores = []
        
        for individual in self.population:
            try:
                fitness = await self._evaluate_individual(individual, task)
                self.fitness_scores.append(fitness)
            except Exception as e:
                logger.error(f"Error evaluating individual: {e}")
                self.fitness_scores.append(float('inf'))
    
    async def _evaluate_individual(self, individual: Any, task: OptimizationTask) -> float:
        """Evaluate fitness of a single individual."""
        try:
            if asyncio.iscoroutinefunction(task.objective_function):
                fitness = await task.objective_function(individual)
            else:
                fitness = task.objective_function(individual)
            
            # Apply constraints
            for constraint in task.constraints:
                if asyncio.iscoroutinefunction(constraint):
                    constraint_value = await constraint(individual)
                else:
                    constraint_value = constraint(individual)
                
                if constraint_value > 0:  # Constraint violated
                    fitness += 1000  # Penalty
            
            return fitness
            
        except Exception as e:
            logger.error(f"Error in fitness evaluation: {e}")
            return float('inf')
    
    def _generate_random_individual(self, bounds: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """Generate a random individual within bounds."""
        individual = {}
        for param, (min_val, max_val) in bounds.items():
            individual[param] = random.uniform(min_val, max_val)
        return individual
    
    def _check_convergence(self) -> bool:
        """Check if optimization has converged."""
        if len(self.convergence_history) < 10:
            return False
        
        recent_values = self.convergence_history[-10:]
        improvement = abs(recent_values[-1] - recent_values[0])
        return improvement < self.config.tolerance
    
    async def _selection(self) -> List[Any]:
        """Select parents using tournament selection."""
        parents = []
        tournament_size = 3
        
        for _ in range(len(self.population)):
            # Tournament selection
            tournament_indices = random.sample(range(len(self.population)), tournament_size)
            tournament_fitness = [self.fitness_scores[i] for i in tournament_indices]
            
            if self.config.fitness_function == FitnessFunction.MINIMIZE:
                winner_index = tournament_indices[tournament_fitness.index(min(tournament_fitness))]
            else:
                winner_index = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
            
            parents.append(self.population[winner_index])
        
        return parents
    
    async def _crossover(self, parents: List[Any]) -> List[Any]:
        """Perform crossover operation."""
        offspring = []
        
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                if random.random() < self.config.crossover_rate:
                    child1, child2 = self._crossover_individuals(parents[i], parents[i + 1])
                    offspring.extend([child1, child2])
                else:
                    offspring.extend([parents[i], parents[i + 1]])
            else:
                offspring.append(parents[i])
        
        return offspring
    
    def _crossover_individuals(self, parent1: Dict[str, float], parent2: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Perform crossover between two individuals."""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Single point crossover
        params = list(parent1.keys())
        crossover_point = random.randint(1, len(params) - 1)
        
        for i in range(crossover_point):
            param = params[i]
            child1[param], child2[param] = child2[param], child1[param]
        
        return child1, child2
    
    async def _mutation(self, offspring: List[Any]):
        """Perform mutation operation."""
        for individual in offspring:
            if random.random() < self.config.mutation_rate:
                self._mutate_individual(individual)
    
    def _mutate_individual(self, individual: Dict[str, float]):
        """Mutate a single individual."""
        param = random.choice(list(individual.keys()))
        # Simple random mutation
        individual[param] = random.gauss(individual[param], 0.1)

class SimulatedAnnealing(OptimizationAlgorithm):
    """Simulated Annealing implementation."""
    
    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        self.temperature = 100.0
        self.cooling_rate = 0.95
    
    async def optimize(self, task: OptimizationTask) -> OptimizationResult:
        """Execute simulated annealing optimization."""
        start_time = time.time()
        self.update_status(OptimizationStatus.RUNNING)
        
        try:
            # Initialize solution
            current_solution = self._generate_random_individual(task.bounds)
            current_fitness = await self._evaluate_individual(current_solution, task)
            
            self.best_solution = current_solution
            self.best_fitness = current_fitness
            
            # Main optimization loop
            for iteration in range(self.config.max_iterations):
                self.current_iteration = iteration
                
                # Generate neighbor
                neighbor = self._generate_neighbor(current_solution, task.bounds)
                neighbor_fitness = await self._evaluate_individual(neighbor, task)
                
                # Accept or reject neighbor
                if self._accept_neighbor(current_fitness, neighbor_fitness):
                    current_solution = neighbor
                    current_fitness = neighbor_fitness
                    
                    # Update best solution
                    if self.config.fitness_function == FitnessFunction.MINIMIZE:
                        if neighbor_fitness < self.best_fitness:
                            self.best_solution = neighbor
                            self.best_fitness = neighbor_fitness
                    else:
                        if neighbor_fitness > self.best_fitness:
                            self.best_solution = neighbor
                            self.best_fitness = neighbor_fitness
                
                # Record fitness
                self.record_iteration(current_fitness)
                
                # Cool down
                self.temperature *= self.cooling_rate
                
                # Check convergence
                if self.temperature < 0.01:
                    break
            
            self.update_status(OptimizationStatus.COMPLETED)
            
            return OptimizationResult(
                success=True,
                best_solution=self.best_solution,
                best_fitness=self.best_fitness,
                iterations=self.current_iteration + 1,
                execution_time=time.time() - start_time,
                convergence_history=self.convergence_history,
                parameters=self.config.custom_parameters,
                metadata={"algorithm": "simulated_annealing"}
            )
            
        except Exception as e:
            self.update_status(OptimizationStatus.FAILED)
            logger.error(f"Simulated annealing failed: {e}")
            raise
    
    def _generate_random_individual(self, bounds: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """Generate a random individual within bounds."""
        individual = {}
        for param, (min_val, max_val) in bounds.items():
            individual[param] = random.uniform(min_val, max_val)
        return individual
    
    def _generate_neighbor(self, solution: Dict[str, float], bounds: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """Generate a neighbor solution."""
        neighbor = solution.copy()
        param = random.choice(list(solution.keys()))
        
        # Gaussian perturbation
        min_val, max_val = bounds[param]
        perturbation = random.gauss(0, (max_val - min_val) * 0.1)
        neighbor[param] = max(min_val, min(max_val, solution[param] + perturbation))
        
        return neighbor
    
    def _accept_neighbor(self, current_fitness: float, neighbor_fitness: float) -> bool:
        """Determine if neighbor should be accepted."""
        if self.config.fitness_function == FitnessFunction.MINIMIZE:
            if neighbor_fitness < current_fitness:
                return True
            else:
                # Accept with probability based on temperature
                delta = neighbor_fitness - current_fitness
                probability = random.exp(-delta / self.temperature)
                return random.random() < probability
        else:
            if neighbor_fitness > current_fitness:
                return True
            else:
                # Accept with probability based on temperature
                delta = current_fitness - neighbor_fitness
                probability = random.exp(-delta / self.temperature)
                return random.random() < probability
    
    async def _evaluate_individual(self, individual: Any, task: OptimizationTask) -> float:
        """Evaluate fitness of a single individual."""
        try:
            if asyncio.iscoroutinefunction(task.objective_function):
                fitness = await task.objective_function(individual)
            else:
                fitness = task.objective_function(individual)
            
            # Apply constraints
            for constraint in task.constraints:
                if asyncio.iscoroutinefunction(constraint):
                    constraint_value = await constraint(individual)
                else:
                    constraint_value = constraint(individual)
                
                if constraint_value > 0:  # Constraint violated
                    fitness += 1000  # Penalty
            
            return fitness
            
        except Exception as e:
            logger.error(f"Error in fitness evaluation: {e}")
            return float('inf')

# ============================================================================
# MAIN OPTIMIZATION MODULE
# ============================================================================

class OptimizationModule(BaseModule):
    """Modular optimization system for Blaze AI."""
    
    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        self.optimization_config = config
        
        # Available algorithms
        self.algorithms: Dict[OptimizationType, OptimizationAlgorithm] = {}
        self.active_tasks: Dict[str, OptimizationTask] = {}
        self.completed_tasks: Dict[str, OptimizationResult] = {}
        
        # Task queue
        self.task_queue: List[OptimizationTask] = []
        self.processing_task: Optional[asyncio.Task] = None
        
        # Initialize algorithms
        self._initialize_algorithms()
    
    def _initialize_algorithms(self):
        """Initialize available optimization algorithms."""
        # Genetic Algorithm
        genetic_config = OptimizationConfig(
            name="genetic_algorithm",
            optimization_type=OptimizationType.GENETIC,
            max_iterations=1000,
            population_size=100,
            mutation_rate=0.1,
            crossover_rate=0.8
        )
        self.algorithms[OptimizationType.GENETIC] = GeneticAlgorithm(genetic_config)
        
        # Simulated Annealing
        sa_config = OptimizationConfig(
            name="simulated_annealing",
            optimization_type=OptimizationType.SIMULATED_ANNEALING,
            max_iterations=1000
        )
        self.algorithms[OptimizationType.SIMULATED_ANNEALING] = SimulatedAnnealing(sa_config)
    
    # ============================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS
    # ============================================================================
    
    async def _initialize_impl(self) -> bool:
        """Initialize the optimization module."""
        try:
            self.logger.info(f"Initializing optimization module: {self.config.name}")
            self.logger.info(f"Available algorithms: {[alg.name for alg in self.algorithms.values()]}")
            
            # Start task processing
            self._start_task_processing()
            
            return True
        except Exception as e:
            self.logger.error(f"Optimization initialization failed: {e}")
            return False
    
    async def _shutdown_impl(self) -> bool:
        """Shutdown the optimization module."""
        try:
            # Stop task processing
            if self.processing_task:
                self.processing_task.cancel()
            
            # Cancel active tasks
            for task_id in list(self.active_tasks.keys()):
                await self.cancel_task(task_id)
            
            self.logger.info(f"Optimization module shutdown completed: {self.config.name}")
            return True
        except Exception as e:
            self.logger.error(f"Optimization shutdown failed: {e}")
            return False
    
    async def _health_check_impl(self) -> HealthStatus:
        """Perform optimization-specific health check."""
        try:
            active_tasks = len(self.active_tasks)
            completed_tasks = len(self.completed_tasks)
            queued_tasks = len(self.task_queue)
            
            if active_tasks == 0 and queued_tasks == 0:
                message = "No active optimization tasks"
                status = ModuleStatus.IDLE
            else:
                message = f"Active: {active_tasks}, Queued: {queued_tasks}, Completed: {completed_tasks}"
                status = ModuleStatus.ACTIVE
            
            return HealthStatus(
                status=status,
                message=message,
                details={
                    "active_tasks": active_tasks,
                    "queued_tasks": queued_tasks,
                    "completed_tasks": completed_tasks,
                    "available_algorithms": len(self.algorithms)
                }
            )
            
        except Exception as e:
            return HealthStatus(
                status=ModuleStatus.ERROR,
                message=f"Health check failed: {str(e)}",
                error=str(e)
            )
    
    # ============================================================================
    # TASK PROCESSING
    # ============================================================================
    
    def _start_task_processing(self):
        """Start background task processing."""
        async def process_tasks():
            while self.status not in [ModuleStatus.SHUTDOWN, ModuleStatus.ERROR]:
                try:
                    if self.task_queue:
                        task = self.task_queue.pop(0)
                        await self._process_task(task)
                    else:
                        await asyncio.sleep(1.0)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Task processing error: {e}")
                    await asyncio.sleep(5.0)
        
        self.processing_task = asyncio.create_task(process_tasks())
    
    async def _process_task(self, task: OptimizationTask):
        """Process a single optimization task."""
        try:
            self.active_tasks[task.id] = task
            
            # Select algorithm
            algorithm = self._select_algorithm(task)
            
            # Execute optimization
            result = await algorithm.optimize(task)
            
            # Store result
            self.completed_tasks[task.id] = result
            del self.active_tasks[task.id]
            
            # Record operation
            self.record_operation(True)
            
            self.logger.info(f"Task {task.name} completed successfully")
            
        except Exception as e:
            self.logger.error(f"Task {task.name} failed: {e}")
            del self.active_tasks[task.id]
            self.record_operation(False)
    
    def _select_algorithm(self, task: OptimizationTask) -> OptimizationAlgorithm:
        """Select appropriate algorithm for the task."""
        # For now, use genetic algorithm as default
        # In a more sophisticated implementation, this could use ML to select the best algorithm
        return self.algorithms[OptimizationType.GENETIC]
    
    # ============================================================================
    # PUBLIC INTERFACE
    # ============================================================================
    
    async def submit_task(
        self,
        name: str,
        objective_function: Callable,
        constraints: Optional[List[Callable]] = None,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        algorithm_type: Optional[OptimizationType] = None,
        **parameters
    ) -> str:
        """Submit an optimization task."""
        task_id = f"task_{int(time.time())}_{random.randint(1000, 9999)}"
        
        task = OptimizationTask(
            id=task_id,
            name=name,
            objective_function=objective_function,
            constraints=constraints or [],
            bounds=bounds or {},
            parameters=parameters
        )
        
        # Add to queue
        self.task_queue.append(task)
        
        self.logger.info(f"Submitted optimization task: {name} (ID: {task_id})")
        return task_id
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task."""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            return {
                "id": task_id,
                "name": task.name,
                "status": "active",
                "created_at": task.created_at,
                "priority": task.priority
            }
        elif task_id in self.completed_tasks:
            result = self.completed_tasks[task_id]
            return {
                "id": task_id,
                "status": "completed",
                "success": result.success,
                "best_fitness": result.best_fitness,
                "iterations": result.iterations,
                "execution_time": result.execution_time
            }
        else:
            return None
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel an active task."""
        if task_id in self.active_tasks:
            del self.active_tasks[task_id]
            self.logger.info(f"Cancelled task: {task_id}")
            return True
        return False
    
    def get_available_algorithms(self) -> List[Dict[str, Any]]:
        """Get list of available optimization algorithms."""
        algorithms = []
        for alg_type, algorithm in self.algorithms.items():
            algorithms.append({
                "type": alg_type.name,
                "name": algorithm.config.name,
                "status": algorithm.status.name,
                "current_iteration": algorithm.current_iteration,
                "best_fitness": algorithm.best_fitness
            })
        return algorithms
    
    def get_task_summary(self) -> Dict[str, Any]:
        """Get summary of all tasks."""
        return {
            "active_tasks": len(self.active_tasks),
            "queued_tasks": len(self.task_queue),
            "completed_tasks": len(self.completed_tasks),
            "total_tasks": len(self.active_tasks) + len(self.task_queue) + len(self.completed_tasks)
        }
    
    def add_algorithm(self, algorithm_type: OptimizationType, algorithm: OptimizationAlgorithm):
        """Add a custom optimization algorithm."""
        self.algorithms[algorithm_type] = algorithm
        self.logger.info(f"Added custom algorithm: {algorithm_type.name}")

# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_optimization_module(
    name: str = "optimization",
    optimization_type: OptimizationType = OptimizationType.GENETIC,
    priority: ModulePriority = ModulePriority.NORMAL
) -> OptimizationModule:
    """Create a new optimization module."""
    config = OptimizationConfig(
        name=name,
        priority=priority,
        optimization_type=optimization_type
    )
    return OptimizationModule(config)

def create_genetic_optimization(name: str = "genetic_optimization") -> OptimizationModule:
    """Create a genetic algorithm optimization module."""
    return create_optimization_module(
        name=name,
        optimization_type=OptimizationType.GENETIC,
        priority=ModulePriority.HIGH
    )

def create_simulated_annealing(name: str = "simulated_annealing") -> OptimizationModule:
    """Create a simulated annealing optimization module."""
    return create_optimization_module(
        name=name,
        optimization_type=OptimizationType.SIMULATED_ANNEALING,
        priority=ModulePriority.NORMAL
    )
