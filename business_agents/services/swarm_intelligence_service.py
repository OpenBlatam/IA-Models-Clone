"""
Swarm Intelligence Service
==========================

Advanced swarm intelligence service for collective behavior,
distributed decision making, and emergent intelligence.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import uuid
import hashlib
import hmac
from cryptography.fernet import Fernet
import base64
import threading
import time
import math
import random
import torch
import torch.nn as nn
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier
import networkx as nx

logger = logging.getLogger(__name__)

class SwarmType(Enum):
    """Types of swarms."""
    PARTICLE_SWARM = "particle_swarm"
    ANT_COLONY = "ant_colony"
    BEE_COLONY = "bee_colony"
    FISH_SCHOOL = "fish_school"
    BIRD_FLOCK = "bird_flock"
    FIREFLY = "firefly"
    CUSTOM = "custom"

class AgentBehavior(Enum):
    """Agent behaviors."""
    EXPLORATION = "exploration"
    EXPLOITATION = "exploitation"
    COOPERATION = "cooperation"
    COMPETITION = "competition"
    ADAPTATION = "adaptation"
    COMMUNICATION = "communication"
    LEADERSHIP = "leadership"
    FOLLOWING = "following"

class SwarmAlgorithm(Enum):
    """Swarm algorithms."""
    PSO = "pso"  # Particle Swarm Optimization
    ACO = "aco"  # Ant Colony Optimization
    ABC = "abc"  # Artificial Bee Colony
    FFA = "ffa"  # Firefly Algorithm
    BFO = "bfo"  # Bacterial Foraging Optimization
    CSO = "cso"  # Cuckoo Search Optimization
    GWO = "gwo"  # Grey Wolf Optimizer
    WOA = "woa"  # Whale Optimization Algorithm

@dataclass
class SwarmAgent:
    """Swarm agent definition."""
    agent_id: str
    swarm_id: str
    position: List[float]
    velocity: List[float]
    fitness: float
    best_position: List[float]
    best_fitness: float
    behavior: AgentBehavior
    neighbors: List[str]
    communication_range: float
    last_update: datetime
    metadata: Dict[str, Any]

@dataclass
class Swarm:
    """Swarm definition."""
    swarm_id: str
    name: str
    swarm_type: SwarmType
    algorithm: SwarmAlgorithm
    agents: List[str]
    global_best_position: List[float]
    global_best_fitness: float
    objective_function: str
    constraints: Dict[str, Any]
    parameters: Dict[str, Any]
    status: str
    created_at: datetime
    last_update: datetime
    metadata: Dict[str, Any]

@dataclass
class SwarmTask:
    """Swarm task definition."""
    task_id: str
    swarm_id: str
    task_type: str
    objective: str
    constraints: Dict[str, Any]
    parameters: Dict[str, Any]
    status: str
    progress: float
    result: Optional[Any]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    metadata: Dict[str, Any]

@dataclass
class SwarmCommunication:
    """Swarm communication definition."""
    communication_id: str
    sender_id: str
    receiver_id: str
    message_type: str
    content: Dict[str, Any]
    timestamp: datetime
    metadata: Dict[str, Any]

class SwarmIntelligenceService:
    """
    Advanced swarm intelligence service.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.swarms = {}
        self.swarm_agents = {}
        self.swarm_tasks = {}
        self.swarm_communications = {}
        self.objective_functions = {}
        self.optimization_algorithms = {}
        
        # Swarm intelligence configurations
        self.swarm_config = config.get("swarm_intelligence", {
            "max_swarms": 100,
            "max_agents_per_swarm": 1000,
            "max_iterations": 1000,
            "convergence_threshold": 0.001,
            "communication_enabled": True,
            "adaptation_enabled": True,
            "learning_enabled": True,
            "real_time_optimization": True
        })
        
    async def initialize(self):
        """Initialize the swarm intelligence service."""
        try:
            await self._initialize_objective_functions()
            await self._initialize_optimization_algorithms()
            await self._load_default_swarms()
            await self._start_swarm_monitoring()
            logger.info("Swarm Intelligence Service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Swarm Intelligence Service: {str(e)}")
            raise
            
    async def _initialize_objective_functions(self):
        """Initialize objective functions."""
        try:
            self.objective_functions = {
                "sphere_function": {
                    "function": self._sphere_function,
                    "dimensions": 2,
                    "global_minimum": 0.0,
                    "bounds": [(-5.12, 5.12), (-5.12, 5.12)],
                    "description": "Simple unimodal function"
                },
                "rosenbrock_function": {
                    "function": self._rosenbrock_function,
                    "dimensions": 2,
                    "global_minimum": 0.0,
                    "bounds": [(-2.048, 2.048), (-2.048, 2.048)],
                    "description": "Classic optimization benchmark"
                },
                "rastrigin_function": {
                    "function": self._rastrigin_function,
                    "dimensions": 2,
                    "global_minimum": 0.0,
                    "bounds": [(-5.12, 5.12), (-5.12, 5.12)],
                    "description": "Multimodal function with many local minima"
                },
                "ackley_function": {
                    "function": self._ackley_function,
                    "dimensions": 2,
                    "global_minimum": 0.0,
                    "bounds": [(-32.768, 32.768), (-32.768, 32.768)],
                    "description": "Complex multimodal function"
                },
                "workflow_optimization": {
                    "function": self._workflow_optimization_function,
                    "dimensions": 10,
                    "global_minimum": 0.0,
                    "bounds": [(0, 100) for _ in range(10)],
                    "description": "Workflow performance optimization"
                },
                "resource_allocation": {
                    "function": self._resource_allocation_function,
                    "dimensions": 5,
                    "global_minimum": 0.0,
                    "bounds": [(0, 1000) for _ in range(5)],
                    "description": "Resource allocation optimization"
                }
            }
            
            logger.info("Objective functions initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize objective functions: {str(e)}")
            
    async def _initialize_optimization_algorithms(self):
        """Initialize optimization algorithms."""
        try:
            self.optimization_algorithms = {
                "pso": {
                    "name": "Particle Swarm Optimization",
                    "parameters": {
                        "w": 0.9,  # inertia weight
                        "c1": 2.0,  # cognitive parameter
                        "c2": 2.0,  # social parameter
                        "v_max": 1.0,  # maximum velocity
                        "population_size": 50
                    },
                    "description": "Particle swarm optimization algorithm"
                },
                "aco": {
                    "name": "Ant Colony Optimization",
                    "parameters": {
                        "alpha": 1.0,  # pheromone importance
                        "beta": 2.0,   # heuristic importance
                        "rho": 0.5,    # evaporation rate
                        "q": 100,      # pheromone quantity
                        "ants": 50
                    },
                    "description": "Ant colony optimization algorithm"
                },
                "abc": {
                    "name": "Artificial Bee Colony",
                    "parameters": {
                        "employed_bees": 25,
                        "onlooker_bees": 25,
                        "scout_bees": 1,
                        "limit": 100,
                        "max_trials": 50
                    },
                    "description": "Artificial bee colony algorithm"
                },
                "ffa": {
                    "name": "Firefly Algorithm",
                    "parameters": {
                        "alpha": 0.5,   # randomization parameter
                        "beta": 1.0,    # attractiveness parameter
                        "gamma": 1.0,   # absorption coefficient
                        "population_size": 50
                    },
                    "description": "Firefly algorithm"
                }
            }
            
            logger.info("Optimization algorithms initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize optimization algorithms: {str(e)}")
            
    async def _load_default_swarms(self):
        """Load default swarms."""
        try:
            # Create sample swarms
            swarms = [
                Swarm(
                    swarm_id="workflow_optimization_swarm",
                    name="Workflow Optimization Swarm",
                    swarm_type=SwarmType.PARTICLE_SWARM,
                    algorithm=SwarmAlgorithm.PSO,
                    agents=[],
                    global_best_position=[0.0, 0.0],
                    global_best_fitness=float('inf'),
                    objective_function="workflow_optimization",
                    constraints={"max_resources": 1000, "min_performance": 0.8},
                    parameters={"w": 0.9, "c1": 2.0, "c2": 2.0, "population_size": 50},
                    status="active",
                    created_at=datetime.utcnow(),
                    last_update=datetime.utcnow(),
                    metadata={"domain": "workflow", "optimization_target": "performance"}
                ),
                Swarm(
                    swarm_id="resource_allocation_swarm",
                    name="Resource Allocation Swarm",
                    swarm_type=SwarmType.ANT_COLONY,
                    algorithm=SwarmAlgorithm.ACO,
                    agents=[],
                    global_best_position=[0.0, 0.0, 0.0, 0.0, 0.0],
                    global_best_fitness=float('inf'),
                    objective_function="resource_allocation",
                    constraints={"budget_limit": 10000, "resource_limits": [100, 200, 150, 300, 250]},
                    parameters={"alpha": 1.0, "beta": 2.0, "rho": 0.5, "ants": 50},
                    status="active",
                    created_at=datetime.utcnow(),
                    last_update=datetime.utcnow(),
                    metadata={"domain": "resources", "optimization_target": "cost"}
                ),
                Swarm(
                    swarm_id="task_scheduling_swarm",
                    name="Task Scheduling Swarm",
                    swarm_type=SwarmType.BEE_COLONY,
                    algorithm=SwarmAlgorithm.ABC,
                    agents=[],
                    global_best_position=[0.0] * 10,
                    global_best_fitness=float('inf'),
                    objective_function="sphere_function",
                    constraints={"deadline": 3600, "dependencies": []},
                    parameters={"employed_bees": 25, "onlooker_bees": 25, "scout_bees": 1},
                    status="active",
                    created_at=datetime.utcnow(),
                    last_update=datetime.utcnow(),
                    metadata={"domain": "scheduling", "optimization_target": "makespan"}
                )
            ]
            
            for swarm in swarms:
                self.swarms[swarm.swarm_id] = swarm
                
            logger.info(f"Loaded {len(swarms)} default swarms")
            
        except Exception as e:
            logger.error(f"Failed to load default swarms: {str(e)}")
            
    async def _start_swarm_monitoring(self):
        """Start swarm monitoring."""
        try:
            # Start background swarm monitoring
            asyncio.create_task(self._monitor_swarms())
            logger.info("Started swarm monitoring")
            
        except Exception as e:
            logger.error(f"Failed to start swarm monitoring: {str(e)}")
            
    async def _monitor_swarms(self):
        """Monitor swarms."""
        while True:
            try:
                # Update each active swarm
                for swarm_id, swarm in self.swarms.items():
                    if swarm.status == "active":
                        await self._update_swarm(swarm)
                        
                # Wait before next monitoring cycle
                await asyncio.sleep(1)  # Monitor every second
                
            except Exception as e:
                logger.error(f"Error in swarm monitoring: {str(e)}")
                await asyncio.sleep(5)  # Wait longer on error
                
    async def _update_swarm(self, swarm: Swarm):
        """Update swarm."""
        try:
            # Update swarm agents
            for agent_id in swarm.agents:
                if agent_id in self.swarm_agents:
                    agent = self.swarm_agents[agent_id]
                    await self._update_agent(agent, swarm)
                    
            # Update global best
            await self._update_global_best(swarm)
            
            # Update swarm status
            swarm.last_update = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Failed to update swarm: {str(e)}")
            
    async def _update_agent(self, agent: SwarmAgent, swarm: Swarm):
        """Update swarm agent."""
        try:
            # Update agent based on algorithm
            if swarm.algorithm == SwarmAlgorithm.PSO:
                await self._update_pso_agent(agent, swarm)
            elif swarm.algorithm == SwarmAlgorithm.ACO:
                await self._update_aco_agent(agent, swarm)
            elif swarm.algorithm == SwarmAlgorithm.ABC:
                await self._update_abc_agent(agent, swarm)
            else:
                await self._update_generic_agent(agent, swarm)
                
            # Update agent timestamp
            agent.last_update = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Failed to update agent: {str(e)}")
            
    async def _update_pso_agent(self, agent: SwarmAgent, swarm: Swarm):
        """Update PSO agent."""
        try:
            # PSO update equations
            w = swarm.parameters.get("w", 0.9)
            c1 = swarm.parameters.get("c1", 2.0)
            c2 = swarm.parameters.get("c2", 2.0)
            v_max = swarm.parameters.get("v_max", 1.0)
            
            # Update velocity
            for i in range(len(agent.position)):
                r1 = random.random()
                r2 = random.random()
                
                cognitive = c1 * r1 * (agent.best_position[i] - agent.position[i])
                social = c2 * r2 * (swarm.global_best_position[i] - agent.position[i])
                
                agent.velocity[i] = w * agent.velocity[i] + cognitive + social
                
                # Limit velocity
                agent.velocity[i] = max(-v_max, min(v_max, agent.velocity[i]))
                
            # Update position
            for i in range(len(agent.position)):
                agent.position[i] += agent.velocity[i]
                
            # Evaluate fitness
            fitness = await self._evaluate_fitness(agent.position, swarm.objective_function)
            agent.fitness = fitness
            
            # Update personal best
            if fitness < agent.best_fitness:
                agent.best_fitness = fitness
                agent.best_position = agent.position.copy()
                
        except Exception as e:
            logger.error(f"Failed to update PSO agent: {str(e)}")
            
    async def _update_aco_agent(self, agent: SwarmAgent, swarm: Swarm):
        """Update ACO agent."""
        try:
            # ACO update logic
            alpha = swarm.parameters.get("alpha", 1.0)
            beta = swarm.parameters.get("beta", 2.0)
            
            # Simple ACO position update
            for i in range(len(agent.position)):
                # Add some randomness and pheromone influence
                pheromone_influence = alpha * random.random()
                heuristic_influence = beta * random.random()
                
                agent.position[i] += (pheromone_influence + heuristic_influence) * 0.1
                
            # Evaluate fitness
            fitness = await self._evaluate_fitness(agent.position, swarm.objective_function)
            agent.fitness = fitness
            
            # Update personal best
            if fitness < agent.best_fitness:
                agent.best_fitness = fitness
                agent.best_position = agent.position.copy()
                
        except Exception as e:
            logger.error(f"Failed to update ACO agent: {str(e)}")
            
    async def _update_abc_agent(self, agent: SwarmAgent, swarm: Swarm):
        """Update ABC agent."""
        try:
            # ABC update logic
            # Simple bee colony position update
            for i in range(len(agent.position)):
                # Add some exploration behavior
                exploration = random.uniform(-0.1, 0.1)
                agent.position[i] += exploration
                
            # Evaluate fitness
            fitness = await self._evaluate_fitness(agent.position, swarm.objective_function)
            agent.fitness = fitness
            
            # Update personal best
            if fitness < agent.best_fitness:
                agent.best_fitness = fitness
                agent.best_position = agent.position.copy()
                
        except Exception as e:
            logger.error(f"Failed to update ABC agent: {str(e)}")
            
    async def _update_generic_agent(self, agent: SwarmAgent, swarm: Swarm):
        """Update generic agent."""
        try:
            # Generic update logic
            for i in range(len(agent.position)):
                # Add some random movement
                movement = random.uniform(-0.1, 0.1)
                agent.position[i] += movement
                
            # Evaluate fitness
            fitness = await self._evaluate_fitness(agent.position, swarm.objective_function)
            agent.fitness = fitness
            
            # Update personal best
            if fitness < agent.best_fitness:
                agent.best_fitness = fitness
                agent.best_position = agent.position.copy()
                
        except Exception as e:
            logger.error(f"Failed to update generic agent: {str(e)}")
            
    async def _update_global_best(self, swarm: Swarm):
        """Update global best solution."""
        try:
            # Find best agent in swarm
            best_agent = None
            best_fitness = float('inf')
            
            for agent_id in swarm.agents:
                if agent_id in self.swarm_agents:
                    agent = self.swarm_agents[agent_id]
                    if agent.best_fitness < best_fitness:
                        best_fitness = agent.best_fitness
                        best_agent = agent
                        
            # Update global best
            if best_agent and best_fitness < swarm.global_best_fitness:
                swarm.global_best_fitness = best_fitness
                swarm.global_best_position = best_agent.best_position.copy()
                
        except Exception as e:
            logger.error(f"Failed to update global best: {str(e)}")
            
    async def _evaluate_fitness(self, position: List[float], objective_function: str) -> float:
        """Evaluate fitness function."""
        try:
            if objective_function in self.objective_functions:
                func = self.objective_functions[objective_function]["function"]
                return func(position)
            else:
                # Default fitness function
                return sum(x**2 for x in position)
                
        except Exception as e:
            logger.error(f"Failed to evaluate fitness: {str(e)}")
            return float('inf')
            
    def _sphere_function(self, position: List[float]) -> float:
        """Sphere function."""
        return sum(x**2 for x in position)
        
    def _rosenbrock_function(self, position: List[float]) -> float:
        """Rosenbrock function."""
        if len(position) < 2:
            return 0.0
        x, y = position[0], position[1]
        return 100 * (y - x**2)**2 + (1 - x)**2
        
    def _rastrigin_function(self, position: List[float]) -> float:
        """Rastrigin function."""
        n = len(position)
        return 10 * n + sum(x**2 - 10 * math.cos(2 * math.pi * x) for x in position)
        
    def _ackley_function(self, position: List[float]) -> float:
        """Ackley function."""
        n = len(position)
        sum1 = sum(x**2 for x in position)
        sum2 = sum(math.cos(2 * math.pi * x) for x in position)
        return -20 * math.exp(-0.2 * math.sqrt(sum1 / n)) - math.exp(sum2 / n) + 20 + math.e
        
    def _workflow_optimization_function(self, position: List[float]) -> float:
        """Workflow optimization function."""
        # Simple workflow optimization objective
        # Minimize execution time while maximizing resource utilization
        execution_time = sum(position) / len(position)
        resource_utilization = 1.0 / (1.0 + sum(x**2 for x in position) / len(position))
        return execution_time - resource_utilization
        
    def _resource_allocation_function(self, position: List[float]) -> float:
        """Resource allocation function."""
        # Simple resource allocation objective
        # Minimize cost while meeting constraints
        cost = sum(position)
        constraint_violation = max(0, sum(position) - 1000)  # Budget constraint
        return cost + constraint_violation
        
    async def create_swarm(self, swarm: Swarm) -> str:
        """Create a new swarm."""
        try:
            # Generate swarm ID if not provided
            if not swarm.swarm_id:
                swarm.swarm_id = f"swarm_{uuid.uuid4().hex[:8]}"
                
            # Set timestamps
            swarm.created_at = datetime.utcnow()
            swarm.last_update = datetime.utcnow()
            
            # Create swarm
            self.swarms[swarm.swarm_id] = swarm
            
            logger.info(f"Created swarm: {swarm.swarm_id}")
            
            return swarm.swarm_id
            
        except Exception as e:
            logger.error(f"Failed to create swarm: {str(e)}")
            raise
            
    async def add_agent_to_swarm(self, swarm_id: str, agent: SwarmAgent) -> str:
        """Add agent to swarm."""
        try:
            # Generate agent ID if not provided
            if not agent.agent_id:
                agent.agent_id = f"agent_{uuid.uuid4().hex[:8]}"
                
            # Set swarm ID
            agent.swarm_id = swarm_id
            
            # Set timestamp
            agent.last_update = datetime.utcnow()
            
            # Create agent
            self.swarm_agents[agent.agent_id] = agent
            
            # Add to swarm
            if swarm_id in self.swarms:
                self.swarms[swarm_id].agents.append(agent.agent_id)
                
            logger.info(f"Added agent {agent.agent_id} to swarm {swarm_id}")
            
            return agent.agent_id
            
        except Exception as e:
            logger.error(f"Failed to add agent to swarm: {str(e)}")
            raise
            
    async def get_swarm(self, swarm_id: str) -> Optional[Swarm]:
        """Get swarm by ID."""
        return self.swarms.get(swarm_id)
        
    async def get_swarms(self, swarm_type: Optional[SwarmType] = None) -> List[Swarm]:
        """Get swarms."""
        swarms = list(self.swarms.values())
        
        if swarm_type:
            swarms = [s for s in swarms if s.swarm_type == swarm_type]
            
        return swarms
        
    async def get_swarm_agents(self, swarm_id: str) -> List[SwarmAgent]:
        """Get swarm agents."""
        agents = []
        if swarm_id in self.swarms:
            for agent_id in self.swarms[swarm_id].agents:
                if agent_id in self.swarm_agents:
                    agents.append(self.swarm_agents[agent_id])
        return agents
        
    async def run_optimization(self, swarm_id: str, max_iterations: int = 100) -> SwarmTask:
        """Run swarm optimization."""
        try:
            if swarm_id not in self.swarms:
                raise ValueError(f"Swarm {swarm_id} not found")
                
            swarm = self.swarms[swarm_id]
            
            # Create optimization task
            task = SwarmTask(
                task_id=f"task_{uuid.uuid4().hex[:8]}",
                swarm_id=swarm_id,
                task_type="optimization",
                objective=swarm.objective_function,
                constraints=swarm.constraints,
                parameters=swarm.parameters,
                status="running",
                progress=0.0,
                result=None,
                created_at=datetime.utcnow(),
                started_at=datetime.utcnow(),
                completed_at=None,
                metadata={"max_iterations": max_iterations}
            )
            
            # Store task
            self.swarm_tasks[task.task_id] = task
            
            # Run optimization
            await self._run_optimization_task(task, max_iterations)
            
            return task
            
        except Exception as e:
            logger.error(f"Failed to run optimization: {str(e)}")
            raise
            
    async def _run_optimization_task(self, task: SwarmTask, max_iterations: int):
        """Run optimization task."""
        try:
            swarm = self.swarms[task.swarm_id]
            
            # Run optimization iterations
            for iteration in range(max_iterations):
                # Update swarm
                await self._update_swarm(swarm)
                
                # Update progress
                task.progress = (iteration + 1) / max_iterations
                
                # Check convergence
                if self._check_convergence(swarm):
                    break
                    
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.01)
                
            # Complete task
            task.status = "completed"
            task.completed_at = datetime.utcnow()
            task.result = {
                "best_position": swarm.global_best_position,
                "best_fitness": swarm.global_best_fitness,
                "iterations": iteration + 1,
                "converged": iteration < max_iterations - 1
            }
            
            logger.info(f"Completed optimization task: {task.task_id}")
            
        except Exception as e:
            logger.error(f"Failed to run optimization task: {str(e)}")
            task.status = "failed"
            task.result = {"error": str(e)}
            
    def _check_convergence(self, swarm: Swarm) -> bool:
        """Check if swarm has converged."""
        try:
            # Simple convergence check
            threshold = self.swarm_config.get("convergence_threshold", 0.001)
            
            # Check if fitness improvement is below threshold
            if swarm.global_best_fitness < threshold:
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Failed to check convergence: {str(e)}")
            return False
            
    async def get_service_status(self) -> Dict[str, Any]:
        """Get swarm intelligence service status."""
        try:
            active_swarms = len([s for s in self.swarms.values() if s.status == "active"])
            total_agents = len(self.swarm_agents)
            total_tasks = len(self.swarm_tasks)
            completed_tasks = len([t for t in self.swarm_tasks.values() if t.status == "completed"])
            
            return {
                "service_status": "active",
                "total_swarms": len(self.swarms),
                "active_swarms": active_swarms,
                "total_agents": total_agents,
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "objective_functions": len(self.objective_functions),
                "optimization_algorithms": len(self.optimization_algorithms),
                "communication_enabled": self.swarm_config.get("communication_enabled", True),
                "adaptation_enabled": self.swarm_config.get("adaptation_enabled", True),
                "learning_enabled": self.swarm_config.get("learning_enabled", True),
                "real_time_optimization": self.swarm_config.get("real_time_optimization", True),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get service status: {str(e)}")
            return {"service_status": "error", "error": str(e)}



























