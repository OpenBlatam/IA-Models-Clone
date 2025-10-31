"""
Swarm Intelligence System - Advanced Multi-Agent Coordination

This module provides comprehensive swarm intelligence capabilities following FastAPI best practices:
- Multi-agent systems and coordination
- Swarm optimization algorithms
- Collective decision making
- Distributed problem solving
- Emergent behavior simulation
- Ant colony optimization
- Particle swarm optimization
- Bee colony algorithms
- Flocking and herding behaviors
- Swarm robotics coordination
"""

import asyncio
import json
import uuid
import time
import math
import secrets
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import hashlib
import base64

logger = logging.getLogger(__name__)

class AgentType(Enum):
    """Swarm agent types"""
    WORKER = "worker"
    EXPLORER = "explorer"
    COORDINATOR = "coordinator"
    SENTINEL = "sentinel"
    COLLECTOR = "collector"
    BUILDER = "builder"
    DEFENDER = "defender"
    COMMUNICATOR = "communicator"

class OptimizationType(Enum):
    """Optimization algorithm types"""
    ANT_COLONY = "ant_colony"
    PARTICLE_SWARM = "particle_swarm"
    BEE_COLONY = "bee_colony"
    GENETIC_ALGORITHM = "genetic_algorithm"
    SIMULATED_ANNEALING = "simulated_annealing"
    TABU_SEARCH = "tabu_search"
    HARMONY_SEARCH = "harmony_search"
    FIREFLY_ALGORITHM = "firefly_algorithm"

class BehaviorType(Enum):
    """Swarm behavior types"""
    FLOCKING = "flocking"
    HERDING = "herding"
    FORAGING = "foraging"
    NEST_BUILDING = "nest_building"
    PREDATOR_AVOIDANCE = "predator_avoidance"
    RESOURCE_COLLECTION = "resource_collection"
    TERRITORY_DEFENSE = "territory_defense"
    MIGRATION = "migration"

@dataclass
class SwarmAgent:
    """Swarm agent data structure"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_type: AgentType = AgentType.WORKER
    position: Dict[str, float] = field(default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0})
    velocity: Dict[str, float] = field(default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0})
    energy: float = 100.0
    memory: Dict[str, Any] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)
    status: str = "active"
    last_update: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SwarmTask:
    """Swarm task data structure"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str = ""
    description: str = ""
    priority: int = 1
    target_position: Dict[str, float] = field(default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0})
    required_capabilities: List[str] = field(default_factory=list)
    estimated_duration: float = 0.0
    assigned_agents: List[str] = field(default_factory=list)
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SwarmDecision:
    """Swarm decision data structure"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    decision_type: str = ""
    options: List[Dict[str, Any]] = field(default_factory=list)
    votes: Dict[str, int] = field(default_factory=dict)
    consensus_threshold: float = 0.6
    final_decision: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OptimizationResult:
    """Optimization result data structure"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    algorithm_type: OptimizationType = OptimizationType.PARTICLE_SWARM
    objective_function: str = ""
    best_solution: List[float] = field(default_factory=list)
    best_fitness: float = 0.0
    iterations: int = 0
    convergence_time: float = 0.0
    population_size: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

# Base service classes
class BaseSwarmService(ABC):
    """Base swarm intelligence service class"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.is_initialized = False
        self._lock = asyncio.Lock()
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize service"""
        pass
    
    @abstractmethod
    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process service request"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup service"""
        pass

class SwarmAgentService(BaseSwarmService):
    """Swarm agent management service"""
    
    def __init__(self):
        super().__init__("SwarmAgent")
        self.agents: Dict[str, SwarmAgent] = {}
        self.agent_communications: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.agent_positions: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    
    async def initialize(self) -> bool:
        """Initialize swarm agent service"""
        try:
            await asyncio.sleep(0.1)
            self.is_initialized = True
            logger.info("Swarm agent service initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize swarm agent service: {e}")
            return False
    
    async def create_agent(self, 
                          agent_type: AgentType,
                          initial_position: Dict[str, float],
                          capabilities: List[str] = None) -> SwarmAgent:
        """Create swarm agent"""
        
        agent = SwarmAgent(
            agent_type=agent_type,
            position=initial_position,
            capabilities=capabilities or self._get_default_capabilities(agent_type),
            energy=100.0,
            status="active"
        )
        
        async with self._lock:
            self.agents[agent.id] = agent
            self.agent_positions[agent.id].append(initial_position.copy())
        
        logger.info(f"Created swarm agent: {agent_type.value} at {initial_position}")
        return agent
    
    def _get_default_capabilities(self, agent_type: AgentType) -> List[str]:
        """Get default capabilities for agent type"""
        capabilities_map = {
            AgentType.WORKER: ["task_execution", "resource_collection"],
            AgentType.EXPLORER: ["navigation", "mapping", "discovery"],
            AgentType.COORDINATOR: ["task_assignment", "communication", "planning"],
            AgentType.SENTINEL: ["monitoring", "alert", "surveillance"],
            AgentType.COLLECTOR: ["resource_gathering", "transport", "storage"],
            AgentType.BUILDER: ["construction", "repair", "maintenance"],
            AgentType.DEFENDER: ["protection", "combat", "security"],
            AgentType.COMMUNICATOR: ["message_relay", "information_sharing", "coordination"]
        }
        return capabilities_map.get(agent_type, ["basic_operation"])
    
    async def move_agent(self, 
                        agent_id: str,
                        target_position: Dict[str, float],
                        speed: float = 1.0) -> bool:
        """Move agent to target position"""
        async with self._lock:
            if agent_id not in self.agents:
                return False
            
            agent = self.agents[agent_id]
            
            # Calculate movement
            dx = target_position["x"] - agent.position["x"]
            dy = target_position["y"] - agent.position["y"]
            dz = target_position["z"] - agent.position["z"]
            
            distance = math.sqrt(dx**2 + dy**2 + dz**2)
            
            if distance > 0:
                # Update velocity
                agent.velocity = {
                    "x": (dx / distance) * speed,
                    "y": (dy / distance) * speed,
                    "z": (dz / distance) * speed
                }
                
                # Update position
                agent.position = target_position.copy()
                agent.last_update = datetime.utcnow()
                
                # Record position history
                self.agent_positions[agent_id].append(target_position.copy())
                
                # Consume energy based on distance
                agent.energy -= distance * 0.1
                
                logger.debug(f"Moved agent {agent_id} to {target_position}")
                return True
            
            return False
    
    async def communicate_with_agent(self, 
                                   from_agent_id: str,
                                   to_agent_id: str,
                                   message: Dict[str, Any]) -> bool:
        """Enable agent-to-agent communication"""
        async with self._lock:
            if from_agent_id not in self.agents or to_agent_id not in self.agents:
                return False
            
            communication = {
                "from": from_agent_id,
                "to": to_agent_id,
                "message": message,
                "timestamp": datetime.utcnow()
            }
            
            self.agent_communications[to_agent_id].append(communication)
            
            logger.debug(f"Agent {from_agent_id} sent message to {to_agent_id}")
            return True
    
    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process swarm agent request"""
        if not self.is_initialized:
            return {"success": False, "error": "Service not initialized"}
        
        operation = request_data.get("operation", "create_agent")
        
        if operation == "create_agent":
            agent = await self.create_agent(
                agent_type=AgentType(request_data.get("agent_type", "worker")),
                initial_position=request_data.get("position", {"x": 0.0, "y": 0.0, "z": 0.0}),
                capabilities=request_data.get("capabilities", [])
            )
            return {"success": True, "result": agent.__dict__, "service": "swarm_agent"}
        
        elif operation == "move_agent":
            success = await self.move_agent(
                agent_id=request_data.get("agent_id", ""),
                target_position=request_data.get("target_position", {}),
                speed=request_data.get("speed", 1.0)
            )
            return {"success": success, "result": "Moved" if success else "Failed", "service": "swarm_agent"}
        
        elif operation == "communicate":
            success = await self.communicate_with_agent(
                from_agent_id=request_data.get("from_agent_id", ""),
                to_agent_id=request_data.get("to_agent_id", ""),
                message=request_data.get("message", {})
            )
            return {"success": success, "result": "Message sent" if success else "Failed", "service": "swarm_agent"}
        
        else:
            return {"success": False, "error": "Unknown operation"}
    
    async def cleanup(self) -> None:
        """Cleanup swarm agent service"""
        self.agents.clear()
        self.agent_communications.clear()
        self.agent_positions.clear()
        self.is_initialized = False
        logger.info("Swarm agent service cleaned up")

class SwarmOptimizationService(BaseSwarmService):
    """Swarm optimization algorithms service"""
    
    def __init__(self):
        super().__init__("SwarmOptimization")
        self.optimization_results: deque = deque(maxlen=1000)
        self.active_optimizations: Dict[str, Dict[str, Any]] = {}
    
    async def initialize(self) -> bool:
        """Initialize swarm optimization service"""
        try:
            await asyncio.sleep(0.1)
            self.is_initialized = True
            logger.info("Swarm optimization service initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize swarm optimization service: {e}")
            return False
    
    async def run_particle_swarm_optimization(self, 
                                            objective_function: str,
                                            dimensions: int,
                                            population_size: int = 30,
                                            max_iterations: int = 100) -> OptimizationResult:
        """Run particle swarm optimization"""
        
        start_time = time.time()
        
        # Initialize particles
        particles = []
        for i in range(population_size):
            particle = {
                "position": [secrets.randbelow(100) - 50 for _ in range(dimensions)],
                "velocity": [0.0 for _ in range(dimensions)],
                "best_position": None,
                "best_fitness": float('inf')
            }
            particles.append(particle)
        
        # Global best
        global_best_position = None
        global_best_fitness = float('inf')
        
        # PSO parameters
        w = 0.9  # inertia weight
        c1 = 2.0  # cognitive parameter
        c2 = 2.0  # social parameter
        
        # Run optimization
        for iteration in range(max_iterations):
            for particle in particles:
                # Evaluate fitness (simplified objective function)
                fitness = self._evaluate_fitness(particle["position"], objective_function)
                
                # Update personal best
                if fitness < particle["best_fitness"]:
                    particle["best_fitness"] = fitness
                    particle["best_position"] = particle["position"].copy()
                
                # Update global best
                if fitness < global_best_fitness:
                    global_best_fitness = fitness
                    global_best_position = particle["position"].copy()
            
            # Update velocities and positions
            for particle in particles:
                for i in range(dimensions):
                    r1, r2 = secrets.randbelow(100) / 100.0, secrets.randbelow(100) / 100.0
                    
                    # Update velocity
                    particle["velocity"][i] = (w * particle["velocity"][i] +
                                             c1 * r1 * (particle["best_position"][i] - particle["position"][i]) +
                                             c2 * r2 * (global_best_position[i] - particle["position"][i]))
                    
                    # Update position
                    particle["position"][i] += particle["velocity"][i]
        
        convergence_time = time.time() - start_time
        
        result = OptimizationResult(
            algorithm_type=OptimizationType.PARTICLE_SWARM,
            objective_function=objective_function,
            best_solution=global_best_position,
            best_fitness=global_best_fitness,
            iterations=max_iterations,
            convergence_time=convergence_time,
            population_size=population_size
        )
        
        async with self._lock:
            self.optimization_results.append(result)
        
        logger.info(f"PSO completed: best fitness = {global_best_fitness:.4f}")
        return result
    
    def _evaluate_fitness(self, position: List[float], objective_function: str) -> float:
        """Evaluate fitness function (simplified)"""
        if objective_function == "sphere":
            return sum(x**2 for x in position)
        elif objective_function == "rosenbrock":
            return sum(100 * (position[i+1] - position[i]**2)**2 + (1 - position[i])**2 
                      for i in range(len(position)-1))
        elif objective_function == "rastrigin":
            return 10 * len(position) + sum(x**2 - 10 * math.cos(2 * math.pi * x) for x in position)
        else:
            return sum(x**2 for x in position)  # Default to sphere function
    
    async def run_ant_colony_optimization(self, 
                                        problem_type: str,
                                        num_ants: int = 20,
                                        num_iterations: int = 100) -> OptimizationResult:
        """Run ant colony optimization"""
        
        start_time = time.time()
        
        # Simulate ACO for TSP-like problem
        num_cities = 10
        cities = [(secrets.randbelow(100), secrets.randbelow(100)) for _ in range(num_cities)]
        
        # Pheromone matrix
        pheromones = [[1.0 for _ in range(num_cities)] for _ in range(num_cities)]
        
        best_tour = None
        best_distance = float('inf')
        
        for iteration in range(num_iterations):
            # Each ant constructs a tour
            for ant in range(num_ants):
                tour = self._construct_tour(cities, pheromones)
                distance = self._calculate_tour_distance(tour, cities)
                
                if distance < best_distance:
                    best_distance = distance
                    best_tour = tour
            
            # Update pheromones
            self._update_pheromones(pheromones, best_tour, best_distance)
        
        convergence_time = time.time() - start_time
        
        result = OptimizationResult(
            algorithm_type=OptimizationType.ANT_COLONY,
            objective_function=problem_type,
            best_solution=best_tour,
            best_fitness=best_distance,
            iterations=num_iterations,
            convergence_time=convergence_time,
            population_size=num_ants
        )
        
        async with self._lock:
            self.optimization_results.append(result)
        
        logger.info(f"ACO completed: best distance = {best_distance:.4f}")
        return result
    
    def _construct_tour(self, cities: List[Tuple[int, int]], pheromones: List[List[float]]) -> List[int]:
        """Construct tour using pheromone trails"""
        num_cities = len(cities)
        tour = [0]  # Start from city 0
        unvisited = set(range(1, num_cities))
        
        while unvisited:
            current_city = tour[-1]
            probabilities = []
            
            for city in unvisited:
                # Calculate probability based on pheromone and distance
                distance = math.sqrt((cities[current_city][0] - cities[city][0])**2 + 
                                   (cities[current_city][1] - cities[city][1])**2)
                probability = pheromones[current_city][city] / (distance + 0.1)
                probabilities.append((city, probability))
            
            # Select next city based on probabilities
            total_prob = sum(p for _, p in probabilities)
            if total_prob > 0:
                rand_val = secrets.randbelow(int(total_prob * 1000)) / 1000.0
                cumulative = 0.0
                for city, prob in probabilities:
                    cumulative += prob
                    if rand_val <= cumulative:
                        tour.append(city)
                        unvisited.remove(city)
                        break
        
        return tour
    
    def _calculate_tour_distance(self, tour: List[int], cities: List[Tuple[int, int]]) -> float:
        """Calculate total tour distance"""
        total_distance = 0.0
        for i in range(len(tour)):
            current_city = tour[i]
            next_city = tour[(i + 1) % len(tour)]
            distance = math.sqrt((cities[current_city][0] - cities[next_city][0])**2 + 
                               (cities[current_city][1] - cities[next_city][1])**2)
            total_distance += distance
        return total_distance
    
    def _update_pheromones(self, pheromones: List[List[float]], tour: List[int], distance: float):
        """Update pheromone trails"""
        evaporation_rate = 0.1
        pheromone_deposit = 1.0 / distance
        
        # Evaporate pheromones
        for i in range(len(pheromones)):
            for j in range(len(pheromones[i])):
                pheromones[i][j] *= (1 - evaporation_rate)
        
        # Deposit pheromones on best tour
        for i in range(len(tour)):
            current_city = tour[i]
            next_city = tour[(i + 1) % len(tour)]
            pheromones[current_city][next_city] += pheromone_deposit
    
    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process swarm optimization request"""
        if not self.is_initialized:
            return {"success": False, "error": "Service not initialized"}
        
        operation = request_data.get("operation", "run_pso")
        
        if operation == "run_pso":
            result = await self.run_particle_swarm_optimization(
                objective_function=request_data.get("objective_function", "sphere"),
                dimensions=request_data.get("dimensions", 2),
                population_size=request_data.get("population_size", 30),
                max_iterations=request_data.get("max_iterations", 100)
            )
            return {"success": True, "result": result.__dict__, "service": "swarm_optimization"}
        
        elif operation == "run_aco":
            result = await self.run_ant_colony_optimization(
                problem_type=request_data.get("problem_type", "tsp"),
                num_ants=request_data.get("num_ants", 20),
                num_iterations=request_data.get("num_iterations", 100)
            )
            return {"success": True, "result": result.__dict__, "service": "swarm_optimization"}
        
        else:
            return {"success": False, "error": "Unknown operation"}
    
    async def cleanup(self) -> None:
        """Cleanup swarm optimization service"""
        self.optimization_results.clear()
        self.active_optimizations.clear()
        self.is_initialized = False
        logger.info("Swarm optimization service cleaned up")

class CollectiveDecisionMakerService(BaseSwarmService):
    """Collective decision making service"""
    
    def __init__(self):
        super().__init__("CollectiveDecisionMaker")
        self.decisions: Dict[str, SwarmDecision] = {}
        self.voting_sessions: Dict[str, Dict[str, Any]] = {}
    
    async def initialize(self) -> bool:
        """Initialize collective decision maker service"""
        try:
            await asyncio.sleep(0.1)
            self.is_initialized = True
            logger.info("Collective decision maker service initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize collective decision maker service: {e}")
            return False
    
    async def create_decision(self, 
                            decision_type: str,
                            options: List[Dict[str, Any]],
                            consensus_threshold: float = 0.6) -> SwarmDecision:
        """Create collective decision"""
        
        decision = SwarmDecision(
            decision_type=decision_type,
            options=options,
            consensus_threshold=consensus_threshold
        )
        
        async with self._lock:
            self.decisions[decision.id] = decision
            self.voting_sessions[decision.id] = {
                "status": "active",
                "participants": [],
                "started_at": datetime.utcnow()
            }
        
        logger.info(f"Created collective decision: {decision_type}")
        return decision
    
    async def cast_vote(self, 
                       decision_id: str,
                       agent_id: str,
                       option_index: int,
                       confidence: float = 1.0) -> bool:
        """Cast vote in collective decision"""
        async with self._lock:
            if decision_id not in self.decisions:
                return False
            
            decision = self.decisions[decision_id]
            
            if option_index >= len(decision.options):
                return False
            
            # Record vote
            vote_key = f"{agent_id}_{option_index}"
            decision.votes[vote_key] = int(confidence * 100)
            
            # Add participant
            if decision_id in self.voting_sessions:
                session = self.voting_sessions[decision_id]
                if agent_id not in session["participants"]:
                    session["participants"].append(agent_id)
            
            logger.debug(f"Agent {agent_id} voted for option {option_index} in decision {decision_id}")
            return True
    
    async def finalize_decision(self, decision_id: str) -> Dict[str, Any]:
        """Finalize collective decision"""
        async with self._lock:
            if decision_id not in self.decisions:
                return {"success": False, "error": "Decision not found"}
            
            decision = self.decisions[decision_id]
            
            # Count votes for each option
            option_votes = [0] * len(decision.options)
            total_votes = 0
            
            for vote_key, vote_weight in decision.votes.items():
                agent_id, option_index = vote_key.split("_")
                option_index = int(option_index)
                if option_index < len(option_votes):
                    option_votes[option_index] += vote_weight
                    total_votes += vote_weight
            
            # Find winning option
            if total_votes > 0:
                max_votes = max(option_votes)
                winning_option_index = option_votes.index(max_votes)
                consensus_ratio = max_votes / total_votes
                
                if consensus_ratio >= decision.consensus_threshold:
                    decision.final_decision = decision.options[winning_option_index]
                    decision.confidence = consensus_ratio
                    
                    # Update session status
                    if decision_id in self.voting_sessions:
                        self.voting_sessions[decision_id]["status"] = "completed"
                        self.voting_sessions[decision_id]["final_decision"] = decision.final_decision
                    
                    result = {
                        "decision_id": decision_id,
                        "final_decision": decision.final_decision,
                        "confidence": consensus_ratio,
                        "total_votes": total_votes,
                        "winning_option_index": winning_option_index,
                        "consensus_reached": True
                    }
                    
                    logger.info(f"Decision {decision_id} finalized with {consensus_ratio:.2f} consensus")
                    return result
                else:
                    result = {
                        "decision_id": decision_id,
                        "consensus_reached": False,
                        "confidence": consensus_ratio,
                        "required_threshold": decision.consensus_threshold,
                        "message": "Insufficient consensus reached"
                    }
                    
                    logger.warning(f"Decision {decision_id} failed to reach consensus: {consensus_ratio:.2f}")
                    return result
            
            return {"success": False, "error": "No votes cast"}
    
    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process collective decision making request"""
        if not self.is_initialized:
            return {"success": False, "error": "Service not initialized"}
        
        operation = request_data.get("operation", "create_decision")
        
        if operation == "create_decision":
            decision = await self.create_decision(
                decision_type=request_data.get("decision_type", "general"),
                options=request_data.get("options", []),
                consensus_threshold=request_data.get("consensus_threshold", 0.6)
            )
            return {"success": True, "result": decision.__dict__, "service": "collective_decision"}
        
        elif operation == "cast_vote":
            success = await self.cast_vote(
                decision_id=request_data.get("decision_id", ""),
                agent_id=request_data.get("agent_id", ""),
                option_index=request_data.get("option_index", 0),
                confidence=request_data.get("confidence", 1.0)
            )
            return {"success": success, "result": "Vote cast" if success else "Failed", "service": "collective_decision"}
        
        elif operation == "finalize_decision":
            result = await self.finalize_decision(
                decision_id=request_data.get("decision_id", "")
            )
            return {"success": True, "result": result, "service": "collective_decision"}
        
        else:
            return {"success": False, "error": "Unknown operation"}
    
    async def cleanup(self) -> None:
        """Cleanup collective decision maker service"""
        self.decisions.clear()
        self.voting_sessions.clear()
        self.is_initialized = False
        logger.info("Collective decision maker service cleaned up")

# Advanced Swarm Intelligence Manager
class SwarmIntelligenceManager:
    """Main swarm intelligence management system"""
    
    def __init__(self):
        self.swarm_tasks: Dict[str, SwarmTask] = {}
        self.swarm_behaviors: Dict[str, Dict[str, Any]] = {}
        
        # Services
        self.agent_service = SwarmAgentService()
        self.optimization_service = SwarmOptimizationService()
        self.decision_service = CollectiveDecisionMakerService()
        
        self._initialized = False
        self._lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """Initialize swarm intelligence system"""
        if self._initialized:
            return
        
        # Initialize services
        await self.agent_service.initialize()
        await self.optimization_service.initialize()
        await self.decision_service.initialize()
        
        self._initialized = True
        logger.info("Swarm intelligence system initialized")
    
    async def shutdown(self) -> None:
        """Shutdown swarm intelligence system"""
        # Cleanup services
        await self.agent_service.cleanup()
        await self.optimization_service.cleanup()
        await self.decision_service.cleanup()
        
        self.swarm_tasks.clear()
        self.swarm_behaviors.clear()
        
        self._initialized = False
        logger.info("Swarm intelligence system shut down")
    
    async def create_swarm_task(self, 
                              task_type: str,
                              description: str,
                              target_position: Dict[str, float],
                              required_capabilities: List[str]) -> SwarmTask:
        """Create swarm task"""
        
        task = SwarmTask(
            task_type=task_type,
            description=description,
            target_position=target_position,
            required_capabilities=required_capabilities,
            estimated_duration=secrets.randbelow(100) + 10  # 10-110 seconds
        )
        
        async with self._lock:
            self.swarm_tasks[task.id] = task
        
        logger.info(f"Created swarm task: {task_type}")
        return task
    
    async def assign_task_to_agents(self, 
                                  task_id: str,
                                  agent_ids: List[str]) -> bool:
        """Assign task to agents"""
        async with self._lock:
            if task_id not in self.swarm_tasks:
                return False
            
            task = self.swarm_tasks[task_id]
            
            # Verify agents have required capabilities
            for agent_id in agent_ids:
                if agent_id in self.agent_service.agents:
                    agent = self.agent_service.agents[agent_id]
                    if all(cap in agent.capabilities for cap in task.required_capabilities):
                        task.assigned_agents.append(agent_id)
                        task.status = "assigned"
                    else:
                        logger.warning(f"Agent {agent_id} lacks required capabilities")
                        return False
                else:
                    logger.warning(f"Agent {agent_id} not found")
                    return False
            
            logger.info(f"Assigned task {task_id} to {len(agent_ids)} agents")
            return True
    
    async def process_swarm_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process swarm intelligence request"""
        if not self._initialized:
            return {"success": False, "error": "Swarm intelligence system not initialized"}
        
        service_type = request_data.get("service_type", "agent")
        
        if service_type == "agent":
            return await self.agent_service.process_request(request_data)
        elif service_type == "optimization":
            return await self.optimization_service.process_request(request_data)
        elif service_type == "decision":
            return await self.decision_service.process_request(request_data)
        else:
            return {"success": False, "error": "Unknown service type"}
    
    def get_swarm_intelligence_summary(self) -> Dict[str, Any]:
        """Get swarm intelligence system summary"""
        return {
            "initialized": self._initialized,
            "active_agents": len(self.agent_service.agents),
            "swarm_tasks": len(self.swarm_tasks),
            "services": {
                "agent_service": self.agent_service.is_initialized,
                "optimization_service": self.optimization_service.is_initialized,
                "decision_service": self.decision_service.is_initialized
            },
            "statistics": {
                "total_optimizations": len(self.optimization_service.optimization_results),
                "total_decisions": len(self.decision_service.decisions),
                "total_communications": sum(len(comms) for comms in self.agent_service.agent_communications.values())
            }
        }

# Global swarm intelligence manager instance
_global_swarm_intelligence_manager: Optional[SwarmIntelligenceManager] = None

def get_swarm_intelligence_manager() -> SwarmIntelligenceManager:
    """Get global swarm intelligence manager instance"""
    global _global_swarm_intelligence_manager
    if _global_swarm_intelligence_manager is None:
        _global_swarm_intelligence_manager = SwarmIntelligenceManager()
    return _global_swarm_intelligence_manager

async def initialize_swarm_intelligence() -> None:
    """Initialize global swarm intelligence system"""
    manager = get_swarm_intelligence_manager()
    await manager.initialize()

async def shutdown_swarm_intelligence() -> None:
    """Shutdown global swarm intelligence system"""
    manager = get_swarm_intelligence_manager()
    await manager.shutdown()

async def create_swarm_task(task_type: str, description: str, target_position: Dict[str, float], required_capabilities: List[str]) -> SwarmTask:
    """Create swarm task using global manager"""
    manager = get_swarm_intelligence_manager()
    return await manager.create_swarm_task(task_type, description, target_position, required_capabilities)





















