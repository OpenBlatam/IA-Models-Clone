"""
Multi-Agent Swarm Intelligence for HeyGen AI

This module provides swarm intelligence capabilities with advanced features:
- Emergent behavior systems
- Adaptive coordination patterns
- Specialized agent types
- Scalable swarm architectures
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import gc
import time
import asyncio
import random
import math

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class SwarmConfig:
    """Configuration for swarm intelligence."""
    
    # Swarm Settings
    num_agents: int = 10
    collaboration_mode: str = "hierarchical"  # hierarchical, decentralized, centralized
    learning_rate: float = 0.01
    
    # Behavior Settings
    enable_emergent_behavior: bool = True
    enable_adaptive_coordination: bool = True
    enable_specialization: bool = True
    
    # Communication Settings
    communication_range: float = 100.0
    communication_frequency: float = 1.0  # Hz
    
    # Performance Settings
    max_iterations: int = 1000
    convergence_threshold: float = 1e-6
    enable_parallel_execution: bool = True


@dataclass
class AgentType:
    """Agent type definitions."""
    
    type: str  # explorer, exploiter, coordinator, specialist
    count: int
    behavior: str
    learning_rate: float
    capabilities: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.capabilities:
            self.capabilities = self._get_default_capabilities()
    
    def _get_default_capabilities(self) -> List[str]:
        """Get default capabilities for agent type."""
        if self.type == "explorer":
            return ["exploration", "discovery", "mapping"]
        elif self.type == "exploiter":
            return ["optimization", "refinement", "efficiency"]
        elif self.type == "coordinator":
            return ["coordination", "communication", "planning"]
        elif self.type == "specialist":
            return ["expertise", "precision", "analysis"]
        else:
            return ["general"]


class SwarmAgent:
    """Individual agent in the swarm."""
    
    def __init__(self, agent_id: str, agent_type: AgentType, position: np.ndarray):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.position = position
        self.velocity = np.random.randn(len(position)) * 0.1
        self.best_position = position.copy()
        self.best_fitness = float('inf')
        self.fitness_history = []
        
        # Communication state
        self.neighbors = []
        self.messages = []
        self.last_communication = time.time()
        
        # Learning state
        self.learning_rate = agent_type.learning_rate
        self.adaptation_factor = 1.0
        
        # Specialization
        self.specialization_level = 0.0
        self.expertise_areas = agent_type.capabilities.copy()
    
    def update_position(self, swarm_center: np.ndarray, global_best: np.ndarray):
        """Update agent position using swarm intelligence algorithms."""
        try:
            # Calculate social influence
            social_influence = (global_best - self.position) * 0.1
            
            # Calculate swarm cohesion
            cohesion = (swarm_center - self.position) * 0.05
            
            # Calculate individual memory
            memory = (self.best_position - self.position) * 0.1
            
            # Update velocity
            self.velocity = (0.7 * self.velocity + 
                           0.2 * social_influence + 
                           0.1 * cohesion + 
                           0.1 * memory)
            
            # Apply velocity limits
            self.velocity = np.clip(self.velocity, -1.0, 1.0)
            
            # Update position
            self.position += self.velocity
            
            # Update best position if improved
            current_fitness = self._calculate_fitness()
            if current_fitness < self.best_fitness:
                self.best_position = self.position.copy()
                self.best_fitness = current_fitness
            
            # Record fitness
            self.fitness_history.append(current_fitness)
            
        except Exception as e:
            logger.warning(f"Failed to update position for agent {self.agent_id}: {e}")
    
    def _calculate_fitness(self) -> float:
        """Calculate fitness value for current position."""
        try:
            # Simple fitness function (distance from origin)
            # In practice, this would be a more sophisticated objective function
            distance = np.linalg.norm(self.position)
            noise = np.random.normal(0, 0.1)
            return distance + noise
            
        except Exception as e:
            logger.warning(f"Failed to calculate fitness for agent {self.agent_id}: {e}")
            return float('inf')
    
    def communicate(self, other_agent: 'SwarmAgent', message: Dict[str, Any]):
        """Communicate with another agent."""
        try:
            if self._can_communicate(other_agent):
                # Send message
                other_agent.messages.append({
                    'from': self.agent_id,
                    'type': message['type'],
                    'data': message['data'],
                    'timestamp': time.time()
                })
                
                # Update communication timestamp
                self.last_communication = time.time()
                other_agent.last_communication = time.time()
                
                return True
            return False
            
        except Exception as e:
            logger.warning(f"Communication failed between {self.agent_id} and {other_agent.agent_id}: {e}")
            return False
    
    def _can_communicate(self, other_agent: 'SwarmAgent') -> bool:
        """Check if communication is possible with another agent."""
        try:
            # Check distance
            distance = np.linalg.norm(self.position - other_agent.position)
            if distance > 100.0:  # Communication range
                return False
            
            # Check communication frequency
            time_since_last = time.time() - self.last_communication
            if time_since_last < 1.0:  # Communication frequency limit
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Communication check failed: {e}")
            return False
    
    def adapt_behavior(self, swarm_state: Dict[str, Any]):
        """Adapt behavior based on swarm state."""
        try:
            # Adapt learning rate based on swarm performance
            if swarm_state.get('improving', False):
                self.adaptation_factor *= 1.1
            else:
                self.adaptation_factor *= 0.9
            
            # Clamp adaptation factor
            self.adaptation_factor = np.clip(self.adaptation_factor, 0.1, 2.0)
            
            # Update learning rate
            self.learning_rate = self.agent_type.learning_rate * self.adaptation_factor
            
            # Adapt specialization based on swarm needs
            if swarm_state.get('need_exploration', False) and 'exploration' in self.expertise_areas:
                self.specialization_level = min(1.0, self.specialization_level + 0.1)
            elif swarm_state.get('need_optimization', False) and 'optimization' in self.expertise_areas:
                self.specialization_level = min(1.0, self.specialization_level + 0.1)
            
        except Exception as e:
            logger.warning(f"Behavior adaptation failed for agent {self.agent_id}: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            'agent_id': self.agent_id,
            'type': self.agent_type.type,
            'position': self.position.tolist(),
            'velocity': self.velocity.tolist(),
            'best_fitness': self.best_fitness,
            'learning_rate': self.learning_rate,
            'specialization_level': self.specialization_level,
            'neighbors_count': len(self.neighbors),
            'messages_count': len(self.messages)
        }


class SwarmCoordinator:
    """Coordinates swarm behavior and communication."""
    
    def __init__(self, config: SwarmConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.coordinator")
        self.agents: Dict[str, SwarmAgent] = {}
        self.communication_graph = {}
        self.swarm_state = {}
        
    def add_agent(self, agent: SwarmAgent) -> bool:
        """Add an agent to the swarm."""
        try:
            if agent.agent_id in self.agents:
                self.logger.warning(f"Agent {agent.agent_id} already exists")
                return False
            
            self.agents[agent.agent_id] = agent
            self.communication_graph[agent.agent_id] = []
            
            self.logger.info(f"Added agent {agent.agent_id} of type {agent.agent_type.type}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add agent {agent.agent_id}: {e}")
            return False
    
    def update_communication_graph(self):
        """Update communication graph based on agent positions."""
        try:
            for agent_id, agent in self.agents.items():
                self.communication_graph[agent_id] = []
                
                for other_id, other_agent in self.agents.items():
                    if agent_id != other_id:
                        distance = np.linalg.norm(agent.position - other_agent.position)
                        if distance <= self.config.communication_range:
                            self.communication_graph[agent_id].append(other_id)
            
        except Exception as e:
            self.logger.error(f"Failed to update communication graph: {e}")
    
    def coordinate_swarm(self) -> Dict[str, Any]:
        """Coordinate swarm behavior."""
        try:
            # Update communication graph
            self.update_communication_graph()
            
            # Calculate swarm center
            positions = [agent.position for agent in self.agents.values()]
            swarm_center = np.mean(positions, axis=0)
            
            # Find global best
            best_agent = min(self.agents.values(), key=lambda a: a.best_fitness)
            global_best = best_agent.best_position
            
            # Update swarm state
            self.swarm_state = {
                'swarm_center': swarm_center.tolist(),
                'global_best': global_best.tolist(),
                'best_fitness': best_agent.best_fitness,
                'improving': self._is_swarm_improving(),
                'need_exploration': self._needs_exploration(),
                'need_optimization': self._needs_optimization()
            }
            
            return self.swarm_state
            
        except Exception as e:
            self.logger.error(f"Swarm coordination failed: {e}")
            return {}
    
    def _is_swarm_improving(self) -> bool:
        """Check if swarm is improving."""
        try:
            if len(self.agents) < 2:
                return False
            
            # Check if best fitness is improving
            best_fitnesses = [agent.best_fitness for agent in self.agents.values()]
            recent_best = min(best_fitnesses[-10:]) if len(best_fitnesses) >= 10 else min(best_fitnesses)
            overall_best = min(best_fitnesses)
            
            return recent_best < overall_best
            
        except Exception as e:
            logger.warning(f"Failed to check swarm improvement: {e}")
            return False
    
    def _needs_exploration(self) -> bool:
        """Check if swarm needs more exploration."""
        try:
            # Check diversity of agent positions
            positions = [agent.position for agent in self.agents.values()]
            if len(positions) < 2:
                return True
            
            # Calculate position variance
            positions_array = np.array(positions)
            variance = np.var(positions_array, axis=0).mean()
            
            # Low variance suggests need for exploration
            return variance < 1.0
            
        except Exception as e:
            logger.warning(f"Failed to check exploration need: {e}")
            return True
    
    def _needs_optimization(self) -> bool:
        """Check if swarm needs optimization."""
        try:
            # Check if agents are converging to similar solutions
            best_positions = [agent.best_position for agent in self.agents.values()]
            if len(best_positions) < 2:
                return False
            
            # Calculate best position variance
            best_positions_array = np.array(best_positions)
            variance = np.var(best_positions_array, axis=0).mean()
            
            # High variance suggests need for optimization
            return variance > 5.0
            
        except Exception as e:
            logger.warning(f"Failed to check optimization need: {e}")
            return False


class MultiAgentSwarmIntelligence:
    """Main swarm intelligence system."""
    
    def __init__(self, config: SwarmConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.swarm_intelligence")
        
        # Initialize components
        self.coordinator = SwarmCoordinator(config)
        self.agents = {}
        
        # Swarm state
        self.iteration = 0
        self.best_global_fitness = float('inf')
        self.convergence_history = []
        
    async def initialize_agents(self):
        """Initialize swarm agents."""
        try:
            self.logger.info("Initializing swarm agents...")
            
            # Create agents based on configuration
            agent_types = self._get_default_agent_types()
            
            agent_id = 0
            for agent_type in agent_types:
                for i in range(agent_type.count):
                    # Generate random position
                    position = np.random.randn(3) * 10.0  # 3D space
                    
                    # Create agent
                    agent = SwarmAgent(
                        agent_id=f"{agent_type.type}_{i}",
                        agent_type=agent_type,
                        position=position
                    )
                    
                    # Add to coordinator
                    self.coordinator.add_agent(agent)
                    self.agents[agent.agent_id] = agent
                    
                    agent_id += 1
            
            self.logger.info(f"Initialized {len(self.agents)} agents")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize agents: {e}")
            raise
    
    def _get_default_agent_types(self) -> List[AgentType]:
        """Get default agent type configuration."""
        return [
            AgentType("explorer", 3, "exploration", 0.02),
            AgentType("exploiter", 4, "exploitation", 0.01),
            AgentType("coordinator", 2, "coordination", 0.005),
            AgentType("specialist", 1, "specialization", 0.015)
        ]
    
    async def execute_collaborative_task(self, task_type: str, task_complexity: str, 
                                       collaboration_mode: str) -> Dict[str, Any]:
        """Execute a collaborative task using swarm intelligence."""
        try:
            self.logger.info(f"Executing collaborative task: {task_type} ({task_complexity})")
            
            # Configure task parameters
            task_config = self._configure_task(task_type, task_complexity)
            
            # Execute swarm optimization
            optimization_result = await self._run_swarm_optimization(task_config)
            
            # Analyze collaboration effectiveness
            collaboration_metrics = self._analyze_collaboration()
            
            return {
                "success": True,
                "task_type": task_type,
                "task_complexity": task_complexity,
                "collaboration_mode": collaboration_mode,
                "optimization_result": optimization_result,
                "collaboration_metrics": collaboration_metrics,
                "final_fitness": self.best_global_fitness,
                "iterations": self.iteration
            }
            
        except Exception as e:
            self.logger.error(f"Collaborative task failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _configure_task(self, task_type: str, task_complexity: str) -> Dict[str, Any]:
        """Configure task parameters."""
        complexity_factors = {
            "low": {"iterations": 100, "convergence_threshold": 1e-4},
            "medium": {"iterations": 500, "convergence_threshold": 1e-5},
            "high": {"iterations": 1000, "convergence_threshold": 1e-6}
        }
        
        config = complexity_factors.get(task_complexity, complexity_factors["medium"])
        config.update({
            "task_type": task_type,
            "task_complexity": task_complexity
        })
        
        return config
    
    async def _run_swarm_optimization(self, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run swarm optimization."""
        try:
            max_iterations = task_config.get("iterations", self.config.max_iterations)
            convergence_threshold = task_config.get("convergence_threshold", self.config.convergence_threshold)
            
            self.logger.info(f"Starting swarm optimization: {max_iterations} iterations")
            
            for iteration in tqdm(range(max_iterations), desc="Swarm Optimization"):
                # Coordinate swarm
                swarm_state = self.coordinator.coordinate_swarm()
                
                # Update all agents
                for agent in self.agents.values():
                    agent.update_position(
                        swarm_center=np.array(swarm_state['swarm_center']),
                        global_best=np.array(swarm_state['global_best'])
                    )
                    
                    # Adapt behavior
                    agent.adapt_behavior(swarm_state)
                
                # Check convergence
                if self._check_convergence(convergence_threshold):
                    self.logger.info(f"Swarm converged at iteration {iteration}")
                    break
                
                # Record convergence history
                self.convergence_history.append(swarm_state['best_fitness'])
                
                # Update iteration counter
                self.iteration = iteration + 1
                
                # Small delay for visualization
                await asyncio.sleep(0.01)
            
            # Final coordination
            final_state = self.coordinator.coordinate_swarm()
            self.best_global_fitness = final_state['best_fitness']
            
            return {
                "iterations_completed": self.iteration,
                "final_fitness": self.best_global_fitness,
                "convergence_history": self.convergence_history,
                "converged": self._check_convergence(convergence_threshold)
            }
            
        except Exception as e:
            self.logger.error(f"Swarm optimization failed: {e}")
            return {"error": str(e)}
    
    def _check_convergence(self, threshold: float) -> bool:
        """Check if swarm has converged."""
        try:
            if len(self.convergence_history) < 10:
                return False
            
            # Check if fitness has stabilized
            recent_fitness = self.convergence_history[-10:]
            fitness_variance = np.var(recent_fitness)
            
            return fitness_variance < threshold
            
        except Exception as e:
            logger.warning(f"Convergence check failed: {e}")
            return False
    
    def _analyze_collaboration(self) -> Dict[str, Any]:
        """Analyze collaboration effectiveness."""
        try:
            # Calculate communication density
            total_connections = sum(len(connections) for connections in self.coordinator.communication_graph.values())
            max_possible_connections = len(self.agents) * (len(self.agents) - 1)
            communication_density = total_connections / max_possible_connections if max_possible_connections > 0 else 0
            
            # Calculate agent specialization
            specialization_levels = [agent.specialization_level for agent in self.agents.values()]
            avg_specialization = np.mean(specialization_levels)
            
            # Calculate position diversity
            positions = [agent.position for agent in self.agents.values()]
            position_variance = np.var(np.array(positions), axis=0).mean()
            
            return {
                "communication_density": communication_density,
                "avg_specialization": avg_specialization,
                "position_diversity": position_variance,
                "total_agents": len(self.agents),
                "active_connections": total_connections
            }
            
        except Exception as e:
            logger.warning(f"Collaboration analysis failed: {e}")
            return {}
    
    def get_swarm_status(self) -> Dict[str, Any]:
        """Get current swarm status."""
        return {
            "total_agents": len(self.agents),
            "iteration": self.iteration,
            "best_global_fitness": self.best_global_fitness,
            "swarm_state": self.coordinator.swarm_state,
            "convergence_history": self.convergence_history,
            "agent_statuses": {agent_id: agent.get_status() for agent_id, agent in self.agents.items()}
        }


# Factory function for creating swarm intelligence systems
def create_swarm_intelligence(config: SwarmConfig) -> MultiAgentSwarmIntelligence:
    """Create a multi-agent swarm intelligence system."""
    return MultiAgentSwarmIntelligence(config)


# Example usage and testing
if __name__ == "__main__":
    # Test swarm intelligence system
    config = SwarmConfig(
        num_agents=10,
        collaboration_mode="hierarchical",
        enable_emergent_behavior=True,
        enable_adaptive_coordination=True
    )
    
    # Create swarm system
    swarm = create_swarm_intelligence(config)
    
    # Initialize agents
    asyncio.run(swarm.initialize_agents())
    
    # Execute collaborative task
    result = asyncio.run(swarm.execute_collaborative_task(
        task_type="optimization",
        task_complexity="medium",
        collaboration_mode="hierarchical"
    ))
    
    print(f"Task result: {result}")
    
    # Get swarm status
    status = swarm.get_swarm_status()
    print(f"Swarm status: {status}")
