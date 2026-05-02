"""
A Study of AI Population Dynamics
==================================

Paper: "A Study of AI Population Dynamics with"

Key concepts:
- Population of AI agents
- Evolution and adaptation
- Agent interactions
- Population-level behaviors
- Emergent properties
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import random

from ..common.agent_base import BaseAgent, AgentState, AgentStatus


class AgentRole(Enum):
    """Agent roles in population."""
    EXPLORER = "explorer"
    EXPLOITER = "exploiter"
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"


@dataclass
class PopulationMetrics:
    """Metrics for agent population."""
    total_agents: int
    active_agents: int
    average_fitness: float
    diversity: float
    cooperation_rate: float
    competition_rate: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AgentInteraction:
    """Interaction between agents."""
    interaction_id: str
    agent1_id: str
    agent2_id: str
    interaction_type: str  # cooperation, competition, communication
    outcome: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


class PopulationDynamicsSystem:
    """
    System for studying AI population dynamics.
    
    Manages population of agents and their interactions.
    """
    
    def __init__(
        self,
        initial_population_size: int = 10,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize population dynamics system.
        
        Args:
            initial_population_size: Initial number of agents
            config: Configuration parameters
        """
        self.config = config or {}
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_roles: Dict[str, AgentRole] = {}
        self.agent_fitness: Dict[str, float] = {}
        
        # Interactions
        self.interactions: List[AgentInteraction] = []
        
        # Population history
        self.population_history: List[PopulationMetrics] = []
        
        # Evolution parameters
        self.mutation_rate = config.get("mutation_rate", 0.1)
        self.selection_pressure = config.get("selection_pressure", 0.5)
        self.cooperation_probability = config.get("cooperation_probability", 0.6)
        
        # Initialize population
        self._initialize_population(initial_population_size)
    
    def _initialize_population(self, size: int):
        """Initialize population with agents."""
        roles = list(AgentRole)
        
        for i in range(size):
            agent_id = f"agent_{i}"
            role = random.choice(roles)
            
            # Create agent (simplified)
            agent = BaseAgent(
                name=agent_id,
                config={"role": role.value}
            )
            
            self.agents[agent_id] = agent
            self.agent_roles[agent_id] = role
            self.agent_fitness[agent_id] = random.uniform(0.5, 1.0)
    
    def step(self) -> PopulationMetrics:
        """
        Execute one step of population dynamics.
        
        Returns:
            Current population metrics
        """
        # Agents interact
        self._agent_interactions()
        
        # Update fitness based on interactions
        self._update_fitness()
        
        # Evolution: selection and mutation
        self._evolve_population()
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        self.population_history.append(metrics)
        
        return metrics
    
    def _agent_interactions(self):
        """Simulate agent interactions."""
        agent_ids = list(self.agents.keys())
        
        # Random pairs interact
        num_interactions = len(agent_ids) // 2
        random.shuffle(agent_ids)
        
        for i in range(0, len(agent_ids) - 1, 2):
            agent1_id = agent_ids[i]
            agent2_id = agent_ids[i + 1]
            
            interaction = self._create_interaction(agent1_id, agent2_id)
            self.interactions.append(interaction)
    
    def _create_interaction(
        self,
        agent1_id: str,
        agent2_id: str
    ) -> AgentInteraction:
        """Create interaction between two agents."""
        # Determine interaction type
        if random.random() < self.cooperation_probability:
            interaction_type = "cooperation"
            outcome = self._cooperation_outcome(agent1_id, agent2_id)
        else:
            interaction_type = "competition"
            outcome = self._competition_outcome(agent1_id, agent2_id)
        
        return AgentInteraction(
            interaction_id=f"interaction_{datetime.now().timestamp()}",
            agent1_id=agent1_id,
            agent2_id=agent2_id,
            interaction_type=interaction_type,
            outcome=outcome
        )
    
    def _cooperation_outcome(
        self,
        agent1_id: str,
        agent2_id: str
    ) -> Dict[str, Any]:
        """Outcome of cooperation."""
        # Both agents benefit from cooperation
        benefit = random.uniform(0.1, 0.3)
        
        return {
            "type": "cooperation",
            "agent1_benefit": benefit,
            "agent2_benefit": benefit,
            "success": True
        }
    
    def _competition_outcome(
        self,
        agent1_id: str,
        agent2_id: str
    ) -> Dict[str, Any]:
        """Outcome of competition."""
        # Winner takes all
        fitness1 = self.agent_fitness.get(agent1_id, 0.5)
        fitness2 = self.agent_fitness.get(agent2_id, 0.5)
        
        if fitness1 > fitness2:
            winner = agent1_id
            loser = agent2_id
        else:
            winner = agent2_id
            loser = agent1_id
        
        return {
            "type": "competition",
            "winner": winner,
            "loser": loser,
            "winner_benefit": 0.2,
            "loser_cost": -0.1
        }
    
    def _update_fitness(self):
        """Update agent fitness based on interactions."""
        # Process recent interactions
        recent_interactions = self.interactions[-len(self.agents):]
        
        for interaction in recent_interactions:
            outcome = interaction.outcome
            
            if outcome["type"] == "cooperation":
                # Both benefit
                self.agent_fitness[interaction.agent1_id] += outcome["agent1_benefit"]
                self.agent_fitness[interaction.agent2_id] += outcome["agent2_benefit"]
            
            elif outcome["type"] == "competition":
                # Winner benefits, loser loses
                winner = outcome["winner"]
                loser = outcome["loser"]
                
                self.agent_fitness[winner] += outcome["winner_benefit"]
                self.agent_fitness[loser] += outcome["loser_cost"]
            
            # Clamp fitness
            for agent_id in self.agent_fitness:
                self.agent_fitness[agent_id] = max(0.0, min(1.0, self.agent_fitness[agent_id]))
    
    def _evolve_population(self):
        """Evolve population through selection and mutation."""
        # Selection: remove low-fitness agents
        fitness_threshold = self.selection_pressure
        
        agents_to_remove = [
            agent_id for agent_id, fitness in self.agent_fitness.items()
            if fitness < fitness_threshold
        ]
        
        for agent_id in agents_to_remove:
            if agent_id in self.agents:
                del self.agents[agent_id]
                del self.agent_roles[agent_id]
                del self.agent_fitness[agent_id]
        
        # Reproduction: create new agents from high-fitness ones
        high_fitness_agents = [
            agent_id for agent_id, fitness in self.agent_fitness.items()
            if fitness > 0.7
        ]
        
        if high_fitness_agents and len(self.agents) < 20:  # Max population
            parent = random.choice(high_fitness_agents)
            self._reproduce_agent(parent)
    
    def _reproduce_agent(self, parent_id: str):
        """Create new agent from parent."""
        new_agent_id = f"agent_{len(self.agents)}"
        
        # Inherit role from parent
        parent_role = self.agent_roles.get(parent_id, AgentRole.EXPLORER)
        
        # Create new agent
        agent = BaseAgent(
            name=new_agent_id,
            config={"role": parent_role.value}
        )
        
        self.agents[new_agent_id] = agent
        self.agent_roles[new_agent_id] = parent_role
        
        # Inherit fitness with mutation
        parent_fitness = self.agent_fitness.get(parent_id, 0.5)
        mutation = random.uniform(-self.mutation_rate, self.mutation_rate)
        self.agent_fitness[new_agent_id] = max(0.0, min(1.0, parent_fitness + mutation))
    
    def _calculate_metrics(self) -> PopulationMetrics:
        """Calculate population metrics."""
        if not self.agents:
            return PopulationMetrics(
                total_agents=0,
                active_agents=0,
                average_fitness=0.0,
                diversity=0.0,
                cooperation_rate=0.0,
                competition_rate=0.0
            )
        
        # Calculate metrics
        total_agents = len(self.agents)
        active_agents = sum(
            1 for agent in self.agents.values()
            if agent.state.status != AgentStatus.COMPLETED
        )
        
        fitnesses = list(self.agent_fitness.values())
        average_fitness = sum(fitnesses) / len(fitnesses) if fitnesses else 0.0
        
        # Diversity: measure of role distribution
        role_counts = {}
        for role in self.agent_roles.values():
            role_counts[role] = role_counts.get(role, 0) + 1
        
        diversity = len(role_counts) / len(AgentRole) if AgentRole else 0.0
        
        # Cooperation/competition rates
        recent_interactions = self.interactions[-20:] if len(self.interactions) > 20 else self.interactions
        if recent_interactions:
            cooperation_count = sum(
                1 for i in recent_interactions
                if i.interaction_type == "cooperation"
            )
            competition_count = sum(
                1 for i in recent_interactions
                if i.interaction_type == "competition"
            )
            total = len(recent_interactions)
            
            cooperation_rate = cooperation_count / total if total > 0 else 0.0
            competition_rate = competition_count / total if total > 0 else 0.0
        else:
            cooperation_rate = 0.0
            competition_rate = 0.0
        
        return PopulationMetrics(
            total_agents=total_agents,
            active_agents=active_agents,
            average_fitness=average_fitness,
            diversity=diversity,
            cooperation_rate=cooperation_rate,
            competition_rate=competition_rate
        )
    
    def get_population_history(self) -> List[PopulationMetrics]:
        """Get population history."""
        return self.population_history
    
    def get_agent_fitness(self, agent_id: str) -> float:
        """Get fitness of an agent."""
        return self.agent_fitness.get(agent_id, 0.0)
    
    def get_population_size(self) -> int:
        """Get current population size."""
        return len(self.agents)



