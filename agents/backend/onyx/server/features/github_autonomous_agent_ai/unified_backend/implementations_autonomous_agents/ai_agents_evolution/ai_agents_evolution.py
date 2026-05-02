"""
AI Agents: Evolution, Architecture, and
========================================

Paper: "AI Agents: Evolution, Architecture, and"

Key concepts:
- Evolution of AI agents
- Agent architecture patterns
- Evolutionary algorithms for agents
- Architecture optimization
- Adaptive agent design
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import random

from ..common.agent_base import BaseAgent, AgentState, AgentStatus


class ArchitecturePattern(Enum):
    """Agent architecture patterns."""
    REACTIVE = "reactive"
    DELIBERATIVE = "deliberative"
    HYBRID = "hybrid"
    LAYERED = "layered"
    MODULAR = "modular"


class EvolutionOperator(Enum):
    """Evolution operators."""
    MUTATION = "mutation"
    CROSSOVER = "crossover"
    SELECTION = "selection"
    ELITISM = "elitism"


@dataclass
class AgentArchitecture:
    """Agent architecture specification."""
    architecture_id: str
    pattern: ArchitecturePattern
    components: List[str]
    parameters: Dict[str, Any]
    fitness: float = 0.0
    generation: int = 0
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class EvolutionGeneration:
    """A generation in the evolution process."""
    generation_id: int
    architectures: List[AgentArchitecture]
    best_fitness: float
    average_fitness: float
    timestamp: datetime = field(default_factory=datetime.now)


class AIAgentsEvolution:
    """
    System for evolving AI agent architectures.
    
    Uses evolutionary algorithms to optimize agent designs.
    """
    
    def __init__(
        self,
        initial_population_size: int = 10,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize evolution system.
        
        Args:
            initial_population_size: Initial number of architectures
            config: Configuration parameters
        """
        self.config = config or {}
        self.population: List[AgentArchitecture] = []
        self.generations: List[EvolutionGeneration] = []
        
        # Evolution parameters
        self.mutation_rate = config.get("mutation_rate", 0.1)
        self.crossover_rate = config.get("crossover_rate", 0.7)
        self.elitism_rate = config.get("elitism_rate", 0.2)
        self.population_size = initial_population_size
        
        # Initialize population
        self._initialize_population()
    
    def _initialize_population(self):
        """Initialize initial population of architectures."""
        patterns = list(ArchitecturePattern)
        
        for i in range(self.population_size):
            architecture = AgentArchitecture(
                architecture_id=f"arch_{i}",
                pattern=random.choice(patterns),
                components=self._generate_components(),
                parameters=self._generate_parameters(),
                generation=0
            )
            self.population.append(architecture)
    
    def _generate_components(self) -> List[str]:
        """Generate random components for architecture."""
        all_components = [
            "reasoning", "memory", "planning", "learning",
            "perception", "action", "communication", "coordination"
        ]
        num_components = random.randint(3, 6)
        return random.sample(all_components, num_components)
    
    def _generate_parameters(self) -> Dict[str, Any]:
        """Generate random parameters for architecture."""
        return {
            "complexity": random.uniform(0.1, 1.0),
            "efficiency": random.uniform(0.1, 1.0),
            "adaptability": random.uniform(0.1, 1.0)
        }
    
    def evolve(self, num_generations: int = 10) -> List[EvolutionGeneration]:
        """
        Evolve agent architectures.
        
        Args:
            num_generations: Number of generations to evolve
            
        Returns:
            List of evolution generations
        """
        for generation in range(num_generations):
            # Evaluate fitness
            self._evaluate_fitness()
            
            # Create new generation
            new_population = self._create_new_generation()
            
            # Record generation
            gen = EvolutionGeneration(
                generation_id=generation + 1,
                architectures=new_population,
                best_fitness=max(a.fitness for a in new_population),
                average_fitness=sum(a.fitness for a in new_population) / len(new_population)
            )
            self.generations.append(gen)
            self.population = new_population
        
        return self.generations
    
    def _evaluate_fitness(self):
        """Evaluate fitness of architectures."""
        for architecture in self.population:
            # Fitness based on parameters and components
            complexity_score = architecture.parameters.get("complexity", 0.5)
            efficiency_score = architecture.parameters.get("efficiency", 0.5)
            adaptability_score = architecture.parameters.get("adaptability", 0.5)
            component_score = len(architecture.components) / 8.0  # Normalize
            
            # Weighted fitness
            architecture.fitness = (
                complexity_score * 0.2 +
                efficiency_score * 0.3 +
                adaptability_score * 0.3 +
                component_score * 0.2
            )
    
    def _create_new_generation(self) -> List[AgentArchitecture]:
        """Create new generation through evolution."""
        new_population = []
        
        # Sort by fitness
        self.population.sort(key=lambda a: a.fitness, reverse=True)
        
        # Elitism: keep best individuals
        num_elites = int(self.population_size * self.elitism_rate)
        elites = self.population[:num_elites]
        new_population.extend(elites)
        
        # Generate rest through crossover and mutation
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate and len(self.population) >= 2:
                # Crossover
                parent1 = self._select_parent()
                parent2 = self._select_parent()
                child = self._crossover(parent1, parent2)
            else:
                # Mutation
                parent = self._select_parent()
                child = self._mutate(parent)
            
            child.generation = self.population[0].generation + 1
            new_population.append(child)
        
        return new_population[:self.population_size]
    
    def _select_parent(self) -> AgentArchitecture:
        """Select parent using fitness-proportional selection."""
        # Tournament selection
        tournament_size = 3
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda a: a.fitness)
    
    def _crossover(
        self,
        parent1: AgentArchitecture,
        parent2: AgentArchitecture
    ) -> AgentArchitecture:
        """Crossover two architectures."""
        # Combine components
        all_components = list(set(parent1.components + parent2.components))
        num_components = random.randint(
            min(len(parent1.components), len(parent2.components)),
            max(len(parent1.components), len(parent2.components))
        )
        child_components = random.sample(all_components, min(num_components, len(all_components)))
        
        # Average parameters
        child_parameters = {
            key: (parent1.parameters.get(key, 0.5) + parent2.parameters.get(key, 0.5)) / 2
            for key in set(list(parent1.parameters.keys()) + list(parent2.parameters.keys()))
        }
        
        # Choose pattern from parents
        child_pattern = random.choice([parent1.pattern, parent2.pattern])
        
        return AgentArchitecture(
            architecture_id=f"arch_{datetime.now().timestamp()}",
            pattern=child_pattern,
            components=child_components,
            parameters=child_parameters,
            generation=parent1.generation
        )
    
    def _mutate(self, architecture: AgentArchitecture) -> AgentArchitecture:
        """Mutate an architecture."""
        # Mutate components
        components = architecture.components.copy()
        if random.random() < self.mutation_rate:
            if random.random() < 0.5 and components:
                # Remove component
                components.remove(random.choice(components))
            else:
                # Add component
                all_components = [
                    "reasoning", "memory", "planning", "learning",
                    "perception", "action", "communication", "coordination"
                ]
                available = [c for c in all_components if c not in components]
                if available:
                    components.append(random.choice(available))
        
        # Mutate parameters
        parameters = architecture.parameters.copy()
        for key in parameters:
            if random.random() < self.mutation_rate:
                parameters[key] = max(0.1, min(1.0, parameters[key] + random.uniform(-0.2, 0.2)))
        
        # Mutate pattern
        pattern = architecture.pattern
        if random.random() < self.mutation_rate:
            patterns = list(ArchitecturePattern)
            patterns.remove(pattern)
            if patterns:
                pattern = random.choice(patterns)
        
        return AgentArchitecture(
            architecture_id=f"arch_{datetime.now().timestamp()}",
            pattern=pattern,
            components=components,
            parameters=parameters,
            generation=architecture.generation
        )
    
    def get_best_architecture(self) -> Optional[AgentArchitecture]:
        """Get best architecture from current population."""
        if not self.population:
            return None
        
        self._evaluate_fitness()
        return max(self.population, key=lambda a: a.fitness)
    
    def get_evolution_history(self) -> List[EvolutionGeneration]:
        """Get evolution history."""
        return self.generations
    
    def get_population_statistics(self) -> Dict[str, Any]:
        """Get population statistics."""
        if not self.population:
            return {}
        
        self._evaluate_fitness()
        
        fitnesses = [a.fitness for a in self.population]
        pattern_counts = {}
        for architecture in self.population:
            pattern_counts[architecture.pattern] = pattern_counts.get(architecture.pattern, 0) + 1
        
        return {
            "population_size": len(self.population),
            "average_fitness": sum(fitnesses) / len(fitnesses),
            "best_fitness": max(fitnesses),
            "worst_fitness": min(fitnesses),
            "pattern_distribution": {p.value: c for p, c in pattern_counts.items()},
            "current_generation": self.population[0].generation if self.population else 0
        }



