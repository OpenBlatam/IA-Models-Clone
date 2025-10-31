"""
Swarm Intelligence Module

This module provides advanced swarm intelligence capabilities including:
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

from .swarm_intelligence_system import (
    SwarmIntelligenceManager,
    SwarmAgent,
    SwarmOptimization,
    CollectiveDecisionMaker,
    DistributedProblemSolver,
    EmergentBehaviorSimulator,
    AntColonyOptimizer,
    ParticleSwarmOptimizer,
    BeeColonyAlgorithm,
    FlockingBehavior,
    SwarmRoboticsCoordinator,
    get_swarm_intelligence_manager,
    initialize_swarm_intelligence,
    shutdown_swarm_intelligence
)

__all__ = [
    "SwarmIntelligenceManager",
    "SwarmAgent",
    "SwarmOptimization",
    "CollectiveDecisionMaker",
    "DistributedProblemSolver",
    "EmergentBehaviorSimulator",
    "AntColonyOptimizer",
    "ParticleSwarmOptimizer",
    "BeeColonyAlgorithm",
    "FlockingBehavior",
    "SwarmRoboticsCoordinator",
    "get_swarm_intelligence_manager",
    "initialize_swarm_intelligence",
    "shutdown_swarm_intelligence"
]





















