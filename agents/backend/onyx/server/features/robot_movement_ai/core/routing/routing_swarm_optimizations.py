"""
Optimizaciones de Inteligencia de Enjambre para Routing.

Este módulo implementa algoritmos de optimización basados en enjambres
inspirados en comportamientos colectivos de animales.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class SwarmAlgorithm(Enum):
    """Algoritmos de enjambre."""
    PARTICLE_SWARM = "pso"
    ANT_COLONY = "aco"
    BEE_COLONY = "abc"
    FIREFLY = "firefly"
    BAT_ALGORITHM = "bat"


@dataclass
class Particle:
    """Partícula en PSO."""
    position: np.ndarray
    velocity: np.ndarray
    best_position: np.ndarray
    best_fitness: float = float('inf')
    fitness: float = float('inf')


@dataclass
class Ant:
    """Hormiga en ACO."""
    path: List[int]
    pheromone_trail: List[Tuple[int, int]]
    fitness: float = float('inf')
    visited: List[int] = field(default_factory=list)


class ParticleSwarmOptimizer:
    """Optimizador de enjambre de partículas (PSO)."""
    
    def __init__(self, num_particles: int = 30, num_dimensions: int = 10):
        self.num_particles = num_particles
        self.num_dimensions = num_dimensions
        self.particles: List[Particle] = []
        self.global_best_position: Optional[np.ndarray] = None
        self.global_best_fitness = float('inf')
        self.w = 0.7  # Inertia weight
        self.c1 = 1.5  # Cognitive coefficient
        self.c2 = 1.5  # Social coefficient
        self.iterations = 0
        
        # Inicializar partículas
        for _ in range(num_particles):
            position = np.random.uniform(-10, 10, num_dimensions)
            velocity = np.random.uniform(-1, 1, num_dimensions)
            particle = Particle(
                position=position,
                velocity=velocity,
                best_position=position.copy()
            )
            self.particles.append(particle)
    
    def optimize(self, cost_function: callable, max_iterations: int = 100) -> np.ndarray:
        """Optimizar usando PSO."""
        for iteration in range(max_iterations):
            # Evaluar fitness
            for particle in self.particles:
                particle.fitness = cost_function(particle.position)
                
                # Update personal best
                if particle.fitness < particle.best_fitness:
                    particle.best_fitness = particle.fitness
                    particle.best_position = particle.position.copy()
                
                # Update global best
                if particle.fitness < self.global_best_fitness:
                    self.global_best_fitness = particle.fitness
                    self.global_best_position = particle.position.copy()
            
            # Update velocities and positions
            for particle in self.particles:
                r1 = np.random.random(self.num_dimensions)
                r2 = np.random.random(self.num_dimensions)
                
                # Velocity update
                particle.velocity = (
                    self.w * particle.velocity +
                    self.c1 * r1 * (particle.best_position - particle.position) +
                    self.c2 * r2 * (self.global_best_position - particle.position)
                )
                
                # Position update
                particle.position += particle.velocity
            
            self.iterations += 1
        
        return self.global_best_position
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            "algorithm": "pso",
            "num_particles": self.num_particles,
            "iterations": self.iterations,
            "best_fitness": self.global_best_fitness
        }


class AntColonyOptimizer:
    """Optimizador de colonia de hormigas (ACO)."""
    
    def __init__(self, num_ants: int = 20, alpha: float = 1.0, beta: float = 2.0,
                 evaporation_rate: float = 0.1, q: float = 100.0):
        self.num_ants = num_ants
        self.alpha = alpha  # Pheromone importance
        self.beta = beta  # Heuristic importance
        self.evaporation_rate = evaporation_rate
        self.q = q  # Pheromone deposit constant
        self.pheromone_matrix: Dict[Tuple[int, int], float] = {}
        self.heuristic_matrix: Dict[Tuple[int, int], float] = {}
        self.ants: List[Ant] = []
        self.best_path: List[int] = []
        self.best_fitness = float('inf')
        self.iterations = 0
    
    def initialize_pheromones(self, nodes: List[Dict[str, Any]]):
        """Inicializar matriz de feromonas."""
        n = len(nodes)
        initial_pheromone = 1.0 / n
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    self.pheromone_matrix[(i, j)] = initial_pheromone
                    
                    # Calcular heurística (distancia inversa)
                    pos_i = nodes[i].get('position', {})
                    pos_j = nodes[j].get('position', {})
                    dist = np.sqrt(
                        sum((pos_i.get(k, 0) - pos_j.get(k, 0)) ** 2 
                            for k in ['x', 'y', 'z'])
                    )
                    self.heuristic_matrix[(i, j)] = 1.0 / max(dist, 0.001)
    
    def construct_solution(self, start_node: int, nodes: List[Dict[str, Any]]) -> List[int]:
        """Construir solución usando hormigas."""
        path = [start_node]
        visited = {start_node}
        current = start_node
        
        while len(path) < len(nodes):
            # Calcular probabilidades
            probabilities = []
            candidates = []
            
            for j in range(len(nodes)):
                if j not in visited:
                    pheromone = self.pheromone_matrix.get((current, j), 0.001)
                    heuristic = self.heuristic_matrix.get((current, j), 0.001)
                    
                    prob = (pheromone ** self.alpha) * (heuristic ** self.beta)
                    probabilities.append(prob)
                    candidates.append(j)
            
            if not candidates:
                break
            
            # Normalizar probabilidades
            total = sum(probabilities)
            probabilities = [p / total for p in probabilities]
            
            # Seleccionar siguiente nodo
            next_node = np.random.choice(candidates, p=probabilities)
            path.append(next_node)
            visited.add(next_node)
            current = next_node
        
        return path
    
    def update_pheromones(self, paths: List[List[int]], fitnesses: List[float]):
        """Actualizar feromonas."""
        # Evaporación
        for key in self.pheromone_matrix:
            self.pheromone_matrix[key] *= (1.0 - self.evaporation_rate)
        
        # Depositar feromonas
        for path, fitness in zip(paths, fitnesses):
            if fitness > 0:
                pheromone_deposit = self.q / fitness
                
                for i in range(len(path) - 1):
                    edge = (path[i], path[i + 1])
                    if edge in self.pheromone_matrix:
                        self.pheromone_matrix[edge] += pheromone_deposit
    
    def optimize(self, nodes: List[Dict[str, Any]], start_node: int = 0,
                max_iterations: int = 50) -> List[int]:
        """Optimizar usando ACO."""
        self.initialize_pheromones(nodes)
        
        for iteration in range(max_iterations):
            paths = []
            fitnesses = []
            
            # Construir soluciones
            for _ in range(self.num_ants):
                path = self.construct_solution(start_node, nodes)
                paths.append(path)
                
                # Calcular fitness (distancia total)
                fitness = 0.0
                for i in range(len(path) - 1):
                    edge = (path[i], path[i + 1])
                    if edge in self.heuristic_matrix:
                        fitness += 1.0 / self.heuristic_matrix[edge]
                
                fitnesses.append(fitness)
                
                # Update best
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_path = path.copy()
            
            # Actualizar feromonas
            self.update_pheromones(paths, fitnesses)
            self.iterations += 1
        
        return self.best_path
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            "algorithm": "aco",
            "num_ants": self.num_ants,
            "iterations": self.iterations,
            "best_fitness": self.best_fitness,
            "pheromone_edges": len(self.pheromone_matrix)
        }


class SwarmOptimizer:
    """Optimizador principal de enjambre."""
    
    def __init__(self, algorithm: SwarmAlgorithm = SwarmAlgorithm.PARTICLE_SWARM,
                 enable_swarm: bool = True):
        self.enable_swarm = enable_swarm
        self.algorithm = algorithm
        
        if algorithm == SwarmAlgorithm.PARTICLE_SWARM:
            self.optimizer = ParticleSwarmOptimizer()
        elif algorithm == SwarmAlgorithm.ANT_COLONY:
            self.optimizer = AntColonyOptimizer()
        else:
            self.optimizer = ParticleSwarmOptimizer()
        
        self.routes_optimized = 0
        self.improvement_rate = 0.0
    
    def optimize_route(self, nodes: List[Dict[str, Any]], 
                      start_node: int = 0) -> Optional[List[int]]:
        """Optimizar ruta usando inteligencia de enjambre."""
        if not self.enable_swarm:
            return None
        
        try:
            if isinstance(self.optimizer, AntColonyOptimizer):
                route = self.optimizer.optimize(nodes, start_node)
            else:
                # Para PSO, convertir a problema de optimización continua
                def cost_function(position: np.ndarray) -> float:
                    # Simplificado: convertir posición a ruta y calcular costo
                    route_indices = np.argsort(position)
                    total_cost = 0.0
                    for i in range(len(route_indices) - 1):
                        from_node = route_indices[i]
                        to_node = route_indices[i + 1]
                        pos_from = nodes[from_node].get('position', {})
                        pos_to = nodes[to_node].get('position', {})
                        dist = np.sqrt(
                            sum((pos_from.get(k, 0) - pos_to.get(k, 0)) ** 2 
                                for k in ['x', 'y', 'z'])
                        )
                        total_cost += dist
                    return total_cost
                
                best_position = self.optimizer.optimize(cost_function)
                route = np.argsort(best_position).tolist()
            
            self.routes_optimized += 1
            return route
        except Exception as e:
            logger.warning(f"Swarm optimization failed: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        if not self.enable_swarm:
            return {
                "swarm_enabled": False
            }
        
        stats = self.optimizer.get_stats()
        stats["swarm_enabled"] = True
        stats["algorithm"] = self.algorithm.value
        stats["routes_optimized"] = self.routes_optimized
        
        return stats


