"""
Swarm Intelligence Engine for Export IA
Advanced swarm-based optimization and collective intelligence for document processing
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio
from collections import defaultdict
import random
import math

# Advanced libraries
try:
    import pyswarm
    import deap
    from deap import base, creator, tools, algorithms
    import networkx as nx
    from scipy.optimize import differential_evolution
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.metrics import silhouette_score
except ImportError:
    print("Installing required libraries...")
    import subprocess
    subprocess.check_call(["pip", "install", "pyswarm", "deap", "networkx", "scipy", "scikit-learn"])

logger = logging.getLogger(__name__)

@dataclass
class SwarmConfig:
    """Configuration for swarm intelligence algorithms"""
    population_size: int = 100
    max_iterations: int = 1000
    convergence_threshold: float = 1e-6
    inertia_weight: float = 0.9
    cognitive_weight: float = 2.0
    social_weight: float = 2.0
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elitism_rate: float = 0.1
    diversity_threshold: float = 0.1
    communication_radius: float = 1.0
    pheromone_decay: float = 0.1
    pheromone_evaporation: float = 0.05

class Particle:
    """Particle for Particle Swarm Optimization"""
    
    def __init__(self, position: np.ndarray, velocity: np.ndarray, bounds: Tuple[float, float]):
        self.position = position.copy()
        self.velocity = velocity.copy()
        self.best_position = position.copy()
        self.best_fitness = float('inf')
        self.fitness = float('inf')
        self.bounds = bounds
        self.age = 0
        self.energy = 1.0
        self.memory = []
        self.social_network = set()
        
    def update_velocity(self, global_best: np.ndarray, config: SwarmConfig, r1: float, r2: float):
        """Update particle velocity using PSO formula"""
        cognitive_component = config.cognitive_weight * r1 * (self.best_position - self.position)
        social_component = config.social_weight * r2 * (global_best - self.position)
        
        self.velocity = (config.inertia_weight * self.velocity + 
                        cognitive_component + 
                        social_component)
        
        # Apply velocity limits
        max_velocity = 0.1 * (self.bounds[1] - self.bounds[0])
        self.velocity = np.clip(self.velocity, -max_velocity, max_velocity)
        
    def update_position(self):
        """Update particle position"""
        self.position += self.velocity
        self.position = np.clip(self.position, self.bounds[0], self.bounds[1])
        self.age += 1
        
    def update_energy(self, fitness_improvement: float):
        """Update particle energy based on performance"""
        if fitness_improvement > 0:
            self.energy = min(1.0, self.energy + 0.1)
        else:
            self.energy = max(0.1, self.energy - 0.05)
            
    def add_to_memory(self, position: np.ndarray, fitness: float):
        """Add experience to particle memory"""
        self.memory.append((position.copy(), fitness))
        if len(self.memory) > 10:  # Limit memory size
            self.memory.pop(0)

class Ant:
    """Ant for Ant Colony Optimization"""
    
    def __init__(self, start_node: int, graph: nx.Graph):
        self.current_node = start_node
        self.visited_nodes = [start_node]
        self.path = []
        self.pheromone_deposited = 0.0
        self.fitness = 0.0
        self.memory = set()
        self.exploration_rate = 0.1
        self.graph = graph
        
    def select_next_node(self, pheromone_matrix: np.ndarray, alpha: float = 1.0, beta: float = 2.0):
        """Select next node using ACO probability formula"""
        current = self.current_node
        unvisited = [node for node in self.graph.neighbors(current) if node not in self.visited_nodes]
        
        if not unvisited:
            return None
            
        probabilities = []
        for node in unvisited:
            pheromone = pheromone_matrix[current][node]
            distance = 1.0 / self.graph[current][node].get('weight', 1.0)
            probability = (pheromone ** alpha) * (distance ** beta)
            probabilities.append(probability)
            
        # Roulette wheel selection
        total_prob = sum(probabilities)
        if total_prob == 0:
            return random.choice(unvisited)
            
        probabilities = [p / total_prob for p in probabilities]
        return np.random.choice(unvisited, p=probabilities)
        
    def move_to_node(self, node: int):
        """Move ant to next node"""
        if node is not None:
            self.path.append((self.current_node, node))
            self.visited_nodes.append(node)
            self.current_node = node
            
    def deposit_pheromone(self, pheromone_matrix: np.ndarray, quality: float):
        """Deposit pheromone on path"""
        for from_node, to_node in self.path:
            pheromone_matrix[from_node][to_node] += quality
            pheromone_matrix[to_node][from_node] += quality  # Undirected graph

class Bee:
    """Bee for Artificial Bee Colony Optimization"""
    
    def __init__(self, position: np.ndarray, bounds: Tuple[float, float]):
        self.position = position.copy()
        self.fitness = float('inf')
        self.trial_count = 0
        self.max_trials = 10
        self.bounds = bounds
        self.dance_strength = 0.0
        self.foraging_radius = 1.0
        self.memory = []
        
    def explore_neighborhood(self, config: SwarmConfig) -> np.ndarray:
        """Explore neighborhood around current position"""
        perturbation = np.random.uniform(-self.foraging_radius, self.foraging_radius, self.position.shape)
        new_position = self.position + perturbation
        new_position = np.clip(new_position, self.bounds[0], self.bounds[1])
        return new_position
        
    def perform_waggle_dance(self, fitness: float) -> float:
        """Perform waggle dance to communicate food source quality"""
        self.dance_strength = min(1.0, fitness / 100.0)  # Normalize dance strength
        return self.dance_strength
        
    def follow_dance(self, target_position: np.ndarray, dance_strength: float) -> np.ndarray:
        """Follow waggle dance to target position"""
        attraction_strength = dance_strength * 0.5
        direction = target_position - self.position
        new_position = self.position + attraction_strength * direction
        new_position = np.clip(new_position, self.bounds[0], self.bounds[1])
        return new_position

class SwarmIntelligenceEngine:
    """Advanced Swarm Intelligence Engine for document processing optimization"""
    
    def __init__(self, config: SwarmConfig):
        self.config = config
        self.particles = []
        self.ants = []
        self.bees = []
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.convergence_history = []
        self.diversity_history = []
        self.performance_metrics = defaultdict(list)
        self.swarm_topology = nx.Graph()
        self.pheromone_matrix = None
        self.iteration = 0
        
        # Advanced features
        self.adaptive_parameters = True
        self.multi_objective = False
        self.dynamic_topology = True
        self.cultural_evolution = True
        self.swarm_memory = []
        self.leader_particles = []
        
    def initialize_pso_swarm(self, problem_dimension: int, bounds: Tuple[float, float]):
        """Initialize Particle Swarm Optimization swarm"""
        self.particles = []
        for _ in range(self.config.population_size):
            position = np.random.uniform(bounds[0], bounds[1], problem_dimension)
            velocity = np.random.uniform(-1, 1, problem_dimension)
            particle = Particle(position, velocity, bounds)
            self.particles.append(particle)
            
        # Initialize swarm topology
        self._create_swarm_topology()
        
    def initialize_aco_swarm(self, graph: nx.Graph):
        """Initialize Ant Colony Optimization swarm"""
        self.ants = []
        self.pheromone_matrix = np.ones((len(graph.nodes), len(graph.nodes)))
        
        for _ in range(self.config.population_size):
            start_node = random.choice(list(graph.nodes))
            ant = Ant(start_node, graph)
            self.ants.append(ant)
            
    def initialize_abc_swarm(self, problem_dimension: int, bounds: Tuple[float, float]):
        """Initialize Artificial Bee Colony swarm"""
        self.bees = []
        for _ in range(self.config.population_size):
            position = np.random.uniform(bounds[0], bounds[1], problem_dimension)
            bee = Bee(position, bounds)
            self.bees.append(bee)
            
    def _create_swarm_topology(self):
        """Create dynamic swarm topology"""
        if self.dynamic_topology:
            # Create small-world network
            self.swarm_topology = nx.watts_strogatz_graph(
                len(self.particles), 
                k=4, 
                p=0.3
            )
        else:
            # Create fully connected topology
            self.swarm_topology = nx.complete_graph(len(self.particles))
            
    def optimize_with_pso(self, objective_function, max_iterations: int = None) -> Dict[str, Any]:
        """Optimize using Particle Swarm Optimization"""
        if max_iterations is None:
            max_iterations = self.config.max_iterations
            
        self.iteration = 0
        convergence_count = 0
        
        while self.iteration < max_iterations and convergence_count < 10:
            # Evaluate fitness
            for particle in self.particles:
                particle.fitness = objective_function(particle.position)
                
                if particle.fitness < particle.best_fitness:
                    particle.best_fitness = particle.fitness
                    particle.best_position = particle.position.copy()
                    
                if particle.fitness < self.global_best_fitness:
                    self.global_best_fitness = particle.fitness
                    self.global_best_position = particle.position.copy()
                    
            # Update velocities and positions
            for i, particle in enumerate(self.particles):
                # Get neighborhood best (local topology)
                neighbors = list(self.swarm_topology.neighbors(i))
                if neighbors:
                    local_best = min(neighbors, key=lambda x: self.particles[x].best_fitness)
                    local_best_position = self.particles[local_best].best_position
                else:
                    local_best_position = self.global_best_position
                    
                r1, r2 = np.random.random(2)
                particle.update_velocity(local_best_position, self.config, r1, r2)
                particle.update_position()
                
                # Update energy and memory
                fitness_improvement = particle.best_fitness - particle.fitness
                particle.update_energy(fitness_improvement)
                particle.add_to_memory(particle.position, particle.fitness)
                
            # Adaptive parameter adjustment
            if self.adaptive_parameters:
                self._adapt_parameters()
                
            # Update topology
            if self.dynamic_topology and self.iteration % 50 == 0:
                self._update_topology()
                
            # Track convergence
            self._track_convergence()
            
            self.iteration += 1
            
        return {
            'best_position': self.global_best_position,
            'best_fitness': self.global_best_fitness,
            'iterations': self.iteration,
            'convergence_history': self.convergence_history,
            'diversity_history': self.diversity_history
        }
        
    def optimize_with_aco(self, graph: nx.Graph, objective_function, max_iterations: int = None) -> Dict[str, Any]:
        """Optimize using Ant Colony Optimization"""
        if max_iterations is None:
            max_iterations = self.config.max_iterations
            
        self.iteration = 0
        best_path = None
        best_fitness = float('inf')
        
        while self.iteration < max_iterations:
            # Reset ants
            for ant in self.ants:
                ant.visited_nodes = [ant.current_node]
                ant.path = []
                
            # Construct solutions
            for ant in self.ants:
                while True:
                    next_node = ant.select_next_node(self.pheromone_matrix)
                    if next_node is None:
                        break
                    ant.move_to_node(next_node)
                    
                # Evaluate solution
                if ant.path:
                    ant.fitness = objective_function(ant.path)
                    if ant.fitness < best_fitness:
                        best_fitness = ant.fitness
                        best_path = ant.path.copy()
                        
            # Update pheromones
            for ant in self.ants:
                if ant.path:
                    quality = 1.0 / (1.0 + ant.fitness)
                    ant.deposit_pheromone(self.pheromone_matrix, quality)
                    
            # Evaporate pheromones
            self.pheromone_matrix *= (1.0 - self.config.pheromone_evaporation)
            
            self.iteration += 1
            
        return {
            'best_path': best_path,
            'best_fitness': best_fitness,
            'iterations': self.iteration,
            'pheromone_matrix': self.pheromone_matrix
        }
        
    def optimize_with_abc(self, objective_function, max_iterations: int = None) -> Dict[str, Any]:
        """Optimize using Artificial Bee Colony"""
        if max_iterations is None:
            max_iterations = self.config.max_iterations
            
        self.iteration = 0
        best_position = None
        best_fitness = float('inf')
        
        while self.iteration < max_iterations:
            # Employed bees phase
            for bee in self.bees:
                new_position = bee.explore_neighborhood(self.config)
                new_fitness = objective_function(new_position)
                
                if new_fitness < bee.fitness:
                    bee.position = new_position
                    bee.fitness = new_fitness
                    bee.trial_count = 0
                else:
                    bee.trial_count += 1
                    
                if bee.fitness < best_fitness:
                    best_fitness = bee.fitness
                    best_position = bee.position.copy()
                    
            # Onlooker bees phase
            fitness_sum = sum(1.0 / (1.0 + bee.fitness) for bee in self.bees)
            for bee in self.bees:
                probability = (1.0 / (1.0 + bee.fitness)) / fitness_sum
                if np.random.random() < probability:
                    # Follow waggle dance
                    target_bee = max(self.bees, key=lambda b: b.dance_strength)
                    new_position = bee.follow_dance(target_bee.position, target_bee.dance_strength)
                    new_fitness = objective_function(new_position)
                    
                    if new_fitness < bee.fitness:
                        bee.position = new_position
                        bee.fitness = new_fitness
                        bee.trial_count = 0
                        
            # Scout bees phase
            for bee in self.bees:
                if bee.trial_count >= bee.max_trials:
                    # Abandon food source and search for new one
                    bee.position = np.random.uniform(bee.bounds[0], bee.bounds[1], bee.position.shape)
                    bee.fitness = objective_function(bee.position)
                    bee.trial_count = 0
                    
            # Update dance strengths
            for bee in self.bees:
                bee.perform_waggle_dance(bee.fitness)
                
            self.iteration += 1
            
        return {
            'best_position': best_position,
            'best_fitness': best_fitness,
            'iterations': self.iteration
        }
        
    def _adapt_parameters(self):
        """Adaptively adjust algorithm parameters"""
        if len(self.convergence_history) < 10:
            return
            
        recent_improvement = (self.convergence_history[-10] - self.convergence_history[-1]) / 10
        
        if recent_improvement < 0.01:  # Slow convergence
            self.config.inertia_weight *= 0.99
            self.config.cognitive_weight *= 1.01
            self.config.social_weight *= 1.01
        else:  # Good convergence
            self.config.inertia_weight = min(0.9, self.config.inertia_weight * 1.01)
            
    def _update_topology(self):
        """Update swarm topology based on performance"""
        # Identify leader particles
        sorted_particles = sorted(enumerate(self.particles), key=lambda x: x[1].fitness)
        self.leader_particles = [i for i, _ in sorted_particles[:5]]
        
        # Create new connections to leaders
        for i, particle in enumerate(self.particles):
            if i not in self.leader_particles:
                # Connect to random leader
                leader = random.choice(self.leader_particles)
                if not self.swarm_topology.has_edge(i, leader):
                    self.swarm_topology.add_edge(i, leader)
                    
    def _track_convergence(self):
        """Track convergence and diversity metrics"""
        # Calculate convergence
        current_best = min(particle.fitness for particle in self.particles)
        self.convergence_history.append(current_best)
        
        # Calculate diversity
        positions = np.array([particle.position for particle in self.particles])
        diversity = np.mean(np.std(positions, axis=0))
        self.diversity_history.append(diversity)
        
    def hybrid_optimization(self, objective_function, problem_type: str = "continuous") -> Dict[str, Any]:
        """Hybrid optimization combining multiple swarm algorithms"""
        results = {}
        
        if problem_type == "continuous":
            # Use PSO + ABC hybrid
            pso_result = self.optimize_with_pso(objective_function, max_iterations=500)
            abc_result = self.optimize_with_abc(objective_function, max_iterations=500)
            
            results = {
                'pso': pso_result,
                'abc': abc_result,
                'best_overall': min(pso_result['best_fitness'], abc_result['best_fitness'])
            }
            
        elif problem_type == "discrete":
            # Use ACO for discrete problems
            # Create a simple graph for demonstration
            graph = nx.complete_graph(10)
            aco_result = self.optimize_with_aco(graph, objective_function, max_iterations=500)
            results = {'aco': aco_result}
            
        return results
        
    def multi_objective_optimization(self, objectives: List[callable], weights: List[float] = None) -> Dict[str, Any]:
        """Multi-objective optimization using weighted sum approach"""
        if weights is None:
            weights = [1.0] * len(objectives)
            
        def combined_objective(x):
            total = 0.0
            for obj, weight in zip(objectives, weights):
                total += weight * obj(x)
            return total
            
        return self.optimize_with_pso(combined_objective)
        
    def get_swarm_statistics(self) -> Dict[str, Any]:
        """Get comprehensive swarm statistics"""
        if not self.particles:
            return {}
            
        positions = np.array([particle.position for particle in self.particles])
        fitnesses = [particle.fitness for particle in self.particles]
        
        return {
            'population_size': len(self.particles),
            'best_fitness': min(fitnesses),
            'worst_fitness': max(fitnesses),
            'average_fitness': np.mean(fitnesses),
            'fitness_std': np.std(fitnesses),
            'position_centroid': np.mean(positions, axis=0),
            'position_std': np.std(positions, axis=0),
            'diversity': np.mean(np.std(positions, axis=0)),
            'convergence_rate': self._calculate_convergence_rate(),
            'topology_connectivity': nx.density(self.swarm_topology)
        }
        
    def _calculate_convergence_rate(self) -> float:
        """Calculate convergence rate"""
        if len(self.convergence_history) < 10:
            return 0.0
            
        recent_history = self.convergence_history[-10:]
        improvements = [recent_history[i] - recent_history[i+1] for i in range(len(recent_history)-1)]
        return np.mean(improvements) if improvements else 0.0

class SwarmDocumentProcessor:
    """Document processor using swarm intelligence for optimization"""
    
    def __init__(self, config: SwarmConfig):
        self.swarm_engine = SwarmIntelligenceEngine(config)
        self.document_features = {}
        self.optimization_history = []
        
    def optimize_document_layout(self, document_content: str, target_format: str) -> Dict[str, Any]:
        """Optimize document layout using swarm intelligence"""
        
        def layout_objective(params):
            # Parameters: [font_size, line_spacing, margin_width, column_count]
            font_size, line_spacing, margin_width, column_count = params
            
            # Simulate layout quality based on parameters
            readability_score = self._calculate_readability(font_size, line_spacing)
            space_efficiency = self._calculate_space_efficiency(margin_width, column_count)
            aesthetic_score = self._calculate_aesthetic_score(params)
            
            # Minimize negative score (maximize quality)
            return -(readability_score + space_efficiency + aesthetic_score)
            
        # Initialize swarm for layout optimization
        self.swarm_engine.initialize_pso_swarm(
            problem_dimension=4,
            bounds=(0.1, 10.0)
        )
        
        result = self.swarm_engine.optimize_with_pso(layout_objective)
        
        optimized_params = {
            'font_size': result['best_position'][0],
            'line_spacing': result['best_position'][1],
            'margin_width': result['best_position'][2],
            'column_count': int(result['best_position'][3])
        }
        
        return {
            'optimized_parameters': optimized_params,
            'quality_score': -result['best_fitness'],
            'optimization_result': result
        }
        
    def optimize_content_structure(self, content_sections: List[str]) -> Dict[str, Any]:
        """Optimize content structure using ant colony optimization"""
        
        # Create graph representing content relationships
        graph = nx.Graph()
        for i, section in enumerate(content_sections):
            graph.add_node(i, content=section)
            
        # Add edges based on content similarity
        for i in range(len(content_sections)):
            for j in range(i+1, len(content_sections)):
                similarity = self._calculate_content_similarity(content_sections[i], content_sections[j])
                if similarity > 0.3:  # Threshold for connection
                    graph.add_edge(i, j, weight=1.0/similarity)
                    
        def structure_objective(path):
            if not path:
                return float('inf')
                
            # Calculate structure quality based on path
            flow_score = self._calculate_content_flow(path, content_sections)
            coherence_score = self._calculate_coherence(path, content_sections)
            
            return -(flow_score + coherence_score)
            
        # Initialize ACO for structure optimization
        self.swarm_engine.initialize_aco_swarm(graph)
        result = self.swarm_engine.optimize_with_aco(graph, structure_objective)
        
        return {
            'optimal_structure': result['best_path'],
            'structure_quality': -result['best_fitness'],
            'optimization_result': result
        }
        
    def optimize_export_parameters(self, document_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize export parameters using hybrid swarm intelligence"""
        
        def export_objective(params):
            # Parameters: [compression_level, image_quality, font_embedding, metadata_inclusion]
            compression, img_quality, font_embed, metadata = params
            
            # Calculate export quality metrics
            file_size_score = self._calculate_file_size_score(compression, img_quality)
            quality_score = self._calculate_export_quality_score(params)
            compatibility_score = self._calculate_compatibility_score(font_embed, metadata)
            
            return -(file_size_score + quality_score + compatibility_score)
            
        # Use hybrid optimization
        result = self.swarm_engine.hybrid_optimization(export_objective, "continuous")
        
        best_params = result['best_overall']
        optimized_export = {
            'compression_level': best_params[0],
            'image_quality': best_params[1],
            'font_embedding': best_params[2] > 0.5,
            'metadata_inclusion': best_params[3] > 0.5
        }
        
        return {
            'optimized_export': optimized_export,
            'export_quality': -result['best_overall'],
            'optimization_results': result
        }
        
    def _calculate_readability(self, font_size: float, line_spacing: float) -> float:
        """Calculate readability score"""
        # Optimal font size around 12pt, line spacing around 1.5
        font_score = 1.0 - abs(font_size - 12.0) / 12.0
        spacing_score = 1.0 - abs(line_spacing - 1.5) / 1.5
        return (font_score + spacing_score) / 2.0
        
    def _calculate_space_efficiency(self, margin_width: float, column_count: float) -> float:
        """Calculate space efficiency score"""
        # Optimal margin around 1 inch, column count around 2
        margin_score = 1.0 - abs(margin_width - 1.0) / 1.0
        column_score = 1.0 - abs(column_count - 2.0) / 2.0
        return (margin_score + column_score) / 2.0
        
    def _calculate_aesthetic_score(self, params: np.ndarray) -> float:
        """Calculate aesthetic score based on golden ratio principles"""
        # Use golden ratio for aesthetic proportions
        golden_ratio = 1.618
        font_size, line_spacing, margin_width, column_count = params
        
        # Check if proportions follow golden ratio
        ratio_score = 1.0 - abs((font_size * line_spacing) / margin_width - golden_ratio) / golden_ratio
        return max(0.0, ratio_score)
        
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between content sections"""
        # Simple word overlap similarity
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
        
    def _calculate_content_flow(self, path: List[Tuple[int, int]], sections: List[str]) -> float:
        """Calculate content flow score based on path"""
        if not path:
            return 0.0
            
        flow_score = 0.0
        for from_idx, to_idx in path:
            similarity = self._calculate_content_similarity(sections[from_idx], sections[to_idx])
            flow_score += similarity
            
        return flow_score / len(path) if path else 0.0
        
    def _calculate_coherence(self, path: List[Tuple[int, int]], sections: List[str]) -> float:
        """Calculate content coherence score"""
        if not path:
            return 0.0
            
        # Check for logical progression
        coherence_score = 0.0
        for i in range(len(path) - 1):
            current_section = sections[path[i][1]]
            next_section = sections[path[i+1][0]]
            
            # Simple coherence based on topic continuity
            topic_words = ['introduction', 'method', 'result', 'conclusion', 'analysis']
            current_topics = [word for word in topic_words if word in current_section.lower()]
            next_topics = [word for word in topic_words if word in next_section.lower()]
            
            if current_topics and next_topics:
                coherence_score += 0.5
                
        return coherence_score / max(1, len(path) - 1)
        
    def _calculate_file_size_score(self, compression: float, img_quality: float) -> float:
        """Calculate file size optimization score"""
        # Balance between compression and quality
        size_score = compression  # Higher compression = smaller file
        quality_penalty = (1.0 - img_quality) * 0.5  # Penalty for low quality
        
        return size_score - quality_penalty
        
    def _calculate_export_quality_score(self, params: np.ndarray) -> float:
        """Calculate overall export quality score"""
        compression, img_quality, font_embed, metadata = params
        
        # Weighted quality score
        quality = (img_quality * 0.4 + 
                  font_embed * 0.3 + 
                  metadata * 0.2 + 
                  (1.0 - compression) * 0.1)  # Less compression = higher quality
        
        return quality
        
    def _calculate_compatibility_score(self, font_embed: float, metadata: float) -> float:
        """Calculate compatibility score"""
        # Font embedding and metadata improve compatibility
        return (font_embed + metadata) / 2.0

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize swarm intelligence engine
    config = SwarmConfig(
        population_size=50,
        max_iterations=100,
        convergence_threshold=1e-6
    )
    
    swarm_engine = SwarmIntelligenceEngine(config)
    doc_processor = SwarmDocumentProcessor(config)
    
    # Test PSO optimization
    def test_objective(x):
        # Rosenbrock function
        return sum(100.0 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1))
    
    print("Testing PSO optimization...")
    swarm_engine.initialize_pso_swarm(problem_dimension=2, bounds=(-2.0, 2.0))
    pso_result = swarm_engine.optimize_with_pso(test_objective, max_iterations=50)
    print(f"PSO Result: {pso_result['best_fitness']:.6f} at {pso_result['best_position']}")
    
    # Test document layout optimization
    print("\nTesting document layout optimization...")
    layout_result = doc_processor.optimize_document_layout(
        "Sample document content for layout optimization testing.",
        "PDF"
    )
    print(f"Optimized layout parameters: {layout_result['optimized_parameters']}")
    print(f"Layout quality score: {layout_result['quality_score']:.4f}")
    
    # Test content structure optimization
    print("\nTesting content structure optimization...")
    content_sections = [
        "Introduction to the topic with background information.",
        "Methodology and approach used in the study.",
        "Results and findings from the analysis.",
        "Conclusion and future work recommendations."
    ]
    structure_result = doc_processor.optimize_content_structure(content_sections)
    print(f"Optimal content structure: {structure_result['optimal_structure']}")
    print(f"Structure quality: {structure_result['structure_quality']:.4f}")
    
    # Test export parameter optimization
    print("\nTesting export parameter optimization...")
    document_data = {
        'content': 'Sample document',
        'images': ['image1.jpg', 'image2.png'],
        'fonts': ['Arial', 'Times New Roman']
    }
    export_result = doc_processor.optimize_export_parameters(document_data)
    print(f"Optimized export parameters: {export_result['optimized_export']}")
    print(f"Export quality: {export_result['export_quality']:.4f}")
    
    # Get swarm statistics
    stats = swarm_engine.get_swarm_statistics()
    print(f"\nSwarm Statistics:")
    print(f"Population size: {stats.get('population_size', 0)}")
    print(f"Best fitness: {stats.get('best_fitness', 0):.6f}")
    print(f"Average fitness: {stats.get('average_fitness', 0):.6f}")
    print(f"Diversity: {stats.get('diversity', 0):.6f}")
    print(f"Convergence rate: {stats.get('convergence_rate', 0):.6f}")
    
    print("\nSwarm Intelligence Engine initialized successfully!")
























