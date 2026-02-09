"""
Optimizaciones de Algoritmos Evolutivos para Routing.

Este módulo implementa algoritmos genéticos y evolutivos para
optimización de rutas mediante evolución de soluciones.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import random

logger = logging.getLogger(__name__)


class EvolutionaryAlgorithm(Enum):
    """Algoritmos evolutivos."""
    GENETIC_ALGORITHM = "ga"
    DIFFERENTIAL_EVOLUTION = "de"
    EVOLUTIONARY_STRATEGY = "es"
    GENETIC_PROGRAMMING = "gp"


@dataclass
class Individual:
    """Individuo en algoritmo evolutivo."""
    genes: List[Any]
    fitness: float = float('inf')
    age: int = 0


class GeneticAlgorithm:
    """Algoritmo genético."""
    
    def __init__(self, population_size: int = 50, mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8, elitism: float = 0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.population: List[Individual] = []
        self.generation = 0
        self.best_individual: Optional[Individual] = None
        self.fitness_history: List[float] = []
    
    def initialize_population(self, num_genes: int, gene_range: Tuple[int, int]):
        """Inicializar población."""
        self.population = []
        for _ in range(self.population_size):
            genes = [random.randint(gene_range[0], gene_range[1]) 
                    for _ in range(num_genes)]
            individual = Individual(genes=genes)
            self.population.append(individual)
    
    def evaluate(self, individual: Individual, fitness_func: Callable) -> float:
        """Evaluar fitness."""
        individual.fitness = fitness_func(individual.genes)
        return individual.fitness
    
    def select(self) -> Individual:
        """Selección por torneo."""
        tournament_size = 3
        tournament = random.sample(self.population, tournament_size)
        return min(tournament, key=lambda x: x.fitness)
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Crossover de un punto."""
        if random.random() > self.crossover_rate:
            return parent1, parent2
        
        point = random.randint(1, len(parent1.genes) - 1)
        
        child1_genes = parent1.genes[:point] + parent2.genes[point:]
        child2_genes = parent2.genes[:point] + parent1.genes[point:]
        
        return Individual(genes=child1_genes), Individual(genes=child2_genes)
    
    def mutate(self, individual: Individual, gene_range: Tuple[int, int]):
        """Mutación."""
        for i in range(len(individual.genes)):
            if random.random() < self.mutation_rate:
                individual.genes[i] = random.randint(gene_range[0], gene_range[1])
    
    def evolve(self, fitness_func: Callable, num_generations: int = 100) -> Individual:
        """Evolucionar población."""
        for generation in range(num_generations):
            # Evaluar
            for individual in self.population:
                self.evaluate(individual, fitness_func)
            
            # Ordenar por fitness
            self.population.sort(key=lambda x: x.fitness)
            
            # Elitismo
            elite_count = int(self.population_size * self.elitism)
            elite = self.population[:elite_count]
            
            # Nueva población
            new_population = elite.copy()
            
            # Generar descendencia
            while len(new_population) < self.population_size:
                parent1 = self.select()
                parent2 = self.select()
                
                child1, child2 = self.crossover(parent1, parent2)
                
                self.mutate(child1, (0, len(parent1.genes) - 1))
                self.mutate(child2, (0, len(parent1.genes) - 1))
                
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            self.population = new_population[:self.population_size]
            self.generation += 1
            
            # Track best
            best = min(self.population, key=lambda x: x.fitness)
            if self.best_individual is None or best.fitness < self.best_individual.fitness:
                self.best_individual = Individual(
                    genes=best.genes.copy(),
                    fitness=best.fitness
                )
            
            self.fitness_history.append(self.best_individual.fitness)
        
        return self.best_individual
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            "algorithm": "genetic_algorithm",
            "population_size": self.population_size,
            "generation": self.generation,
            "best_fitness": self.best_individual.fitness if self.best_individual else float('inf'),
            "avg_fitness": np.mean([ind.fitness for ind in self.population]) if self.population else 0.0
        }


class DifferentialEvolution:
    """Evolución diferencial."""
    
    def __init__(self, population_size: int = 50, F: float = 0.5, CR: float = 0.7):
        self.population_size = population_size
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability
        self.population: List[np.ndarray] = []
        self.fitness: List[float] = []
        self.generation = 0
        self.best_solution: Optional[np.ndarray] = None
        self.best_fitness = float('inf')
    
    def initialize(self, num_dimensions: int, bounds: Tuple[float, float]):
        """Inicializar población."""
        self.population = []
        for _ in range(self.population_size):
            individual = np.random.uniform(bounds[0], bounds[1], num_dimensions)
            self.population.append(individual)
        self.fitness = [float('inf')] * self.population_size
    
    def evolve(self, fitness_func: Callable, max_generations: int = 100) -> np.ndarray:
        """Evolucionar."""
        for generation in range(max_generations):
            for i, target in enumerate(self.population):
                # Seleccionar 3 individuos diferentes
                candidates = [j for j in range(self.population_size) if j != i]
                a, b, c = random.sample(candidates, 3)
                
                # Mutación
                mutant = self.population[a] + self.F * (self.population[b] - self.population[c])
                
                # Crossover
                trial = target.copy()
                j_rand = random.randint(0, len(target) - 1)
                for j in range(len(target)):
                    if random.random() < self.CR or j == j_rand:
                        trial[j] = mutant[j]
                
                # Evaluar
                trial_fitness = fitness_func(trial)
                
                # Selección
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    
                    if trial_fitness < self.best_fitness:
                        self.best_fitness = trial_fitness
                        self.best_solution = trial.copy()
            
            self.generation += 1
        
        return self.best_solution
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            "algorithm": "differential_evolution",
            "population_size": self.population_size,
            "generation": self.generation,
            "best_fitness": self.best_fitness
        }


class EvolutionaryOptimizer:
    """Optimizador principal evolutivo."""
    
    def __init__(self, algorithm: EvolutionaryAlgorithm = EvolutionaryAlgorithm.GENETIC_ALGORITHM,
                 enable_evolutionary: bool = True):
        self.enable_evolutionary = enable_evolutionary
        self.algorithm = algorithm
        
        if algorithm == EvolutionaryAlgorithm.GENETIC_ALGORITHM:
            self.optimizer = GeneticAlgorithm()
        elif algorithm == EvolutionaryAlgorithm.DIFFERENTIAL_EVOLUTION:
            self.optimizer = DifferentialEvolution()
        else:
            self.optimizer = GeneticAlgorithm()
        
        self.routes_optimized = 0
    
    def optimize_route(self, nodes: List[Dict[str, Any]], 
                      fitness_func: Optional[Callable] = None) -> Optional[List[int]]:
        """Optimizar ruta usando algoritmos evolutivos."""
        if not self.enable_evolutionary:
            return None
        
        try:
            # Definir función de fitness si no se proporciona
            if fitness_func is None:
                def default_fitness(route: List[int]) -> float:
                    total_cost = 0.0
                    for i in range(len(route) - 1):
                        from_node = nodes[route[i]]
                        to_node = nodes[route[i + 1]]
                        pos_from = from_node.get('position', {})
                        pos_to = to_node.get('position', {})
                        dist = np.sqrt(
                            sum((pos_from.get(k, 0) - pos_to.get(k, 0)) ** 2 
                                for k in ['x', 'y', 'z'])
                        )
                        total_cost += dist
                    return total_cost
                fitness_func = default_fitness
            
            if isinstance(self.optimizer, GeneticAlgorithm):
                # Inicializar población
                self.optimizer.initialize_population(
                    num_genes=len(nodes),
                    gene_range=(0, len(nodes) - 1)
                )
                
                # Evolucionar
                best = self.optimizer.evolve(fitness_func)
                route = best.genes
            else:
                # Differential Evolution
                self.optimizer.initialize(
                    num_dimensions=len(nodes),
                    bounds=(0, len(nodes) - 1)
                )
                best = self.optimizer.evolve(fitness_func)
                route = best.astype(int).tolist()
            
            self.routes_optimized += 1
            return route
        except Exception as e:
            logger.warning(f"Evolutionary optimization failed: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        if not self.enable_evolutionary:
            return {
                "evolutionary_enabled": False
            }
        
        stats = self.optimizer.get_stats()
        stats["evolutionary_enabled"] = True
        stats["algorithm"] = self.algorithm.value
        stats["routes_optimized"] = self.routes_optimized
        
        return stats


