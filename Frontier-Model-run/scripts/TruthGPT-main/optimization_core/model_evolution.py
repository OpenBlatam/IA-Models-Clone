"""
Advanced Model Evolution System for TruthGPT Optimization Core
Genetic algorithms, evolutionary strategies, and neuro-evolution
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

import logging
logger = logging.getLogger(__name__)


class EvolutionStrategy(Enum):
    GENETIC_ALGORITHM = "genetic_algorithm"
    CMA_ES = "cma_es"
    SIMPLE_ES = "simple_es"
    NEURO_EVOLUTION = "neuro_evolution"


@dataclass
class EvolutionConfig:
    strategy: EvolutionStrategy = EvolutionStrategy.GENETIC_ALGORITHM

    # population
    population_size: int = 50
    elite_fraction: float = 0.1
    mutation_rate: float = 0.1
    mutation_scale: float = 0.1
    crossover_rate: float = 0.7

    # evolution
    generations: int = 50
    tournament_size: int = 3

    # ES
    sigma_init: float = 0.5
    sigma_decay: float = 0.99

    # search space
    chromosome_length: int = 32


class Individual:
    def __init__(self, genes: np.ndarray):
        self.genes = genes.astype(np.float32)
        self.fitness: Optional[float] = None

    def clone(self) -> "Individual":
        c = Individual(self.genes.copy())
        c.fitness = self.fitness
        return c


def _random_genes(length: int) -> np.ndarray:
    return np.random.uniform(-1.0, 1.0, size=(length,)).astype(np.float32)


def _evaluate(ind: Individual, objective: Callable[[np.ndarray], float]) -> float:
    if ind.fitness is None:
        ind.fitness = float(objective(ind.genes))
    return ind.fitness


class GeneticAlgorithm:
    def __init__(self, cfg: EvolutionConfig, objective: Callable[[np.ndarray], float]):
        self.cfg = cfg
        self.objective = objective
        self.population: List[Individual] = [Individual(_random_genes(cfg.chromosome_length)) for _ in range(cfg.population_size)]

    def _select_parent(self) -> Individual:
        k = self.cfg.tournament_size
        contenders = random.sample(self.population, k)
        contenders.sort(key=lambda i: _evaluate(i, self.objective), reverse=True)
        return contenders[0]

    def _crossover(self, a: Individual, b: Individual) -> Tuple[Individual, Individual]:
        if random.random() > self.cfg.crossover_rate:
            return a.clone(), b.clone()
        point = random.randint(1, len(a.genes) - 2)
        child1 = np.concatenate([a.genes[:point], b.genes[point:]])
        child2 = np.concatenate([b.genes[:point], a.genes[point:]])
        return Individual(child1), Individual(child2)

    def _mutate(self, ind: Individual) -> None:
        mask = np.random.rand(len(ind.genes)) < self.cfg.mutation_rate
        noise = np.random.normal(0.0, self.cfg.mutation_scale, size=ind.genes.shape).astype(ind.genes.dtype)
        ind.genes[mask] += noise[mask]

    def evolve(self, generations: Optional[int] = None) -> Dict[str, Any]:
        gens = generations or self.cfg.generations
        history: List[Dict[str, Any]] = []

        for g in range(gens):
            # evaluate
            for ind in self.population:
                _evaluate(ind, self.objective)

            self.population.sort(key=lambda i: i.fitness or -1e9, reverse=True)
            elite_count = max(1, int(self.cfg.elite_fraction * len(self.population)))
            elites = [self.population[i].clone() for i in range(elite_count)]

            best = elites[0]
            avg_fitness = float(np.mean([i.fitness for i in self.population if i.fitness is not None]))
            history.append({"generation": g, "best_fitness": best.fitness, "avg_fitness": avg_fitness})

            # produce offspring
            next_population: List[Individual] = elites
            while len(next_population) < self.cfg.population_size:
                p1, p2 = self._select_parent(), self._select_parent()
                c1, c2 = self._crossover(p1, p2)
                self._mutate(c1)
                if len(next_population) < self.cfg.population_size:
                    next_population.append(c1)
                self._mutate(c2)
                if len(next_population) < self.cfg.population_size:
                    next_population.append(c2)

            self.population = next_population

        self.population.sort(key=lambda i: i.fitness or -1e9, reverse=True)
        return {"best": self.population[0], "history": history}


class SimpleES:
    def __init__(self, cfg: EvolutionConfig, objective: Callable[[np.ndarray], float]):
        self.cfg = cfg
        self.objective = objective
        self.center = _random_genes(cfg.chromosome_length)
        self.sigma = cfg.sigma_init

    def step(self) -> Tuple[np.ndarray, float]:
        noises = np.random.normal(0.0, self.sigma, size=(self.cfg.population_size, len(self.center))).astype(np.float32)
        samples = self.center + noises
        rewards = np.asarray([self.objective(s) for s in samples], dtype=np.float32)
        norm_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        grad = (noises.T @ norm_rewards) / self.cfg.population_size
        lr = 0.1
        self.center = self.center + lr * grad
        self.sigma *= self.cfg.sigma_decay
        return self.center.copy(), float(rewards.max())

    def optimize(self, generations: Optional[int] = None) -> Dict[str, Any]:
        gens = generations or self.cfg.generations
        history: List[Dict[str, Any]] = []
        best_reward = -1e9
        best_params = self.center.copy()
        for g in range(gens):
            params, reward = self.step()
            history.append({"generation": g, "best_reward": reward, "sigma": self.sigma})
            if reward > best_reward:
                best_reward = reward
                best_params = params.copy()
        return {"best_params": best_params, "best_reward": best_reward, "history": history}


class NeuroEvolution:
    """Evolve simple feedforward networks parameter vectors using GA."""

    def __init__(self, cfg: EvolutionConfig, layer_sizes: List[int], objective_fn: Callable[[Callable[[np.ndarray], np.ndarray]], float]):
        self.cfg = cfg
        self.layer_sizes = layer_sizes
        self.objective_fn = objective_fn
        self.shapes = [(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]
        self.num_params = int(sum(s[0] * s[1] + s[1] for s in self.shapes))
        self.ga = GeneticAlgorithm(EvolutionConfig(
            strategy=EvolutionStrategy.GENETIC_ALGORITHM,
            population_size=cfg.population_size,
            elite_fraction=cfg.elite_fraction,
            mutation_rate=cfg.mutation_rate,
            mutation_scale=cfg.mutation_scale,
            crossover_rate=cfg.crossover_rate,
            generations=cfg.generations,
            tournament_size=cfg.tournament_size,
            chromosome_length=self.num_params,
        ), self._wrapped_objective)

    def _params_to_network(self, params: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        idx = 0
        weights: List[np.ndarray] = []
        biases: List[np.ndarray] = []
        for in_dim, out_dim in self.shapes:
            w_size = in_dim * out_dim
            W = params[idx: idx + w_size].reshape(in_dim, out_dim)
            idx += w_size
            b = params[idx: idx + out_dim]
            idx += out_dim
            weights.append(W.astype(np.float32))
            biases.append(b.astype(np.float32))

        def net(x: np.ndarray) -> np.ndarray:
            h = x.astype(np.float32)
            for i in range(len(weights)):
                h = h @ weights[i] + biases[i]
                if i < len(weights) - 1:
                    h = np.maximum(h, 0)  # ReLU
            return h

        return net

    def _wrapped_objective(self, genes: np.ndarray) -> float:
        net = self._params_to_network(genes)
        score = float(self.objective_fn(net))
        return score

    def evolve(self) -> Dict[str, Any]:
        result = self.ga.evolve()
        best: Individual = result["best"]
        net = self._params_to_network(best.genes)
        return {"network": net, "best_fitness": best.fitness, "history": result["history"]}


# Factories

def create_evolution_config(**kwargs: Any) -> EvolutionConfig:
    return EvolutionConfig(**kwargs)


def create_ga(cfg: EvolutionConfig, objective: Callable[[np.ndarray], float]) -> GeneticAlgorithm:
    return GeneticAlgorithm(cfg, objective)


def create_simple_es(cfg: EvolutionConfig, objective: Callable[[np.ndarray], float]) -> SimpleES:
    return SimpleES(cfg, objective)


def create_neuro_evolution(cfg: EvolutionConfig, layer_sizes: List[int], objective_fn: Callable[[Callable[[np.ndarray], np.ndarray]], float]) -> NeuroEvolution:
    return NeuroEvolution(cfg, layer_sizes, objective_fn)


# Example

def example_model_evolution() -> Dict[str, Any]:
    # Objective for GA: maximize negative Sphere function (equivalent to minimize sum(x^2))
    def ga_objective(x: np.ndarray) -> float:
        return -float(np.sum(x * x))

    cfg = create_evolution_config(population_size=40, generations=30, chromosome_length=24)
    ga = create_ga(cfg, ga_objective)
    ga_result = ga.evolve()

    # Simple ES on same objective
    es = create_simple_es(create_evolution_config(population_size=60, generations=25, sigma_init=0.6, chromosome_length=24), ga_objective)
    es_result = es.optimize()

    # Neuro-evolution: XOR-like objective sampled with synthetic data
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(200, 2)).astype(np.float32)
    y = (X[:, 0] ^ X[:, 1]).astype(np.float32).reshape(-1, 1)

    def neuro_obj(net: Callable[[np.ndarray], np.ndarray]) -> float:
        preds = net(X)
        preds = 1 / (1 + np.exp(-preds))
        loss = np.mean((preds - y) ** 2)
        return -float(loss)  # maximize negative loss

    ne_cfg = create_evolution_config(population_size=50, generations=25)
    ne = create_neuro_evolution(ne_cfg, [2, 8, 1], neuro_obj)
    ne_result = ne.evolve()

    print("✅ Model Evolution Example Complete!")
    print(f"GA best fitness: {ga_result['best'].fitness:.6f}")
    print(f"ES best reward: {es_result['best_reward']:.6f}")
    print(f"Neuro-evolution best fitness: {ne_result['best_fitness']:.6f}")

    return {
        "ga": ga_result,
        "es": es_result,
        "neuro": ne_result,
    }


__all__ = [
    "EvolutionStrategy",
    "EvolutionConfig",
    "Individual",
    "GeneticAlgorithm",
    "SimpleES",
    "NeuroEvolution",
    "create_evolution_config",
    "create_ga",
    "create_simple_es",
    "create_neuro_evolution",
    "example_model_evolution",
]


if __name__ == "__main__":
    example_model_evolution()
