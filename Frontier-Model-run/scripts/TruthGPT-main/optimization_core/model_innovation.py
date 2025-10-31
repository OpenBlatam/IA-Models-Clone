"""
Advanced Model Innovation System for TruthGPT Optimization Core
Novel architecture discovery, creative algorithm design, and breakthrough search
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Tuple, Callable, Optional

import numpy as np

import logging
logger = logging.getLogger(__name__)


class InnovationStrategy(Enum):
    NOVELTY_SEARCH = "novelty_search"
    QUALITY_DIVERSITY = "quality_diversity"
    PROGRAM_SYNTHESIS = "program_synthesis"
    GRAMMAR_EVOLUTION = "grammar_evolution"
    META_INNOVATION = "meta_innovation"


@dataclass
class InnovationConfig:
    strategy: InnovationStrategy = InnovationStrategy.QUALITY_DIVERSITY
    population_size: int = 64
    generations: int = 30
    elite_fraction: float = 0.1
    mutation_rate: float = 0.15
    crossover_rate: float = 0.6

    # QD grid (for MAP-Elites-like)
    qd_bins: Tuple[int, int] = (8, 8)

    # Novelty parameters
    novelty_k: int = 10
    novelty_weight: float = 0.4
    quality_weight: float = 0.6

    # Grammar / program synthesis
    max_depth: int = 6


# Simple building blocks for architecture DSL
PRIMITIVES = [
    "Conv3x3", "Conv5x5", "DWConv3x3", "MaxPool", "AvgPool", "BN", "ReLU", "GELU", "SE",
    "Skip", "Concat", "Attention", "MLP", "Dropout"
]


def random_architecture(max_blocks: int = 8) -> List[str]:
    blocks = random.randint(3, max_blocks)
    return [random.choice(PRIMITIVES) for _ in range(blocks)]


def mutate_architecture(arch: List[str], rate: float) -> List[str]:
    out = arch.copy()
    for i in range(len(out)):
        if random.random() < rate:
            out[i] = random.choice(PRIMITIVES)
    # occasional insert/delete
    if random.random() < rate and len(out) < 12:
        out.insert(random.randrange(0, len(out)), random.choice(PRIMITIVES))
    if random.random() < rate and len(out) > 3:
        del out[random.randrange(0, len(out))]
    return out


def crossover_architecture(a: List[str], b: List[str]) -> Tuple[List[str], List[str]]:
    if not a or not b:
        return a.copy(), b.copy()
    pa = random.randrange(1, len(a))
    pb = random.randrange(1, len(b))
    c1 = a[:pa] + b[pb:]
    c2 = b[:pb] + a[pa:]
    return c1, c2


def architecture_embedding(arch: List[str]) -> np.ndarray:
    vocab = {p: i for i, p in enumerate(PRIMITIVES)}
    idxs = np.array([vocab[x] for x in arch], dtype=np.int32)
    # simple positional hashing into fixed-size vector
    vec = np.zeros((len(PRIMITIVES),), dtype=np.float32)
    for j, t in enumerate(idxs):
        vec[t] += 1.0 / (1.0 + j)
    return vec / (np.linalg.norm(vec) + 1e-8)


def behavioral_descriptor(arch: List[str]) -> np.ndarray:
    # features: depth, conv_ratio, attn_presence, skip_presence, mlp_ratio
    depth = len(arch)
    conv_ratio = sum(1 for x in arch if "Conv" in x) / max(1, depth)
    attn = 1.0 if any("Attention" in x for x in arch) else 0.0
    skip = 1.0 if any("Skip" in x for x in arch) else 0.0
    mlp_ratio = sum(1 for x in arch if x == "MLP") / max(1, depth)
    return np.array([depth / 16.0, conv_ratio, attn, skip, mlp_ratio], dtype=np.float32)


def simulated_quality(arch: List[str]) -> float:
    # stand-in differentiable-free scorer combining heuristic priors
    depth = len(arch)
    has_bn = any(x == "BN" for x in arch)
    has_activation = any(x in ("ReLU", "GELU") for x in arch)
    has_pool = any("Pool" in x for x in arch)
    has_attention = any("Attention" in x for x in arch)
    conv_layers = sum(1 for x in arch if "Conv" in x)

    score = 0.0
    score += 0.15 * has_bn
    score += 0.15 * has_activation
    score += 0.10 * has_pool
    score += 0.20 * has_attention
    score += 0.25 * np.tanh((conv_layers - 3) / 4.0)
    score += 0.15 * np.tanh((8 - abs(depth - 8)) / 5.0)  # prefer medium depth
    return float(max(0.0, min(1.0, score)))


class InnovationPopulation:
    def __init__(self, cfg: InnovationConfig):
        self.cfg = cfg
        self.population: List[List[str]] = [random_architecture() for _ in range(cfg.population_size)]
        self.archive: List[np.ndarray] = []  # embeddings for novelty
        self.qd_grid: Dict[Tuple[int, int], Tuple[List[str], float]] = {}

    def _novelty(self, emb: np.ndarray) -> float:
        if not self.archive:
            return 1.0
        D = np.stack(self.archive, axis=0)
        dists = np.linalg.norm(D - emb, axis=1)
        k = min(self.cfg.novelty_k, len(dists))
        return float(np.mean(np.partition(dists, k - 1)[:k]))

    def _qd_coords(self, bd: np.ndarray) -> Tuple[int, int]:
        bx, by = self.cfg.qd_bins
        i = int(np.clip(bd[0] * bx, 0, bx - 1))
        j = int(np.clip(bd[1] * by, 0, by - 1))
        return i, j

    def evaluate(self, arch: List[str]) -> Dict[str, Any]:
        emb = architecture_embedding(arch)
        nov = self._novelty(emb)
        qual = simulated_quality(arch)
        bd = behavioral_descriptor(arch)
        score = self.cfg.quality_weight * qual + self.cfg.novelty_weight * nov
        return {"embedding": emb, "novelty": nov, "quality": qual, "descriptor": bd, "score": score}

    def step(self) -> Dict[str, Any]:
        # Evaluate
        evals = [self.evaluate(a) for a in self.population]
        for e in evals:
            self.archive.append(e["embedding"])  # grow archive

        # QD insertion
        for arch, e in zip(self.population, evals):
            i, j = self._qd_coords(e["descriptor"])
            key = (i, j)
            incumbent = self.qd_grid.get(key)
            if incumbent is None or e["quality"] > incumbent[1]:
                self.qd_grid[key] = (arch, e["quality"])

        # Selection (by combined score)
        idx = list(range(len(self.population)))
        idx.sort(key=lambda k: evals[k]["score"], reverse=True)
        elite_count = max(1, int(self.cfg.elite_fraction * len(idx)))
        elites = [self.population[i] for i in idx[:elite_count]]

        # Variation
        next_pop: List[List[str]] = [a.copy() for a in elites]
        while len(next_pop) < self.cfg.population_size:
            if random.random() < self.cfg.crossover_rate and len(elites) >= 2:
                p1, p2 = random.sample(elites, 2)
                c1, c2 = crossover_architecture(p1, p2)
                c1 = mutate_architecture(c1, self.cfg.mutation_rate)
                c2 = mutate_architecture(c2, self.cfg.mutation_rate)
                next_pop.append(c1)
                if len(next_pop) < self.cfg.population_size:
                    next_pop.append(c2)
            else:
                p = random.choice(elites)
                next_pop.append(mutate_architecture(p, self.cfg.mutation_rate))

        self.population = next_pop[: self.cfg.population_size]
        best_idx = idx[0]
        return {
            "best_architecture": self.population[0],
            "best_eval": evals[best_idx],
            "qd_coverage": len(self.qd_grid),
        }

    def innovate(self, generations: Optional[int] = None) -> Dict[str, Any]:
        gens = generations or self.cfg.generations
        history: List[Dict[str, Any]] = []
        best_arch: List[str] = []
        best_quality = -1.0
        for g in range(gens):
            out = self.step()
            history.append({
                "generation": g,
                "qd_coverage": out["qd_coverage"],
                "quality": out["best_eval"]["quality"],
                "novelty": out["best_eval"]["novelty"],
            })
            q = out["best_eval"]["quality"]
            if q > best_quality:
                best_quality = q
                best_arch = out["best_architecture"]
        return {"best_architecture": best_arch, "best_quality": best_quality, "history": history, "qd_grid": self.qd_grid}


# Factories

def create_innovation_config(**kwargs: Any) -> InnovationConfig:
    return InnovationConfig(**kwargs)


def create_innovation_population(cfg: InnovationConfig) -> InnovationPopulation:
    return InnovationPopulation(cfg)


# Example

def example_model_innovation() -> Dict[str, Any]:
    cfg = create_innovation_config(
        strategy=InnovationStrategy.QUALITY_DIVERSITY,
        population_size=48,
        generations=20,
        elite_fraction=0.12,
        mutation_rate=0.18,
        crossover_rate=0.55,
        qd_bins=(10, 10),
        novelty_k=8,
        novelty_weight=0.35,
        quality_weight=0.65,
    )

    pop = create_innovation_population(cfg)
    result = pop.innovate()

    print("✅ Model Innovation Example Complete!")
    print(f"Best quality: {result['best_quality']:.4f}")
    print(f"QD coverage (cells filled): {len(result['qd_grid'])}")
    print("Best architecture:", "-".join(result["best_architecture"]))

    return result


__all__ = [
    "InnovationStrategy",
    "InnovationConfig",
    "InnovationPopulation",
    "create_innovation_config",
    "create_innovation_population",
    "example_model_innovation",
]


if __name__ == "__main__":
    example_model_innovation()
