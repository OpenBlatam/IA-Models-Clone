"""
Advanced Model Collaboration System for TruthGPT Optimization Core
Collaborative training, distributed learning, and peer-to-peer learning
"""

from __future__ import annotations

import time
import random
import json
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.distributed as dist
except Exception:  # lightweight import fallback
    torch = None
    nn = None
    dist = None

import logging
logger = logging.getLogger(__name__)


class CollaborationMode(Enum):
    CENTRALIZED = "centralized"
    FEDERATED = "federated"
    DECENTRALIZED = "decentralized"
    PEER_TO_PEER = "peer_to_peer"


class AggregationStrategy(Enum):
    FEDAVG = "fedavg"
    WEIGHTED = "weighted"
    KRUM = "krum"
    MEDIAN = "median"
    TRIMMED_MEAN = "trimmed_mean"


class Topology(Enum):
    STAR = "star"
    RING = "ring"
    MESH = "mesh"
    RANDOM = "random"


@dataclass
class CollaborationConfig:
    mode: CollaborationMode = CollaborationMode.FEDERATED
    aggregation: AggregationStrategy = AggregationStrategy.FEDAVG
    topology: Topology = Topology.STAR

    num_clients: int = 5
    rounds: int = 10
    local_epochs: int = 1
    participation_rate: float = 1.0  # fraction of clients per round

    # robustness and privacy
    byzantine_robust: bool = True
    differential_privacy: bool = False
    dp_noise_std: float = 0.01

    # bandwidth/latency simulation
    simulate_network: bool = True
    latency_ms_range: Tuple[int, int] = (5, 50)
    packet_loss_prob: float = 0.0

    # weighting
    weighted_by_samples: bool = True


def _simulate_latency(cfg: CollaborationConfig) -> None:
    if not cfg.simulate_network:
        return
    delay_ms = random.randint(*cfg.latency_ms_range)
    time.sleep(delay_ms / 1000.0)


def _apply_dp_noise(tensor_list: List[np.ndarray], std: float) -> List[np.ndarray]:
    return [t + np.random.normal(0, std, size=t.shape).astype(t.dtype) for t in tensor_list]


class ClientNode:
    def __init__(self, client_id: int, num_samples: int) -> None:
        self.client_id = client_id
        self.num_samples = num_samples
        self.state: Dict[str, np.ndarray] = {}
        self.last_loss: float = 1.0

    def set_state(self, state: Dict[str, np.ndarray]) -> None:
        self.state = {k: v.copy() for k, v in state.items()}

    def local_train(self, epochs: int = 1) -> Dict[str, Any]:
        improvement = random.uniform(0.01, 0.05) * epochs
        self.last_loss = max(0.0, self.last_loss * (1.0 - improvement))
        # simulate small drift in parameters
        for k in self.state:
            self.state[k] = self.state[k] + np.random.normal(0, 1e-3, size=self.state[k].shape).astype(self.state[k].dtype)
        return {"client_id": self.client_id, "loss": self.last_loss, "samples": self.num_samples}

    def get_state(self) -> Dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self.state.items()}


class Aggregator:
    def __init__(self, cfg: CollaborationConfig) -> None:
        self.cfg = cfg

    def aggregate(self, client_states: List[Tuple[Dict[str, np.ndarray], int]]) -> Dict[str, np.ndarray]:
        if self.cfg.aggregation == AggregationStrategy.FEDAVG:
            return self._fedavg(client_states)
        if self.cfg.aggregation == AggregationStrategy.WEIGHTED:
            return self._weighted(client_states)
        if self.cfg.aggregation == AggregationStrategy.MEDIAN:
            return self._median(client_states)
        if self.cfg.aggregation == AggregationStrategy.TRIMMED_MEAN:
            return self._trimmed_mean(client_states, trim_ratio=0.1)
        # KRUM and other robust methods would require distance metrics; provide simple fallback
        return self._fedavg(client_states)

    def _stack_param(self, client_states: List[Tuple[Dict[str, np.ndarray], int]], key: str) -> Tuple[np.ndarray, np.ndarray]:
        tensors = [state[key] for state, _ in client_states]
        weights = np.asarray([w for _, w in client_states], dtype=np.float64)
        return np.stack(tensors, axis=0), weights

    def _fedavg(self, client_states: List[Tuple[Dict[str, np.ndarray], int]]) -> Dict[str, np.ndarray]:
        total = float(sum(w for _, w in client_states)) if self.cfg.weighted_by_samples else float(len(client_states))
        keys = client_states[0][0].keys()
        out: Dict[str, np.ndarray] = {}
        for k in keys:
            stacked, weights = self._stack_param(client_states, k)
            if not self.cfg.weighted_by_samples:
                weights = np.ones_like(weights)
            w = (weights / weights.sum()).reshape((-1,) + (1,) * (stacked.ndim - 1))
            out[k] = np.sum(w * stacked, axis=0)
        return out

    def _weighted(self, client_states: List[Tuple[Dict[str, np.ndarray], int]]) -> Dict[str, np.ndarray]:
        return self._fedavg(client_states)

    def _median(self, client_states: List[Tuple[Dict[str, np.ndarray], int]]) -> Dict[str, np.ndarray]:
        keys = client_states[0][0].keys()
        out: Dict[str, np.ndarray] = {}
        for k in keys:
            stacked, _ = self._stack_param(client_states, k)
            out[k] = np.median(stacked, axis=0)
        return out

    def _trimmed_mean(self, client_states: List[Tuple[Dict[str, np.ndarray], int]], trim_ratio: float) -> Dict[str, np.ndarray]:
        keys = client_states[0][0].keys()
        out: Dict[str, np.ndarray] = {}
        trim = max(1, int(trim_ratio * len(client_states)))
        for k in keys:
            stacked, _ = self._stack_param(client_states, k)
            sorted_vals = np.sort(stacked, axis=0)
            trimmed = sorted_vals[trim:-trim] if 2 * trim < len(stacked) else sorted_vals
            out[k] = np.mean(trimmed, axis=0)
        return out


class CollaborationCoordinator:
    def __init__(self, cfg: CollaborationConfig, initial_state: Dict[str, np.ndarray]) -> None:
        self.cfg = cfg
        self.aggregator = Aggregator(cfg)
        self.clients: List[ClientNode] = [ClientNode(i, num_samples=random.randint(500, 5000)) for i in range(cfg.num_clients)]
        for c in self.clients:
            c.set_state(initial_state)
        self.global_state: Dict[str, np.ndarray] = {k: v.copy() for k, v in initial_state.items()}
        self.history: List[Dict[str, Any]] = []

    def _sample_clients(self) -> List[ClientNode]:
        k = max(1, int(self.cfg.participation_rate * len(self.clients)))
        return random.sample(self.clients, k)

    def run_round(self, round_idx: int) -> Dict[str, Any]:
        _simulate_latency(self.cfg)
        selected = self._sample_clients()
        stats: List[Dict[str, Any]] = []
        client_states: List[Tuple[Dict[str, np.ndarray], int]] = []

        for client in selected:
            client.set_state(self.global_state)
            s = client.local_train(epochs=self.cfg.local_epochs)
            stats.append(s)
            client_states.append((client.get_state(), client.num_samples))

        if self.cfg.differential_privacy:
            client_states = [({k: v for k, v in _apply_dp_noise(list(state.values()), self.cfg.dp_noise_std)}, n)  # type: ignore
                             for state, n in client_states]

        new_global = self.aggregator.aggregate(client_states)
        self.global_state = new_global

        round_summary = {
            "round": round_idx,
            "participating_clients": len(selected),
            "avg_loss": float(np.mean([s["loss"] for s in stats])) if stats else None,
            "total_samples": int(sum(s["samples"] for s in stats)) if stats else 0,
        }
        self.history.append(round_summary)
        return round_summary

    def collaborate(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {"rounds": []}
        for r in range(self.cfg.rounds):
            summary["rounds"].append(self.run_round(r))
        summary["final_state_checksum"] = self._checksum(self.global_state)
        return summary

    @staticmethod
    def _checksum(state: Dict[str, np.ndarray]) -> str:
        acc = 0.0
        for v in state.values():
            acc += float(v.mean()) + float(v.std())
        return f"ck_{abs(int(acc * 1e6))}"


# Factories

def create_collaboration_config(**kwargs: Any) -> CollaborationConfig:
    return CollaborationConfig(**kwargs)


def create_collaboration_coordinator(cfg: CollaborationConfig, initial_state: Dict[str, np.ndarray]) -> CollaborationCoordinator:
    return CollaborationCoordinator(cfg, initial_state)


# Example

def example_model_collaboration() -> CollaborationCoordinator:
    rng = np.random.default_rng(42)
    initial_state = {
        "layer1.weight": rng.normal(0, 0.02, size=(64, 32)).astype(np.float32),
        "layer1.bias": rng.normal(0, 0.02, size=(64,)).astype(np.float32),
        "layer2.weight": rng.normal(0, 0.02, size=(32, 10)).astype(np.float32),
        "layer2.bias": rng.normal(0, 0.02, size=(10,)).astype(np.float32),
    }

    cfg = create_collaboration_config(
        mode=CollaborationMode.FEDERATED,
        aggregation=AggregationStrategy.FEDAVG,
        topology=Topology.STAR,
        num_clients=7,
        rounds=8,
        local_epochs=1,
        participation_rate=0.6,
        byzantine_robust=True,
        differential_privacy=False,
        simulate_network=True,
        latency_ms_range=(3, 25),
        packet_loss_prob=0.0,
    )

    coordinator = create_collaboration_coordinator(cfg, initial_state)
    results = coordinator.collaborate()

    print("✅ Model Collaboration Example Complete!")
    print(f"Rounds: {len(results['rounds'])}")
    print(f"Final State Checksum: {results['final_state_checksum']}")
    if results["rounds"]:
        last = results["rounds"][-1]
        print(f"Last Round — clients: {last['participating_clients']}, avg_loss: {last['avg_loss']:.4f}")

    return coordinator


__all__ = [
    "CollaborationMode",
    "AggregationStrategy",
    "Topology",
    "CollaborationConfig",
    "ClientNode",
    "Aggregator",
    "CollaborationCoordinator",
    "create_collaboration_config",
    "create_collaboration_coordinator",
    "example_model_collaboration",
]


if __name__ == "__main__":
    example_model_collaboration()
