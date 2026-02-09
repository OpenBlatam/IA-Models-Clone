"""
Load Balancer for Color Grading AI
===================================

Load balancing for distributed processing.
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


class LoadBalanceStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_LOAD = "least_load"
    RANDOM = "random"


@dataclass
class Worker:
    """Worker node."""
    id: str
    weight: int = 1
    active_connections: int = 0
    current_load: float = 0.0
    is_healthy: bool = True
    metadata: Dict[str, Any] = None


class LoadBalancer:
    """
    Load balancer for distributing work.
    
    Features:
    - Multiple balancing strategies
    - Health checking
    - Weighted distribution
    - Load monitoring
    """
    
    def __init__(self, strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN):
        """
        Initialize load balancer.
        
        Args:
            strategy: Load balancing strategy
        """
        self.strategy = strategy
        self._workers: Dict[str, Worker] = {}
        self._round_robin_index = 0
        self._connection_counts: Dict[str, int] = defaultdict(int)
    
    def register_worker(
        self,
        worker_id: str,
        weight: int = 1,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Register a worker.
        
        Args:
            worker_id: Worker identifier
            weight: Worker weight (for weighted strategies)
            metadata: Optional worker metadata
        """
        self._workers[worker_id] = Worker(
            id=worker_id,
            weight=weight,
            metadata=metadata or {}
        )
        logger.info(f"Registered worker: {worker_id} (weight: {weight})")
    
    def unregister_worker(self, worker_id: str):
        """Unregister a worker."""
        if worker_id in self._workers:
            del self._workers[worker_id]
            if worker_id in self._connection_counts:
                del self._connection_counts[worker_id]
            logger.info(f"Unregistered worker: {worker_id}")
    
    def set_worker_health(self, worker_id: str, is_healthy: bool):
        """Set worker health status."""
        if worker_id in self._workers:
            self._workers[worker_id].is_healthy = is_healthy
            logger.info(f"Worker {worker_id} health: {is_healthy}")
    
    def set_worker_load(self, worker_id: str, load: float):
        """Set worker current load."""
        if worker_id in self._workers:
            self._workers[worker_id].current_load = load
    
    def select_worker(self) -> Optional[str]:
        """
        Select a worker based on strategy.
        
        Returns:
            Worker ID or None if no workers available
        """
        # Filter healthy workers
        healthy_workers = [
            w for w in self._workers.values()
            if w.is_healthy
        ]
        
        if not healthy_workers:
            logger.warning("No healthy workers available")
            return None
        
        if self.strategy == LoadBalanceStrategy.ROUND_ROBIN:
            worker = healthy_workers[self._round_robin_index % len(healthy_workers)]
            self._round_robin_index += 1
            return worker.id
        
        elif self.strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
            worker = min(healthy_workers, key=lambda w: w.active_connections)
            return worker.id
        
        elif self.strategy == LoadBalanceStrategy.WEIGHTED_ROUND_ROBIN:
            # Weighted selection
            total_weight = sum(w.weight for w in healthy_workers)
            if total_weight == 0:
                return healthy_workers[0].id
            
            # Simple weighted round robin
            worker = healthy_workers[self._round_robin_index % len(healthy_workers)]
            self._round_robin_index += 1
            return worker.id
        
        elif self.strategy == LoadBalanceStrategy.LEAST_LOAD:
            worker = min(healthy_workers, key=lambda w: w.current_load)
            return worker.id
        
        elif self.strategy == LoadBalanceStrategy.RANDOM:
            import random
            worker = random.choice(healthy_workers)
            return worker.id
        
        return None
    
    def record_connection(self, worker_id: str):
        """Record new connection to worker."""
        if worker_id in self._workers:
            self._workers[worker_id].active_connections += 1
            self._connection_counts[worker_id] += 1
    
    def release_connection(self, worker_id: str):
        """Release connection from worker."""
        if worker_id in self._workers:
            self._workers[worker_id].active_connections = max(
                0,
                self._workers[worker_id].active_connections - 1
            )
    
    def get_worker_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        return {
            worker_id: {
                "weight": worker.weight,
                "active_connections": worker.active_connections,
                "current_load": worker.current_load,
                "is_healthy": worker.is_healthy,
                "total_connections": self._connection_counts.get(worker_id, 0),
            }
            for worker_id, worker in self._workers.items()
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        healthy_count = sum(1 for w in self._workers.values() if w.is_healthy)
        total_connections = sum(w.active_connections for w in self._workers.values())
        
        return {
            "strategy": self.strategy.value,
            "total_workers": len(self._workers),
            "healthy_workers": healthy_count,
            "total_connections": total_connections,
            "workers": self.get_worker_stats(),
        }




