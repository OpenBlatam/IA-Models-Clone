"""
Scalability Optimizations

Optimizations for:
- Horizontal scaling
- Load balancing
- Auto-scaling
- Resource management
- Distributed systems
"""

import logging
import os
from typing import Optional, Dict, Any, List
import asyncio
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ScalingStrategy(Enum):
    """Scaling strategies."""
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    AUTO = "auto"


@dataclass
class ResourceMetrics:
    """Resource usage metrics."""
    cpu_percent: float
    memory_percent: float
    request_rate: float
    error_rate: float
    queue_length: int


class AutoScaler:
    """Auto-scaling based on metrics."""
    
    def __init__(
        self,
        min_instances: int = 1,
        max_instances: int = 10,
        target_cpu: float = 70.0,
        target_memory: float = 80.0
    ):
        """
        Initialize auto-scaler.
        
        Args:
            min_instances: Minimum instances
            max_instances: Maximum instances
            target_cpu: Target CPU percentage
            target_memory: Target memory percentage
        """
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.target_cpu = target_cpu
        self.target_memory = target_memory
        self.current_instances = min_instances
    
    def should_scale_up(self, metrics: ResourceMetrics) -> bool:
        """
        Check if should scale up.
        
        Args:
            metrics: Current resource metrics
            
        Returns:
            True if should scale up
        """
        if self.current_instances >= self.max_instances:
            return False
        
        # Scale up if CPU or memory is high
        if metrics.cpu_percent > self.target_cpu:
            return True
        
        if metrics.memory_percent > self.target_memory:
            return True
        
        # Scale up if error rate is high (might need more capacity)
        if metrics.error_rate > 0.05:  # 5% error rate
            return True
        
        return False
    
    def should_scale_down(self, metrics: ResourceMetrics) -> bool:
        """
        Check if should scale down.
        
        Args:
            metrics: Current resource metrics
            
        Returns:
            True if should scale down
        """
        if self.current_instances <= self.min_instances:
            return False
        
        # Scale down if resources are low
        if (metrics.cpu_percent < self.target_cpu * 0.5 and
            metrics.memory_percent < self.target_memory * 0.5 and
            metrics.request_rate < 10):  # Low traffic
            return True
        
        return False
    
    def get_scaling_decision(self, metrics: ResourceMetrics) -> int:
        """
        Get scaling decision.
        
        Args:
            metrics: Current resource metrics
            
        Returns:
            Number of instances to add/remove (positive = scale up, negative = scale down)
        """
        if self.should_scale_up(metrics):
            return 1
        elif self.should_scale_down(metrics):
            return -1
        return 0


class LoadBalancer:
    """Load balancing strategies."""
    
    @staticmethod
    def round_robin(instances: List[str], current_index: int = 0) -> tuple[str, int]:
        """
        Round-robin load balancing.
        
        Args:
            instances: List of instance URLs
            current_index: Current index
            
        Returns:
            (selected_instance, new_index)
        """
        if not instances:
            raise ValueError("No instances available")
        
        selected = instances[current_index % len(instances)]
        new_index = (current_index + 1) % len(instances)
        
        return selected, new_index
    
    @staticmethod
    def least_connections(instances: Dict[str, int]) -> str:
        """
        Least connections load balancing.
        
        Args:
            instances: Dictionary of {instance_url: connection_count}
            
        Returns:
            Instance with least connections
        """
        if not instances:
            raise ValueError("No instances available")
        
        return min(instances.items(), key=lambda x: x[1])[0]
    
    @staticmethod
    def weighted_round_robin(
        instances: List[tuple[str, float]],
        current_weights: Dict[str, float]
    ) -> str:
        """
        Weighted round-robin load balancing.
        
        Args:
            instances: List of (instance_url, weight) tuples
            current_weights: Current weight distribution
            
        Returns:
            Selected instance
        """
        if not instances:
            raise ValueError("No instances available")
        
        # Select based on weights
        total_weight = sum(weight for _, weight in instances)
        import random
        r = random.uniform(0, total_weight)
        
        cumulative = 0
        for instance, weight in instances:
            cumulative += weight
            if r <= cumulative:
                return instance
        
        return instances[0][0]  # Fallback


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_max_calls: int = 3
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening
            recovery_timeout: Time to wait before trying again
            half_open_max_calls: Max calls in half-open state
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
        self.half_open_calls = 0
    
    def call(self, func, *args, **kwargs):
        """
        Call function with circuit breaker.
        
        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        import time
        
        # Check state
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half_open"
                self.half_open_calls = 0
            else:
                raise Exception("Circuit breaker is OPEN")
        
        # Try call
        try:
            result = func(*args, **kwargs)
            
            # Success - reset if half-open
            if self.state == "half_open":
                self.half_open_calls += 1
                if self.half_open_calls >= self.half_open_max_calls:
                    self.state = "closed"
                    self.failure_count = 0
            
            return result
            
        except Exception as e:
            # Failure
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            
            raise


class ResourceManager:
    """Resource management and allocation."""
    
    @staticmethod
    def get_optimal_worker_count(
        cpu_count: Optional[int] = None,
        memory_gb: Optional[float] = None
    ) -> int:
        """
        Calculate optimal worker count.
        
        Args:
            cpu_count: Number of CPUs
            memory_gb: Available memory in GB
            
        Returns:
            Optimal worker count
        """
        import os
        
        if cpu_count is None:
            cpu_count = os.cpu_count() or 1
        
        if memory_gb is None:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Formula: workers = min(CPU * 2, memory_gb / 2)
        workers_by_cpu = cpu_count * 2
        workers_by_memory = int(memory_gb / 2)
        
        return min(workers_by_cpu, workers_by_memory, 32)  # Cap at 32
    
    @staticmethod
    def allocate_resources(
        total_cpu: float,
        total_memory_gb: float,
        num_services: int
    ) -> List[Dict[str, float]]:
        """
        Allocate resources across services.
        
        Args:
            total_cpu: Total CPU available
            total_memory_gb: Total memory available
            num_services: Number of services
            
        Returns:
            List of resource allocations
        """
        cpu_per_service = total_cpu / num_services
        memory_per_service = total_memory_gb / num_services
        
        return [
            {
                'cpu': cpu_per_service,
                'memory_gb': memory_per_service
            }
            for _ in range(num_services)
        ]








