"""
Advanced Load Balancer
Multiple load balancing strategies for microservices
"""

import logging
import hashlib
import time
from typing import List, Optional, Dict, Any
from enum import Enum

from microservices.service_discovery import ServiceInstance, ServiceRegistry, get_service_registry

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    IP_HASH = "ip_hash"
    LEAST_RESPONSE_TIME = "least_response_time"


class LoadBalancer:
    """
    Advanced load balancer for microservices
    
    Features:
    - Multiple balancing strategies
    - Health-aware routing
    - Weighted distribution
    - Connection tracking
    - Response time tracking
    """
    
    def __init__(self, registry: Optional[ServiceRegistry] = None):
        self.registry = registry or get_service_registry()
        self._round_robin_index: Dict[str, int] = {}
        self._connection_counts: Dict[str, Dict[str, int]] = {}
        self._response_times: Dict[str, Dict[str, float]] = {}
        self._last_used: Dict[str, Dict[str, float]] = {}
    
    def select_instance(
        self,
        service_name: str,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN,
        client_ip: Optional[str] = None,
        **kwargs
    ) -> Optional[ServiceInstance]:
        """
        Select service instance using load balancing strategy
        
        Args:
            service_name: Name of service
            strategy: Load balancing strategy
            client_ip: Client IP for IP hash strategy
            **kwargs: Additional parameters
            
        Returns:
            Selected service instance
        """
        instances = self.registry.discover(service_name, healthy_only=True)
        
        if not instances:
            return None
        
        if len(instances) == 1:
            return instances[0]
        
        if strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin(service_name, instances)
        elif strategy == LoadBalancingStrategy.RANDOM:
            import random
            return random.choice(instances)
        elif strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections(service_name, instances)
        elif strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin(service_name, instances)
        elif strategy == LoadBalancingStrategy.IP_HASH:
            return self._ip_hash(service_name, instances, client_ip)
        elif strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return self._least_response_time(service_name, instances)
        else:
            return instances[0]
    
    def _round_robin(self, service_name: str, instances: List[ServiceInstance]) -> ServiceInstance:
        """Round-robin selection"""
        if service_name not in self._round_robin_index:
            self._round_robin_index[service_name] = 0
        
        index = self._round_robin_index[service_name]
        instance = instances[index % len(instances)]
        self._round_robin_index[service_name] = (index + 1) % len(instances)
        
        return instance
    
    def _least_connections(self, service_name: str, instances: List[ServiceInstance]) -> ServiceInstance:
        """Least connections selection"""
        if service_name not in self._connection_counts:
            self._connection_counts[service_name] = {i.instance_id: 0 for i in instances}
        
        counts = self._connection_counts[service_name]
        instance = min(instances, key=lambda i: counts.get(i.instance_id, 0))
        
        # Increment connection count
        counts[instance.instance_id] = counts.get(instance.instance_id, 0) + 1
        
        return instance
    
    def _weighted_round_robin(
        self,
        service_name: str,
        instances: List[ServiceInstance]
    ) -> ServiceInstance:
        """Weighted round-robin selection"""
        # Get weights from metadata or use default
        weights = [
            int(i.metadata.get("weight", 1)) for i in instances
        ]
        total_weight = sum(weights)
        
        # Simple weighted selection (in production, use more sophisticated algorithm)
        import random
        rand = random.uniform(0, total_weight)
        cumulative = 0
        
        for instance, weight in zip(instances, weights):
            cumulative += weight
            if rand <= cumulative:
                return instance
        
        return instances[0]
    
    def _ip_hash(
        self,
        service_name: str,
        instances: List[ServiceInstance],
        client_ip: Optional[str]
    ) -> ServiceInstance:
        """IP hash selection (sticky sessions)"""
        if not client_ip:
            return instances[0]
        
        hash_value = int(hashlib.md5(client_ip.encode()).hexdigest(), 16)
        index = hash_value % len(instances)
        return instances[index]
    
    def _least_response_time(
        self,
        service_name: str,
        instances: List[ServiceInstance]
    ) -> ServiceInstance:
        """Least response time selection"""
        if service_name not in self._response_times:
            self._response_times[service_name] = {
                i.instance_id: 1000.0 for i in instances
            }
        
        times = self._response_times[service_name]
        instance = min(instances, key=lambda i: times.get(i.instance_id, 1000.0))
        return instance
    
    def record_response_time(
        self,
        service_name: str,
        instance_id: str,
        response_time: float
    ) -> None:
        """Record response time for instance"""
        if service_name not in self._response_times:
            self._response_times[service_name] = {}
        
        # Exponential moving average
        current = self._response_times[service_name].get(instance_id, response_time)
        self._response_times[service_name][instance_id] = (
            current * 0.7 + response_time * 0.3
        )
    
    def release_connection(self, service_name: str, instance_id: str) -> None:
        """Release connection (decrement count)"""
        if service_name in self._connection_counts:
            counts = self._connection_counts[service_name]
            if instance_id in counts:
                counts[instance_id] = max(0, counts[instance_id] - 1)


# Global load balancer
_balancer: Optional[LoadBalancer] = None


def get_load_balancer() -> LoadBalancer:
    """Get global load balancer"""
    global _balancer
    if _balancer is None:
        _balancer = LoadBalancer()
    return _balancer















