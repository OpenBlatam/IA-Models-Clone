from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import asyncio
import random
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime
import aiohttp
from .service_discovery import ServiceInstance
from typing import Any, List, Dict, Optional
"""
Load Balancer Implementation
===========================

Advanced load balancing strategies for microservices:
- Round Robin
- Weighted Round Robin
- Least Connections
- Health-based routing
- Geographic routing
"""



logger = logging.getLogger(__name__)

@dataclass
class LoadBalancerStats:
    """Load balancer statistics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    backend_stats: Dict[str, Dict[str, Any]] = None
    
    def __post_init__(self) -> Any:
        if self.backend_stats is None:
            self.backend_stats = {}


class LoadBalancingStrategy(ABC):
    """Abstract base class for load balancing strategies."""
    
    @abstractmethod
    def select_instance(self, instances: List[ServiceInstance], request_context: Dict = None) -> Optional[ServiceInstance]:
        """Select an instance based on the strategy."""
        pass


class RoundRobinStrategy(LoadBalancingStrategy):
    """Round robin load balancing strategy."""
    
    def __init__(self) -> Any:
        self.current_index = 0
    
    def select_instance(self, instances: List[ServiceInstance], request_context: Dict = None) -> Optional[ServiceInstance]:
        """Select instance using round robin."""
        if not instances:
            return None
        
        instance = instances[self.current_index % len(instances)]
        self.current_index += 1
        return instance


class WeightedRoundRobinStrategy(LoadBalancingStrategy):
    """Weighted round robin load balancing strategy."""
    
    def __init__(self) -> Any:
        self.current_weights: Dict[str, int] = {}
        self.total_weight = 0
    
    def select_instance(self, instances: List[ServiceInstance], request_context: Dict = None) -> Optional[ServiceInstance]:
        """Select instance using weighted round robin."""
        if not instances:
            return None
        
        # Calculate weights (can be based on instance metadata)
        weighted_instances = []
        for instance in instances:
            weight = int(instance.metadata.get("weight", 1))
            weighted_instances.extend([instance] * weight)
        
        if not weighted_instances:
            return instances[0]
        
        index = getattr(self, '_index', 0) % len(weighted_instances)
        self._index = index + 1
        return weighted_instances[index]


class LeastConnectionsStrategy(LoadBalancingStrategy):
    """Least connections load balancing strategy."""
    
    def __init__(self) -> Any:
        self.connection_counts: Dict[str, int] = {}
    
    def select_instance(self, instances: List[ServiceInstance], request_context: Dict = None) -> Optional[ServiceInstance]:
        """Select instance with least connections."""
        if not instances:
            return None
        
        # Find instance with minimum connections
        min_connections = float('inf')
        selected_instance = None
        
        for instance in instances:
            connections = self.connection_counts.get(instance.id, 0)
            if connections < min_connections:
                min_connections = connections
                selected_instance = instance
        
        if selected_instance:
            self.connection_counts[selected_instance.id] = min_connections + 1
        
        return selected_instance
    
    def release_connection(self, instance_id: str):
        """Release a connection for an instance."""
        if instance_id in self.connection_counts:
            self.connection_counts[instance_id] = max(0, self.connection_counts[instance_id] - 1)


class HealthBasedStrategy(LoadBalancingStrategy):
    """Health-based load balancing strategy."""
    
    def __init__(self, fallback_strategy: LoadBalancingStrategy = None):
        
    """__init__ function."""
self.fallback_strategy = fallback_strategy or RoundRobinStrategy()
        self.health_cache: Dict[str, bool] = {}
        
    async def check_instance_health(self, instance: ServiceInstance) -> bool:
        """Check if instance is healthy."""
        try:
            async with aiohttp.ClientSession() as session:
                health_url = instance.health_check_url or f"{instance.url}/health"
                async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    is_healthy = response.status == 200
                    self.health_cache[instance.id] = is_healthy
                    return is_healthy
        except:
            self.health_cache[instance.id] = False
            return False
    
    def select_instance(self, instances: List[ServiceInstance], request_context: Dict = None) -> Optional[ServiceInstance]:
        """Select healthy instance."""
        if not instances:
            return None
        
        # Filter healthy instances
        healthy_instances = [
            instance for instance in instances
            if self.health_cache.get(instance.id, True) and instance.is_healthy
        ]
        
        if healthy_instances:
            return self.fallback_strategy.select_instance(healthy_instances, request_context)
        
        # Fallback to any instance if none are marked as healthy
        return self.fallback_strategy.select_instance(instances, request_context)


class LoadBalancerManager:
    """Advanced load balancer with multiple strategies."""
    
    def __init__(self, strategy: LoadBalancingStrategy = None):
        
    """__init__ function."""
self.strategy = strategy or RoundRobinStrategy()
        self.stats = LoadBalancerStats()
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if not self.session:
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=10,
                keepalive_timeout=30
            )
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout
            )
        return self.session
    
    async async def route_request(self, 
                          instances: List[ServiceInstance],
                          path: str = "/",
                          method: str = "GET",
                          **kwargs) -> Dict[str, Any]:
        """Route request to selected instance."""
        if not instances:
            return {
                "error": "No instances available",
                "status_code": 503
            }
        
        # Select instance
        selected_instance = self.strategy.select_instance(instances)
        if not selected_instance:
            return {
                "error": "No instance selected",
                "status_code": 503
            }
        
        # Make request
        start_time = asyncio.get_event_loop().time()
        
        try:
            session = await self._get_session()
            url = f"{selected_instance.url}{path}"
            
            async with session.request(method, url, **kwargs) as response:
                end_time = asyncio.get_event_loop().time()
                response_time = end_time - start_time
                
                # Update stats
                self._update_stats(selected_instance.id, response_time, response.status < 400)
                
                # Read response
                try:
                    if response.headers.get('content-type', '').startswith('application/json'):
                        response_data = await response.json()
                    else:
                        response_data = await response.text()
                except:
                    response_data = await response.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                
                return {
                    "data": response_data,
                    "status_code": response.status,
                    "headers": dict(response.headers),
                    "instance": selected_instance.id,
                    "response_time": response_time
                }
                
        except Exception as e:
            end_time = asyncio.get_event_loop().time()
            response_time = end_time - start_time
            
            # Update stats for failed request
            self._update_stats(selected_instance.id, response_time, False)
            
            logger.error(f"Error routing request to {selected_instance.url}: {e}")
            return {
                "error": str(e),
                "status_code": 500,
                "instance": selected_instance.id,
                "response_time": response_time
            }
    
    def _update_stats(self, instance_id: str, response_time: float, success: bool):
        """Update load balancer statistics."""
        self.stats.total_requests += 1
        
        if success:
            self.stats.successful_requests += 1
        else:
            self.stats.failed_requests += 1
        
        # Update average response time
        total_time = self.stats.average_response_time * (self.stats.total_requests - 1) + response_time
        self.stats.average_response_time = total_time / self.stats.total_requests
        
        # Update backend stats
        if instance_id not in self.stats.backend_stats:
            self.stats.backend_stats[instance_id] = {
                "requests": 0,
                "successful": 0,
                "failed": 0,
                "avg_response_time": 0.0
            }
        
        backend_stats = self.stats.backend_stats[instance_id]
        backend_stats["requests"] += 1
        
        if success:
            backend_stats["successful"] += 1
        else:
            backend_stats["failed"] += 1
        
        # Update backend average response time
        total_backend_time = backend_stats["avg_response_time"] * (backend_stats["requests"] - 1) + response_time
        backend_stats["avg_response_time"] = total_backend_time / backend_stats["requests"]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        return {
            "total_requests": self.stats.total_requests,
            "successful_requests": self.stats.successful_requests,
            "failed_requests": self.stats.failed_requests,
            "success_rate": self.stats.successful_requests / max(1, self.stats.total_requests),
            "average_response_time": self.stats.average_response_time,
            "strategy": self.strategy.__class__.__name__,
            "backend_stats": self.stats.backend_stats
        }
    
    def set_strategy(self, strategy: LoadBalancingStrategy):
        """Change load balancing strategy."""
        self.strategy = strategy
        logger.info(f"Changed load balancing strategy to {strategy.__class__.__name__}")
    
    async def health_check_instances(self, instances: List[ServiceInstance]) -> Dict[str, bool]:
        """Health check all instances."""
        if not isinstance(self.strategy, HealthBasedStrategy):
            # Basic health check
            results = {}
            session = await self._get_session()
            
            for instance in instances:
                try:
                    health_url = instance.health_check_url or f"{instance.url}/health"
                    async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                        results[instance.id] = response.status == 200
                except:
                    results[instance.id] = False
            
            return results
        else:
            # Use health-based strategy's health check
            results = {}
            for instance in instances:
                results[instance.id] = await self.strategy.check_instance_health(instance)
            return results
    
    async def close(self) -> Any:
        """Close HTTP session."""
        if self.session:
            await self.session.close() 