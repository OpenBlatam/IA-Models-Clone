from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import asyncio
import aiohttp
import logging
import time
import random
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
import structlog
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Load Balancer for OS Content UGC Video Generator
Provides horizontal scaling and high availability
"""


logger = structlog.get_logger("os_content.load_balancer")

class HealthStatus(Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class BackendServer:
    """Backend server configuration"""
    url: str
    weight: int = 1
    max_connections: int = 100
    timeout: float = 30.0
    health_check_interval: float = 30.0
    health_check_path: str = "/os-content/health"
    
    # Runtime state
    status: HealthStatus = HealthStatus.UNKNOWN
    active_connections: int = 0
    last_health_check: float = 0.0
    response_time: float = 0.0
    error_count: int = 0
    success_count: int = 0

class LoadBalancer:
    """Advanced load balancer with health checking and multiple algorithms"""
    
    def __init__(self, algorithm: str = "round_robin"):
        
    """__init__ function."""
self.backends: List[BackendServer] = []
        self.algorithm = algorithm
        self.current_index = 0
        self.session: Optional[aiohttp.ClientSession] = None
        self.running = False
        
        # Statistics
        self.total_requests = 0
        self.failed_requests = 0
        self.start_time = time.time()
    
    async def add_backend(self, url: str, weight: int = 1, **kwargs) -> None:
        """Add a backend server"""
        backend = BackendServer(url=url, weight=weight, **kwargs)
        self.backends.append(backend)
        logger.info(f"Added backend: {url} with weight {weight}")
    
    async def remove_backend(self, url: str) -> None:
        """Remove a backend server"""
        self.backends = [b for b in self.backends if b.url != url]
        logger.info(f"Removed backend: {url}")
    
    async def start(self) -> None:
        """Start the load balancer"""
        if self.running:
            return
        
        self.running = True
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(timeout=timeout)
        
        # Start health checking
        asyncio.create_task(self._health_check_loop())
        
        logger.info("Load balancer started")
    
    async def stop(self) -> None:
        """Stop the load balancer"""
        self.running = False
        if self.session:
            await self.session.close()
        logger.info("Load balancer stopped")
    
    def _select_backend(self) -> Optional[BackendServer]:
        """Select backend using the configured algorithm"""
        healthy_backends = [b for b in self.backends if b.status == HealthStatus.HEALTHY]
        
        if not healthy_backends:
            return None
        
        if self.algorithm == "round_robin":
            return self._round_robin_select(healthy_backends)
        elif self.algorithm == "least_connections":
            return self._least_connections_select(healthy_backends)
        elif self.algorithm == "weighted_round_robin":
            return self._weighted_round_robin_select(healthy_backends)
        elif self.algorithm == "random":
            return self._random_select(healthy_backends)
        else:
            return self._round_robin_select(healthy_backends)
    
    def _round_robin_select(self, backends: List[BackendServer]) -> BackendServer:
        """Round robin selection"""
        backend = backends[self.current_index % len(backends)]
        self.current_index += 1
        return backend
    
    def _least_connections_select(self, backends: List[BackendServer]) -> BackendServer:
        """Least connections selection"""
        return min(backends, key=lambda b: b.active_connections)
    
    def _weighted_round_robin_select(self, backends: List[BackendServer]) -> BackendServer:
        """Weighted round robin selection"""
        total_weight = sum(b.weight for b in backends)
        if total_weight == 0:
            return random.choice(backends)
        
        # Simple weighted selection
        weights = [b.weight for b in backends]
        return random.choices(backends, weights=weights)[0]
    
    def _random_select(self, backends: List[BackendServer]) -> BackendServer:
        """Random selection"""
        return random.choice(backends)
    
    async def _health_check_loop(self) -> None:
        """Health check loop for all backends"""
        while self.running:
            tasks = []
            for backend in self.backends:
                task = asyncio.create_task(self._check_backend_health(backend))
                tasks.append(task)
            
            await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(30)  # Check every 30 seconds
    
    async def _check_backend_health(self, backend: BackendServer) -> None:
        """Check health of a single backend"""
        try:
            start_time = time.time()
            health_url = f"{backend.url.rstrip('/')}{backend.health_check_path}"
            
            async with self.session.get(health_url) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    backend.status = HealthStatus.HEALTHY
                    backend.response_time = response_time
                    backend.success_count += 1
                    backend.error_count = 0
                else:
                    backend.status = HealthStatus.UNHEALTHY
                    backend.error_count += 1
                
                backend.last_health_check = time.time()
                
        except Exception as e:
            backend.status = HealthStatus.UNHEALTHY
            backend.error_count += 1
            logger.warning(f"Health check failed for {backend.url}: {e}")
    
    async async def forward_request(self, method: str, path: str, **kwargs) -> aiohttp.ClientResponse:
        """Forward request to selected backend"""
        backend = self._select_backend()
        if not backend:
            raise Exception("No healthy backends available")
        
        try:
            backend.active_connections += 1
            self.total_requests += 1
            
            url = f"{backend.url.rstrip('/')}{path}"
            start_time = time.time()
            
            async with self.session.request(method, url, **kwargs) as response:
                response_time = time.time() - start_time
                backend.response_time = response_time
                backend.success_count += 1
                
                return response
                
        except Exception as e:
            backend.error_count += 1
            self.failed_requests += 1
            logger.error(f"Request failed for {backend.url}: {e}")
            raise
        finally:
            backend.active_connections = max(0, backend.active_connections - 1)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        uptime = time.time() - self.start_time
        healthy_backends = sum(1 for b in self.backends if b.status == HealthStatus.HEALTHY)
        
        return {
            "uptime": uptime,
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "success_rate": ((self.total_requests - self.failed_requests) / self.total_requests * 100) if self.total_requests > 0 else 0,
            "total_backends": len(self.backends),
            "healthy_backends": healthy_backends,
            "algorithm": self.algorithm,
            "backends": [
                {
                    "url": b.url,
                    "status": b.status.value,
                    "active_connections": b.active_connections,
                    "response_time": b.response_time,
                    "success_count": b.success_count,
                    "error_count": b.error_count,
                    "weight": b.weight
                }
                for b in self.backends
            ]
        }

# Global load balancer instance
load_balancer = LoadBalancer()

async def initialize_load_balancer(backends: List[str], algorithm: str = "round_robin"):
    """Initialize the load balancer with backend servers"""
    global load_balancer
    load_balancer = LoadBalancer(algorithm=algorithm)
    
    for backend_url in backends:
        await load_balancer.add_backend(backend_url)
    
    await load_balancer.start()
    logger.info(f"Load balancer initialized with {len(backends)} backends")

async def cleanup_load_balancer():
    """Cleanup the load balancer"""
    await load_balancer.stop()
    logger.info("Load balancer cleaned up") 