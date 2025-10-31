"""
Intelligent Load Balancer for Blaze AI System.

This module provides advanced load balancing with intelligent routing,
health monitoring, adaptive strategies, and performance optimization.
"""

from __future__ import annotations

import asyncio
import time
import random
import hashlib
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Awaitable, Protocol
from collections import defaultdict, deque
import weakref
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from ..core.interfaces import CoreConfig, SystemHealth, HealthStatus
from ..utils.logging import get_logger
from ..utils.metrics import MetricsCollector

# =============================================================================
# Load Balancing Types
# =============================================================================

class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"
    CONSISTENT_HASH = "consistent_hash"
    ADAPTIVE = "adaptive"
    POWER_OF_TWO = "power_of_two"
    LEAST_LOADED = "least_loaded"

class HealthStatus(Enum):
    """Health status for load balancer."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"

@dataclass
class LoadBalancerConfig:
    """Configuration for load balancer."""
    strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE
    health_check_interval: float = 30.0
    health_check_timeout: float = 10.0
    max_retries: int = 3
    retry_delay: float = 1.0
    enable_sticky_sessions: bool = True
    sticky_session_timeout: float = 300.0
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    enable_rate_limiting: bool = True
    rate_limit_per_second: int = 1000
    enable_adaptive_routing: bool = True
    performance_window_size: int = 100
    enable_predictive_routing: bool = True

# =============================================================================
# Backend Server Management
# =============================================================================

@dataclass
class BackendServer:
    """Represents a backend server in the load balancer."""
    id: str
    host: str
    port: int
    weight: int = 100
    max_connections: int = 1000
    health_check_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Runtime state
    current_connections: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_response_time: float = 0.0
    average_response_time: float = 0.0
    last_health_check: float = 0.0
    health_status: HealthStatus = HealthStatus.UNKNOWN
    consecutive_failures: int = 0
    last_failure_time: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests
    
    @property
    def connection_utilization(self) -> float:
        """Calculate connection utilization."""
        if self.max_connections == 0:
            return 0.0
        return self.current_connections / self.max_connections
    
    @property
    def is_available(self) -> bool:
        """Check if server is available for requests."""
        return (self.health_status == HealthStatus.HEALTHY and 
                self.current_connections < self.max_connections and
                self.consecutive_failures < 5)

class BackendPool:
    """Manages a pool of backend servers."""
    
    def __init__(self, config: LoadBalancerConfig):
        self.config = config
        self.logger = get_logger("backend_pool")
        self.servers: Dict[str, BackendServer] = {}
        self.healthy_servers: List[str] = []
        self.unhealthy_servers: List[str] = []
        self._lock = threading.RLock()
        
        # Health monitoring
        self._health_check_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        self._start_background_tasks()
    
    def add_server(self, server: BackendServer) -> None:
        """Add a server to the pool."""
        with self._lock:
            self.servers[server.id] = server
            if server.health_status == HealthStatus.HEALTHY:
                self.healthy_servers.append(server.id)
            else:
                self.unhealthy_servers.append(server.id)
            
            self.logger.info(f"Added server {server.id} ({server.host}:{server.port})")
    
    def remove_server(self, server_id: str) -> bool:
        """Remove a server from the pool."""
        with self._lock:
            if server_id in self.servers:
                server = self.servers[server_id]
                
                # Remove from appropriate lists
                if server_id in self.healthy_servers:
                    self.healthy_servers.remove(server_id)
                if server_id in self.unhealthy_servers:
                    self.unhealthy_servers.remove(server_id)
                
                del self.servers[server_id]
                self.logger.info(f"Removed server {server_id}")
                return True
        
        return False
    
    def get_server(self, server_id: str) -> Optional[BackendServer]:
        """Get a server by ID."""
        return self.servers.get(server_id)
    
    def get_healthy_servers(self) -> List[BackendServer]:
        """Get list of healthy servers."""
        with self._lock:
            return [self.servers[server_id] for server_id in self.healthy_servers 
                   if server_id in self.servers]
    
    def get_available_servers(self) -> List[BackendServer]:
        """Get list of available servers."""
        with self._lock:
            return [server for server in self.servers.values() if server.is_available]
    
    def update_server_health(self, server_id: str, health_status: HealthStatus) -> None:
        """Update server health status."""
        with self._lock:
            if server_id in self.servers:
                server = self.servers[server_id]
                old_status = server.health_status
                server.health_status = health_status
                server.last_health_check = time.time()
                
                # Update lists
                if health_status == HealthStatus.HEALTHY:
                    if server_id in self.unhealthy_servers:
                        self.unhealthy_servers.remove(server_id)
                    if server_id not in self.healthy_servers:
                        self.healthy_servers.append(server_id)
                else:
                    if server_id in self.healthy_servers:
                        self.healthy_servers.remove(server_id)
                    if server_id not in self.unhealthy_servers:
                        self.unhealthy_servers.append(server_id)
                
                if old_status != health_status:
                    self.logger.info(f"Server {server_id} health changed: {old_status.value} -> {health_status.value}")
    
    async def _health_check_loop(self):
        """Background health check loop."""
        while not self._shutdown_event.is_set():
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.config.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(60)
    
    async def _perform_health_checks(self):
        """Perform health checks on all servers."""
        health_check_tasks = []
        
        for server in self.servers.values():
            task = asyncio.create_task(self._check_server_health(server))
            health_check_tasks.append(task)
        
        if health_check_tasks:
            await asyncio.gather(*health_check_tasks, return_exceptions=True)
    
    async def _check_server_health(self, server: BackendServer) -> None:
        """Check health of a specific server."""
        try:
            # Simple TCP connection check
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(server.host, server.port),
                timeout=self.config.health_check_timeout
            )
            writer.close()
            await writer.wait_closed()
            
            # Server is healthy
            server.consecutive_failures = 0
            self.update_server_health(server.id, HealthStatus.HEALTHY)
            
        except Exception as e:
            # Server is unhealthy
            server.consecutive_failures += 1
            server.last_failure_time = time.time()
            
            if server.consecutive_failures >= 3:
                self.update_server_health(server.id, HealthStatus.UNHEALTHY)
            else:
                self.update_server_health(server.id, HealthStatus.DEGRADED)
    
    def _start_background_tasks(self):
        """Start background tasks."""
        if self._health_check_task is None or self._health_check_task.done():
            self._health_check_task = asyncio.create_task(self._health_check_loop())
    
    async def shutdown(self):
        """Shutdown the backend pool."""
        self._shutdown_event.set()
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

# =============================================================================
# Load Balancing Strategies
# =============================================================================

class LoadBalancingStrategy(ABC):
    """Abstract base class for load balancing strategies."""
    
    def __init__(self, backend_pool: BackendPool):
        self.backend_pool = backend_pool
    
    @abstractmethod
    async def select_server(self, request: Dict[str, Any]) -> Optional[BackendServer]:
        """Select a server for the request."""
        pass

class RoundRobinStrategy(LoadBalancingStrategy):
    """Round-robin load balancing strategy."""
    
    def __init__(self, backend_pool: BackendPool):
        super().__init__(backend_pool)
        self.current_index = 0
        self._lock = threading.Lock()
    
    async def select_server(self, request: Dict[str, Any]) -> Optional[BackendServer]:
        """Select next server in round-robin fashion."""
        available_servers = self.backend_pool.get_available_servers()
        if not available_servers:
            return None
        
        with self._lock:
            server = available_servers[self.current_index % len(available_servers)]
            self.current_index += 1
            return server

class LeastConnectionsStrategy(LoadBalancingStrategy):
    """Least connections load balancing strategy."""
    
    async def select_server(self, request: Dict[str, Any]) -> Optional[BackendServer]:
        """Select server with least connections."""
        available_servers = self.backend_pool.get_available_servers()
        if not available_servers:
            return None
        
        return min(available_servers, key=lambda s: s.current_connections)

class WeightedRoundRobinStrategy(LoadBalancingStrategy):
    """Weighted round-robin load balancing strategy."""
    
    def __init__(self, backend_pool: BackendPool):
        super().__init__(backend_pool)
        self.current_weight = 0
        self.current_index = 0
        self._lock = threading.Lock()
    
    async def select_server(self, request: Dict[str, Any]) -> Optional[BackendServer]:
        """Select server using weighted round-robin."""
        available_servers = self.backend_pool.get_available_servers()
        if not available_servers:
            return None
        
        with self._lock:
            while True:
                server = available_servers[self.current_index % len(available_servers)]
                self.current_weight += 1
                
                if self.current_weight >= server.weight:
                    self.current_weight = 0
                    self.current_index += 1
                
                if server.is_available:
                    return server

class LeastResponseTimeStrategy(LoadBalancingStrategy):
    """Least response time load balancing strategy."""
    
    async def select_server(self, request: Dict[str, Any]) -> Optional[BackendServer]:
        """Select server with lowest average response time."""
        available_servers = self.backend_pool.get_available_servers()
        if not available_servers:
            return None
        
        return min(available_servers, key=lambda s: s.average_response_time)

class ConsistentHashStrategy(LoadBalancingStrategy):
    """Consistent hashing load balancing strategy."""
    
    def __init__(self, backend_pool: BackendPool, virtual_nodes: int = 150):
        super().__init__(backend_pool)
        self.virtual_nodes = virtual_nodes
        self.hash_ring: Dict[int, str] = {}
        self._build_hash_ring()
    
    def _build_hash_ring(self):
        """Build the consistent hash ring."""
        for server in self.backend_pool.servers.values():
            for i in range(self.virtual_nodes):
                virtual_node_key = f"{server.id}-{i}"
                hash_value = self._hash(virtual_node_key)
                self.hash_ring[hash_value] = server.id
    
    def _hash(self, key: str) -> int:
        """Generate hash for a key."""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
    
    async def select_server(self, request: Dict[str, Any]) -> Optional[BackendServer]:
        """Select server using consistent hashing."""
        if not self.hash_ring:
            return None
        
        # Use request ID or generate hash from request
        request_key = str(request.get("id", hash(str(request))))
        request_hash = self._hash(request_key)
        
        # Find the next server in the hash ring
        sorted_hashes = sorted(self.hash_ring.keys())
        for hash_value in sorted_hashes:
            if hash_value >= request_hash:
                server_id = self.hash_ring[hash_value]
                server = self.backend_pool.get_server(server_id)
                if server and server.is_available:
                    return server
        
        # Wrap around to the first server
        if sorted_hashes:
            server_id = self.hash_ring[sorted_hashes[0]]
            server = self.backend_pool.get_server(server_id)
            if server and server.is_available:
                return server
        
        return None

class AdaptiveStrategy(LoadBalancingStrategy):
    """Adaptive load balancing strategy that learns from performance."""
    
    def __init__(self, backend_pool: BackendPool, window_size: int = 100):
        super().__init__(backend_pool)
        self.window_size = window_size
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.strategy_weights = {
            "round_robin": 1.0,
            "least_connections": 1.0,
            "least_response_time": 1.0,
            "weighted": 1.0
        }
    
    async def select_server(self, request: Dict[str, Any]) -> Optional[BackendServer]:
        """Select server using adaptive strategy."""
        available_servers = self.backend_pool.get_available_servers()
        if not available_servers:
            return None
        
        # Analyze current performance
        self._analyze_performance()
        
        # Choose strategy based on weights
        strategy = self._choose_strategy()
        
        if strategy == "round_robin":
            return await RoundRobinStrategy(self.backend_pool).select_server(request)
        elif strategy == "least_connections":
            return await LeastConnectionsStrategy(self.backend_pool).select_server(request)
        elif strategy == "least_response_time":
            return await LeastResponseTimeStrategy(self.backend_pool).select_server(request)
        else:
            return await WeightedRoundRobinStrategy(self.backend_pool).select_server(request)
    
    def _analyze_performance(self):
        """Analyze performance and adjust strategy weights."""
        for server_id, history in self.performance_history.items():
            if len(history) >= 10:
                avg_response_time = statistics.mean(history)
                success_rate = sum(1 for h in history if h > 0) / len(history)
                
                # Adjust weights based on performance
                if avg_response_time < 100 and success_rate > 0.95:
                    # Good performance - increase weight
                    self.strategy_weights["least_response_time"] *= 1.1
                elif avg_response_time > 500 or success_rate < 0.8:
                    # Poor performance - decrease weight
                    self.strategy_weights["least_response_time"] *= 0.9
    
    def _choose_strategy(self) -> str:
        """Choose strategy based on current weights."""
        total_weight = sum(self.strategy_weights.values())
        if total_weight == 0:
            return "round_robin"
        
        # Normalize weights
        normalized_weights = {k: v / total_weight for k, v in self.strategy_weights.items()}
        
        # Choose strategy probabilistically
        rand = random.random()
        cumulative = 0
        for strategy, weight in normalized_weights.items():
            cumulative += weight
            if rand <= cumulative:
                return strategy
        
        return "round_robin"

# =============================================================================
# Main Intelligent Load Balancer
# =============================================================================

class IntelligentLoadBalancer:
    """Main intelligent load balancer coordinating all strategies."""
    
    def __init__(self, config: Optional[LoadBalancerConfig] = None):
        self.config = config or LoadBalancerConfig()
        self.logger = get_logger("intelligent_load_balancer")
        
        # Initialize components
        self.backend_pool = BackendPool(self.config)
        self.strategies = self._initialize_strategies()
        self.current_strategy = self.strategies[self.config.strategy]
        
        # Session management
        self.sticky_sessions: Dict[str, str] = {}
        self.session_timestamps: Dict[str, float] = {}
        
        # Circuit breaker
        self.circuit_breaker_states: Dict[str, Dict[str, Any]] = {}
        
        # Rate limiting
        self.request_counts: Dict[str, int] = defaultdict(int)
        self.last_reset_time = time.time()
        
        # Performance tracking
        self.request_history: deque = deque(maxlen=1000)
        self.performance_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "total_response_time": 0.0
        }
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        self._start_background_tasks()
    
    def _initialize_strategies(self) -> Dict[LoadBalancingStrategy, LoadBalancingStrategy]:
        """Initialize all load balancing strategies."""
        return {
            LoadBalancingStrategy.ROUND_ROBIN: RoundRobinStrategy(self.backend_pool),
            LoadBalancingStrategy.LEAST_CONNECTIONS: LeastConnectionsStrategy(self.backend_pool),
            LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN: WeightedRoundRobinStrategy(self.backend_pool),
            LoadBalancingStrategy.LEAST_RESPONSE_TIME: LeastResponseTimeStrategy(self.backend_pool),
            LoadBalancingStrategy.CONSISTENT_HASH: ConsistentHashStrategy(self.backend_pool),
            LoadBalancingStrategy.ADAPTIVE: AdaptiveStrategy(self.backend_pool),
            LoadBalancingStrategy.POWER_OF_TWO: RoundRobinStrategy(self.backend_pool),  # Simplified
            LoadBalancingStrategy.LEAST_LOADED: LeastConnectionsStrategy(self.backend_pool)  # Simplified
        }
    
    def add_backend_server(self, server: BackendServer) -> None:
        """Add a backend server."""
        self.backend_pool.add_server(server)
    
    def remove_backend_server(self, server_id: str) -> bool:
        """Remove a backend server."""
        return self.backend_pool.remove_server(server_id)
    
    def set_strategy(self, strategy: LoadBalancingStrategy) -> None:
        """Set the load balancing strategy."""
        if strategy in self.strategies:
            self.current_strategy = self.strategies[strategy]
            self.logger.info(f"Load balancing strategy changed to: {strategy.value}")
        else:
            self.logger.warning(f"Unknown strategy: {strategy.value}")
    
    async def route_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Route a request to an appropriate backend server."""
        start_time = time.time()
        
        try:
            # Check rate limiting
            if not self._check_rate_limit(request):
                return {"error": "Rate limit exceeded", "status": 429}
            
            # Check circuit breaker
            if self._is_circuit_open(request):
                return {"error": "Circuit breaker open", "status": 503}
            
            # Handle sticky sessions
            session_id = request.get("session_id")
            if session_id and self.config.enable_sticky_sessions:
                server_id = self.sticky_sessions.get(session_id)
                if server_id:
                    server = self.backend_pool.get_server(server_id)
                    if server and server.is_available:
                        return await self._process_request(server, request)
            
            # Select server using current strategy
            server = await self.current_strategy.select_server(request)
            if not server:
                return {"error": "No available servers", "status": 503}
            
            # Update sticky session
            if session_id and self.config.enable_sticky_sessions:
                self.sticky_sessions[session_id] = server.id
                self.session_timestamps[session_id] = time.time()
            
            # Process request
            result = await self._process_request(server, request)
            
            # Update performance metrics
            self._update_performance_metrics(True, time.time() - start_time)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Request routing failed: {e}")
            self._update_performance_metrics(False, time.time() - start_time)
            return {"error": str(e), "status": 500}
    
    async def _process_request(self, server: BackendServer, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process request on selected server."""
        # Increment connection count
        server.current_connections += 1
        server.total_requests += 1
        
        try:
            # Simulate request processing
            await asyncio.sleep(0.01)  # Simulate processing time
            
            # Simulate success/failure
            success = random.random() > 0.1  # 90% success rate
            
            if success:
                server.successful_requests += 1
                response_time = random.uniform(10, 100)  # 10-100ms
                server.total_response_time += response_time
                server.average_response_time = server.total_response_time / server.total_requests
                
                # Update circuit breaker
                self._update_circuit_breaker(server.id, True)
                
                return {
                    "server_id": server.id,
                    "status": "success",
                    "response_time": response_time,
                    "message": f"Request processed by {server.host}:{server.port}"
                }
            else:
                server.failed_requests += 1
                
                # Update circuit breaker
                self._update_circuit_breaker(server.id, False)
                
                return {
                    "server_id": server.id,
                    "status": "error",
                    "message": "Request failed"
                }
                
        finally:
            # Decrement connection count
            server.current_connections = max(0, server.current_connections - 1)
    
    def _check_rate_limit(self, request: Dict[str, Any]) -> bool:
        """Check if request is within rate limits."""
        if not self.config.enable_rate_limiting:
            return True
        
        current_time = time.time()
        client_id = request.get("client_id", "default")
        
        # Reset counter every second
        if current_time - self.last_reset_time >= 1.0:
            self.request_counts.clear()
            self.last_reset_time = current_time
        
        # Check rate limit
        if self.request_counts[client_id] >= self.config.rate_limit_per_second:
            return False
        
        self.request_counts[client_id] += 1
        return True
    
    def _is_circuit_open(self, request: Dict[str, Any]) -> bool:
        """Check if circuit breaker is open for the request."""
        if not self.config.enable_circuit_breaker:
            return False
        
        # Simplified circuit breaker check
        return False
    
    def _update_circuit_breaker(self, server_id: str, success: bool) -> None:
        """Update circuit breaker state."""
        if not self.config.enable_circuit_breaker:
            return
        
        if server_id not in self.circuit_breaker_states:
            self.circuit_breaker_states[server_id] = {
                "failure_count": 0,
                "success_count": 0,
                "last_failure_time": 0.0,
                "state": "closed"
            }
        
        state = self.circuit_breaker_states[server_id]
        
        if success:
            state["success_count"] += 1
            state["failure_count"] = 0
        else:
            state["failure_count"] += 1
            state["last_failure_time"] = time.time()
            
            if state["failure_count"] >= self.config.circuit_breaker_threshold:
                state["state"] = "open"
    
    def _update_performance_metrics(self, success: bool, response_time: float) -> None:
        """Update performance metrics."""
        self.performance_metrics["total_requests"] += 1
        
        if success:
            self.performance_metrics["successful_requests"] += 1
        else:
            self.performance_metrics["failed_requests"] += 1
        
        self.performance_metrics["total_response_time"] += response_time
        self.performance_metrics["average_response_time"] = (
            self.performance_metrics["total_response_time"] / 
            self.performance_metrics["total_requests"]
        )
        
        # Store in history
        self.request_history.append({
            "timestamp": time.time(),
            "success": success,
            "response_time": response_time
        })
    
    async def get_load_balancer_status(self) -> Dict[str, Any]:
        """Get load balancer status and statistics."""
        return {
            "strategy": self.config.strategy.value,
            "backend_servers": len(self.backend_pool.servers),
            "healthy_servers": len(self.backend_pool.healthy_servers),
            "unhealthy_servers": len(self.backend_pool.unhealthy_servers),
            "sticky_sessions": len(self.sticky_sessions),
            "performance_metrics": self.performance_metrics,
            "rate_limiting": {
                "enabled": self.config.enable_rate_limiting,
                "current_requests": dict(self.request_counts)
            },
            "circuit_breaker": {
                "enabled": self.config.enable_circuit_breaker,
                "states": self.circuit_breaker_states
            }
        }
    
    async def _cleanup_loop(self):
        """Background cleanup loop."""
        while not self._shutdown_event.is_set():
            try:
                # Clean up expired sticky sessions
                current_time = time.time()
                expired_sessions = [
                    session_id for session_id, timestamp in self.session_timestamps.items()
                    if current_time - timestamp > self.config.sticky_session_timeout
                ]
                
                for session_id in expired_sessions:
                    del self.sticky_sessions[session_id]
                    del self.session_timestamps[session_id]
                
                # Clean up old request history
                if len(self.request_history) > 1000:
                    # Keep only last 1000 requests
                    while len(self.request_history) > 1000:
                        self.request_history.popleft()
                
                await asyncio.sleep(60)  # Cleanup every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(120)
    
    def _start_background_tasks(self):
        """Start background tasks."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def shutdown(self):
        """Shutdown the load balancer."""
        self.logger.info("Shutting down intelligent load balancer...")
        self._shutdown_event.set()
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        await self.backend_pool.shutdown()
        
        self.logger.info("Intelligent load balancer shutdown complete")

# =============================================================================
# Factory Functions
# =============================================================================

def create_intelligent_load_balancer(config: Optional[LoadBalancerConfig] = None) -> IntelligentLoadBalancer:
    """Create an intelligent load balancer instance."""
    return IntelligentLoadBalancer(config)

# Export main classes
__all__ = [
    "IntelligentLoadBalancer",
    "BackendPool",
    "BackendServer",
    "LoadBalancerConfig",
    "LoadBalancingStrategy",
    "HealthStatus",
    "RoundRobinStrategy",
    "LeastConnectionsStrategy",
    "WeightedRoundRobinStrategy",
    "LeastResponseTimeStrategy",
    "ConsistentHashStrategy",
    "AdaptiveStrategy",
    "create_intelligent_load_balancer"
]


