"""
Refactored Router Engine for the Blaze AI module.

High-performance routing engine with advanced load balancing strategies,
intelligent request distribution, and dynamic resource management.
"""

from __future__ import annotations

import asyncio
import time
import random
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json

from . import Engine, EngineStatus
from ..core.interfaces import CoreConfig
from ..utils.logging import get_logger

class LoadBalancingStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"
    IP_HASH = "ip_hash"
    ADAPTIVE = "adaptive"

@dataclass
class RouterConfig:
    strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN
    health_check_interval: float = 30.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    max_retries: int = 3
    retry_delay: float = 1.0
    enable_sticky_sessions: bool = False
    sticky_session_timeout: float = 300.0
    enable_adaptive_routing: bool = True
    adaptive_window_size: int = 100
    enable_health_checks: bool = True
    health_check_timeout: float = 5.0

@dataclass
class RouteTarget:
    id: str
    engine: Engine
    weight: int = 1
    max_connections: int = 100
    current_connections: int = 0
    response_times: List[float] = field(default_factory=list)
    error_count: int = 0
    last_health_check: float = 0.0
    is_healthy: bool = True
    circuit_breaker_state: str = "CLOSED"
    circuit_breaker_failures: int = 0
    circuit_breaker_last_failure: float = 0.0

@dataclass
class RoutingRequest:
    operation: str
    params: Dict[str, Any]
    source_ip: Optional[str] = None
    session_id: Optional[str] = None
    priority: int = 1
    timeout: Optional[float] = None
    retry_count: int = 0

@dataclass
class RoutingResponse:
    target_id: str
    result: Any
    processing_time: float
    retry_count: int = 0
    error: Optional[str] = None

class HealthChecker:
    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout
        self._lock = asyncio.Lock()
    
    async def check_health(self, target: RouteTarget) -> bool:
        async with self._lock:
            try:
                start_time = time.time()
                health_status = await asyncio.wait_for(
                    target.engine.get_health_status(),
                    timeout=self.timeout
                )
                
                target.last_health_check = time.time()
                target.is_healthy = health_status.get("status") == "healthy"
                
                return target.is_healthy
                
            except asyncio.TimeoutError:
                target.is_healthy = False
                target.last_health_check = time.time()
                return False
            except Exception:
                target.is_healthy = False
                target.last_health_check = time.time()
                return False

class CircuitBreaker:
    def __init__(self, threshold: int = 5, timeout: float = 60.0):
        self.threshold = threshold
        self.timeout = timeout
        self._lock = asyncio.Lock()
    
    async def should_allow_request(self, target: RouteTarget) -> bool:
        async with self._lock:
            current_time = time.time()
            
            if target.circuit_breaker_state == "OPEN":
                if current_time - target.circuit_breaker_last_failure > self.timeout:
                    target.circuit_breaker_state = "HALF_OPEN"
                    return True
                return False
            
            elif target.circuit_breaker_state == "HALF_OPEN":
                return True
            
            return True
    
    async def record_success(self, target: RouteTarget):
        async with self._lock:
            if target.circuit_breaker_state == "HALF_OPEN":
                target.circuit_breaker_state = "CLOSED"
                target.circuit_breaker_failures = 0
    
    async def record_failure(self, target: RouteTarget):
        async with self._lock:
            target.circuit_breaker_failures += 1
            target.circuit_breaker_last_failure = time.time()
            
            if target.circuit_breaker_failures >= self.threshold:
                target.circuit_breaker_state = "OPEN"

class LoadBalancer:
    def __init__(self, strategy: LoadBalancingStrategy):
        self.strategy = strategy
        self._round_robin_index = 0
        self._lock = asyncio.Lock()
    
    async def select_target(self, targets: List[RouteTarget], request: RoutingRequest) -> Optional[RouteTarget]:
        if not targets:
            return None
        
        healthy_targets = [t for t in targets if t.is_healthy]
        if not healthy_targets:
            return None
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return await self._round_robin(healthy_targets)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return await self._least_connections(healthy_targets)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return await self._weighted_round_robin(healthy_targets)
        elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return await self._least_response_time(healthy_targets)
        elif self.strategy == LoadBalancingStrategy.IP_HASH:
            return await self._ip_hash(healthy_targets, request)
        elif self.strategy == LoadBalancingStrategy.ADAPTIVE:
            return await self._adaptive(healthy_targets, request)
        else:
            return healthy_targets[0]
    
    async def _round_robin(self, targets: List[RouteTarget]) -> RouteTarget:
        async with self._lock:
            target = targets[self._round_robin_index % len(targets)]
            self._round_robin_index = (self._round_robin_index + 1) % len(targets)
            return target
    
    async def _least_connections(self, targets: List[RouteTarget]) -> RouteTarget:
        return min(targets, key=lambda t: t.current_connections)
    
    async def _weighted_round_robin(self, targets: List[RouteTarget]) -> RouteTarget:
        total_weight = sum(t.weight for t in targets)
        if total_weight == 0:
            return targets[0]
        
        async with self._lock:
            target = targets[self._round_robin_index % len(targets)]
            self._round_robin_index = (self._round_robin_index + 1) % len(targets)
            return target
    
    async def _least_response_time(self, targets: List[RouteTarget]) -> RouteTarget:
        def get_avg_response_time(target: RouteTarget) -> float:
            if not target.response_times:
                return float('inf')
            return sum(target.response_times) / len(target.response_times)
        
        return min(targets, key=get_avg_response_time)
    
    async def _ip_hash(self, targets: List[RouteTarget], request: RoutingRequest) -> RouteTarget:
        if not request.source_ip:
            return targets[0]
        
        hash_value = int(hashlib.md5(request.source_ip.encode()).hexdigest(), 16)
        return targets[hash_value % len(targets)]
    
    async def _adaptive(self, targets: List[RouteTarget], request: RoutingRequest) -> RouteTarget:
        def calculate_score(target: RouteTarget) -> float:
            connection_score = 1.0 / (target.current_connections + 1)
            
            if target.response_times:
                avg_response_time = sum(target.response_times) / len(target.response_times)
                response_score = 1.0 / (avg_response_time + 1)
            else:
                response_score = 1.0
            
            error_score = 1.0 / (target.error_count + 1)
            
            return connection_score * response_score * error_score * target.weight
        
        return max(targets, key=calculate_score)

class RouterEngine(Engine):
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.router_config = RouterConfig(**config)
        self.targets: Dict[str, RouteTarget] = {}
        self.load_balancer = LoadBalancer(self.router_config.strategy)
        self.health_checker = HealthChecker(self.router_config.health_check_timeout)
        self.circuit_breaker = CircuitBreaker(
            self.router_config.circuit_breaker_threshold,
            self.router_config.circuit_breaker_timeout
        )
        self.session_map: Dict[str, str] = {}
        self._health_check_task: Optional[asyncio.Task] = None
        self._adaptive_task: Optional[asyncio.Task] = None
    
    async def _initialize_engine(self) -> None:
        if self.router_config.enable_health_checks:
            self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        if self.router_config.enable_adaptive_routing:
            self._adaptive_task = asyncio.create_task(self._adaptive_routing_loop())
    
    async def add_target(self, target_id: str, engine: Engine, weight: int = 1, max_connections: int = 100):
        target = RouteTarget(
            id=target_id,
            engine=engine,
            weight=weight,
            max_connections=max_connections
        )
        self.targets[target_id] = target
    
    async def remove_target(self, target_id: str):
        if target_id in self.targets:
            del self.targets[target_id]
    
    async def _execute_operation(self, operation: str, params: Dict[str, Any]) -> Any:
        if operation == "route":
            return await self._route_request(params)
        elif operation == "get_targets":
            return await self._get_targets_info()
        elif operation == "update_target_weight":
            return await self._update_target_weight(params)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    async def _route_request(self, params: Dict[str, Any]) -> RoutingResponse:
        request = RoutingRequest(**params)
        
        if self.router_config.enable_sticky_sessions and request.session_id:
            target_id = self.session_map.get(request.session_id)
            if target_id and target_id in self.targets:
                target = self.targets[target_id]
                if target.is_healthy and await self.circuit_breaker.should_allow_request(target):
                    return await self._execute_on_target(target, request)
        
        target = await self._select_target(request)
        if not target:
            raise RuntimeError("No healthy targets available")
        
        if self.router_config.enable_sticky_sessions and request.session_id:
            self.session_map[request.session_id] = target.id
        
        return await self._execute_on_target(target, request)
    
    async def _select_target(self, request: RoutingRequest) -> Optional[RouteTarget]:
        available_targets = []
        
        for target in self.targets.values():
            if not target.is_healthy:
                continue
            
            if not await self.circuit_breaker.should_allow_request(target):
                continue
            
            if target.current_connections >= target.max_connections:
                continue
            
            available_targets.append(target)
        
        if not available_targets:
            return None
        
        return await self.load_balancer.select_target(available_targets, request)
    
    async def _execute_on_target(self, target: RouteTarget, request: RoutingRequest) -> RoutingResponse:
        start_time = time.time()
        
        try:
            target.current_connections += 1
            
            result = await asyncio.wait_for(
                target.engine.execute(request.operation, request.params),
                timeout=request.timeout or 30.0
            )
            
            processing_time = time.time() - start_time
            
            await self.circuit_breaker.record_success(target)
            
            target.response_times.append(processing_time)
            if len(target.response_times) > self.router_config.adaptive_window_size:
                target.response_times.pop(0)
            
            return RoutingResponse(
                target_id=target.id,
                result=result,
                processing_time=processing_time,
                retry_count=request.retry_count
            )
            
        except Exception as e:
            await self.circuit_breaker.record_failure(target)
            target.error_count += 1
            
            if request.retry_count < self.router_config.max_retries:
                await asyncio.sleep(self.router_config.retry_delay)
                request.retry_count += 1
                return await self._route_request({
                    "operation": request.operation,
                    "params": request.params,
                    "source_ip": request.source_ip,
                    "session_id": request.session_id,
                    "priority": request.priority,
                    "timeout": request.timeout,
                    "retry_count": request.retry_count
                })
            
            return RoutingResponse(
                target_id=target.id,
                result=None,
                processing_time=time.time() - start_time,
                retry_count=request.retry_count,
                error=str(e)
            )
        
        finally:
            target.current_connections = max(0, target.current_connections - 1)
    
    async def _get_targets_info(self) -> Dict[str, Any]:
        return {
            "targets": {
                target_id: {
                    "id": target.id,
                    "weight": target.weight,
                    "current_connections": target.current_connections,
                    "max_connections": target.max_connections,
                    "is_healthy": target.is_healthy,
                    "error_count": target.error_count,
                    "circuit_breaker_state": target.circuit_breaker_state,
                    "last_health_check": target.last_health_check
                }
                for target_id, target in self.targets.items()
            },
            "strategy": self.router_config.strategy.value,
            "total_targets": len(self.targets),
            "healthy_targets": sum(1 for t in self.targets.values() if t.is_healthy)
        }
    
    async def _update_target_weight(self, params: Dict[str, Any]) -> bool:
        target_id = params.get("target_id")
        new_weight = params.get("weight")
        
        if target_id not in self.targets or new_weight is None:
            return False
        
        self.targets[target_id].weight = new_weight
        return True
    
    async def _health_check_loop(self):
        while not self._shutdown_event.is_set():
            try:
                for target in self.targets.values():
                    await self.health_checker.check_health(target)
                
                await asyncio.sleep(self.router_config.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(5.0)
    
    async def _adaptive_routing_loop(self):
        while not self._shutdown_event.is_set():
            try:
                await self._update_adaptive_weights()
                await asyncio.sleep(60.0)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Adaptive routing loop error: {e}")
                await asyncio.sleep(30.0)
    
    async def _update_adaptive_weights(self):
        for target in self.targets.values():
            if not target.response_times:
                continue
            
            avg_response_time = sum(target.response_times) / len(target.response_times)
            error_rate = target.error_count / max(len(target.response_times), 1)
            
            new_weight = max(1, int(target.weight * (1.0 - error_rate) * (1.0 / avg_response_time)))
            target.weight = min(new_weight, 10)
    
    async def shutdown(self):
        await super().shutdown()
        
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        if self._adaptive_task:
            self._adaptive_task.cancel()
            try:
                await self._adaptive_task
            except asyncio.CancelledError:
                pass


