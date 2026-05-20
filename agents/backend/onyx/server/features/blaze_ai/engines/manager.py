"""
Engine Manager for Blaze AI System.

This module provides centralized engine management, monitoring,
and orchestration capabilities.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Awaitable
from contextlib import asynccontextmanager

from .base import Engine, EngineStatus, EngineType, EnginePriority
from .factory import EngineFactory, get_engine_factory
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from ..core.interfaces import CoreConfig, SystemHealth, HealthStatus
from ..utils.logging import get_logger

# =============================================================================
# Engine Manager Configuration
# =============================================================================

@dataclass
class EngineManagerConfig:
    """Configuration for the engine manager."""
    max_concurrent_requests: int = 100
    monitoring_interval: float = 30.0
    health_check_timeout: float = 10.0
    enable_auto_recovery: bool = True
    auto_recovery_threshold: int = 3
    enable_load_balancing: bool = True
    enable_circuit_breaker: bool = True
    default_circuit_breaker_config: Optional[CircuitBreakerConfig] = None
    enable_metrics_collection: bool = True
    metrics_retention_period: float = 3600.0  # 1 hour
    enable_auto_scaling: bool = False
    min_engines_per_type: int = 1
    max_engines_per_type: int = 5
    scaling_threshold: float = 0.8  # 80% utilization
    
    def __post_init__(self):
        if self.default_circuit_breaker_config is None:
            self.default_circuit_breaker_config = CircuitBreakerConfig()

@dataclass
class EngineInstance:
    """Information about an engine instance."""
    name: str
    engine: Engine
    template_name: str
    created_at: float
    last_used: float
    usage_count: int = 0
    error_count: int = 0
    circuit_breaker: Optional[CircuitBreaker] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

# =============================================================================
# Engine Manager Implementation
# =============================================================================

class EngineManager:
    """Centralized engine manager for the Blaze AI module."""
    
    def __init__(self, config: Optional[CoreConfig] = None, factory: Optional[EngineFactory] = None):
        self.config = config or CoreConfig()
        self.manager_config = EngineManagerConfig()
        self.factory = factory or get_engine_factory()
        self.logger = get_logger("engine_manager")
        
        # Engine management
        self.engines: Dict[str, EngineInstance] = {}
        self.engine_groups: Dict[EngineType, List[str]] = defaultdict(list)
        self.engine_metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # System state
        self.system_health = SystemHealth()
        self._semaphore = asyncio.Semaphore(self.manager_config.max_concurrent_requests)
        self._shutdown_event = asyncio.Event()
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._recovery_task: Optional[asyncio.Task] = None
        self._scaling_task: Optional[asyncio.Task] = None
        self._metrics_task: Optional[asyncio.Task] = None
        
        # Initialize manager
        self._initialize_manager()
    
    def _initialize_manager(self):
        """Initialize the engine manager."""
        self.logger.info("Initializing engine manager...")
        
        # Create default engines
        self._create_default_engines()
        
        # Start background tasks
        self._start_background_tasks()
        
        self.logger.info("Engine manager initialized successfully")
    
    def _create_default_engines(self):
        """Create default engines using the factory."""
        default_engines = [
            ("llm", "llm", {}),
            ("diffusion", "diffusion", {}),
            ("router", "router", {})
        ]
        
        for engine_name, template_name, config in default_engines:
            try:
                self.create_engine(template_name, config, engine_name)
            except Exception as e:
                self.logger.warning(f"Failed to create default engine {engine_name}: {e}")
    
    def create_engine(self, 
                     template_name: str, 
                     config: Optional[Dict[str, Any]] = None,
                     instance_name: Optional[str] = None) -> str:
        """Create a new engine instance."""
        try:
            # Create engine using factory
            engine = self.factory.create_engine(template_name, config, instance_name)
            
            # Create circuit breaker if enabled
            circuit_breaker = None
            if self.manager_config.enable_circuit_breaker:
                circuit_breaker = CircuitBreaker(self.manager_config.default_circuit_breaker_config)
            
            # Create engine instance
            instance = EngineInstance(
                name=engine.name,
                engine=engine,
                template_name=template_name,
                created_at=time.time(),
                last_used=time.time(),
                circuit_breaker=circuit_breaker,
                metadata=self.factory.get_engine_metadata(engine.name) or {}
            )
            
            # Register engine
            self.engines[engine.name] = instance
            engine_type = engine._get_engine_type()
            self.engine_groups[engine_type].append(engine.name)
            
            # Initialize engine
            asyncio.create_task(self._initialize_engine_safe(instance))
            
            self.logger.info(f"Created engine instance: {engine.name}")
            return engine.name
            
        except Exception as e:
            self.logger.error(f"Failed to create engine {template_name}: {e}")
            raise
    
    async def _initialize_engine_safe(self, instance: EngineInstance):
        """Safely initialize an engine instance."""
        try:
            await instance.engine.initialize()
            self.logger.info(f"Engine {instance.name} initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize engine {instance.name}: {e}")
            instance.error_count += 1
    
    def register_engine(self, name: str, engine: Engine, template_name: str = "custom"):
        """Register an existing engine instance."""
        if name in self.engines:
            raise ValueError(f"Engine '{name}' already registered")
        
        # Create circuit breaker if enabled
        circuit_breaker = None
        if self.manager_config.enable_circuit_breaker:
            circuit_breaker = CircuitBreaker(self.manager_config.default_circuit_breaker_config)
        
        # Create engine instance
        instance = EngineInstance(
            name=name,
            engine=engine,
            template_name=template_name,
            created_at=time.time(),
            last_used=time.time(),
            circuit_breaker=circuit_breaker
        )
        
        # Register engine
        self.engines[name] = instance
        engine_type = engine._get_engine_type()
        self.engine_groups[engine_type].append(name)
        
        self.logger.info(f"Registered engine: {name}")
    
    def unregister_engine(self, name: str) -> bool:
        """Unregister an engine instance."""
        if name not in self.engines:
            return False
        
        instance = self.engines[name]
        engine_type = instance.engine._get_engine_type()
        
        # Remove from groups
        if name in self.engine_groups[engine_type]:
            self.engine_groups[engine_type].remove(name)
        
        # Remove engine
        del self.engines[name]
        
        # Clean up metrics
        if name in self.engine_metrics:
            del self.engine_metrics[name]
        
        self.logger.info(f"Unregistered engine: {name}")
        return True
    
    @asynccontextmanager
    async def _engine_context(self, engine_name: str):
        """Context manager for engine operations."""
        if engine_name not in self.engines:
            raise ValueError(f"Engine '{engine_name}' not found")
        
        async with self._semaphore:
            instance = self.engines[engine_name]
            try:
                # Update usage statistics
                instance.last_used = time.time()
                instance.usage_count += 1
                
                yield instance
            except Exception as e:
                instance.error_count += 1
                self.logger.error(f"Engine {engine_name} operation failed: {e}")
                raise
    
    async def dispatch(self, engine_name: str, operation: str, params: Dict[str, Any]) -> Any:
        """Dispatch request to specific engine with context management."""
        async with self._engine_context(engine_name) as instance:
            # Use circuit breaker if available
            if instance.circuit_breaker:
                return await instance.circuit_breaker.call(
                    instance.engine.execute, operation, params
                )
            else:
                return await instance.engine.execute(operation, params)
    
    async def dispatch_batch(self, requests: List[Dict[str, Any]]) -> List[Any]:
        """Dispatch multiple requests in batch with improved error handling."""
        tasks = []
        for i, request in enumerate(requests):
            engine_name = request.get("engine")
            operation = request.get("operation")
            params = request.get("params", {})
            
            if engine_name and operation:
                task = self._dispatch_single_with_metadata(i, engine_name, operation, params)
                tasks.append(task)
            else:
                tasks.append(asyncio.create_task(self._return_error(f"Invalid request format at index {i}")))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
    
    async def _dispatch_single_with_metadata(self, index: int, engine_name: str, operation: str, params: Dict[str, Any]):
        """Dispatch single request with metadata tracking."""
        try:
            result = await self.dispatch(engine_name, operation, params)
            return {"index": index, "success": True, "result": result}
        except Exception as e:
            return {"index": index, "success": False, "error": str(e)}
    
    async def _return_error(self, message: str):
        """Return error result for invalid requests."""
        return {"success": False, "error": message}
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get status of all engines."""
        status = {}
        for name, instance in self.engines.items():
            engine = instance.engine
            status[name] = {
                "template": instance.template_name,
                "status": engine.status.value,
                "created_at": instance.created_at,
                "last_used": instance.last_used,
                "usage_count": instance.usage_count,
                "error_count": instance.error_count,
                "circuit_breaker_state": instance.circuit_breaker.get_state().value if instance.circuit_breaker else "disabled",
                "metrics": engine.get_metrics(),
                "capabilities": {
                    "supported_operations": engine.capabilities.supported_operations,
                    "max_batch_size": engine.capabilities.max_batch_size,
                    "max_concurrent_requests": engine.capabilities.max_concurrent_requests
                }
            }
        return status
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide metrics."""
        total_engines = len(self.engines)
        healthy_engines = len([e for e in self.engines.values() if e.engine.is_healthy()])
        total_requests = sum(e.engine.get_metrics().get("total_requests", 0) for e in self.engines.values())
        total_errors = sum(e.error_count for e in self.engines.values())
        
        return {
            "total_engines": total_engines,
            "healthy_engines": healthy_engines,
            "unhealthy_engines": total_engines - healthy_engines,
            "health_ratio": healthy_engines / total_engines if total_engines > 0 else 0.0,
            "total_requests": total_requests,
            "total_errors": total_errors,
            "error_rate": total_errors / total_requests if total_requests > 0 else 0.0,
            "engine_groups": {t.value: len(names) for t, names in self.engine_groups.items()}
        }
    
    def get_engines_by_type(self, engine_type: EngineType) -> List[str]:
        """Get list of engine names by type."""
        return self.engine_groups.get(engine_type, [])
    
    def get_engines_by_priority(self, priority: EnginePriority) -> List[str]:
        """Get list of engine names by priority."""
        return [
            name for name, instance in self.engines.items()
            if instance.engine.metadata.priority == priority
        ]
    
    def find_engine_for_operation(self, operation: str, engine_type: Optional[EngineType] = None) -> Optional[str]:
        """Find the best engine for a specific operation."""
        candidates = []
        
        for name, instance in self.engines.items():
            engine = instance.engine
            
            # Check if engine can handle the operation
            if not engine.can_handle_operation(operation):
                continue
            
            # Check engine type if specified
            if engine_type and engine.metadata.type != engine_type:
                continue
            
            # Check if engine is healthy
            if not engine.is_healthy():
                continue
            
            # Calculate score based on priority, health, and utilization
            score = self._calculate_engine_score(instance)
            candidates.append((name, score))
        
        if not candidates:
            return None
        
        # Return engine with highest score
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    
    def _calculate_engine_score(self, instance: EngineInstance) -> float:
        """Calculate engine score for load balancing."""
        engine = instance.engine
        
        # Base score from priority
        priority_scores = {
            EnginePriority.CRITICAL: 100,
            EnginePriority.HIGH: 80,
            EnginePriority.NORMAL: 60,
            EnginePriority.LOW: 40,
            EnginePriority.BACKGROUND: 20
        }
        score = priority_scores.get(engine.metadata.priority, 50)
        
        # Health bonus
        if engine.is_healthy():
            score += 20
        
        # Utilization penalty (lower utilization = higher score)
        utilization = engine.get_utilization()
        score -= utilization * 0.3
        
        # Error penalty
        score -= instance.error_count * 5
        
        return max(0, score)
    
    def _start_background_tasks(self):
        """Start background monitoring and recovery tasks."""
        if self._monitoring_task is None or self._monitoring_task.done():
            self._monitoring_task = asyncio.create_task(self._monitor_engines())
        
        if self.manager_config.enable_auto_recovery and (self._recovery_task is None or self._recovery_task.done()):
            self._recovery_task = asyncio.create_task(self._auto_recovery_loop())
        
        if self.manager_config.enable_auto_scaling and (self._scaling_task is None or self._scaling_task.done()):
            self._scaling_task = asyncio.create_task(self._auto_scaling_loop())
        
        if self.manager_config.enable_metrics_collection and (self._metrics_task is None or self._metrics_task.done()):
            self._metrics_task = asyncio.create_task(self._metrics_collection_loop())
    
    async def _monitor_engines(self):
        """Background engine monitoring with improved error handling."""
        while not self._shutdown_event.is_set():
            try:
                for name, instance in self.engines.items():
                    try:
                        health_status = instance.engine.get_health_status()
                        await self.system_health.update_component(
                            name,
                            health_status.status,
                            health_status.message,
                            health_status.details
                        )
                    except Exception as e:
                        self.logger.error(f"Health check failed for engine {name}: {e}")
                
                await asyncio.sleep(self.manager_config.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Engine monitoring error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _auto_recovery_loop(self):
        """Automatic engine recovery loop."""
        while not self._shutdown_event.is_set():
            try:
                for name, instance in self.engines.items():
                    if instance.engine.status == EngineStatus.ERROR:
                        await self._attempt_engine_recovery(name, instance)
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Auto-recovery error: {e}")
                await asyncio.sleep(120)
    
    async def _attempt_engine_recovery(self, name: str, instance: EngineInstance):
        """Attempt to recover a failed engine."""
        try:
            self.logger.info(f"Attempting to recover engine {name}")
            await instance.engine.initialize()
            self.logger.info(f"Successfully recovered engine {name}")
        except Exception as e:
            self.logger.error(f"Failed to recover engine {name}: {e}")
    
    async def _auto_scaling_loop(self):
        """Automatic engine scaling loop."""
        while not self._shutdown_event.is_set():
            try:
                await self._check_scaling_needs()
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Auto-scaling error: {e}")
                await asyncio.sleep(600)
    
    async def _check_scaling_needs(self):
        """Check if scaling is needed and perform scaling."""
        for engine_type, engine_names in self.engine_groups.items():
            if len(engine_names) < self.manager_config.min_engines_per_type:
                # Scale up
                await self._scale_up_engines(engine_type)
            elif len(engine_names) > self.manager_config.max_engines_per_type:
                # Scale down
                await self._scale_down_engines(engine_type)
    
    async def _scale_up_engines(self, engine_type: EngineType):
        """Scale up engines of a specific type."""
        # Find template for this engine type
        template_name = self._find_template_for_type(engine_type)
        if not template_name:
            return
        
        try:
            new_engine_name = f"{engine_type.value}_{len(self.engine_groups[engine_type]) + 1}"
            self.create_engine(template_name, {}, new_engine_name)
            self.logger.info(f"Scaled up {engine_type.value} engine: {new_engine_name}")
        except Exception as e:
            self.logger.error(f"Failed to scale up {engine_type.value} engine: {e}")
    
    async def _scale_down_engines(self, engine_type: EngineType):
        """Scale down engines of a specific type."""
        engine_names = self.engine_groups[engine_type]
        if len(engine_names) <= self.manager_config.min_engines_per_type:
            return
        
        # Remove the least used engine
        least_used = min(engine_names, key=lambda name: self.engines[name].usage_count)
        if self.unregister_engine(least_used):
            self.logger.info(f"Scaled down {engine_type.value} engine: {least_used}")
    
    def _find_template_for_type(self, engine_type: EngineType) -> Optional[str]:
        """Find template name for a specific engine type."""
        for name, template in self.factory.engine_templates.items():
            if (hasattr(template.engine_class, '_get_engine_type') and
                template.engine_class._get_engine_type() == engine_type):
                return name
        return None
    
    async def _metrics_collection_loop(self):
        """Background metrics collection loop."""
        while not self._shutdown_event.is_set():
            try:
                await self._collect_engine_metrics()
                await asyncio.sleep(60)  # Collect every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(120)
    
    async def _collect_engine_metrics(self):
        """Collect metrics from all engines."""
        current_time = time.time()
        
        for name, instance in self.engines.items():
            try:
                # Get engine metrics
                engine_metrics = instance.engine.get_metrics()
                
                # Add instance metrics
                instance_metrics = {
                    "created_at": instance.created_at,
                    "last_used": instance.last_used,
                    "usage_count": instance.usage_count,
                    "error_count": instance.error_count,
                    "uptime": current_time - instance.created_at,
                    "idle_time": current_time - instance.last_used
                }
                
                # Add circuit breaker metrics if available
                if instance.circuit_breaker:
                    cb_metrics = instance.circuit_breaker.get_metrics()
                    instance_metrics["circuit_breaker"] = cb_metrics
                
                # Store combined metrics
                self.engine_metrics[name] = {
                    "engine": engine_metrics,
                    "instance": instance_metrics,
                    "timestamp": current_time
                }
                
            except Exception as e:
                self.logger.error(f"Failed to collect metrics for engine {name}: {e}")
    
    async def shutdown(self):
        """Shutdown all engines and background tasks."""
        self.logger.info("Shutting down engine manager...")
        self._shutdown_event.set()
        
        # Cancel background tasks
        for task in [self._monitoring_task, self._recovery_task, self._scaling_task, self._metrics_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Shutdown engines
        shutdown_tasks = []
        for name, instance in self.engines.items():
            shutdown_tasks.append(self._shutdown_engine_safe(name, instance))
        
        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        
        self.logger.info("Engine manager shutdown complete")
    
    async def _shutdown_engine_safe(self, name: str, instance: EngineInstance):
        """Safely shutdown an engine instance."""
        try:
            await instance.engine.shutdown()
        except Exception as e:
            self.logger.error(f"Error shutting down engine {name}: {e}")

# =============================================================================
# Global Instance Management
# =============================================================================

_default_engine_manager: Optional[EngineManager] = None

def get_engine_manager(config: Optional[CoreConfig] = None, factory: Optional[EngineFactory] = None) -> EngineManager:
    """Get the global engine manager instance."""
    global _default_engine_manager
    if _default_engine_manager is None:
        _default_engine_manager = EngineManager(config, factory)
    return _default_engine_manager

async def shutdown_engine_manager():
    """Shutdown the global engine manager."""
    global _default_engine_manager
    if _default_engine_manager:
        await _default_engine_manager.shutdown()
        _default_engine_manager = None


