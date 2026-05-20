"""
Blaze AI Engines Module v7.2.0

This module integrates the advanced engine system with the modular architecture,
providing quantum optimization, neural turbo acceleration, and real-time performance
through the modular system interface.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Type
from dataclasses import dataclass, field
from enum import Enum

from .base import BaseModule, ModuleConfig, ModuleStatus
from ..engines import (
    BlazeEngine, QuantumEngine, NeuralTurboEngine, MararealEngine, 
    HybridOptimizationEngine, EngineRegistry, EngineManager,
    EngineConfig, QuantumConfig, NeuralTurboConfig, MararealConfig,
    EngineType, OptimizationLevel, create_engine_registry, create_engine_manager
)

logger = logging.getLogger(__name__)

# ============================================================================
# ENGINE MODULE CONFIGURATION
# ============================================================================

class EngineModuleConfig(ModuleConfig):
    """Configuration for the Engines Module."""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="engines",
            module_type="ENGINE",
            priority=1,  # High priority for engine management
            **kwargs
        )
        
        # Engine-specific configurations
        self.default_engine_type: EngineType = kwargs.get("default_engine_type", EngineType.HYBRID)
        self.auto_initialize_engines: bool = kwargs.get("auto_initialize_engines", True)
        self.max_concurrent_engines: int = kwargs.get("max_concurrent_engines", 10)
        self.engine_timeout: float = kwargs.get("engine_timeout", 30.0)
        self.enable_health_monitoring: bool = kwargs.get("enable_health_monitoring", True)
        self.health_check_interval: float = kwargs.get("health_check_interval", 5.0)
        
        # Performance configurations
        self.enable_quantum_optimization: bool = kwargs.get("enable_quantum_optimization", True)
        self.enable_neural_turbo: bool = kwargs.get("enable_neural_turbo", True)
        self.enable_marareal: bool = kwargs.get("enable_marareal", True)
        self.enable_hybrid: bool = kwargs.get("enable_hybrid", True)

class EngineMetrics:
    """Metrics specific to engine operations."""
    
    def __init__(self):
        self.engines_created: int = 0
        self.engines_active: int = 0
        self.engines_shutdown: int = 0
        self.total_executions: int = 0
        self.successful_executions: int = 0
        self.failed_executions: int = 0
        self.average_execution_time: float = 0.0
        self.quantum_optimizations: int = 0
        self.neural_accelerations: int = 0
        self.real_time_executions: int = 0
        self.hybrid_optimizations: int = 0

# ============================================================================
# ENGINE MODULE IMPLEMENTATION
# ============================================================================

class EnginesModule(BaseModule):
    """
    Engines Module - Manages all Blaze AI engines through the modular system.
    
    This module provides:
    - Engine creation and management
    - Performance optimization
    - Health monitoring
    - Metrics collection
    - Task execution through various engines
    """
    
    def __init__(self, config: EngineModuleConfig):
        super().__init__(config)
        self.engine_registry: Optional[EngineRegistry] = None
        self.engine_manager: Optional[EngineManager] = None
        self.active_engines: Dict[str, BlazeEngine] = {}
        self.engine_metrics = EngineMetrics()
        self.health_monitor_task: Optional[asyncio.Task] = None
        
    async def initialize(self) -> bool:
        """Initialize the Engines Module."""
        try:
            logger.info("Initializing Engines Module...")
            
            # Create engine registry and manager
            self.engine_registry = create_engine_registry()
            self.engine_manager = create_engine_manager(self.engine_registry)
            
            # Auto-initialize default engines if configured
            if self.config.auto_initialize_engines:
                await self._initialize_default_engines()
            
            # Start health monitoring if enabled
            if self.config.enable_health_monitoring:
                self.health_monitor_task = asyncio.create_task(
                    self._health_monitor_loop()
                )
            
            self.status = ModuleStatus.ACTIVE
            logger.info("Engines Module initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Engines Module: {e}")
            self.status = ModuleStatus.ERROR
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the Engines Module."""
        try:
            logger.info("Shutting down Engines Module...")
            
            # Stop health monitoring
            if self.health_monitor_task:
                self.health_monitor_task.cancel()
                try:
                    await self.health_monitor_task
                except asyncio.CancelledError:
                    pass
            
            # Shutdown all active engines
            if self.engine_manager:
                await self.engine_manager.shutdown_all()
            
            # Clear references
            self.active_engines.clear()
            self.engine_manager = None
            self.engine_registry = None
            
            self.status = ModuleStatus.SHUTDOWN
            logger.info("Engines Module shutdown successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during Engines Module shutdown: {e}")
            return False
    
    async def _initialize_default_engines(self):
        """Initialize default engines based on configuration."""
        try:
            # Create default engine configurations
            if self.config.enable_quantum_optimization:
                quantum_config = EngineConfig(
                    engine_type=EngineType.QUANTUM,
                    optimization_level=OptimizationLevel.QUANTUM
                )
                await self.create_engine("quantum", quantum_config)
            
            if self.config.enable_neural_turbo:
                neural_config = EngineConfig(
                    engine_type=EngineType.NEURAL_TURBO,
                    optimization_level=OptimizationLevel.NEURAL_TURBO
                )
                await self.create_engine("neural_turbo", neural_config)
            
            if self.config.enable_marareal:
                marareal_config = EngineConfig(
                    engine_type=EngineType.MARAREAL,
                    optimization_level=OptimizationLevel.MARAREAL
                )
                await self.create_engine("marareal", marareal_config)
            
            if self.config.enable_hybrid:
                hybrid_config = EngineConfig(
                    engine_type=EngineType.HYBRID,
                    optimization_level=OptimizationLevel.MARAREAL
                )
                await self.create_engine("hybrid", hybrid_config)
                
        except Exception as e:
            logger.warning(f"Some default engines failed to initialize: {e}")
    
    async def create_engine(self, name: str, config: EngineConfig) -> Optional[BlazeEngine]:
        """Create and register a new engine."""
        try:
            if not self.engine_manager:
                raise RuntimeError("Engine manager not initialized")
            
            # Check if we're at the limit
            if len(self.active_engines) >= self.config.max_concurrent_engines:
                logger.warning(f"Maximum engine limit reached ({self.config.max_concurrent_engines})")
                return None
            
            # Create the engine
            engine = await self.engine_manager.create_engine(name, config)
            if engine:
                self.active_engines[name] = engine
                self.engine_metrics.engines_created += 1
                self.engine_metrics.engines_active += 1
                logger.info(f"Engine created successfully: {name}")
                return engine
            else:
                logger.error(f"Failed to create engine: {name}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating engine {name}: {e}")
            return None
    
    async def get_engine(self, name: str) -> Optional[BlazeEngine]:
        """Get an active engine by name."""
        return self.active_engines.get(name)
    
    async def shutdown_engine(self, name: str) -> bool:
        """Shutdown and remove an engine."""
        try:
            if not self.engine_manager:
                return False
            
            success = await self.engine_manager.shutdown_engine(name)
            if success and name in self.active_engines:
                del self.active_engines[name]
                self.engine_metrics.engines_active -= 1
                self.engine_metrics.engines_shutdown += 1
                logger.info(f"Engine shutdown: {name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error shutting down engine {name}: {e}")
            return False
    
    async def execute_with_engine(self, engine_name: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task using a specific engine."""
        try:
            engine = await self.get_engine(engine_name)
            if not engine:
                raise ValueError(f"Engine not found: {engine_name}")
            
            start_time = asyncio.get_event_loop().time()
            
            # Execute the task
            result = await asyncio.wait_for(
                engine.execute(task_data),
                timeout=self.config.engine_timeout
            )
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            # Update metrics
            self.engine_metrics.total_executions += 1
            self.engine_metrics.successful_executions += 1
            self.engine_metrics.average_execution_time = (
                (self.engine_metrics.average_execution_time * (self.engine_metrics.total_executions - 1) + execution_time) /
                self.engine_metrics.total_executions
            )
            
            # Update specific engine metrics
            if engine_name == "quantum":
                self.engine_metrics.quantum_optimizations += 1
            elif engine_name == "neural_turbo":
                self.engine_metrics.neural_accelerations += 1
            elif engine_name == "marareal":
                self.engine_metrics.real_time_executions += 1
            elif engine_name == "hybrid":
                self.engine_metrics.hybrid_optimizations += 1
            
            result["execution_time"] = execution_time
            result["engine_used"] = engine_name
            result["module"] = "engines"
            
            return result
            
        except asyncio.TimeoutError:
            self.engine_metrics.failed_executions += 1
            raise TimeoutError(f"Engine execution timed out: {engine_name}")
        except Exception as e:
            self.engine_metrics.failed_executions += 1
            logger.error(f"Engine execution failed: {e}")
            raise
    
    async def execute_optimization(self, optimization_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute optimization using the best available engine."""
        try:
            # Determine best engine for optimization
            if "quantum" in self.active_engines:
                return await self.execute_with_engine("quantum", optimization_data)
            elif "hybrid" in self.active_engines:
                return await self.execute_with_engine("hybrid", optimization_data)
            else:
                raise RuntimeError("No optimization engine available")
                
        except Exception as e:
            logger.error(f"Optimization execution failed: {e}")
            raise
    
    async def execute_neural_acceleration(self, neural_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute neural network acceleration."""
        try:
            if "neural_turbo" in self.active_engines:
                return await self.execute_with_engine("neural_turbo", neural_data)
            elif "hybrid" in self.active_engines:
                return await self.execute_with_engine("hybrid", neural_data)
            else:
                raise RuntimeError("No neural acceleration engine available")
                
        except Exception as e:
            logger.error(f"Neural acceleration failed: {e}")
            raise
    
    async def execute_real_time(self, real_time_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute real-time task."""
        try:
            if "marareal" in self.active_engines:
                return await self.execute_with_engine("marareal", real_time_data)
            elif "hybrid" in self.active_engines:
                return await self.execute_with_engine("hybrid", real_time_data)
            else:
                raise RuntimeError("No real-time engine available")
                
        except Exception as e:
            logger.error(f"Real-time execution failed: {e}")
            raise
    
    async def get_engine_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all engines."""
        try:
            if not self.engine_manager:
                return {"error": "Engine manager not initialized"}
            
            status = await self.engine_manager.get_all_engines_status()
            status["module_status"] = self.status.value
            status["active_engines"] = list(self.active_engines.keys())
            status["metrics"] = {
                "engines_created": self.engine_metrics.engines_created,
                "engines_active": self.engine_metrics.engines_active,
                "total_executions": self.engine_metrics.total_executions,
                "successful_executions": self.engine_metrics.successful_executions,
                "failed_executions": self.engine_metrics.failed_executions,
                "average_execution_time": self.engine_metrics.average_execution_time
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting engine status: {e}")
            return {"error": str(e)}
    
    async def _health_monitor_loop(self):
        """Background health monitoring loop."""
        while self.status == ModuleStatus.ACTIVE:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                # Check engine health
                for name, engine in self.active_engines.items():
                    try:
                        health = await engine.health_check()
                        if health.get("status") == "error":
                            logger.warning(f"Engine {name} health check failed: {health}")
                    except Exception as e:
                        logger.warning(f"Health check failed for engine {name}: {e}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitor loop: {e}")
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get module metrics."""
        return {
            "module": "engines",
            "status": self.status.value,
            "active_engines": len(self.active_engines),
            "engine_metrics": {
                "engines_created": self.engine_metrics.engines_created,
                "engines_active": self.engine_metrics.engines_active,
                "engines_shutdown": self.engine_metrics.engines_shutdown,
                "total_executions": self.engine_metrics.total_executions,
                "successful_executions": self.engine_metrics.successful_executions,
                "failed_executions": self.engine_metrics.failed_executions,
                "average_execution_time": self.engine_metrics.average_execution_time,
                "quantum_optimizations": self.engine_metrics.quantum_optimizations,
                "neural_accelerations": self.engine_metrics.neural_accelerations,
                "real_time_executions": self.engine_metrics.real_time_executions,
                "hybrid_optimizations": self.engine_metrics.hybrid_optimizations
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check module health."""
        try:
            health_status = "healthy"
            issues = []
            
            # Check if engine manager is available
            if not self.engine_manager:
                health_status = "unhealthy"
                issues.append("Engine manager not initialized")
            
            # Check active engines
            if len(self.active_engines) == 0:
                health_status = "warning"
                issues.append("No active engines")
            
            # Check for failed executions
            if self.engine_metrics.failed_executions > 0:
                failure_rate = self.engine_metrics.failed_executions / max(self.engine_metrics.total_executions, 1)
                if failure_rate > 0.1:  # More than 10% failure rate
                    health_status = "warning"
                    issues.append(f"High failure rate: {failure_rate:.2%}")
            
            return {
                "status": health_status,
                "issues": issues,
                "active_engines": len(self.active_engines),
                "total_executions": self.engine_metrics.total_executions,
                "uptime": self.get_uptime()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "issues": [f"Health check failed: {e}"],
                "error": str(e)
            }

# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_engines_module(**kwargs) -> EnginesModule:
    """Create an Engines Module instance."""
    config = EngineModuleConfig(**kwargs)
    return EnginesModule(config)

def create_engines_module_with_defaults() -> EnginesModule:
    """Create an Engines Module with default configurations."""
    return create_engines_module(
        auto_initialize_engines=True,
        enable_quantum_optimization=True,
        enable_neural_turbo=True,
        enable_marareal=True,
        enable_hybrid=True,
        max_concurrent_engines=8,
        health_check_interval=10.0
    )

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "EnginesModule",
    "EngineModuleConfig",
    "EngineMetrics",
    "create_engines_module",
    "create_engines_module_with_defaults"
]
