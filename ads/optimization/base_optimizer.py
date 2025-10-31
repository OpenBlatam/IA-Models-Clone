"""
🚀 ADS Optimization - Base Optimizer

Base classes and interfaces for the consolidated optimization system.
This consolidates all scattered optimization functionality into a single,
organized architecture.
"""

import asyncio
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timezone
import weakref

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Optimization strategy types."""
    PERFORMANCE = "performance"
    MEMORY = "memory"
    GPU = "gpu"
    NETWORK = "network"
    DATABASE = "database"
    CACHE = "cache"
    COMPREHENSIVE = "comprehensive"


class OptimizationLevel(Enum):
    """Optimization intensity levels."""
    LIGHT = "light"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"
    EXTREME = "extreme"


@dataclass
class OptimizationResult:
    """Result of an optimization operation."""
    strategy: OptimizationStrategy
    level: OptimizationLevel
    success: bool
    metrics_before: Dict[str, float]
    metrics_after: Dict[str, float]
    improvement_percentage: float
    duration_ms: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    details: Dict[str, Any] = field(default_factory=dict)
    errors: Optional[List[str]] = None
    warnings: Optional[List[str]] = None


@dataclass
class OptimizationContext:
    """Context for optimization operations."""
    target_entity: str  # ad, campaign, group
    entity_id: str
    optimization_type: OptimizationStrategy
    level: OptimizationLevel
    parameters: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class BaseOptimizer(ABC):
    """
    🚀 Base Optimizer - Consolidated optimization functionality.
    
    This base class consolidates all scattered optimization implementations
    into a single, organized system that provides consistent behavior
    across all optimization types.
    """
    
    def __init__(self, name: str, strategy: OptimizationStrategy):
        self.name = name
        self.strategy = strategy
        self.optimization_history: List[OptimizationResult] = []
        self.active_optimizations: Dict[str, OptimizationContext] = {}
        self.performance_metrics: Dict[str, List[float]] = {}
        self.error_count = 0
        self.success_count = 0
        
        # Performance tracking
        self.total_optimization_time = 0.0
        self.avg_optimization_time = 0.0
        self.total_improvement = 0.0
        self.avg_improvement = 0.0
        
        # Resource monitoring
        self.resource_usage: Dict[str, float] = {}
        self.optimization_thresholds: Dict[str, float] = {}
        
        # Callbacks and hooks
        self.before_optimization_hooks: List[Callable] = []
        self.after_optimization_hooks: List[Callable] = []
        self.error_handlers: List[Callable] = []
        
        logger.info(f"Initialized {self.name} optimizer with strategy: {self.strategy.value}")
    
    @abstractmethod
    async def optimize(self, context: OptimizationContext) -> OptimizationResult:
        """Execute optimization with the given context."""
        pass
    
    @abstractmethod
    def can_optimize(self, context: OptimizationContext) -> bool:
        """Check if this optimizer can handle the given context."""
        pass
    
    @abstractmethod
    def get_optimization_capabilities(self) -> Dict[str, Any]:
        """Get the capabilities and limitations of this optimizer."""
        pass
    
    async def execute_optimization(self, context: OptimizationContext) -> OptimizationResult:
        """Execute optimization with full lifecycle management."""
        start_time = time.time()
        optimization_id = f"{context.entity_id}_{int(start_time)}"
        
        try:
            # Validate context
            if not self._validate_context(context):
                raise ValueError(f"Invalid optimization context: {context}")
            
            # Check if optimization is possible
            if not self.can_optimize(context):
                raise ValueError(f"Optimizer {self.name} cannot handle context: {context}")
            
            # Execute before hooks
            await self._execute_before_hooks(context)
            
            # Collect metrics before optimization
            metrics_before = await self._collect_metrics_before(context)
            
            # Execute optimization
            result = await self.optimize(context)
            
            # Collect metrics after optimization
            metrics_after = await self._collect_metrics_after(context)
            
            # Calculate improvement
            improvement_percentage = self._calculate_improvement(metrics_before, metrics_after)
            
            # Create optimization result
            optimization_result = OptimizationResult(
                strategy=self.strategy,
                level=context.level,
                success=True,
                metrics_before=metrics_before,
                metrics_after=metrics_after,
                improvement_percentage=improvement_percentage,
                duration_ms=(time.time() - start_time) * 1000,
                details=result.details if hasattr(result, 'details') else {},
                timestamp=datetime.now(timezone.utc)
            )
            
            # Update statistics
            self._update_statistics(optimization_result)
            
            # Execute after hooks
            await self._execute_after_hooks(context, optimization_result)
            
            # Log success
            logger.info(f"Optimization completed successfully: {optimization_id}, "
                       f"Improvement: {improvement_percentage:.2f}%")
            
            return optimization_result
            
        except Exception as e:
            # Handle errors
            error_result = OptimizationResult(
                strategy=self.strategy,
                level=context.level,
                success=False,
                metrics_before={},
                metrics_after={},
                improvement_percentage=0.0,
                duration_ms=(time.time() - start_time) * 1000,
                errors=[str(e)],
                timestamp=datetime.now(timezone.utc)
            )
            
            # Update error statistics
            self.error_count += 1
            
            # Execute error handlers
            await self._execute_error_handlers(context, e)
            
            # Log error
            logger.error(f"Optimization failed: {optimization_id}, Error: {e}")
            
            return error_result
    
    def _validate_context(self, context: OptimizationContext) -> bool:
        """Validate optimization context."""
        if not context.target_entity:
            return False
        if not context.entity_id:
            return False
        if not context.optimization_type:
            return False
        if not context.level:
            return False
        return True
    
    async def _collect_metrics_before(self, context: OptimizationContext) -> Dict[str, float]:
        """Collect performance metrics before optimization."""
        try:
            metrics = {}
            
            # Collect system metrics
            metrics['cpu_usage'] = self._get_cpu_usage()
            metrics['memory_usage'] = self._get_memory_usage()
            metrics['gpu_usage'] = self._get_gpu_usage()
            metrics['network_usage'] = self._get_network_usage()
            
            # Collect application-specific metrics
            app_metrics = await self._collect_application_metrics(context)
            metrics.update(app_metrics)
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Failed to collect metrics before optimization: {e}")
            return {}
    
    async def _collect_metrics_after(self, context: OptimizationContext) -> Dict[str, float]:
        """Collect performance metrics after optimization."""
        try:
            metrics = {}
            
            # Collect system metrics
            metrics['cpu_usage'] = self._get_cpu_usage()
            metrics['memory_usage'] = self._get_memory_usage()
            metrics['gpu_usage'] = self._get_gpu_usage()
            metrics['network_usage'] = self._get_network_usage()
            
            # Collect application-specific metrics
            app_metrics = await self._collect_application_metrics(context)
            metrics.update(app_metrics)
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Failed to collect metrics after optimization: {e}")
            return {}
    
    async def _collect_application_metrics(self, context: OptimizationContext) -> Dict[str, float]:
        """Collect application-specific metrics."""
        # Override in subclasses for specific metrics
        return {}
    
    def _calculate_improvement(self, metrics_before: Dict[str, float], 
                              metrics_after: Dict[str, float]) -> float:
        """Calculate improvement percentage between before and after metrics."""
        if not metrics_before or not metrics_after:
            return 0.0
        
        improvements = []
        for key in metrics_before:
            if key in metrics_after:
                before = metrics_before[key]
                after = metrics_after[key]
                
                if before > 0:
                    # For metrics where lower is better (CPU, memory usage)
                    if key in ['cpu_usage', 'memory_usage', 'gpu_usage', 'network_usage']:
                        improvement = ((before - after) / before) * 100
                    else:
                        improvement = ((after - before) / before) * 100
                    improvements.append(improvement)
        
        return sum(improvements) / len(improvements) if improvements else 0.0
    
    def _update_statistics(self, result: OptimizationResult):
        """Update optimizer statistics."""
        if result.success:
            self.success_count += 1
            self.total_optimization_time += result.duration_ms
            self.total_improvement += result.improvement_percentage
            
            # Update averages
            self.avg_optimization_time = self.total_optimization_time / self.success_count
            self.avg_improvement = self.total_improvement / self.success_count
        
        # Add to history
        self.optimization_history.append(result)
        
        # Keep only recent history
        if len(self.optimization_history) > 100:
            self.optimization_history = self.optimization_history[-100:]
    
    async def _execute_before_hooks(self, context: OptimizationContext):
        """Execute before optimization hooks."""
        for hook in self.before_optimization_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(context)
                else:
                    hook(context)
            except Exception as e:
                logger.warning(f"Before hook failed: {e}")
    
    async def _execute_after_hooks(self, context: OptimizationContext, result: OptimizationResult):
        """Execute after optimization hooks."""
        for hook in self.after_optimization_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(context, result)
                else:
                    hook(context, result)
            except Exception as e:
                logger.warning(f"After hook failed: {e}")
    
    async def _execute_error_handlers(self, context: OptimizationContext, error: Exception):
        """Execute error handlers."""
        for handler in self.error_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(context, error)
                else:
                    handler(context, error)
            except Exception as e:
                logger.warning(f"Error handler failed: {e}")
    
    def add_before_hook(self, hook: Callable):
        """Add a before optimization hook."""
        self.before_optimization_hooks.append(hook)
    
    def add_after_hook(self, hook: Callable):
        """Add an after optimization hook."""
        self.after_optimization_hooks.append(hook)
    
    def add_error_handler(self, handler: Callable):
        """Add an error handler."""
        self.error_handlers.append(handler)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get optimizer statistics."""
        return {
            'name': self.name,
            'strategy': self.strategy.value,
            'total_optimizations': self.success_count + self.error_count,
            'successful_optimizations': self.success_count,
            'failed_optimizations': self.error_count,
            'success_rate': (self.success_count / (self.success_count + self.error_count) * 100) 
                           if (self.success_count + self.error_count) > 0 else 0,
            'avg_optimization_time_ms': self.avg_optimization_time,
            'avg_improvement_percentage': self.avg_improvement,
            'total_optimization_time_ms': self.total_optimization_time,
            'total_improvement_percentage': self.total_improvement,
            'recent_optimizations': [
                {
                    'timestamp': result.timestamp.isoformat(),
                    'success': result.success,
                    'improvement': result.improvement_percentage,
                    'duration_ms': result.duration_ms
                }
                for result in self.optimization_history[-10:]  # Last 10 optimizations
            ]
        }
    
    def get_optimization_history(self, limit: int = 50) -> List[OptimizationResult]:
        """Get optimization history."""
        return self.optimization_history[-limit:]
    
    def clear_history(self):
        """Clear optimization history."""
        self.optimization_history.clear()
    
    # System metric collection methods
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            import psutil
            return psutil.cpu_percent(interval=0.1)
        except ImportError:
            return 0.0
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage percentage."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return memory.percent
        except ImportError:
            return 0.0
    
    def _get_gpu_usage(self) -> float:
        """Get current GPU usage percentage."""
        try:
            # Try to get GPU usage from various sources
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return info.gpu
        except ImportError:
            return 0.0
        except Exception:
            return 0.0
    
    def _get_network_usage(self) -> float:
        """Get current network usage."""
        try:
            import psutil
            # This is a simplified network usage metric
            return 0.0  # Would need more complex implementation for actual network usage
        except ImportError:
            return 0.0
    
    def cleanup(self):
        """Clean up optimizer resources."""
        self.optimization_history.clear()
        self.active_optimizations.clear()
        self.performance_metrics.clear()
        self.before_optimization_hooks.clear()
        self.after_optimization_hooks.clear()
        self.error_handlers.clear()
        
        logger.info(f"Cleaned up {self.name} optimizer")
