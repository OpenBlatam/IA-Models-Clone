"""
Optimization Module - Character Clothing Changer AI
====================================================

Optimization and performance utilities.
"""

# Re-export for backward compatibility
from ...models.auto_optimizer import AutoOptimizer
from ...models.auto_optimizer_v2 import AutoOptimizerV2, OptimizationTarget, OptimizationResult
from ...models.memory_optimizer import MemoryOptimizer
from ...models.performance_tracker import PerformanceTracker, PerformanceMetric
from ...models.performance_monitor import PerformanceMonitor, PerformanceMetrics
from ...models.resolution_handler import ResolutionHandler
from ...models.resource_optimizer import ResourceOptimizer, ResourceUsage

__all__ = [
    "AutoOptimizer",
    "AutoOptimizerV2",
    "OptimizationTarget",
    "OptimizationResult",
    "MemoryOptimizer",
    "PerformanceTracker",
    "PerformanceMetric",
    "PerformanceMonitor",
    "PerformanceMetrics",
    "ResolutionHandler",
    "ResourceOptimizer",
    "ResourceUsage",
]

