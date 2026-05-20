"""
🚀 ADS Optimization - Optimization Factory

Factory pattern implementation for creating and managing optimizers.
This consolidates all scattered optimizer creation logic into a single,
organized system.
"""

import logging
from typing import Dict, List, Any, Optional, Type
from enum import Enum

from .base_optimizer import BaseOptimizer, OptimizationStrategy, OptimizationLevel, OptimizationContext
from .performance_optimizer import PerformanceOptimizer
from .profiling_optimizer import ProfilingOptimizer
from .gpu_optimizer import GPUOptimizer

logger = logging.getLogger(__name__)


class OptimizerType(Enum):
    """Types of available optimizers."""
    PERFORMANCE = "performance"
    PROFILING = "profiling"
    GPU = "gpu"
    MEMORY = "memory"
    NETWORK = "network"
    DATABASE = "database"
    CACHE = "cache"
    COMPREHENSIVE = "comprehensive"


class OptimizationFactory:
    """
    🚀 Optimization Factory - Consolidated optimizer creation.
    
    This factory consolidates all scattered optimizer creation logic into a single,
    organized system that provides consistent optimizer instantiation across the
    entire advertising platform.
    """
    
    def __init__(self):
        self.registered_optimizers: Dict[str, Type[BaseOptimizer]] = {}
        self.optimizer_instances: Dict[str, BaseOptimizer] = {}
        self.optimizer_configs: Dict[str, Dict[str, Any]] = {}
        
        # Register default optimizers
        self._register_default_optimizers()
        
        logger.info("Optimization factory initialized")
    
    def _register_default_optimizers(self):
        """Register the default set of optimizers."""
        self.register_optimizer(
            OptimizerType.PERFORMANCE.value,
            PerformanceOptimizer,
            {
                'name': 'Performance Optimizer',
                'description': 'Optimizes general performance metrics',
                'capabilities': ['cpu_optimization', 'memory_optimization', 'response_time_optimization']
            }
        )
        
        self.register_optimizer(
            OptimizerType.PROFILING.value,
            ProfilingOptimizer,
            {
                'name': 'Profiling Optimizer',
                'description': 'Optimizes based on profiling data',
                'capabilities': ['code_profiling', 'bottleneck_detection', 'performance_analysis']
            }
        )
        
        self.register_optimizer(
            OptimizerType.GPU.value,
            GPUOptimizer,
            {
                'name': 'GPU Optimizer',
                'description': 'Optimizes GPU-related operations',
                'capabilities': ['gpu_memory_optimization', 'cuda_optimization', 'tensor_optimization']
            }
        )
        
        logger.info(f"Registered {len(self.registered_optimizers)} default optimizers")
    
    def register_optimizer(self, optimizer_type: str, optimizer_class: Type[BaseOptimizer], 
                          config: Dict[str, Any]):
        """Register a new optimizer type."""
        if optimizer_type in self.registered_optimizers:
            logger.warning(f"Overwriting existing optimizer: {optimizer_type}")
        
        self.registered_optimizers[optimizer_type] = optimizer_class
        self.optimizer_configs[optimizer_type] = config
        
        logger.info(f"Registered optimizer: {optimizer_type} -> {optimizer_class.__name__}")
    
    def create_optimizer(self, optimizer_type: str, **kwargs) -> BaseOptimizer:
        """Create an optimizer instance of the specified type."""
        if optimizer_type not in self.registered_optimizers:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        optimizer_class = self.registered_optimizers[optimizer_type]
        config = self.optimizer_configs[optimizer_type]
        
        # Only pass the 'name' parameter to the constructor
        creation_params = {'name': config.get('name', optimizer_type.title())}
        creation_params.update(kwargs)
        
        try:
            optimizer = optimizer_class(**creation_params)
            logger.info(f"Created optimizer: {optimizer_type}")
            return optimizer
            
        except Exception as e:
            logger.error(f"Failed to create optimizer {optimizer_type}: {e}")
            raise
    
    def get_or_create_optimizer(self, optimizer_type: str, **kwargs) -> BaseOptimizer:
        """Get existing optimizer instance or create a new one."""
        instance_key = f"{optimizer_type}_{hash(str(kwargs))}"
        
        if instance_key in self.optimizer_instances:
            return self.optimizer_instances[instance_key]
        
        optimizer = self.create_optimizer(optimizer_type, **kwargs)
        self.optimizer_instances[instance_key] = optimizer
        
        return optimizer
    
    def get_optimizer_info(self, optimizer_type: str) -> Optional[Dict[str, Any]]:
        """Get information about a registered optimizer."""
        if optimizer_type not in self.registered_optimizers:
            return None
        
        config = self.optimizer_configs[optimizer_type]
        optimizer_class = self.registered_optimizers[optimizer_type]
        
        return {
            'type': optimizer_type,
            'class_name': optimizer_class.__name__,
            'module': optimizer_class.__module__,
            'config': config,
            'capabilities': config.get('capabilities', [])
        }
    
    def list_available_optimizers(self) -> List[Dict[str, Any]]:
        """List all available optimizer types."""
        optimizers = []
        
        for optimizer_type in self.registered_optimizers:
            info = self.get_optimizer_info(optimizer_type)
            if info:
                optimizers.append(info)
        
        return optimizers
    
    def get_optimizer_capabilities(self, optimizer_type: str) -> List[str]:
        """Get capabilities of a specific optimizer type."""
        info = self.get_optimizer_info(optimizer_type)
        return info.get('capabilities', []) if info else []
    
    def can_handle_optimization(self, optimizer_type: str, context: OptimizationContext) -> bool:
        """Check if an optimizer can handle a specific optimization context."""
        if optimizer_type not in self.registered_optimizers:
            return False
        
        optimizer = self.get_or_create_optimizer(optimizer_type)
        return optimizer.can_optimize(context)
    
    def get_optimal_optimizer(self, context: OptimizationContext) -> Optional[str]:
        """Get the optimal optimizer type for a given context."""
        best_optimizer = None
        best_score = 0
        
        for optimizer_type in self.registered_optimizers:
            if self.can_handle_optimization(optimizer_type, context):
                # Calculate score based on context requirements
                score = self._calculate_optimizer_score(optimizer_type, context)
                
                if score > best_score:
                    best_score = score
                    best_optimizer = optimizer_type
        
        return best_optimizer
    
    def _calculate_optimizer_score(self, optimizer_type: str, context: OptimizationContext) -> float:
        """Calculate how well an optimizer matches a context."""
        score = 0.0
        
        # Get optimizer capabilities
        capabilities = self.get_optimizer_capabilities(optimizer_type)
        
        # Score based on strategy match
        if optimizer_type in [OptimizerType.PERFORMANCE.value, OptimizerType.PROFILING.value]:
            if context.optimization_type == OptimizationStrategy.PERFORMANCE:
                score += 10.0
        
        if optimizer_type == OptimizerType.GPU.value:
            if context.optimization_type == OptimizationStrategy.GPU:
                score += 10.0
        
        # Score based on level match
        if context.level == OptimizationLevel.LIGHT:
            if optimizer_type == OptimizerType.PERFORMANCE.value:
                score += 5.0
        elif context.level == OptimizationLevel.STANDARD:
            if optimizer_type in [OptimizerType.PERFORMANCE.value, OptimizerType.PROFILING.value]:
                score += 5.0
        elif context.level == OptimizationLevel.AGGRESSIVE:
            if optimizer_type in [OptimizerType.PROFILING.value, OptimizerType.GPU.value]:
                score += 5.0
        elif context.level == OptimizationLevel.EXTREME:
            if optimizer_type == OptimizerType.GPU.value:
                score += 5.0
        
        # Score based on entity type
        if context.target_entity == 'ad':
            if optimizer_type == OptimizerType.PERFORMANCE.value:
                score += 3.0
        elif context.target_entity == 'campaign':
            if optimizer_type == OptimizerType.PROFILING.value:
                score += 3.0
        
        return score
    
    async def execute_optimization(self, context: OptimizationContext, 
                                   optimizer_type: Optional[str] = None) -> Optional[Any]:
        """Execute optimization using the best available optimizer."""
        try:
            # Determine optimizer type if not specified
            if not optimizer_type:
                optimizer_type = self.get_optimal_optimizer(context)
            
            if not optimizer_type:
                logger.warning(f"No suitable optimizer found for context: {context}")
                return None
            
            # Get or create optimizer
            optimizer = self.get_or_create_optimizer(optimizer_type)
            
            # Execute optimization
            result = await optimizer.execute_optimization(context)
            
            logger.info(f"Optimization executed successfully using {optimizer_type}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to execute optimization: {e}")
            return None
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all optimizer instances."""
        stats = {
            'total_optimizers': len(self.registered_optimizers),
            'active_instances': len(self.optimizer_instances),
            'optimizer_types': list(self.registered_optimizers.keys()),
            'instance_statistics': {}
        }
        
        # Collect statistics from all instances
        for instance_key, optimizer in self.optimizer_instances.items():
            try:
                instance_stats = optimizer.get_statistics()
                stats['instance_statistics'][instance_key] = instance_stats
            except Exception as e:
                logger.warning(f"Failed to get statistics for {instance_key}: {e}")
        
        return stats
    
    def cleanup_optimizer(self, optimizer_type: str):
        """Clean up a specific optimizer type."""
        if optimizer_type in self.registered_optimizers:
            del self.registered_optimizers[optimizer_type]
            del self.optimizer_configs[optimizer_type]
            
            # Clean up instances
            instances_to_remove = [
                key for key in self.optimizer_instances.keys()
                if key.startswith(optimizer_type)
            ]
            
            for key in instances_to_remove:
                try:
                    self.optimizer_instances[key].cleanup()
                    del self.optimizer_instances[key]
                except Exception as e:
                    logger.warning(f"Failed to cleanup optimizer instance {key}: {e}")
            
            logger.info(f"Cleaned up optimizer type: {optimizer_type}")
    
    def cleanup_all(self):
        """Clean up all optimizers and instances."""
        # Clean up all instances
        for instance_key, optimizer in self.optimizer_instances.items():
            try:
                optimizer.cleanup()
            except Exception as e:
                logger.warning(f"Failed to cleanup optimizer instance {instance_key}: {e}")
        
        # Clear all collections
        self.registered_optimizers.clear()
        self.optimizer_instances.clear()
        self.optimizer_configs.clear()
        
        logger.info("Cleaned up all optimizers")


# Global factory instance
_optimization_factory: Optional[OptimizationFactory] = None


def get_optimization_factory() -> OptimizationFactory:
    """Get or create the global optimization factory instance."""
    global _optimization_factory
    if _optimization_factory is None:
        _optimization_factory = OptimizationFactory()
    return _optimization_factory


def register_optimizer(optimizer_type: str, optimizer_class: Type[BaseOptimizer], 
                      config: Dict[str, Any]):
    """Register an optimizer with the global factory."""
    factory = get_optimization_factory()
    factory.register_optimizer(optimizer_type, optimizer_class, config)


def create_optimizer(optimizer_type: str, **kwargs) -> BaseOptimizer:
    """Create an optimizer using the global factory."""
    factory = get_optimization_factory()
    return factory.create_optimizer(optimizer_type, **kwargs)


def get_optimal_optimizer(context: OptimizationContext) -> Optional[str]:
    """Get the optimal optimizer for a context using the global factory."""
    factory = get_optimization_factory()
    return factory.get_optimal_optimizer(context)
