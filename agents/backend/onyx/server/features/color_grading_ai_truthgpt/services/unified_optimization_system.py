"""
Unified Optimization System for Color Grading AI
=================================================

Consolidates all optimization services into a single unified system:
- OptimizationEngine (parameter optimization)
- MLOptimizer (ML-based optimization)
- PerformanceOptimizer (resource optimization)
- AdaptiveOptimizer (adaptive optimization)
- ResourceOptimizer (resource allocation)
- AutoTuner (auto-tuning)

Features:
- Parameter optimization
- ML-based learning
- Resource optimization
- Adaptive strategies
- Auto-tuning
- Unified interface
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .optimization_engine import OptimizationEngine, OptimizationResult
from .ml_optimizer import MLOptimizer
from .performance_optimizer import PerformanceOptimizer, SystemResources
from .adaptive_optimizer import AdaptiveOptimizer, OptimizationPattern
from .resource_optimizer import ResourceOptimizer, ResourceUsage, ResourceAllocation
from .auto_tuner import AutoTuner, TuningResult

logger = logging.getLogger(__name__)


class OptimizationMode(Enum):
    """Optimization modes."""
    PARAMETER = "parameter"  # Color grading parameters
    PERFORMANCE = "performance"  # System performance
    RESOURCE = "resource"  # Resource allocation
    ML = "ml"  # Machine learning
    ADAPTIVE = "adaptive"  # Adaptive optimization
    AUTO = "auto"  # Auto-tuning
    FULL = "full"  # All optimizations


@dataclass
class UnifiedOptimizationResult:
    """Unified optimization result."""
    mode: OptimizationMode
    result: Any
    quality_score: Optional[float] = None
    performance_gain: Optional[float] = None
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class UnifiedOptimizationSystem:
    """
    Unified optimization system.
    
    Consolidates:
    - OptimizationEngine: Parameter optimization
    - MLOptimizer: ML-based optimization
    - PerformanceOptimizer: Resource optimization
    - AdaptiveOptimizer: Adaptive strategies
    - ResourceOptimizer: Resource allocation
    - AutoTuner: Auto-tuning
    
    Features:
    - Unified interface for all optimizations
    - Mode-based optimization
    - Combined strategies
    - Performance tracking
    - Recommendations
    """
    
    def __init__(self, default_mode: OptimizationMode = OptimizationMode.FULL):
        """
        Initialize unified optimization system.
        
        Args:
            default_mode: Default optimization mode
        """
        self.default_mode = default_mode
        
        # Initialize components
        self.optimization_engine = OptimizationEngine()
        self.ml_optimizer = MLOptimizer()
        self.performance_optimizer = PerformanceOptimizer()
        self.adaptive_optimizer = AdaptiveOptimizer()
        self.resource_optimizer = ResourceOptimizer()
        self.auto_tuner = AutoTuner()
        
        logger.info(f"Initialized UnifiedOptimizationSystem (mode={default_mode.value})")
    
    def optimize_parameters(
        self,
        current_params: Dict[str, Any],
        mode: Optional[OptimizationMode] = None,
        target_quality: float = 0.9,
        **kwargs
    ) -> UnifiedOptimizationResult:
        """
        Optimize color grading parameters.
        
        Args:
            current_params: Current parameters
            mode: Optimization mode
            target_quality: Target quality score
            **kwargs: Additional parameters
            
        Returns:
            Unified optimization result
        """
        mode = mode or self.default_mode
        
        if mode in [OptimizationMode.PARAMETER, OptimizationMode.FULL]:
            result = self.optimization_engine.optimize_parameters(
                current_params,
                target_quality=target_quality,
                **kwargs
            )
            return UnifiedOptimizationResult(
                mode=OptimizationMode.PARAMETER,
                result=result,
                quality_score=result.quality_score,
                performance_gain=result.performance_gain,
                recommendations=result.recommendations
            )
        
        return UnifiedOptimizationResult(
            mode=mode,
            result=current_params,
            quality_score=0.5
        )
    
    def optimize_with_ml(
        self,
        user_id: str,
        input_analysis: Dict[str, Any],
        mode: Optional[OptimizationMode] = None
    ) -> UnifiedOptimizationResult:
        """
        Optimize using ML.
        
        Args:
            user_id: User identifier
            input_analysis: Input media analysis
            mode: Optimization mode
            
        Returns:
            Unified optimization result
        """
        mode = mode or self.default_mode
        
        if mode in [OptimizationMode.ML, OptimizationMode.FULL]:
            optimal_params = self.ml_optimizer.predict_optimal_params(
                user_id,
                input_analysis
            )
            return UnifiedOptimizationResult(
                mode=OptimizationMode.ML,
                result=optimal_params,
                recommendations=["ML-based optimization applied"]
            )
        
        return UnifiedOptimizationResult(
            mode=mode,
            result={}
        )
    
    def optimize_resources(
        self,
        mode: Optional[OptimizationMode] = None
    ) -> UnifiedOptimizationResult:
        """
        Optimize system resources.
        
        Args:
            mode: Optimization mode
            
        Returns:
            Unified optimization result
        """
        mode = mode or self.default_mode
        
        if mode in [OptimizationMode.RESOURCE, OptimizationMode.PERFORMANCE, OptimizationMode.FULL]:
            resources = self.performance_optimizer.get_system_resources()
            allocation = self.resource_optimizer.optimize_allocation(
                ResourceUsage(
                    cpu_percent=resources.cpu_percent,
                    memory_percent=resources.memory_percent,
                    disk_io=0.0,
                    network_io=0.0
                )
            )
            
            return UnifiedOptimizationResult(
                mode=OptimizationMode.RESOURCE,
                result=allocation,
                recommendations=["Resource allocation optimized"]
            )
        
        return UnifiedOptimizationResult(
            mode=mode,
            result={}
        )
    
    def should_throttle(self) -> bool:
        """
        Check if processing should be throttled.
        
        Returns:
            True if should throttle
        """
        return self.performance_optimizer.should_throttle()
    
    def get_optimal_workers(self, base_workers: int = 3) -> int:
        """
        Get optimal number of workers.
        
        Args:
            base_workers: Base number of workers
            
        Returns:
            Optimal number of workers
        """
        return self.performance_optimizer.get_optimal_workers(base_workers)
    
    def learn_from_preference(
        self,
        user_id: str,
        input_analysis: Dict[str, Any],
        applied_params: Dict[str, Any],
        user_rating: float
    ):
        """
        Learn from user preference.
        
        Args:
            user_id: User identifier
            input_analysis: Input media analysis
            applied_params: Applied parameters
            user_rating: User rating (0-1)
        """
        self.ml_optimizer.learn_from_preference(
            user_id,
            input_analysis,
            applied_params,
            user_rating
        )
        logger.info(f"Learned from user preference: {user_id} (rating: {user_rating})")
    
    def auto_tune(
        self,
        target_metric: str,
        current_config: Dict[str, Any],
        mode: Optional[OptimizationMode] = None
    ) -> UnifiedOptimizationResult:
        """
        Auto-tune configuration.
        
        Args:
            target_metric: Target metric to optimize
            current_config: Current configuration
            mode: Optimization mode
            
        Returns:
            Unified optimization result
        """
        mode = mode or self.default_mode
        
        if mode in [OptimizationMode.AUTO, OptimizationMode.FULL]:
            tuning_result = self.auto_tuner.tune(
                target_metric,
                current_config
            )
            return UnifiedOptimizationResult(
                mode=OptimizationMode.AUTO,
                result=tuning_result,
                performance_gain=tuning_result.improvement,
                recommendations=["Auto-tuning completed"]
            )
        
        return UnifiedOptimizationResult(
            mode=mode,
            result=current_config
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return {
            "mode": self.default_mode.value,
            "resource_stats": self.performance_optimizer.get_resource_stats(),
            "optimization_history": len(self.optimization_engine._optimization_history),
        }


