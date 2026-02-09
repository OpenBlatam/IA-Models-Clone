"""
Auto Optimizer V2 for Flux2 Clothing Changer
===========================================

Advanced automatic optimization system.
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class OptimizationTarget:
    """Optimization target."""
    metric_name: str
    target_value: float
    optimization_direction: str  # "minimize" or "maximize"
    weight: float = 1.0


@dataclass
class OptimizationResult:
    """Optimization result."""
    parameter_name: str
    old_value: Any
    new_value: Any
    improvement: float
    confidence: float


class AutoOptimizerV2:
    """Advanced automatic optimization system."""
    
    def __init__(
        self,
        learning_rate: float = 0.1,
        exploration_rate: float = 0.2,
    ):
        """
        Initialize auto optimizer.
        
        Args:
            learning_rate: Learning rate for optimization
            exploration_rate: Exploration rate for trying new values
        """
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        
        self.targets: List[OptimizationTarget] = []
        self.parameter_history: Dict[str, List[Any]] = defaultdict(list)
        self.metric_history: Dict[str, List[float]] = defaultdict(list)
        self.optimization_results: List[OptimizationResult] = []
    
    def add_target(
        self,
        metric_name: str,
        target_value: float,
        optimization_direction: str = "maximize",
        weight: float = 1.0,
    ) -> OptimizationTarget:
        """
        Add optimization target.
        
        Args:
            metric_name: Metric name to optimize
            target_value: Target value
            optimization_direction: "minimize" or "maximize"
            weight: Target weight
            
        Returns:
            Created target
        """
        target = OptimizationTarget(
            metric_name=metric_name,
            target_value=target_value,
            optimization_direction=optimization_direction,
            weight=weight,
        )
        
        self.targets.append(target)
        logger.info(f"Added optimization target: {metric_name}")
        return target
    
    def record_metric(
        self,
        metric_name: str,
        value: float,
    ) -> None:
        """
        Record metric value.
        
        Args:
            metric_name: Metric name
            value: Metric value
        """
        self.metric_history[metric_name].append(value)
    
    def optimize_parameter(
        self,
        parameter_name: str,
        current_value: float,
        parameter_range: tuple,
        metric_name: str,
    ) -> Optional[OptimizationResult]:
        """
        Optimize a parameter.
        
        Args:
            parameter_name: Parameter name
            current_value: Current parameter value
            parameter_range: (min, max) range
            metric_name: Metric to optimize
            
        Returns:
            Optimization result or None
        """
        if metric_name not in self.metric_history or len(self.metric_history[metric_name]) < 2:
            return None
        
        # Get recent metric values
        recent_metrics = self.metric_history[metric_name][-10:]
        current_metric = recent_metrics[-1]
        previous_metric = recent_metrics[-2] if len(recent_metrics) > 1 else current_metric
        
        # Calculate improvement direction
        metric_change = current_metric - previous_metric
        
        # Find target for this metric
        target = next((t for t in self.targets if t.metric_name == metric_name), None)
        if not target:
            return None
        
        # Determine optimization direction
        if target.optimization_direction == "maximize":
            should_increase = metric_change > 0
        else:
            should_increase = metric_change < 0
        
        # Calculate new value
        min_val, max_val = parameter_range
        step = (max_val - min_val) * self.learning_rate
        
        if should_increase:
            new_value = min(current_value + step, max_val)
        else:
            new_value = max(current_value - step, min_val)
        
        # Exploration: sometimes try random value
        import random
        if random.random() < self.exploration_rate:
            new_value = random.uniform(min_val, max_val)
        
        # Calculate improvement
        improvement = abs(new_value - current_value) / (max_val - min_val) if max_val > min_val else 0.0
        
        # Confidence based on metric trend
        confidence = min(1.0, abs(metric_change) / max(abs(current_metric), 1.0))
        
        result = OptimizationResult(
            parameter_name=parameter_name,
            old_value=current_value,
            new_value=new_value,
            improvement=improvement,
            confidence=confidence,
        )
        
        self.optimization_results.append(result)
        self.parameter_history[parameter_name].append(new_value)
        
        logger.info(f"Optimized {parameter_name}: {current_value} -> {new_value} (confidence: {confidence:.2%})")
        return result
    
    def get_optimization_history(
        self,
        parameter_name: Optional[str] = None,
    ) -> List[OptimizationResult]:
        """
        Get optimization history.
        
        Args:
            parameter_name: Optional parameter name filter
            
        Returns:
            List of optimization results
        """
        if parameter_name:
            return [r for r in self.optimization_results if r.parameter_name == parameter_name]
        return self.optimization_results.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get optimizer statistics."""
        return {
            "total_targets": len(self.targets),
            "total_optimizations": len(self.optimization_results),
            "tracked_metrics": list(self.metric_history.keys()),
            "tracked_parameters": list(self.parameter_history.keys()),
        }


