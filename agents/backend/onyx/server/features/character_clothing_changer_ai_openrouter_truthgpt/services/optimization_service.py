"""
Optimization Service
====================

Service for optimizing workflows and operations.
"""

import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Optimization result"""
    success: bool
    improvements: List[str]
    metrics: Dict[str, Any]
    timestamp: datetime


class OptimizationService:
    """
    Service for optimizing workflows and operations.
    
    Features:
    - Workflow optimization
    - Parameter tuning
    - Performance optimization
    - Resource optimization
    """
    
    def __init__(self):
        """Initialize optimization service"""
        self.optimization_history: List[OptimizationResult] = []
    
    def optimize_workflow_parameters(
        self,
        current_params: Dict[str, Any],
        target_quality: str = "balanced"
    ) -> Dict[str, Any]:
        """
        Optimize workflow parameters based on target quality.
        
        Args:
            current_params: Current workflow parameters
            target_quality: Target quality ("fast", "balanced", "high")
            
        Returns:
            Optimized parameters
        """
        optimized = current_params.copy()
        
        if target_quality == "fast":
            # Optimize for speed
            optimized["num_steps"] = min(optimized.get("num_steps", 12), 8)
            optimized["guidance_scale"] = min(optimized.get("guidance_scale", 50.0), 40.0)
        elif target_quality == "high":
            # Optimize for quality
            optimized["num_steps"] = max(optimized.get("num_steps", 12), 20)
            optimized["guidance_scale"] = max(optimized.get("guidance_scale", 50.0), 60.0)
        else:  # balanced
            # Balanced optimization
            optimized["num_steps"] = optimized.get("num_steps", 12)
            optimized["guidance_scale"] = optimized.get("guidance_scale", 50.0)
        
        logger.info(f"Optimized parameters for {target_quality} quality")
        return optimized
    
    def suggest_improvements(
        self,
        metrics: Dict[str, Any]
    ) -> List[str]:
        """
        Suggest improvements based on metrics.
        
        Args:
            metrics: Performance metrics
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        # Check duration
        duration = metrics.get("duration", 0)
        if duration > 60:
            suggestions.append("Consider reducing num_steps for faster processing")
        
        # Check success rate
        success_rate = metrics.get("success_rate", 1.0)
        if success_rate < 0.9:
            suggestions.append("Review error logs to identify common failure patterns")
        
        # Check resource usage
        cpu_usage = metrics.get("cpu_usage", 0)
        if cpu_usage > 80:
            suggestions.append("High CPU usage detected - consider scaling horizontally")
        
        memory_usage = metrics.get("memory_usage", 0)
        if memory_usage > 80:
            suggestions.append("High memory usage - consider optimizing batch sizes")
        
        return suggestions
    
    def optimize_batch_size(
        self,
        current_batch_size: int,
        avg_duration: float,
        max_concurrent: int
    ) -> int:
        """
        Optimize batch size based on performance.
        
        Args:
            current_batch_size: Current batch size
            avg_duration: Average operation duration
            max_concurrent: Maximum concurrent operations
            
        Returns:
            Optimized batch size
        """
        # Simple heuristic: adjust based on duration
        if avg_duration < 5:
            # Fast operations - can handle larger batches
            optimized = min(current_batch_size * 2, max_concurrent * 2)
        elif avg_duration > 30:
            # Slow operations - reduce batch size
            optimized = max(current_batch_size // 2, 1)
        else:
            optimized = current_batch_size
        
        logger.info(f"Optimized batch size: {current_batch_size} -> {optimized}")
        return optimized
    
    def record_optimization(
        self,
        success: bool,
        improvements: List[str],
        metrics: Dict[str, Any]
    ) -> None:
        """
        Record optimization result.
        
        Args:
            success: Whether optimization was successful
            improvements: List of improvements made
            metrics: Performance metrics
        """
        result = OptimizationResult(
            success=success,
            improvements=improvements,
            metrics=metrics,
            timestamp=datetime.now()
        )
        
        self.optimization_history.append(result)
        
        # Keep only recent history
        if len(self.optimization_history) > 100:
            self.optimization_history = self.optimization_history[-100:]
    
    def get_optimization_history(
        self,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get optimization history.
        
        Args:
            limit: Maximum number of results
            
        Returns:
            List of optimization results
        """
        recent = self.optimization_history[-limit:]
        
        return [
            {
                "success": r.success,
                "improvements": r.improvements,
                "metrics": r.metrics,
                "timestamp": r.timestamp.isoformat()
            }
            for r in recent
        ]

