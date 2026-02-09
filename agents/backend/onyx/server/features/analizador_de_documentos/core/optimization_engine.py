"""
Optimization Engine for Document Analyzer
==========================================

Advanced optimization engine for performance tuning and resource management.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class OptimizationStrategy(Enum):
    """Optimization strategies"""
    AUTO = "auto"
    PERFORMANCE = "performance"
    MEMORY = "memory"
    BALANCED = "balanced"

@dataclass
class OptimizationResult:
    """Result of optimization"""
    strategy: str
    improvements: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    estimated_gain: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

class OptimizationEngine:
    """Advanced optimization engine"""
    
    def __init__(self):
        self.optimizations: Dict[str, OptimizationResult] = {}
        self.metrics_history: List[Dict[str, Any]] = []
        logger.info("OptimizationEngine initialized")
    
    def analyze_performance(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance metrics and identify bottlenecks"""
        bottlenecks = []
        recommendations = []
        
        # Check CPU usage
        if metrics.get("cpu_percent", 0) > 80:
            bottlenecks.append("High CPU usage")
            recommendations.append("Consider reducing batch size or optimizing model inference")
        
        # Check memory usage
        if metrics.get("memory_percent", 0) > 85:
            bottlenecks.append("High memory usage")
            recommendations.append("Consider enabling memory cleanup or reducing cache size")
        
        # Check GPU usage
        if metrics.get("gpu_percent", 0) > 90:
            bottlenecks.append("High GPU usage")
            recommendations.append("Consider distributing load or optimizing model")
        
        # Check latency
        if metrics.get("avg_latency", 0) > 5.0:
            bottlenecks.append("High latency")
            recommendations.append("Consider caching, batch processing, or model optimization")
        
        return {
            "bottlenecks": bottlenecks,
            "recommendations": recommendations,
            "severity": "high" if len(bottlenecks) >= 3 else "medium" if len(bottlenecks) >= 1 else "low"
        }
    
    def optimize_batch_size(
        self,
        current_size: int,
        avg_time_per_item: float,
        memory_usage: float,
        target_time: float = 30.0
    ) -> int:
        """Optimize batch size based on metrics"""
        # Calculate optimal based on target time
        optimal_by_time = int(target_time / avg_time_per_item) if avg_time_per_item > 0 else current_size
        
        # Adjust based on memory
        if memory_usage > 80:
            optimal_by_time = int(optimal_by_time * 0.7)  # Reduce if memory is high
        
        # Ensure reasonable bounds
        optimal = max(5, min(optimal_by_time, 100))
        
        return optimal
    
    def recommend_optimizations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Cache recommendations
        if metrics.get("cache_hit_rate", 0) < 0.5:
            recommendations.append("Increase cache size or improve cache strategy")
        
        # Batch processing
        if metrics.get("avg_batch_time", 0) > 60:
            recommendations.append("Reduce batch size or optimize batch processing")
        
        # Model optimization
        if metrics.get("model_inference_time", 0) > 2.0:
            recommendations.append("Consider model quantization or pruning")
        
        # Memory optimization
        if metrics.get("memory_usage", 0) > 85:
            recommendations.append("Enable memory cleanup or reduce working set")
        
        return recommendations
    
    def create_optimization_plan(
        self,
        strategy: OptimizationStrategy,
        metrics: Dict[str, Any]
    ) -> OptimizationResult:
        """Create optimization plan based on strategy and metrics"""
        analysis = self.analyze_performance(metrics)
        recommendations = self.recommend_optimizations(metrics)
        
        improvements = {}
        
        if strategy == OptimizationStrategy.PERFORMANCE:
            improvements["batch_size"] = self.optimize_batch_size(
                metrics.get("batch_size", 10),
                metrics.get("avg_time_per_item", 1.0),
                metrics.get("memory_usage", 50)
            )
            improvements["cache_strategy"] = "aggressive"
        
        elif strategy == OptimizationStrategy.MEMORY:
            improvements["batch_size"] = max(5, int(metrics.get("batch_size", 10) * 0.7))
            improvements["cache_strategy"] = "conservative"
            improvements["enable_cleanup"] = True
        
        elif strategy == OptimizationStrategy.BALANCED:
            improvements["batch_size"] = self.optimize_batch_size(
                metrics.get("batch_size", 10),
                metrics.get("avg_time_per_item", 1.0),
                metrics.get("memory_usage", 50)
            )
            improvements["cache_strategy"] = "balanced"
        
        # Calculate estimated gain
        estimated_gain = 0.0
        if improvements:
            # Simple estimation (can be improved)
            estimated_gain = 15.0  # 15% improvement estimate
        
        result = OptimizationResult(
            strategy=strategy.value,
            improvements=improvements,
            recommendations=recommendations,
            estimated_gain=estimated_gain
        )
        
        self.optimizations[strategy.value] = result
        
        return result

# Global instance
optimization_engine = OptimizationEngine()
















