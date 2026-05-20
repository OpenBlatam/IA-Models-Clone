"""
Cache cost optimization.

Provides cost optimization for cache operations.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class CostMetric(Enum):
    """Cost metrics."""
    MEMORY = "memory"
    COMPUTE = "compute"
    NETWORK = "network"
    STORAGE = "storage"


@dataclass
class CostProfile:
    """Cost profile."""
    metric: CostMetric
    cost_per_unit: float
    current_usage: float
    total_cost: float


class CacheCostOptimizer:
    """
    Cache cost optimizer.
    
    Optimizes cache for cost efficiency.
    """
    
    def __init__(self, cache: Any):
        """
        Initialize cost optimizer.
        
        Args:
            cache: Cache instance
        """
        self.cache = cache
        self.cost_profiles: Dict[CostMetric, CostProfile] = {}
        self.usage_history: List[Dict[str, Any]] = []
    
    def set_cost_profile(
        self,
        metric: CostMetric,
        cost_per_unit: float
    ) -> None:
        """
        Set cost profile for metric.
        
        Args:
            metric: Cost metric
            cost_per_unit: Cost per unit
        """
        stats = self.cache.get_stats()
        
        if metric == CostMetric.MEMORY:
            current_usage = stats.get("memory_mb", 0.0)
        elif metric == CostMetric.COMPUTE:
            current_usage = stats.get("operations_count", 0.0)
        else:
            current_usage = 0.0
        
        profile = CostProfile(
            metric=metric,
            cost_per_unit=cost_per_unit,
            current_usage=current_usage,
            total_cost=current_usage * cost_per_unit
        )
        
        self.cost_profiles[metric] = profile
    
    def calculate_total_cost(self) -> float:
        """
        Calculate total cost.
        
        Returns:
            Total cost
        """
        total = 0.0
        
        for profile in self.cost_profiles.values():
            total += profile.total_cost
        
        return total
    
    def get_cost_breakdown(self) -> Dict[str, Any]:
        """
        Get cost breakdown.
        
        Returns:
            Cost breakdown
        """
        breakdown = {}
        
        for metric, profile in self.cost_profiles.items():
            breakdown[metric.value] = {
                "cost_per_unit": profile.cost_per_unit,
                "usage": profile.current_usage,
                "total_cost": profile.total_cost
            }
        
        breakdown["total"] = self.calculate_total_cost()
        
        return breakdown
    
    def optimize_for_cost(self, target_cost: float) -> Dict[str, Any]:
        """
        Optimize cache for target cost.
        
        Args:
            target_cost: Target cost
            
        Returns:
            Optimization results
        """
        current_cost = self.calculate_total_cost()
        
        if current_cost <= target_cost:
            return {
                "optimized": False,
                "reason": "Already below target cost",
                "current_cost": current_cost,
                "target_cost": target_cost
            }
        
        optimizations = []
        
        # Optimize memory cost
        if CostMetric.MEMORY in self.cost_profiles:
            memory_cost = self.cost_profiles[CostMetric.MEMORY].total_cost
            
            if memory_cost > target_cost * 0.5:
                # Enable compression
                if not self.cache.config.use_compression:
                    self.cache.config.use_compression = True
                    optimizations.append("Enabled compression")
                
                # Enable quantization
                if not self.cache.config.use_quantization:
                    self.cache.config.use_quantization = True
                    optimizations.append("Enabled quantization")
        
        # Reduce cache size
        if current_cost > target_cost:
            stats = self.cache.get_stats()
            current_size = self.cache.config.max_tokens
            
            # Reduce by 20%
            new_size = int(current_size * 0.8)
            self.cache.config.max_tokens = new_size
            optimizations.append(f"Reduced cache size from {current_size} to {new_size}")
        
        new_cost = self.calculate_total_cost()
        
        return {
            "optimized": True,
            "optimizations": optimizations,
            "previous_cost": current_cost,
            "new_cost": new_cost,
            "savings": current_cost - new_cost
        }
    
    def track_usage(self) -> None:
        """Track current usage."""
        import time
        
        stats = self.cache.get_stats()
        usage = {
            "timestamp": time.time(),
            "memory_mb": stats.get("memory_mb", 0.0),
            "cache_size": stats.get("cache_size", 0),
            "operations": stats.get("operations_count", 0)
        }
        
        self.usage_history.append(usage)
        
        # Keep only recent history
        if len(self.usage_history) > 1000:
            self.usage_history = self.usage_history[-1000:]
    
    def get_cost_trend(self) -> Dict[str, Any]:
        """
        Get cost trend.
        
        Returns:
            Cost trend analysis
        """
        if len(self.usage_history) < 2:
            return {"trend": "insufficient_data"}
        
        recent = self.usage_history[-10:]
        older = self.usage_history[-20:-10] if len(self.usage_history) >= 20 else self.usage_history[:10]
        
        recent_cost = sum(self._calculate_cost_for_usage(u) for u in recent) / len(recent)
        older_cost = sum(self._calculate_cost_for_usage(u) for u in older) / len(older) if older else recent_cost
        
        trend = "increasing" if recent_cost > older_cost else "decreasing" if recent_cost < older_cost else "stable"
        
        return {
            "trend": trend,
            "recent_avg_cost": recent_cost,
            "older_avg_cost": older_cost,
            "change_percent": ((recent_cost - older_cost) / older_cost * 100) if older_cost > 0 else 0
        }
    
    def _calculate_cost_for_usage(self, usage: Dict[str, Any]) -> float:
        """Calculate cost for usage snapshot."""
        cost = 0.0
        
        if CostMetric.MEMORY in self.cost_profiles:
            memory_cost = usage.get("memory_mb", 0.0) * self.cost_profiles[CostMetric.MEMORY].cost_per_unit
            cost += memory_cost
        
        return cost

