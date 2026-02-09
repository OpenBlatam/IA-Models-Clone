"""
Cost Optimization

Optimizations for:
- Resource cost tracking
- Cost optimization strategies
- Billing optimization
- Resource right-sizing
- Cost alerts
"""

import logging
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class ResourceCost:
    """Resource cost information."""
    resource_type: str
    usage: float
    unit_cost: float
    total_cost: float
    timestamp: datetime


class CostTracker:
    """Cost tracking and optimization."""
    
    def __init__(self):
        """Initialize cost tracker."""
        self.costs: List[ResourceCost] = []
        self.daily_budget: Optional[float] = None
        self.monthly_budget: Optional[float] = None
    
    def track_cost(
        self,
        resource_type: str,
        usage: float,
        unit_cost: float
    ) -> None:
        """
        Track resource cost.
        
        Args:
            resource_type: Type of resource (GPU, CPU, Storage, etc.)
            usage: Usage amount
            unit_cost: Cost per unit
        """
        cost = ResourceCost(
            resource_type=resource_type,
            usage=usage,
            unit_cost=unit_cost,
            total_cost=usage * unit_cost,
            timestamp=datetime.now()
        )
        
        self.costs.append(cost)
        
        # Check budget alerts
        self._check_budget()
    
    def get_daily_cost(self, date: Optional[datetime] = None) -> float:
        """
        Get total cost for day.
        
        Args:
            date: Date to check (today if None)
            
        Returns:
            Total daily cost
        """
        if date is None:
            date = datetime.now()
        
        day_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        
        daily_costs = [
            cost for cost in self.costs
            if day_start <= cost.timestamp < day_end
        ]
        
        return sum(cost.total_cost for cost in daily_costs)
    
    def get_monthly_cost(self, month: Optional[datetime] = None) -> float:
        """
        Get total cost for month.
        
        Args:
            month: Month to check (current month if None)
            
        Returns:
            Total monthly cost
        """
        if month is None:
            month = datetime.now()
        
        month_start = month.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if month.month == 12:
            month_end = month_start.replace(year=month.year + 1, month=1)
        else:
            month_end = month_start.replace(month=month.month + 1)
        
        monthly_costs = [
            cost for cost in self.costs
            if month_start <= cost.timestamp < month_end
        ]
        
        return sum(cost.total_cost for cost in monthly_costs)
    
    def set_budget(self, daily: Optional[float] = None, monthly: Optional[float] = None) -> None:
        """
        Set budget limits.
        
        Args:
            daily: Daily budget
            monthly: Monthly budget
        """
        self.daily_budget = daily
        self.monthly_budget = monthly
    
    def _check_budget(self) -> None:
        """Check if budget is exceeded."""
        if self.daily_budget:
            daily_cost = self.get_daily_cost()
            if daily_cost > self.daily_budget:
                logger.warning(f"Daily budget exceeded: ${daily_cost:.2f} / ${self.daily_budget:.2f}")
        
        if self.monthly_budget:
            monthly_cost = self.get_monthly_cost()
            if monthly_cost > self.monthly_budget:
                logger.warning(f"Monthly budget exceeded: ${monthly_cost:.2f} / ${self.monthly_budget:.2f}")
    
    def get_cost_breakdown(self) -> Dict[str, float]:
        """Get cost breakdown by resource type."""
        breakdown = {}
        
        for cost in self.costs:
            if cost.resource_type not in breakdown:
                breakdown[cost.resource_type] = 0.0
            breakdown[cost.resource_type] += cost.total_cost
        
        return breakdown


class CostOptimizer:
    """Cost optimization strategies."""
    
    @staticmethod
    def estimate_gpu_cost(
        hours: float,
        gpu_type: str = "A100",
        region: str = "us-east-1"
    ) -> float:
        """
        Estimate GPU cost.
        
        Args:
            hours: Hours of usage
            gpu_type: GPU type
            region: AWS region
            
        Returns:
            Estimated cost
        """
        # Simplified pricing (actual prices vary)
        pricing = {
            "A100": 3.06,  # per hour
            "V100": 2.48,
            "T4": 0.35,
            "CPU": 0.10
        }
        
        hourly_rate = pricing.get(gpu_type, 1.0)
        return hours * hourly_rate
    
    @staticmethod
    def optimize_for_cost(
        current_config: Dict[str, Any],
        target_cost_reduction: float = 0.2
    ) -> Dict[str, Any]:
        """
        Optimize configuration for cost reduction.
        
        Args:
            current_config: Current configuration
            target_cost_reduction: Target cost reduction (0.0 to 1.0)
            
        Returns:
            Optimized configuration
        """
        optimized = current_config.copy()
        
        # Reduce workers if possible
        if 'workers' in optimized:
            optimized['workers'] = max(1, int(optimized['workers'] * (1 - target_cost_reduction)))
        
        # Reduce pool size
        if 'pool_size' in optimized:
            optimized['pool_size'] = max(5, int(optimized['pool_size'] * (1 - target_cost_reduction)))
        
        # Enable caching more aggressively
        optimized['enable_cache'] = True
        optimized['cache_ttl'] = optimized.get('cache_ttl', 300) * 2
        
        # Use smaller models if available
        if 'model_size' in optimized:
            if optimized['model_size'] == 'large':
                optimized['model_size'] = 'medium'
            elif optimized['model_size'] == 'medium':
                optimized['model_size'] = 'small'
        
        return optimized
    
    @staticmethod
    def recommend_right_sizing(
        current_usage: Dict[str, float],
        target_utilization: float = 0.7
    ) -> Dict[str, Any]:
        """
        Recommend right-sizing based on usage.
        
        Args:
            current_usage: Current resource usage
            target_utilization: Target utilization (0.0 to 1.0)
            
        Returns:
            Right-sizing recommendations
        """
        recommendations = {}
        
        for resource, usage in current_usage.items():
            if usage < target_utilization * 0.5:
                recommendations[resource] = "downsize"
            elif usage > target_utilization * 1.2:
                recommendations[resource] = "upsize"
            else:
                recommendations[resource] = "optimal"
        
        return recommendations








