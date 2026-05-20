"""
Cost Optimizer for Color Grading AI
====================================

Cost optimization and resource usage management.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class CostType(Enum):
    """Cost types."""
    COMPUTE = "compute"  # CPU/GPU compute
    STORAGE = "storage"  # Storage costs
    NETWORK = "network"  # Network bandwidth
    API = "api"  # External API calls
    TOTAL = "total"  # Total cost


@dataclass
class CostEstimate:
    """Cost estimate."""
    cost_type: CostType
    amount: float
    unit: str = "USD"
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationRecommendation:
    """Optimization recommendation."""
    category: str
    recommendation: str
    potential_savings: float
    impact: str  # low, medium, high
    effort: str  # low, medium, high


class CostOptimizer:
    """
    Cost optimization system.
    
    Features:
    - Cost tracking
    - Resource usage analysis
    - Optimization recommendations
    - Cost estimation
    - Budget management
    """
    
    def __init__(self):
        """Initialize cost optimizer."""
        self._cost_history: List[CostEstimate] = []
        self._resource_usage: Dict[str, float] = {}
        self._cost_rates: Dict[CostType, float] = {
            CostType.COMPUTE: 0.001,  # per CPU-hour
            CostType.STORAGE: 0.0001,  # per GB-month
            CostType.NETWORK: 0.00001,  # per GB
            CostType.API: 0.01,  # per API call
        }
    
    def record_cost(
        self,
        cost_type: CostType,
        amount: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Record a cost.
        
        Args:
            cost_type: Cost type
            amount: Cost amount
            metadata: Optional metadata
        """
        estimate = CostEstimate(
            cost_type=cost_type,
            amount=amount,
            metadata=metadata or {}
        )
        self._cost_history.append(estimate)
        logger.debug(f"Recorded cost: {cost_type.value} = {amount}")
    
    def estimate_operation_cost(
        self,
        operation: str,
        duration_seconds: float,
        resource_usage: Dict[str, float]
    ) -> CostEstimate:
        """
        Estimate cost for an operation.
        
        Args:
            operation: Operation name
            duration_seconds: Operation duration
            resource_usage: Resource usage dictionary
            
        Returns:
            Cost estimate
        """
        total_cost = 0.0
        
        # Compute cost
        cpu_hours = (resource_usage.get("cpu_percent", 0) / 100) * (duration_seconds / 3600)
        compute_cost = cpu_hours * self._cost_rates[CostType.COMPUTE]
        total_cost += compute_cost
        
        # Storage cost (if applicable)
        storage_gb = resource_usage.get("storage_gb", 0)
        storage_cost = storage_gb * self._cost_rates[CostType.STORAGE]
        total_cost += storage_cost
        
        # Network cost (if applicable)
        network_gb = resource_usage.get("network_gb", 0)
        network_cost = network_gb * self._cost_rates[CostType.NETWORK]
        total_cost += network_cost
        
        return CostEstimate(
            cost_type=CostType.TOTAL,
            amount=total_cost,
            metadata={
                "operation": operation,
                "duration_seconds": duration_seconds,
                "compute_cost": compute_cost,
                "storage_cost": storage_cost,
                "network_cost": network_cost,
            }
        )
    
    def get_cost_summary(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get cost summary for a period.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Cost summary
        """
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now()
        
        filtered_costs = [
            cost for cost in self._cost_history
            if start_date <= cost.timestamp <= end_date
        ]
        
        total_by_type = {}
        for cost_type in CostType:
            total_by_type[cost_type.value] = sum(
                c.amount for c in filtered_costs
                if c.cost_type == cost_type
            )
        
        total_cost = sum(c.amount for c in filtered_costs)
        
        return {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "total_cost": total_cost,
            "costs_by_type": total_by_type,
            "transaction_count": len(filtered_costs),
        }
    
    def get_optimization_recommendations(self) -> List[OptimizationRecommendation]:
        """
        Get cost optimization recommendations.
        
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Analyze costs
        total_cost = sum(c.amount for c in self._cost_history)
        compute_cost = sum(
            c.amount for c in self._cost_history
            if c.cost_type == CostType.COMPUTE
        )
        
        # Recommendation: Optimize compute usage
        if compute_cost > total_cost * 0.5:
            recommendations.append(OptimizationRecommendation(
                category="compute",
                recommendation="Consider using batch processing to reduce compute costs",
                potential_savings=compute_cost * 0.2,
                impact="high",
                effort="medium"
            ))
        
        # Recommendation: Cache optimization
        if len(self._cost_history) > 100:
            recommendations.append(OptimizationRecommendation(
                category="caching",
                recommendation="Increase cache hit rate to reduce redundant processing",
                potential_savings=total_cost * 0.15,
                impact="medium",
                effort="low"
            ))
        
        return recommendations
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cost statistics."""
        total_cost = sum(c.amount for c in self._cost_history)
        
        return {
            "total_cost": total_cost,
            "total_transactions": len(self._cost_history),
            "cost_rates": {
                cost_type.value: rate
                for cost_type, rate in self._cost_rates.items()
            },
        }

