"""
Cost Optimizer for Flux2 Clothing Changer
==========================================

Cost optimization and budget management.
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from collections import defaultdict, deque
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class CostRecord:
    """Cost record."""
    service: str
    operation: str
    cost: float
    currency: str = "USD"
    timestamp: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if self.metadata is None:
            self.metadata = {}


class CostOptimizer:
    """Cost optimization and budget management system."""
    
    def __init__(
        self,
        budget_daily: Optional[float] = None,
        budget_monthly: Optional[float] = None,
    ):
        """
        Initialize cost optimizer.
        
        Args:
            budget_daily: Daily budget limit
            budget_monthly: Monthly budget limit
        """
        self.budget_daily = budget_daily
        self.budget_monthly = budget_monthly
        
        self.cost_records: deque = deque(maxlen=10000)
        self.cost_by_service: Dict[str, float] = defaultdict(float)
        self.cost_by_operation: Dict[str, float] = defaultdict(float)
        
        # Pricing models
        self.pricing = {
            "inference": 0.001,  # per request
            "storage": 0.0001,  # per MB per day
            "api_calls": 0.0005,  # per API call
        }
    
    def record_cost(
        self,
        service: str,
        operation: str,
        cost: float,
        currency: str = "USD",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a cost.
        
        Args:
            service: Service name
            operation: Operation name
            cost: Cost amount
            currency: Currency code
            metadata: Optional metadata
        """
        record = CostRecord(
            service=service,
            operation=operation,
            cost=cost,
            currency=currency,
            metadata=metadata or {},
        )
        
        self.cost_records.append(record)
        self.cost_by_service[service] += cost
        self.cost_by_operation[operation] += cost
    
    def estimate_cost(
        self,
        service: str,
        operation: str,
        quantity: int = 1,
    ) -> float:
        """
        Estimate cost for operation.
        
        Args:
            service: Service name
            operation: Operation name
            quantity: Quantity
            
        Returns:
            Estimated cost
        """
        # Use pricing model or historical average
        if operation in self.pricing:
            return self.pricing[operation] * quantity
        
        # Calculate from history
        historical_costs = [
            r.cost for r in self.cost_records
            if r.service == service and r.operation == operation
        ]
        
        if historical_costs:
            avg_cost = sum(historical_costs) / len(historical_costs)
            return avg_cost * quantity
        
        return 0.0
    
    def get_daily_cost(
        self,
        date: Optional[str] = None,
    ) -> float:
        """
        Get daily cost.
        
        Args:
            date: Date string (YYYY-MM-DD) or None for today
            
        Returns:
            Daily cost
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        date_start = datetime.strptime(date, "%Y-%m-%d").timestamp()
        date_end = date_start + 86400  # 24 hours
        
        daily_costs = [
            r.cost for r in self.cost_records
            if date_start <= r.timestamp < date_end
        ]
        
        return sum(daily_costs)
    
    def get_monthly_cost(
        self,
        year: Optional[int] = None,
        month: Optional[int] = None,
    ) -> float:
        """
        Get monthly cost.
        
        Args:
            year: Year (None for current)
            month: Month (None for current)
            
        Returns:
            Monthly cost
        """
        if year is None:
            year = datetime.now().year
        if month is None:
            month = datetime.now().month
        
        month_start = datetime(year, month, 1).timestamp()
        if month == 12:
            month_end = datetime(year + 1, 1, 1).timestamp()
        else:
            month_end = datetime(year, month + 1, 1).timestamp()
        
        monthly_costs = [
            r.cost for r in self.cost_records
            if month_start <= r.timestamp < month_end
        ]
        
        return sum(monthly_costs)
    
    def check_budget(
        self,
        estimated_cost: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Check budget status.
        
        Args:
            estimated_cost: Estimated cost for upcoming operation
            
        Returns:
            Budget status
        """
        daily_cost = self.get_daily_cost()
        monthly_cost = self.get_monthly_cost()
        
        status = {
            "daily_cost": daily_cost,
            "monthly_cost": monthly_cost,
            "estimated_cost": estimated_cost,
            "budget_status": "ok",
            "warnings": [],
        }
        
        # Check daily budget
        if self.budget_daily:
            remaining_daily = self.budget_daily - daily_cost - estimated_cost
            if remaining_daily < 0:
                status["budget_status"] = "exceeded"
                status["warnings"].append("Daily budget exceeded")
            elif remaining_daily < self.budget_daily * 0.1:
                status["budget_status"] = "warning"
                status["warnings"].append("Daily budget nearly exhausted")
        
        # Check monthly budget
        if self.budget_monthly:
            remaining_monthly = self.budget_monthly - monthly_cost - estimated_cost
            if remaining_monthly < 0:
                status["budget_status"] = "exceeded"
                status["warnings"].append("Monthly budget exceeded")
            elif remaining_monthly < self.budget_monthly * 0.1:
                status["budget_status"] = "warning"
                status["warnings"].append("Monthly budget nearly exhausted")
        
        return status
    
    def get_cost_breakdown(
        self,
        time_range: Optional[timedelta] = None,
    ) -> Dict[str, Any]:
        """
        Get cost breakdown.
        
        Args:
            time_range: Time range to analyze
            
        Returns:
            Cost breakdown
        """
        cutoff_time = time.time() - time_range.total_seconds() if time_range else 0
        
        relevant_records = [
            r for r in self.cost_records
            if r.timestamp >= cutoff_time
        ]
        
        total_cost = sum(r.cost for r in relevant_records)
        
        by_service = defaultdict(float)
        by_operation = defaultdict(float)
        
        for record in relevant_records:
            by_service[record.service] += record.cost
            by_operation[record.operation] += record.cost
        
        return {
            "total_cost": total_cost,
            "by_service": dict(by_service),
            "by_operation": dict(by_operation),
            "record_count": len(relevant_records),
        }
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get cost optimization recommendations."""
        recommendations = []
        
        # Analyze costs
        breakdown = self.get_cost_breakdown()
        
        # Find expensive operations
        if breakdown["by_operation"]:
            expensive_ops = sorted(
                breakdown["by_operation"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            for op, cost in expensive_ops:
                recommendations.append(
                    f"Operation '{op}' is expensive (${cost:.2f}). Consider optimization."
                )
        
        # Check for unused services
        recent_services = set(
            r.service for r in list(self.cost_records)[-100:]
        )
        all_services = set(self.cost_by_service.keys())
        unused = all_services - recent_services
        
        if unused:
            recommendations.append(
                f"Unused services detected: {', '.join(unused)}. Consider removing."
            )
        
        if not recommendations:
            recommendations.append("Costs are optimized")
        
        return recommendations
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cost statistics."""
        return {
            "total_records": len(self.cost_records),
            "total_cost_all_time": sum(r.cost for r in self.cost_records),
            "daily_cost": self.get_daily_cost(),
            "monthly_cost": self.get_monthly_cost(),
            "cost_by_service": dict(self.cost_by_service),
            "cost_by_operation": dict(self.cost_by_operation),
            "budget_daily": self.budget_daily,
            "budget_monthly": self.budget_monthly,
        }


