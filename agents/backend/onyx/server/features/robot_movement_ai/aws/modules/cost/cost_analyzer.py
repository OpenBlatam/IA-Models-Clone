"""
Cost Analyzer
=============

Analyze and optimize costs.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class CostItem:
    """Cost item."""
    service: str
    resource: str
    cost: float
    unit: str = "USD"
    period: str = "hour"
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class CostAnalyzer:
    """Cost analyzer and optimizer."""
    
    def __init__(self):
        self._costs: List[CostItem] = []
        self._budgets: Dict[str, float] = {}
        self._alerts: List[Dict[str, Any]] = []
    
    def record_cost(
        self,
        service: str,
        resource: str,
        cost: float,
        unit: str = "USD",
        period: str = "hour"
    ):
        """Record cost."""
        cost_item = CostItem(
            service=service,
            resource=resource,
            cost=cost,
            unit=unit,
            period=period
        )
        
        self._costs.append(cost_item)
        
        # Check budget
        self._check_budget(service, cost)
        
        logger.debug(f"Recorded cost: {service}/{resource} = {cost} {unit}/{period}")
    
    def set_budget(self, service: str, budget: float, period: str = "month"):
        """Set budget for service."""
        self._budgets[service] = budget
        logger.info(f"Set budget for {service}: {budget} USD/{period}")
    
    def _check_budget(self, service: str, cost: float):
        """Check if cost exceeds budget."""
        if service not in self._budgets:
            return
        
        budget = self._budgets[service]
        
        # Calculate current period cost
        current_cost = self.get_service_cost(service, period_days=30)
        
        if current_cost > budget:
            alert = {
                "service": service,
                "budget": budget,
                "current_cost": current_cost,
                "exceeded_by": current_cost - budget,
                "timestamp": datetime.now().isoformat()
            }
            self._alerts.append(alert)
            logger.warning(f"Budget exceeded for {service}: {current_cost} > {budget}")
    
    def get_service_cost(self, service: str, period_days: int = 30) -> float:
        """Get service cost for period."""
        cutoff = datetime.now() - timedelta(days=period_days)
        
        service_costs = [
            item.cost for item in self._costs
            if item.service == service and item.timestamp > cutoff
        ]
        
        return sum(service_costs)
    
    def get_total_cost(self, period_days: int = 30) -> float:
        """Get total cost for period."""
        cutoff = datetime.now() - timedelta(days=period_days)
        
        period_costs = [
            item.cost for item in self._costs
            if item.timestamp > cutoff
        ]
        
        return sum(period_costs)
    
    def get_cost_breakdown(self, period_days: int = 30) -> Dict[str, float]:
        """Get cost breakdown by service."""
        cutoff = datetime.now() - timedelta(days=period_days)
        
        breakdown = {}
        for item in self._costs:
            if item.timestamp > cutoff:
                if item.service not in breakdown:
                    breakdown[item.service] = 0.0
                breakdown[item.service] += item.cost
        
        return breakdown
    
    def get_cost_recommendations(self) -> List[Dict[str, Any]]:
        """Get cost optimization recommendations."""
        recommendations = []
        
        # Analyze costs
        breakdown = self.get_cost_breakdown(period_days=30)
        total = sum(breakdown.values())
        
        for service, cost in breakdown.items():
            percentage = (cost / total * 100) if total > 0 else 0
            
            if percentage > 50:
                recommendations.append({
                    "type": "high_cost",
                    "service": service,
                    "cost": cost,
                    "percentage": percentage,
                    "recommendation": f"Service {service} accounts for {percentage:.1f}% of costs. Consider optimization."
                })
        
        return recommendations
    
    def get_cost_stats(self) -> Dict[str, Any]:
        """Get cost statistics."""
        return {
            "total_cost_30d": self.get_total_cost(period_days=30),
            "total_cost_7d": self.get_total_cost(period_days=7),
            "breakdown": self.get_cost_breakdown(period_days=30),
            "budgets": self._budgets.copy(),
            "alerts": len(self._alerts),
            "recent_alerts": self._alerts[-10:]
        }















