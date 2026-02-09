"""
Budget Manager
==============

Advanced budget management and alerts.
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class BudgetAlertLevel(Enum):
    """Budget alert levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Budget:
    """Budget definition."""
    name: str
    amount: float
    period: str  # daily, weekly, monthly, yearly
    alert_thresholds: List[float] = None  # [50%, 80%, 100%]
    
    def __post_init__(self):
        if self.alert_thresholds is None:
            self.alert_thresholds = [0.5, 0.8, 1.0]


@dataclass
class BudgetAlert:
    """Budget alert."""
    budget_name: str
    level: BudgetAlertLevel
    current_spend: float
    budget_amount: float
    percentage: float
    message: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class BudgetManager:
    """Budget manager with alerts."""
    
    def __init__(self):
        self._budgets: Dict[str, Budget] = {}
        self._spending: Dict[str, List[Dict[str, Any]]] = {}
        self._alerts: List[BudgetAlert] = []
        self._alert_callbacks: Dict[str, List[Callable]] = {}
    
    def create_budget(
        self,
        name: str,
        amount: float,
        period: str = "monthly",
        alert_thresholds: Optional[List[float]] = None
    ):
        """Create budget."""
        budget = Budget(
            name=name,
            amount=amount,
            period=period,
            alert_thresholds=alert_thresholds or [0.5, 0.8, 1.0]
        )
        
        self._budgets[name] = budget
        self._spending[name] = []
        logger.info(f"Created budget: {name} = {amount} USD/{period}")
    
    def record_spending(self, budget_name: str, amount: float, description: str = ""):
        """Record spending against budget."""
        if budget_name not in self._budgets:
            logger.warning(f"Budget {budget_name} not found")
            return
        
        self._spending[budget_name].append({
            "amount": amount,
            "description": description,
            "timestamp": datetime.now()
        })
        
        # Check alerts
        self._check_alerts(budget_name)
    
    def _check_alerts(self, budget_name: str):
        """Check budget alerts."""
        if budget_name not in self._budgets:
            return
        
        budget = self._budgets[budget_name]
        current_spend = self.get_current_spend(budget_name)
        percentage = current_spend / budget.amount if budget.amount > 0 else 0
        
        # Check thresholds
        for threshold in budget.alert_thresholds:
            if percentage >= threshold:
                level = self._get_alert_level(percentage)
                
                alert = BudgetAlert(
                    budget_name=budget_name,
                    level=level,
                    current_spend=current_spend,
                    budget_amount=budget.amount,
                    percentage=percentage,
                    message=f"Budget {budget_name} is at {percentage:.1%} ({current_spend:.2f}/{budget.amount:.2f} USD)"
                )
                
                self._alerts.append(alert)
                
                # Trigger callbacks
                if budget_name in self._alert_callbacks:
                    for callback in self._alert_callbacks[budget_name]:
                        try:
                            callback(alert)
                        except Exception as e:
                            logger.error(f"Alert callback failed: {e}")
                
                logger.warning(alert.message)
    
    def _get_alert_level(self, percentage: float) -> BudgetAlertLevel:
        """Get alert level based on percentage."""
        if percentage >= 1.0:
            return BudgetAlertLevel.CRITICAL
        elif percentage >= 0.8:
            return BudgetAlertLevel.WARNING
        else:
            return BudgetAlertLevel.INFO
    
    def get_current_spend(self, budget_name: str, period_days: Optional[int] = None) -> float:
        """Get current spending for budget."""
        if budget_name not in self._spending:
            return 0.0
        
        spending = self._spending[budget_name]
        
        if period_days:
            cutoff = datetime.now() - timedelta(days=period_days)
            spending = [s for s in spending if s["timestamp"] > cutoff]
        
        return sum(s["amount"] for s in spending)
    
    def register_alert_callback(self, budget_name: str, callback: Callable):
        """Register alert callback."""
        if budget_name not in self._alert_callbacks:
            self._alert_callbacks[budget_name] = []
        self._alert_callbacks[budget_name].append(callback)
    
    def get_budget_status(self, budget_name: str) -> Dict[str, Any]:
        """Get budget status."""
        if budget_name not in self._budgets:
            return {}
        
        budget = self._budgets[budget_name]
        current_spend = self.get_current_spend(budget_name)
        percentage = current_spend / budget.amount if budget.amount > 0 else 0
        
        return {
            "budget_name": budget_name,
            "budget_amount": budget.amount,
            "current_spend": current_spend,
            "remaining": budget.amount - current_spend,
            "percentage": percentage,
            "period": budget.period,
            "status": "over_budget" if percentage >= 1.0 else "on_track"
        }
    
    def get_all_budgets_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all budgets."""
        return {
            name: self.get_budget_status(name)
            for name in self._budgets.keys()
        }
    
    def get_alerts(self, level: Optional[BudgetAlertLevel] = None, limit: int = 100) -> List[BudgetAlert]:
        """Get budget alerts."""
        alerts = self._alerts
        
        if level:
            alerts = [a for a in alerts if a.level == level]
        
        return alerts[-limit:]

