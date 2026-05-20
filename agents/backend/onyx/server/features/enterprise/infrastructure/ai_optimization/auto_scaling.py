from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, List, Dict, Optional
"""
Intelligent Auto-Scaling with AI
================================

AI-powered auto-scaling using reinforcement learning and predictive analytics.
"""


logger = logging.getLogger(__name__)

class ScalingAction(Enum):
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"
    EMERGENCY_SCALE = "emergency_scale"


@dataclass
class ScalingDecision:
    """Auto-scaling decision with reasoning."""
    action: ScalingAction
    target_instances: int
    confidence: float
    reasoning: str
    predicted_load: float
    cost_impact: float


class IntelligentAutoScaler:
    """AI-powered auto-scaler with predictive capabilities."""
    
    def __init__(self, min_instances: int = 1, max_instances: int = 20):
        
    """__init__ function."""
self.min_instances = min_instances
        self.max_instances = max_instances
        self.current_instances = min_instances
        
        # Learning components
        self.load_history = []
        self.scaling_history = []
        self.performance_history = []
        
        # Prediction models (simplified)
        self.load_predictor = None
        self.cost_predictor = None
        
    async def make_scaling_decision(self, current_metrics: Dict[str, Any]) -> ScalingDecision:
        """Make intelligent scaling decision based on current metrics and predictions."""
        
        # Extract key metrics
        cpu_usage = current_metrics.get('cpu_usage', 0.5)
        memory_usage = current_metrics.get('memory_usage', 0.5)
        request_rate = current_metrics.get('request_rate', 100)
        response_time = current_metrics.get('response_time', 200)
        error_rate = current_metrics.get('error_rate', 0.01)
        
        # Predict future load
        predicted_load = await self._predict_load(current_metrics)
        
        # Calculate scaling need
        scaling_need = self._calculate_scaling_need(current_metrics, predicted_load)
        
        # Determine action
        if scaling_need > 0.7:
            action = ScalingAction.SCALE_UP
            target_instances = min(self.current_instances + 1, self.max_instances)
        elif scaling_need < -0.5:
            action = ScalingAction.SCALE_DOWN
            target_instances = max(self.current_instances - 1, self.min_instances)
        elif scaling_need > 0.9:  # Emergency scaling
            action = ScalingAction.EMERGENCY_SCALE
            target_instances = min(self.current_instances + 2, self.max_instances)
        else:
            action = ScalingAction.MAINTAIN
            target_instances = self.current_instances
        
        # Calculate confidence and cost
        confidence = self._calculate_confidence(current_metrics, predicted_load)
        cost_impact = self._estimate_cost_impact(target_instances)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(current_metrics, predicted_load, scaling_need)
        
        decision = ScalingDecision(
            action=action,
            target_instances=target_instances,
            confidence=confidence,
            reasoning=reasoning,
            predicted_load=predicted_load,
            cost_impact=cost_impact
        )
        
        # Record decision for learning
        self.scaling_history.append({
            'timestamp': datetime.utcnow(),
            'decision': decision,
            'metrics': current_metrics
        })
        
        return decision
    
    async def _predict_load(self, current_metrics: Dict[str, Any]) -> float:
        """Predict future load based on current metrics and history."""
        # Simple time-series prediction (can be enhanced with ML models)
        if len(self.load_history) < 5:
            return current_metrics.get('cpu_usage', 0.5)
        
        # Use recent trend
        recent_loads = [h['cpu_usage'] for h in self.load_history[-10:]]
        trend = np.polyfit(range(len(recent_loads)), recent_loads, 1)[0]
        
        # Predict next period
        current_load = current_metrics.get('cpu_usage', 0.5)
        predicted_load = current_load + (trend * 5)  # 5 time periods ahead
        
        return max(0.0, min(1.0, predicted_load))
    
    def _calculate_scaling_need(self, current_metrics: Dict[str, Any], predicted_load: float) -> float:
        """Calculate scaling need score (-1 to 1)."""
        cpu_pressure = current_metrics.get('cpu_usage', 0.5) - 0.7  # Target 70% CPU
        memory_pressure = current_metrics.get('memory_usage', 0.5) - 0.8  # Target 80% memory
        response_pressure = (current_metrics.get('response_time', 200) - 200) / 500  # Target 200ms
        error_pressure = current_metrics.get('error_rate', 0.01) * 10  # Scale errors
        
        # Future load pressure
        load_pressure = predicted_load - 0.7
        
        # Combine pressures
        total_pressure = (
            cpu_pressure * 0.3 +
            memory_pressure * 0.2 +
            response_pressure * 0.2 +
            error_pressure * 0.1 +
            load_pressure * 0.2
        )
        
        return max(-1.0, min(1.0, total_pressure))
    
    def _calculate_confidence(self, current_metrics: Dict[str, Any], predicted_load: float) -> float:
        """Calculate confidence in the scaling decision."""
        # Base confidence on prediction accuracy history
        if len(self.performance_history) < 5:
            return 0.7
        
        # Calculate recent prediction accuracy
        recent_accuracy = np.mean([h.get('prediction_accuracy', 0.7) for h in self.performance_history[-10:]])
        
        # Factor in metrics stability
        stability_score = 1.0 - abs(current_metrics.get('cpu_usage', 0.5) - predicted_load)
        
        confidence = (recent_accuracy * 0.6) + (stability_score * 0.4)
        return max(0.1, min(1.0, confidence))
    
    def _estimate_cost_impact(self, target_instances: int) -> float:
        """Estimate cost impact of scaling decision."""
        current_cost = self.current_instances * 100  # $100 per instance
        target_cost = target_instances * 100
        cost_change = target_cost - current_cost
        
        return cost_change / max(current_cost, 1)  # Percentage change
    
    def _generate_reasoning(self, current_metrics: Dict[str, Any], predicted_load: float, scaling_need: float) -> str:
        """Generate human-readable reasoning for the scaling decision."""
        factors = []
        
        cpu = current_metrics.get('cpu_usage', 0.5)
        memory = current_metrics.get('memory_usage', 0.5)
        response_time = current_metrics.get('response_time', 200)
        
        if cpu > 0.8:
            factors.append(f"high CPU ({cpu:.1%})")
        if memory > 0.8:
            factors.append(f"high memory ({memory:.1%})")
        if response_time > 500:
            factors.append(f"slow responses ({response_time:.0f}ms)")
        if predicted_load > 0.8:
            factors.append(f"predicted load increase ({predicted_load:.1%})")
        
        factor_str = ", ".join(factors) if factors else "normal metrics"
        
        return f"Scaling need: {scaling_need:.2f}, factors: {factor_str}"
    
    def update_performance(self, actual_metrics: Dict[str, Any]):
        """Update performance history for learning."""
        self.load_history.append(actual_metrics)
        
        # Calculate prediction accuracy
        if len(self.scaling_history) > 0:
            last_decision = self.scaling_history[-1]
            predicted = last_decision['decision'].predicted_load
            actual = actual_metrics.get('cpu_usage', 0.5)
            accuracy = 1.0 - abs(predicted - actual)
            
            self.performance_history.append({
                'timestamp': datetime.utcnow(),
                'prediction_accuracy': accuracy,
                'actual_metrics': actual_metrics
            })
        
        # Maintain history size
        if len(self.load_history) > 1000:
            self.load_history = self.load_history[-500:]
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-500:]
    
    def get_scaling_insights(self) -> Dict[str, Any]:
        """Get insights about scaling performance."""
        if not self.performance_history:
            return {"message": "No performance data available"}
        
        avg_accuracy = np.mean([h['prediction_accuracy'] for h in self.performance_history[-50:]])
        
        scaling_actions = [h['decision'].action.value for h in self.scaling_history[-20:]]
        action_counts = {action: scaling_actions.count(action) for action in set(scaling_actions)}
        
        return {
            "current_instances": self.current_instances,
            "prediction_accuracy": avg_accuracy,
            "recent_actions": action_counts,
            "total_scaling_decisions": len(self.scaling_history),
            "performance_data_points": len(self.performance_history)
        } 