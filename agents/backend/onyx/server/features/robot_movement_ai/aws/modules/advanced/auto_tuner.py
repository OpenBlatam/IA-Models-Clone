"""
Auto Tuner
==========

Automatic performance tuning based on metrics.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import statistics

logger = logging.getLogger(__name__)


@dataclass
class TuningParameter:
    """Tuning parameter definition."""
    name: str
    current_value: float
    min_value: float
    max_value: float
    step: float
    target_metric: str
    optimize_for: str = "minimize"  # minimize or maximize


@dataclass
class TuningResult:
    """Tuning result."""
    parameter: str
    old_value: float
    new_value: float
    improvement: float
    metric_before: float
    metric_after: float


class AutoTuner:
    """Automatic performance tuner."""
    
    def __init__(self):
        self._parameters: Dict[str, TuningParameter] = {}
        self._metrics_history: Dict[str, list] = {}
        self._tuning_history: list = []
        self._tuning_active = False
    
    def register_parameter(
        self,
        name: str,
        initial_value: float,
        min_value: float,
        max_value: float,
        step: float,
        target_metric: str,
        optimize_for: str = "minimize"
    ):
        """Register tuning parameter."""
        self._parameters[name] = TuningParameter(
            name=name,
            current_value=initial_value,
            min_value=min_value,
            max_value=max_value,
            step=step,
            target_metric=target_metric,
            optimize_for=optimize_for
        )
        
        if target_metric not in self._metrics_history:
            self._metrics_history[target_metric] = []
        
        logger.info(f"Registered tuning parameter: {name}")
    
    def record_metric(self, metric_name: str, value: float):
        """Record metric value."""
        if metric_name not in self._metrics_history:
            self._metrics_history[metric_name] = []
        
        self._metrics_history[metric_name].append({
            "value": value,
            "timestamp": datetime.now()
        })
        
        # Keep only last 1000 entries
        if len(self._metrics_history[metric_name]) > 1000:
            self._metrics_history[metric_name] = self._metrics_history[metric_name][-500:]
    
    def get_average_metric(self, metric_name: str, last_minutes: int = 5) -> Optional[float]:
        """Get average metric value."""
        if metric_name not in self._metrics_history:
            return None
        
        cutoff = datetime.now() - timedelta(minutes=last_minutes)
        recent = [
            m["value"] for m in self._metrics_history[metric_name]
            if m["timestamp"] > cutoff
        ]
        
        if not recent:
            return None
        
        return statistics.mean(recent)
    
    async def tune_parameter(self, parameter_name: str) -> Optional[TuningResult]:
        """Tune single parameter."""
        if parameter_name not in self._parameters:
            return None
        
        param = self._parameters[parameter_name]
        
        # Get current metric
        current_metric = self.get_average_metric(param.target_metric)
        if current_metric is None:
            logger.warning(f"No metric data for {param.target_metric}")
            return None
        
        # Try increasing value
        new_value = min(param.current_value + param.step, param.max_value)
        old_value = param.current_value
        param.current_value = new_value
        
        # Wait for metrics to stabilize
        await asyncio.sleep(10)
        
        # Get new metric
        new_metric = self.get_average_metric(param.target_metric)
        
        if new_metric is None:
            # Revert
            param.current_value = old_value
            return None
        
        # Calculate improvement
        if param.optimize_for == "minimize":
            improvement = (current_metric - new_metric) / current_metric * 100
            better = new_metric < current_metric
        else:
            improvement = (new_metric - current_metric) / current_metric * 100
            better = new_metric > current_metric
        
        if not better:
            # Revert
            param.current_value = old_value
            improvement = -improvement
        
        result = TuningResult(
            parameter=parameter_name,
            old_value=old_value,
            new_value=param.current_value,
            improvement=improvement,
            metric_before=current_metric,
            metric_after=new_metric
        )
        
        self._tuning_history.append(result)
        logger.info(f"Tuned {parameter_name}: {old_value} -> {param.current_value} ({improvement:.2f}% improvement)")
        
        return result
    
    async def auto_tune(self, interval: float = 60.0):
        """Start automatic tuning."""
        self._tuning_active = True
        
        while self._tuning_active:
            for param_name in list(self._parameters.keys()):
                if not self._tuning_active:
                    break
                
                await self.tune_parameter(param_name)
                await asyncio.sleep(interval)
    
    def stop_tuning(self):
        """Stop automatic tuning."""
        self._tuning_active = False
        logger.info("Auto-tuning stopped")
    
    def get_tuning_stats(self) -> Dict[str, Any]:
        """Get tuning statistics."""
        return {
            "parameters": len(self._parameters),
            "tuning_attempts": len(self._tuning_history),
            "current_values": {
                name: param.current_value
                for name, param in self._parameters.items()
            }
        }















