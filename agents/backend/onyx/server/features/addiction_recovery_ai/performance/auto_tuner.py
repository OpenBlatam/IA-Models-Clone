"""
Auto-Tuner
Automatically tunes performance parameters based on metrics
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
import time

logger = logging.getLogger(__name__)


@dataclass
class TuningMetric:
    """Performance metric for tuning"""
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    target: Optional[float] = None
    weight: float = 1.0


@dataclass
class TuningParameter:
    """Tunable parameter"""
    name: str
    current_value: float
    min_value: float
    max_value: float
    step: float
    impact: float = 1.0  # Impact on performance (0-1)


class AutoTuner:
    """
    Auto-tuner for performance parameters
    
    Features:
    - Automatic parameter tuning
    - Metric-based optimization
    - Gradient descent-like optimization
    - Safety bounds
    - Learning from history
    """
    
    def __init__(
        self,
        parameters: Dict[str, TuningParameter],
        objective_function: Callable[[Dict[str, float]], float],
        learning_rate: float = 0.1
    ):
        self.parameters = parameters
        self.objective_function = objective_function
        self.learning_rate = learning_rate
        
        self._metrics_history: deque = deque(maxlen=1000)
        self._parameter_history: deque = deque(maxlen=100)
        self._best_score: Optional[float] = None
        self._best_parameters: Optional[Dict[str, float]] = None
        
        self._tuning_active = False
        self._tuning_task: Optional[asyncio.Task] = None
        
        logger.info(f"✅ Auto-tuner initialized with {len(parameters)} parameters")
    
    def record_metric(self, metric: TuningMetric):
        """Record performance metric"""
        self._metrics_history.append(metric)
    
    def get_current_parameters(self) -> Dict[str, float]:
        """Get current parameter values"""
        return {
            name: param.current_value
            for name, param in self.parameters.items()
        }
    
    async def tune(self):
        """Perform one tuning iteration"""
        if not self._tuning_active:
            return
        
        # Get current performance score
        current_params = self.get_current_parameters()
        current_score = await self._evaluate_performance(current_params)
        
        # Try adjusting each parameter
        improvements = {}
        
        for param_name, param in self.parameters.items():
            # Try increasing
            if param.current_value < param.max_value:
                test_params = current_params.copy()
                test_params[param_name] = min(
                    param.max_value,
                    param.current_value + param.step
                )
                test_score = await self._evaluate_performance(test_params)
                
                if test_score > current_score:
                    improvements[param_name] = {
                        "direction": "increase",
                        "score": test_score,
                        "value": test_params[param_name]
                    }
            
            # Try decreasing
            if param.current_value > param.min_value:
                test_params = current_params.copy()
                test_params[param_name] = max(
                    param.min_value,
                    param.current_value - param.step
                )
                test_score = await self._evaluate_performance(test_params)
                
                if test_score > current_score:
                    if param_name not in improvements or test_score > improvements[param_name]["score"]:
                        improvements[param_name] = {
                            "direction": "decrease",
                            "score": test_score,
                            "value": test_params[param_name]
                        }
        
        # Apply best improvement
        if improvements:
            best_param = max(improvements.items(), key=lambda x: x[1]["score"])
            param_name, improvement = best_param
            
            old_value = self.parameters[param_name].current_value
            self.parameters[param_name].current_value = improvement["value"]
            
            logger.info(
                f"Tuned {param_name}: {old_value:.2f} -> {improvement['value']:.2f} "
                f"(score: {current_score:.2f} -> {improvement['score']:.2f})"
            )
            
            # Record in history
            self._parameter_history.append({
                "timestamp": time.time(),
                "parameter": param_name,
                "old_value": old_value,
                "new_value": improvement["value"],
                "score_improvement": improvement["score"] - current_score
            })
            
            # Update best if better
            if self._best_score is None or improvement["score"] > self._best_score:
                self._best_score = improvement["score"]
                self._best_parameters = self.get_current_parameters()
    
    async def _evaluate_performance(self, parameters: Dict[str, float]) -> float:
        """Evaluate performance with given parameters"""
        try:
            # Call objective function
            if asyncio.iscoroutinefunction(self.objective_function):
                score = await self.objective_function(parameters)
            else:
                score = self.objective_function(parameters)
            
            return score
        except Exception as e:
            logger.error(f"Error evaluating performance: {e}")
            return 0.0
    
    def start_tuning(self, interval: float = 60.0):
        """Start automatic tuning"""
        if self._tuning_active:
            return
        
        self._tuning_active = True
        self._tuning_task = asyncio.create_task(self._tuning_loop(interval))
        logger.info(f"Auto-tuning started (interval: {interval}s)")
    
    def stop_tuning(self):
        """Stop automatic tuning"""
        self._tuning_active = False
        if self._tuning_task:
            self._tuning_task.cancel()
            try:
                asyncio.get_event_loop().run_until_complete(self._tuning_task)
            except asyncio.CancelledError:
                pass
        logger.info("Auto-tuning stopped")
    
    async def _tuning_loop(self, interval: float):
        """Tuning loop"""
        while self._tuning_active:
            try:
                await self.tune()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Tuning error: {e}")
                await asyncio.sleep(interval)
    
    def get_tuning_stats(self) -> Dict[str, Any]:
        """Get tuning statistics"""
        return {
            "active": self._tuning_active,
            "parameters": self.get_current_parameters(),
            "best_score": self._best_score,
            "best_parameters": self._best_parameters,
            "history_size": len(self._parameter_history),
            "metrics_count": len(self._metrics_history)
        }
    
    def reset_to_best(self):
        """Reset parameters to best known values"""
        if self._best_parameters:
            for param_name, value in self._best_parameters.items():
                if param_name in self.parameters:
                    self.parameters[param_name].current_value = value
            logger.info("Reset parameters to best known values")


# Global tuner instance
_tuner: Optional[AutoTuner] = None


def create_auto_tuner(
    parameters: Dict[str, TuningParameter],
    objective_function: Callable[[Dict[str, float]], float]
) -> AutoTuner:
    """Create auto-tuner instance"""
    return AutoTuner(parameters, objective_function)















