"""
Auto-Scaling System with Load Prediction
Following functional programming principles and intelligent scaling strategies
"""

import asyncio
import time
import math
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
from collections import deque
import statistics

logger = logging.getLogger(__name__)


# Pure functions for auto-scaling

class ScalingAction(str, Enum):
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"
    EMERGENCY_SCALE = "emergency_scale"


class ScalingTrigger(str, Enum):
    CPU_THRESHOLD = "cpu_threshold"
    MEMORY_THRESHOLD = "memory_threshold"
    REQUEST_RATE = "request_rate"
    RESPONSE_TIME = "response_time"
    PREDICTIVE = "predictive"
    MANUAL = "manual"


@dataclass(frozen=True)
class ScalingDecision:
    """Immutable scaling decision - pure data structure"""
    action: ScalingAction
    trigger: ScalingTrigger
    current_instances: int
    target_instances: int
    reason: str
    confidence: float
    estimated_impact: Dict[str, Any]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary - pure function"""
        return {
            "action": self.action.value,
            "trigger": self.trigger.value,
            "current_instances": self.current_instances,
            "target_instances": self.target_instances,
            "reason": self.reason,
            "confidence": self.confidence,
            "estimated_impact": self.estimated_impact,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass(frozen=True)
class LoadMetrics:
    """Immutable load metrics - pure data structure"""
    cpu_usage: float
    memory_usage: float
    request_rate: float
    response_time: float
    active_connections: int
    queue_length: int
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary - pure function"""
        return {
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "request_rate": self.request_rate,
            "response_time": self.response_time,
            "active_connections": self.active_connections,
            "queue_length": self.queue_length,
            "timestamp": self.timestamp.isoformat()
        }


def calculate_load_score(metrics: LoadMetrics) -> float:
    """Calculate overall load score - pure function"""
    # Weighted combination of metrics
    weights = {
        "cpu": 0.3,
        "memory": 0.2,
        "request_rate": 0.2,
        "response_time": 0.2,
        "connections": 0.1
    }
    
    # Normalize metrics to 0-1 scale
    cpu_score = min(1.0, metrics.cpu_usage / 100.0)
    memory_score = min(1.0, metrics.memory_usage / 100.0)
    request_score = min(1.0, metrics.request_rate / 1000.0)  # 1000 req/s max
    response_score = min(1.0, metrics.response_time / 5.0)  # 5s max
    connection_score = min(1.0, metrics.active_connections / 1000.0)  # 1000 conn max
    
    # Calculate weighted score
    load_score = (
        weights["cpu"] * cpu_score +
        weights["memory"] * memory_score +
        weights["request_rate"] * request_score +
        weights["response_time"] * response_score +
        weights["connections"] * connection_score
    )
    
    return max(0.0, min(1.0, load_score))


def predict_future_load(
    historical_metrics: List[LoadMetrics],
    prediction_horizon_minutes: int = 15
) -> float:
    """Predict future load using simple trend analysis - pure function"""
    if len(historical_metrics) < 3:
        return 0.5  # Default prediction
    
    # Extract load scores
    load_scores = [calculate_load_score(m) for m in historical_metrics]
    
    # Simple linear trend
    if len(load_scores) >= 2:
        # Calculate trend
        recent_scores = load_scores[-10:]  # Last 10 measurements
        if len(recent_scores) >= 2:
            trend = (recent_scores[-1] - recent_scores[0]) / len(recent_scores)
            
            # Predict future value
            future_score = recent_scores[-1] + (trend * prediction_horizon_minutes)
            
            # Apply some smoothing
            current_score = recent_scores[-1]
            predicted_score = (current_score * 0.7) + (future_score * 0.3)
            
            return max(0.0, min(1.0, predicted_score))
    
    # Fallback to recent average
    return sum(load_scores[-5:]) / min(5, len(load_scores))


def calculate_optimal_instances(
    current_load: float,
    target_load: float,
    current_instances: int,
    min_instances: int = 1,
    max_instances: int = 100
) -> int:
    """Calculate optimal number of instances - pure function"""
    if current_load <= 0 or target_load <= 0:
        return current_instances
    
    # Calculate required instances based on load ratio
    required_instances = math.ceil(
        (current_load / target_load) * current_instances
    )
    
    # Apply constraints
    optimal_instances = max(min_instances, min(max_instances, required_instances))
    
    return optimal_instances


def determine_scaling_action(
    current_instances: int,
    target_instances: int,
    scaling_threshold: float = 0.1
) -> ScalingAction:
    """Determine scaling action - pure function"""
    if target_instances > current_instances:
        difference = target_instances - current_instances
        if difference >= current_instances * scaling_threshold:
            return ScalingAction.SCALE_UP
    elif target_instances < current_instances:
        difference = current_instances - target_instances
        if difference >= current_instances * scaling_threshold:
            return ScalingAction.SCALE_DOWN
    
    return ScalingAction.MAINTAIN


def calculate_scaling_confidence(
    historical_accuracy: List[float],
    prediction_horizon: int,
    current_metrics: LoadMetrics
) -> float:
    """Calculate confidence in scaling decision - pure function"""
    if not historical_accuracy:
        return 0.5  # Default confidence
    
    # Base confidence on historical accuracy
    avg_accuracy = sum(historical_accuracy) / len(historical_accuracy)
    
    # Adjust for prediction horizon (longer = less confident)
    horizon_factor = max(0.5, 1.0 - (prediction_horizon / 60.0))
    
    # Adjust for current load stability
    stability_factor = 1.0
    if current_metrics.cpu_usage > 80 or current_metrics.memory_usage > 80:
        stability_factor = 0.8  # Less confident when system is stressed
    
    confidence = avg_accuracy * horizon_factor * stability_factor
    
    return max(0.1, min(1.0, confidence))


def estimate_scaling_impact(
    current_instances: int,
    target_instances: int,
    current_metrics: LoadMetrics
) -> Dict[str, Any]:
    """Estimate impact of scaling action - pure function"""
    if current_instances == target_instances:
        return {
            "load_reduction": 0.0,
            "cost_change_percent": 0.0,
            "estimated_response_time": current_metrics.response_time,
            "estimated_throughput": current_metrics.request_rate
        }
    
    # Calculate load reduction
    load_reduction = (target_instances - current_instances) / current_instances
    
    # Estimate cost change
    cost_change_percent = ((target_instances - current_instances) / current_instances) * 100
    
    # Estimate performance impact
    if target_instances > current_instances:
        # Scaling up
        estimated_response_time = current_metrics.response_time * (current_instances / target_instances)
        estimated_throughput = current_metrics.request_rate * (target_instances / current_instances)
    else:
        # Scaling down
        estimated_response_time = current_metrics.response_time * (current_instances / target_instances)
        estimated_throughput = current_metrics.request_rate * (target_instances / current_instances)
    
    return {
        "load_reduction": load_reduction,
        "cost_change_percent": cost_change_percent,
        "estimated_response_time": max(0.1, estimated_response_time),
        "estimated_throughput": max(0.1, estimated_throughput)
    }


# Auto-Scaling System Class

class AutoScalingSystem:
    """Auto-Scaling System with Load Prediction following functional principles"""
    
    def __init__(
        self,
        min_instances: int = 1,
        max_instances: int = 100,
        target_cpu_usage: float = 70.0,
        target_memory_usage: float = 80.0,
        target_response_time: float = 2.0,
        scaling_cooldown: int = 300,  # 5 minutes
        prediction_horizon: int = 15  # 15 minutes
    ):
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.target_cpu_usage = target_cpu_usage
        self.target_memory_usage = target_memory_usage
        self.target_response_time = target_response_time
        self.scaling_cooldown = scaling_cooldown
        self.prediction_horizon = prediction_horizon
        
        # Current state
        self.current_instances = min_instances
        self.last_scaling_time = None
        
        # Historical data
        self.metrics_history: deque = deque(maxlen=1000)
        self.scaling_history: List[ScalingDecision] = []
        self.prediction_accuracy: List[float] = []
        
        # Scaling callbacks
        self.scaling_callbacks: List[Callable] = []
        
        # Background tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        # Statistics
        self.stats = {
            "total_scaling_actions": 0,
            "scale_up_count": 0,
            "scale_down_count": 0,
            "maintain_count": 0,
            "average_prediction_accuracy": 0.0,
            "last_scaling_action": None
        }
    
    async def start(self) -> None:
        """Start auto-scaling system"""
        if self.is_running:
            return
        
        self.is_running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Auto-scaling system started")
    
    async def stop(self) -> None:
        """Stop auto-scaling system"""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Auto-scaling system stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.is_running:
            try:
                # Get current metrics (this would be provided by the monitoring system)
                current_metrics = await self._get_current_metrics()
                
                if current_metrics:
                    self.metrics_history.append(current_metrics)
                    
                    # Make scaling decision
                    decision = await self._make_scaling_decision(current_metrics)
                    
                    if decision and decision.action != ScalingAction.MAINTAIN:
                        await self._execute_scaling_decision(decision)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error("Error in auto-scaling monitoring loop", error=str(e))
                await asyncio.sleep(60)
    
    async def _get_current_metrics(self) -> Optional[LoadMetrics]:
        """Get current system metrics"""
        try:
            # This would integrate with the actual monitoring system
            # For now, return mock data
            return LoadMetrics(
                cpu_usage=50.0,
                memory_usage=60.0,
                request_rate=100.0,
                response_time=1.5,
                active_connections=50,
                queue_length=5,
                timestamp=datetime.utcnow()
            )
        except Exception as e:
            logger.error("Error getting current metrics", error=str(e))
            return None
    
    async def _make_scaling_decision(self, current_metrics: LoadMetrics) -> Optional[ScalingDecision]:
        """Make scaling decision based on current and predicted metrics"""
        try:
            # Check cooldown
            if self.last_scaling_time:
                time_since_last = (datetime.utcnow() - self.last_scaling_time).total_seconds()
                if time_since_last < self.scaling_cooldown:
                    return None
            
            # Calculate current load score
            current_load = calculate_load_score(current_metrics)
            
            # Predict future load
            historical_metrics = list(self.metrics_history)
            predicted_load = predict_future_load(historical_metrics, self.prediction_horizon)
            
            # Determine trigger
            trigger = self._determine_scaling_trigger(current_metrics, predicted_load)
            
            # Calculate target instances
            target_load = (self.target_cpu_usage + self.target_memory_usage) / 200.0  # Average target
            target_instances = calculate_optimal_instances(
                predicted_load, target_load, self.current_instances,
                self.min_instances, self.max_instances
            )
            
            # Determine action
            action = determine_scaling_action(
                self.current_instances, target_instances, 0.1
            )
            
            if action == ScalingAction.MAINTAIN:
                return None
            
            # Calculate confidence
            confidence = calculate_scaling_confidence(
                self.prediction_accuracy, self.prediction_horizon, current_metrics
            )
            
            # Estimate impact
            impact = estimate_scaling_impact(
                self.current_instances, target_instances, current_metrics
            )
            
            # Create decision
            decision = ScalingDecision(
                action=action,
                trigger=trigger,
                current_instances=self.current_instances,
                target_instances=target_instances,
                reason=self._generate_scaling_reason(action, trigger, current_metrics),
                confidence=confidence,
                estimated_impact=impact,
                timestamp=datetime.utcnow()
            )
            
            return decision
            
        except Exception as e:
            logger.error("Error making scaling decision", error=str(e))
            return None
    
    def _determine_scaling_trigger(
        self,
        current_metrics: LoadMetrics,
        predicted_load: float
    ) -> ScalingTrigger:
        """Determine what triggered the scaling decision"""
        if current_metrics.cpu_usage > self.target_cpu_usage:
            return ScalingTrigger.CPU_THRESHOLD
        elif current_metrics.memory_usage > self.target_memory_usage:
            return ScalingTrigger.MEMORY_THRESHOLD
        elif current_metrics.response_time > self.target_response_time:
            return ScalingTrigger.RESPONSE_TIME
        elif predicted_load > 0.8:
            return ScalingTrigger.PREDICTIVE
        else:
            return ScalingTrigger.REQUEST_RATE
    
    def _generate_scaling_reason(
        self,
        action: ScalingAction,
        trigger: ScalingTrigger,
        metrics: LoadMetrics
    ) -> str:
        """Generate human-readable scaling reason"""
        if action == ScalingAction.SCALE_UP:
            if trigger == ScalingTrigger.CPU_THRESHOLD:
                return f"CPU usage {metrics.cpu_usage:.1f}% exceeds target {self.target_cpu_usage}%"
            elif trigger == ScalingTrigger.MEMORY_THRESHOLD:
                return f"Memory usage {metrics.memory_usage:.1f}% exceeds target {self.target_memory_usage}%"
            elif trigger == ScalingTrigger.RESPONSE_TIME:
                return f"Response time {metrics.response_time:.2f}s exceeds target {self.target_response_time}s"
            else:
                return f"Predicted load increase requires scaling up"
        else:
            return f"System load is low, scaling down to optimize costs"
    
    async def _execute_scaling_decision(self, decision: ScalingDecision) -> None:
        """Execute scaling decision"""
        try:
            logger.info(f"Executing scaling decision: {decision.action.value} to {decision.target_instances} instances")
            
            # Update current instances
            self.current_instances = decision.target_instances
            self.last_scaling_time = datetime.utcnow()
            
            # Update statistics
            self.stats["total_scaling_actions"] += 1
            if decision.action == ScalingAction.SCALE_UP:
                self.stats["scale_up_count"] += 1
            elif decision.action == ScalingAction.SCALE_DOWN:
                self.stats["scale_down_count"] += 1
            else:
                self.stats["maintain_count"] += 1
            
            self.stats["last_scaling_action"] = decision.action.value
            
            # Store decision
            self.scaling_history.append(decision)
            
            # Notify callbacks
            for callback in self.scaling_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(decision)
                    else:
                        callback(decision)
                except Exception as e:
                    logger.error("Error in scaling callback", error=str(e))
            
            logger.info(f"Scaling completed: {decision.action.value} to {decision.target_instances} instances")
            
        except Exception as e:
            logger.error("Error executing scaling decision", error=str(e))
    
    def add_scaling_callback(self, callback: Callable) -> None:
        """Add scaling callback"""
        self.scaling_callbacks.append(callback)
    
    def remove_scaling_callback(self, callback: Callable) -> None:
        """Remove scaling callback"""
        self.scaling_callbacks.remove(callback)
    
    def get_scaling_statistics(self) -> Dict[str, Any]:
        """Get scaling statistics"""
        return {
            "current_instances": self.current_instances,
            "min_instances": self.min_instances,
            "max_instances": self.max_instances,
            "statistics": self.stats.copy(),
            "recent_decisions": [d.to_dict() for d in self.scaling_history[-10:]],
            "prediction_accuracy": self.prediction_accuracy[-20:] if self.prediction_accuracy else [],
            "is_running": self.is_running
        }
    
    def get_load_trends(self) -> Dict[str, Any]:
        """Get load trend analysis"""
        if len(self.metrics_history) < 2:
            return {"trend": "insufficient_data"}
        
        # Calculate trends
        recent_metrics = list(self.metrics_history)[-20:]  # Last 20 measurements
        
        cpu_trend = self._calculate_trend([m.cpu_usage for m in recent_metrics])
        memory_trend = self._calculate_trend([m.memory_usage for m in recent_metrics])
        response_time_trend = self._calculate_trend([m.response_time for m in recent_metrics])
        
        return {
            "cpu_trend": cpu_trend,
            "memory_trend": memory_trend,
            "response_time_trend": response_time_trend,
            "data_points": len(recent_metrics),
            "time_range_minutes": (recent_metrics[-1].timestamp - recent_metrics[0].timestamp).total_seconds() / 60
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction - pure function"""
        if len(values) < 2:
            return "stable"
        
        # Simple linear trend
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        change_percent = ((second_avg - first_avg) / first_avg) * 100
        
        if change_percent > 5:
            return "increasing"
        elif change_percent < -5:
            return "decreasing"
        else:
            return "stable"
    
    async def manual_scale(self, target_instances: int, reason: str = "Manual scaling") -> bool:
        """Manually scale to target instances"""
        try:
            if target_instances < self.min_instances or target_instances > self.max_instances:
                logger.error(f"Target instances {target_instances} outside range [{self.min_instances}, {self.max_instances}]")
                return False
            
            # Create manual scaling decision
            decision = ScalingDecision(
                action=ScalingAction.SCALE_UP if target_instances > self.current_instances else ScalingAction.SCALE_DOWN,
                trigger=ScalingTrigger.MANUAL,
                current_instances=self.current_instances,
                target_instances=target_instances,
                reason=reason,
                confidence=1.0,
                estimated_impact=estimate_scaling_impact(
                    self.current_instances, target_instances, 
                    self.metrics_history[-1] if self.metrics_history else LoadMetrics(0, 0, 0, 0, 0, 0, datetime.utcnow())
                ),
                timestamp=datetime.utcnow()
            )
            
            await self._execute_scaling_decision(decision)
            return True
            
        except Exception as e:
            logger.error("Error in manual scaling", error=str(e))
            return False


# Factory functions

def create_auto_scaling_system(
    min_instances: int = 1,
    max_instances: int = 100,
    target_cpu_usage: float = 70.0,
    target_memory_usage: float = 80.0,
    target_response_time: float = 2.0
) -> AutoScalingSystem:
    """Create auto-scaling system - pure function"""
    return AutoScalingSystem(
        min_instances, max_instances, target_cpu_usage,
        target_memory_usage, target_response_time
    )


async def get_auto_scaling_system() -> AutoScalingSystem:
    """Get auto-scaling system instance"""
    system = create_auto_scaling_system()
    await system.start()
    return system

