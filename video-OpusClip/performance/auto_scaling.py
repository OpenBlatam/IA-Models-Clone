#!/usr/bin/env python3
"""
Auto-Scaling and Load Balancing System

Advanced auto-scaling system with:
- Dynamic resource allocation
- Load balancing strategies
- Performance-based scaling
- Predictive scaling
- Resource optimization
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union, Tuple
import asyncio
import time
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import structlog
import statistics
from collections import deque, defaultdict

logger = structlog.get_logger("auto_scaling")

# =============================================================================
# AUTO-SCALING MODELS
# =============================================================================

class ScalingStrategy(Enum):
    """Auto-scaling strategies."""
    CPU_BASED = "cpu_based"
    MEMORY_BASED = "memory_based"
    REQUEST_BASED = "request_based"
    RESPONSE_TIME_BASED = "response_time_based"
    HYBRID = "hybrid"
    PREDICTIVE = "predictive"

class ScalingAction(Enum):
    """Scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_ACTION = "no_action"
    EMERGENCY_SCALE_UP = "emergency_scale_up"

@dataclass
class ResourceMetrics:
    """Resource metrics for scaling decisions."""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: float
    request_rate: float
    response_time: float
    error_rate: float
    active_connections: int
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "disk_usage": self.disk_usage,
            "network_io": self.network_io,
            "request_rate": self.request_rate,
            "response_time": self.response_time,
            "error_rate": self.error_rate,
            "active_connections": self.active_connections,
            "timestamp": self.timestamp.isoformat()
        }

@dataclass
class ScalingDecision:
    """Scaling decision data."""
    action: ScalingAction
    reason: str
    current_metrics: ResourceMetrics
    target_instances: int
    confidence: float
    estimated_impact: Dict[str, Any]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action": self.action.value,
            "reason": self.reason,
            "current_metrics": self.current_metrics.to_dict(),
            "target_instances": self.target_instances,
            "confidence": self.confidence,
            "estimated_impact": self.estimated_impact,
            "timestamp": self.timestamp.isoformat()
        }

@dataclass
class ScalingConfig:
    """Auto-scaling configuration."""
    min_instances: int
    max_instances: int
    target_cpu_usage: float
    target_memory_usage: float
    target_response_time: float
    scale_up_threshold: float
    scale_down_threshold: float
    cooldown_period: int
    strategy: ScalingStrategy
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "min_instances": self.min_instances,
            "max_instances": self.max_instances,
            "target_cpu_usage": self.target_cpu_usage,
            "target_memory_usage": self.target_memory_usage,
            "target_response_time": self.target_response_time,
            "scale_up_threshold": self.scale_up_threshold,
            "scale_down_threshold": self.scale_down_threshold,
            "cooldown_period": self.cooldown_period,
            "strategy": self.strategy.value
        }

# =============================================================================
# AUTO-SCALING ENGINE
# =============================================================================

class AutoScalingEngine:
    """Advanced auto-scaling engine."""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.current_instances = config.min_instances
        self.last_scaling_action = None
        self.last_scaling_time = 0
        self.metrics_history: deque = deque(maxlen=100)
        self.scaling_history: deque = deque(maxlen=50)
        
        # Performance tracking
        self.performance_tracking = {
            'scaling_events': 0,
            'successful_scales': 0,
            'failed_scales': 0,
            'average_scale_time': 0.0
        }
        
        # Predictive scaling
        self.predictive_model = None
        self.prediction_accuracy = 0.0
    
    async def evaluate_scaling_need(self, metrics: ResourceMetrics) -> ScalingDecision:
        """Evaluate if scaling is needed based on current metrics."""
        try:
            # Add metrics to history
            self.metrics_history.append(metrics)
            
            # Check cooldown period
            if self._is_in_cooldown():
                return ScalingDecision(
                    action=ScalingAction.NO_ACTION,
                    reason="In cooldown period",
                    current_metrics=metrics,
                    target_instances=self.current_instances,
                    confidence=1.0,
                    estimated_impact={},
                    timestamp=datetime.utcnow()
                )
            
            # Evaluate based on strategy
            if self.config.strategy == ScalingStrategy.CPU_BASED:
                decision = await self._evaluate_cpu_based_scaling(metrics)
            elif self.config.strategy == ScalingStrategy.MEMORY_BASED:
                decision = await self._evaluate_memory_based_scaling(metrics)
            elif self.config.strategy == ScalingStrategy.REQUEST_BASED:
                decision = await self._evaluate_request_based_scaling(metrics)
            elif self.config.strategy == ScalingStrategy.RESPONSE_TIME_BASED:
                decision = await self._evaluate_response_time_based_scaling(metrics)
            elif self.config.strategy == ScalingStrategy.HYBRID:
                decision = await self._evaluate_hybrid_scaling(metrics)
            elif self.config.strategy == ScalingStrategy.PREDICTIVE:
                decision = await self._evaluate_predictive_scaling(metrics)
            else:
                decision = await self._evaluate_hybrid_scaling(metrics)
            
            # Log scaling decision
            logger.info(
                "Scaling decision made",
                action=decision.action.value,
                reason=decision.reason,
                current_instances=self.current_instances,
                target_instances=decision.target_instances,
                confidence=decision.confidence
            )
            
            return decision
            
        except Exception as e:
            logger.error("Failed to evaluate scaling need", error=str(e))
            return ScalingDecision(
                action=ScalingAction.NO_ACTION,
                reason=f"Error in evaluation: {str(e)}",
                current_metrics=metrics,
                target_instances=self.current_instances,
                confidence=0.0,
                estimated_impact={},
                timestamp=datetime.utcnow()
            )
    
    async def execute_scaling_action(self, decision: ScalingDecision) -> bool:
        """Execute the scaling action."""
        try:
            if decision.action == ScalingAction.NO_ACTION:
                return True
            
            # Calculate target instances
            if decision.action == ScalingAction.SCALE_UP:
                target_instances = min(
                    self.current_instances + 1,
                    self.config.max_instances
                )
            elif decision.action == ScalingAction.SCALE_DOWN:
                target_instances = max(
                    self.current_instances - 1,
                    self.config.min_instances
                )
            elif decision.action == ScalingAction.EMERGENCY_SCALE_UP:
                target_instances = min(
                    self.current_instances + 2,
                    self.config.max_instances
                )
            else:
                target_instances = self.current_instances
            
            # Execute scaling
            success = await self._execute_scaling(target_instances)
            
            if success:
                self.current_instances = target_instances
                self.last_scaling_action = decision.action
                self.last_scaling_time = time.time()
                
                # Update performance tracking
                self.performance_tracking['scaling_events'] += 1
                self.performance_tracking['successful_scales'] += 1
                
                # Add to scaling history
                self.scaling_history.append({
                    'action': decision.action.value,
                    'from_instances': self.current_instances - (1 if decision.action == ScalingAction.SCALE_UP else -1),
                    'to_instances': target_instances,
                    'timestamp': datetime.utcnow().isoformat(),
                    'reason': decision.reason
                })
                
                logger.info(
                    "Scaling action executed successfully",
                    action=decision.action.value,
                    from_instances=self.current_instances - (1 if decision.action == ScalingAction.SCALE_UP else -1),
                    to_instances=target_instances
                )
            else:
                self.performance_tracking['failed_scales'] += 1
                logger.error("Scaling action failed", action=decision.action.value)
            
            return success
            
        except Exception as e:
            logger.error("Failed to execute scaling action", error=str(e))
            self.performance_tracking['failed_scales'] += 1
            return False
    
    def _is_in_cooldown(self) -> bool:
        """Check if we're in cooldown period."""
        if self.last_scaling_time == 0:
            return False
        
        elapsed_time = time.time() - self.last_scaling_time
        return elapsed_time < self.config.cooldown_period
    
    async def _evaluate_cpu_based_scaling(self, metrics: ResourceMetrics) -> ScalingDecision:
        """Evaluate CPU-based scaling."""
        cpu_usage = metrics.cpu_usage
        
        if cpu_usage > self.config.scale_up_threshold:
            return ScalingDecision(
                action=ScalingAction.SCALE_UP,
                reason=f"CPU usage {cpu_usage:.2f}% exceeds threshold {self.config.scale_up_threshold}%",
                current_metrics=metrics,
                target_instances=self.current_instances + 1,
                confidence=min(1.0, (cpu_usage - self.config.scale_up_threshold) / 20),
                estimated_impact={'cpu_reduction': cpu_usage * 0.3},
                timestamp=datetime.utcnow()
            )
        elif cpu_usage < self.config.scale_down_threshold and self.current_instances > self.config.min_instances:
            return ScalingDecision(
                action=ScalingAction.SCALE_DOWN,
                reason=f"CPU usage {cpu_usage:.2f}% below threshold {self.config.scale_down_threshold}%",
                current_metrics=metrics,
                target_instances=self.current_instances - 1,
                confidence=min(1.0, (self.config.scale_down_threshold - cpu_usage) / 20),
                estimated_impact={'cpu_increase': cpu_usage * 0.2},
                timestamp=datetime.utcnow()
            )
        else:
            return ScalingDecision(
                action=ScalingAction.NO_ACTION,
                reason="CPU usage within normal range",
                current_metrics=metrics,
                target_instances=self.current_instances,
                confidence=1.0,
                estimated_impact={},
                timestamp=datetime.utcnow()
            )
    
    async def _evaluate_memory_based_scaling(self, metrics: ResourceMetrics) -> ScalingDecision:
        """Evaluate memory-based scaling."""
        memory_usage = metrics.memory_usage
        
        if memory_usage > self.config.scale_up_threshold:
            return ScalingDecision(
                action=ScalingAction.SCALE_UP,
                reason=f"Memory usage {memory_usage:.2f}% exceeds threshold {self.config.scale_up_threshold}%",
                current_metrics=metrics,
                target_instances=self.current_instances + 1,
                confidence=min(1.0, (memory_usage - self.config.scale_up_threshold) / 20),
                estimated_impact={'memory_reduction': memory_usage * 0.3},
                timestamp=datetime.utcnow()
            )
        elif memory_usage < self.config.scale_down_threshold and self.current_instances > self.config.min_instances:
            return ScalingDecision(
                action=ScalingAction.SCALE_DOWN,
                reason=f"Memory usage {memory_usage:.2f}% below threshold {self.config.scale_down_threshold}%",
                current_metrics=metrics,
                target_instances=self.current_instances - 1,
                confidence=min(1.0, (self.config.scale_down_threshold - memory_usage) / 20),
                estimated_impact={'memory_increase': memory_usage * 0.2},
                timestamp=datetime.utcnow()
            )
        else:
            return ScalingDecision(
                action=ScalingAction.NO_ACTION,
                reason="Memory usage within normal range",
                current_metrics=metrics,
                target_instances=self.current_instances,
                confidence=1.0,
                estimated_impact={},
                timestamp=datetime.utcnow()
            )
    
    async def _evaluate_request_based_scaling(self, metrics: ResourceMetrics) -> ScalingDecision:
        """Evaluate request-based scaling."""
        request_rate = metrics.request_rate
        current_capacity = self.current_instances * 100  # Assume 100 requests per instance
        
        if request_rate > current_capacity * 0.8:  # 80% capacity threshold
            return ScalingDecision(
                action=ScalingAction.SCALE_UP,
                reason=f"Request rate {request_rate:.2f} exceeds 80% of current capacity {current_capacity}",
                current_metrics=metrics,
                target_instances=self.current_instances + 1,
                confidence=min(1.0, (request_rate - current_capacity * 0.8) / (current_capacity * 0.2)),
                estimated_impact={'capacity_increase': 100},
                timestamp=datetime.utcnow()
            )
        elif request_rate < current_capacity * 0.3 and self.current_instances > self.config.min_instances:
            return ScalingDecision(
                action=ScalingAction.SCALE_DOWN,
                reason=f"Request rate {request_rate:.2f} below 30% of current capacity {current_capacity}",
                current_metrics=metrics,
                target_instances=self.current_instances - 1,
                confidence=min(1.0, (current_capacity * 0.3 - request_rate) / (current_capacity * 0.3)),
                estimated_impact={'capacity_decrease': 100},
                timestamp=datetime.utcnow()
            )
        else:
            return ScalingDecision(
                action=ScalingAction.NO_ACTION,
                reason="Request rate within normal range",
                current_metrics=metrics,
                target_instances=self.current_instances,
                confidence=1.0,
                estimated_impact={},
                timestamp=datetime.utcnow()
            )
    
    async def _evaluate_response_time_based_scaling(self, metrics: ResourceMetrics) -> ScalingDecision:
        """Evaluate response time-based scaling."""
        response_time = metrics.response_time
        
        if response_time > self.config.target_response_time * 2:  # 2x target response time
            return ScalingDecision(
                action=ScalingAction.EMERGENCY_SCALE_UP,
                reason=f"Response time {response_time:.2f}ms exceeds 2x target {self.config.target_response_time}ms",
                current_metrics=metrics,
                target_instances=self.current_instances + 2,
                confidence=min(1.0, (response_time - self.config.target_response_time * 2) / self.config.target_response_time),
                estimated_impact={'response_time_reduction': response_time * 0.4},
                timestamp=datetime.utcnow()
            )
        elif response_time > self.config.target_response_time * 1.5:  # 1.5x target response time
            return ScalingDecision(
                action=ScalingAction.SCALE_UP,
                reason=f"Response time {response_time:.2f}ms exceeds 1.5x target {self.config.target_response_time}ms",
                current_metrics=metrics,
                target_instances=self.current_instances + 1,
                confidence=min(1.0, (response_time - self.config.target_response_time * 1.5) / (self.config.target_response_time * 0.5)),
                estimated_impact={'response_time_reduction': response_time * 0.3},
                timestamp=datetime.utcnow()
            )
        elif response_time < self.config.target_response_time * 0.5 and self.current_instances > self.config.min_instances:
            return ScalingDecision(
                action=ScalingAction.SCALE_DOWN,
                reason=f"Response time {response_time:.2f}ms below 0.5x target {self.config.target_response_time}ms",
                current_metrics=metrics,
                target_instances=self.current_instances - 1,
                confidence=min(1.0, (self.config.target_response_time * 0.5 - response_time) / (self.config.target_response_time * 0.5)),
                estimated_impact={'response_time_increase': response_time * 0.2},
                timestamp=datetime.utcnow()
            )
        else:
            return ScalingDecision(
                action=ScalingAction.NO_ACTION,
                reason="Response time within normal range",
                current_metrics=metrics,
                target_instances=self.current_instances,
                confidence=1.0,
                estimated_impact={},
                timestamp=datetime.utcnow()
            )
    
    async def _evaluate_hybrid_scaling(self, metrics: ResourceMetrics) -> ScalingDecision:
        """Evaluate hybrid scaling based on multiple metrics."""
        # Calculate weighted scores
        cpu_score = metrics.cpu_usage / 100.0
        memory_score = metrics.memory_usage / 100.0
        response_time_score = min(1.0, metrics.response_time / (self.config.target_response_time * 2))
        request_rate_score = min(1.0, metrics.request_rate / 1000.0)  # Assume 1000 requests/second max
        
        # Weighted average
        overall_score = (
            cpu_score * 0.3 +
            memory_score * 0.3 +
            response_time_score * 0.25 +
            request_rate_score * 0.15
        )
        
        if overall_score > 0.8:
            return ScalingDecision(
                action=ScalingAction.SCALE_UP,
                reason=f"Overall system load {overall_score:.2f} exceeds 80% threshold",
                current_metrics=metrics,
                target_instances=self.current_instances + 1,
                confidence=overall_score,
                estimated_impact={'load_reduction': overall_score * 0.3},
                timestamp=datetime.utcnow()
            )
        elif overall_score < 0.3 and self.current_instances > self.config.min_instances:
            return ScalingDecision(
                action=ScalingAction.SCALE_DOWN,
                reason=f"Overall system load {overall_score:.2f} below 30% threshold",
                current_metrics=metrics,
                target_instances=self.current_instances - 1,
                confidence=1.0 - overall_score,
                estimated_impact={'load_increase': overall_score * 0.2},
                timestamp=datetime.utcnow()
            )
        else:
            return ScalingDecision(
                action=ScalingAction.NO_ACTION,
                reason="Overall system load within normal range",
                current_metrics=metrics,
                target_instances=self.current_instances,
                confidence=1.0 - abs(overall_score - 0.5) * 2,
                estimated_impact={},
                timestamp=datetime.utcnow()
            )
    
    async def _evaluate_predictive_scaling(self, metrics: ResourceMetrics) -> ScalingDecision:
        """Evaluate predictive scaling based on historical patterns."""
        if len(self.metrics_history) < 10:
            # Not enough data for prediction, fall back to hybrid
            return await self._evaluate_hybrid_scaling(metrics)
        
        # Simple trend analysis
        recent_metrics = list(self.metrics_history)[-10:]
        cpu_trend = self._calculate_trend([m.cpu_usage for m in recent_metrics])
        memory_trend = self._calculate_trend([m.memory_usage for m in recent_metrics])
        response_time_trend = self._calculate_trend([m.response_time for m in recent_metrics])
        
        # Predict future values
        predicted_cpu = metrics.cpu_usage + cpu_trend * 5  # 5 minutes ahead
        predicted_memory = metrics.memory_usage + memory_trend * 5
        predicted_response_time = metrics.response_time + response_time_trend * 5
        
        # Make scaling decision based on predictions
        if predicted_cpu > self.config.scale_up_threshold or predicted_response_time > self.config.target_response_time * 1.5:
            return ScalingDecision(
                action=ScalingAction.SCALE_UP,
                reason=f"Predicted CPU {predicted_cpu:.2f}% or response time {predicted_response_time:.2f}ms will exceed thresholds",
                current_metrics=metrics,
                target_instances=self.current_instances + 1,
                confidence=min(1.0, abs(cpu_trend) + abs(response_time_trend) / 100),
                estimated_impact={'predicted_improvement': 0.3},
                timestamp=datetime.utcnow()
            )
        elif predicted_cpu < self.config.scale_down_threshold and predicted_memory < self.config.scale_down_threshold and self.current_instances > self.config.min_instances:
            return ScalingDecision(
                action=ScalingAction.SCALE_DOWN,
                reason=f"Predicted CPU {predicted_cpu:.2f}% and memory {predicted_memory:.2f}% will be below thresholds",
                current_metrics=metrics,
                target_instances=self.current_instances - 1,
                confidence=min(1.0, abs(cpu_trend) + abs(memory_trend)),
                estimated_impact={'predicted_efficiency': 0.2},
                timestamp=datetime.utcnow()
            )
        else:
            return ScalingDecision(
                action=ScalingAction.NO_ACTION,
                reason="Predicted metrics within normal range",
                current_metrics=metrics,
                target_instances=self.current_instances,
                confidence=0.8,
                estimated_impact={},
                timestamp=datetime.utcnow()
            )
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend slope for a series of values."""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x = list(range(n))
        y = values
        
        # Simple linear regression
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        return slope
    
    async def _execute_scaling(self, target_instances: int) -> bool:
        """Execute the actual scaling operation."""
        try:
            # This would integrate with your container orchestration platform
            # For now, we'll simulate the scaling operation
            
            logger.info(
                "Executing scaling operation",
                from_instances=self.current_instances,
                to_instances=target_instances
            )
            
            # Simulate scaling delay
            await asyncio.sleep(2)
            
            # In a real implementation, this would:
            # 1. Call Kubernetes API to scale deployment
            # 2. Call cloud provider API to scale instances
            # 3. Update load balancer configuration
            # 4. Wait for new instances to be ready
            
            return True
            
        except Exception as e:
            logger.error("Failed to execute scaling", error=str(e))
            return False
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status and statistics."""
        return {
            'current_instances': self.current_instances,
            'config': self.config.to_dict(),
            'last_scaling_action': self.last_scaling_action.value if self.last_scaling_action else None,
            'last_scaling_time': self.last_scaling_time,
            'in_cooldown': self._is_in_cooldown(),
            'performance_tracking': self.performance_tracking,
            'recent_scaling_history': list(self.scaling_history)[-10:],
            'metrics_history_size': len(self.metrics_history)
        }

# =============================================================================
# LOAD BALANCER
# =============================================================================

class LoadBalancer:
    """Advanced load balancer with multiple strategies."""
    
    def __init__(self, strategy: str = "round_robin"):
        self.strategy = strategy
        self.instances = []
        self.current_index = 0
        self.instance_weights = {}
        self.instance_health = {}
        self.request_counts = defaultdict(int)
        self.response_times = defaultdict(list)
    
    def add_instance(self, instance_id: str, weight: int = 1) -> None:
        """Add an instance to the load balancer."""
        self.instances.append(instance_id)
        self.instance_weights[instance_id] = weight
        self.instance_health[instance_id] = True
        self.request_counts[instance_id] = 0
        self.response_times[instance_id] = []
    
    def remove_instance(self, instance_id: str) -> None:
        """Remove an instance from the load balancer."""
        if instance_id in self.instances:
            self.instances.remove(instance_id)
            self.instance_weights.pop(instance_id, None)
            self.instance_health.pop(instance_id, None)
            self.request_counts.pop(instance_id, None)
            self.response_times.pop(instance_id, None)
    
    def get_next_instance(self) -> Optional[str]:
        """Get the next instance based on the load balancing strategy."""
        if not self.instances:
            return None
        
        # Filter healthy instances
        healthy_instances = [inst for inst in self.instances if self.instance_health.get(inst, True)]
        
        if not healthy_instances:
            return None
        
        if self.strategy == "round_robin":
            return self._round_robin(healthy_instances)
        elif self.strategy == "weighted_round_robin":
            return self._weighted_round_robin(healthy_instances)
        elif self.strategy == "least_connections":
            return self._least_connections(healthy_instances)
        elif self.strategy == "least_response_time":
            return self._least_response_time(healthy_instances)
        else:
            return self._round_robin(healthy_instances)
    
    def _round_robin(self, instances: List[str]) -> str:
        """Round robin load balancing."""
        instance = instances[self.current_index % len(instances)]
        self.current_index += 1
        return instance
    
    def _weighted_round_robin(self, instances: List[str]) -> str:
        """Weighted round robin load balancing."""
        # Simple weighted round robin implementation
        total_weight = sum(self.instance_weights.get(inst, 1) for inst in instances)
        if total_weight == 0:
            return instances[0]
        
        # Find instance with highest weight that hasn't been used recently
        best_instance = instances[0]
        best_score = float('inf')
        
        for instance in instances:
            weight = self.instance_weights.get(instance, 1)
            request_count = self.request_counts.get(instance, 0)
            score = request_count / weight
            
            if score < best_score:
                best_score = score
                best_instance = instance
        
        return best_instance
    
    def _least_connections(self, instances: List[str]) -> str:
        """Least connections load balancing."""
        return min(instances, key=lambda inst: self.request_counts.get(inst, 0))
    
    def _least_response_time(self, instances: List[str]) -> str:
        """Least response time load balancing."""
        def get_avg_response_time(instance):
            times = self.response_times.get(instance, [])
            return statistics.mean(times) if times else 0
        
        return min(instances, key=get_avg_response_time)
    
    def record_request(self, instance_id: str, response_time: float) -> None:
        """Record a request for an instance."""
        self.request_counts[instance_id] += 1
        self.response_times[instance_id].append(response_time)
        
        # Keep only recent response times (last 100)
        if len(self.response_times[instance_id]) > 100:
            self.response_times[instance_id] = self.response_times[instance_id][-100:]
    
    def set_instance_health(self, instance_id: str, is_healthy: bool) -> None:
        """Set the health status of an instance."""
        self.instance_health[instance_id] = is_healthy
    
    def get_load_balancer_status(self) -> Dict[str, Any]:
        """Get load balancer status and statistics."""
        return {
            'strategy': self.strategy,
            'instances': self.instances,
            'instance_weights': self.instance_weights,
            'instance_health': self.instance_health,
            'request_counts': dict(self.request_counts),
            'average_response_times': {
                inst: statistics.mean(times) if times else 0
                for inst, times in self.response_times.items()
            },
            'healthy_instances': len([inst for inst in self.instances if self.instance_health.get(inst, True)])
        }

# =============================================================================
# GLOBAL AUTO-SCALING INSTANCES
# =============================================================================

# Default scaling configuration
default_scaling_config = ScalingConfig(
    min_instances=2,
    max_instances=10,
    target_cpu_usage=70.0,
    target_memory_usage=80.0,
    target_response_time=500.0,
    scale_up_threshold=80.0,
    scale_down_threshold=30.0,
    cooldown_period=300,  # 5 minutes
    strategy=ScalingStrategy.HYBRID
)

# Global instances
auto_scaling_engine = AutoScalingEngine(default_scaling_config)
load_balancer = LoadBalancer("weighted_round_robin")

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'ScalingStrategy',
    'ScalingAction',
    'ResourceMetrics',
    'ScalingDecision',
    'ScalingConfig',
    'AutoScalingEngine',
    'LoadBalancer',
    'auto_scaling_engine',
    'load_balancer'
]





























