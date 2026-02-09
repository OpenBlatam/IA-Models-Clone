"""
Auto Scaler for Flux2 Clothing Changer
=======================================

Automatic scaling based on load and metrics.
"""

import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class ScalingDecision:
    """Scaling decision."""
    action: str  # "scale_up", "scale_down", "no_action"
    reason: str
    target_count: int
    current_count: int
    metrics: Dict[str, float]
    timestamp: float


class AutoScaler:
    """Automatic scaling system."""
    
    def __init__(
        self,
        min_instances: int = 1,
        max_instances: int = 10,
        target_cpu_percent: float = 70.0,
        target_memory_percent: float = 80.0,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.3,
        cooldown_period: float = 300.0,  # 5 minutes
    ):
        """
        Initialize auto scaler.
        
        Args:
            min_instances: Minimum number of instances
            max_instances: Maximum number of instances
            target_cpu_percent: Target CPU usage percentage
            target_memory_percent: Target memory usage percentage
            scale_up_threshold: Threshold for scaling up
            scale_down_threshold: Threshold for scaling down
            cooldown_period: Cooldown period in seconds between scaling actions
        """
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.target_cpu_percent = target_cpu_percent
        self.target_memory_percent = target_memory_percent
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.cooldown_period = cooldown_period
        
        self.current_instances = min_instances
        self.last_scale_time = 0.0
        self.scaling_history: deque = deque(maxlen=100)
        self.metrics_history: deque = deque(maxlen=100)
        
        # Scaling callbacks
        self.scale_up_callback: Optional[Callable[[int], None]] = None
        self.scale_down_callback: Optional[Callable[[int], None]] = None
    
    def register_scale_up_callback(self, callback: Callable[[int], None]) -> None:
        """Register callback for scale up."""
        self.scale_up_callback = callback
    
    def register_scale_down_callback(self, callback: Callable[[int], None]) -> None:
        """Register callback for scale down."""
        self.scale_down_callback = callback
    
    def record_metrics(
        self,
        cpu_percent: float,
        memory_percent: float,
        queue_size: int = 0,
        active_requests: int = 0,
    ) -> None:
        """
        Record metrics for scaling decisions.
        
        Args:
            cpu_percent: CPU usage percentage
            memory_percent: Memory usage percentage
            queue_size: Queue size
            active_requests: Active requests
        """
        self.metrics_history.append({
            "timestamp": time.time(),
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "queue_size": queue_size,
            "active_requests": active_requests,
        })
    
    def evaluate_scaling(self) -> Optional[ScalingDecision]:
        """
        Evaluate if scaling is needed.
        
        Returns:
            Scaling decision or None
        """
        # Check cooldown
        if time.time() - self.last_scale_time < self.cooldown_period:
            return None
        
        if len(self.metrics_history) < 5:
            return None
        
        # Get recent metrics
        recent_metrics = list(self.metrics_history)[-10:]
        
        # Calculate averages
        avg_cpu = sum(m["cpu_percent"] for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m["memory_percent"] for m in recent_metrics) / len(recent_metrics)
        avg_queue = sum(m["queue_size"] for m in recent_metrics) / len(recent_metrics)
        avg_requests = sum(m["active_requests"] for m in recent_metrics) / len(recent_metrics)
        
        metrics = {
            "cpu_percent": avg_cpu,
            "memory_percent": avg_memory,
            "queue_size": avg_queue,
            "active_requests": avg_requests,
        }
        
        # Determine scaling action
        cpu_ratio = avg_cpu / self.target_cpu_percent if self.target_cpu_percent > 0 else 0
        memory_ratio = avg_memory / self.target_memory_percent if self.target_memory_percent > 0 else 0
        max_ratio = max(cpu_ratio, memory_ratio)
        
        # Scale up conditions
        if (
            max_ratio > self.scale_up_threshold and
            self.current_instances < self.max_instances
        ):
            target_count = min(
                self.max_instances,
                int(self.current_instances * max_ratio) + 1
            )
            
            decision = ScalingDecision(
                action="scale_up",
                reason=f"High resource usage: CPU {avg_cpu:.1f}%, Memory {avg_memory:.1f}%",
                target_count=target_count,
                current_count=self.current_instances,
                metrics=metrics,
                timestamp=time.time(),
            )
            
            self._execute_scale_up(target_count)
            return decision
        
        # Scale down conditions
        elif (
            max_ratio < self.scale_down_threshold and
            avg_queue < 5 and
            avg_requests < 2 and
            self.current_instances > self.min_instances
        ):
            target_count = max(
                self.min_instances,
                int(self.current_instances * max_ratio)
            )
            
            decision = ScalingDecision(
                action="scale_down",
                reason=f"Low resource usage: CPU {avg_cpu:.1f}%, Memory {avg_memory:.1f}%",
                target_count=target_count,
                current_count=self.current_instances,
                metrics=metrics,
                timestamp=time.time(),
            )
            
            self._execute_scale_down(target_count)
            return decision
        
        return None
    
    def _execute_scale_up(self, target_count: int) -> None:
        """Execute scale up."""
        old_count = self.current_instances
        self.current_instances = target_count
        self.last_scale_time = time.time()
        
        self.scaling_history.append({
            "action": "scale_up",
            "from": old_count,
            "to": target_count,
            "timestamp": time.time(),
        })
        
        if self.scale_up_callback:
            try:
                self.scale_up_callback(target_count)
            except Exception as e:
                logger.error(f"Scale up callback failed: {e}")
        
        logger.info(f"Scaled up from {old_count} to {target_count} instances")
    
    def _execute_scale_down(self, target_count: int) -> None:
        """Execute scale down."""
        old_count = self.current_instances
        self.current_instances = target_count
        self.last_scale_time = time.time()
        
        self.scaling_history.append({
            "action": "scale_down",
            "from": old_count,
            "to": target_count,
            "timestamp": time.time(),
        })
        
        if self.scale_down_callback:
            try:
                self.scale_down_callback(target_count)
            except Exception as e:
                logger.error(f"Scale down callback failed: {e}")
        
        logger.info(f"Scaled down from {old_count} to {target_count} instances")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get scaling statistics."""
        return {
            "current_instances": self.current_instances,
            "min_instances": self.min_instances,
            "max_instances": self.max_instances,
            "scaling_actions": len(self.scaling_history),
            "recent_scaling": list(self.scaling_history)[-5:] if self.scaling_history else [],
            "cooldown_remaining": max(
                0,
                self.cooldown_period - (time.time() - self.last_scale_time)
            ),
        }


