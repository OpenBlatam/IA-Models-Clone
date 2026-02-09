"""
Cache auto-scaling.

Provides auto-scaling capabilities for cache.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ScalingAction(Enum):
    """Scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_ACTION = "no_action"


@dataclass
class ScalingDecision:
    """Scaling decision."""
    action: ScalingAction
    reason: str
    current_metrics: Dict[str, Any]
    target_metrics: Dict[str, Any]


class CacheAutoScaler:
    """
    Cache auto-scaler.
    
    Provides auto-scaling capabilities.
    """
    
    def __init__(
        self,
        cache: Any,
        min_size: int = 100,
        max_size: int = 10000,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.3
    ):
        """
        Initialize auto-scaler.
        
        Args:
            cache: Cache instance
            min_size: Minimum cache size
            max_size: Maximum cache size
            scale_up_threshold: Threshold for scaling up (hit rate)
            scale_down_threshold: Threshold for scaling down (hit rate)
        """
        self.cache = cache
        self.min_size = min_size
        self.max_size = max_size
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.scaling_history: list[Dict[str, Any]] = []
    
    def should_scale(self) -> ScalingDecision:
        """
        Determine if scaling is needed.
        
        Returns:
            Scaling decision
        """
        stats = self.cache.get_stats()
        hit_rate = stats.get("hit_rate", 0.0)
        current_size = self.cache.config.max_tokens
        
        current_metrics = {
            "hit_rate": hit_rate,
            "cache_size": current_size,
            "memory_mb": stats.get("memory_mb", 0.0)
        }
        
        # Scale up if hit rate is low and not at max
        if hit_rate < self.scale_up_threshold and current_size < self.max_size:
            new_size = min(int(current_size * 1.5), self.max_size)
            
            return ScalingDecision(
                action=ScalingAction.SCALE_UP,
                reason=f"Hit rate {hit_rate:.2%} below threshold {self.scale_up_threshold:.2%}",
                current_metrics=current_metrics,
                target_metrics={"cache_size": new_size}
            )
        
        # Scale down if hit rate is high and not at min
        if hit_rate > (1.0 - self.scale_down_threshold) and current_size > self.min_size:
            new_size = max(int(current_size * 0.8), self.min_size)
            
            return ScalingDecision(
                action=ScalingAction.SCALE_DOWN,
                reason=f"Hit rate {hit_rate:.2%} above threshold {(1.0 - self.scale_down_threshold):.2%}",
                current_metrics=current_metrics,
                target_metrics={"cache_size": new_size}
            )
        
        return ScalingDecision(
            action=ScalingAction.NO_ACTION,
            reason="Metrics within acceptable range",
            current_metrics=current_metrics,
            target_metrics={}
        )
    
    def scale(self, decision: ScalingDecision) -> bool:
        """
        Execute scaling decision.
        
        Args:
            decision: Scaling decision
            
        Returns:
            True if scaled
        """
        if decision.action == ScalingAction.NO_ACTION:
            return False
        
        try:
            if decision.action == ScalingAction.SCALE_UP:
                new_size = decision.target_metrics.get("cache_size", self.cache.config.max_tokens)
                old_size = self.cache.config.max_tokens
                self.cache.config.max_tokens = new_size
                
                logger.info(f"Scaled up from {old_size} to {new_size}")
                
                self.scaling_history.append({
                    "timestamp": time.time(),
                    "action": "scale_up",
                    "from_size": old_size,
                    "to_size": new_size,
                    "reason": decision.reason
                })
                
                return True
            
            elif decision.action == ScalingAction.SCALE_DOWN:
                new_size = decision.target_metrics.get("cache_size", self.cache.config.max_tokens)
                old_size = self.cache.config.max_tokens
                self.cache.config.max_tokens = new_size
                
                logger.info(f"Scaled down from {old_size} to {new_size}")
                
                self.scaling_history.append({
                    "timestamp": time.time(),
                    "action": "scale_down",
                    "from_size": old_size,
                    "to_size": new_size,
                    "reason": decision.reason
                })
                
                return True
        
        except Exception as e:
            logger.error(f"Scaling failed: {e}")
            return False
    
    def auto_scale(self, interval: float = 60.0) -> None:
        """
        Start auto-scaling.
        
        Args:
            interval: Check interval in seconds
        """
        import threading
        
        def scale_loop():
            while True:
                time.sleep(interval)
                
                decision = self.should_scale()
                
                if decision.action != ScalingAction.NO_ACTION:
                    self.scale(decision)
        
        thread = threading.Thread(target=scale_loop, daemon=True)
        thread.start()
        
        logger.info("Auto-scaling started")
    
    def get_scaling_history(self) -> list[Dict[str, Any]]:
        """
        Get scaling history.
        
        Returns:
            Scaling history
        """
        return self.scaling_history.copy()

