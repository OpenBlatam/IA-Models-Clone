"""
Cache health monitoring and diagnostics.

Provides health checks and diagnostic capabilities.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, Any, List, Optional
from enum import Enum

from kv_cache.types import StatsDict

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Cache health status."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class CacheHealthMonitor:
    """
    Cache health monitor.
    
    Monitors cache health and provides diagnostics.
    """
    
    def __init__(
        self,
        cache: Any,
        check_interval: int = 100,
        warning_thresholds: Optional[Dict[str, float]] = None,
        critical_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize cache health monitor.
        
        Args:
            cache: Cache instance
            check_interval: Operations between health checks
            warning_thresholds: Thresholds for warnings
            critical_thresholds: Thresholds for critical status
        """
        self.cache = cache
        self.check_interval = check_interval
        self.operation_count = 0
        
        self.warning_thresholds = warning_thresholds or {
            "hit_rate": 0.6,
            "memory_mb": 1000.0,
            "eviction_rate": 0.3
        }
        
        self.critical_thresholds = critical_thresholds or {
            "hit_rate": 0.3,
            "memory_mb": 2000.0,
            "eviction_rate": 0.7
        }
        
        self.health_history: List[Dict[str, Any]] = []
    
    def check_health(self) -> Dict[str, Any]:
        """
        Check cache health.
        
        Returns:
            Dictionary with health status and diagnostics
        """
        stats = self.cache.get_stats()
        
        health = {
            "timestamp": time.time(),
            "status": HealthStatus.UNKNOWN,
            "metrics": {},
            "issues": [],
            "recommendations": []
        }
        
        # Check hit rate
        hit_rate = stats.get("hit_rate", 0.0)
        health["metrics"]["hit_rate"] = hit_rate
        
        if hit_rate < self.critical_thresholds["hit_rate"]:
            health["status"] = HealthStatus.CRITICAL
            health["issues"].append({
                "type": "low_hit_rate",
                "severity": "critical",
                "value": hit_rate,
                "threshold": self.critical_thresholds["hit_rate"],
                "message": f"Hit rate critically low: {hit_rate:.2%}"
            })
        elif hit_rate < self.warning_thresholds["hit_rate"]:
            if health["status"] == HealthStatus.UNKNOWN:
                health["status"] = HealthStatus.WARNING
            health["issues"].append({
                "type": "low_hit_rate",
                "severity": "warning",
                "value": hit_rate,
                "threshold": self.warning_thresholds["hit_rate"],
                "message": f"Hit rate below threshold: {hit_rate:.2%}"
            })
        
        # Check memory
        memory_mb = stats.get("storage_memory_mb", 0.0)
        health["metrics"]["memory_mb"] = memory_mb
        
        if memory_mb > self.critical_thresholds["memory_mb"]:
            health["status"] = HealthStatus.CRITICAL
            health["issues"].append({
                "type": "high_memory",
                "severity": "critical",
                "value": memory_mb,
                "threshold": self.critical_thresholds["memory_mb"],
                "message": f"Memory usage critically high: {memory_mb:.2f} MB"
            })
        elif memory_mb > self.warning_thresholds["memory_mb"]:
            if health["status"] == HealthStatus.UNKNOWN:
                health["status"] = HealthStatus.WARNING
            health["issues"].append({
                "type": "high_memory",
                "severity": "warning",
                "value": memory_mb,
                "threshold": self.warning_thresholds["memory_mb"],
                "message": f"Memory usage high: {memory_mb:.2f} MB"
            })
        
        # Check eviction rate
        num_entries = stats.get("num_entries", 0)
        evictions = stats.get("evictions", 0)
        eviction_rate = evictions / max(num_entries, 1)
        health["metrics"]["eviction_rate"] = eviction_rate
        
        if eviction_rate > self.critical_thresholds["eviction_rate"]:
            health["status"] = HealthStatus.CRITICAL
            health["issues"].append({
                "type": "high_eviction",
                "severity": "critical",
                "value": eviction_rate,
                "threshold": self.critical_thresholds["eviction_rate"],
                "message": f"Eviction rate critically high: {eviction_rate:.2%}"
            })
        elif eviction_rate > self.warning_thresholds["eviction_rate"]:
            if health["status"] == HealthStatus.UNKNOWN:
                health["status"] = HealthStatus.WARNING
            health["issues"].append({
                "type": "high_eviction",
                "severity": "warning",
                "value": eviction_rate,
                "threshold": self.warning_thresholds["eviction_rate"],
                "message": f"Eviction rate high: {eviction_rate:.2%}"
            })
        
        # Set healthy if no issues
        if health["status"] == HealthStatus.UNKNOWN:
            health["status"] = HealthStatus.HEALTHY
        
        # Generate recommendations
        if health["status"] != HealthStatus.HEALTHY:
            health["recommendations"] = self._generate_recommendations(health)
        
        self.health_history.append(health)
        
        # Keep only recent history
        if len(self.health_history) > 100:
            self.health_history = self.health_history[-100:]
        
        return health
    
    def _generate_recommendations(self, health: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on health status."""
        recommendations = []
        
        for issue in health["issues"]:
            if issue["type"] == "low_hit_rate":
                recommendations.append("Increase cache size or improve eviction strategy")
                recommendations.append("Enable cache warmup")
            elif issue["type"] == "high_memory":
                recommendations.append("Enable compression or quantization")
                recommendations.append("Reduce cache size")
            elif issue["type"] == "high_eviction":
                recommendations.append("Increase max_tokens")
                recommendations.append("Use adaptive eviction strategy")
        
        return list(set(recommendations))  # Remove duplicates
    
    def auto_check(self) -> Optional[Dict[str, Any]]:
        """
        Auto-check health if interval reached.
        
        Returns:
            Health status if check ran, None otherwise
        """
        self.operation_count += 1
        if self.operation_count % self.check_interval == 0:
            return self.check_health()
        return None
    
    def get_health_summary(self) -> Dict[str, Any]:
        """
        Get health summary.
        
        Returns:
            Dictionary with health summary
        """
        if not self.health_history:
            return {"status": HealthStatus.UNKNOWN, "message": "No health data"}
        
        recent = self.health_history[-10:]
        
        status_counts = {}
        for health in recent:
            status = health["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
        
        most_common_status = max(status_counts.items(), key=lambda x: x[1])[0]
        
        return {
            "current_status": recent[-1]["status"].value if recent else "unknown",
            "most_common_status": most_common_status.value,
            "total_checks": len(self.health_history),
            "recent_issues": sum(len(h.get("issues", [])) for h in recent),
            "last_check": recent[-1]["timestamp"] if recent else None
        }

