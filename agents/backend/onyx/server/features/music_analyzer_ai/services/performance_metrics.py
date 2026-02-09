"""
Servicio de métricas de rendimiento del sistema
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import time

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Métricas de rendimiento del sistema"""
    
    def __init__(self):
        self.logger = logger
        self.metrics = {
            "request_count": 0,
            "total_response_time": 0.0,
            "endpoint_metrics": defaultdict(lambda: {"count": 0, "total_time": 0.0, "errors": 0}),
            "error_count": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "start_time": datetime.now()
        }
    
    def record_request(self, endpoint: str, response_time: float, success: bool = True):
        """Registra una petición"""
        self.metrics["request_count"] += 1
        self.metrics["total_response_time"] += response_time
        
        endpoint_metric = self.metrics["endpoint_metrics"][endpoint]
        endpoint_metric["count"] += 1
        endpoint_metric["total_time"] += response_time
        
        if not success:
            self.metrics["error_count"] += 1
            endpoint_metric["errors"] += 1
    
    def record_cache(self, hit: bool):
        """Registra un cache hit/miss"""
        if hit:
            self.metrics["cache_hits"] += 1
        else:
            self.metrics["cache_misses"] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas generales"""
        request_count = self.metrics["request_count"]
        avg_response_time = (
            self.metrics["total_response_time"] / request_count
            if request_count > 0 else 0
        )
        
        uptime = (datetime.now() - self.metrics["start_time"]).total_seconds()
        
        cache_total = self.metrics["cache_hits"] + self.metrics["cache_misses"]
        cache_hit_rate = (
            self.metrics["cache_hits"] / cache_total
            if cache_total > 0 else 0
        )
        
        error_rate = (
            self.metrics["error_count"] / request_count
            if request_count > 0 else 0
        )
        
        return {
            "total_requests": request_count,
            "average_response_time": round(avg_response_time, 3),
            "total_response_time": round(self.metrics["total_response_time"], 2),
            "error_count": self.metrics["error_count"],
            "error_rate": round(error_rate, 3),
            "cache_hits": self.metrics["cache_hits"],
            "cache_misses": self.metrics["cache_misses"],
            "cache_hit_rate": round(cache_hit_rate, 3),
            "uptime_seconds": round(uptime, 2),
            "uptime_hours": round(uptime / 3600, 2),
            "start_time": self.metrics["start_time"].isoformat()
        }
    
    def get_endpoint_metrics(self, limit: int = 10) -> Dict[str, Any]:
        """Obtiene métricas por endpoint"""
        endpoint_stats = []
        
        for endpoint, metric in self.metrics["endpoint_metrics"].items():
            count = metric["count"]
            avg_time = metric["total_time"] / count if count > 0 else 0
            error_rate = metric["errors"] / count if count > 0 else 0
            
            endpoint_stats.append({
                "endpoint": endpoint,
                "request_count": count,
                "average_response_time": round(avg_time, 3),
                "total_time": round(metric["total_time"], 2),
                "error_count": metric["errors"],
                "error_rate": round(error_rate, 3)
            })
        
        # Ordenar por número de requests
        endpoint_stats.sort(key=lambda x: x["request_count"], reverse=True)
        
        return {
            "endpoints": endpoint_stats[:limit],
            "total_endpoints": len(endpoint_stats)
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de rendimiento"""
        metrics = self.get_metrics()
        endpoint_metrics = self.get_endpoint_metrics(5)
        
        # Evaluar rendimiento
        performance_level = "Excellent"
        if metrics["average_response_time"] > 1.0:
            performance_level = "Good"
        if metrics["average_response_time"] > 2.0:
            performance_level = "Fair"
        if metrics["average_response_time"] > 5.0:
            performance_level = "Poor"
        
        if metrics["error_rate"] > 0.1:
            performance_level = "Needs Attention"
        
        return {
            "performance_level": performance_level,
            "metrics": metrics,
            "top_endpoints": endpoint_metrics["endpoints"],
            "recommendations": self._generate_recommendations(metrics)
        }
    
    def _generate_recommendations(self, metrics: Dict) -> List[str]:
        """Genera recomendaciones basadas en métricas"""
        recommendations = []
        
        if metrics["average_response_time"] > 2.0:
            recommendations.append("Consider optimizing slow endpoints")
        
        if metrics["cache_hit_rate"] < 0.3:
            recommendations.append("Consider increasing cache usage")
        
        if metrics["error_rate"] > 0.05:
            recommendations.append("High error rate detected - review error logs")
        
        if metrics["total_requests"] > 10000 and metrics["average_response_time"] > 1.0:
            recommendations.append("Consider implementing request queuing for high load")
        
        return recommendations if recommendations else ["System performance is optimal"]
    
    def reset_metrics(self):
        """Resetea las métricas"""
        self.metrics = {
            "request_count": 0,
            "total_response_time": 0.0,
            "endpoint_metrics": defaultdict(lambda: {"count": 0, "total_time": 0.0, "errors": 0}),
            "error_count": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "start_time": datetime.now()
        }

