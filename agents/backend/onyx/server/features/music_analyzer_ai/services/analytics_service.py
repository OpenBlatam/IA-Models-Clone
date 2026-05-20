"""
Servicio de analytics y métricas
"""

import time
from typing import Dict, List, Any, Optional
from collections import defaultdict
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class AnalyticsService:
    """Servicio para tracking de métricas y analytics"""
    
    def __init__(self):
        self.metrics = {
            "requests": defaultdict(int),
            "endpoints": defaultdict(int),
            "errors": defaultdict(int),
            "response_times": defaultdict(list),
            "tracks_analyzed": set(),
            "users": set()
        }
        self.logger = logger
        self.start_time = datetime.now()
    
    def track_request(self, endpoint: str, method: str = "GET",
                     user_id: Optional[str] = None,
                     response_time: Optional[float] = None,
                     status_code: int = 200):
        """Registra una petición"""
        key = f"{method}:{endpoint}"
        self.metrics["requests"][key] += 1
        self.metrics["endpoints"][endpoint] += 1
        
        if response_time:
            self.metrics["response_times"][endpoint].append(response_time)
            # Mantener solo los últimos 1000 tiempos
            if len(self.metrics["response_times"][endpoint]) > 1000:
                self.metrics["response_times"][endpoint] = \
                    self.metrics["response_times"][endpoint][-1000:]
        
        if status_code >= 400:
            error_key = f"{status_code}:{endpoint}"
            self.metrics["errors"][error_key] += 1
        
        if user_id:
            self.metrics["users"].add(user_id)
    
    def track_analysis(self, track_id: str, user_id: Optional[str] = None):
        """Registra un análisis realizado"""
        self.metrics["tracks_analyzed"].add(track_id)
        if user_id:
            self.metrics["users"].add(user_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas generales"""
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        total_requests = sum(self.metrics["requests"].values())
        total_errors = sum(self.metrics["errors"].values())
        
        # Calcular tiempos promedio de respuesta
        avg_response_times = {}
        for endpoint, times in self.metrics["response_times"].items():
            if times:
                avg_response_times[endpoint] = {
                    "avg": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                    "count": len(times)
                }
        
        # Top endpoints
        top_endpoints = sorted(
            self.metrics["endpoints"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Top errores
        top_errors = sorted(
            self.metrics["errors"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            "uptime_seconds": uptime,
            "uptime_formatted": str(timedelta(seconds=int(uptime))),
            "total_requests": total_requests,
            "total_errors": total_errors,
            "error_rate": (total_errors / total_requests * 100) if total_requests > 0 else 0,
            "unique_tracks_analyzed": len(self.metrics["tracks_analyzed"]),
            "unique_users": len(self.metrics["users"]),
            "top_endpoints": [
                {"endpoint": endpoint, "count": count}
                for endpoint, count in top_endpoints
            ],
            "top_errors": [
                {"error": error, "count": count}
                for error, count in top_errors
            ],
            "average_response_times": avg_response_times,
            "requests_per_minute": (total_requests / (uptime / 60)) if uptime > 0 else 0
        }
    
    def reset_stats(self):
        """Resetea las estadísticas"""
        self.metrics = {
            "requests": defaultdict(int),
            "endpoints": defaultdict(int),
            "errors": defaultdict(int),
            "response_times": defaultdict(list),
            "tracks_analyzed": set(),
            "users": set()
        }
        self.start_time = datetime.now()
        self.logger.info("Analytics stats reset")


# Instancia global
analytics_service = AnalyticsService()

