"""
Metrics Collector - Recolector de Métricas
===========================================

Recolecta métricas para monitoreo y análisis.
"""

import logging
import time
from typing import Dict, Any, List
from datetime import datetime
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Recolector de métricas"""

    def __init__(self):
        """Inicializa el recolector de métricas"""
        self.metrics = {
            "requests_total": 0,
            "requests_by_endpoint": defaultdict(int),
            "requests_by_status": defaultdict(int),
            "response_times": deque(maxlen=1000),
            "projects_generated": 0,
            "projects_failed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "rate_limit_hits": 0,
        }
        self.start_time = time.time()

    def record_request(
        self,
        endpoint: str,
        status_code: int,
        response_time: float,
    ):
        """
        Registra una request.

        Args:
            endpoint: Endpoint llamado
            status_code: Código de estado HTTP
            response_time: Tiempo de respuesta en segundos
        """
        self.metrics["requests_total"] += 1
        self.metrics["requests_by_endpoint"][endpoint] += 1
        self.metrics["requests_by_status"][status_code] += 1
        self.metrics["response_times"].append(response_time)

    def record_project_generated(self, success: bool = True):
        """Registra generación de proyecto"""
        if success:
            self.metrics["projects_generated"] += 1
        else:
            self.metrics["projects_failed"] += 1

    def record_cache_hit(self, hit: bool = True):
        """Registra hit/miss de cache"""
        if hit:
            self.metrics["cache_hits"] += 1
        else:
            self.metrics["cache_misses"] += 1

    def record_rate_limit_hit(self):
        """Registra rate limit hit"""
        self.metrics["rate_limit_hits"] += 1

    def get_metrics(self) -> Dict[str, Any]:
        """
        Obtiene todas las métricas.

        Returns:
            Diccionario con todas las métricas
        """
        response_times = list(self.metrics["response_times"])
        avg_response_time = (
            sum(response_times) / len(response_times) if response_times else 0
        )

        uptime_seconds = time.time() - self.start_time

        return {
            "requests": {
                "total": self.metrics["requests_total"],
                "by_endpoint": dict(self.metrics["requests_by_endpoint"]),
                "by_status": dict(self.metrics["requests_by_status"]),
                "average_response_time_seconds": round(avg_response_time, 3),
            },
            "projects": {
                "generated": self.metrics["projects_generated"],
                "failed": self.metrics["projects_failed"],
                "success_rate": round(
                    self.metrics["projects_generated"]
                    / (self.metrics["projects_generated"] + self.metrics["projects_failed"])
                    * 100,
                    2
                ) if (self.metrics["projects_generated"] + self.metrics["projects_failed"]) > 0 else 0,
            },
            "cache": {
                "hits": self.metrics["cache_hits"],
                "misses": self.metrics["cache_misses"],
                "hit_rate": round(
                    self.metrics["cache_hits"]
                    / (self.metrics["cache_hits"] + self.metrics["cache_misses"])
                    * 100,
                    2
                ) if (self.metrics["cache_hits"] + self.metrics["cache_misses"]) > 0 else 0,
            },
            "rate_limiting": {
                "hits": self.metrics["rate_limit_hits"],
            },
            "system": {
                "uptime_seconds": round(uptime_seconds, 2),
                "uptime_hours": round(uptime_seconds / 3600, 2),
            },
        }

    def get_prometheus_metrics(self) -> str:
        """
        Obtiene métricas en formato Prometheus.

        Returns:
            Métricas en formato Prometheus
        """
        metrics = self.get_metrics()
        lines = []

        # Requests
        lines.append(f'# HELP requests_total Total number of requests')
        lines.append(f'# TYPE requests_total counter')
        lines.append(f'requests_total {metrics["requests"]["total"]}')

        # Response time
        lines.append(f'# HELP response_time_seconds Average response time')
        lines.append(f'# TYPE response_time_seconds gauge')
        lines.append(f'response_time_seconds {metrics["requests"]["average_response_time_seconds"]}')

        # Projects
        lines.append(f'# HELP projects_generated_total Total projects generated')
        lines.append(f'# TYPE projects_generated_total counter')
        lines.append(f'projects_generated_total {metrics["projects"]["generated"]}')

        # Cache
        lines.append(f'# HELP cache_hits_total Total cache hits')
        lines.append(f'# TYPE cache_hits_total counter')
        lines.append(f'cache_hits_total {metrics["cache"]["hits"]}')

        return "\n".join(lines)


