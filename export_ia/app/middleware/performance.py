"""
Performance Middleware - Middleware de rendimiento avanzado
"""

import time
import asyncio
import logging
from typing import Dict, Any, Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import psutil
import threading
from collections import defaultdict, deque
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Métricas de rendimiento."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.request_times: deque = deque(maxlen=max_history)
        self.endpoint_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "count": 0,
            "total_time": 0.0,
            "min_time": float('inf'),
            "max_time": 0.0,
            "avg_time": 0.0,
            "errors": 0,
            "last_request": None
        })
        self.system_stats = {
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
            "memory_used_mb": 0.0,
            "disk_usage_percent": 0.0,
            "active_connections": 0,
            "last_update": None
        }
        self._lock = threading.Lock()
    
    def record_request(self, endpoint: str, duration: float, success: bool = True):
        """Registrar una request."""
        with self._lock:
            self.request_times.append({
                "endpoint": endpoint,
                "duration": duration,
                "timestamp": datetime.now(),
                "success": success
            })
            
            stats = self.endpoint_stats[endpoint]
            stats["count"] += 1
            stats["total_time"] += duration
            stats["min_time"] = min(stats["min_time"], duration)
            stats["max_time"] = max(stats["max_time"], duration)
            stats["avg_time"] = stats["total_time"] / stats["count"]
            stats["last_request"] = datetime.now()
            
            if not success:
                stats["errors"] += 1
    
    def update_system_stats(self):
        """Actualizar estadísticas del sistema."""
        try:
            self.system_stats.update({
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_used_mb": psutil.virtual_memory().used / (1024 * 1024),
                "disk_usage_percent": psutil.disk_usage('/').percent,
                "active_connections": len(psutil.net_connections()),
                "last_update": datetime.now()
            })
        except Exception as e:
            logger.error(f"Error al actualizar estadísticas del sistema: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        with self._lock:
            # Calcular estadísticas generales
            if self.request_times:
                recent_requests = [
                    req for req in self.request_times
                    if datetime.now() - req["timestamp"] < timedelta(minutes=5)
                ]
                
                if recent_requests:
                    avg_response_time = sum(req["duration"] for req in recent_requests) / len(recent_requests)
                    requests_per_minute = len(recent_requests) / 5
                    error_rate = sum(1 for req in recent_requests if not req["success"]) / len(recent_requests) * 100
                else:
                    avg_response_time = 0.0
                    requests_per_minute = 0.0
                    error_rate = 0.0
            else:
                avg_response_time = 0.0
                requests_per_minute = 0.0
                error_rate = 0.0
            
            return {
                "general": {
                    "total_requests": len(self.request_times),
                    "avg_response_time_ms": avg_response_time * 1000,
                    "requests_per_minute": requests_per_minute,
                    "error_rate_percent": error_rate,
                    "uptime_seconds": (datetime.now() - self.request_times[0]["timestamp"]).total_seconds() if self.request_times else 0
                },
                "endpoints": dict(self.endpoint_stats),
                "system": self.system_stats.copy()
            }


class PerformanceMiddleware(BaseHTTPMiddleware):
    """
    Middleware de rendimiento avanzado.
    """
    
    def __init__(self, app: ASGIApp, update_interval: int = 30):
        super().__init__(app)
        self.metrics = PerformanceMetrics()
        self.update_interval = update_interval
        self._system_update_task = None
        self._start_system_monitoring()
    
    def _start_system_monitoring(self):
        """Iniciar monitoreo del sistema."""
        def update_loop():
            while True:
                try:
                    self.metrics.update_system_stats()
                    time.sleep(self.update_interval)
                except Exception as e:
                    logger.error(f"Error en monitoreo del sistema: {e}")
                    time.sleep(self.update_interval)
        
        thread = threading.Thread(target=update_loop, daemon=True)
        thread.start()
    
    async def dispatch(self, request: Request, call_next):
        """Procesar request y medir rendimiento."""
        start_time = time.time()
        
        # Agregar información de rendimiento al request
        request.state.performance_start = start_time
        
        try:
            # Procesar request
            response = await call_next(request)
            
            # Calcular duración
            duration = time.time() - start_time
            
            # Determinar endpoint
            endpoint = f"{request.method} {request.url.path}"
            
            # Registrar métricas
            self.metrics.record_request(endpoint, duration, success=True)
            
            # Agregar headers de rendimiento
            response.headers["X-Process-Time"] = f"{duration:.4f}s"
            response.headers["X-Endpoint"] = endpoint
            
            return response
            
        except Exception as e:
            # Calcular duración incluso en caso de error
            duration = time.time() - start_time
            endpoint = f"{request.method} {request.url.path}"
            
            # Registrar error
            self.metrics.record_request(endpoint, duration, success=False)
            
            # Re-lanzar excepción
            raise e
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de rendimiento."""
        return self.metrics.get_stats()
    
    def get_endpoint_stats(self, endpoint: Optional[str] = None) -> Dict[str, Any]:
        """Obtener estadísticas de endpoint específico."""
        stats = self.metrics.get_stats()
        
        if endpoint:
            return stats["endpoints"].get(endpoint, {})
        
        return stats["endpoints"]
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del sistema."""
        return self.metrics.system_stats.copy()
    
    def get_slowest_endpoints(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Obtener endpoints más lentos."""
        stats = self.metrics.get_stats()
        endpoints = []
        
        for endpoint, data in stats["endpoints"].items():
            if data["count"] > 0:
                endpoints.append({
                    "endpoint": endpoint,
                    "avg_time_ms": data["avg_time"] * 1000,
                    "max_time_ms": data["max_time"] * 1000,
                    "count": data["count"],
                    "error_rate": (data["errors"] / data["count"]) * 100 if data["count"] > 0 else 0
                })
        
        # Ordenar por tiempo promedio
        endpoints.sort(key=lambda x: x["avg_time_ms"], reverse=True)
        
        return endpoints[:limit]
    
    def get_most_used_endpoints(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Obtener endpoints más utilizados."""
        stats = self.metrics.get_stats()
        endpoints = []
        
        for endpoint, data in stats["endpoints"].items():
            if data["count"] > 0:
                endpoints.append({
                    "endpoint": endpoint,
                    "count": data["count"],
                    "avg_time_ms": data["avg_time"] * 1000,
                    "total_time_ms": data["total_time"] * 1000,
                    "last_request": data["last_request"].isoformat() if data["last_request"] else None
                })
        
        # Ordenar por cantidad de requests
        endpoints.sort(key=lambda x: x["count"], reverse=True)
        
        return endpoints[:limit]
    
    def get_error_endpoints(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Obtener endpoints con más errores."""
        stats = self.metrics.get_stats()
        endpoints = []
        
        for endpoint, data in stats["endpoints"].items():
            if data["errors"] > 0:
                error_rate = (data["errors"] / data["count"]) * 100 if data["count"] > 0 else 0
                endpoints.append({
                    "endpoint": endpoint,
                    "errors": data["errors"],
                    "error_rate": error_rate,
                    "count": data["count"],
                    "last_request": data["last_request"].isoformat() if data["last_request"] else None
                })
        
        # Ordenar por tasa de error
        endpoints.sort(key=lambda x: x["error_rate"], reverse=True)
        
        return endpoints[:limit]
    
    def reset_stats(self):
        """Resetear estadísticas."""
        self.metrics = PerformanceMetrics()
        logger.info("Estadísticas de rendimiento reseteadas")
    
    def health_check(self) -> Dict[str, Any]:
        """Verificar salud del middleware de rendimiento."""
        try:
            stats = self.metrics.get_stats()
            
            # Verificar si hay problemas de rendimiento
            avg_response_time = stats["general"]["avg_response_time_ms"]
            error_rate = stats["general"]["error_rate_percent"]
            cpu_percent = stats["system"]["cpu_percent"]
            memory_percent = stats["system"]["memory_percent"]
            
            status = "healthy"
            issues = []
            
            if avg_response_time > 5000:  # 5 segundos
                status = "degraded"
                issues.append(f"Tiempo de respuesta alto: {avg_response_time:.2f}ms")
            
            if error_rate > 10:  # 10%
                status = "degraded"
                issues.append(f"Tasa de error alta: {error_rate:.2f}%")
            
            if cpu_percent > 90:
                status = "degraded"
                issues.append(f"CPU alto: {cpu_percent:.2f}%")
            
            if memory_percent > 90:
                status = "degraded"
                issues.append(f"Memoria alta: {memory_percent:.2f}%")
            
            return {
                "status": status,
                "issues": issues,
                "stats": stats,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error en health check de rendimiento: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }




