"""
Monitor de rendimiento del sistema
"""

import logging
import time
import psutil
import threading
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Métricas de rendimiento"""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_available_mb: float = 0.0
    disk_usage_percent: float = 0.0
    active_connections: int = 0
    request_rate: float = 0.0
    error_rate: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


class PerformanceMonitor:
    """Monitor de rendimiento del sistema"""
    
    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        self.metrics_history: deque = deque(maxlen=history_size)
        self.request_times: deque = deque(maxlen=1000)
        self.error_count = 0
        self.request_count = 0
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.start_time = time.time()
    
    def start_monitoring(self, interval: float = 5.0):
        """Inicia monitoreo en background"""
        if self.monitoring:
            return
        
        self.monitoring = True
        
        def monitor_loop():
            while self.monitoring:
                try:
                    metrics = self.collect_metrics()
                    self.metrics_history.append(metrics)
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"Error en monitor: {e}")
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Performance monitor iniciado")
    
    def stop_monitoring(self):
        """Detiene monitoreo"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        logger.info("Performance monitor detenido")
    
    def collect_metrics(self) -> PerformanceMetrics:
        """Recolecta métricas actuales"""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Memoria
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / (1024 * 1024)
            memory_available_mb = memory.available / (1024 * 1024)
            
            # Disco
            disk = psutil.disk_usage('/')
            disk_usage_percent = disk.percent
            
            # Calcular request rate
            uptime = time.time() - self.start_time
            request_rate = self.request_count / max(uptime, 1) * 60  # requests per minute
            error_rate = self.error_count / max(self.request_count, 1) * 100  # error percentage
            
            return PerformanceMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_mb=memory_used_mb,
                memory_available_mb=memory_available_mb,
                disk_usage_percent=disk_usage_percent,
                request_rate=request_rate,
                error_rate=error_rate
            )
        except Exception as e:
            logger.error(f"Error recolectando métricas: {e}")
            return PerformanceMetrics()
    
    def record_request(self, duration: float, is_error: bool = False):
        """Registra una petición"""
        self.request_count += 1
        if is_error:
            self.error_count += 1
        self.request_times.append(duration)
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Obtiene métricas actuales"""
        return self.collect_metrics()
    
    def get_average_response_time(self) -> float:
        """Obtiene tiempo promedio de respuesta"""
        if not self.request_times:
            return 0.0
        return sum(self.request_times) / len(self.request_times)
    
    def get_p95_response_time(self) -> float:
        """Obtiene percentil 95 de tiempo de respuesta"""
        if not self.request_times:
            return 0.0
        sorted_times = sorted(self.request_times)
        index = int(len(sorted_times) * 0.95)
        return sorted_times[index] if index < len(sorted_times) else sorted_times[-1]
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de métricas"""
        current = self.get_current_metrics()
        
        return {
            "current": {
                "cpu_percent": current.cpu_percent,
                "memory_percent": current.memory_percent,
                "memory_used_mb": current.memory_used_mb,
                "disk_usage_percent": current.disk_usage_percent,
                "request_rate": current.request_rate,
                "error_rate": current.error_rate
            },
            "performance": {
                "avg_response_time": self.get_average_response_time(),
                "p95_response_time": self.get_p95_response_time(),
                "total_requests": self.request_count,
                "total_errors": self.error_count
            },
            "uptime_seconds": time.time() - self.start_time
        }


# Singleton global
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Obtiene instancia singleton del monitor"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor




