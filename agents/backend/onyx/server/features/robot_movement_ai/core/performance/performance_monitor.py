"""
Performance Monitor System
===========================

Sistema de monitoreo de performance avanzado.
"""

import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PerformanceSnapshot:
    """Snapshot de performance."""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    active_requests: int
    response_time: float
    throughput: float
    error_rate: float


class PerformanceMonitor:
    """
    Monitor de performance.
    
    Monitorea y analiza el performance del sistema.
    """
    
    def __init__(self, history_size: int = 1000):
        """
        Inicializar monitor.
        
        Args:
            history_size: Tamaño del historial
        """
        self.history_size = history_size
        self.snapshots: deque = deque(maxlen=history_size)
        self.request_times: deque = deque(maxlen=history_size)
        self.error_count = 0
        self.request_count = 0
        self.start_time = time.time()
    
    def record_request(self, duration: float, success: bool = True) -> None:
        """
        Registrar solicitud.
        
        Args:
            duration: Duración en segundos
            success: Si fue exitosa
        """
        self.request_count += 1
        self.request_times.append(duration)
        
        if not success:
            self.error_count += 1
    
    def take_snapshot(self) -> PerformanceSnapshot:
        """
        Tomar snapshot de performance.
        
        Returns:
            Snapshot de performance
        """
        import psutil
        
        process = psutil.Process()
        cpu_usage = process.cpu_percent(interval=0.1)
        memory_info = process.memory_info()
        memory_usage = memory_info.rss / (1024 * 1024)  # MB
        
        # Calcular métricas de requests
        active_requests = len(self.request_times)
        
        if self.request_times:
            response_time = sum(self.request_times) / len(self.request_times)
        else:
            response_time = 0.0
        
        # Throughput (requests por segundo)
        uptime = time.time() - self.start_time
        throughput = self.request_count / uptime if uptime > 0 else 0.0
        
        # Error rate
        error_rate = self.error_count / self.request_count if self.request_count > 0 else 0.0
        
        snapshot = PerformanceSnapshot(
            timestamp=time.time(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            active_requests=active_requests,
            response_time=response_time,
            throughput=throughput,
            error_rate=error_rate
        )
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def get_performance_metrics(
        self,
        window_seconds: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Obtener métricas de performance.
        
        Args:
            window_seconds: Ventana de tiempo (None = todo el historial)
            
        Returns:
            Métricas de performance
        """
        if not self.snapshots:
            return {
                "cpu_usage": 0.0,
                "memory_usage": 0.0,
                "response_time": 0.0,
                "throughput": 0.0,
                "error_rate": 0.0
            }
        
        # Filtrar por ventana de tiempo
        if window_seconds:
            cutoff_time = time.time() - window_seconds
            relevant_snapshots = [
                s for s in self.snapshots
                if s.timestamp >= cutoff_time
            ]
        else:
            relevant_snapshots = list(self.snapshots)
        
        if not relevant_snapshots:
            return {
                "cpu_usage": 0.0,
                "memory_usage": 0.0,
                "response_time": 0.0,
                "throughput": 0.0,
                "error_rate": 0.0
            }
        
        # Calcular promedios
        avg_cpu = sum(s.cpu_usage for s in relevant_snapshots) / len(relevant_snapshots)
        avg_memory = sum(s.memory_usage for s in relevant_snapshots) / len(relevant_snapshots)
        avg_response_time = sum(s.response_time for s in relevant_snapshots) / len(relevant_snapshots)
        avg_throughput = sum(s.throughput for s in relevant_snapshots) / len(relevant_snapshots)
        avg_error_rate = sum(s.error_rate for s in relevant_snapshots) / len(relevant_snapshots)
        
        # Calcular percentiles de response time
        response_times = [s.response_time for s in relevant_snapshots if s.response_time > 0]
        if response_times:
            sorted_times = sorted(response_times)
            p50 = sorted_times[len(sorted_times) // 2]
            p95 = sorted_times[int(len(sorted_times) * 0.95)]
            p99 = sorted_times[int(len(sorted_times) * 0.99)]
        else:
            p50 = p95 = p99 = 0.0
        
        return {
            "cpu_usage": avg_cpu,
            "memory_usage": avg_memory,
            "response_time": avg_response_time,
            "response_time_p50": p50,
            "response_time_p95": p95,
            "response_time_p99": p99,
            "throughput": avg_throughput,
            "error_rate": avg_error_rate,
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "uptime": time.time() - self.start_time
        }
    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """
        Detectar anomalías en performance.
        
        Returns:
            Lista de anomalías detectadas
        """
        if len(self.snapshots) < 10:
            return []
        
        anomalies = []
        recent_snapshots = list(self.snapshots)[-10:]
        
        # Calcular promedios históricos
        historical_cpu = sum(s.cpu_usage for s in self.snapshots) / len(self.snapshots)
        historical_memory = sum(s.memory_usage for s in self.snapshots) / len(self.snapshots)
        historical_response = sum(s.response_time for s in self.snapshots) / len(self.snapshots)
        
        # Detectar anomalías en snapshots recientes
        for snapshot in recent_snapshots:
            if snapshot.cpu_usage > historical_cpu * 2:
                anomalies.append({
                    "type": "high_cpu",
                    "timestamp": snapshot.timestamp,
                    "value": snapshot.cpu_usage,
                    "threshold": historical_cpu * 2
                })
            
            if snapshot.memory_usage > historical_memory * 1.5:
                anomalies.append({
                    "type": "high_memory",
                    "timestamp": snapshot.timestamp,
                    "value": snapshot.memory_usage,
                    "threshold": historical_memory * 1.5
                })
            
            if snapshot.response_time > historical_response * 3:
                anomalies.append({
                    "type": "slow_response",
                    "timestamp": snapshot.timestamp,
                    "value": snapshot.response_time,
                    "threshold": historical_response * 3
                })
        
        return anomalies


# Instancia global
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor(history_size: int = 1000) -> PerformanceMonitor:
    """Obtener instancia global del monitor de performance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor(history_size=history_size)
    return _performance_monitor






