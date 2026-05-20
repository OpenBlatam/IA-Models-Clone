"""
Monitoring Service - Servicio de Monitoreo
===========================================

Sistema de monitoreo y métricas del sistema.
"""

import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class MonitoringService:
    """Servicio de monitoreo"""
    
    def __init__(self):
        """Inicializar servicio de monitoreo"""
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.counters: Dict[str, int] = defaultdict(int)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        logger.info("Monitoring Service inicializado")
    
    def increment_counter(self, name: str, value: int = 1):
        """
        Incrementar contador
        
        Args:
            name: Nombre del contador
            value: Valor a incrementar
        """
        self.counters[name] += value
        self.metrics[f"counter:{name}"].append({
            "timestamp": datetime.now().isoformat(),
            "value": self.counters[name]
        })
    
    def record_timing(self, name: str, duration: float):
        """
        Registrar tiempo de ejecución
        
        Args:
            name: Nombre de la operación
            duration: Duración en segundos
        """
        self.timers[name].append(duration)
        if len(self.timers[name]) > 1000:
            self.timers[name] = self.timers[name][-1000:]
        
        self.metrics[f"timer:{name}"].append({
            "timestamp": datetime.now().isoformat(),
            "duration": duration
        })
    
    def record_metric(self, name: str, value: Any):
        """
        Registrar métrica
        
        Args:
            name: Nombre de la métrica
            value: Valor
        """
        self.metrics[name].append({
            "timestamp": datetime.now().isoformat(),
            "value": value
        })
    
    def get_counter(self, name: str) -> int:
        """
        Obtener valor de contador
        
        Args:
            name: Nombre del contador
            
        Returns:
            Valor del contador
        """
        return self.counters.get(name, 0)
    
    def get_timing_stats(self, name: str) -> Dict[str, float]:
        """
        Obtener estadísticas de timing
        
        Args:
            name: Nombre de la operación
            
        Returns:
            Dict con estadísticas
        """
        timings = self.timers.get(name, [])
        if not timings:
            return {
                "count": 0,
                "avg": 0.0,
                "min": 0.0,
                "max": 0.0
            }
        
        return {
            "count": len(timings),
            "avg": sum(timings) / len(timings),
            "min": min(timings),
            "max": max(timings),
            "p95": sorted(timings)[int(len(timings) * 0.95)] if timings else 0.0
        }
    
    def get_metrics_summary(
        self,
        hours: int = 24
    ) -> Dict[str, Any]:
        """
        Obtener resumen de métricas
        
        Args:
            hours: Horas hacia atrás
            
        Returns:
            Dict con resumen
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        summary = {
            "period_hours": hours,
            "counters": dict(self.counters),
            "timings": {},
            "recent_metrics": {}
        }
        
        # Estadísticas de timings
        for name in self.timers.keys():
            summary["timings"][name] = self.get_timing_stats(name)
        
        # Métricas recientes
        for metric_name, values in self.metrics.items():
            recent = [
                v for v in values
                if datetime.fromisoformat(v["timestamp"]) >= cutoff_time
            ]
            if recent:
                summary["recent_metrics"][metric_name] = recent[-10:]  # Últimas 10
        
        return summary
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Obtener estado de salud del sistema
        
        Returns:
            Dict con estado de salud
        """
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "counters": dict(self.counters),
            "active_timers": len(self.timers),
            "total_metrics": sum(len(v) for v in self.metrics.values())
        }
    
    def reset(self):
        """Resetear todas las métricas"""
        self.counters.clear()
        self.timers.clear()
        self.metrics.clear()
        logger.info("Métricas reseteadas")
    
    def get_alerts(
        self,
        threshold_config: Dict[str, Dict[str, float]]
    ) -> List[Dict[str, Any]]:
        """
        Obtener alertas basadas en umbrales
        
        Args:
            threshold_config: Configuración de umbrales
                Ej: {"error_rate": {"max": 0.1}, "response_time": {"max": 2.0}}
            
        Returns:
            Lista de alertas
        """
        alerts = []
        
        for metric_name, thresholds in threshold_config.items():
            if metric_name.startswith("counter:"):
                counter_name = metric_name.replace("counter:", "")
                value = self.get_counter(counter_name)
                
                if "max" in thresholds and value > thresholds["max"]:
                    alerts.append({
                        "metric": metric_name,
                        "value": value,
                        "threshold": thresholds["max"],
                        "severity": "high" if value > thresholds["max"] * 2 else "medium"
                    })
            
            elif metric_name.startswith("timer:"):
                timer_name = metric_name.replace("timer:", "")
                stats = self.get_timing_stats(timer_name)
                
                if "max" in thresholds and stats.get("avg", 0) > thresholds["max"]:
                    alerts.append({
                        "metric": metric_name,
                        "value": stats["avg"],
                        "threshold": thresholds["max"],
                        "severity": "high" if stats["avg"] > thresholds["max"] * 2 else "medium"
                    })
        
        return alerts
    
    def export_metrics(
        self,
        file_path: str,
        format: str = "json"
    ) -> bool:
        """
        Exportar métricas a archivo
        
        Args:
            file_path: Ruta del archivo
            format: Formato (json, csv)
            
        Returns:
            True si se exportó exitosamente
        """
        try:
            if format == "json":
                import json
                data = {
                    "exported_at": datetime.now().isoformat(),
                    "counters": dict(self.counters),
                    "timings": {
                        name: self.get_timing_stats(name)
                        for name in self.timers.keys()
                    },
                    "metrics": {
                        name: list(values)
                        for name, values in self.metrics.items()
                    }
                }
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, default=str)
            
            elif format == "csv":
                import csv
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Metric", "Type", "Value"])
                    
                    for name, value in self.counters.items():
                        writer.writerow([name, "counter", value])
                    
                    for name in self.timers.keys():
                        stats = self.get_timing_stats(name)
                        writer.writerow([name, "timer_avg", stats["avg"]])
                        writer.writerow([name, "timer_max", stats["max"]])
            
            logger.info(f"Métricas exportadas a {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exportando métricas: {e}")
            return False


class TimingContext:
    """Context manager para medir tiempo"""
    
    def __init__(self, monitoring: MonitoringService, operation_name: str):
        self.monitoring = monitoring
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.monitoring.record_timing(self.operation_name, duration)



