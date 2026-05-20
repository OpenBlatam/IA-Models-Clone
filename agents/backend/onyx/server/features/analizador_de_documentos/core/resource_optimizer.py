"""
Optimizador de Recursos
========================

Sistema para optimización automática de recursos del sistema.
"""

import logging
import gc
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

# Importar psutil si está disponible
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil no disponible, funcionalidad limitada")


@dataclass
class ResourceMetrics:
    """Métricas de recursos"""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    network_sent_mb: float
    network_recv_mb: float


class ResourceOptimizer:
    """
    Optimizador de recursos
    
    Proporciona:
    - Monitoreo de recursos del sistema
    - Optimización automática
    - Limpieza de memoria
    - Recomendaciones de optimización
    - Alertas de recursos
    """
    
    def __init__(self):
        """Inicializar optimizador"""
        self.metrics_history: List[ResourceMetrics] = []
        self.optimization_history: List[Dict[str, Any]] = []
        logger.info("ResourceOptimizer inicializado")
    
    def get_current_metrics(self) -> ResourceMetrics:
        """Obtener métricas actuales del sistema"""
        if not PSUTIL_AVAILABLE:
            # Retornar métricas vacías si psutil no está disponible
            return ResourceMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_mb=0.0,
                memory_available_mb=0.0,
                disk_usage_percent=0.0,
                network_sent_mb=0.0,
                network_recv_mb=0.0
            )
        
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            metrics = ResourceMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                memory_available_mb=memory.available / (1024 * 1024),
                disk_usage_percent=disk.percent,
                network_sent_mb=network.bytes_sent / (1024 * 1024),
                network_recv_mb=network.bytes_recv / (1024 * 1024)
            )
            
            self.metrics_history.append(metrics)
            
            # Mantener solo últimos 1000 registros
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-1000:]
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error obteniendo métricas: {e}")
            # Retornar métricas vacías en caso de error
            return ResourceMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_mb=0.0,
                memory_available_mb=0.0,
                disk_usage_percent=0.0,
                network_sent_mb=0.0,
                network_recv_mb=0.0
            )
    
    def optimize_memory(self) -> Dict[str, Any]:
        """Optimizar memoria"""
        if not PSUTIL_AVAILABLE:
            # Solo hacer garbage collection
            gc.collect()
            return {
                "type": "memory",
                "timestamp": datetime.now().isoformat(),
                "memory_freed_mb": 0.0,
                "note": "psutil no disponible, solo se ejecutó garbage collection"
            }
        
        try:
            before = psutil.virtual_memory().used / (1024 * 1024)
            
            # Forzar garbage collection
            gc.collect()
            
            after = psutil.virtual_memory().used / (1024 * 1024)
            freed = before - after
            
            optimization = {
                "type": "memory",
                "timestamp": datetime.now().isoformat(),
                "memory_freed_mb": freed,
                "before_mb": before,
                "after_mb": after
            }
            
            self.optimization_history.append(optimization)
            
            logger.info(f"Memoria optimizada: {freed:.2f} MB liberados")
            
            return optimization
            
        except Exception as e:
            logger.error(f"Error optimizando memoria: {e}")
            return {"error": str(e)}
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Obtener recomendaciones de optimización"""
        recommendations = []
        
        if not self.metrics_history:
            return recommendations
        
        latest = self.metrics_history[-1]
        
        if latest.memory_percent > 85:
            recommendations.append({
                "type": "memory",
                "priority": "high",
                "message": f"Uso de memoria alto ({latest.memory_percent:.1f}%). Considera optimizar memoria.",
                "action": "optimize_memory"
            })
        
        if latest.cpu_percent > 90:
            recommendations.append({
                "type": "cpu",
                "priority": "high",
                "message": f"Uso de CPU alto ({latest.cpu_percent:.1f}%). Considera escalar recursos.",
                "action": "scale_up"
            })
        
        if latest.disk_usage_percent > 90:
            recommendations.append({
                "type": "disk",
                "priority": "critical",
                "message": f"Espacio en disco bajo ({latest.disk_usage_percent:.1f}%). Considera limpiar espacio.",
                "action": "cleanup_disk"
            })
        
        return recommendations
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Obtener resumen de recursos"""
        if not self.metrics_history:
            return {"status": "no_data"}
        
        latest = self.metrics_history[-1]
        
        # Calcular promedios de últimos 10 registros
        recent = self.metrics_history[-10:]
        avg_cpu = sum(m.cpu_percent for m in recent) / len(recent)
        avg_memory = sum(m.memory_percent for m in recent) / len(recent)
        
        return {
            "current": {
                "cpu_percent": latest.cpu_percent,
                "memory_percent": latest.memory_percent,
                "memory_used_mb": latest.memory_used_mb,
                "memory_available_mb": latest.memory_available_mb,
                "disk_usage_percent": latest.disk_usage_percent
            },
            "averages": {
                "cpu_percent": avg_cpu,
                "memory_percent": avg_memory
            },
            "recommendations": self.get_optimization_recommendations()
        }


# Instancia global
_resource_optimizer: Optional[ResourceOptimizer] = None


def get_resource_optimizer() -> ResourceOptimizer:
    """Obtener instancia global del optimizador"""
    global _resource_optimizer
    if _resource_optimizer is None:
        _resource_optimizer = ResourceOptimizer()
    return _resource_optimizer

