"""
Pipeline Monitoring (optimizado)

Sistema de monitoreo y observabilidad para pipelines.
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict

from .base import BasePipeline, PipelineContext, PipelineStatus

logger = logging.getLogger(__name__)


class PipelineMonitor:
    """
    Monitor de pipelines (optimizado).
    
    Rastrea ejecuciones, métricas, y estadísticas de pipelines.
    """
    
    def __init__(self):
        """Inicializar monitor (optimizado)."""
        self.executions: List[Dict[str, Any]] = []
        self.metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._max_executions = 1000
    
    def record_execution(
        self,
        pipeline: BasePipeline,
        result: Any,
        execution_time: float,
        success: bool
    ):
        """
        Registrar ejecución de pipeline (optimizado).
        
        Args:
            pipeline: Pipeline ejecutado
            result: Resultado de la ejecución
            execution_time: Tiempo de ejecución
            success: Si fue exitoso
        """
        execution_record = {
            "pipeline_name": pipeline.name,
            "pipeline_id": pipeline.context.pipeline_id if pipeline.context else None,
            "success": success,
            "execution_time": execution_time,
            "timestamp": datetime.utcnow().isoformat(),
            "stage_count": len(pipeline.stages)
        }
        
        self.executions.append(execution_record)
        
        # Limitar tamaño
        if len(self.executions) > self._max_executions:
            self.executions = self.executions[-self._max_executions:]
        
        # Actualizar métricas
        self._update_metrics(pipeline.name, execution_time, success)
    
    def _update_metrics(self, pipeline_name: str, execution_time: float, success: bool):
        """
        Actualizar métricas de pipeline (optimizado).
        
        Args:
            pipeline_name: Nombre del pipeline
            execution_time: Tiempo de ejecución
            success: Si fue exitoso
        """
        if pipeline_name not in self.metrics:
            self.metrics[pipeline_name] = {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "total_time": 0.0,
                "min_time": float('inf'),
                "max_time": 0.0
            }
        
        metrics = self.metrics[pipeline_name]
        metrics["total_executions"] += 1
        metrics["total_time"] += execution_time
        
        if success:
            metrics["successful_executions"] += 1
        else:
            metrics["failed_executions"] += 1
        
        metrics["min_time"] = min(metrics["min_time"], execution_time)
        metrics["max_time"] = max(metrics["max_time"], execution_time)
    
    def get_metrics(self, pipeline_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtener métricas (optimizado).
        
        Args:
            pipeline_name: Nombre del pipeline (opcional, si None retorna todos)
            
        Returns:
            Dict con métricas
        """
        if pipeline_name:
            metrics = self.metrics.get(pipeline_name, {})
            if metrics:
                # Calcular promedios
                total = metrics["total_executions"]
                if total > 0:
                    metrics["average_time"] = metrics["total_time"] / total
                    metrics["success_rate"] = metrics["successful_executions"] / total
                else:
                    metrics["average_time"] = 0.0
                    metrics["success_rate"] = 0.0
            return metrics
        
        # Retornar todas las métricas
        all_metrics = {}
        for name, metrics in self.metrics.items():
            total = metrics["total_executions"]
            if total > 0:
                metrics["average_time"] = metrics["total_time"] / total
                metrics["success_rate"] = metrics["successful_executions"] / total
            else:
                metrics["average_time"] = 0.0
                metrics["success_rate"] = 0.0
            all_metrics[name] = metrics
        
        return all_metrics
    
    def get_recent_executions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Obtener ejecuciones recientes (optimizado).
        
        Args:
            limit: Número máximo de ejecuciones
            
        Returns:
            Lista de ejecuciones recientes
        """
        return self.executions[-limit:] if limit <= len(self.executions) else self.executions.copy()
    
    def get_execution_history(
        self,
        pipeline_name: Optional[str] = None,
        since: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Obtener historial de ejecuciones (optimizado).
        
        Args:
            pipeline_name: Filtrar por nombre de pipeline (opcional)
            since: Filtrar desde fecha (opcional)
            
        Returns:
            Lista de ejecuciones filtradas
        """
        filtered = self.executions
        
        if pipeline_name:
            filtered = [e for e in filtered if e["pipeline_name"] == pipeline_name]
        
        if since:
            since_iso = since.isoformat()
            filtered = [e for e in filtered if e["timestamp"] >= since_iso]
        
        return filtered
    
    def clear_metrics(self, pipeline_name: Optional[str] = None):
        """
        Limpiar métricas (optimizado).
        
        Args:
            pipeline_name: Nombre del pipeline (opcional, si None limpia todos)
        """
        if pipeline_name:
            if pipeline_name in self.metrics:
                del self.metrics[pipeline_name]
                logger.info(f"Cleared metrics for pipeline: {pipeline_name}")
        else:
            self.metrics.clear()
            logger.info("Cleared all pipeline metrics")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Obtener resumen de monitoreo (optimizado).
        
        Returns:
            Dict con resumen de todas las métricas
        """
        total_pipelines = len(self.metrics)
        total_executions = sum(m["total_executions"] for m in self.metrics.values())
        total_successful = sum(m["successful_executions"] for m in self.metrics.values())
        total_failed = sum(m["failed_executions"] for m in self.metrics.values())
        
        return {
            "total_pipelines": total_pipelines,
            "total_executions": total_executions,
            "total_successful": total_successful,
            "total_failed": total_failed,
            "overall_success_rate": (
                total_successful / total_executions
                if total_executions > 0
                else 0.0
            ),
            "pipelines": list(self.metrics.keys())
        }


# Instancia global del monitor
_global_monitor = PipelineMonitor()


def get_monitor() -> PipelineMonitor:
    """
    Obtener instancia global del monitor (optimizado).
    
    Returns:
        Instancia de PipelineMonitor
    """
    return _global_monitor

