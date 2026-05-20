"""
Metrics Tracker Module
=====================

Tracking centralizado de métricas del agente.
Proporciona una interfaz estructurada para registrar métricas.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .metrics_manager import MetricsManager

logger = logging.getLogger(__name__)


class MetricsTracker:
    """
    Tracker centralizado de métricas.
    
    Proporciona una interfaz estructurada para registrar
    diferentes tipos de métricas del agente.
    """
    
    def __init__(self, metrics_manager: MetricsManager):
        """
        Inicializar tracker de métricas.
        
        Args:
            metrics_manager: Manager de métricas
        """
        self.metrics_manager = metrics_manager
        self._tracking_enabled = True
    
    def track_llm_call(
        self,
        tokens_used: int,
        response_time: float,
        success: bool = True,
        error: Optional[str] = None
    ) -> None:
        """
        Registrar una llamada LLM.
        
        Args:
            tokens_used: Tokens utilizados
            response_time: Tiempo de respuesta en segundos
            success: Si la llamada fue exitosa
            error: Mensaje de error si falló
        """
        if not self._tracking_enabled:
            return
        
        try:
            self.metrics_manager.record_llm_call(tokens_used, response_time)
            if not success:
                self.metrics_manager.record_error()
                logger.debug(f"LLM call failed: {error}")
        except Exception as e:
            logger.warning(f"Error tracking LLM call: {e}")
    
    def track_task_processed(self, task_id: Optional[str] = None) -> None:
        """
        Registrar que una tarea fue procesada.
        
        Args:
            task_id: ID de la tarea (opcional)
        """
        if not self._tracking_enabled:
            return
        
        try:
            self.metrics_manager.record_task_processed()
            logger.debug(f"Task processed: {task_id}" if task_id else "Task processed")
        except Exception as e:
            logger.warning(f"Error tracking task processed: {e}")
    
    def track_task_completed(
        self,
        task_id: Optional[str] = None,
        duration: Optional[float] = None
    ) -> None:
        """
        Registrar que una tarea fue completada.
        
        Args:
            task_id: ID de la tarea (opcional)
            duration: Duración de la tarea en segundos (opcional)
        """
        if not self._tracking_enabled:
            return
        
        try:
            self.metrics_manager.record_task_completed()
            if duration:
                logger.debug(f"Task completed in {duration:.2f}s: {task_id}" if task_id else f"Task completed in {duration:.2f}s")
            else:
                logger.debug(f"Task completed: {task_id}" if task_id else "Task completed")
        except Exception as e:
            logger.warning(f"Error tracking task completed: {e}")
    
    def track_task_failed(
        self,
        task_id: Optional[str] = None,
        error: Optional[str] = None
    ) -> None:
        """
        Registrar que una tarea falló.
        
        Args:
            task_id: ID de la tarea (opcional)
            error: Mensaje de error (opcional)
        """
        if not self._tracking_enabled:
            return
        
        try:
            self.metrics_manager.record_task_failed()
            self.metrics_manager.record_error()
            logger.debug(f"Task failed: {task_id}" + (f" - {error}" if error else ""))
        except Exception as e:
            logger.warning(f"Error tracking task failed: {e}")
    
    def track_error(
        self,
        error_type: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Registrar un error.
        
        Args:
            error_type: Tipo de error (opcional)
            context: Contexto adicional (opcional)
        """
        if not self._tracking_enabled:
            return
        
        try:
            self.metrics_manager.record_error()
            log_msg = f"Error tracked: {error_type}" if error_type else "Error tracked"
            if context:
                log_msg += f" - Context: {context}"
            logger.debug(log_msg)
        except Exception as e:
            logger.warning(f"Error tracking error: {e}")
    
    def track_activity(self) -> None:
        """Registrar actividad del agente."""
        if not self._tracking_enabled:
            return
        
        try:
            self.metrics_manager.update_activity()
        except Exception as e:
            logger.warning(f"Error tracking activity: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Obtener métricas actuales.
        
        Returns:
            Dict con métricas
        """
        try:
            return self.metrics_manager.get_metrics()
        except Exception as e:
            logger.warning(f"Error getting metrics: {e}")
            return {}
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Obtener resumen de métricas.
        
        Returns:
            Dict con resumen de métricas
        """
        metrics = self.get_metrics()
        
        return {
            "total_tasks": metrics.get("total_tasks_processed", 0),
            "completed": metrics.get("total_tasks_completed", 0),
            "failed": metrics.get("total_tasks_failed", 0),
            "success_rate": metrics.get("success_rate", 0.0),
            "llm_calls": metrics.get("total_llm_calls", 0),
            "tokens_used": metrics.get("total_tokens_used", 0),
            "errors": metrics.get("errors_count", 0),
            "avg_response_time": metrics.get("average_response_time", 0.0)
        }
    
    def enable_tracking(self) -> None:
        """Habilitar tracking de métricas."""
        self._tracking_enabled = True
        logger.debug("Metrics tracking enabled")
    
    def disable_tracking(self) -> None:
        """Deshabilitar tracking de métricas."""
        self._tracking_enabled = False
        logger.debug("Metrics tracking disabled")
    
    def is_tracking_enabled(self) -> bool:
        """
        Verificar si el tracking está habilitado.
        
        Returns:
            True si está habilitado
        """
        return self._tracking_enabled
    
    def reset_metrics(self) -> None:
        """Reiniciar métricas (crea nuevas métricas)."""
        try:
            # Crear nuevo MetricsManager
            self.metrics_manager = MetricsManager()
            logger.debug("Metrics reset")
        except Exception as e:
            logger.warning(f"Error resetting metrics: {e}")


def create_metrics_tracker(metrics_manager: MetricsManager) -> MetricsTracker:
    """
    Factory function para crear MetricsTracker.
    
    Args:
        metrics_manager: Manager de métricas
        
    Returns:
        Instancia de MetricsTracker
    """
    return MetricsTracker(metrics_manager)


