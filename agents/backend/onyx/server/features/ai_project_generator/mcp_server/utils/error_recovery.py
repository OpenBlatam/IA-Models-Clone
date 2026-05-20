"""
Error Recovery - Utilidades avanzadas de recuperación de errores
=================================================================

Funciones helper para recuperación automática y manejo inteligente de errores.
"""

import logging
from typing import Callable, Any, Optional, Dict, List, Type
from datetime import datetime, timedelta
from collections import defaultdict

from ..exceptions import MCPError, MCPConnectorError, MCPOperationError

logger = logging.getLogger(__name__)


class ErrorTracker:
    """
    Rastreador de errores para análisis y recuperación.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Inicializar rastreador de errores.
        
        Args:
            window_size: Tamaño de la ventana de errores a mantener
        """
        self.window_size = window_size
        self.errors: List[Dict[str, Any]] = []
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.last_error: Optional[Dict[str, Any]] = None
    
    def record_error(
        self,
        error_type: str,
        error_message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Registrar un error.
        
        Args:
            error_type: Tipo de error
            error_message: Mensaje de error
            context: Contexto adicional
        """
        error_record = {
            "timestamp": datetime.utcnow(),
            "error_type": error_type,
            "error_message": error_message,
            "context": context or {}
        }
        
        self.errors.append(error_record)
        self.error_counts[error_type] += 1
        self.last_error = error_record
        
        # Mantener solo los últimos N errores
        if len(self.errors) > self.window_size:
            self.errors.pop(0)
    
    def get_error_rate(self, error_type: Optional[str] = None, minutes: int = 5) -> float:
        """
        Obtener tasa de errores.
        
        Args:
            error_type: Tipo de error específico (opcional)
            minutes: Ventana de tiempo en minutos
        
        Returns:
            Tasa de errores (errores por minuto)
        """
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        recent_errors = [
            e for e in self.errors
            if e["timestamp"] >= cutoff and
            (error_type is None or e["error_type"] == error_type)
        ]
        
        return len(recent_errors) / minutes if minutes > 0 else 0.0
    
    def get_most_common_errors(self, limit: int = 5) -> List[tuple]:
        """
        Obtener errores más comunes.
        
        Args:
            limit: Número de errores a retornar
        
        Returns:
            Lista de tuplas (error_type, count)
        """
        return sorted(
            self.error_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
    
    def should_trigger_recovery(self, threshold: float = 5.0) -> bool:
        """
        Determinar si se debe activar recuperación.
        
        Args:
            threshold: Umbral de tasa de errores (errores por minuto)
        
        Returns:
            True si se debe activar recuperación
        """
        return self.get_error_rate(minutes=5) >= threshold


class ErrorRecoveryManager:
    """
    Gestor de recuperación de errores.
    
    Coordina estrategias de recuperación basadas en tipos de error.
    """
    
    def __init__(self):
        """Inicializar gestor de recuperación"""
        self.strategies: Dict[Type[Exception], Callable] = {}
        self.tracker = ErrorTracker()
        self.recovery_history: List[Dict[str, Any]] = []
    
    def register_strategy(
        self,
        error_type: Type[Exception],
        recovery_func: Callable
    ) -> None:
        """
        Registrar estrategia de recuperación para un tipo de error.
        
        Args:
            error_type: Tipo de excepción
            recovery_func: Función de recuperación
        """
        self.strategies[error_type] = recovery_func
        logger.info(f"Registered recovery strategy for {error_type.__name__}")
    
    async def attempt_recovery(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """
        Intentar recuperación de un error.
        
        Args:
            error: Excepción a recuperar
            context: Contexto adicional
        
        Returns:
            Resultado de la recuperación o None si no hay estrategia
        """
        # Registrar error
        self.tracker.record_error(
            error_type=type(error).__name__,
            error_message=str(error),
            context=context
        )
        
        # Buscar estrategia de recuperación
        error_type = type(error)
        recovery_func = self.strategies.get(error_type)
        
        if not recovery_func:
            # Buscar en la jerarquía de clases
            for registered_type, func in self.strategies.items():
                if issubclass(error_type, registered_type):
                    recovery_func = func
                    break
        
        if recovery_func:
            try:
                result = await recovery_func(error, context)
                self.recovery_history.append({
                    "timestamp": datetime.utcnow(),
                    "error_type": error_type.__name__,
                    "success": True,
                    "result": result
                })
                logger.info(f"Recovery successful for {error_type.__name__}")
                return result
            except Exception as recovery_error:
                logger.error(
                    f"Recovery failed for {error_type.__name__}: {recovery_error}",
                    exc_info=True
                )
                self.recovery_history.append({
                    "timestamp": datetime.utcnow(),
                    "error_type": error_type.__name__,
                    "success": False,
                    "error": str(recovery_error)
                })
        
        return None
    
    def get_recovery_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas de recuperación.
        
        Returns:
            Diccionario con estadísticas
        """
        successful = sum(1 for r in self.recovery_history if r.get("success"))
        total = len(self.recovery_history)
        
        return {
            "total_attempts": total,
            "successful": successful,
            "failed": total - successful,
            "success_rate": successful / total if total > 0 else 0.0,
            "error_rate": self.tracker.get_error_rate(),
            "most_common_errors": self.tracker.get_most_common_errors()
        }


def create_default_recovery_manager() -> ErrorRecoveryManager:
    """
    Crear gestor de recuperación con estrategias por defecto.
    
    Returns:
        ErrorRecoveryManager configurado
    """
    manager = ErrorRecoveryManager()
    
    # Estrategia para errores de connector
    async def connector_recovery(error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Recuperación para errores de connector"""
        logger.info("Attempting connector recovery...")
        # Aquí se podría implementar lógica de reconexión, etc.
        return {"status": "recovered", "action": "reconnect"}
    
    manager.register_strategy(MCPConnectorError, connector_recovery)
    
    # Estrategia para errores de operación
    async def operation_recovery(error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Recuperación para errores de operación"""
        logger.info("Attempting operation recovery...")
        return {"status": "recovered", "action": "retry"}
    
    manager.register_strategy(MCPOperationError, operation_recovery)
    
    return manager

