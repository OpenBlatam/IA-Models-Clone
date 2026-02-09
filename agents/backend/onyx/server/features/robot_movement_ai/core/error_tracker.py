"""
Error Tracker System
===================

Sistema de seguimiento de errores.
"""

import logging
import traceback
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ErrorRecord:
    """Registro de error."""
    error_id: str
    error_type: str
    message: str
    traceback: str
    timestamp: str
    context: Dict[str, Any] = field(default_factory=dict)
    count: int = 1
    first_occurrence: str = ""
    last_occurrence: str = ""


class ErrorTracker:
    """
    Rastreador de errores.
    
    Rastrea y analiza errores del sistema.
    """
    
    def __init__(self, max_errors: int = 10000):
        """
        Inicializar rastreador.
        
        Args:
            max_errors: Máximo de errores a mantener
        """
        self.max_errors = max_errors
        self.errors: Dict[str, ErrorRecord] = {}
        self.error_history: List[ErrorRecord] = []
        self.error_counts: Dict[str, int] = defaultdict(int)
    
    def record_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> ErrorRecord:
        """
        Registrar error.
        
        Args:
            error: Excepción
            context: Contexto adicional
            
        Returns:
            Registro de error
        """
        error_type = type(error).__name__
        error_message = str(error)
        error_traceback = traceback.format_exc()
        
        # Crear ID único basado en tipo y mensaje
        error_id = f"{error_type}:{hash(error_message)}"
        
        timestamp = datetime.now().isoformat()
        
        if error_id in self.errors:
            # Actualizar error existente
            record = self.errors[error_id]
            record.count += 1
            record.last_occurrence = timestamp
            if context:
                record.context.update(context)
        else:
            # Crear nuevo registro
            record = ErrorRecord(
                error_id=error_id,
                error_type=error_type,
                message=error_message,
                traceback=error_traceback,
                timestamp=timestamp,
                context=context or {},
                first_occurrence=timestamp,
                last_occurrence=timestamp
            )
            self.errors[error_id] = record
        
        self.error_counts[error_type] += 1
        self.error_history.append(record)
        
        # Limitar tamaño del historial
        if len(self.error_history) > self.max_errors:
            self.error_history = self.error_history[-self.max_errors:]
        
        logger.error(f"Error recorded: {error_type} - {error_message}")
        
        return record
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de errores."""
        total_errors = len(self.error_history)
        unique_errors = len(self.errors)
        
        # Errores más frecuentes
        top_errors = sorted(
            self.errors.items(),
            key=lambda x: x[1].count,
            reverse=True
        )[:10]
        
        # Errores por tipo
        errors_by_type = {}
        for error_type, count in self.error_counts.items():
            errors_by_type[error_type] = count
        
        return {
            "total_errors": total_errors,
            "unique_errors": unique_errors,
            "errors_by_type": errors_by_type,
            "top_errors": [
                {
                    "error_id": error_id,
                    "error_type": record.error_type,
                    "message": record.message,
                    "count": record.count,
                    "first_occurrence": record.first_occurrence,
                    "last_occurrence": record.last_occurrence
                }
                for error_id, record in top_errors
            ]
        }
    
    def get_recent_errors(self, limit: int = 100) -> List[ErrorRecord]:
        """
        Obtener errores recientes.
        
        Args:
            limit: Límite de resultados
            
        Returns:
            Lista de errores recientes
        """
        return self.error_history[-limit:]
    
    def get_error_by_id(self, error_id: str) -> Optional[ErrorRecord]:
        """Obtener error por ID."""
        return self.errors.get(error_id)
    
    def clear_errors(self) -> None:
        """Limpiar todos los errores."""
        self.errors.clear()
        self.error_history.clear()
        self.error_counts.clear()
        logger.info("All errors cleared")


# Instancia global
_error_tracker: Optional[ErrorTracker] = None


def get_error_tracker(max_errors: int = 10000) -> ErrorTracker:
    """Obtener instancia global del rastreador de errores."""
    global _error_tracker
    if _error_tracker is None:
        _error_tracker = ErrorTracker(max_errors=max_errors)
    return _error_tracker






