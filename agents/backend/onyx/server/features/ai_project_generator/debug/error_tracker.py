"""
Error Tracker - Rastreador de errores avanzado
===============================================

Tracking y análisis de errores con contexto y estadísticas.
"""

import logging
import traceback
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class ErrorTracker:
    """
    Rastreador de errores con:
    - Tracking de errores con contexto
    - Estadísticas de errores
    - Agrupación de errores similares
    - Historial de errores
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Args:
            max_history: Máximo de errores en historial
        """
        self.max_history = max_history
        self.errors: List[Dict[str, Any]] = []
        self.error_stats: Dict[str, int] = defaultdict(int)
        self.error_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    def track_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """
        Rastrea un error.
        
        Args:
            error: Excepción
            context: Contexto adicional
            request_id: ID de request
            user_id: ID de usuario
        """
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "context": context or {},
            "request_id": request_id,
            "user_id": user_id
        }
        
        # Agregar a historial
        self.errors.append(error_info)
        if len(self.errors) > self.max_history:
            self.errors.pop(0)
        
        # Actualizar estadísticas
        error_key = f"{error_info['error_type']}:{error_info['error_message'][:50]}"
        self.error_stats[error_key] += 1
        
        # Agrupar errores similares
        self.error_groups[error_key].append(error_info)
        if len(self.error_groups[error_key]) > 100:
            self.error_groups[error_key].pop(0)
        
        logger.error(f"Error tracked: {error_info['error_type']}", exc_info=True)
    
    def get_recent_errors(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Obtiene errores recientes"""
        return self.errors[-limit:]
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de errores"""
        return {
            "total_errors": len(self.errors),
            "error_types": dict(self.error_stats),
            "top_errors": sorted(
                self.error_stats.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }
    
    def get_error_group(self, error_key: str) -> List[Dict[str, Any]]:
        """Obtiene grupo de errores similares"""
        return self.error_groups.get(error_key, [])
    
    def get_errors_by_type(self, error_type: str) -> List[Dict[str, Any]]:
        """Obtiene errores por tipo"""
        return [
            e for e in self.errors
            if e["error_type"] == error_type
        ]
    
    def get_errors_by_time_range(
        self,
        start: datetime,
        end: datetime
    ) -> List[Dict[str, Any]]:
        """Obtiene errores en rango de tiempo"""
        return [
            e for e in self.errors
            if start <= datetime.fromisoformat(e["timestamp"]) <= end
        ]
    
    def clear_errors(self):
        """Limpia historial de errores"""
        self.errors.clear()
        self.error_stats.clear()
        self.error_groups.clear()


# Instancia global
_error_tracker: Optional[ErrorTracker] = None


def get_error_tracker() -> ErrorTracker:
    """Obtiene instancia de error tracker"""
    global _error_tracker
    if _error_tracker is None:
        _error_tracker = ErrorTracker()
    return _error_tracker















