"""
Change History - Gestión del historial de cambios
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class ChangeHistory:
    """Gestor del historial de cambios"""

    def __init__(self, max_history: int = 1000):
        """
        Inicializar el gestor de historial.

        Args:
            max_history: Número máximo de registros a mantener
        """
        self.max_history = max_history
        self.changes: List[Dict[str, Any]] = []

    def record_change(self, change: Dict[str, Any]) -> str:
        """
        Registrar un cambio en el historial.

        Args:
            change: Diccionario con información del cambio

        Returns:
            ID del cambio registrado
        """
        change_id = str(uuid.uuid4())
        record = {
            "id": change_id,
            "timestamp": datetime.utcnow().isoformat(),
            **change
        }
        
        self.changes.append(record)
        
        # Limitar tamaño del historial
        if len(self.changes) > self.max_history:
            self.changes = self.changes[-self.max_history:]
        
        logger.debug(f"Cambio registrado: {change_id}")
        return change_id

    def get_recent_changes(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Obtener los cambios recientes.

        Args:
            limit: Número máximo de cambios a retornar

        Returns:
            Lista de cambios recientes
        """
        return self.changes[-limit:][::-1]  # Más recientes primero

    def get_change_by_id(self, change_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtener un cambio específico por ID.

        Args:
            change_id: ID del cambio

        Returns:
            Diccionario con el cambio o None si no se encuentra
        """
        for change in self.changes:
            if change.get("id") == change_id:
                return change
        return None

    def clear_history(self):
        """Limpiar todo el historial"""
        self.changes = []
        logger.info("Historial limpiado")






