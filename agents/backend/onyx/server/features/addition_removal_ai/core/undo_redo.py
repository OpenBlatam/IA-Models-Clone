"""
Undo/Redo - Sistema de deshacer y rehacer operaciones
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class UndoRedoManager:
    """Gestor de operaciones undo/redo"""

    def __init__(self, max_history: int = 100):
        """
        Inicializar el gestor de undo/redo.

        Args:
            max_history: Número máximo de estados a mantener
        """
        self.max_history = max_history
        self.undo_stack: List[Dict[str, Any]] = []
        self.redo_stack: List[Dict[str, Any]] = []

    def save_state(
        self,
        content: str,
        operation: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Guardar un estado para poder deshacerlo.

        Args:
            content: Contenido del estado
            operation: Tipo de operación realizada
            metadata: Metadatos adicionales

        Returns:
            ID del estado guardado
        """
        state_id = str(uuid.uuid4())
        state = {
            "id": state_id,
            "content": content,
            "operation": operation,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.undo_stack.append(state)
        
        # Limitar tamaño del stack
        if len(self.undo_stack) > self.max_history:
            self.undo_stack = self.undo_stack[-self.max_history:]
        
        # Limpiar redo stack cuando se hace una nueva operación
        self.redo_stack.clear()
        
        logger.debug(f"Estado guardado: {state_id} ({operation})")
        return state_id

    def undo(self, current_content: str) -> Optional[Dict[str, Any]]:
        """
        Deshacer la última operación.

        Args:
            current_content: Contenido actual

        Returns:
            Estado anterior o None si no hay nada que deshacer
        """
        if not self.undo_stack:
            return None
        
        # Guardar estado actual en redo stack
        current_state = {
            "content": current_content,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.redo_stack.append(current_state)
        
        # Obtener estado anterior
        previous_state = self.undo_stack.pop()
        
        logger.info(f"Deshaciendo operación: {previous_state.get('operation')}")
        return previous_state

    def redo(self, current_content: str) -> Optional[Dict[str, Any]]:
        """
        Rehacer la última operación deshecha.

        Args:
            current_content: Contenido actual

        Returns:
            Estado siguiente o None si no hay nada que rehacer
        """
        if not self.redo_stack:
            return None
        
        # Guardar estado actual en undo stack
        current_state = {
            "content": current_content,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.undo_stack.append(current_state)
        
        # Obtener estado siguiente
        next_state = self.redo_stack.pop()
        
        logger.info("Rehaciendo operación")
        return next_state

    def can_undo(self) -> bool:
        """Verificar si se puede deshacer"""
        return len(self.undo_stack) > 0

    def can_redo(self) -> bool:
        """Verificar si se puede rehacer"""
        return len(self.redo_stack) > 0

    def get_history(self, limit: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """
        Obtener historial de undo/redo.

        Args:
            limit: Número máximo de estados a retornar

        Returns:
            Diccionario con historial de undo y redo
        """
        return {
            "undo": list(self.undo_stack[-limit:])[::-1],
            "redo": list(self.redo_stack[-limit:])[::-1]
        }

    def clear(self):
        """Limpiar todo el historial"""
        self.undo_stack.clear()
        self.redo_stack.clear()
        logger.info("Historial de undo/redo limpiado")

    def get_state_count(self) -> Dict[str, int]:
        """Obtener conteo de estados"""
        return {
            "undo_count": len(self.undo_stack),
            "redo_count": len(self.redo_stack),
            "total": len(self.undo_stack) + len(self.redo_stack)
        }






