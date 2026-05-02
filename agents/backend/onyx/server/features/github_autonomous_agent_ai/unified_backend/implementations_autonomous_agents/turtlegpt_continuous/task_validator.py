"""
Task Validator Module
====================

Validación y normalización de tareas antes del procesamiento.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

from .models import AgentTask
from .constants import PRIORITY_MIN, PRIORITY_MAX
from .utils import validate_priority

logger = logging.getLogger(__name__)


class TaskValidator:
    """
    Validador de tareas.
    
    Proporciona validación y normalización de tareas
    antes de que sean procesadas.
    """
    
    def __init__(
        self,
        min_priority: int = PRIORITY_MIN,
        max_priority: int = PRIORITY_MAX,
        min_description_length: int = 1,
        max_description_length: int = 10000
    ):
        """
        Inicializar validador.
        
        Args:
            min_priority: Prioridad mínima permitida
            max_priority: Prioridad máxima permitida
            min_description_length: Longitud mínima de descripción
            max_description_length: Longitud máxima de descripción
        """
        self.min_priority = min_priority
        self.max_priority = max_priority
        self.min_description_length = min_description_length
        self.max_description_length = max_description_length
    
    def validate_task(
        self,
        description: str,
        priority: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> tuple[bool, Optional[str]]:
        """
        Validar una tarea.
        
        Args:
            description: Descripción de la tarea
            priority: Prioridad de la tarea
            metadata: Metadatos adicionales
            
        Returns:
            Tupla (es_válida, mensaje_error)
        """
        # Validar descripción
        if not description or not isinstance(description, str):
            return False, "Description must be a non-empty string"
        
        description = description.strip()
        if len(description) < self.min_description_length:
            return False, f"Description too short (minimum {self.min_description_length} characters)"
        
        if len(description) > self.max_description_length:
            return False, f"Description too long (maximum {self.max_description_length} characters)"
        
        # Validar prioridad
        try:
            validated_priority = validate_priority(priority, self.min_priority, self.max_priority)
        except ValueError as e:
            return False, str(e)
        
        # Validar metadata si existe
        if metadata is not None and not isinstance(metadata, dict):
            return False, "Metadata must be a dictionary or None"
        
        return True, None
    
    def normalize_task(
        self,
        description: str,
        priority: int,
        metadata: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Normalizar una tarea.
        
        Args:
            description: Descripción de la tarea
            priority: Prioridad de la tarea
            metadata: Metadatos adicionales
            task_id: ID de la tarea (se genera si no se proporciona)
            
        Returns:
            Dict con tarea normalizada
        """
        # Normalizar descripción
        description = description.strip() if description else ""
        
        # Normalizar prioridad
        validated_priority = validate_priority(priority, self.min_priority, self.max_priority)
        
        # Normalizar metadata
        normalized_metadata = metadata.copy() if metadata else {}
        normalized_metadata.setdefault("created_at", datetime.now().isoformat())
        
        # Generar ID si no se proporciona
        if not task_id:
            task_id = str(uuid.uuid4())
        
        return {
            "task_id": task_id,
            "description": description,
            "priority": validated_priority,
            "metadata": normalized_metadata
        }
    
    def validate_and_normalize(
        self,
        description: str,
        priority: int,
        metadata: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Validar y normalizar una tarea en un solo paso.
        
        Args:
            description: Descripción de la tarea
            priority: Prioridad de la tarea
            metadata: Metadatos adicionales
            task_id: ID de la tarea
            
        Returns:
            Tupla (tarea_normalizada, mensaje_error)
        """
        is_valid, error_message = self.validate_task(description, priority, metadata)
        
        if not is_valid:
            return {}, error_message
        
        normalized = self.normalize_task(description, priority, metadata, task_id)
        return normalized, None


def create_task_validator(
    min_priority: int = PRIORITY_MIN,
    max_priority: int = PRIORITY_MAX,
    min_description_length: int = 1,
    max_description_length: int = 10000
) -> TaskValidator:
    """
    Factory function para crear TaskValidator.
    
    Args:
        min_priority: Prioridad mínima permitida
        max_priority: Prioridad máxima permitida
        min_description_length: Longitud mínima de descripción
        max_description_length: Longitud máxima de descripción
        
    Returns:
        Instancia de TaskValidator
    """
    return TaskValidator(
        min_priority=min_priority,
        max_priority=max_priority,
        min_description_length=min_description_length,
        max_description_length=max_description_length
    )


