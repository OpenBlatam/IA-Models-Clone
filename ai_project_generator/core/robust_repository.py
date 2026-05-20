"""
Robust Repository - Repositorio robusto con validación y resiliencia
====================================================================

Repositorio base mejorado con validación robusta y manejo de errores avanzado.
"""

import logging
from abc import abstractmethod
from typing import Optional, Dict, Any, List
from datetime import datetime

from .base_repository import BaseRepository
from .exceptions import RepositoryError, ValidationError
from .validators import validate_project_name, validate_description

logger = logging.getLogger(__name__)


class RobustRepository(BaseRepository):
    """
    Repositorio robusto con:
    - Validación estricta
    - Sanitización de datos
    - Manejo de errores mejorado
    - Validación de integridad
    """
    
    def __init__(self, strict_validation: bool = True):
        """
        Args:
            strict_validation: Si usar validación estricta
        """
        self.strict_validation = strict_validation
    
    def _validate_project_data(self, data: Dict[str, Any]) -> None:
        """
        Valida datos de proyecto.
        
        Args:
            data: Datos a validar
        
        Raises:
            ValidationError: Si los datos no son válidos
        """
        if not data:
            raise ValidationError("Project data cannot be empty")
        
        # Validar descripción
        if "description" in data:
            if not validate_description(data["description"]):
                raise ValidationError(
                    "Description must be 10-2000 characters and contain at least 5 unique words"
                )
        
        # Validar nombre si está presente
        if "project_name" in data and data["project_name"]:
            if not validate_project_name(data["project_name"]):
                raise ValidationError(
                    "Project name must be 3-50 characters and contain only "
                    "letters, numbers, hyphens, and underscores"
                )
        
        # Validar autor
        if "author" in data:
            if not data["author"] or len(data["author"]) < 1:
                raise ValidationError("Author cannot be empty")
    
    def _sanitize_project_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitiza datos de proyecto.
        
        Args:
            data: Datos a sanitizar
        
        Returns:
            Datos sanitizados
        """
        sanitized = {}
        
        # Sanitizar descripción
        if "description" in data:
            sanitized["description"] = data["description"].strip()
        
        # Sanitizar nombre
        if "project_name" in data and data["project_name"]:
            sanitized["project_name"] = data["project_name"].strip().lower().replace(" ", "_")
        
        # Sanitizar autor
        if "author" in data:
            sanitized["author"] = data["author"].strip()
        
        # Copiar otros campos
        for key in ["version", "priority", "tags", "metadata"]:
            if key in data:
                sanitized[key] = data[key]
        
        return sanitized
    
    async def _create_impl(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Implementación de create con validación"""
        # Validar
        if self.strict_validation:
            self._validate_project_data(data)
        
        # Sanitizar
        data = self._sanitize_project_data(data)
        
        # Agregar timestamps
        data["created_at"] = datetime.now().isoformat()
        
        # Llamar implementación específica
        return await self._create_impl_specific(data)
    
    async def _update_impl(
        self,
        id: str,
        data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Implementación de update con validación"""
        # Validar
        if self.strict_validation and data:
            self._validate_project_data(data)
        
        # Sanitizar
        if data:
            data = self._sanitize_project_data(data)
            data["updated_at"] = datetime.now().isoformat()
        
        # Llamar implementación específica
        return await self._update_impl_specific(id, data)
    
    @abstractmethod
    async def _create_impl_specific(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Implementación específica de create"""
        pass
    
    @abstractmethod
    async def _update_impl_specific(
        self,
        id: str,
        data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Implementación específica de update"""
        pass
    
    async def _list_impl(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List con validación de parámetros"""
        # Validar y sanitizar filtros
        if filters:
            filters = self._sanitize_filters(filters)
        
        return await self._list_impl_specific(filters, limit, offset)
    
    def _sanitize_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitiza filtros"""
        sanitized = {}
        
        # Sanitizar status
        if "status" in filters and filters["status"]:
            valid_statuses = ["queued", "processing", "completed", "failed", "cancelled"]
            if filters["status"] in valid_statuses:
                sanitized["status"] = filters["status"]
        
        # Sanitizar author
        if "author" in filters and filters["author"]:
            sanitized["author"] = filters["author"].strip()
        
        # Sanitizar project_name
        if "project_name" in filters and filters["project_name"]:
            sanitized["project_name"] = filters["project_name"].strip()
        
        return sanitized
    
    @abstractmethod
    async def _list_impl_specific(
        self,
        filters: Optional[Dict[str, Any]],
        limit: int,
        offset: int
    ) -> List[Dict[str, Any]]:
        """Implementación específica de list"""
        pass

