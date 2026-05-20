"""
Data Validator - Validador robusto de datos
==========================================

Validación robusta y estricta de datos con sanitización.
"""

import re
import logging
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime

logger = logging.getLogger(__name__)


class RobustValidator:
    """Validador robusto de datos"""
    
    @staticmethod
    def validate_and_sanitize_project_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valida y sanitiza datos de proyecto.
        
        Args:
            data: Datos a validar y sanitizar
        
        Returns:
            Datos validados y sanitizados
        
        Raises:
            ValidationError: Si los datos no son válidos
        """
        from .exceptions import ValidationError
        
        if not isinstance(data, dict):
            raise ValidationError("Data must be a dictionary")
        
        sanitized = {}
        
        # Descripción (requerida)
        if "description" not in data:
            raise ValidationError("Description is required")
        
        description = data["description"]
        if not isinstance(description, str):
            raise ValidationError("Description must be a string")
        
        description = description.strip()
        if len(description) < 10:
            raise ValidationError("Description must be at least 10 characters")
        if len(description) > 2000:
            raise ValidationError("Description must be at most 2000 characters")
        
        # Verificar palabras únicas
        words = set(description.lower().split())
        if len(words) < 5:
            raise ValidationError("Description must contain at least 5 unique words")
        
        sanitized["description"] = description
        
        # Nombre de proyecto (opcional pero validado si presente)
        if "project_name" in data and data["project_name"]:
            name = data["project_name"]
            if not isinstance(name, str):
                raise ValidationError("Project name must be a string")
            
            name = name.strip()
            if len(name) < 3:
                raise ValidationError("Project name must be at least 3 characters")
            if len(name) > 50:
                raise ValidationError("Project name must be at most 50 characters")
            
            if not re.match(r'^[a-zA-Z0-9_-]+$', name):
                raise ValidationError(
                    "Project name can only contain letters, numbers, hyphens, and underscores"
                )
            
            sanitized["project_name"] = name.lower().replace(" ", "_")
        
        # Autor (requerido)
        if "author" not in data:
            raise ValidationError("Author is required")
        
        author = data["author"]
        if not isinstance(author, str):
            raise ValidationError("Author must be a string")
        
        author = author.strip()
        if len(author) < 1:
            raise ValidationError("Author cannot be empty")
        if len(author) > 100:
            raise ValidationError("Author must be at most 100 characters")
        
        sanitized["author"] = author
        
        # Versión (opcional, validada si presente)
        if "version" in data and data["version"]:
            version = data["version"]
            if not isinstance(version, str):
                raise ValidationError("Version must be a string")
            
            # Validar formato semántico (simplificado)
            if not re.match(r'^\d+\.\d+\.\d+', version):
                raise ValidationError("Version must be in semantic format (e.g., 1.0.0)")
            
            sanitized["version"] = version
        else:
            sanitized["version"] = "1.0.0"
        
        # Prioridad (opcional, validada)
        if "priority" in data:
            priority = data["priority"]
            if not isinstance(priority, int):
                raise ValidationError("Priority must be an integer")
            if priority < -10 or priority > 10:
                raise ValidationError("Priority must be between -10 and 10")
            sanitized["priority"] = priority
        else:
            sanitized["priority"] = 0
        
        # Tags (opcional, validada)
        if "tags" in data:
            tags = data["tags"]
            if not isinstance(tags, list):
                raise ValidationError("Tags must be a list")
            
            sanitized_tags = []
            for tag in tags:
                if not isinstance(tag, str):
                    continue
                tag = tag.strip()
                if tag and len(tag) <= 50:
                    sanitized_tags.append(tag.lower())
            
            sanitized["tags"] = sanitized_tags[:10]  # Máximo 10 tags
        else:
            sanitized["tags"] = []
        
        # Metadata (opcional, validada)
        if "metadata" in data:
            metadata = data["metadata"]
            if not isinstance(metadata, dict):
                raise ValidationError("Metadata must be a dictionary")
            
            # Limitar tamaño de metadata
            if len(str(metadata)) > 10000:
                raise ValidationError("Metadata is too large (max 10KB)")
            
            sanitized["metadata"] = metadata
        else:
            sanitized["metadata"] = {}
        
        return sanitized
    
    @staticmethod
    def validate_filters(filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valida y sanitiza filtros.
        
        Args:
            filters: Filtros a validar
        
        Returns:
            Filtros validados
        """
        if not isinstance(filters, dict):
            return {}
        
        validated = {}
        
        # Status
        if "status" in filters:
            valid_statuses = ["queued", "processing", "completed", "failed", "cancelled"]
            if filters["status"] in valid_statuses:
                validated["status"] = filters["status"]
        
        # Author
        if "author" in filters and filters["author"]:
            author = str(filters["author"]).strip()
            if author:
                validated["author"] = author
        
        # Project name
        if "project_name" in filters and filters["project_name"]:
            name = str(filters["project_name"]).strip()
            if name:
                validated["project_name"] = name
        
        return validated
    
    @staticmethod
    def validate_pagination(limit: int, offset: int) -> tuple:
        """
        Valida y normaliza paginación.
        
        Args:
            limit: Límite
            offset: Offset
        
        Returns:
            (limit, offset) validados
        """
        if not isinstance(limit, int) or limit < 1:
            limit = 100
        if limit > 1000:
            limit = 1000
        
        if not isinstance(offset, int) or offset < 0:
            offset = 0
        
        return limit, offset


class ProjectDataModel(BaseModel):
    """Modelo Pydantic para validación robusta de datos de proyecto"""
    
    description: str = Field(..., min_length=10, max_length=2000)
    project_name: Optional[str] = Field(None, min_length=3, max_length=50)
    author: str = Field(..., min_length=1, max_length=100)
    version: str = Field(default="1.0.0", regex=r'^\d+\.\d+\.\d+')
    priority: int = Field(default=0, ge=-10, le=10)
    tags: List[str] = Field(default_factory=list, max_items=10)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('description')
    def validate_description(cls, v):
        words = set(v.lower().split())
        if len(words) < 5:
            raise ValueError("Description must contain at least 5 unique words")
        return v.strip()
    
    @validator('project_name')
    def validate_project_name(cls, v):
        if v:
            if not re.match(r'^[a-zA-Z0-9_-]+$', v):
                raise ValueError(
                    "Project name can only contain letters, numbers, hyphens, and underscores"
                )
            return v.strip().lower().replace(" ", "_")
        return v
    
    @validator('author')
    def validate_author(cls, v):
        return v.strip()
    
    @validator('tags')
    def validate_tags(cls, v):
        return [tag.strip().lower() for tag in v if tag and len(tag.strip()) <= 50][:10]















