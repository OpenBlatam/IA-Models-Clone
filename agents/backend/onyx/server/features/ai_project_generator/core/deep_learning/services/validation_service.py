"""
Validation Service - Servicio de validación (optimizado)
=========================================================

Servicio especializado para validar inputs y parámetros.
Encapsula toda la lógica de validación en un servicio independiente.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from ..utils.validators import (
    validate_generator_key,
    validate_project_path,
    get_target_directory
)

logger = logging.getLogger(__name__)


class ValidationService:
    """
    Servicio de validación (optimizado).
    
    Encapsula toda la lógica de validación de inputs y parámetros.
    Proporciona validaciones consistentes y reutilizables.
    """
    
    def __init__(self):
        """Inicializar servicio de validación"""
        self.logger = logging.getLogger(f"{__name__}.ValidationService")
        self._validation_errors: List[str] = []
    
    def validate_generation_request(
        self,
        project_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any]
    ) -> bool:
        """
        Validar request completo de generación (optimizado).
        
        Args:
            project_dir: Directorio del proyecto
            keywords: Keywords extraídos
            project_info: Información del proyecto
            
        Returns:
            True si es válido
            
        Raises:
            ValueError: Si la validación falla
        """
        self._validation_errors.clear()
        
        # Validar project_dir
        try:
            validate_project_path(project_dir)
        except ValueError as e:
            self._validation_errors.append(f"Invalid project_dir: {e}")
        
        # Validar keywords
        if not keywords:
            self._validation_errors.append("keywords cannot be empty")
        elif not isinstance(keywords, dict):
            self._validation_errors.append("keywords must be a dictionary")
        
        # Validar project_info
        if not project_info:
            self._validation_errors.append("project_info cannot be empty")
        elif not isinstance(project_info, dict):
            self._validation_errors.append("project_info must be a dictionary")
        
        # Si hay errores, lanzar excepción
        if self._validation_errors:
            error_msg = "; ".join(self._validation_errors)
            self.logger.error(f"Validation failed: {error_msg}")
            raise ValueError(f"Validation failed: {error_msg}")
        
        return True
    
    def validate_generator_key(self, generator_key: str) -> bool:
        """
        Validar clave de generador (optimizado).
        
        Args:
            generator_key: Clave del generador
            
        Returns:
            True si es válido
            
        Raises:
            ValueError: Si la validación falla
        """
        try:
            validate_generator_key(generator_key)
            return True
        except ValueError as e:
            self.logger.error(f"Invalid generator key: {e}")
            raise
    
    def get_target_directory(
        self,
        project_dir: Path,
        generator_key: str
    ) -> Optional[Path]:
        """
        Obtener directorio objetivo (optimizado).
        
        Args:
            project_dir: Directorio del proyecto
            generator_key: Clave del generador
            
        Returns:
            Directorio objetivo o None
        """
        try:
            return get_target_directory(project_dir, generator_key)
        except Exception as e:
            self.logger.warning(f"Error getting target directory: {e}")
            return None
    
    def get_validation_errors(self) -> List[str]:
        """
        Obtener errores de validación (optimizado).
        
        Returns:
            Lista de errores
        """
        return self._validation_errors.copy()

