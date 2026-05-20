"""
Validation Service - Servicio de validación
============================================

Servicio independiente para validación de proyectos.
"""

import logging
from typing import Dict, Any
from pathlib import Path

from ..utils.validator import ProjectValidator
from ..domain.models import ValidationResult

logger = logging.getLogger(__name__)


class ValidationService:
    """Servicio para validación de proyectos"""
    
    def __init__(self, validator: ProjectValidator = None):
        self.validator = validator or ProjectValidator()
    
    async def validate_project(self, project_path: str) -> Dict[str, Any]:
        """
        Valida un proyecto.
        
        Args:
            project_path: Ruta del proyecto
        
        Returns:
            Resultado de validación
        """
        try:
            path = Path(project_path)
            if not path.exists():
                return ValidationResult(
                    valid=False,
                    errors=[f"Project path does not exist: {project_path}"]
                ).dict()
            
            result = self.validator.validate_project(str(path))
            return ValidationResult(**result).dict()
        except Exception as e:
            logger.error(f"Error validating project: {e}", exc_info=True)
            return ValidationResult(
                valid=False,
                errors=[str(e)]
            ).dict()















