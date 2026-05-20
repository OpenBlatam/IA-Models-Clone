"""
Generation Stats - Estadísticas de generación
==============================================

Sistema de tracking de estadísticas para generación de código.
Optimizado siguiendo mejores prácticas de Python y FastAPI.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List
from copy import deepcopy

logger = logging.getLogger(__name__)


def _calculate_success_rate(successful: int, total: int) -> float:
    """
    Calcula la tasa de éxito (función pura).
    
    Args:
        successful: Número de éxitos
        total: Número total
        
    Returns:
        Tasa de éxito (0.0 a 1.0)
    """
    if total == 0:
        return 0.0
    return successful / total


@dataclass(frozen=False)
class GenerationStats:
    """
    Estadísticas de generación.
    
    Trackea el éxito, fallos y omisiones de generadores.
    Optimizado con mejor manejo de datos.
    """
    total_generators: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    generators_run: List[str] = field(default_factory=list)
    errors: Dict[str, str] = field(default_factory=dict)
    
    def add_success(self, generator_key: str) -> None:
        """
        Agregar generador exitoso.
        
        Args:
            generator_key: Clave del generador
            
        Raises:
            ValueError: Si generator_key está vacío
        """
        if not generator_key:
            raise ValueError("generator_key cannot be empty")
        
        self.successful += 1
        self.total_generators += 1
        self.generators_run.append(generator_key)
        logger.debug(f"Generator '{generator_key}' succeeded")
    
    def add_failure(self, generator_key: str, error: str) -> None:
        """
        Agregar generador fallido.
        
        Args:
            generator_key: Clave del generador
            error: Mensaje de error
            
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        if not generator_key:
            raise ValueError("generator_key cannot be empty")
        
        if not error:
            raise ValueError("error cannot be empty")
        
        self.failed += 1
        self.total_generators += 1
        self.errors[generator_key] = error
        logger.warning(f"Generator '{generator_key}' failed: {error}")
    
    def add_skipped(self, generator_key: str) -> None:
        """
        Agregar generador omitido.
        
        Args:
            generator_key: Clave del generador
            
        Raises:
            ValueError: Si generator_key está vacío
        """
        if not generator_key:
            raise ValueError("generator_key cannot be empty")
        
        self.skipped += 1
        self.total_generators += 1
        logger.debug(f"Generator '{generator_key}' skipped")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Obtener resumen de estadísticas.
        
        Returns:
            Diccionario con resumen de estadísticas
        """
        success_rate = _calculate_success_rate(self.successful, self.total_generators)
        
        return {
            "total": self.total_generators,
            "successful": self.successful,
            "failed": self.failed,
            "skipped": self.skipped,
            "success_rate": success_rate,
            "generators_run": deepcopy(self.generators_run),
            "errors": deepcopy(self.errors)
        }
    
    def reset(self) -> None:
        """Resetear estadísticas."""
        self.total_generators = 0
        self.successful = 0
        self.failed = 0
        self.skipped = 0
        self.generators_run.clear()
        self.errors.clear()
        logger.debug("Generation stats reset")
