"""
Base Generator - Clase base para generadores
=============================================

Clase abstracta base que define la interfaz común para todos los generadores.
Optimizado siguiendo mejores prácticas de Python y FastAPI.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


def _validate_name(name: str) -> None:
    """
    Validar nombre del generador (función pura).
    
    Args:
        name: Nombre del generador
        
    Raises:
        ValueError: Si el nombre es inválido
    """
    if not name or not name.strip():
        raise ValueError("Generator name cannot be empty")


def _validate_target_dir(target_dir: Path) -> None:
    """
    Validar directorio destino (función pura).
    
    Args:
        target_dir: Directorio destino
        
    Raises:
        ValueError: Si el directorio es inválido
        TypeError: Si no es un Path
    """
    if target_dir is None:
        raise ValueError("target_dir cannot be None")
    
    if not isinstance(target_dir, Path):
        raise TypeError("target_dir must be a Path object")


def _validate_keywords(keywords: Dict[str, Any]) -> None:
    """
    Validar keywords (función pura).
    
    Args:
        keywords: Keywords extraídos
        
    Raises:
        ValueError: Si keywords es inválido
        TypeError: Si no es un diccionario
    """
    if not isinstance(keywords, dict):
        raise TypeError("keywords must be a dictionary")


def _validate_project_info(project_info: Dict[str, Any]) -> None:
    """
    Validar project_info (función pura).
    
    Args:
        project_info: Información del proyecto
        
    Raises:
        ValueError: Si project_info es inválido
        TypeError: Si no es un diccionario
    """
    if not isinstance(project_info, dict):
        raise TypeError("project_info must be a dictionary")


def _ensure_directory_exists(target_dir: Path) -> Path:
    """
    Asegurar que el directorio existe (función pura).
    
    Args:
        target_dir: Directorio a crear
        
    Returns:
        Path del directorio creado
    """
    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
    
    return target_dir


class BaseGenerator(ABC):
    """
    Clase base abstracta para todos los generadores.
    
    Define la interfaz común y proporciona utilidades compartidas.
    Implementa el patrón Template Method para consistencia.
    Optimizado con funciones puras y mejor manejo de errores.
    """
    
    def __init__(self, name: str, description: str = "") -> None:
        """
        Inicializar generador base.
        
        Args:
            name: Nombre del generador
            description: Descripción del generador
            
        Raises:
            ValueError: Si el nombre es inválido
        """
        _validate_name(name)
        
        self.name = name.strip()
        self.description = description
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
    
    @abstractmethod
    def generate(
        self,
        target_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any]
    ) -> None:
        """
        Generar código (método abstracto).
        
        Args:
            target_dir: Directorio destino
            keywords: Keywords extraídos del proyecto
            project_info: Información del proyecto
            
        Raises:
            NotImplementedError: Debe ser implementado por subclases
        """
        raise NotImplementedError("Subclasses must implement generate method")
    
    def validate_inputs(
        self,
        target_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any]
    ) -> bool:
        """
        Validar inputs antes de generar.
        
        Args:
            target_dir: Directorio destino
            keywords: Keywords extraídos
            project_info: Información del proyecto
            
        Returns:
            True si los inputs son válidos
            
        Raises:
            ValueError: Si los inputs son inválidos
            TypeError: Si los tipos son incorrectos
        """
        _validate_target_dir(target_dir)
        _validate_keywords(keywords)
        _validate_project_info(project_info)
        
        return True
    
    def ensure_directory(self, target_dir: Path) -> Path:
        """
        Asegurar que el directorio existe.
        
        Args:
            target_dir: Directorio a crear
            
        Returns:
            Path del directorio creado
        """
        return _ensure_directory_exists(target_dir)
    
    def should_generate(self, keywords: Dict[str, Any]) -> bool:
        """
        Determinar si se debe generar este componente.
        
        Args:
            keywords: Keywords extraídos
            
        Returns:
            True si se debe generar
        """
        return True
    
    def generate_with_validation(
        self,
        target_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any]
    ) -> None:
        """
        Generar con validación previa.
        
        Args:
            target_dir: Directorio destino
            keywords: Keywords extraídos
            project_info: Información del proyecto
            
        Raises:
            ValueError: Si los inputs son inválidos
            TypeError: Si los tipos son incorrectos
            Exception: Si hay error durante la generación
        """
        self.validate_inputs(target_dir, keywords, project_info)
        
        if not self.should_generate(keywords):
            self.logger.info(f"Skipping {self.name} generation (not required)")
            return
        
        target_dir = self.ensure_directory(target_dir)
        
        try:
            self.logger.info(f"Generating {self.name}...")
            self.generate(target_dir, keywords, project_info)
            self.logger.info(f"Successfully generated {self.name}")
        except Exception as e:
            self.logger.error(f"Error generating {self.name}: {e}", exc_info=True)
            raise
