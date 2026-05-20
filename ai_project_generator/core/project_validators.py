"""
Project Validators - Validadores compartidos para proyectos
===========================================================

Funciones de validación reutilizables para proyectos, directorios y keywords.
Consolida validaciones duplicadas en un solo lugar.
"""

from pathlib import Path
from typing import Dict, Any


def validate_project_dir(project_dir: Path) -> None:
    """
    Valida el directorio del proyecto.
    
    Args:
        project_dir: Directorio del proyecto
        
    Raises:
        ValueError: Si el directorio es inválido
    """
    if project_dir is None:
        raise ValueError("project_dir cannot be None")


def validate_project_name(project_name: str) -> None:
    """
    Valida el nombre del proyecto.
    
    Args:
        project_name: Nombre del proyecto
        
    Raises:
        ValueError: Si el nombre es inválido
    """
    if not project_name or not project_name.strip():
        raise ValueError("project_name cannot be empty")


def validate_keywords(keywords: Dict[str, Any]) -> None:
    """
    Valida keywords del proyecto.
    
    Args:
        keywords: Keywords del proyecto
        
    Raises:
        ValueError: Si keywords es inválido
        TypeError: Si no es un diccionario
    """
    if not isinstance(keywords, dict):
        raise TypeError("keywords must be a dictionary")


def validate_project_info(project_info: Dict[str, Any]) -> None:
    """
    Valida project_info.
    
    Args:
        project_info: Información del proyecto
        
    Raises:
        ValueError: Si project_info es inválido
        TypeError: Si no es un diccionario
    """
    if not isinstance(project_info, dict):
        raise TypeError("project_info must be a dictionary")
    
    if not project_info:
        raise ValueError("project_info cannot be empty")






