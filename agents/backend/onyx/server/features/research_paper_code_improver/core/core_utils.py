"""
Core Utilities - Utilidades compartidas para módulos core
=========================================================
"""

import logging
from pathlib import Path
from typing import Optional
from functools import lru_cache

logger = logging.getLogger(__name__)


def get_logger(name: str) -> logging.Logger:
    """
    Obtiene logger consistente para un módulo.
    
    Args:
        name: Nombre del módulo (típicamente __name__)
        
    Returns:
        Logger configurado
    """
    return logging.getLogger(name)


def ensure_dir(path: str) -> Path:
    """
    Asegura que un directorio existe, creándolo si es necesario.
    
    Args:
        path: Ruta al directorio
        
    Returns:
        Path object del directorio
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


@lru_cache(maxsize=1)
def get_paper_storage(storage_dir: str = "data/papers"):
    """
    Factory function para obtener instancia singleton de PaperStorage.
    
    Args:
        storage_dir: Directorio de almacenamiento
        
    Returns:
        Instancia de PaperStorage
    """
    from .paper_storage import PaperStorage
    return PaperStorage(storage_dir=storage_dir)


def safe_file_operation(operation, *args, default=None, **kwargs):
    """
    Ejecuta operación de archivo de forma segura con manejo de errores.
    
    Args:
        operation: Función a ejecutar
        *args: Argumentos posicionales
        default: Valor por defecto si falla
        **kwargs: Argumentos nombrados
        
    Returns:
        Resultado de la operación o default si falla
    """
    try:
        return operation(*args, **kwargs)
    except Exception as e:
        logger.warning(f"Error en operación de archivo: {e}")
        return default

