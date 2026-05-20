"""
File Helpers
============
Utilidades para manejo de archivos.
"""

import os
import json
from typing import Optional, Dict, Any
from pathlib import Path


def ensure_directory(path: str) -> None:
    """
    Asegurar que un directorio existe.
    
    Args:
        path: Ruta del directorio
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def read_json_file(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Leer archivo JSON.
    
    Args:
        file_path: Ruta del archivo
        
    Returns:
        Contenido del archivo o None si no existe
    """
    if not os.path.exists(file_path):
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def write_json_file(file_path: str, data: Dict[str, Any], ensure_dir: bool = True) -> bool:
    """
    Escribir archivo JSON.
    
    Args:
        file_path: Ruta del archivo
        data: Datos a escribir
        ensure_dir: Crear directorio si no existe
        
    Returns:
        True si se escribió correctamente
    """
    try:
        if ensure_dir:
            ensure_directory(os.path.dirname(file_path))
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return True
    except (IOError, TypeError):
        return False


def get_file_size(file_path: str) -> Optional[int]:
    """
    Obtener tamaño de archivo en bytes.
    
    Args:
        file_path: Ruta del archivo
        
    Returns:
        Tamaño en bytes o None si no existe
    """
    if not os.path.exists(file_path):
        return None
    
    return os.path.getsize(file_path)


def file_exists(file_path: str) -> bool:
    """
    Verificar si un archivo existe.
    
    Args:
        file_path: Ruta del archivo
        
    Returns:
        True si existe
    """
    return os.path.exists(file_path) and os.path.isfile(file_path)

