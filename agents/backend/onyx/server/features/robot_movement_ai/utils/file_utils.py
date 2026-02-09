"""
File Utilities - Utilidades de archivos
========================================

Utilidades para trabajar con archivos y directorios.
"""

from pathlib import Path
from typing import Optional, List, Union
import os
import shutil
import hashlib


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Asegurar que un directorio existe, creándolo si es necesario.
    
    Args:
        path: Ruta del directorio
    
    Returns:
        Path del directorio
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def safe_remove(path: Union[str, Path]) -> bool:
    """
    Eliminar archivo o directorio de forma segura.
    
    Args:
        path: Ruta a eliminar
    
    Returns:
        True si se eliminó exitosamente
    """
    try:
        path_obj = Path(path)
        if path_obj.is_file():
            path_obj.unlink()
        elif path_obj.is_dir():
            shutil.rmtree(path_obj)
        return True
    except Exception:
        return False


def get_file_hash(
    file_path: Union[str, Path],
    algorithm: str = 'sha256',
    chunk_size: int = 8192
) -> Optional[str]:
    """
    Calcular hash de un archivo.
    
    Args:
        file_path: Ruta del archivo
        algorithm: Algoritmo de hash (md5, sha1, sha256)
        chunk_size: Tamaño de chunk para lectura
    
    Returns:
        Hash hexadecimal o None si falla
    """
    try:
        hash_obj = hashlib.new(algorithm)
        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    except Exception:
        return None


def get_file_size(file_path: Union[str, Path]) -> Optional[int]:
    """
    Obtener tamaño de archivo.
    
    Args:
        file_path: Ruta del archivo
    
    Returns:
        Tamaño en bytes o None si falla
    """
    try:
        return Path(file_path).stat().st_size
    except Exception:
        return None


def list_files(
    directory: Union[str, Path],
    pattern: Optional[str] = None,
    recursive: bool = False
) -> List[Path]:
    """
    Listar archivos en un directorio.
    
    Args:
        directory: Directorio
        pattern: Patrón glob opcional (ej: "*.py")
        recursive: Si True, busca recursivamente
    
    Returns:
        Lista de paths de archivos
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        return []
    
    if pattern:
        if recursive:
            return list(dir_path.rglob(pattern))
        else:
            return list(dir_path.glob(pattern))
    else:
        if recursive:
            return [p for p in dir_path.rglob('*') if p.is_file()]
        else:
            return [p for p in dir_path.iterdir() if p.is_file()]


def copy_file(
    source: Union[str, Path],
    destination: Union[str, Path],
    overwrite: bool = False
) -> bool:
    """
    Copiar archivo.
    
    Args:
        source: Archivo origen
        destination: Archivo destino
        overwrite: Si True, sobrescribe si existe
    
    Returns:
        True si se copió exitosamente
    """
    try:
        src = Path(source)
        dst = Path(destination)
        
        if not src.is_file():
            return False
        
        if dst.exists() and not overwrite:
            return False
        
        ensure_dir(dst.parent)
        shutil.copy2(src, dst)
        return True
    except Exception:
        return False


def move_file(
    source: Union[str, Path],
    destination: Union[str, Path],
    overwrite: bool = False
) -> bool:
    """
    Mover archivo.
    
    Args:
        source: Archivo origen
        destination: Archivo destino
        overwrite: Si True, sobrescribe si existe
    
    Returns:
        True si se movió exitosamente
    """
    try:
        src = Path(source)
        dst = Path(destination)
        
        if not src.is_file():
            return False
        
        if dst.exists() and not overwrite:
            return False
        
        ensure_dir(dst.parent)
        shutil.move(str(src), str(dst))
        return True
    except Exception:
        return False


def read_text_file(file_path: Union[str, Path], encoding: str = 'utf-8') -> Optional[str]:
    """
    Leer archivo de texto.
    
    Args:
        file_path: Ruta del archivo
        encoding: Codificación
    
    Returns:
        Contenido del archivo o None si falla
    """
    try:
        return Path(file_path).read_text(encoding=encoding)
    except Exception:
        return None


def write_text_file(
    file_path: Union[str, Path],
    content: str,
    encoding: str = 'utf-8',
    create_dirs: bool = True
) -> bool:
    """
    Escribir archivo de texto.
    
    Args:
        file_path: Ruta del archivo
        content: Contenido a escribir
        encoding: Codificación
        create_dirs: Si True, crea directorios si no existen
    
    Returns:
        True si se escribió exitosamente
    """
    try:
        path = Path(file_path)
        if create_dirs:
            ensure_dir(path.parent)
        path.write_text(content, encoding=encoding)
        return True
    except Exception:
        return False


def get_file_extension(file_path: Union[str, Path]) -> str:
    """
    Obtener extensión de archivo.
    
    Args:
        file_path: Ruta del archivo
    
    Returns:
        Extensión (sin punto)
    """
    return Path(file_path).suffix.lstrip('.')


def get_file_name(file_path: Union[str, Path], with_extension: bool = True) -> str:
    """
    Obtener nombre de archivo.
    
    Args:
        file_path: Ruta del archivo
        with_extension: Si True, incluye extensión
    
    Returns:
        Nombre del archivo
    """
    path = Path(file_path)
    if with_extension:
        return path.name
    else:
        return path.stem

