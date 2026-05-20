"""
File Utils - Utilidades de Archivos
===================================

Utilidades avanzadas para manejo de archivos y directorios.
"""

import logging
import shutil
import hashlib
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
from datetime import datetime
import os

logger = logging.getLogger(__name__)

# Intentar importar aiofiles para operaciones async
try:
    import aiofiles
    _has_aiofiles = True
except ImportError:
    _has_aiofiles = False


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Asegurar que un directorio existe.
    
    Args:
        path: Ruta del directorio
        
    Returns:
        Path del directorio
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_delete(path: Union[str, Path], missing_ok: bool = True) -> bool:
    """
    Eliminar archivo o directorio de forma segura.
    
    Args:
        path: Ruta a eliminar
        missing_ok: Si no lanzar error si no existe
        
    Returns:
        True si se eliminó
    """
    path = Path(path)
    
    if not path.exists():
        return missing_ok
    
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        return True
    except Exception as e:
        logger.error(f"Error deleting {path}: {e}")
        return False


def get_file_hash(
    filepath: Union[str, Path],
    algorithm: str = "sha256"
) -> Optional[str]:
    """
    Obtener hash de archivo.
    
    Args:
        filepath: Ruta del archivo
        algorithm: Algoritmo de hash (md5, sha1, sha256, sha512)
        
    Returns:
        Hash hexadecimal o None si falla
    """
    try:
        hash_obj = hashlib.new(algorithm)
        
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        
        return hash_obj.hexdigest()
    except Exception as e:
        logger.error(f"Error hashing file {filepath}: {e}")
        return None


async def get_file_hash_async(
    filepath: Union[str, Path],
    algorithm: str = "sha256"
) -> Optional[str]:
    """
    Obtener hash de archivo (async).
    
    Args:
        filepath: Ruta del archivo
        algorithm: Algoritmo de hash
        
    Returns:
        Hash hexadecimal o None si falla
    """
    if not _has_aiofiles:
        return get_file_hash(filepath, algorithm)
    
    try:
        hash_obj = hashlib.new(algorithm)
        
        async with aiofiles.open(filepath, 'rb') as f:
            while True:
                chunk = await f.read(4096)
                if not chunk:
                    break
                hash_obj.update(chunk)
        
        return hash_obj.hexdigest()
    except Exception as e:
        logger.error(f"Error hashing file {filepath}: {e}")
        return None


def get_file_size(filepath: Union[str, Path]) -> int:
    """
    Obtener tamaño de archivo.
    
    Args:
        filepath: Ruta del archivo
        
    Returns:
        Tamaño en bytes
    """
    try:
        return Path(filepath).stat().st_size
    except Exception:
        return 0


def get_directory_size(directory: Union[str, Path]) -> int:
    """
    Obtener tamaño total de directorio.
    
    Args:
        directory: Ruta del directorio
        
    Returns:
        Tamaño total en bytes
    """
    total = 0
    directory = Path(directory)
    
    try:
        for entry in directory.rglob('*'):
            if entry.is_file():
                try:
                    total += entry.stat().st_size
                except (OSError, PermissionError):
                    pass
    except Exception as e:
        logger.error(f"Error calculating directory size: {e}")
    
    return total


def find_files(
    directory: Union[str, Path],
    pattern: str = "*",
    recursive: bool = True
) -> List[Path]:
    """
    Buscar archivos por patrón.
    
    Args:
        directory: Directorio base
        pattern: Patrón de búsqueda (glob)
        recursive: Si buscar recursivamente
        
    Returns:
        Lista de archivos encontrados
    """
    directory = Path(directory)
    
    if recursive:
        return list(directory.rglob(pattern))
    else:
        return list(directory.glob(pattern))


def find_files_by_extension(
    directory: Union[str, Path],
    extension: str,
    recursive: bool = True
) -> List[Path]:
    """
    Buscar archivos por extensión.
    
    Args:
        directory: Directorio base
        extension: Extensión (con o sin punto)
        recursive: Si buscar recursivamente
        
    Returns:
        Lista de archivos encontrados
    """
    if not extension.startswith('.'):
        extension = f'.{extension}'
    
    pattern = f'*{extension}'
    return find_files(directory, pattern, recursive)


def copy_file_safe(
    source: Union[str, Path],
    destination: Union[str, Path],
    overwrite: bool = False
) -> bool:
    """
    Copiar archivo de forma segura.
    
    Args:
        source: Archivo origen
        destination: Archivo destino
        overwrite: Si sobrescribir si existe
        
    Returns:
        True si se copió
    """
    source = Path(source)
    destination = Path(destination)
    
    if not source.exists():
        logger.error(f"Source file does not exist: {source}")
        return False
    
    if destination.exists() and not overwrite:
        logger.warning(f"Destination file exists: {destination}")
        return False
    
    try:
        ensure_dir(destination.parent)
        shutil.copy2(source, destination)
        return True
    except Exception as e:
        logger.error(f"Error copying file: {e}")
        return False


def move_file_safe(
    source: Union[str, Path],
    destination: Union[str, Path],
    overwrite: bool = False
) -> bool:
    """
    Mover archivo de forma segura.
    
    Args:
        source: Archivo origen
        destination: Archivo destino
        overwrite: Si sobrescribir si existe
        
    Returns:
        True si se movió
    """
    source = Path(source)
    destination = Path(destination)
    
    if not source.exists():
        logger.error(f"Source file does not exist: {source}")
        return False
    
    if destination.exists() and not overwrite:
        logger.warning(f"Destination file exists: {destination}")
        return False
    
    try:
        ensure_dir(destination.parent)
        shutil.move(str(source), str(destination))
        return True
    except Exception as e:
        logger.error(f"Error moving file: {e}")
        return False


def get_file_info(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Obtener información completa de archivo.
    
    Args:
        filepath: Ruta del archivo
        
    Returns:
        Diccionario con información
    """
    path = Path(filepath)
    
    if not path.exists():
        return {}
    
    stat = path.stat()
    
    return {
        "path": str(path),
        "name": path.name,
        "stem": path.stem,
        "suffix": path.suffix,
        "size": stat.st_size,
        "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "accessed_at": datetime.fromtimestamp(stat.st_atime).isoformat(),
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
        "is_symlink": path.is_symlink(),
        "parent": str(path.parent)
    }


def clean_directory(
    directory: Union[str, Path],
    pattern: Optional[str] = None,
    older_than_days: Optional[int] = None
) -> int:
    """
    Limpiar directorio eliminando archivos según criterios.
    
    Args:
        directory: Directorio a limpiar
        pattern: Patrón de archivos a eliminar (opcional)
        older_than_days: Eliminar archivos más antiguos que N días (opcional)
        
    Returns:
        Número de archivos eliminados
    """
    directory = Path(directory)
    deleted = 0
    
    if not directory.exists():
        return 0
    
    cutoff_date = None
    if older_than_days:
        cutoff_date = datetime.now().timestamp() - (older_than_days * 86400)
    
    try:
        files = find_files(directory, pattern or "*", recursive=True) if pattern else directory.rglob("*")
        
        for file_path in files:
            if not file_path.is_file():
                continue
            
            # Verificar patrón
            if pattern and not file_path.match(pattern):
                continue
            
            # Verificar antigüedad
            if cutoff_date:
                if file_path.stat().st_mtime > cutoff_date:
                    continue
            
            try:
                file_path.unlink()
                deleted += 1
            except Exception as e:
                logger.warning(f"Could not delete {file_path}: {e}")
    
    except Exception as e:
        logger.error(f"Error cleaning directory: {e}")
    
    return deleted




