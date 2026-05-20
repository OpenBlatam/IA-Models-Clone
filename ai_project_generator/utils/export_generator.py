"""
Export Generator - Generador de Exportaciones
==============================================

Genera exportaciones de proyectos en diferentes formatos.
Optimizado siguiendo mejores prácticas de Python y FastAPI.
"""

import logging
import json
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, Any, Optional, Set
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


_EXCLUDED_PATTERNS: Set[str] = {
    '__pycache__',
    '.pyc',
    '.git',
    'node_modules',
    '.pytest_cache',
    '.coverage',
    'dist',
    'build',
    '.venv',
    'venv',
    'env',
    '.env',
    '.DS_Store',
}


def _should_exclude_file(file_path: Path) -> bool:
    """
    Verifica si un archivo debe ser excluido (función pura).
    
    Args:
        file_path: Ruta del archivo
        
    Returns:
        True si debe ser excluido, False en caso contrario
    """
    path_str = str(file_path)
    return any(pattern in path_str for pattern in _EXCLUDED_PATTERNS)


def _tar_filter(tarinfo: tarfile.TarInfo) -> Optional[tarfile.TarInfo]:
    """
    Filtra archivos innecesarios en TAR (función pura).
    
    Args:
        tarinfo: Información del archivo TAR
        
    Returns:
        TarInfo si debe incluirse, None si debe excluirse
    """
    if _should_exclude_file(Path(tarinfo.name)):
        return None
    return tarinfo


def _get_tar_mode(compression: str) -> str:
    """
    Obtiene el modo de TAR según la compresión (función pura).
    
    Args:
        compression: Tipo de compresión
        
    Returns:
        Modo de TAR
    """
    if compression == 'gz':
        return 'w:gz'
    if compression == 'bz2':
        return 'w:bz2'
    if compression == 'xz':
        return 'w:xz'
    return 'w'


def _get_tar_extension(compression: str) -> str:
    """
    Obtiene la extensión del archivo TAR (función pura).
    
    Args:
        compression: Tipo de compresión
        
    Returns:
        Extensión del archivo
    """
    if compression == 'gz':
        return '.tar.gz'
    if compression == 'bz2':
        return '.tar.bz2'
    if compression == 'xz':
        return '.tar.xz'
    return '.tar'


def _count_files(directory: Path) -> Dict[str, int]:
    """
    Cuenta archivos en un directorio (función pura).
    
    Args:
        directory: Directorio a contar
        
    Returns:
        Diccionario con estadísticas de archivos
    """
    if not directory.exists():
        return {"total": 0, "python": 0, "typescript": 0, "json": 0}
    
    total = 0
    python = 0
    typescript = 0
    json_files = 0
    
    for file_path in directory.rglob('*'):
        if file_path.is_file() and not _should_exclude_file(file_path):
            total += 1
            suffix = file_path.suffix.lower()
            if suffix == '.py':
                python += 1
            elif suffix in ['.ts', '.tsx']:
                typescript += 1
            elif suffix == '.json':
                json_files += 1
    
    return {
        "total": total,
        "python": python,
        "typescript": typescript,
        "json": json_files,
    }


def _write_file_safe(file_path: Path, content: str, encoding: str = "utf-8") -> None:
    """
    Escribir archivo de forma segura (función pura).
    
    Args:
        file_path: Ruta del archivo
        content: Contenido a escribir
        encoding: Codificación del archivo
        
    Raises:
        IOError: Si no se puede escribir el archivo
    """
    try:
        file_path.write_text(content, encoding=encoding)
    except IOError as e:
        logger.error(f"Failed to write file {file_path}: {e}")
        raise


class ExportGenerator:
    """
    Generador de exportaciones de proyectos.
    Optimizado con funciones puras y mejor manejo de errores.
    """
    
    def __init__(self) -> None:
        """Inicializa el generador de exportaciones."""
        pass
    
    async def export_to_zip(
        self,
        project_dir: Path,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Exporta el proyecto a un archivo ZIP.

        Args:
            project_dir: Directorio del proyecto
            output_path: Ruta de salida (opcional)

        Returns:
            Ruta del archivo ZIP generado
            
        Raises:
            ValueError: Si los parámetros son inválidos
            IOError: Si no se puede crear el archivo ZIP
        """
        if not project_dir:
            raise ValueError("project_dir cannot be None")
        
        if not project_dir.exists():
            raise ValueError(f"Project directory does not exist: {project_dir}")
        
        if output_path is None:
            output_path = project_dir.parent / f"{project_dir.name}.zip"
        
        try:
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in project_dir.rglob('*'):
                    if file_path.is_file() and not _should_exclude_file(file_path):
                        arcname = file_path.relative_to(project_dir)
                        zipf.write(file_path, arcname)
            
            logger.info(f"Project exported to ZIP: {output_path}")
            return output_path
        
        except (IOError, zipfile.BadZipFile) as e:
            logger.error(f"Error exporting to ZIP: {e}")
            raise IOError(f"Failed to create ZIP file: {e}") from e
    
    async def export_to_tar(
        self,
        project_dir: Path,
        output_path: Optional[Path] = None,
        compression: str = 'gz',
    ) -> Path:
        """
        Exporta el proyecto a un archivo TAR.

        Args:
            project_dir: Directorio del proyecto
            output_path: Ruta de salida (opcional)
            compression: Compresión ('gz', 'bz2', 'xz', o None)

        Returns:
            Ruta del archivo TAR generado
            
        Raises:
            ValueError: Si los parámetros son inválidos
            IOError: Si no se puede crear el archivo TAR
        """
        if not project_dir:
            raise ValueError("project_dir cannot be None")
        
        if not project_dir.exists():
            raise ValueError(f"Project directory does not exist: {project_dir}")
        
        if compression not in [None, 'gz', 'bz2', 'xz']:
            raise ValueError(f"Invalid compression type: {compression}")
        
        if output_path is None:
            ext = _get_tar_extension(compression or '')
            output_path = project_dir.parent / f"{project_dir.name}{ext}"
        
        try:
            mode = _get_tar_mode(compression or '')
            
            with tarfile.open(output_path, mode) as tarf:
                tarf.add(project_dir, arcname=project_dir.name, filter=_tar_filter)
            
            logger.info(f"Project exported to TAR: {output_path}")
            return output_path
        
        except (IOError, tarfile.TarError) as e:
            logger.error(f"Error exporting to TAR: {e}")
            raise IOError(f"Failed to create TAR file: {e}") from e
    
    async def export_metadata(
        self,
        project_dir: Path,
        project_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Exporta metadata del proyecto.

        Args:
            project_dir: Directorio del proyecto
            project_info: Información del proyecto

        Returns:
            Metadata del proyecto
            
        Raises:
            ValueError: Si los parámetros son inválidos
            IOError: Si no se puede escribir el archivo
        """
        if not project_dir:
            raise ValueError("project_dir cannot be None")
        
        if not project_info:
            raise ValueError("project_info cannot be empty")
        
        metadata = {
            "project_name": project_info.get("name"),
            "description": project_info.get("description"),
            "author": project_info.get("author"),
            "version": project_info.get("version"),
            "created_at": project_info.get("created_at"),
            "keywords": project_info.get("keywords", {}),
            "structure": {
                "backend": project_info.get("backend", {}),
                "frontend": project_info.get("frontend", {}),
            },
            "files": _count_files(project_dir),
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }
        
        # Guardar metadata
        metadata_path = project_dir / "project_metadata.json"
        try:
            _write_file_safe(
                metadata_path,
                json.dumps(metadata, indent=2, ensure_ascii=False)
            )
            logger.info(f"Metadata exported to: {metadata_path}")
        except IOError as e:
            logger.error(f"Failed to write metadata: {e}")
            raise
        
        return metadata
