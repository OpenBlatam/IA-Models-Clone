"""
Path Utils - Utilidades de Path Avanzadas
=========================================

Utilidades avanzadas para manipulación de rutas y paths.
"""

import logging
from typing import Union, List, Optional
from pathlib import Path
import os

logger = logging.getLogger(__name__)


def normalize_path(path: Union[str, Path]) -> Path:
    """
    Normalizar path (resolver .., ., etc.).
    
    Args:
        path: Path a normalizar
        
    Returns:
        Path normalizado
    """
    return Path(path).resolve()


def relative_path(path: Union[str, Path], base: Union[str, Path]) -> Path:
    """
    Obtener path relativo desde base.
    
    Args:
        path: Path objetivo
        base: Path base
        
    Returns:
        Path relativo
    """
    return Path(path).relative_to(Path(base))


def safe_relative_path(path: Union[str, Path], base: Union[str, Path]) -> Optional[Path]:
    """
    Obtener path relativo de forma segura (retorna None si no es relativo).
    
    Args:
        path: Path objetivo
        base: Path base
        
    Returns:
        Path relativo o None
    """
    try:
        return Path(path).relative_to(Path(base))
    except ValueError:
        return None


def join_paths(*parts: Union[str, Path]) -> Path:
    """
    Unir múltiples partes de path.
    
    Args:
        *parts: Partes del path
        
    Returns:
        Path unido
    """
    result = Path(parts[0]) if parts else Path()
    for part in parts[1:]:
        result = result / part
    return result


def ensure_extension(path: Union[str, Path], extension: str) -> Path:
    """
    Asegurar que path tenga extensión.
    
    Args:
        path: Path
        extension: Extensión (con o sin punto)
        
    Returns:
        Path con extensión
    """
    path = Path(path)
    if not extension.startswith('.'):
        extension = f'.{extension}'
    
    if path.suffix != extension:
        return path.with_suffix(extension)
    return path


def change_extension(path: Union[str, Path], new_extension: str) -> Path:
    """
    Cambiar extensión de path.
    
    Args:
        path: Path
        new_extension: Nueva extensión (con o sin punto)
        
    Returns:
        Path con nueva extensión
    """
    path = Path(path)
    if not new_extension.startswith('.'):
        new_extension = f'.{new_extension}'
    
    return path.with_suffix(new_extension)


def remove_extension(path: Union[str, Path]) -> Path:
    """
    Remover extensión de path.
    
    Args:
        path: Path
        
    Returns:
        Path sin extensión
    """
    return Path(path).with_suffix('')


def get_common_path(paths: List[Union[str, Path]]) -> Optional[Path]:
    """
    Obtener path común de múltiples paths.
    
    Args:
        paths: Lista de paths
        
    Returns:
        Path común o None
    """
    if not paths:
        return None
    
    paths = [Path(p).resolve() for p in paths]
    common_parts = []
    
    # Encontrar partes comunes
    for i, part in enumerate(paths[0].parts):
        if all(len(p.parts) > i and p.parts[i] == part for p in paths[1:]):
            common_parts.append(part)
        else:
            break
    
    if common_parts:
        return Path(*common_parts)
    
    return None


def expand_user(path: Union[str, Path]) -> Path:
    """
    Expandir ~ en path.
    
    Args:
        path: Path con ~
        
    Returns:
        Path expandido
    """
    return Path(path).expanduser()


def expand_vars(path: Union[str, Path]) -> Path:
    """
    Expandir variables de entorno en path.
    
    Args:
        path: Path con variables
        
    Returns:
        Path expandido
    """
    return Path(os.path.expandvars(str(path)))


def expand_all(path: Union[str, Path]) -> Path:
    """
    Expandir ~ y variables de entorno.
    
    Args:
        path: Path
        
    Returns:
        Path expandido
    """
    return expand_vars(expand_user(path))


def is_absolute(path: Union[str, Path]) -> bool:
    """
    Verificar si path es absoluto.
    
    Args:
        path: Path a verificar
        
    Returns:
        True si es absoluto
    """
    return Path(path).is_absolute()


def is_relative(path: Union[str, Path]) -> bool:
    """
    Verificar si path es relativo.
    
    Args:
        path: Path a verificar
        
    Returns:
        True si es relativo
    """
    return not Path(path).is_absolute()


def get_path_depth(path: Union[str, Path]) -> int:
    """
    Obtener profundidad de path.
    
    Args:
        path: Path
        
    Returns:
        Profundidad (número de niveles)
    """
    return len(Path(path).parts)


def get_path_components(path: Union[str, Path]) -> List[str]:
    """
    Obtener componentes de path.
    
    Args:
        path: Path
        
    Returns:
        Lista de componentes
    """
    return list(Path(path).parts)


def sanitize_filename(filename: str, replacement: str = "_") -> str:
    """
    Sanitizar nombre de archivo removiendo caracteres inválidos.
    
    Args:
        filename: Nombre de archivo
        replacement: Carácter de reemplazo
        
    Returns:
        Nombre sanitizado
    """
    # Caracteres inválidos en nombres de archivo
    invalid_chars = '<>:"/\\|?*'
    
    sanitized = filename
    for char in invalid_chars:
        sanitized = sanitized.replace(char, replacement)
    
    # Remover caracteres de control
    sanitized = ''.join(c for c in sanitized if ord(c) >= 32)
    
    # Limitar longitud
    max_length = 255
    if len(sanitized) > max_length:
        # Mantener extensión si existe
        if '.' in sanitized:
            name, ext = sanitized.rsplit('.', 1)
            max_name_length = max_length - len(ext) - 1
            sanitized = name[:max_name_length] + '.' + ext
        else:
            sanitized = sanitized[:max_length]
    
    return sanitized


def make_unique_path(path: Union[str, Path], max_attempts: int = 1000) -> Path:
    """
    Crear path único agregando número si existe.
    
    Args:
        path: Path base
        max_attempts: Intentos máximos
        
    Returns:
        Path único
    """
    path = Path(path)
    
    if not path.exists():
        return path
    
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    
    for i in range(1, max_attempts + 1):
        new_path = parent / f"{stem}_{i}{suffix}"
        if not new_path.exists():
            return new_path
    
    raise ValueError(f"Could not create unique path after {max_attempts} attempts")


def get_temp_path(prefix: str = "tmp", suffix: str = "", directory: Optional[Union[str, Path]] = None) -> Path:
    """
    Obtener path temporal único.
    
    Args:
        prefix: Prefijo
        suffix: Sufijo (extensión)
        directory: Directorio (usa temp por defecto)
        
    Returns:
        Path temporal único
    """
    import tempfile
    
    if directory:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
    else:
        directory = Path(tempfile.gettempdir())
    
    if suffix and not suffix.startswith('.'):
        suffix = f'.{suffix}'
    
    # Generar nombre único
    import uuid
    unique_id = str(uuid.uuid4())[:8]
    temp_path = directory / f"{prefix}_{unique_id}{suffix}"
    
    return temp_path


def is_subpath(path: Union[str, Path], parent: Union[str, Path]) -> bool:
    """
    Verificar si path es subpath de parent.
    
    Args:
        path: Path a verificar
        parent: Path padre
        
    Returns:
        True si es subpath
    """
    try:
        Path(path).relative_to(Path(parent))
        return True
    except ValueError:
        return False


def get_relative_depth(path: Union[str, Path], base: Union[str, Path]) -> int:
    """
    Obtener profundidad relativa desde base.
    
    Args:
        path: Path objetivo
        base: Path base
        
    Returns:
        Profundidad relativa
    """
    try:
        relative = Path(path).relative_to(Path(base))
        return len(relative.parts)
    except ValueError:
        return -1

