"""
Version Utilities
=================
Utilidades para manejo de versiones.
"""

import re
from typing import Optional, Tuple


def parse_version(version_string: str) -> Tuple[int, int, int, Optional[str]]:
    """
    Parsear string de versión.
    
    Args:
        version_string: String de versión (ej: "1.2.3" o "1.2.3-beta")
        
    Returns:
        Tupla con (major, minor, patch, prerelease)
    """
    pattern = r'^(\d+)\.(\d+)\.(\d+)(?:-(.+))?$'
    match = re.match(pattern, version_string)
    
    if not match:
        raise ValueError(f"Invalid version format: {version_string}")
    
    major = int(match.group(1))
    minor = int(match.group(2))
    patch = int(match.group(3))
    prerelease = match.group(4)
    
    return (major, minor, patch, prerelease)


def compare_versions(version1: str, version2: str) -> int:
    """
    Comparar dos versiones.
    
    Args:
        version1: Primera versión
        version2: Segunda versión
        
    Returns:
        -1 si version1 < version2, 0 si son iguales, 1 si version1 > version2
    """
    v1 = parse_version(version1)
    v2 = parse_version(version2)
    
    # Comparar major, minor, patch
    for i in range(3):
        if v1[i] < v2[i]:
            return -1
        elif v1[i] > v2[i]:
            return 1
    
    # Si hay prerelease, la versión con prerelease es menor
    if v1[3] and not v2[3]:
        return -1
    elif not v1[3] and v2[3]:
        return 1
    
    return 0


def is_compatible_version(version: str, min_version: str, max_version: Optional[str] = None) -> bool:
    """
    Verificar si una versión es compatible con un rango.
    
    Args:
        version: Versión a verificar
        min_version: Versión mínima requerida
        max_version: Versión máxima permitida (opcional)
        
    Returns:
        True si es compatible
    """
    if compare_versions(version, min_version) < 0:
        return False
    
    if max_version and compare_versions(version, max_version) > 0:
        return False
    
    return True

