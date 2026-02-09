"""
Compatibility Utilities
======================

Utilidades para compatibilidad entre versiones y sistemas.
"""

import sys
import platform
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def get_system_info() -> Dict[str, Any]:
    """
    Obtener información del sistema.
    
    Returns:
        Diccionario con información del sistema
    """
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version,
        "python_version_info": sys.version_info._asdict()
    }


def check_python_version(min_version: tuple = (3, 8)) -> bool:
    """
    Verificar versión de Python.
    
    Args:
        min_version: Versión mínima requerida (major, minor)
        
    Returns:
        True si la versión es suficiente
    """
    current = sys.version_info[:2]
    if current < min_version:
        logger.warning(
            f"Python {current[0]}.{current[1]} detected, "
            f"but {min_version[0]}.{min_version[1]}+ is recommended"
        )
        return False
    return True


def check_dependencies() -> Dict[str, bool]:
    """
    Verificar dependencias opcionales.
    
    Returns:
        Diccionario de {dependency: available}
    """
    dependencies = {
        "numba": False,
        "torch": False,
        "cv2": False,
        "rclpy": False,
    }
    
    # Verificar numba
    try:
        import numba
        dependencies["numba"] = True
    except ImportError:
        pass
    
    # Verificar torch
    try:
        import torch
        dependencies["torch"] = True
    except ImportError:
        pass
    
    # Verificar opencv
    try:
        import cv2
        dependencies["cv2"] = True
    except ImportError:
        pass
    
    # Verificar ROS2
    try:
        import rclpy
        dependencies["rclpy"] = True
    except ImportError:
        pass
    
    return dependencies


def get_optimal_num_threads() -> int:
    """
    Obtener número óptimo de threads.
    
    Returns:
        Número de threads recomendado
    """
    import os
    
    # Verificar variable de entorno
    env_threads = os.getenv("OMP_NUM_THREADS") or os.getenv("NUMBA_NUM_THREADS")
    if env_threads:
        try:
            return int(env_threads)
        except ValueError:
            pass
    
    # Usar número de CPUs disponibles
    try:
        import os
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        try:
            import multiprocessing
            return multiprocessing.cpu_count()
        except:
            return 1


def is_windows() -> bool:
    """Verificar si estamos en Windows."""
    return platform.system() == "Windows"


def is_linux() -> bool:
    """Verificar si estamos en Linux."""
    return platform.system() == "Linux"


def is_macos() -> bool:
    """Verificar si estamos en macOS."""
    return platform.system() == "Darwin"


def get_path_separator() -> str:
    """Obtener separador de rutas del sistema."""
    return "\\" if is_windows() else "/"


def normalize_path(path: str) -> str:
    """
    Normalizar ruta según el sistema.
    
    Args:
        path: Ruta a normalizar
        
    Returns:
        Ruta normalizada
    """
    if is_windows():
        return path.replace("/", "\\")
    else:
        return path.replace("\\", "/")


class FeatureFlags:
    """
    Gestor de feature flags.
    
    Permite habilitar/deshabilitar características dinámicamente.
    """
    
    def __init__(self):
        """Inicializar feature flags."""
        self.flags: Dict[str, bool] = {
            "use_numba": False,
            "use_gpu": False,
            "use_ros": False,
            "use_llm": False,
            "advanced_optimization": True,
            "caching": True,
        }
    
    def enable(self, flag: str) -> None:
        """Habilitar feature flag."""
        self.flags[flag] = True
        logger.info(f"Feature flag {flag} enabled")
    
    def disable(self, flag: str) -> None:
        """Deshabilitar feature flag."""
        self.flags[flag] = False
        logger.info(f"Feature flag {flag} disabled")
    
    def is_enabled(self, flag: str) -> bool:
        """Verificar si feature flag está habilitado."""
        return self.flags.get(flag, False)
    
    def set_all(self, flags: Dict[str, bool]) -> None:
        """Establecer múltiples feature flags."""
        self.flags.update(flags)
    
    def get_all(self) -> Dict[str, bool]:
        """Obtener todos los feature flags."""
        return self.flags.copy()


# Instancia global
_feature_flags: Optional[FeatureFlags] = None


def get_feature_flags() -> FeatureFlags:
    """Obtener instancia global de feature flags."""
    global _feature_flags
    if _feature_flags is None:
        _feature_flags = FeatureFlags()
    return _feature_flags






