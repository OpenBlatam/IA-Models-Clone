"""
Optional Imports - Manejo de imports opcionales
===============================================
"""

import logging
from typing import Optional, Any, Callable
from functools import wraps

logger = logging.getLogger(__name__)


def optional_import(module_name: str, package_name: Optional[str] = None, default: Any = None):
    """
    Intenta importar un módulo opcional.
    
    Args:
        module_name: Nombre del módulo a importar
        package_name: Nombre del paquete para mensaje de error (opcional)
        default: Valor por defecto si falla
        
    Returns:
        Módulo importado o default
    """
    try:
        return __import__(module_name)
    except ImportError:
        pkg = package_name or module_name
        logger.debug(f"{pkg} no disponible. Instalar con: pip install {pkg}")
        return default


def check_imports(*imports: tuple[str, str]) -> dict[str, bool]:
    """
    Verifica múltiples imports opcionales.
    
    Args:
        *imports: Tuplas de (module_name, package_name)
        
    Returns:
        Dict con estado de cada import
    """
    results = {}
    for module_name, package_name in imports:
        try:
            __import__(module_name)
            results[module_name] = True
        except ImportError:
            logger.debug(f"{package_name} no disponible")
            results[module_name] = False
    return results


# Common optional imports
def get_chromadb():
    """Obtiene chromadb si está disponible"""
    return optional_import("chromadb", "chromadb")


def get_sentence_transformers():
    """Obtiene sentence_transformers si está disponible"""
    return optional_import("sentence_transformers", "sentence-transformers")


def get_openai():
    """Obtiene openai si está disponible"""
    return optional_import("openai", "openai")


def get_anthropic():
    """Obtiene anthropic si está disponible"""
    return optional_import("anthropic", "anthropic")


def get_pymupdf():
    """Obtiene PyMuPDF (fitz) si está disponible"""
    return optional_import("fitz", "pymupdf")


def get_pdfplumber():
    """Obtiene pdfplumber si está disponible"""
    return optional_import("pdfplumber", "pdfplumber")


def get_pypdf2():
    """Obtiene PyPDF2 si está disponible"""
    return optional_import("PyPDF2", "PyPDF2")


def get_httpx():
    """Obtiene httpx si está disponible"""
    return optional_import("httpx", "httpx")


def get_requests():
    """Obtiene requests si está disponible"""
    return optional_import("requests", "requests")

