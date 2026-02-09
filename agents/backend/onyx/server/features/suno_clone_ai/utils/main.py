"""
Utils Main - Funciones base y entry points del módulo de utilidades

Rol en el Ecosistema IA:
- Funciones helper reutilizables, sin dependencias
- Procesamiento de datos, validaciones, transformaciones
- Utilidades compartidas entre módulos
"""

from typing import Any, Dict, List, Optional
from .helpers import Helpers
from .validators import Validators
from .service import UtilService


# Instancia global del servicio
_util_service: Optional[UtilService] = None


def get_util_service() -> UtilService:
    """
    Obtiene la instancia global del servicio de utilidades.
    
    Returns:
        UtilService: Servicio de utilidades
    """
    global _util_service
    if _util_service is None:
        _util_service = UtilService()
    return _util_service


def safe_get(data: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Obtiene un valor de forma segura de un diccionario.
    
    Args:
        data: Diccionario
        key: Clave a buscar
        default: Valor por defecto
        
    Returns:
        Valor encontrado o default
    """
    return Helpers.safe_get(data, key, default)


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Divide una lista en chunks de tamaño fijo.
    Útil para procesamiento por lotes.
    
    Args:
        items: Lista a dividir
        chunk_size: Tamaño de cada chunk
        
    Returns:
        Lista de chunks
    """
    return Helpers.chunk_list(items, chunk_size)


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Aplana un diccionario anidado.
    
    Args:
        d: Diccionario anidado
        parent_key: Prefijo para claves
        sep: Separador entre niveles
        
    Returns:
        Diccionario aplanado
    """
    return Helpers.flatten_dict(d, parent_key, sep)


def is_email(email: str) -> bool:
    """
    Valida formato de email.
    
    Args:
        email: String a validar
        
    Returns:
        True si es email válido
    """
    return Validators.is_email(email)


def is_url(url: str) -> bool:
    """
    Valida formato de URL.
    
    Args:
        url: String a validar
        
    Returns:
        True si es URL válida
    """
    return Validators.is_url(url)


def validate_length(value: str, min_length: int = 0, max_length: Optional[int] = None) -> bool:
    """
    Valida longitud de string.
    
    Args:
        value: String a validar
        min_length: Longitud mínima
        max_length: Longitud máxima
        
    Returns:
        True si cumple con la longitud
    """
    return Validators.validate_length(value, min_length, max_length)

