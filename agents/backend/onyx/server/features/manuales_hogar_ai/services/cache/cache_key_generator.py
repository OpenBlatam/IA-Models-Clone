"""
Cache Key Generator
==================

Generador especializado de claves de cache.
"""

import hashlib
import json
from typing import Dict, Any
from ...core.base.service_base import BaseService


class CacheKeyGenerator(BaseService):
    """Generador de claves de cache."""
    
    def __init__(self):
        """Inicializar generador."""
        super().__init__(logger_name=__name__)
    
    def generate(
        self,
        text: str,
        category: str,
        **kwargs
    ) -> str:
        """
        Generar clave de cache.
        
        Args:
            text: Texto base
            category: Categoría
            **kwargs: Otros parámetros
        
        Returns:
            Clave de cache (MD5 hash)
        """
        normalized_text = text.lower().strip()[:200]
        key_string = f"{normalized_text}:{category}:{json.dumps(kwargs, sort_keys=True)}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def generate_description_hash(self, text: str) -> str:
        """
        Generar hash de descripción.
        
        Args:
            text: Texto a hashear
        
        Returns:
            Hash MD5
        """
        normalized = text.lower().strip()[:200]
        return hashlib.md5(normalized.encode()).hexdigest()

