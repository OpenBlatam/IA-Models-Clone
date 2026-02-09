"""
Context Search Service - Servicio de búsqueda contextual
"""

from typing import List, Dict, Any
from .base import BaseContextSearch


class ContextSearchService:
    """Servicio para búsqueda contextual"""

    def __init__(self):
        """Inicializa el servicio de búsqueda contextual"""
        pass

    async def search(self, query: str, context: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """Busca en el contexto"""
        # Implementación básica de búsqueda
        results = []
        query_lower = query.lower()

        # Buscar en mensajes del contexto
        messages = context.get("messages", [])
        for msg in messages:
            content = str(msg).lower()
            if query_lower in content:
                results.append(msg)
                if len(results) >= limit:
                    break

        return results

