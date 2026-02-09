"""
Search Use Case
Caso de uso para búsqueda
"""

import logging
from typing import Dict, Any, Optional
from modules.search import SearchModule

logger = logging.getLogger(__name__)


class SearchUseCase:
    """
    Caso de uso para búsqueda
    Orquesta la lógica de negocio
    """
    
    def __init__(self, search_module: SearchModule):
        self.search_module = search_module
    
    async def search_songs(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Busca canciones
        
        Args:
            query: Consulta de búsqueda
            filters: Filtros adicionales
            limit: Límite de resultados
            offset: Offset para paginación
            
        Returns:
            Resultados de búsqueda
        """
        try:
            # Validar entrada
            if not query or len(query) < 1:
                raise ValueError("Query must not be empty")
            
            # Buscar usando el módulo
            results = await self.search_module.search(
                query=query,
                filters=filters,
                limit=limit,
                offset=offset
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error in search use case: {e}", exc_info=True)
            raise















