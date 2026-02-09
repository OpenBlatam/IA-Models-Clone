"""
Search Module Implementation
"""

import logging
from typing import Dict, Any, List, Optional
from modules.base import BaseModule, ModuleConfig, ModuleStatus

logger = logging.getLogger(__name__)


class SearchModule(BaseModule):
    """
    Módulo de búsqueda
    Puede funcionar como microservicio independiente
    """
    
    def __init__(self, config: ModuleConfig):
        super().__init__(config)
        self._search_engine = None
    
    async def _initialize(self):
        """Inicializa el motor de búsqueda"""
        try:
            from services.search_engine import SearchEngine
            
            self._search_engine = SearchEngine()
            logger.info("Search engine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize search engine: {e}")
            raise
    
    async def _shutdown(self):
        """Cierra el motor de búsqueda"""
        self._search_engine = None
        logger.info("Search engine shut down")
    
    async def search(
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
        if self.status != ModuleStatus.ACTIVE:
            raise RuntimeError(f"Module {self.name} is not active")
        
        try:
            results = await self._search_engine.search(
                query=query,
                filters=filters or {},
                limit=limit,
                offset=offset
            )
            return results
        except Exception as e:
            logger.error(f"Error searching: {e}", exc_info=True)
            raise















