"""
Advanced Search
==============

Búsqueda avanzada que compone parser y filter builder.
"""

from typing import List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from ...core.base.service_base import BaseService
from ...database.models import Manual
from .query_parser import QueryParser
from .filter_builder import FilterBuilder


class AdvancedSearch(BaseService):
    """Utilidades de búsqueda avanzada."""
    
    def __init__(self, db: AsyncSession):
        """
        Inicializar búsqueda avanzada.
        
        Args:
            db: Sesión de base de datos
        """
        super().__init__(logger_name=__name__)
        self.db = db
        self.query_parser = QueryParser()
        self.filter_builder = FilterBuilder()
    
    def parse_search_query(self, query: str) -> Dict[str, Any]:
        """
        Parsear query de búsqueda avanzada.
        
        Args:
            query: Query de búsqueda
        
        Returns:
            Diccionario con filtros parseados
        """
        return self.query_parser.parse(query)
    
    async def search(
        self,
        query: str,
        limit: int = 20,
        offset: int = 0
    ) -> List[Manual]:
        """
        Realizar búsqueda avanzada.
        
        Args:
            query: Query de búsqueda
            limit: Límite de resultados
            offset: Offset para paginación
        
        Returns:
            Lista de manuales encontrados
        """
        try:
            filters = self.parse_search_query(query)
            
            db_query = select(Manual).where(Manual.is_public == True)
            db_query = self.filter_builder.build_filters(db_query, filters)
            db_query = self.filter_builder.build_text_search(db_query, filters.get("text", []))
            
            db_query = db_query.order_by(
                desc(Manual.created_at)
            ).limit(limit).offset(offset)
            
            result = await self.db.execute(db_query)
            return list(result.scalars().all())
        
        except Exception as e:
            self.log_error(f"Error en búsqueda avanzada: {str(e)}")
            return []

