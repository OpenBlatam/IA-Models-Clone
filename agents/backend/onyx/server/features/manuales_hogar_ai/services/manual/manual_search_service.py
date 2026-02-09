"""
Manual Search Service
====================

Servicio especializado para búsqueda de manuales.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, desc

from ...core.base.service_base import BaseService
from ...database.models import Manual
from ...utils.search.advanced_search import AdvancedSearch
from ...utils.validation.validators import Validators


class ManualSearchService(BaseService):
    """Servicio para búsqueda de manuales."""
    
    def __init__(self, db: AsyncSession):
        """
        Inicializar servicio de búsqueda.
        
        Args:
            db: Sesión de base de datos
        """
        super().__init__(logger_name=__name__)
        self.db = db
        self.search = AdvancedSearch(db)
        self.validator = Validators()
    
    async def search(
        self,
        category: Optional[str] = None,
        search_term: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
        difficulty: Optional[str] = None,
        min_rating: Optional[float] = None,
        max_rating: Optional[float] = None,
        tags: Optional[List[str]] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        advanced_query: Optional[str] = None
    ) -> List[Manual]:
        """
        Buscar manuales con filtros avanzados.
        
        Args:
            category: Filtrar por categoría
            search_term: Término de búsqueda
            limit: Límite de resultados
            offset: Offset para paginación
            difficulty: Filtrar por dificultad
            min_rating: Rating mínimo
            max_rating: Rating máximo
            tags: Lista de tags
            date_from: Fecha desde
            date_to: Fecha hasta
            advanced_query: Query avanzada con sintaxis especial
        
        Returns:
            Lista de manuales
        """
        try:
            query = select(Manual)
            
            if advanced_query:
                filters = self.search.parse_search_query(advanced_query)
                # Construir condiciones desde filtros parseados
                from sqlalchemy import and_, or_
                conditions = []
                
                if filters.get("category"):
                    conditions.append(Manual.category == filters["category"])
                if filters.get("difficulty"):
                    conditions.append(Manual.difficulty == filters["difficulty"])
                if filters.get("min_rating") is not None:
                    conditions.append(Manual.average_rating >= filters["min_rating"])
                if filters.get("max_rating") is not None:
                    conditions.append(Manual.average_rating <= filters["max_rating"])
                if filters.get("tags"):
                    tag_conditions = [Manual.tags.ilike(f"%{tag}%") for tag in filters["tags"]]
                    if tag_conditions:
                        conditions.append(or_(*tag_conditions))
                if filters.get("date_from"):
                    conditions.append(Manual.created_at >= filters["date_from"])
                if filters.get("date_to"):
                    conditions.append(Manual.created_at <= filters["date_to"])
                if filters.get("text"):
                    text_conditions = []
                    for term in filters["text"]:
                        text_conditions.append(Manual.problem_description.ilike(f"%{term}%"))
                        if Manual.title:
                            text_conditions.append(Manual.title.ilike(f"%{term}%"))
                    if text_conditions:
                        conditions.append(or_(*text_conditions))
            else:
                conditions = self._build_basic_conditions(
                    category, search_term, difficulty,
                    min_rating, max_rating, tags, date_from, date_to
                )
            
            conditions.append(Manual.is_public == True)
            
            if conditions:
                query = query.where(and_(*conditions))
            
            query = query.order_by(
                desc(Manual.average_rating),
                desc(Manual.view_count),
                desc(Manual.created_at)
            )
            
            query = query.limit(limit).offset(offset)
            
            result = await self.db.execute(query)
            return list(result.scalars().all())
        
        except Exception as e:
            self.log_error(f"Error buscando manuales: {str(e)}")
            return []
    
    def _build_basic_conditions(
        self,
        category: Optional[str],
        search_term: Optional[str],
        difficulty: Optional[str],
        min_rating: Optional[float],
        max_rating: Optional[float],
        tags: Optional[List[str]],
        date_from: Optional[datetime],
        date_to: Optional[datetime]
    ) -> List:
        """Construir condiciones básicas de búsqueda."""
        conditions = []
        
        if category:
            is_valid, _ = self.validator.validate_category(category)
            if is_valid:
                conditions.append(Manual.category == category.lower())
        
        if search_term:
            conditions.append(
                Manual.problem_description.ilike(f"%{search_term}%")
            )
        
        if difficulty:
            is_valid, _ = self.validator.validate_difficulty(difficulty)
            if is_valid:
                conditions.append(Manual.difficulty == difficulty)
        
        if min_rating is not None:
            conditions.append(Manual.average_rating >= min_rating)
        
        if max_rating is not None:
            conditions.append(Manual.average_rating <= max_rating)
        
        if tags:
            tag_conditions = [Manual.tags.ilike(f"%{tag}%") for tag in tags]
            if tag_conditions:
                conditions.append(or_(*tag_conditions))
        
        if date_from:
            conditions.append(Manual.created_at >= date_from)
        
        if date_to:
            conditions.append(Manual.created_at <= date_to)
        
        return conditions

