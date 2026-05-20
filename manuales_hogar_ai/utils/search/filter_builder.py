"""
Filter Builder
=============

Constructor especializado de filtros SQL.
"""

from typing import Dict, Any, Optional
from datetime import datetime
from sqlalchemy import and_, or_
from sqlalchemy.orm import Query

from ...core.base.service_base import BaseService
from ...database.models import Manual


class FilterBuilder(BaseService):
    """Constructor de filtros SQL."""
    
    def __init__(self):
        """Inicializar constructor."""
        super().__init__(logger_name=__name__)
    
    def build_filters(
        self,
        query: Query,
        filters: Dict[str, Any]
    ) -> Query:
        """
        Construir filtros SQL desde diccionario.
        
        Args:
            query: Query SQLAlchemy
            filters: Diccionario de filtros
        
        Returns:
            Query con filtros aplicados
        """
        conditions = []
        
        if filters.get("category"):
            conditions.append(Manual.category == filters["category"])
        
        if filters.get("difficulty"):
            conditions.append(Manual.difficulty == filters["difficulty"])
        
        if filters.get("min_rating"):
            conditions.append(Manual.average_rating >= filters["min_rating"])
        
        if filters.get("max_rating"):
            conditions.append(Manual.average_rating <= filters["max_rating"])
        
        if filters.get("tags"):
            # Buscar en tags (asumiendo que tags es un campo JSON o texto)
            tag_conditions = [
                Manual.tags.contains(tag) for tag in filters["tags"]
            ]
            if tag_conditions:
                conditions.append(or_(*tag_conditions))
        
        if filters.get("date_from"):
            conditions.append(Manual.created_at >= filters["date_from"])
        
        if filters.get("date_to"):
            conditions.append(Manual.created_at <= filters["date_to"])
        
        if conditions:
            query = query.where(and_(*conditions))
        
        return query
    
    def build_text_search(
        self,
        query: Query,
        text_parts: list
    ) -> Query:
        """
        Construir búsqueda de texto.
        
        Args:
            query: Query SQLAlchemy
            text_parts: Lista de términos de búsqueda
        
        Returns:
            Query con búsqueda de texto aplicada
        """
        if not text_parts:
            return query
        
        text_conditions = []
        for term in text_parts:
            text_conditions.append(
                or_(
                    Manual.title.contains(term),
                    Manual.problem_description.contains(term),
                    Manual.content.contains(term)
                )
            )
        
        if text_conditions:
            query = query.where(and_(*text_conditions))
        
        return query

