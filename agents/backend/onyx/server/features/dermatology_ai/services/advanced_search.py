"""
Sistema de búsqueda avanzada
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class SearchOperator(str, Enum):
    """Operadores de búsqueda"""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    IN = "in"
    BETWEEN = "between"


@dataclass
class SearchFilter:
    """Filtro de búsqueda"""
    field: str
    operator: SearchOperator
    value: Any


@dataclass
class SortOption:
    """Opción de ordenamiento"""
    field: str
    direction: str = "asc"  # "asc" or "desc"


class AdvancedSearchEngine:
    """Motor de búsqueda avanzada"""
    
    def __init__(self):
        """Inicializa el motor de búsqueda"""
        pass
    
    def search_analyses(self, analyses: List[Dict], filters: List[SearchFilter],
                       sort: Optional[SortOption] = None,
                       limit: Optional[int] = None) -> List[Dict]:
        """
        Busca en análisis
        
        Args:
            analyses: Lista de análisis
            filters: Lista de filtros
            sort: Opción de ordenamiento
            limit: Límite de resultados
            
        Returns:
            Lista de análisis filtrados
        """
        results = analyses
        
        # Aplicar filtros
        for filter_obj in filters:
            results = self._apply_filter(results, filter_obj)
        
        # Aplicar ordenamiento
        if sort:
            results = self._apply_sort(results, sort)
        
        # Aplicar límite
        if limit:
            results = results[:limit]
        
        return results
    
    def _apply_filter(self, items: List[Dict], filter_obj: SearchFilter) -> List[Dict]:
        """Aplica un filtro"""
        filtered = []
        
        for item in items:
            value = self._get_nested_value(item, filter_obj.field)
            
            if self._matches_filter(value, filter_obj.operator, filter_obj.value):
                filtered.append(item)
        
        return filtered
    
    def _matches_filter(self, value: Any, operator: SearchOperator,
                       filter_value: Any) -> bool:
        """Verifica si un valor coincide con el filtro"""
        if operator == SearchOperator.EQUALS:
            return value == filter_value
        elif operator == SearchOperator.NOT_EQUALS:
            return value != filter_value
        elif operator == SearchOperator.GREATER_THAN:
            return value > filter_value
        elif operator == SearchOperator.LESS_THAN:
            return value < filter_value
        elif operator == SearchOperator.CONTAINS:
            return filter_value in str(value)
        elif operator == SearchOperator.STARTS_WITH:
            return str(value).startswith(str(filter_value))
        elif operator == SearchOperator.ENDS_WITH:
            return str(value).endswith(str(filter_value))
        elif operator == SearchOperator.IN:
            return value in filter_value
        elif operator == SearchOperator.BETWEEN:
            return filter_value[0] <= value <= filter_value[1]
        
        return False
    
    def _apply_sort(self, items: List[Dict], sort: SortOption) -> List[Dict]:
        """Aplica ordenamiento"""
        reverse = sort.direction == "desc"
        
        try:
            return sorted(
                items,
                key=lambda x: self._get_nested_value(x, sort.field),
                reverse=reverse
            )
        except Exception:
            return items
    
    def _get_nested_value(self, item: Dict, field: str) -> Any:
        """Obtiene valor anidado de un diccionario"""
        keys = field.split(".")
        value = item
        
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None
        
        return value
    
    def build_search_query(self, query_params: Dict) -> tuple:
        """
        Construye query de búsqueda desde parámetros
        
        Args:
            query_params: Parámetros de búsqueda
            
        Returns:
            Tupla (filters, sort, limit)
        """
        filters = []
        
        # Construir filtros
        for field, value in query_params.items():
            if field in ["sort", "limit", "page"]:
                continue
            
            operator = SearchOperator.EQUALS
            if "__" in field:
                field, op = field.split("__", 1)
                try:
                    operator = SearchOperator(op)
                except ValueError:
                    operator = SearchOperator.EQUALS
            
            filters.append(SearchFilter(
                field=field,
                operator=operator,
                value=value
            ))
        
        # Ordenamiento
        sort = None
        if "sort" in query_params:
            sort_field = query_params["sort"]
            direction = "desc" if sort_field.startswith("-") else "asc"
            field = sort_field.lstrip("-")
            sort = SortOption(field=field, direction=direction)
        
        # Límite
        limit = query_params.get("limit")
        if limit:
            limit = int(limit)
        
        return filters, sort, limit






