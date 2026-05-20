"""
Search Engine Service - Motor de búsqueda avanzado
==================================================

Sistema de búsqueda avanzado con filtros y ranking.
"""

import logging
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class SearchFilter:
    """Filtro de búsqueda"""
    field: str
    operator: str  # equals, contains, greater_than, less_than, in, between
    value: Any


@dataclass
class SearchResult:
    """Resultado de búsqueda"""
    id: str
    score: float
    data: Dict[str, Any]
    highlights: List[str] = None


class SearchEngineService:
    """Servicio de motor de búsqueda"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.index: Dict[str, List[Dict[str, Any]]] = {}  # type -> [items]
        logger.info("SearchEngineService initialized")
    
    def index_item(self, item_type: str, item: Dict[str, Any]):
        """Indexar item"""
        if item_type not in self.index:
            self.index[item_type] = []
        
        self.index[item_type].append(item)
        logger.debug(f"Item indexed: {item_type}")
    
    def search(
        self,
        query: str,
        item_type: Optional[str] = None,
        filters: Optional[List[SearchFilter]] = None,
        limit: int = 20,
        offset: int = 0
    ) -> List[SearchResult]:
        """Buscar items"""
        results = []
        
        # Obtener items a buscar
        items_to_search = []
        if item_type:
            items_to_search = self.index.get(item_type, [])
        else:
            for items in self.index.values():
                items_to_search.extend(items)
        
        # Aplicar filtros
        if filters:
            items_to_search = self._apply_filters(items_to_search, filters)
        
        # Buscar por query
        query_lower = query.lower()
        for item in items_to_search:
            score = self._calculate_relevance_score(item, query_lower)
            if score > 0:
                highlights = self._extract_highlights(item, query_lower)
                results.append(SearchResult(
                    id=item.get("id", ""),
                    score=score,
                    data=item,
                    highlights=highlights
                ))
        
        # Ordenar por score
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Paginación
        return results[offset:offset + limit]
    
    def _apply_filters(
        self,
        items: List[Dict[str, Any]],
        filters: List[SearchFilter]
    ) -> List[Dict[str, Any]]:
        """Aplicar filtros"""
        filtered = items
        
        for filter_obj in filters:
            filtered = [
                item for item in filtered
                if self._match_filter(item, filter_obj)
            ]
        
        return filtered
    
    def _match_filter(self, item: Dict[str, Any], filter_obj: SearchFilter) -> bool:
        """Verificar si item coincide con filtro"""
        field_value = item.get(filter_obj.field)
        
        if filter_obj.operator == "equals":
            return field_value == filter_obj.value
        elif filter_obj.operator == "contains":
            return str(filter_obj.value).lower() in str(field_value).lower()
        elif filter_obj.operator == "greater_than":
            return field_value > filter_obj.value
        elif filter_obj.operator == "less_than":
            return field_value < filter_obj.value
        elif filter_obj.operator == "in":
            return field_value in filter_obj.value
        elif filter_obj.operator == "between":
            return filter_obj.value[0] <= field_value <= filter_obj.value[1]
        
        return False
    
    def _calculate_relevance_score(self, item: Dict[str, Any], query: str) -> float:
        """Calcular score de relevancia"""
        score = 0.0
        item_text = json.dumps(item).lower()
        
        # Coincidencias exactas
        if query in item_text:
            score += 10.0
        
        # Coincidencias de palabras
        query_words = query.split()
        for word in query_words:
            if word in item_text:
                score += 1.0
        
        # Boost en campos importantes
        for field in ["title", "name", "description"]:
            if field in item:
                field_value = str(item[field]).lower()
                if query in field_value:
                    score += 5.0
        
        return score
    
    def _extract_highlights(self, item: Dict[str, Any], query: str) -> List[str]:
        """Extraer highlights"""
        highlights = []
        query_words = query.split()
        
        for key, value in item.items():
            if isinstance(value, str):
                value_lower = value.lower()
                for word in query_words:
                    if word in value_lower:
                        # Extraer contexto
                        idx = value_lower.find(word)
                        start = max(0, idx - 20)
                        end = min(len(value), idx + len(word) + 20)
                        highlight = value[start:end]
                        if highlight not in highlights:
                            highlights.append(highlight)
        
        return highlights[:5]  # Máximo 5 highlights

