"""
Search Engine
=============

Motor de búsqueda avanzado con fuzzy matching.
"""

import re
from typing import List, Dict, Any, Optional, Callable
from difflib import SequenceMatcher
from datetime import datetime


class SearchEngine:
    """Motor de búsqueda avanzado."""
    
    def __init__(self, min_similarity: float = 0.6):
        """
        Inicializar motor de búsqueda.
        
        Args:
            min_similarity: Similaridad mínima (0-1)
        """
        self.min_similarity = min_similarity
    
    def fuzzy_search(
        self,
        query: str,
        items: List[Dict[str, Any]],
        fields: List[str],
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Búsqueda difusa.
        
        Args:
            query: Consulta
            items: Items a buscar
            fields: Campos a buscar
            limit: Límite de resultados
        
        Returns:
            Resultados ordenados por relevancia
        """
        query_lower = query.lower()
        results = []
        
        for item in items:
            max_score = 0.0
            
            for field in fields:
                if field in item:
                    value = str(item[field]).lower()
                    score = self._calculate_similarity(query_lower, value)
                    max_score = max(max_score, score)
            
            if max_score >= self.min_similarity:
                item_copy = item.copy()
                item_copy["_score"] = max_score
                results.append(item_copy)
        
        # Ordenar por score
        results.sort(key=lambda x: x["_score"], reverse=True)
        
        if limit:
            results = results[:limit]
        
        return results
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """
        Calcular similaridad entre strings.
        
        Args:
            str1: String 1
            str2: String 2
        
        Returns:
            Similaridad (0-1)
        """
        # Exact match
        if str1 == str2:
            return 1.0
        
        # Contains
        if str1 in str2 or str2 in str1:
            return 0.9
        
        # Sequence matcher
        return SequenceMatcher(None, str1, str2).ratio()
    
    def regex_search(
        self,
        pattern: str,
        items: List[Dict[str, Any]],
        fields: List[str],
        case_sensitive: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Búsqueda con regex.
        
        Args:
            pattern: Patrón regex
            items: Items a buscar
            fields: Campos a buscar
            case_sensitive: Si es case sensitive
        
        Returns:
            Resultados
        """
        flags = 0 if case_sensitive else re.IGNORECASE
        regex = re.compile(pattern, flags)
        results = []
        
        for item in items:
            for field in fields:
                if field in item:
                    value = str(item[field])
                    if regex.search(value):
                        results.append(item)
                        break
        
        return results
    
    def filter_by_date_range(
        self,
        items: List[Dict[str, Any]],
        date_field: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Filtrar por rango de fechas.
        
        Args:
            items: Items a filtrar
            date_field: Campo de fecha
            start_date: Fecha inicio
            end_date: Fecha fin
        
        Returns:
            Items filtrados
        """
        results = []
        
        for item in items:
            if date_field not in item:
                continue
            
            item_date = item[date_field]
            if isinstance(item_date, str):
                item_date = datetime.fromisoformat(item_date)
            
            if start_date and item_date < start_date:
                continue
            
            if end_date and item_date > end_date:
                continue
            
            results.append(item)
        
        return results
    
    def multi_field_search(
        self,
        query: Dict[str, Any],
        items: List[Dict[str, Any]],
        operators: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Búsqueda multi-campo.
        
        Args:
            query: Consulta con campos
            items: Items a buscar
            operators: Operadores por campo (AND/OR)
        
        Returns:
            Resultados
        """
        operators = operators or {}
        results = items.copy()
        
        for field, value in query.items():
            operator = operators.get(field, "AND")
            field_results = []
            
            for item in results:
                if field in item:
                    item_value = str(item[field]).lower()
                    search_value = str(value).lower()
                    
                    if search_value in item_value:
                        field_results.append(item)
            
            if operator == "AND":
                results = field_results
            else:  # OR
                results.extend(field_results)
        
        # Eliminar duplicados
        seen = set()
        unique_results = []
        for item in results:
            item_id = id(item)
            if item_id not in seen:
                seen.add(item_id)
                unique_results.append(item)
        
        return unique_results




