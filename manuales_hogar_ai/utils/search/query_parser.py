"""
Query Parser
============

Parser especializado para queries de búsqueda avanzada.
"""

import re
from typing import Dict, Any
from datetime import datetime
from ...core.base.service_base import BaseService


class QueryParser(BaseService):
    """Parser de queries de búsqueda avanzada."""
    
    PATTERNS = {
        "category": r'category:(\w+)',
        "difficulty": r'difficulty:(\w+)',
        "rating": r'rating:([><=]?\d+(?:\.\d+)?)',
        "tags": r'tags:(\w+)',
        "date": r'date:([><=]?\d{4}-\d{2}-\d{2})'
    }
    
    def __init__(self):
        """Inicializar parser."""
        super().__init__(logger_name=__name__)
    
    def parse(self, query: str) -> Dict[str, Any]:
        """
        Parsear query de búsqueda avanzada.
        
        Soporta:
        - "category:plomeria" - Filtrar por categoría
        - "difficulty:fácil" - Filtrar por dificultad
        - "rating:>4" - Filtrar por rating
        - "tags:emergencia" - Filtrar por tags
        - "date:>2024-01-01" - Filtrar por fecha
        
        Args:
            query: Query de búsqueda
        
        Returns:
            Diccionario con filtros parseados
        """
        filters = {
            "text": [],
            "category": None,
            "difficulty": None,
            "min_rating": None,
            "max_rating": None,
            "tags": [],
            "date_from": None,
            "date_to": None
        }
        
        for key, pattern in self.PATTERNS.items():
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                filters.update(self._process_match(key, matches[0], filters))
        
        text_query = self._extract_text_query(query)
        filters["text"] = text_query
        
        return filters
    
    def _process_match(
        self,
        key: str,
        match: str,
        filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Procesar match de patrón."""
        if key == "category":
            return {"category": match.lower()}
        elif key == "difficulty":
            return {"difficulty": match.capitalize()}
        elif key == "rating":
            return self._parse_rating(match)
        elif key == "tags":
            return {"tags": filters.get("tags", []) + [match.lower()]}
        elif key == "date":
            return self._parse_date(match)
        return {}
    
    def _parse_rating(self, rating_str: str) -> Dict[str, Any]:
        """Parsear filtro de rating."""
        if rating_str.startswith(">"):
            return {"min_rating": float(rating_str[1:])}
        elif rating_str.startswith("<"):
            return {"max_rating": float(rating_str[1:])}
        elif rating_str.startswith("="):
            val = float(rating_str[1:])
            return {"min_rating": val, "max_rating": val}
        else:
            return {"min_rating": float(rating_str)}
    
    def _parse_date(self, date_str: str) -> Dict[str, Any]:
        """Parsear filtro de fecha."""
        try:
            if date_str.startswith(">"):
                return {"date_from": datetime.strptime(date_str[1:], "%Y-%m-%d")}
            elif date_str.startswith("<"):
                return {"date_to": datetime.strptime(date_str[1:], "%Y-%m-%d")}
            else:
                date = datetime.strptime(date_str, "%Y-%m-%d")
                return {"date_from": date, "date_to": date}
        except ValueError:
            return {}
    
    def _extract_text_query(self, query: str) -> list:
        """Extraer texto de búsqueda."""
        text_query = query
        for pattern in self.PATTERNS.values():
            text_query = re.sub(pattern, '', text_query, flags=re.IGNORECASE)
        
        text_parts = [
            t.strip()
            for t in text_query.split()
            if t.strip() and len(t.strip()) > 2
        ]
        return text_parts

