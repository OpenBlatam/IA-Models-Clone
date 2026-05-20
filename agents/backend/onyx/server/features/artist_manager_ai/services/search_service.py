"""
Search Service
==============

Servicio de búsqueda avanzada.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import re

logger = logging.getLogger(__name__)


class SearchService:
    """Servicio de búsqueda."""
    
    def __init__(self):
        """Inicializar servicio de búsqueda."""
        self._logger = logger
    
    def search_events(
        self,
        events: List[Dict[str, Any]],
        query: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Buscar eventos.
        
        Args:
            events: Lista de eventos
            query: Consulta de búsqueda
            filters: Filtros adicionales
        
        Returns:
            Eventos encontrados
        """
        results = []
        query_lower = query.lower()
        
        for event in events:
            score = 0
            
            # Búsqueda en título
            if query_lower in event.get("title", "").lower():
                score += 10
            
            # Búsqueda en descripción
            if query_lower in event.get("description", "").lower():
                score += 5
            
            # Búsqueda en ubicación
            if query_lower in event.get("location", "").lower():
                score += 3
            
            # Aplicar filtros
            if filters:
                if "event_type" in filters and event.get("event_type") != filters["event_type"]:
                    continue
                if "date_from" in filters:
                    event_date = datetime.fromisoformat(event.get("start_time", ""))
                    if event_date < filters["date_from"]:
                        continue
                if "date_to" in filters:
                    event_date = datetime.fromisoformat(event.get("start_time", ""))
                    if event_date > filters["date_to"]:
                        continue
            
            if score > 0:
                event["_search_score"] = score
                results.append(event)
        
        # Ordenar por score
        results.sort(key=lambda x: x.get("_search_score", 0), reverse=True)
        return results
    
    def search_routines(
        self,
        routines: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Buscar rutinas.
        
        Args:
            routines: Lista de rutinas
            query: Consulta de búsqueda
        
        Returns:
            Rutinas encontradas
        """
        results = []
        query_lower = query.lower()
        
        for routine in routines:
            score = 0
            
            if query_lower in routine.get("title", "").lower():
                score += 10
            if query_lower in routine.get("description", "").lower():
                score += 5
            if query_lower in routine.get("routine_type", "").lower():
                score += 3
            
            if score > 0:
                routine["_search_score"] = score
                results.append(routine)
        
        results.sort(key=lambda x: x.get("_search_score", 0), reverse=True)
        return results
    
    def fuzzy_search(
        self,
        items: List[Dict[str, Any]],
        query: str,
        fields: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Búsqueda difusa (fuzzy).
        
        Args:
            items: Lista de items
            query: Consulta
            fields: Campos a buscar
        
        Returns:
            Items encontrados
        """
        results = []
        query_lower = query.lower()
        query_words = query_lower.split()
        
        for item in items:
            score = 0
            
            for field in fields:
                field_value = str(item.get(field, "")).lower()
                
                # Coincidencia exacta
                if query_lower in field_value:
                    score += 10
                
                # Coincidencia de palabras
                for word in query_words:
                    if word in field_value:
                        score += 2
                
                # Coincidencia parcial (primeros caracteres)
                if field_value.startswith(query_lower[:3]):
                    score += 1
            
            if score > 0:
                item["_search_score"] = score
                results.append(item)
        
        results.sort(key=lambda x: x.get("_search_score", 0), reverse=True)
        return results




