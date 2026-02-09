"""
Utilidades para filtrado y búsqueda
"""

from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from pydantic import BaseModel, Field


class SongFilters(BaseModel):
    """Filtros para búsqueda de canciones"""
    user_id: Optional[str] = Field(None, description="Filtrar por usuario")
    genre: Optional[str] = Field(None, description="Filtrar por género")
    status: Optional[str] = Field(None, description="Filtrar por estado")
    date_from: Optional[datetime] = Field(None, description="Fecha desde")
    date_to: Optional[datetime] = Field(None, description="Fecha hasta")
    min_duration: Optional[int] = Field(None, ge=1, description="Duración mínima en segundos")
    max_duration: Optional[int] = Field(None, ge=1, description="Duración máxima en segundos")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte filtros a diccionario"""
        filters = {}
        if self.user_id:
            filters["user_id"] = self.user_id
        if self.genre:
            filters["genre"] = self.genre
        if self.status:
            filters["status"] = self.status
        if self.date_from:
            filters["date_from"] = self.date_from
        if self.date_to:
            filters["date_to"] = self.date_to
        if self.min_duration:
            filters["min_duration"] = self.min_duration
        if self.max_duration:
            filters["max_duration"] = self.max_duration
        return filters


def apply_filters(
    items: List[Dict[str, Any]],
    filters: SongFilters
) -> List[Dict[str, Any]]:
    """Aplica filtros a una lista de items"""
    filtered = items
    
    if filters.user_id:
        filtered = [item for item in filtered if item.get("user_id") == filters.user_id]
    
    if filters.genre:
        filtered = [
            item for item in filtered
            if item.get("metadata", {}).get("genre") == filters.genre
        ]
    
    if filters.status:
        filtered = [item for item in filtered if item.get("status") == filters.status]
    
    if filters.date_from:
        filtered = [
            item for item in filtered
            if datetime.fromisoformat(item.get("created_at", "")) >= filters.date_from
        ]
    
    if filters.date_to:
        filtered = [
            item for item in filtered
            if datetime.fromisoformat(item.get("created_at", "")) <= filters.date_to
        ]
    
    if filters.min_duration:
        filtered = [
            item for item in filtered
            if item.get("metadata", {}).get("duration", 0) >= filters.min_duration
        ]
    
    if filters.max_duration:
        filtered = [
            item for item in filtered
            if item.get("metadata", {}).get("duration", float('inf')) <= filters.max_duration
        ]
    
    return filtered

