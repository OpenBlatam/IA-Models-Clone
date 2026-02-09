"""
Utilidades para paginación
"""

from typing import Generic, TypeVar, List, Optional
from pydantic import BaseModel, Field

T = TypeVar('T')


class PaginationParams(BaseModel):
    """Parámetros de paginación"""
    limit: int = Field(50, ge=1, le=100, description="Número máximo de resultados")
    offset: int = Field(0, ge=0, description="Offset para paginación")


class PaginatedResponse(BaseModel, Generic[T]):
    """Response paginado"""
    items: List[T]
    total: int
    limit: int
    offset: int
    has_more: bool
    
    @property
    def next_offset(self) -> Optional[int]:
        """Calcula el siguiente offset"""
        if self.has_more:
            return self.offset + self.limit
        return None
    
    @property
    def prev_offset(self) -> Optional[int]:
        """Calcula el offset anterior"""
        if self.offset > 0:
            return max(0, self.offset - self.limit)
        return None


def create_paginated_response(
    items: List[T],
    total: int,
    limit: int,
    offset: int
) -> PaginatedResponse[T]:
    """Crea una respuesta paginada"""
    return PaginatedResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
        has_more=(offset + limit) < total
    )

