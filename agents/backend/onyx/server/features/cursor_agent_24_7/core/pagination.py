"""
Pagination - Paginación avanzada
==================================

Utilidades para paginación de resultados.
"""

from typing import Generic, TypeVar, List, Optional
from pydantic import BaseModel

T = TypeVar('T')


class PaginationParams(BaseModel):
    """Parámetros de paginación."""
    page: int = 1
    page_size: int = 20
    max_page_size: int = 100
    
    def __init__(self, **data):
        super().__init__(**data)
        # Validar y ajustar
        if self.page < 1:
            self.page = 1
        if self.page_size < 1:
            self.page_size = 20
        if self.page_size > self.max_page_size:
            self.page_size = self.max_page_size
    
    @property
    def offset(self) -> int:
        """Calcular offset."""
        return (self.page - 1) * self.page_size
    
    @property
    def limit(self) -> int:
        """Obtener límite."""
        return self.page_size


class PaginatedResponse(BaseModel, Generic[T]):
    """Response paginado."""
    items: List[T]
    total: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_previous: bool
    
    @classmethod
    def create(
        cls,
        items: List[T],
        total: int,
        page: int,
        page_size: int
    ) -> 'PaginatedResponse[T]':
        """Crear response paginado."""
        total_pages = (total + page_size - 1) // page_size if total > 0 else 0
        
        return cls(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_previous=page > 1
        )




