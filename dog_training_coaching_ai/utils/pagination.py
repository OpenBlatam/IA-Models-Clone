"""
Pagination Utilities
====================
Utilidades para paginación.
"""

from typing import List, Any, Optional, Dict, Generic, TypeVar
from pydantic import BaseModel

T = TypeVar('T')


class PaginationParams(BaseModel):
    """Parámetros de paginación."""
    page: int = 1
    page_size: int = 20
    max_page_size: int = 100
    
    def validate(self):
        """Validar parámetros."""
        if self.page < 1:
            self.page = 1
        if self.page_size < 1:
            self.page_size = 1
        if self.page_size > self.max_page_size:
            self.page_size = self.max_page_size


class PaginatedResponse(BaseModel, Generic[T]):
    """Respuesta paginada."""
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
    ) -> "PaginatedResponse[T]":
        """Crear respuesta paginada."""
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


def paginate(items: List[Any], page: int = 1, page_size: int = 20) -> Dict[str, Any]:
    """
    Paginar lista de items.
    
    Args:
        items: Lista de items
        page: Número de página
        page_size: Tamaño de página
        
    Returns:
        Diccionario con items paginados y metadata
    """
    params = PaginationParams(page=page, page_size=page_size)
    params.validate()
    
    total = len(items)
    start = (params.page - 1) * params.page_size
    end = start + params.page_size
    
    paginated_items = items[start:end]
    
    return PaginatedResponse.create(
        items=paginated_items,
        total=total,
        page=params.page,
        page_size=params.page_size
    ).dict()


def get_pagination_links(
    base_url: str,
    page: int,
    page_size: int,
    total: int
) -> Dict[str, Optional[str]]:
    """
    Generar links de paginación.
    
    Args:
        base_url: URL base
        page: Página actual
        page_size: Tamaño de página
        total: Total de items
        
    Returns:
        Diccionario con links
    """
    total_pages = (total + page_size - 1) // page_size if total > 0 else 0
    
    links = {
        "first": f"{base_url}?page=1&page_size={page_size}" if total_pages > 0 else None,
        "last": f"{base_url}?page={total_pages}&page_size={page_size}" if total_pages > 0 else None,
        "next": f"{base_url}?page={page + 1}&page_size={page_size}" if page < total_pages else None,
        "prev": f"{base_url}?page={page - 1}&page_size={page_size}" if page > 1 else None
    }
    
    return links

