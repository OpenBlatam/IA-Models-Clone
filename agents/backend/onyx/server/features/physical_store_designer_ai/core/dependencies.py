"""
Dependencies for FastAPI routes
"""

from typing import Optional, Any
from fastapi import Header, Depends, Query
from ..config.settings import settings
from .exceptions import AuthenticationError, ValidationError


async def verify_api_key(x_api_key: Optional[str] = Header(None)) -> Optional[str]:
    """
    Dependency to verify API key if configured
    
    Usage:
        @router.get("/protected")
        async def protected_route(api_key: str = Depends(verify_api_key)):
            ...
    """
    if not settings.api_key_header:
        # API key verification disabled
        return None
    
    if not x_api_key:
        raise AuthenticationError("API key requerida")
    
    # In production, verify against database or secret manager
    # For now, just check if it's provided
    return x_api_key


def get_pagination_params(
    page: int = Query(1, ge=1, description="Número de página"),
    page_size: int = Query(20, ge=1, le=100, description="Tamaño de página")
) -> tuple[int, int]:
    """
    Dependency to get pagination parameters
    
    Usage:
        @router.get("/items")
        async def list_items(pagination: tuple[int, int] = Depends(get_pagination_params)):
            offset, limit = pagination
            ...
    """
    offset = (page - 1) * page_size
    return offset, page_size


def get_sort_params(
    sort_by: Optional[str] = Query(None, description="Campo para ordenar"),
    sort_order: str = Query("asc", regex="^(asc|desc)$", description="Orden: asc o desc")
) -> Optional[tuple[str, str]]:
    """
    Dependency to get sorting parameters
    
    Usage:
        @router.get("/items")
        async def list_items(sort: Optional[tuple[str, str]] = Depends(get_sort_params)):
            if sort:
                sort_by, sort_order = sort
            ...
    """
    if sort_by:
        return sort_by, sort_order
    return None


def get_filter_params(
    search: Optional[str] = Query(None, description="Búsqueda de texto"),
    min_value: Optional[float] = Query(None, description="Valor mínimo"),
    max_value: Optional[float] = Query(None, description="Valor máximo"),
    status: Optional[str] = Query(None, description="Filtro por estado")
) -> dict[str, Any]:
    """
    Dependency to get filter parameters
    
    Usage:
        @router.get("/items")
        async def list_items(filters: dict = Depends(get_filter_params)):
            if filters["search"]:
                ...
    """
    filters = {}
    if search:
        filters["search"] = search.strip()
    if min_value is not None:
        filters["min_value"] = min_value
    if max_value is not None:
        filters["max_value"] = max_value
    if status:
        filters["status"] = status
    return filters


def get_date_range_params(
    start_date: Optional[str] = Query(None, description="Fecha inicio (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="Fecha fin (YYYY-MM-DD)")
) -> Optional[tuple[str, str]]:
    """
    Dependency to get date range parameters
    
    Usage:
        @router.get("/items")
        async def list_items(date_range: Optional[tuple[str, str]] = Depends(get_date_range_params)):
            if date_range:
                start_date, end_date = date_range
            ...
    """
    if start_date and end_date:
        # Simple validation
        from datetime import datetime
        try:
            datetime.strptime(start_date, "%Y-%m-%d")
            datetime.strptime(end_date, "%Y-%m-%d")
            return start_date, end_date
        except ValueError:
            raise ValidationError("Formato de fecha inválido. Use YYYY-MM-DD")
    return None

