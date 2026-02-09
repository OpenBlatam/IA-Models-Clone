"""
Rutas de Búsqueda Avanzada
===========================

Endpoints para búsqueda avanzada de manuales.
"""

import logging
from fastapi import APIRouter, HTTPException, Depends, Query, Path
from typing import Optional, List
from pydantic import BaseModel
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession

from ...database.session import get_async_session
from ...services.manual.manual_service import ManualService
from ...services.semantic_search_service import SemanticSearchService
from ...utils.search.advanced_search import AdvancedSearch
from ...utils.validation.validators import Validators

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["search"])


# Modelos Pydantic
class AdvancedSearchRequest(BaseModel):
    """Request para búsqueda avanzada."""
    query: Optional[str] = None
    category: Optional[str] = None
    difficulty: Optional[str] = None
    min_rating: Optional[float] = None
    max_rating: Optional[float] = None
    tags: Optional[List[str]] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    limit: int = 20
    offset: int = 0


# Dependencies
async def get_db_session() -> AsyncSession:
    """Obtener sesión de base de datos."""
    async for session in get_async_session():
        yield session


# Endpoints
@router.post("/search/semantic")
async def semantic_search(
    query: str = Query(..., description="Query de búsqueda"),
    category: Optional[str] = Query(None, description="Filtrar por categoría"),
    limit: int = Query(10, ge=1, le=50),
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Umbral de similitud"),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Búsqueda semántica de manuales usando embeddings.
    
    - **query**: Query de búsqueda
    - **category**: Filtrar por categoría (opcional)
    - **limit**: Número de resultados
    - **threshold**: Umbral mínimo de similitud
    """
    try:
        service = SemanticSearchService(db)
        results = await service.search_semantic(
            query=query,
            limit=limit,
            threshold=threshold,
            category=category
        )
        
        from ...api.routes.history import ManualListItem
        
        return {
            "success": True,
            "query": query,
            "results": [
                {
                    "manual": ManualListItem.from_orm(r["manual"]).dict(),
                    "similarity": r["similarity"],
                    "score": r["score"]
                }
                for r in results
            ],
            "total": len(results)
        }
    
    except Exception as e:
        logger.error(f"Error en búsqueda semántica: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en búsqueda: {str(e)}")


@router.post("/search/advanced")
async def advanced_search(
    request: AdvancedSearchRequest,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Búsqueda avanzada de manuales.
    
    Soporta query avanzada con sintaxis:
    - "category:plomeria" - Filtrar por categoría
    - "difficulty:fácil" - Filtrar por dificultad
    - "rating:>4" - Filtrar por rating
    - "tags:emergencia" - Filtrar por tags
    - "date:>2024-01-01" - Filtrar por fecha
    
    Ejemplo: "fuga category:plomeria rating:>4"
    """
    try:
        service = ManualService(db)
        validator = Validators()
        
        # Validar parámetros
        if request.category:
            is_valid, error = validator.validate_category(request.category)
            if not is_valid:
                raise HTTPException(status_code=400, detail=error)
        
        if request.difficulty:
            is_valid, error = validator.validate_difficulty(request.difficulty)
            if not is_valid:
                raise HTTPException(status_code=400, detail=error)
        
        # Parsear fechas
        date_from = None
        date_to = None
        if request.date_from:
            try:
                date_from = datetime.fromisoformat(request.date_from.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(status_code=400, detail="Formato de fecha inválido (use ISO 8601)")
        
        if request.date_to:
            try:
                date_to = datetime.fromisoformat(request.date_to.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(status_code=400, detail="Formato de fecha inválido (use ISO 8601)")
        
        # Validar rango de fechas
        is_valid, error = validator.validate_date_range(date_from, date_to)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error)
        
        # Realizar búsqueda
        manuals = await service.search_manuals(
            category=request.category,
            search_term=request.query,
            limit=request.limit,
            offset=request.offset,
            difficulty=request.difficulty,
            min_rating=request.min_rating,
            max_rating=request.max_rating,
            tags=request.tags,
            date_from=date_from,
            date_to=date_to,
            advanced_query=request.query if request.query and any(
                keyword in request.query for keyword in ['category:', 'difficulty:', 'rating:', 'tags:', 'date:']
            ) else None
        )
        
        from ...api.routes.history import ManualListItem
        
        return {
            "success": True,
            "results": [ManualListItem.from_orm(m) for m in manuals],
            "total": len(manuals),
            "limit": request.limit,
            "offset": request.offset
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en búsqueda avanzada: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en búsqueda: {str(e)}")


@router.get("/search")
async def simple_search(
    q: str = Query(..., description="Término de búsqueda"),
    category: Optional[str] = Query(None, description="Filtrar por categoría"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Búsqueda simple de manuales.
    
    - **q**: Término de búsqueda
    - **category**: Filtrar por categoría (opcional)
    - **limit**: Límite de resultados
    - **offset**: Offset para paginación
    """
    try:
        service = ManualService(db)
        manuals = await service.search_manuals(
            category=category,
            search_term=q,
            limit=limit,
            offset=offset
        )
        
        from ...api.routes.history import ManualListItem
        
        return {
            "success": True,
            "query": q,
            "results": [ManualListItem.from_orm(m) for m in manuals],
            "total": len(manuals),
            "limit": limit,
            "offset": offset
        }
    
    except Exception as e:
        logger.error(f"Error en búsqueda: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en búsqueda: {str(e)}")


@router.get("/search/suggestions")
async def get_search_suggestions(
    q: str = Query(..., description="Término de búsqueda"),
    limit: int = Query(10, ge=1, le=20),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Obtener sugerencias de búsqueda.
    
    - **q**: Término de búsqueda
    - **limit**: Número de sugerencias
    """
    try:
        service = ManualService(db)
        
        # Buscar manuales que coincidan
        manuals = await service.search_manuals(
            search_term=q,
            limit=limit
        )
        
        # Extraer sugerencias de títulos y categorías
        suggestions = set()
        for manual in manuals:
            if manual.title:
                suggestions.add(manual.title)
            suggestions.add(manual.category)
            if manual.tags:
                for tag in manual.tags.split(','):
                    if tag.strip().lower().startswith(q.lower()):
                        suggestions.add(tag.strip())
        
        return {
            "success": True,
            "query": q,
            "suggestions": sorted(list(suggestions))[:limit]
        }
    
    except Exception as e:
        logger.error(f"Error obteniendo sugerencias: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo sugerencias: {str(e)}")

