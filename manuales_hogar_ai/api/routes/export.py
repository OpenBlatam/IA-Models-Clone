"""
Rutas de Exportación
=====================

Endpoints para exportar manuales a diferentes formatos.
"""

import logging
from fastapi import APIRouter, HTTPException, Depends, Path, Query
from fastapi.responses import Response
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession

from ...database.session import get_async_session
from ...services.manual.manual_service import ManualService
from ...utils.export.manual_exporter import ManualExporter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["export"])


# Dependencies
async def get_db_session() -> AsyncSession:
    """Obtener sesión de base de datos."""
    async for session in get_async_session():
        yield session


# Endpoints
@router.get("/manuals/{manual_id}/export/markdown")
async def export_to_markdown(
    manual_id: int = Path(..., description="ID del manual"),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Exportar manual a Markdown.
    
    - **manual_id**: ID del manual
    """
    try:
        service = ManualService(db)
        manual = await service.get_manual_by_id(manual_id)
        
        if not manual:
            raise HTTPException(status_code=404, detail="Manual no encontrado")
        
        # Incrementar contador de vistas
        manual.view_count += 1
        await db.commit()
        
        # Preparar metadata
        metadata = {
            "title": manual.title,
            "category": manual.category,
            "difficulty": manual.difficulty,
            "estimated_time": manual.estimated_time
        }
        
        # Exportar
        exporter = ManualExporter()
        markdown_content = exporter.export_to_markdown(
            manual.manual_content,
            metadata=metadata
        )
        
        return Response(
            content=markdown_content,
            media_type="text/markdown",
            headers={
                "Content-Disposition": f'attachment; filename="manual_{manual_id}.md"'
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exportando a markdown: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error exportando: {str(e)}")


@router.get("/manuals/{manual_id}/export/text")
async def export_to_text(
    manual_id: int = Path(..., description="ID del manual"),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Exportar manual a texto plano.
    
    - **manual_id**: ID del manual
    """
    try:
        service = ManualService(db)
        manual = await service.get_manual_by_id(manual_id)
        
        if not manual:
            raise HTTPException(status_code=404, detail="Manual no encontrado")
        
        # Incrementar contador de vistas
        manual.view_count += 1
        await db.commit()
        
        # Preparar metadata
        metadata = {
            "title": manual.title,
            "category": manual.category,
            "difficulty": manual.difficulty,
            "estimated_time": manual.estimated_time
        }
        
        # Exportar
        exporter = ManualExporter()
        text_content = exporter.export_to_text(
            manual.manual_content,
            metadata=metadata
        )
        
        return Response(
            content=text_content,
            media_type="text/plain",
            headers={
                "Content-Disposition": f'attachment; filename="manual_{manual_id}.txt"'
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exportando a texto: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error exportando: {str(e)}")


@router.get("/manuals/{manual_id}/export/json")
async def export_to_json(
    manual_id: int = Path(..., description="ID del manual"),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Exportar manual a JSON.
    
    - **manual_id**: ID del manual
    """
    try:
        import json
        
        service = ManualService(db)
        manual = await service.get_manual_by_id(manual_id)
        
        if not manual:
            raise HTTPException(status_code=404, detail="Manual no encontrado")
        
        # Incrementar contador de vistas
        manual.view_count += 1
        await db.commit()
        
        # Preparar metadata
        metadata = {
            "id": manual.id,
            "title": manual.title,
            "category": manual.category,
            "difficulty": manual.difficulty,
            "estimated_time": manual.estimated_time,
            "tools_required": manual.tools_required,
            "materials_required": manual.materials_required,
            "safety_warnings": manual.safety_warnings,
            "tags": manual.tags,
            "average_rating": manual.average_rating,
            "rating_count": manual.rating_count,
            "view_count": manual.view_count,
            "favorite_count": manual.favorite_count,
            "created_at": manual.created_at.isoformat() if manual.created_at else None
        }
        
        # Exportar
        exporter = ManualExporter()
        json_data = exporter.export_to_json(
            manual.manual_content,
            metadata=metadata
        )
        
        return Response(
            content=json.dumps(json_data, indent=2, ensure_ascii=False),
            media_type="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="manual_{manual_id}.json"'
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exportando a JSON: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error exportando: {str(e)}")




