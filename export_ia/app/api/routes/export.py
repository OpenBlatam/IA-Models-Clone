"""
Export Routes - Rutas de exportación
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from ...core.engine import get_export_engine
from ...core.models import ExportConfig, ExportFormat, DocumentType, QualityLevel

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/export")
async def export_document(request: Request):
    """
    Exportar un documento.
    
    Body:
        - content: Contenido del documento
        - format: Formato de exportación (pdf, docx, html, etc.)
        - document_type: Tipo de documento (report, presentation, etc.)
        - quality_level: Nivel de calidad (draft, standard, professional, etc.)
    """
    try:
        data = await request.json()
        content = data.get("content")
        format_name = data.get("format", "pdf")
        document_type = data.get("document_type", "report")
        quality_level = data.get("quality_level", "professional")
        
        if not content:
            raise HTTPException(status_code=400, detail="El contenido es requerido")
        
        # Crear configuración
        config = ExportConfig(
            format=ExportFormat(format_name),
            document_type=DocumentType(document_type),
            quality_level=QualityLevel(quality_level)
        )
        
        # Exportar documento
        engine = get_export_engine()
        task_id = await engine.export_document(content, config)
        
        return {
            "task_id": task_id,
            "status": "pending",
            "message": "Tarea de exportación creada",
            "format": format_name,
            "document_type": document_type,
            "quality_level": quality_level
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error en exportación: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@router.get("/export/{task_id}/status")
async def get_export_status(task_id: str):
    """
    Obtener estado de una tarea de exportación.
    
    Args:
        task_id: ID de la tarea
    """
    try:
        engine = get_export_engine()
        status = await engine.get_task_status(task_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Tarea no encontrada")
        
        return status
        
    except Exception as e:
        logger.error(f"Error obteniendo estado: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@router.get("/export/{task_id}/download")
async def download_export(task_id: str):
    """
    Descargar archivo exportado.
    
    Args:
        task_id: ID de la tarea
    """
    try:
        engine = get_export_engine()
        status = await engine.get_task_status(task_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Tarea no encontrada")
        
        if status.get("status") != "completed":
            raise HTTPException(status_code=400, detail="La tarea no está completada")
        
        file_path = status.get("file_path")
        if not file_path:
            raise HTTPException(status_code=404, detail="Archivo no encontrado")
        
        return FileResponse(
            path=file_path,
            filename=file_path.split("/")[-1],
            media_type="application/octet-stream"
        )
        
    except Exception as e:
        logger.error(f"Error descargando archivo: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@router.post("/validate")
async def validate_content(request: Request):
    """
    Validar contenido de documento.
    
    Body:
        - content: Contenido del documento
        - format: Formato de exportación
        - document_type: Tipo de documento
        - quality_level: Nivel de calidad
    """
    try:
        data = await request.json()
        content = data.get("content")
        format_name = data.get("format", "pdf")
        document_type = data.get("document_type", "report")
        quality_level = data.get("quality_level", "professional")
        
        if not content:
            raise HTTPException(status_code=400, detail="El contenido es requerido")
        
        # Crear configuración
        config = ExportConfig(
            format=ExportFormat(format_name),
            document_type=DocumentType(document_type),
            quality_level=QualityLevel(quality_level)
        )
        
        # Validar contenido
        engine = get_export_engine()
        result = engine.validate_content(content, config)
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error en validación: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@router.get("/formats")
async def get_supported_formats():
    """Obtener formatos soportados."""
    try:
        engine = get_export_engine()
        formats = engine.list_supported_formats()
        
        return {
            "formats": formats,
            "count": len(formats)
        }
        
    except Exception as e:
        logger.error(f"Error obteniendo formatos: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@router.get("/templates/{doc_type}")
async def get_template(doc_type: str):
    """
    Obtener plantilla de documento.
    
    Args:
        doc_type: Tipo de documento
    """
    try:
        engine = get_export_engine()
        template = engine.get_document_template(DocumentType(doc_type))
        
        return {
            "document_type": doc_type,
            "template": template
        }
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Tipo de documento inválido")
    except Exception as e:
        logger.error(f"Error obteniendo plantilla: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")




