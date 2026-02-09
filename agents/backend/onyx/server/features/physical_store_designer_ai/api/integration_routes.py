"""
Integration Routes - Endpoints para integraciones y funcionalidades avanzadas
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Optional, List
import logging

from ..services.external_apis_service import ExternalAPIsService
from ..services.backup_service import BackupService
from ..services.export_service import ExportService
from ..services.webhook_service import WebhookService, WebhookEvent
from ..services.cache_service import CacheService
from ..services.storage_service import StorageService
from ..services.auth_service import AuthService

logger = logging.getLogger(__name__)

router = APIRouter()

# Inicializar servicios
external_apis = ExternalAPIsService()
backup_service = BackupService()
export_service = ExportService()
webhook_service = WebhookService()
cache_service = CacheService()
storage_service = StorageService()
auth_service = AuthService()


def verify_token(authorization: Optional[str] = None):
    """Verificar token (opcional para algunos endpoints)"""
    if authorization:
        token = authorization.replace("Bearer ", "")
        payload = auth_service.verify_token(token)
        if payload:
            return payload
    return None


@router.get("/external/location/{location}")
async def get_location_details(location: str):
    """Obtener detalles de ubicación"""
    details = await external_apis.get_location_details(location)
    return details


@router.get("/external/weather/{location}")
async def get_weather(location: str):
    """Obtener información del clima"""
    weather = await external_apis.get_weather_info(location)
    return weather


@router.get("/external/nearby/{location}")
async def get_nearby_places(location: str, place_type: str = "store"):
    """Obtener lugares cercanos"""
    places = await external_apis.get_nearby_places(location, place_type)
    return places


@router.post("/backup/create")
async def create_backup(backup_name: Optional[str] = None):
    """Crear backup de todos los diseños"""
    all_designs = storage_service.list_designs()
    
    designs = []
    for design_summary in all_designs:
        design = storage_service.load_design(design_summary["store_id"])
        if design:
            designs.append(design.dict())
    
    backup = backup_service.create_backup(designs, backup_name)
    return backup


@router.get("/backup/list")
async def list_backups():
    """Listar todos los backups"""
    backups = backup_service.list_backups()
    return {"backups": backups}


@router.post("/backup/restore/{backup_id}")
async def restore_backup(backup_id: str):
    """Restaurar backup"""
    try:
        restore_result = backup_service.restore_backup(backup_id)
        return restore_result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/backup/{backup_id}")
async def delete_backup(backup_id: str):
    """Eliminar backup"""
    deleted = backup_service.delete_backup(backup_id)
    
    if not deleted:
        raise HTTPException(status_code=404, detail="Backup no encontrado")
    
    return {"message": "Backup eliminado", "backup_id": backup_id}


@router.get("/export/cad/{store_id}")
async def export_to_cad(store_id: str):
    """Exportar diseño a formato CAD"""
    design = storage_service.load_design(store_id)
    if not design:
        raise HTTPException(status_code=404, detail="Diseño no encontrado")
    
    cad_export = export_service.export_to_cad(design)
    return cad_export


@router.get("/export/3d/{store_id}")
async def export_to_3d(store_id: str):
    """Exportar diseño a formato 3D"""
    design = storage_service.load_design(store_id)
    if not design:
        raise HTTPException(status_code=404, detail="Diseño no encontrado")
    
    export_3d = export_service.export_to_3d(design)
    return export_3d


@router.get("/export/svg/{store_id}")
async def export_to_svg(store_id: str):
    """Exportar diseño a formato SVG"""
    design = storage_service.load_design(store_id)
    if not design:
        raise HTTPException(status_code=404, detail="Diseño no encontrado")
    
    svg = export_service.export_to_svg(design)
    
    from fastapi.responses import Response
    return Response(content=svg, media_type="image/svg+xml")


@router.get("/export/pdf-advanced/{store_id}")
async def export_to_pdf_advanced(store_id: str):
    """Exportar diseño a PDF avanzado"""
    design = storage_service.load_design(store_id)
    if not design:
        raise HTTPException(status_code=404, detail="Diseño no encontrado")
    
    pdf_export = export_service.export_to_pdf_advanced(design)
    return pdf_export


@router.post("/webhooks/register")
async def register_webhook(
    user_id: str,
    url: str,
    events: List[str],
    secret: Optional[str] = None
):
    """Registrar webhook"""
    try:
        webhook_events = [WebhookEvent(e) for e in events]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Evento inválido: {e}")
    
    webhook = webhook_service.register_webhook(user_id, url, webhook_events, secret)
    return webhook


@router.get("/webhooks/{user_id}")
async def get_webhooks(user_id: str):
    """Obtener webhooks de usuario"""
    webhooks = webhook_service.get_webhooks(user_id)
    return {"user_id": user_id, "webhooks": webhooks}


@router.delete("/webhooks/{user_id}/{webhook_id}")
async def unregister_webhook(user_id: str, webhook_id: str):
    """Desregistrar webhook"""
    unregistered = webhook_service.unregister_webhook(user_id, webhook_id)
    
    if not unregistered:
        raise HTTPException(status_code=404, detail="Webhook no encontrado")
    
    return {"message": "Webhook desregistrado", "webhook_id": webhook_id}


@router.get("/cache/stats")
async def get_cache_stats():
    """Obtener estadísticas del caché"""
    stats = cache_service.get_stats()
    return stats


@router.post("/cache/clear")
async def clear_cache():
    """Limpiar caché"""
    count = cache_service.clear()
    return {"message": f"Caché limpiado", "items_removed": count}


@router.post("/cache/cleanup")
async def cleanup_cache():
    """Limpiar items expirados del caché"""
    count = cache_service.cleanup_expired()
    return {"message": f"Items expirados eliminados", "count": count}




