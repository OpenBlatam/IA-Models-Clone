"""
Premium Routes - Endpoints para funcionalidades premium
"""

from fastapi import APIRouter, HTTPException
from typing import List, Optional
import logging

from ..services.reporting_service import ReportingService
from ..services.collaboration_service import CollaborationService, SharePermission
from ..services.dashboard_service import DashboardService
from ..services.template_service import TemplateService
from ..services.trends_service import TrendsService
from ..services.notification_service import NotificationService, NotificationType
from ..services.storage_service import StorageService

logger = logging.getLogger(__name__)

router = APIRouter()

# Inicializar servicios
reporting_service = ReportingService()
collaboration_service = CollaborationService()
dashboard_service = DashboardService()
template_service = TemplateService()
trends_service = TrendsService()
notification_service = NotificationService()
storage_service = StorageService()


@router.get("/reports/{store_id}")
async def get_comprehensive_report(store_id: str):
    """Obtener reporte completo de diseño"""
    design = storage_service.load_design(store_id)
    if not design:
        raise HTTPException(status_code=404, detail="Diseño no encontrado")
    
    report = reporting_service.generate_comprehensive_report(design)
    return report


@router.get("/reports/{store_id}/pdf")
async def get_pdf_report(store_id: str):
    """Obtener reporte en PDF"""
    design = storage_service.load_design(store_id)
    if not design:
        raise HTTPException(status_code=404, detail="Diseño no encontrado")
    
    pdf_info = reporting_service.generate_pdf_report(design)
    return {"message": pdf_info, "download_url": f"/reports/{store_id}/pdf/download"}


@router.get("/reports/{store_id}/excel")
async def get_excel_report(store_id: str):
    """Obtener reporte en Excel"""
    design = storage_service.load_design(store_id)
    if not design:
        raise HTTPException(status_code=404, detail="Diseño no encontrado")
    
    excel_info = reporting_service.generate_excel_report(design)
    return {"message": excel_info, "download_url": f"/reports/{store_id}/excel/download"}


@router.post("/share/{store_id}")
async def share_design(
    store_id: str,
    shared_by: str,
    permission: str = "view",
    expires_in_days: Optional[int] = None,
    password: Optional[str] = None
):
    """Compartir diseño"""
    design = storage_service.load_design(store_id)
    if not design:
        raise HTTPException(status_code=404, detail="Diseño no encontrado")
    
    try:
        perm = SharePermission(permission)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Permiso inválido: {permission}")
    
    share_info = collaboration_service.share_design(
        store_id=store_id,
        shared_by=shared_by,
        permission=perm,
        expires_in_days=expires_in_days,
        password=password
    )
    
    return share_info


@router.get("/share/{share_id}")
async def get_shared_design(share_id: str, password: Optional[str] = None):
    """Obtener diseño compartido"""
    share_info = collaboration_service.get_shared_design(share_id, password)
    
    if not share_info:
        raise HTTPException(status_code=404, detail="Diseño compartido no encontrado o expirado")
    
    if "error" in share_info:
        raise HTTPException(status_code=403, detail=share_info["error"])
    
    design = storage_service.load_design(share_info["store_id"])
    if not design:
        raise HTTPException(status_code=404, detail="Diseño no encontrado")
    
    return {
        "share_info": share_info,
        "design": design.dict()
    }


@router.post("/share/{share_id}/revoke")
async def revoke_share(share_id: str, revoked_by: str):
    """Revocar compartir"""
    revoked = collaboration_service.revoke_share(share_id, revoked_by)
    
    if not revoked:
        raise HTTPException(status_code=404, detail="Compartir no encontrado o sin permisos")
    
    return {"message": "Compartir revocado exitosamente", "share_id": share_id}


@router.post("/comments/{store_id}")
async def add_comment(
    store_id: str,
    commenter: str,
    content: str,
    section: Optional[str] = None
):
    """Agregar comentario"""
    design = storage_service.load_design(store_id)
    if not design:
        raise HTTPException(status_code=404, detail="Diseño no encontrado")
    
    comment = collaboration_service.add_comment(
        store_id=store_id,
        commenter=commenter,
        content=content,
        section=section
    )
    
    return comment


@router.get("/comments/{store_id}")
async def get_comments(store_id: str):
    """Obtener comentarios"""
    comments = collaboration_service.get_comments(store_id)
    return {"store_id": store_id, "comments": comments}


@router.get("/dashboard")
async def get_dashboard(time_range: str = "all"):
    """Obtener dashboard"""
    # En producción, esto obtendría diseños del usuario actual
    all_designs = storage_service.list_designs()
    
    # Convertir a formato esperado
    designs = []
    for design_summary in all_designs:
        design = storage_service.load_design(design_summary["store_id"])
        if design:
            designs.append(design.dict())
    
    dashboard = dashboard_service.generate_dashboard(designs, time_range)
    return dashboard


@router.get("/templates")
async def get_templates(
    store_type: Optional[str] = None,
    style: Optional[str] = None
):
    """Obtener templates"""
    from ..core.models import StoreType, DesignStyle
    
    store_type_enum = None
    if store_type:
        try:
            store_type_enum = StoreType(store_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Tipo de tienda inválido: {store_type}")
    
    style_enum = None
    if style:
        try:
            style_enum = DesignStyle(style)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Estilo inválido: {style}")
    
    templates = template_service.get_templates(store_type_enum, style_enum)
    return {"templates": templates}


@router.get("/templates/{template_id}")
async def get_template(template_id: str):
    """Obtener template específico"""
    template = template_service.get_template(template_id)
    
    if not template:
        raise HTTPException(status_code=404, detail="Template no encontrado")
    
    return template


@router.post("/templates/{template_id}/apply")
async def apply_template(template_id: str, customizations: Optional[dict] = None):
    """Aplicar template"""
    try:
        result = template_service.apply_template(template_id, customizations)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/trends")
async def get_trends(period: str = "month"):
    """Obtener análisis de tendencias"""
    all_designs = storage_service.list_designs()
    
    designs = []
    for design_summary in all_designs:
        design = storage_service.load_design(design_summary["store_id"])
        if design:
            designs.append(design.dict())
    
    trends = trends_service.analyze_trends(designs, period)
    return trends


@router.get("/notifications/{user_id}")
async def get_notifications(user_id: str, unread_only: bool = False):
    """Obtener notificaciones"""
    notifications = notification_service.get_notifications(user_id, unread_only)
    return {
        "user_id": user_id,
        "notifications": notifications,
        "unread_count": notification_service.get_unread_count(user_id)
    }


@router.post("/notifications/{user_id}/read/{notification_id}")
async def mark_notification_read(user_id: str, notification_id: str):
    """Marcar notificación como leída"""
    marked = notification_service.mark_as_read(user_id, notification_id)
    
    if not marked:
        raise HTTPException(status_code=404, detail="Notificación no encontrada")
    
    return {"message": "Notificación marcada como leída"}


@router.post("/notifications/{user_id}/read-all")
async def mark_all_notifications_read(user_id: str):
    """Marcar todas las notificaciones como leídas"""
    count = notification_service.mark_all_as_read(user_id)
    return {"message": f"{count} notificaciones marcadas como leídas"}




