"""
Notifications API Routes - Rutas API para sistema de notificaciones
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import logging

from ..notifications.notification_system import NotificationSystem, NotificationType, NotificationPriority

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/notifications", tags=["Notifications"])

# Instancia global del sistema de notificaciones
notification_manager = NotificationSystem()


# Modelos Pydantic
class CreateTemplateRequest(BaseModel):
    name: str
    notification_type: str
    subject: str
    content: str
    variables: List[str] = Field(default_factory=list)


class CreateRecipientRequest(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    push_token: Optional[str] = None
    webhook_url: Optional[str] = None
    preferences: Dict[str, Any] = Field(default_factory=dict)


class SendNotificationRequest(BaseModel):
    notification_type: str
    subject: str
    content: str
    recipients: List[str]
    priority: str = "normal"
    template_id: Optional[str] = None
    scheduled_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SendTemplateNotificationRequest(BaseModel):
    template_id: str
    recipients: List[str]
    variables: Dict[str, str] = Field(default_factory=dict)
    priority: str = "normal"
    scheduled_at: Optional[datetime] = None


# Rutas de Plantillas
@router.post("/templates")
async def create_template(request: CreateTemplateRequest):
    """Crear plantilla de notificación."""
    try:
        notification_type = NotificationType(request.notification_type)
        
        template_id = await notification_manager.create_template(
            name=request.name,
            notification_type=notification_type,
            subject=request.subject,
            content=request.content,
            variables=request.variables
        )
        
        return {
            "template_id": template_id,
            "success": True,
            "message": "Plantilla creada exitosamente",
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Tipo de notificación inválido: {e}")
    except Exception as e:
        logger.error(f"Error al crear plantilla: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/templates")
async def get_templates():
    """Obtener todas las plantillas."""
    try:
        templates = []
        for template_id, template in notification_manager.templates.items():
            templates.append({
                "template_id": template.template_id,
                "name": template.name,
                "notification_type": template.notification_type.value,
                "subject": template.subject,
                "content": template.content,
                "variables": template.variables,
                "created_at": template.created_at.isoformat(),
                "updated_at": template.updated_at.isoformat()
            })
        
        return {
            "templates": templates,
            "count": len(templates),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al obtener plantillas: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/templates/{template_id}")
async def get_template(template_id: str):
    """Obtener plantilla específica."""
    try:
        if template_id not in notification_manager.templates:
            raise HTTPException(status_code=404, detail="Plantilla no encontrada")
        
        template = notification_manager.templates[template_id]
        
        return {
            "template": {
                "template_id": template.template_id,
                "name": template.name,
                "notification_type": template.notification_type.value,
                "subject": template.subject,
                "content": template.content,
                "variables": template.variables,
                "created_at": template.created_at.isoformat(),
                "updated_at": template.updated_at.isoformat()
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al obtener plantilla: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Rutas de Destinatarios
@router.post("/recipients")
async def create_recipient(request: CreateRecipientRequest):
    """Crear destinatario."""
    try:
        recipient_id = await notification_manager.create_recipient(
            name=request.name,
            email=request.email,
            phone=request.phone,
            push_token=request.push_token,
            webhook_url=request.webhook_url,
            preferences=request.preferences
        )
        
        return {
            "recipient_id": recipient_id,
            "success": True,
            "message": "Destinatario creado exitosamente",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al crear destinatario: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recipients")
async def get_recipients():
    """Obtener todos los destinatarios."""
    try:
        recipients = []
        for recipient_id, recipient in notification_manager.recipients.items():
            recipients.append({
                "recipient_id": recipient.recipient_id,
                "name": recipient.name,
                "email": recipient.email,
                "phone": recipient.phone,
                "push_token": recipient.push_token,
                "webhook_url": recipient.webhook_url,
                "preferences": recipient.preferences,
                "is_active": recipient.is_active
            })
        
        return {
            "recipients": recipients,
            "count": len(recipients),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al obtener destinatarios: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recipients/{recipient_id}")
async def get_recipient(recipient_id: str):
    """Obtener destinatario específico."""
    try:
        if recipient_id not in notification_manager.recipients:
            raise HTTPException(status_code=404, detail="Destinatario no encontrado")
        
        recipient = notification_manager.recipients[recipient_id]
        
        return {
            "recipient": {
                "recipient_id": recipient.recipient_id,
                "name": recipient.name,
                "email": recipient.email,
                "phone": recipient.phone,
                "push_token": recipient.push_token,
                "webhook_url": recipient.webhook_url,
                "preferences": recipient.preferences,
                "is_active": recipient.is_active
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al obtener destinatario: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Rutas de Notificaciones
@router.post("/send")
async def send_notification(request: SendNotificationRequest):
    """Enviar notificación."""
    try:
        notification_type = NotificationType(request.notification_type)
        priority = NotificationPriority(request.priority)
        
        notification_id = await notification_manager.send_notification(
            notification_type=notification_type,
            subject=request.subject,
            content=request.content,
            recipients=request.recipients,
            priority=priority,
            template_id=request.template_id,
            scheduled_at=request.scheduled_at,
            metadata=request.metadata
        )
        
        return {
            "notification_id": notification_id,
            "success": True,
            "message": "Notificación enviada exitosamente",
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Parámetro inválido: {e}")
    except Exception as e:
        logger.error(f"Error al enviar notificación: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/send-template")
async def send_template_notification(request: SendTemplateNotificationRequest):
    """Enviar notificación usando plantilla."""
    try:
        priority = NotificationPriority(request.priority)
        
        notification_id = await notification_manager.send_notification_with_template(
            template_id=request.template_id,
            recipients=request.recipients,
            variables=request.variables,
            priority=priority,
            scheduled_at=request.scheduled_at
        )
        
        return {
            "notification_id": notification_id,
            "success": True,
            "message": "Notificación con plantilla enviada exitosamente",
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Parámetro inválido: {e}")
    except Exception as e:
        logger.error(f"Error al enviar notificación con plantilla: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{notification_id}")
async def get_notification_status(notification_id: str):
    """Obtener estado de notificación."""
    try:
        status = await notification_manager.get_notification_status(notification_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Notificación no encontrada")
        
        return {
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al obtener estado de notificación: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_notification_stats():
    """Obtener estadísticas de notificaciones."""
    try:
        stats = await notification_manager.get_notification_stats()
        
        return {
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al obtener estadísticas de notificaciones: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def notifications_health_check():
    """Verificar salud del sistema de notificaciones."""
    try:
        health = await notification_manager.health_check()
        
        return {
            "health": health,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error en health check de notificaciones: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Rutas de utilidad
@router.get("/types")
async def get_notification_types():
    """Obtener tipos de notificaciones disponibles."""
    return {
        "notification_types": [
            {
                "value": notification_type.value,
                "name": notification_type.name,
                "description": f"Notificación de tipo {notification_type.value}"
            }
            for notification_type in NotificationType
        ],
        "timestamp": datetime.now().isoformat()
    }


@router.get("/priorities")
async def get_notification_priorities():
    """Obtener prioridades de notificaciones disponibles."""
    return {
        "priorities": [
            {
                "value": priority.value,
                "name": priority.name,
                "description": f"Prioridad {priority.value}"
            }
            for priority in NotificationPriority
        ],
        "timestamp": datetime.now().isoformat()
    }


# Rutas de ejemplo
@router.post("/examples/send-alert")
async def send_alert_example(
    alert_type: str,
    severity: str,
    message: str,
    recipients: List[str]
):
    """Ejemplo: Enviar alerta del sistema."""
    try:
        notification_id = await notification_manager.send_notification_with_template(
            template_id="alert_notification",
            recipients=recipients,
            variables={
                "alert_type": alert_type,
                "severity": severity,
                "message": message,
                "timestamp": datetime.now().isoformat()
            },
            priority=NotificationPriority.HIGH
        )
        
        return {
            "notification_id": notification_id,
            "success": True,
            "message": f"Alerta '{alert_type}' enviada a {len(recipients)} destinatarios",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al enviar alerta: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/examples/send-welcome")
async def send_welcome_example(
    user_name: str,
    user_email: str
):
    """Ejemplo: Enviar email de bienvenida."""
    try:
        # Crear destinatario temporal
        recipient_id = await notification_manager.create_recipient(
            name=user_name,
            email=user_email
        )
        
        # Enviar notificación de bienvenida
        notification_id = await notification_manager.send_notification_with_template(
            template_id="default_email",
            recipients=[recipient_id],
            variables={
                "name": user_name,
                "message": "¡Bienvenido al sistema Export IA! Estamos emocionados de tenerte a bordo."
            },
            priority=NotificationPriority.NORMAL
        )
        
        return {
            "notification_id": notification_id,
            "recipient_id": recipient_id,
            "success": True,
            "message": f"Email de bienvenida enviado a {user_name}",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al enviar email de bienvenida: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/examples/send-webhook")
async def send_webhook_example(
    webhook_url: str,
    event_type: str,
    message: str
):
    """Ejemplo: Enviar webhook."""
    try:
        # Crear destinatario temporal
        recipient_id = await notification_manager.create_recipient(
            name="Webhook Recipient",
            webhook_url=webhook_url
        )
        
        # Enviar notificación webhook
        notification_id = await notification_manager.send_notification_with_template(
            template_id="default_webhook",
            recipients=[recipient_id],
            variables={
                "event_type": event_type,
                "message": message,
                "timestamp": datetime.now().isoformat()
            },
            priority=NotificationPriority.NORMAL
        )
        
        return {
            "notification_id": notification_id,
            "recipient_id": recipient_id,
            "success": True,
            "message": f"Webhook enviado a {webhook_url}",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al enviar webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))




