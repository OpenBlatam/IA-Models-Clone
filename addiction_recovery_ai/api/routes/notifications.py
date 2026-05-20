"""
Notifications routes - Refactored with best practices
"""

from fastapi import APIRouter, HTTPException, status

try:
    from schemas.notifications import (
        NotificationResponse,
        NotificationsListResponse,
        ReminderResponse,
        RemindersListResponse
    )
    from schemas.common import ErrorResponse, SuccessResponse
    from dependencies import NotificationServiceDep
except ImportError:
    from ...schemas.notifications import (
        NotificationResponse,
        NotificationsListResponse,
        ReminderResponse,
        RemindersListResponse
    )
    from ...schemas.common import ErrorResponse, SuccessResponse
    from ...dependencies import NotificationServiceDep

router = APIRouter(prefix="/notifications", tags=["Notifications"])


@router.get(
    "/{user_id}",
    response_model=NotificationsListResponse,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse, "description": "User not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def get_notifications(
    user_id: str,
    notification_service: NotificationServiceDep
) -> NotificationsListResponse:
    """
    Obtiene notificaciones pendientes del usuario
    
    - **user_id**: ID del usuario
    """
    # Guard clause: Validate user_id
    if not user_id or not user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id es requerido"
        )
    
    try:
        pending = notification_service.get_pending_notifications(user_id)
        
        notifications = [
            NotificationResponse(
                notification_id=n.get("notification_id", ""),
                user_id=user_id,
                title=n.get("title", ""),
                message=n.get("message", ""),
                type=n.get("type", "info"),
                priority=n.get("priority", "normal"),
                read=n.get("read", False),
                created_at=n.get("created_at"),
                read_at=n.get("read_at")
            )
            for n in pending
        ]
        
        unread_count = len([n for n in notifications if not n.read])
        
        return NotificationsListResponse(
            user_id=user_id,
            notifications=notifications,
            unread_count=unread_count,
            total_count=len(notifications)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo notificaciones: {str(e)}"
        )


@router.post(
    "/{notification_id}/read",
    response_model=SuccessResponse,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse, "description": "Notification not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def mark_notification_read(
    notification_id: str,
    notification_service: NotificationServiceDep
) -> SuccessResponse:
    """
    Marca una notificación como leída
    
    - **notification_id**: ID de la notificación
    """
    # Guard clause: Validate notification_id
    if not notification_id or not notification_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="notification_id es requerido"
        )
    
    try:
        success = notification_service.mark_notification_read(notification_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Notificación no encontrada"
            )
        
        return SuccessResponse(
            message="Notificación marcada como leída"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error marcando notificación: {str(e)}"
        )


@router.get(
    "/reminders/{user_id}",
    response_model=RemindersListResponse,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse, "description": "User not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def get_reminders(
    user_id: str,
    notification_service: NotificationServiceDep
) -> RemindersListResponse:
    """
    Obtiene recordatorios diarios del usuario
    
    - **user_id**: ID del usuario
    """
    # Guard clause: Validate user_id
    if not user_id or not user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id es requerido"
        )
    
    try:
        reminders_data = notification_service.get_daily_reminders(user_id)
        
        reminders = [
            ReminderResponse(
                reminder_id=r.get("reminder_id", ""),
                user_id=user_id,
                title=r.get("title", ""),
                message=r.get("message", ""),
                scheduled_time=r.get("scheduled_time"),
                completed=r.get("completed", False),
                created_at=r.get("created_at")
            )
            for r in reminders_data
        ]
        
        from datetime import datetime
        upcoming_count = len([
            r for r in reminders
            if not r.completed and r.scheduled_time > datetime.now()
        ])
        
        return RemindersListResponse(
            user_id=user_id,
            reminders=reminders,
            upcoming_count=upcoming_count
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo recordatorios: {str(e)}"
        )

