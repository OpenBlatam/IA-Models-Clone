from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
import logging

from onyx.auth.users import current_user
from onyx.db.engine import get_session
from onyx.db.models import User
from onyx.db.notification import dismiss_notification
from onyx.db.notification import get_notification_by_id
from onyx.db.notification import get_notifications
from onyx.server.settings.models import Notification as NotificationModel
from onyx.utils.logger import setup_logger
from ..utils.error_system import (
from typing import Any, List, Dict, Optional
import asyncio
    error_factory, 
    ErrorContext, 
    ValidationError, 
    ResourceNotFoundError, 
    AuthorizationError,
    SystemError,
    handle_errors,
    ErrorCategory
)

logger = setup_logger(__name__)

router = APIRouter(prefix="/notifications")


@router.get("", response_model=List[NotificationModel])
@handle_errors(ErrorCategory.DATABASE, operation="get_notifications")
def get_notifications_api(
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> List[NotificationModel]:
    """
    Get user notifications with proper error handling and logging.
    
    Args:
        user: Current authenticated user
        db_session: Database session
        
    Returns:
        List of user notifications
        
    Raises:
        HTTPException: If there's an error retrieving notifications
    """
    try:
        logger.info(f"Retrieving notifications for user {user.id}")
        
        notifications = [
            NotificationModel.from_model(notif)
            for notif in get_notifications(user, db_session, include_dismissed=False)
        ]
        
        logger.info(f"Successfully retrieved {len(notifications)} notifications for user {user.id}")
        return notifications
        
    except Exception as e:
        context = ErrorContext(
            user_id=str(user.id),
            operation="get_notifications",
            resource_type="notifications"
        )
        
        raise error_factory.create_system_error(
            f"Failed to retrieve notifications: {str(e)}",
            component="notifications_api",
            context=context,
            original_exception=e
        )


@router.post("/{notification_id}/dismiss", status_code=status.HTTP_200_OK)
@handle_errors(ErrorCategory.DATABASE, operation="dismiss_notification")
def dismiss_notification_endpoint(
    notification_id: int,
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> dict:
    """
    Dismiss a notification with proper error handling and logging.
    
    Args:
        notification_id: ID of the notification to dismiss
        user: Current authenticated user
        db_session: Database session
        
    Returns:
        Success message
        
    Raises:
        HTTPException: If there's an error dismissing the notification
    """
    try:
        logger.info(f"User {user.id} attempting to dismiss notification {notification_id}")
        
        # Validate notification ID
        if notification_id <= 0:
            context = ErrorContext(
                user_id=str(user.id),
                operation="dismiss_notification",
                resource_type="notification",
                resource_id=str(notification_id)
            )
            
            raise error_factory.create_validation_error(
                "Invalid notification ID provided",
                field="notification_id",
                value=notification_id,
                context=context
            )
        
        notification = get_notification_by_id(notification_id, user, db_session)
        
        # Check if notification exists
        if not notification:
            context = ErrorContext(
                user_id=str(user.id),
                operation="dismiss_notification",
                resource_type="notification",
                resource_id=str(notification_id)
            )
            
            raise error_factory.create_resource_not_found_error(
                "Notification not found",
                resource_type="notification",
                resource_id=str(notification_id),
                context=context
            )
        
        dismiss_notification(notification, db_session)
        
        logger.info(f"Successfully dismissed notification {notification_id} for user {user.id}")
        return {
            "message": "Notification dismissed successfully",
            "notification_id": notification_id
        }
        
    except (ValidationError, ResourceNotFoundError, AuthorizationError) as e:
        # Convert Onyx errors to HTTP exceptions
        if isinstance(e, ValidationError):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=e.user_friendly_message
            )
        elif isinstance(e, ResourceNotFoundError):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=e.user_friendly_message
            )
        elif isinstance(e, AuthorizationError):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=e.user_friendly_message
            )
        
    except PermissionError as e:
        context = ErrorContext(
            user_id=str(user.id),
            operation="dismiss_notification",
            resource_type="notification",
            resource_id=str(notification_id)
        )
        
        raise error_factory.create_authorization_error(
            f"Permission denied to dismiss notification: {str(e)}",
            required_permission="dismiss_notification",
            context=context,
            original_exception=e
        )
        
    except ValueError as e:
        context = ErrorContext(
            user_id=str(user.id),
            operation="dismiss_notification",
            resource_type="notification",
            resource_id=str(notification_id)
        )
        
        raise error_factory.create_resource_not_found_error(
            f"Notification not found: {str(e)}",
            resource_type="notification",
            resource_id=str(notification_id),
            context=context,
            original_exception=e
        )
        
    except Exception as e:
        context = ErrorContext(
            user_id=str(user.id),
            operation="dismiss_notification",
            resource_type="notification",
            resource_id=str(notification_id)
        )
        
        raise error_factory.create_system_error(
            f"Failed to dismiss notification: {str(e)}",
            component="notifications_api",
            context=context,
            original_exception=e
        )
