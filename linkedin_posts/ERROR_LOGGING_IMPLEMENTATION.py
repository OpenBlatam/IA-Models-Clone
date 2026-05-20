from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
import logging
import traceback
import json
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Dict, Optional
"""
Error Logging Implementation: Proper Error Logging and User-Friendly Messages

This module demonstrates how to implement comprehensive error logging
with user-friendly error messages for production systems.
"""


# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# ERROR LOGGING UTILITIES
# ============================================================================

class ErrorLogger:
    """Centralized error logging utility"""
    
    @staticmethod
    def log_error(
        error: Exception,
        context: Dict[str, Any],
        user_id: Optional[str] = None,
        operation: str = "unknown",
        severity: str = "ERROR"
    ) -> str:
        """Log error with structured context and return error ID"""
        error_id = f"err_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(error))}"
        
        error_data = {
            "error_id": error_id,
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "user_id": user_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "traceback": traceback.format_exc(),
            "severity": severity
        }
        
        # Log the error
        logger.error(f"Error ID: {error_id} | Operation: {operation} | User: {user_id} | Error: {error}")
        logger.debug(f"Full error context: {json.dumps(error_data, indent=2)}")
        
        # Store error for monitoring
        ErrorMonitor.record_error(error_data)
        
        return error_id
    
    @staticmethod
    def log_warning(
        message: str,
        context: Dict[str, Any],
        user_id: Optional[str] = None,
        operation: str = "unknown"
    ) -> None:
        """Log warning with context"""
        logger.warning(f"Warning | Operation: {operation} | User: {user_id} | Message: {message}")
        logger.debug(f"Warning context: {context}")
    
    @staticmethod
    def log_info(
        message: str,
        context: Dict[str, Any],
        user_id: Optional[str] = None,
        operation: str = "unknown"
    ) -> None:
        """Log info with context"""
        logger.info(f"Info | Operation: {operation} | User: {user_id} | Message: {message}")
        logger.debug(f"Info context: {context}")

class ErrorMonitor:
    """Error monitoring and alerting system"""
    
    error_counts = {}
    alert_thresholds = {
        "ERROR": 10,      # Alert after 10 errors in 5 minutes
        "WARNING": 50,    # Alert after 50 warnings in 5 minutes
        "CRITICAL": 5     # Alert after 5 critical errors in 1 minute
    }
    alert_cooldown = timedelta(minutes=5)
    last_alerts = {}
    
    @classmethod
    def record_error(cls, error_data: Dict[str, Any]) -> None:
        """Record error for monitoring"""
        error_key = f"{error_data['operation']}_{error_data['error_type']}"
        current_time = datetime.now()
        
        if error_key not in cls.error_counts:
            cls.error_counts[error_key] = []
        
        # Add error with timestamp
        cls.error_counts[error_key].append({
            "timestamp": current_time,
            "severity": error_data["severity"],
            "user_id": error_data.get("user_id"),
            "error_id": error_data["error_id"]
        })
        
        # Clean old errors (older than 1 hour)
        cutoff_time = current_time - timedelta(hours=1)
        cls.error_counts[error_key] = [
            error for error in cls.error_counts[error_key]
            if error["timestamp"] > cutoff_time
        ]
        
        # Check for alerts
        cls.check_alerts(error_key, error_data["severity"])
    
    @classmethod
    def check_alerts(cls, error_key: str, severity: str) -> None:
        """Check if alerts should be triggered"""
        current_time = datetime.now()
        errors = cls.error_counts.get(error_key, [])
        
        # Count recent errors by severity
        recent_errors = [
            error for error in errors
            if error["timestamp"] > current_time - timedelta(minutes=5)
        ]
        
        severity_count = len([e for e in recent_errors if e["severity"] == severity])
        threshold = cls.alert_thresholds.get(severity, 10)
        
        if severity_count >= threshold:
            # Check cooldown
            last_alert = cls.last_alerts.get(error_key, datetime.min)
            if current_time - last_alert > cls.alert_cooldown:
                cls.trigger_alert(error_key, severity, severity_count, recent_errors)
                cls.last_alerts[error_key] = current_time
    
    @classmethod
    def trigger_alert(cls, error_key: str, severity: str, count: int, errors: list) -> None:
        """Trigger alert for error threshold exceeded"""
        alert_data = {
            "alert_id": f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "error_key": error_key,
            "severity": severity,
            "error_count": count,
            "threshold": cls.alert_thresholds[severity],
            "recent_errors": errors[:5]  # Last 5 errors
        }
        
        logger.critical(f"ALERT: {severity} threshold exceeded for {error_key} | Count: {count}")
        
        # Send alert to monitoring system (implementation would go here)
        cls.send_alert_to_monitoring_system(alert_data)
    
    @classmethod
    def send_alert_to_monitoring_system(cls, alert_data: Dict[str, Any]) -> None:
        """Send alert to external monitoring system"""
        # Implementation for sending to monitoring system (e.g., Sentry, DataDog, etc.)
        logger.info(f"Alert sent to monitoring system: {alert_data['alert_id']}")

class UserFriendlyMessages:
    """Centralized user-friendly error messages"""
    
    MESSAGES = {
        # Authentication errors
        "USER_NOT_FOUND": "We couldn't find your account. Please check your login and try again.",
        "ACCOUNT_DEACTIVATED": "Your account has been deactivated. Please contact support for assistance.",
        "INVALID_CREDENTIALS": "The provided credentials are incorrect. Please check and try again.",
        
        # Authorization errors
        "UNAUTHORIZED_ACCESS": "You don't have permission to perform this action.",
        "POSTS_PRIVATE": "This content is private and not available for viewing.",
        "ACCESS_DENIED": "Access denied. Please contact the content owner for permission.",
        
        # Validation errors
        "CONTENT_TOO_SHORT": "Your post is too short. Please add more content (minimum 10 characters).",
        "CONTENT_TOO_LONG": "Your post is too long. Please shorten it (maximum 3000 characters).",
        "INVALID_FILE_TYPE": "This file type is not supported. Please use images only (JPEG, PNG, GIF, WebP).",
        "FILE_TOO_LARGE": "File is too large. Please use a file smaller than 5MB.",
        "MISSING_USER_ID": "Please provide a valid user ID.",
        "MISSING_CONTENT": "Please provide post content.",
        "MISSING_POST_ID": "Please provide a valid post ID.",
        
        # Business rule errors
        "DAILY_LIMIT_EXCEEDED": "You've reached your daily post limit. Please try again tomorrow.",
        "RATE_LIMIT_EXCEEDED": "You're posting too quickly. Please wait a moment before trying again.",
        "DUPLICATE_CONTENT": "This content appears to be a duplicate. Please create unique content.",
        "EDIT_TIME_EXPIRED": "This post can no longer be edited (24-hour limit).",
        "POST_BEING_EDITED": "This post is currently being edited. Please try again in a moment.",
        "POST_DELETED": "This post has been deleted and cannot be modified.",
        
        # System errors
        "DATABASE_ERROR": "We're experiencing technical difficulties. Please try again in a few minutes.",
        "NETWORK_ERROR": "Connection issue detected. Please check your internet and try again.",
        "SERVICE_UNAVAILABLE": "This service is temporarily unavailable. Please try again later.",
        "UNKNOWN_ERROR": "Something unexpected happened. Please try again or contact support if the problem persists.",
        "POST_CREATION_FAILED": "Unable to create your post. Please try again later.",
        "POST_UPDATE_FAILED": "Unable to update your post. Please try again later.",
        "FETCH_FAILED": "Unable to retrieve the requested information. Please try again later."
    }
    
    @staticmethod
    def get_message(error_code: str, default: str = None) -> str:
        """Get user-friendly message for error code"""
        return UserFriendlyMessages.MESSAGES.get(error_code, default or "An error occurred. Please try again.")

class ErrorResponse:
    """Standardized error response structure"""
    
    @staticmethod
    def create(
        error_code: str,
        user_message: str,
        technical_message: str = None,
        error_id: str = None,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Create standardized error response"""
        response = {
            "status": "failed",
            "error": {
                "code": error_code,
                "message": user_message,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        if error_id:
            response["error"]["id"] = error_id
        
        if context:
            response["error"]["context"] = context
        
        # Only include technical message in development
        if technical_message and os.getenv("ENVIRONMENT") == "development":
            response["error"]["technical"] = technical_message
        
        return response
    
    @staticmethod
    def success(data: Dict[str, Any], message: str = None) -> Dict[str, Any]:
        """Create standardized success response"""
        response = {
            "status": "success",
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        if message:
            response["message"] = message
        
        return response

# ============================================================================
# MOCK DATABASE FUNCTIONS
# ============================================================================

class DatabaseConnectionError(Exception):
    """Custom exception for database connection errors"""
    pass

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

async def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    """Mock function to get user by ID"""
    await asyncio.sleep(0.01)
    
    # Simulate database connection error
    if user_id == "db_error_user":
        raise DatabaseConnectionError("Database connection failed")
    
    if user_id == "valid_user":
        return {
            "id": user_id,
            "name": "John Doe",
            "is_active": True,
            "email": "john@example.com",
            "role": "user",
            "posts_public": True
        }
    elif user_id == "inactive_user":
        return {
            "id": user_id,
            "name": "Jane Doe",
            "is_active": False,
            "email": "jane@example.com",
            "role": "user",
            "posts_public": True
        }
    elif user_id == "private_user":
        return {
            "id": user_id,
            "name": "Private User",
            "is_active": True,
            "email": "private@example.com",
            "role": "user",
            "posts_public": False
        }
    return None

async def get_post_by_id(post_id: str) -> Optional[Dict[str, Any]]:
    """Mock function to get post by ID"""
    await asyncio.sleep(0.01)
    
    # Simulate database connection error
    if post_id == "db_error_post":
        raise DatabaseConnectionError("Database connection failed")
    
    if post_id == "valid_post":
        return {
            "id": post_id,
            "user_id": "valid_user",
            "content": "Original content",
            "created_at": datetime.now() - timedelta(hours=2),
            "is_deleted": False,
            "is_public": True,
            "is_being_edited": False,
            "status": "published"
        }
    elif post_id == "deleted_post":
        return {
            "id": post_id,
            "user_id": "valid_user",
            "content": "Deleted content",
            "created_at": datetime.now() - timedelta(days=1),
            "is_deleted": True,
            "is_public": True,
            "is_being_edited": False,
            "status": "deleted"
        }
    elif post_id == "editing_post":
        return {
            "id": post_id,
            "user_id": "valid_user",
            "content": "Content being edited",
            "created_at": datetime.now() - timedelta(hours=1),
            "is_deleted": False,
            "is_public": True,
            "is_being_edited": True,
            "status": "published"
        }
    return None

async def create_post_in_database(user_id: str, content: str, hashtags: List[str] = None) -> Dict[str, Any]:
    """Mock function to create post in database"""
    await asyncio.sleep(0.05)
    
    # Simulate database error for specific content
    if "error" in content.lower():
        raise DatabaseConnectionError("Failed to create post in database")
    
    return {
        "id": f"post_{uuid.uuid4().hex[:8]}",
        "user_id": user_id,
        "content": content,
        "hashtags": hashtags or [],
        "created_at": datetime.now(),
        "status": "published"
    }

async def update_post_in_database(post_id: str, new_content: str) -> Dict[str, Any]:
    """Mock function to update post in database"""
    await asyncio.sleep(0.05)
    
    # Simulate database error for specific content
    if "error" in new_content.lower():
        raise DatabaseConnectionError("Failed to update post in database")
    
    return {
        "id": post_id,
        "content": new_content,
        "updated_at": datetime.now()
    }

async def get_posts_from_database(user_id: str) -> List[Dict[str, Any]]:
    """Mock function to get posts from database"""
    await asyncio.sleep(0.02)
    
    # Simulate database error for specific user
    if user_id == "db_error_user":
        raise DatabaseConnectionError("Failed to fetch posts from database")
    
    return [
        {"id": "post1", "content": "First post", "created_at": datetime.now()},
        {"id": "post2", "content": "Second post", "created_at": datetime.now()}
    ]

async def is_duplicate_content(content: str, user_id: str) -> bool:
    """Mock function to check for duplicate content"""
    await asyncio.sleep(0.01)
    return "duplicate" in content.lower()

async def check_rate_limit(user_id: str, action: str) -> bool:
    """Mock function to check rate limit"""
    await asyncio.sleep(0.01)
    return user_id != "rate_limited_user"

# ============================================================================
# ERROR LOGGING IMPLEMENTATIONS
# ============================================================================

class PostService:
    """Service class demonstrating proper error logging"""
    
    @staticmethod
    async def create_post_with_error_logging(user_id: str, content: str, hashtags: List[str] = None) -> Dict[str, Any]:
        """
        Create a new post with comprehensive error logging.
        
        Demonstrates proper error logging with user-friendly messages.
        """
        operation = "create_post"
        context = {
            "user_id": user_id,
            "content_length": len(content) if content else 0,
            "hashtags_count": len(hashtags) if hashtags else 0
        }
        
        try:
            # ============================================================================
            # VALIDATION PHASE (Early returns with proper error responses)
            # ============================================================================
            
            # Input validation
            if not user_id:
                return ErrorResponse.create(
                    error_code="MISSING_USER_ID",
                    user_message=UserFriendlyMessages.get_message("MISSING_USER_ID")
                )
            
            if not content:
                return ErrorResponse.create(
                    error_code="MISSING_CONTENT",
                    user_message=UserFriendlyMessages.get_message("MISSING_CONTENT")
                )
            
            # Content validation
            content = content.strip()
            if len(content) < 10:
                return ErrorResponse.create(
                    error_code="CONTENT_TOO_SHORT",
                    user_message=UserFriendlyMessages.get_message("CONTENT_TOO_SHORT")
                )
            
            if len(content) > 3000:
                return ErrorResponse.create(
                    error_code="CONTENT_TOO_LONG",
                    user_message=UserFriendlyMessages.get_message("CONTENT_TOO_LONG")
                )
            
            # User validation
            user = await get_user_by_id(user_id)
            if not user:
                error_id = ErrorLogger.log_error(
                    error=ValueError(f"User not found: {user_id}"),
                    context=context,
                    user_id=user_id,
                    operation=operation,
                    severity="WARNING"
                )
                return ErrorResponse.create(
                    error_code="USER_NOT_FOUND",
                    user_message=UserFriendlyMessages.get_message("USER_NOT_FOUND"),
                    technical_message=f"User not found in database: {user_id}",
                    error_id=error_id,
                    context={"user_id": user_id}
                )
            
            if not user["is_active"]:
                error_id = ErrorLogger.log_error(
                    error=ValueError(f"Account deactivated: {user_id}"),
                    context=context,
                    user_id=user_id,
                    operation=operation,
                    severity="WARNING"
                )
                return ErrorResponse.create(
                    error_code="ACCOUNT_DEACTIVATED",
                    user_message=UserFriendlyMessages.get_message("ACCOUNT_DEACTIVATED"),
                    error_id=error_id
                )
            
            # Business rule validation
            if await is_duplicate_content(content, user_id):
                ErrorLogger.log_warning(
                    message=f"Duplicate content detected for user {user_id}",
                    context=context,
                    user_id=user_id,
                    operation=operation
                )
                return ErrorResponse.create(
                    error_code="DUPLICATE_CONTENT",
                    user_message=UserFriendlyMessages.get_message("DUPLICATE_CONTENT")
                )
            
            if not await check_rate_limit(user_id, "post_creation"):
                ErrorLogger.log_warning(
                    message=f"Rate limit exceeded for user {user_id}",
                    context=context,
                    user_id=user_id,
                    operation=operation
                )
                return ErrorResponse.create(
                    error_code="RATE_LIMIT_EXCEEDED",
                    user_message=UserFriendlyMessages.get_message("RATE_LIMIT_EXCEEDED")
                )
            
            # ============================================================================
            # MAIN BUSINESS LOGIC
            # ============================================================================
            
            post = await create_post_in_database(user_id, content, hashtags)
            
            # Log success
            ErrorLogger.log_info(
                message=f"Post created successfully | Post ID: {post['id']}",
                context={**context, "post_id": post["id"]},
                user_id=user_id,
                operation=operation
            )
            
            return ErrorResponse.success(
                data={"post_id": post["id"], "created_at": post["created_at"].isoformat()},
                message="Post created successfully"
            )
            
        except DatabaseConnectionError as e:
            error_id = ErrorLogger.log_error(
                error=e,
                context=context,
                user_id=user_id,
                operation=operation,
                severity="ERROR"
            )
            
            return ErrorResponse.create(
                error_code="DATABASE_ERROR",
                user_message=UserFriendlyMessages.get_message("DATABASE_ERROR"),
                technical_message=str(e),
                error_id=error_id
            )
            
        except Exception as e:
            error_id = ErrorLogger.log_error(
                error=e,
                context=context,
                user_id=user_id,
                operation=operation,
                severity="ERROR"
            )
            
            return ErrorResponse.create(
                error_code="POST_CREATION_FAILED",
                user_message=UserFriendlyMessages.get_message("POST_CREATION_FAILED"),
                technical_message=str(e),
                error_id=error_id
            )
    
    @staticmethod
    async def update_post_with_error_logging(post_id: str, user_id: str, new_content: str) -> Dict[str, Any]:
        """
        Update post with comprehensive error logging.
        
        Demonstrates proper error logging for update operations.
        """
        operation = "update_post"
        context = {
            "post_id": post_id,
            "user_id": user_id,
            "content_length": len(new_content) if new_content else 0
        }
        
        try:
            # ============================================================================
            # VALIDATION PHASE
            # ============================================================================
            
            # Input validation
            if not post_id:
                return ErrorResponse.create(
                    error_code="MISSING_POST_ID",
                    user_message=UserFriendlyMessages.get_message("MISSING_POST_ID")
                )
            
            if not user_id:
                return ErrorResponse.create(
                    error_code="MISSING_USER_ID",
                    user_message=UserFriendlyMessages.get_message("MISSING_USER_ID")
                )
            
            if not new_content:
                return ErrorResponse.create(
                    error_code="MISSING_CONTENT",
                    user_message=UserFriendlyMessages.get_message("MISSING_CONTENT")
                )
            
            # Content validation
            new_content = new_content.strip()
            if len(new_content) < 10:
                return ErrorResponse.create(
                    error_code="CONTENT_TOO_SHORT",
                    user_message=UserFriendlyMessages.get_message("CONTENT_TOO_SHORT")
                )
            
            if len(new_content) > 3000:
                return ErrorResponse.create(
                    error_code="CONTENT_TOO_LONG",
                    user_message=UserFriendlyMessages.get_message("CONTENT_TOO_LONG")
                )
            
            # Post validation
            post = await get_post_by_id(post_id)
            if not post:
                error_id = ErrorLogger.log_error(
                    error=ValueError(f"Post not found: {post_id}"),
                    context=context,
                    user_id=user_id,
                    operation=operation,
                    severity="WARNING"
                )
                return ErrorResponse.create(
                    error_code="POST_NOT_FOUND",
                    user_message="The post you're trying to edit doesn't exist",
                    technical_message=f"Post not found in database: {post_id}",
                    error_id=error_id
                )
            
            # Authorization
            if post["user_id"] != user_id:
                error_id = ErrorLogger.log_error(
                    error=PermissionError(f"Unauthorized post update attempt | User: {user_id} | Post: {post_id}"),
                    context=context,
                    user_id=user_id,
                    operation=operation,
                    severity="WARNING"
                )
                return ErrorResponse.create(
                    error_code="UNAUTHORIZED_UPDATE",
                    user_message=UserFriendlyMessages.get_message("UNAUTHORIZED_ACCESS"),
                    error_id=error_id
                )
            
            # State validation
            if post["is_deleted"]:
                return ErrorResponse.create(
                    error_code="POST_DELETED",
                    user_message=UserFriendlyMessages.get_message("POST_DELETED")
                )
            
            if post["is_being_edited"]:
                return ErrorResponse.create(
                    error_code="POST_BEING_EDITED",
                    user_message=UserFriendlyMessages.get_message("POST_BEING_EDITED")
                )
            
            # Time-based validation
            post_age = datetime.now() - post["created_at"]
            if post_age > timedelta(hours=24):
                return ErrorResponse.create(
                    error_code="EDIT_TIME_EXPIRED",
                    user_message=UserFriendlyMessages.get_message("EDIT_TIME_EXPIRED")
                )
            
            # ============================================================================
            # MAIN BUSINESS LOGIC
            # ============================================================================
            
            updated_post = await update_post_in_database(post_id, new_content)
            
            # Log success
            ErrorLogger.log_info(
                message=f"Post updated successfully | Post ID: {post_id}",
                context={**context, "updated_at": updated_post["updated_at"].isoformat()},
                user_id=user_id,
                operation=operation
            )
            
            return ErrorResponse.success(
                data={"post_id": post_id, "updated_at": updated_post["updated_at"].isoformat()},
                message="Post updated successfully"
            )
            
        except DatabaseConnectionError as e:
            error_id = ErrorLogger.log_error(
                error=e,
                context=context,
                user_id=user_id,
                operation=operation,
                severity="ERROR"
            )
            
            return ErrorResponse.create(
                error_code="DATABASE_ERROR",
                user_message=UserFriendlyMessages.get_message("DATABASE_ERROR"),
                technical_message=str(e),
                error_id=error_id
            )
            
        except Exception as e:
            error_id = ErrorLogger.log_error(
                error=e,
                context=context,
                user_id=user_id,
                operation=operation,
                severity="ERROR"
            )
            
            return ErrorResponse.create(
                error_code="POST_UPDATE_FAILED",
                user_message=UserFriendlyMessages.get_message("POST_UPDATE_FAILED"),
                technical_message=str(e),
                error_id=error_id
            )
    
    @staticmethod
    async def get_user_posts_with_error_logging(user_id: str, requester_id: str) -> Dict[str, Any]:
        """
        Get user posts with comprehensive error logging.
        
        Demonstrates proper error logging for read operations.
        """
        operation = "get_user_posts"
        context = {"user_id": user_id, "requester_id": requester_id}
        
        try:
            # ============================================================================
            # VALIDATION PHASE
            # ============================================================================
            
            # Input validation
            if not requester_id:
                return ErrorResponse.create(
                    error_code="MISSING_REQUESTER_ID",
                    user_message="Please provide your user ID"
                )
            
            if not user_id:
                return ErrorResponse.create(
                    error_code="MISSING_USER_ID",
                    user_message=UserFriendlyMessages.get_message("MISSING_USER_ID")
                )
            
            # Authentication
            requester = await get_user_by_id(requester_id)
            if not requester:
                error_id = ErrorLogger.log_error(
                    error=ValueError(f"Requester not found: {requester_id}"),
                    context=context,
                    user_id=requester_id,
                    operation=operation,
                    severity="WARNING"
                )
                return ErrorResponse.create(
                    error_code="USER_NOT_FOUND",
                    user_message=UserFriendlyMessages.get_message("USER_NOT_FOUND"),
                    error_id=error_id
                )
            
            if not requester["is_active"]:
                error_id = ErrorLogger.log_error(
                    error=ValueError(f"Requester account deactivated: {requester_id}"),
                    context=context,
                    user_id=requester_id,
                    operation=operation,
                    severity="WARNING"
                )
                return ErrorResponse.create(
                    error_code="ACCOUNT_DEACTIVATED",
                    user_message=UserFriendlyMessages.get_message("ACCOUNT_DEACTIVATED"),
                    error_id=error_id
                )
            
            # Authorization
            if requester_id != user_id and not requester["posts_public"]:
                error_id = ErrorLogger.log_error(
                    error=PermissionError(f"Unauthorized access attempt | Requester: {requester_id} | Target: {user_id}"),
                    context=context,
                    user_id=requester_id,
                    operation=operation,
                    severity="WARNING"
                )
                return ErrorResponse.create(
                    error_code="POSTS_PRIVATE",
                    user_message=UserFriendlyMessages.get_message("POSTS_PRIVATE"),
                    error_id=error_id
                )
            
            # ============================================================================
            # MAIN BUSINESS LOGIC
            # ============================================================================
            
            posts = await get_posts_from_database(user_id)
            
            # Log success
            ErrorLogger.log_info(
                message=f"Posts retrieved successfully | User: {user_id} | Count: {len(posts)}",
                context={**context, "posts_count": len(posts)},
                user_id=requester_id,
                operation=operation
            )
            
            return ErrorResponse.success(
                data={"posts": posts, "count": len(posts)},
                message="Posts retrieved successfully"
            )
            
        except DatabaseConnectionError as e:
            error_id = ErrorLogger.log_error(
                error=e,
                context=context,
                user_id=requester_id,
                operation=operation,
                severity="ERROR"
            )
            
            return ErrorResponse.create(
                error_code="DATABASE_ERROR",
                user_message=UserFriendlyMessages.get_message("DATABASE_ERROR"),
                technical_message=str(e),
                error_id=error_id
            )
            
        except Exception as e:
            error_id = ErrorLogger.log_error(
                error=e,
                context=context,
                user_id=requester_id,
                operation=operation,
                severity="ERROR"
            )
            
            return ErrorResponse.create(
                error_code="FETCH_FAILED",
                user_message=UserFriendlyMessages.get_message("FETCH_FAILED"),
                technical_message=str(e),
                error_id=error_id
            )

# ============================================================================
# GRACEFUL DEGRADATION EXAMPLE
# ============================================================================

class PostServiceWithFallback:
    """Service class demonstrating graceful degradation"""
    
    @staticmethod
    async def create_post_fallback(user_id: str, content: str) -> Dict[str, Any]:
        """Fallback method for post creation"""
        await asyncio.sleep(0.1)  # Simulate slower fallback
        return {
            "id": f"fallback_post_{uuid.uuid4().hex[:8]}",
            "user_id": user_id,
            "content": content,
            "created_at": datetime.now(),
            "status": "published",
            "fallback_used": True
        }
    
    @staticmethod
    async def create_post_with_fallback(user_id: str, content: str) -> Dict[str, Any]:
        """
        Create post with graceful degradation.
        
        Demonstrates error recovery and fallback mechanisms.
        """
        operation = "create_post_with_fallback"
        context = {"user_id": user_id, "content_length": len(content) if content else 0}
        
        try:
            # Primary operation
            post = await create_post_in_database(user_id, content)
            
            ErrorLogger.log_info(
                message=f"Post created successfully (primary) | Post ID: {post['id']}",
                context={**context, "post_id": post["id"]},
                user_id=user_id,
                operation=operation
            )
            
            return ErrorResponse.success(
                data={"post_id": post["id"], "fallback_used": False},
                message="Post created successfully"
            )
            
        except DatabaseConnectionError as e:
            # Log the primary error
            ErrorLogger.log_error(
                error=e,
                context=context,
                user_id=user_id,
                operation=operation,
                severity="ERROR"
            )
            
            # Try fallback operation
            try:
                ErrorLogger.log_warning(
                    message=f"Attempting fallback for user {user_id}",
                    context=context,
                    user_id=user_id,
                    operation=operation
                )
                
                post = await PostServiceWithFallback.create_post_fallback(user_id, content)
                
                ErrorLogger.log_info(
                    message=f"Post created successfully (fallback) | Post ID: {post['id']}",
                    context={**context, "post_id": post["id"], "fallback_used": True},
                    user_id=user_id,
                    operation=operation
                )
                
                return ErrorResponse.success(
                    data={"post_id": post["id"], "fallback_used": True},
                    message="Post created (using backup system)"
                )
                
            except Exception as fallback_error:
                ErrorLogger.log_error(
                    error=fallback_error,
                    context={**context, "fallback_failed": True},
                    user_id=user_id,
                    operation=f"{operation}_fallback",
                    severity="CRITICAL"
                )
                
                return ErrorResponse.create(
                    error_code="SERVICE_UNAVAILABLE",
                    user_message=UserFriendlyMessages.get_message("SERVICE_UNAVAILABLE"),
                    technical_message="Both primary and fallback systems failed"
                )
        
        except Exception as e:
            error_id = ErrorLogger.log_error(
                error=e,
                context=context,
                user_id=user_id,
                operation=operation,
                severity="ERROR"
            )
            
            return ErrorResponse.create(
                error_code="UNKNOWN_ERROR",
                user_message=UserFriendlyMessages.get_message("UNKNOWN_ERROR"),
                technical_message=str(e),
                error_id=error_id
            )

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

async def demonstrate_error_logging():
    """Demonstrate the error logging pattern with various scenarios"""
    
    post_service = PostService()
    fallback_service = PostServiceWithFallback()
    
    print("=" * 80)
    print("ERROR LOGGING PATTERN DEMONSTRATION")
    print("=" * 80)
    
    # Test 1: Successful post creation
    print("\n1. SUCCESSFUL POST CREATION:")
    result = await post_service.create_post_with_error_logging(
        user_id="valid_user",
        content="This is a test post with sufficient content length to pass validation.",
        hashtags=["test", "demo"]
    )
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # Test 2: Failed post creation (validation error)
    print("\n2. FAILED POST CREATION (Content too short):")
    result = await post_service.create_post_with_error_logging(
        user_id="valid_user",
        content="Short",
        hashtags=["test"]
    )
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # Test 3: Failed post creation (user not found)
    print("\n3. FAILED POST CREATION (User not found):")
    result = await post_service.create_post_with_error_logging(
        user_id="nonexistent_user",
        content="This is a test post with sufficient content length.",
        hashtags=["test"]
    )
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # Test 4: Failed post creation (inactive user)
    print("\n4. FAILED POST CREATION (Inactive user):")
    result = await post_service.create_post_with_error_logging(
        user_id="inactive_user",
        content="This is a test post with sufficient content length.",
        hashtags=["test"]
    )
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # Test 5: Failed post creation (database error)
    print("\n5. FAILED POST CREATION (Database error):")
    result = await post_service.create_post_with_error_logging(
        user_id="db_error_user",
        content="This is a test post with sufficient content length.",
        hashtags=["test"]
    )
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # Test 6: Failed post creation (simulated database error)
    print("\n6. FAILED POST CREATION (Simulated database error):")
    result = await post_service.create_post_with_error_logging(
        user_id="valid_user",
        content="This post contains error to simulate database failure.",
        hashtags=["test"]
    )
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # Test 7: Successful post update
    print("\n7. SUCCESSFUL POST UPDATE:")
    result = await post_service.update_post_with_error_logging(
        post_id="valid_post",
        user_id="valid_user",
        new_content="This is the updated content with sufficient length to pass all validations."
    )
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # Test 8: Failed post update (unauthorized)
    print("\n8. FAILED POST UPDATE (Unauthorized):")
    result = await post_service.update_post_with_error_logging(
        post_id="valid_post",
        user_id="private_user",
        new_content="This should fail due to unauthorized access."
    )
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # Test 9: Failed post update (post being edited)
    print("\n9. FAILED POST UPDATE (Post being edited):")
    result = await post_service.update_post_with_error_logging(
        post_id="editing_post",
        user_id="valid_user",
        new_content="This should fail because the post is being edited."
    )
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # Test 10: Successful post retrieval
    print("\n10. SUCCESSFUL POST RETRIEVAL:")
    result = await post_service.get_user_posts_with_error_logging(
        user_id="valid_user",
        requester_id="valid_user"
    )
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # Test 11: Failed post retrieval (database error)
    print("\n11. FAILED POST RETRIEVAL (Database error):")
    result = await post_service.get_user_posts_with_error_logging(
        user_id="db_error_user",
        requester_id="valid_user"
    )
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # Test 12: Graceful degradation (fallback)
    print("\n12. GRACEFUL DEGRADATION (Fallback):")
    result = await fallback_service.create_post_with_fallback(
        user_id="valid_user",
        content="This post contains error to simulate database failure and trigger fallback."
    )
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # Test 13: Error monitoring summary
    print("\n13. ERROR MONITORING SUMMARY:")
    print(f"Total errors recorded: {len(ErrorMonitor.error_counts)}")
    for error_key, errors in ErrorMonitor.error_counts.items():
        print(f"  {error_key}: {len(errors)} errors")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(demonstrate_error_logging()) 