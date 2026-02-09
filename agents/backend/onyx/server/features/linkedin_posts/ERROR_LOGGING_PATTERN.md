# Error Logging Pattern: Proper Error Logging and User-Friendly Messages

## Core Principle: Comprehensive Error Handling with Clear Communication

Implement proper error logging for debugging while providing user-friendly error messages. This creates:
- **Detailed logging** for developers and system administrators
- **User-friendly messages** that don't expose sensitive information
- **Structured error tracking** for monitoring and alerting
- **Consistent error handling** across the application

## 1. Error Logging Structure

### ❌ **Poor Error Handling (Bad)**
```python
async def create_post_bad(user_id: str, content: str) -> Dict[str, Any]:
    try:
        if not user_id:
            return {"error": "User ID required"}
        
        user = await get_user_by_id(user_id)
        if not user:
            return {"error": "User not found"}
        
        post = await create_post_in_database(user_id, content)
        return {"status": "success", "post_id": post.id}
        
    except Exception as e:
        print(f"Error: {e}")  # Poor logging
        return {"error": "Something went wrong"}  # Generic message
```

### ✅ **Proper Error Logging (Good)**
```python
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, Optional

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ErrorLogger:
    """Centralized error logging utility"""
    
    @staticmethod
    def log_error(
        error: Exception,
        context: Dict[str, Any],
        user_id: Optional[str] = None,
        operation: str = "unknown",
        severity: str = "ERROR"
    ) -> None:
        """Log error with structured context"""
        error_id = f"err_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(error))}"
        
        log_data = {
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
        
        logger.error(f"Error ID: {error_id} | Operation: {operation} | User: {user_id} | Error: {error}")
        logger.debug(f"Full error context: {log_data}")
        
        # Store error for monitoring (optional)
        ErrorLogger.store_error_for_monitoring(log_data)
    
    @staticmethod
    def store_error_for_monitoring(error_data: Dict[str, Any]) -> None:
        """Store error data for monitoring and alerting"""
        # Implementation for storing errors in monitoring system
        pass

async def create_post_good(user_id: str, content: str) -> Dict[str, Any]:
    """Create post with proper error logging"""
    operation = "create_post"
    context = {"user_id": user_id, "content_length": len(content) if content else 0}
    
    try:
        # Input validation
        if not user_id:
            return {
                "status": "failed",
                "error": "User ID is required",
                "error_code": "MISSING_USER_ID",
                "user_message": "Please provide a valid user ID"
            }
        
        if not content:
            return {
                "status": "failed",
                "error": "Content is required",
                "error_code": "MISSING_CONTENT",
                "user_message": "Please provide post content"
            }
        
        # User validation
        user = await get_user_by_id(user_id)
        if not user:
            ErrorLogger.log_error(
                error=ValueError(f"User not found: {user_id}"),
                context=context,
                user_id=user_id,
                operation=operation,
                severity="WARNING"
            )
            return {
                "status": "failed",
                "error": "User not found",
                "error_code": "USER_NOT_FOUND",
                "user_message": "User account not found"
            }
        
        # Create post
        post = await create_post_in_database(user_id, content)
        
        logger.info(f"Post created successfully | User: {user_id} | Post ID: {post.id}")
        return {
            "status": "success",
            "post_id": post.id,
            "message": "Post created successfully"
        }
        
    except Exception as e:
        ErrorLogger.log_error(
            error=e,
            context=context,
            user_id=user_id,
            operation=operation,
            severity="ERROR"
        )
        
        return {
            "status": "failed",
            "error": "Failed to create post",
            "error_code": "POST_CREATION_FAILED",
            "user_message": "Unable to create post. Please try again later.",
            "error_id": f"err_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(e))}"
        }
```

## 2. User-Friendly Error Messages

### ❌ **Poor User Messages (Bad)**
```python
async def update_post_bad(post_id: str, user_id: str, content: str) -> Dict[str, Any]:
    try:
        post = await get_post_by_id(post_id)
        if not post:
            return {"error": "Post not found"}  # Too generic
        
        if post.user_id != user_id:
            return {"error": "Unauthorized"}  # Too technical
        
        if len(content) < 10:
            return {"error": "Content too short"}  # Not helpful
        
        # Update logic...
        
    except DatabaseConnectionError as e:
        return {"error": f"Database error: {e}"}  # Exposes technical details
        
    except Exception as e:
        return {"error": "Internal server error"}  # Too generic
```

### ✅ **User-Friendly Messages (Good)**
```python
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
        
        # Business rule errors
        "DAILY_LIMIT_EXCEEDED": "You've reached your daily post limit. Please try again tomorrow.",
        "RATE_LIMIT_EXCEEDED": "You're posting too quickly. Please wait a moment before trying again.",
        "DUPLICATE_CONTENT": "This content appears to be a duplicate. Please create unique content.",
        "EDIT_TIME_EXPIRED": "This post can no longer be edited (24-hour limit).",
        
        # System errors
        "DATABASE_ERROR": "We're experiencing technical difficulties. Please try again in a few minutes.",
        "NETWORK_ERROR": "Connection issue detected. Please check your internet and try again.",
        "SERVICE_UNAVAILABLE": "This service is temporarily unavailable. Please try again later.",
        "UNKNOWN_ERROR": "Something unexpected happened. Please try again or contact support if the problem persists."
    }
    
    @staticmethod
    def get_message(error_code: str, default: str = None) -> str:
        """Get user-friendly message for error code"""
        return UserFriendlyMessages.MESSAGES.get(error_code, default or "An error occurred. Please try again.")

async def update_post_good(post_id: str, user_id: str, content: str) -> Dict[str, Any]:
    """Update post with user-friendly error messages"""
    operation = "update_post"
    context = {"post_id": post_id, "user_id": user_id, "content_length": len(content) if content else 0}
    
    try:
        # Input validation
        if not post_id:
            return {
                "status": "failed",
                "error_code": "MISSING_POST_ID",
                "user_message": "Please provide a valid post ID"
            }
        
        if not content:
            return {
                "status": "failed",
                "error_code": "MISSING_CONTENT",
                "user_message": "Please provide new content for the post"
            }
        
        # Content validation
        if len(content) < 10:
            return {
                "status": "failed",
                "error_code": "CONTENT_TOO_SHORT",
                "user_message": UserFriendlyMessages.get_message("CONTENT_TOO_SHORT")
            }
        
        # Post validation
        post = await get_post_by_id(post_id)
        if not post:
            ErrorLogger.log_error(
                error=ValueError(f"Post not found: {post_id}"),
                context=context,
                user_id=user_id,
                operation=operation,
                severity="WARNING"
            )
            return {
                "status": "failed",
                "error_code": "POST_NOT_FOUND",
                "user_message": "The post you're trying to edit doesn't exist"
            }
        
        # Authorization
        if post.user_id != user_id:
            ErrorLogger.log_error(
                error=PermissionError(f"Unauthorized post update attempt | User: {user_id} | Post: {post_id}"),
                context=context,
                user_id=user_id,
                operation=operation,
                severity="WARNING"
            )
            return {
                "status": "failed",
                "error_code": "UNAUTHORIZED_UPDATE",
                "user_message": UserFriendlyMessages.get_message("UNAUTHORIZED_ACCESS")
            }
        
        # Update post
        updated_post = await update_post_in_database(post_id, content)
        
        logger.info(f"Post updated successfully | User: {user_id} | Post: {post_id}")
        return {
            "status": "success",
            "message": "Post updated successfully"
        }
        
    except DatabaseConnectionError as e:
        ErrorLogger.log_error(
            error=e,
            context=context,
            user_id=user_id,
            operation=operation,
            severity="ERROR"
        )
        return {
            "status": "failed",
            "error_code": "DATABASE_ERROR",
            "user_message": UserFriendlyMessages.get_message("DATABASE_ERROR")
        }
        
    except Exception as e:
        ErrorLogger.log_error(
            error=e,
            context=context,
            user_id=user_id,
            operation=operation,
            severity="ERROR"
        )
        return {
            "status": "failed",
            "error_code": "UNKNOWN_ERROR",
            "user_message": UserFriendlyMessages.get_message("UNKNOWN_ERROR")
        }
```

## 3. Structured Error Response

### ❌ **Inconsistent Error Responses (Bad)**
```python
# Different error response formats
return {"error": "User not found"}
return {"status": "failed", "message": "Invalid input"}
return {"success": False, "error_msg": "Database error"}
return {"result": "error", "details": "Something went wrong"}
```

### ✅ **Structured Error Responses (Good)**
```python
import os
from datetime import datetime
from typing import Dict, Any, Optional

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

# Usage example
async def get_user_posts(user_id: str, requester_id: str) -> Dict[str, Any]:
    try:
        if not requester_id:
            return ErrorResponse.create(
                error_code="MISSING_REQUESTER_ID",
                user_message="Please provide your user ID"
            )
        
        user = await get_user_by_id(requester_id)
        if not user:
            return ErrorResponse.create(
                error_code="USER_NOT_FOUND",
                user_message=UserFriendlyMessages.get_message("USER_NOT_FOUND"),
                technical_message=f"User not found in database: {requester_id}",
                context={"requester_id": requester_id}
            )
        
        posts = await get_posts_from_database(user_id)
        return ErrorResponse.success(
            data={"posts": posts, "count": len(posts)},
            message="Posts retrieved successfully"
        )
        
    except Exception as e:
        error_id = f"err_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(e))}"
        
        ErrorLogger.log_error(
            error=e,
            context={"user_id": user_id, "requester_id": requester_id},
            user_id=requester_id,
            operation="get_user_posts",
            severity="ERROR"
        )
        
        return ErrorResponse.create(
            error_code="FETCH_FAILED",
            user_message=UserFriendlyMessages.get_message("DATABASE_ERROR"),
            technical_message=str(e),
            error_id=error_id
        )
```

## 4. Error Monitoring and Alerting

### **Error Monitoring Setup**
```python
import json
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

class ErrorMonitor:
    """Error monitoring and alerting system"""
    
    def __init__(self):
        self.error_counts = {}
        self.alert_thresholds = {
            "ERROR": 10,  # Alert after 10 errors in 5 minutes
            "WARNING": 50,  # Alert after 50 warnings in 5 minutes
            "CRITICAL": 5   # Alert after 5 critical errors in 1 minute
        }
        self.alert_cooldown = timedelta(minutes=5)
        self.last_alerts = {}
    
    def record_error(self, error_data: Dict[str, Any]) -> None:
        """Record error for monitoring"""
        error_key = f"{error_data['operation']}_{error_data['error_type']}"
        current_time = datetime.now()
        
        if error_key not in self.error_counts:
            self.error_counts[error_key] = []
        
        # Add error with timestamp
        self.error_counts[error_key].append({
            "timestamp": current_time,
            "severity": error_data["severity"],
            "user_id": error_data.get("user_id"),
            "error_id": error_data["error_id"]
        })
        
        # Clean old errors (older than 1 hour)
        cutoff_time = current_time - timedelta(hours=1)
        self.error_counts[error_key] = [
            error for error in self.error_counts[error_key]
            if error["timestamp"] > cutoff_time
        ]
        
        # Check for alerts
        self.check_alerts(error_key, error_data["severity"])
    
    def check_alerts(self, error_key: str, severity: str) -> None:
        """Check if alerts should be triggered"""
        current_time = datetime.now()
        errors = self.error_counts.get(error_key, [])
        
        # Count recent errors by severity
        recent_errors = [
            error for error in errors
            if error["timestamp"] > current_time - timedelta(minutes=5)
        ]
        
        severity_count = len([e for e in recent_errors if e["severity"] == severity])
        threshold = self.alert_thresholds.get(severity, 10)
        
        if severity_count >= threshold:
            # Check cooldown
            last_alert = self.last_alerts.get(error_key, datetime.min)
            if current_time - last_alert > self.alert_cooldown:
                self.trigger_alert(error_key, severity, severity_count, recent_errors)
                self.last_alerts[error_key] = current_time
    
    def trigger_alert(self, error_key: str, severity: str, count: int, errors: list) -> None:
        """Trigger alert for error threshold exceeded"""
        alert_data = {
            "alert_id": f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "error_key": error_key,
            "severity": severity,
            "error_count": count,
            "threshold": self.alert_thresholds[severity],
            "recent_errors": errors[:5]  # Last 5 errors
        }
        
        logger.critical(f"ALERT: {severity} threshold exceeded for {error_key} | Count: {count}")
        
        # Send alert to monitoring system
        self.send_alert_to_monitoring_system(alert_data)
    
    def send_alert_to_monitoring_system(self, alert_data: Dict[str, Any]) -> None:
        """Send alert to external monitoring system"""
        # Implementation for sending to monitoring system (e.g., Sentry, DataDog, etc.)
        logger.info(f"Alert sent to monitoring system: {alert_data['alert_id']}")

# Global error monitor instance
error_monitor = ErrorMonitor()

# Enhanced ErrorLogger with monitoring
class ErrorLogger:
    @staticmethod
    def log_error(
        error: Exception,
        context: Dict[str, Any],
        user_id: Optional[str] = None,
        operation: str = "unknown",
        severity: str = "ERROR"
    ) -> None:
        """Log error with structured context and monitoring"""
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
        logger.debug(f"Full error context: {error_data}")
        
        # Record for monitoring
        error_monitor.record_error(error_data)
```

## 5. Error Recovery and Graceful Degradation

### **Graceful Degradation Pattern**
```python
async def create_post_with_fallback(user_id: str, content: str) -> Dict[str, Any]:
    """Create post with graceful degradation"""
    operation = "create_post"
    context = {"user_id": user_id, "content_length": len(content) if content else 0}
    
    try:
        # Primary operation
        post = await create_post_in_database(user_id, content)
        return ErrorResponse.success({"post_id": post.id}, "Post created successfully")
        
    except DatabaseConnectionError as e:
        # Log the error
        ErrorLogger.log_error(
            error=e,
            context=context,
            user_id=user_id,
            operation=operation,
            severity="ERROR"
        )
        
        # Try fallback operation
        try:
            logger.warning(f"Attempting fallback for user {user_id}")
            post = await create_post_fallback(user_id, content)
            return ErrorResponse.success(
                {"post_id": post.id, "fallback_used": True},
                "Post created (using backup system)"
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
        ErrorLogger.log_error(
            error=e,
            context=context,
            user_id=user_id,
            operation=operation,
            severity="ERROR"
        )
        
        return ErrorResponse.create(
            error_code="UNKNOWN_ERROR",
            user_message=UserFriendlyMessages.get_message("UNKNOWN_ERROR")
        )
```

## 6. Function Structure Template

### **Standard Error Logging Structure**
```python
async def function_with_error_logging(param1: str, param2: str) -> Dict[str, Any]:
    operation = "function_name"
    context = {"param1": param1, "param2": param2}
    
    try:
        # ============================================================================
        # VALIDATION PHASE (Early returns with proper error responses)
        # ============================================================================
        
        if not param1:
            return ErrorResponse.create(
                error_code="MISSING_PARAM1",
                user_message="Parameter 1 is required"
            )
        
        # ============================================================================
        # MAIN BUSINESS LOGIC
        # ============================================================================
        
        result = await perform_main_operation(param1, param2)
        
        # Log success
        logger.info(f"Operation completed successfully | User: {context.get('user_id')} | Operation: {operation}")
        
        return ErrorResponse.success({"result": result}, "Operation completed successfully")
        
    except SpecificException as e:
        # Handle specific exceptions
        ErrorLogger.log_error(
            error=e,
            context=context,
            user_id=context.get("user_id"),
            operation=operation,
            severity="WARNING"
        )
        
        return ErrorResponse.create(
            error_code="SPECIFIC_ERROR",
            user_message=UserFriendlyMessages.get_message("SPECIFIC_ERROR"),
            technical_message=str(e)
        )
        
    except Exception as e:
        # Handle unexpected exceptions
        ErrorLogger.log_error(
            error=e,
            context=context,
            user_id=context.get("user_id"),
            operation=operation,
            severity="ERROR"
        )
        
        return ErrorResponse.create(
            error_code="UNKNOWN_ERROR",
            user_message=UserFriendlyMessages.get_message("UNKNOWN_ERROR")
        )
```

## 7. Benefits of Proper Error Logging

### **Developer Benefits**
- **Detailed debugging information** with full context
- **Structured error tracking** for monitoring
- **Error correlation** with unique error IDs
- **Performance insights** from error patterns

### **User Benefits**
- **Clear, actionable error messages** that don't expose technical details
- **Consistent error experience** across the application
- **Helpful guidance** on how to resolve issues
- **Professional error handling** that builds trust

### **Operational Benefits**
- **Proactive monitoring** with alerting thresholds
- **Error trend analysis** for system improvements
- **Quick incident response** with detailed error context
- **Reduced support burden** with better error messages

## 8. When to Use Proper Error Logging

### **✅ Use For:**
- All user-facing operations
- Database operations
- External API calls
- File operations
- Authentication and authorization
- Business logic operations
- System integration points

### **❌ Avoid For:**
- Internal utility functions with no user impact
- Performance-critical operations where logging overhead is significant
- Temporary debugging code
- Sensitive operations where logging could expose secrets

Proper error logging transforms your application into a robust, maintainable system that provides excellent user experience while enabling effective debugging and monitoring. 