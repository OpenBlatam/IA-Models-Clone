"""
Task Handlers
Modular task handlers for background processing
"""

import logging
from typing import Dict, Any, Callable, Optional
from core.interfaces import IBackgroundTask
from core.service_container import get_container

logger = logging.getLogger(__name__)


class TaskHandlerRegistry:
    """Registry for task handlers"""
    
    def __init__(self):
        self._handlers: Dict[str, IBackgroundTask] = {}
    
    def register(self, task_name: str, handler: IBackgroundTask) -> None:
        """Register task handler"""
        self._handlers[task_name] = handler
        logger.info(f"Registered task handler: {task_name}")
    
    def get(self, task_name: str) -> Optional[IBackgroundTask]:
        """Get task handler by name"""
        return self._handlers.get(task_name)
    
    async def execute(self, task_name: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task using registered handler"""
        handler = self.get(task_name)
        
        if not handler:
            raise ValueError(f"No handler for task: {task_name}")
        
        try:
            if hasattr(handler, "execute"):
                result = await handler.execute(task_data)
            elif callable(handler):
                result = await handler(task_data) if hasattr(handler, "__call__") else handler(task_data)
            else:
                raise ValueError(f"Invalid handler type: {type(handler)}")
            
            return {"status": "success", "result": result}
        except Exception as e:
            logger.error(f"Error executing task {task_name}: {str(e)}")
            raise


# Task handler implementations
class GenerateReportTask:
    """Handler for generate report task"""
    
    async def execute(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute generate report task"""
        user_id = task_data.get("user_id")
        report_type = task_data.get("report_type", "monthly")
        
        # Get services
        container = get_container()
        storage_service = container.get_storage_service()
        
        # Generate report (simplified)
        report = {
            "id": f"report_{user_id}_{report_type}",
            "user_id": user_id,
            "type": report_type,
            "generated_at": task_data.get("timestamp")
        }
        
        # Store report
        await storage_service.put(report)
        
        return {"report_id": report["id"], "status": "completed"}


class SendNotificationTask:
    """Handler for send notification task"""
    
    async def execute(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute send notification task"""
        user_id = task_data.get("user_id")
        message = task_data.get("message")
        notification_type = task_data.get("notification_type", "general")
        
        # Get services
        container = get_container()
        notification_service = container.get_notification_service()
        
        # Send notification
        await notification_service.send(
            recipient=user_id,
            message=message,
            metadata={"type": notification_type}
        )
        
        return {"status": "sent", "user_id": user_id}


class UpdateAnalyticsTask:
    """Handler for update analytics task"""
    
    async def execute(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute update analytics task"""
        user_id = task_data.get("user_id")
        
        # Get services
        container = get_container()
        storage_service = container.get_storage_service()
        metrics_service = container.get_metrics_service()
        
        # Update analytics (simplified)
        analytics = await storage_service.get(f"analytics_{user_id}")
        
        if not analytics:
            analytics = {"user_id": user_id, "updates": 0}
        
        analytics["updates"] = analytics.get("updates", 0) + 1
        
        await storage_service.put(analytics)
        
        # Record metric
        await metrics_service.increment("analytics_updates", labels={"user_id": user_id})
        
        return {"status": "updated", "user_id": user_id}


def register_task_handlers(registry: TaskHandlerRegistry) -> None:
    """Register all task handlers"""
    registry.register("generate_report", GenerateReportTask())
    registry.register("send_notification", SendNotificationTask())
    registry.register("update_analytics", UpdateAnalyticsTask())
    
    logger.info("Task handlers registered")















