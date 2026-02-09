"""
Background Workers for Async Task Processing
Implements Celery-like patterns using SQS and Lambda
"""

import json
import logging
from typing import Dict, Any, Callable, Optional
from datetime import datetime, timedelta
from enum import Enum
import asyncio

from aws.aws_services import SQSService, SNSService
from aws.retry_handler import retry_with_backoff
from config.aws_settings import get_aws_settings

logger = logging.getLogger(__name__)
aws_settings = get_aws_settings()


class TaskStatus(Enum):
    """Task status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILURE = "failure"
    RETRY = "retry"


class BackgroundWorker:
    """
    Background worker for async task processing
    
    Features:
    - Task queue management
    - Retry logic
    - Task status tracking
    - Dead letter queue support
    """
    
    def __init__(self, queue_url: Optional[str] = None):
        self.sqs = SQSService()
        self.sns = SNSService()
        self.queue_url = queue_url or aws_settings.sqs_queue_url
        self.task_registry: Dict[str, Callable] = {}
        self.max_retries = 3
        self.retry_delay = 60  # seconds
    
    def register_task(self, task_name: str, task_func: Callable) -> None:
        """Register a task handler"""
        self.task_registry[task_name] = task_func
        logger.info(f"Registered task: {task_name}")
    
    @retry_with_backoff(max_retries=3, backoff_factor=2.0)
    def enqueue_task(self, task_name: str, task_data: Dict[str, Any],
                    delay_seconds: int = 0, priority: int = 0) -> str:
        """
        Enqueue a task for processing
        
        Args:
            task_name: Name of the task
            task_data: Task data
            delay_seconds: Delay before processing
            priority: Task priority (higher = more important)
            
        Returns:
            Task ID
        """
        task = {
            "task_name": task_name,
            "task_id": f"{task_name}_{datetime.utcnow().timestamp()}",
            "task_data": task_data,
            "status": TaskStatus.PENDING.value,
            "created_at": datetime.utcnow().isoformat(),
            "priority": priority,
            "retry_count": 0,
            "max_retries": self.max_retries
        }
        
        # Send to SQS
        message_attributes = {
            "task_name": {"StringValue": task_name, "DataType": "String"},
            "priority": {"StringValue": str(priority), "DataType": "Number"}
        }
        
        message_id = self.sqs.send_message(
            message_body=task,
            queue_url=self.queue_url,
            delay_seconds=delay_seconds,
            message_attributes=message_attributes
        )
        
        logger.info(f"Task enqueued: {task_name} (ID: {task['task_id']})")
        return task["task_id"]
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single task
        
        Args:
            task: Task data
            
        Returns:
            Processing result
        """
        task_name = task.get("task_name")
        task_data = task.get("task_data", {})
        task_id = task.get("task_id")
        
        if not task_name:
            raise ValueError("Task name not specified")
        
        handler = self.task_registry.get(task_name)
        
        if not handler:
            raise ValueError(f"No handler registered for task: {task_name}")
        
        try:
            # Update status
            task["status"] = TaskStatus.PROCESSING.value
            task["started_at"] = datetime.utcnow().isoformat()
            
            # Execute task
            if asyncio.iscoroutinefunction(handler):
                result = await handler(task_data)
            else:
                result = handler(task_data)
            
            # Update status
            task["status"] = TaskStatus.SUCCESS.value
            task["completed_at"] = datetime.utcnow().isoformat()
            task["result"] = result
            
            logger.info(f"Task completed: {task_name} (ID: {task_id})")
            return task
            
        except Exception as e:
            # Handle failure
            task["status"] = TaskStatus.FAILURE.value
            task["error"] = str(e)
            task["failed_at"] = datetime.utcnow().isoformat()
            
            # Retry if possible
            retry_count = task.get("retry_count", 0)
            max_retries = task.get("max_retries", self.max_retries)
            
            if retry_count < max_retries:
                task["status"] = TaskStatus.RETRY.value
                task["retry_count"] = retry_count + 1
                task["retry_at"] = (datetime.utcnow() + timedelta(seconds=self.retry_delay)).isoformat()
                
                # Re-enqueue with delay
                self.enqueue_task(
                    task_name=task_name,
                    task_data=task_data,
                    delay_seconds=self.retry_delay
                )
                
                logger.warning(f"Task will retry: {task_name} (ID: {task_id}, attempt {retry_count + 1})")
            else:
                # Send to dead letter queue
                logger.error(f"Task failed after max retries: {task_name} (ID: {task_id})")
                self._send_to_dlq(task)
            
            raise
    
    def _send_to_dlq(self, task: Dict[str, Any]) -> None:
        """Send failed task to dead letter queue"""
        dlq_url = aws_settings.sqs_queue_url.replace("/queue/", "/dlq/")
        
        try:
            self.sqs.send_message(
                message_body=task,
                queue_url=dlq_url
            )
            logger.info(f"Task sent to DLQ: {task.get('task_id')}")
        except Exception as e:
            logger.error(f"Error sending to DLQ: {str(e)}")


# Common task handlers
class TaskHandlers:
    """Common task handlers for recovery AI"""
    
    @staticmethod
    async def generate_report(task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recovery report"""
        user_id = task_data.get("user_id")
        report_type = task_data.get("report_type", "monthly")
        
        # Import here to avoid circular dependencies
        from services.report_service import ReportService
        report_service = ReportService()
        
        report = await report_service.generate_report(user_id, report_type)
        return {"report_id": report.get("id"), "status": "completed"}
    
    @staticmethod
    async def send_notification(task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send notification to user"""
        user_id = task_data.get("user_id")
        notification_type = task_data.get("notification_type")
        message = task_data.get("message")
        
        from services.notification_service import NotificationService
        notification_service = NotificationService()
        
        await notification_service.send_notification(user_id, notification_type, message)
        return {"status": "sent"}
    
    @staticmethod
    async def update_analytics(task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update user analytics"""
        user_id = task_data.get("user_id")
        
        from services.analytics_service import AnalyticsService
        analytics_service = AnalyticsService()
        
        await analytics_service.update_user_analytics(user_id)
        return {"status": "updated"}
    
    @staticmethod
    async def process_ml_prediction(task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process ML prediction"""
        user_id = task_data.get("user_id")
        prediction_type = task_data.get("prediction_type")
        
        from core.ultra_fast_engine import create_ultra_fast_engine
        engine = create_ultra_fast_engine()
        
        # Perform prediction
        result = await engine.predict_relapse_risk(user_id)
        return {"prediction": result}


# Lambda handler for background worker
def lambda_worker_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler for background worker
    
    Processes tasks from SQS queue
    """
    worker = BackgroundWorker()
    
    # Register common tasks
    handlers = TaskHandlers()
    worker.register_task("generate_report", handlers.generate_report)
    worker.register_task("send_notification", handlers.send_notification)
    worker.register_task("update_analytics", handlers.update_analytics)
    worker.register_task("process_ml_prediction", handlers.process_ml_prediction)
    
    results = []
    
    # Process each SQS record
    for record in event.get("Records", []):
        try:
            task = json.loads(record.get("body", "{}"))
            result = asyncio.run(worker.process_task(task))
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing task: {str(e)}")
            results.append({"status": "error", "error": str(e)})
    
    return {
        "statusCode": 200,
        "body": json.dumps({
            "processed": len(results),
            "results": results
        })
    }

