"""
AI Integration System - Celery Tasks
Background task processing for integration operations
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from celery import Celery, Task
from celery.exceptions import Retry
from sqlalchemy.orm import Session

from .integration_engine import (
    AIIntegrationEngine,
    IntegrationRequest,
    IntegrationResult,
    ContentType,
    IntegrationStatus,
    integration_engine
)
from .models import (
    IntegrationRequest as DBIntegrationRequest,
    IntegrationResult as DBIntegrationResult,
    IntegrationLog,
    IntegrationMetrics,
    WebhookEvent,
    Base
)
from .config import settings
from .database import get_db_session

# Configure logging
logger = logging.getLogger(__name__)

# Initialize Celery
celery_app = Celery(
    "ai_integration_system",
    broker=settings.redis.url,
    backend=settings.redis.url,
    include=["ai_integration_system.tasks"]
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,  # 5 minutes
    task_soft_time_limit=240,  # 4 minutes
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_disable_rate_limits=False,
    task_routes={
        "ai_integration_system.tasks.process_integration_request": {"queue": "integration"},
        "ai_integration_system.tasks.process_bulk_integration": {"queue": "bulk"},
        "ai_integration_system.tasks.process_webhook_event": {"queue": "webhooks"},
        "ai_integration_system.tasks.cleanup_old_data": {"queue": "maintenance"},
        "ai_integration_system.tasks.health_check_platforms": {"queue": "monitoring"},
    }
)

class DatabaseTask(Task):
    """Base task class with database session management"""
    
    def __init__(self):
        self._db_session = None
    
    @property
    def db_session(self) -> Session:
        """Get database session"""
        if self._db_session is None:
            self._db_session = get_db_session()
        return self._db_session
    
    def after_return(self, status, retval, task_id, args, kwargs, einfo):
        """Clean up database session after task completion"""
        if self._db_session:
            self._db_session.close()
            self._db_session = None

@celery_app.task(bind=True, base=DatabaseTask, max_retries=3)
def process_integration_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single integration request
    """
    try:
        logger.info(f"Processing integration request: {request_data.get('content_id')}")
        
        # Create IntegrationRequest object
        integration_request = IntegrationRequest(
            content_id=request_data["content_id"],
            content_type=ContentType(request_data["content_type"]),
            content_data=request_data["content_data"],
            target_platforms=request_data["target_platforms"],
            priority=request_data.get("priority", 1),
            max_retries=request_data.get("max_retries", 3),
            metadata=request_data.get("metadata", {})
        )
        
        # Log the start of processing
        self.log_integration_action(
            request_data["content_id"],
            "process_started",
            IntegrationStatus.IN_PROGRESS,
            "Integration processing started"
        )
        
        # Process the request
        results = asyncio.run(process_single_request_async(integration_request))
        
        # Store results in database
        self.store_integration_results(request_data["content_id"], results)
        
        # Log completion
        self.log_integration_action(
            request_data["content_id"],
            "process_completed",
            IntegrationStatus.COMPLETED,
            f"Integration completed with {len(results)} results"
        )
        
        return {
            "status": "success",
            "content_id": request_data["content_id"],
            "results_count": len(results),
            "results": [result.__dict__ for result in results]
        }
        
    except Exception as e:
        logger.error(f"Error processing integration request: {str(e)}")
        
        # Log error
        self.log_integration_action(
            request_data.get("content_id", "unknown"),
            "process_failed",
            IntegrationStatus.FAILED,
            f"Integration failed: {str(e)}"
        )
        
        # Retry if we haven't exceeded max retries
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying integration request (attempt {self.request.retries + 1})")
            raise self.retry(countdown=60 * (2 ** self.request.retries))  # Exponential backoff
        
        return {
            "status": "failed",
            "content_id": request_data.get("content_id", "unknown"),
            "error": str(e)
        }

async def process_single_request_async(integration_request: IntegrationRequest) -> List[IntegrationResult]:
    """Async processing of integration request"""
    results = []
    
    for platform in integration_request.target_platforms:
        if platform not in integration_engine.connectors:
            logger.error(f"No connector found for platform: {platform}")
            continue
        
        try:
            connector = integration_engine.connectors[platform]
            
            # Authenticate if needed
            if not await connector.authenticate():
                logger.error(f"Authentication failed for platform: {platform}")
                continue
            
            # Create content
            result = await connector.create_content(integration_request.content_data)
            results.append(result)
            
            logger.info(f"Integration completed for {platform}: {result.status}")
            
        except Exception as e:
            logger.error(f"Error processing {platform}: {str(e)}")
            result = IntegrationResult(
                request_id=integration_request.content_id,
                platform=platform,
                status=IntegrationStatus.FAILED,
                error_message=str(e)
            )
            results.append(result)
    
    return results

@celery_app.task(bind=True, base=DatabaseTask)
def process_bulk_integration(self, bulk_requests: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Process multiple integration requests in bulk
    """
    try:
        logger.info(f"Processing bulk integration with {len(bulk_requests)} requests")
        
        results = []
        successful_count = 0
        failed_count = 0
        
        for request_data in bulk_requests:
            try:
                # Process each request
                result = process_integration_request.delay(request_data)
                results.append({
                    "content_id": request_data["content_id"],
                    "task_id": result.id,
                    "status": "queued"
                })
                successful_count += 1
                
            except Exception as e:
                logger.error(f"Error queuing request {request_data.get('content_id')}: {str(e)}")
                results.append({
                    "content_id": request_data.get("content_id", "unknown"),
                    "status": "failed",
                    "error": str(e)
                })
                failed_count += 1
        
        return {
            "status": "completed",
            "total_requests": len(bulk_requests),
            "successful_queued": successful_count,
            "failed": failed_count,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error processing bulk integration: {str(e)}")
        return {
            "status": "failed",
            "error": str(e)
        }

@celery_app.task(bind=True, base=DatabaseTask)
def process_webhook_event(self, webhook_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process webhook events from integrated platforms
    """
    try:
        platform = webhook_data.get("platform")
        event_type = webhook_data.get("event_type")
        payload = webhook_data.get("payload", {})
        
        logger.info(f"Processing webhook event: {platform} - {event_type}")
        
        # Store webhook event
        webhook_event = WebhookEvent(
            platform=platform,
            event_type=event_type,
            payload=payload
        )
        
        self.db_session.add(webhook_event)
        self.db_session.commit()
        
        # Process based on event type
        if event_type == "content.created":
            await handle_content_created_webhook(platform, payload)
        elif event_type == "content.updated":
            await handle_content_updated_webhook(platform, payload)
        elif event_type == "content.deleted":
            await handle_content_deleted_webhook(platform, payload)
        elif event_type == "campaign.sent":
            await handle_campaign_sent_webhook(platform, payload)
        else:
            logger.info(f"Unhandled webhook event type: {event_type}")
        
        # Mark as processed
        webhook_event.processed = True
        webhook_event.processed_at = datetime.utcnow()
        self.db_session.commit()
        
        return {
            "status": "success",
            "platform": platform,
            "event_type": event_type,
            "processed": True
        }
        
    except Exception as e:
        logger.error(f"Error processing webhook event: {str(e)}")
        
        # Mark as failed
        if 'webhook_event' in locals():
            webhook_event.processed = False
            webhook_event.processing_error = str(e)
            self.db_session.commit()
        
        return {
            "status": "failed",
            "error": str(e)
        }

async def handle_content_created_webhook(platform: str, payload: Dict[str, Any]):
    """Handle content created webhook"""
    logger.info(f"Content created on {platform}: {payload.get('id')}")

async def handle_content_updated_webhook(platform: str, payload: Dict[str, Any]):
    """Handle content updated webhook"""
    logger.info(f"Content updated on {platform}: {payload.get('id')}")

async def handle_content_deleted_webhook(platform: str, payload: Dict[str, Any]):
    """Handle content deleted webhook"""
    logger.info(f"Content deleted on {platform}: {payload.get('id')}")

async def handle_campaign_sent_webhook(platform: str, payload: Dict[str, Any]):
    """Handle campaign sent webhook"""
    logger.info(f"Campaign sent on {platform}: {payload.get('id')}")

@celery_app.task(bind=True, base=DatabaseTask)
def health_check_platforms(self) -> Dict[str, Any]:
    """
    Perform health checks on all configured platforms
    """
    try:
        logger.info("Performing platform health checks")
        
        health_results = {}
        
        for platform_name, connector in integration_engine.connectors.items():
            try:
                # Test connection
                is_healthy = asyncio.run(connector.authenticate())
                
                health_results[platform_name] = {
                    "status": "healthy" if is_healthy else "unhealthy",
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                # Store metrics
                metric = IntegrationMetrics(
                    platform=platform_name,
                    metric_type="health_check",
                    metric_value="1" if is_healthy else "0",
                    metadata={"timestamp": datetime.utcnow().isoformat()}
                )
                self.db_session.add(metric)
                
            except Exception as e:
                logger.error(f"Health check failed for {platform_name}: {str(e)}")
                health_results[platform_name] = {
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
        
        self.db_session.commit()
        
        return {
            "status": "completed",
            "timestamp": datetime.utcnow().isoformat(),
            "results": health_results
        }
        
    except Exception as e:
        logger.error(f"Error performing health checks: {str(e)}")
        return {
            "status": "failed",
            "error": str(e)
        }

@celery_app.task(bind=True, base=DatabaseTask)
def cleanup_old_data(self, days_to_keep: int = 30) -> Dict[str, Any]:
    """
    Clean up old data from the database
    """
    try:
        logger.info(f"Cleaning up data older than {days_to_keep} days")
        
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        
        # Clean up old integration logs
        old_logs = self.db_session.query(IntegrationLog).filter(
            IntegrationLog.created_at < cutoff_date
        ).delete()
        
        # Clean up old webhook events
        old_webhooks = self.db_session.query(WebhookEvent).filter(
            WebhookEvent.received_at < cutoff_date,
            WebhookEvent.processed == True
        ).delete()
        
        # Clean up old metrics (keep only last 7 days for detailed metrics)
        old_metrics = self.db_session.query(IntegrationMetrics).filter(
            IntegrationMetrics.timestamp < cutoff_date
        ).delete()
        
        self.db_session.commit()
        
        return {
            "status": "completed",
            "logs_deleted": old_logs,
            "webhooks_deleted": old_webhooks,
            "metrics_deleted": old_metrics,
            "cutoff_date": cutoff_date.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        return {
            "status": "failed",
            "error": str(e)
        }

@celery_app.task(bind=True, base=DatabaseTask)
def retry_failed_integrations(self) -> Dict[str, Any]:
    """
    Retry failed integrations that haven't exceeded max retries
    """
    try:
        logger.info("Retrying failed integrations")
        
        # Find failed integrations that can be retried
        failed_requests = self.db_session.query(DBIntegrationRequest).filter(
            DBIntegrationRequest.status == IntegrationStatus.FAILED,
            DBIntegrationRequest.retry_count < DBIntegrationRequest.max_retries
        ).all()
        
        retry_count = 0
        
        for request in failed_requests:
            try:
                # Increment retry count
                request.retry_count += 1
                request.status = IntegrationStatus.PENDING
                
                # Queue for retry
                request_data = {
                    "content_id": request.content_id,
                    "content_type": request.content_type.value,
                    "content_data": request.content_data,
                    "target_platforms": request.target_platforms,
                    "priority": request.priority,
                    "max_retries": request.max_retries,
                    "metadata": request.metadata
                }
                
                process_integration_request.delay(request_data)
                retry_count += 1
                
            except Exception as e:
                logger.error(f"Error retrying request {request.content_id}: {str(e)}")
        
        self.db_session.commit()
        
        return {
            "status": "completed",
            "retry_count": retry_count,
            "total_failed": len(failed_requests)
        }
        
    except Exception as e:
        logger.error(f"Error retrying failed integrations: {str(e)}")
        return {
            "status": "failed",
            "error": str(e)
        }

# Helper methods for DatabaseTask
def log_integration_action(self, content_id: str, action: str, status: IntegrationStatus, message: str):
    """Log integration action to database"""
    try:
        log_entry = IntegrationLog(
            content_id=content_id,
            platform="system",
            action=action,
            status=status,
            message=message
        )
        self.db_session.add(log_entry)
        self.db_session.commit()
    except Exception as e:
        logger.error(f"Error logging integration action: {str(e)}")

def store_integration_results(self, content_id: str, results: List[IntegrationResult]):
    """Store integration results in database"""
    try:
        # Find the integration request
        db_request = self.db_session.query(DBIntegrationRequest).filter(
            DBIntegrationRequest.content_id == content_id
        ).first()
        
        if not db_request:
            logger.error(f"Integration request not found: {content_id}")
            return
        
        # Update request status
        if all(r.status == IntegrationStatus.COMPLETED for r in results):
            db_request.status = IntegrationStatus.COMPLETED
        elif any(r.status == IntegrationStatus.COMPLETED for r in results):
            db_request.status = IntegrationStatus.COMPLETED  # Partial success
        else:
            db_request.status = IntegrationStatus.FAILED
        
        db_request.completed_at = datetime.utcnow()
        
        # Store results
        for result in results:
            db_result = DBIntegrationResult(
                request_id=db_request.id,
                platform=result.platform,
                status=result.status,
                external_id=result.external_id,
                error_message=result.error_message,
                response_data=result.response_data
            )
            self.db_session.add(db_result)
        
        self.db_session.commit()
        
    except Exception as e:
        logger.error(f"Error storing integration results: {str(e)}")

# Periodic tasks
@celery_app.task
def scheduled_health_checks():
    """Scheduled health checks for all platforms"""
    return health_check_platforms.delay()

@celery_app.task
def scheduled_cleanup():
    """Scheduled cleanup of old data"""
    return cleanup_old_data.delay(days_to_keep=30)

@celery_app.task
def scheduled_retry_failed():
    """Scheduled retry of failed integrations"""
    return retry_failed_integrations.delay()

# Celery beat schedule
celery_app.conf.beat_schedule = {
    'health-check-platforms': {
        'task': 'ai_integration_system.tasks.scheduled_health_checks',
        'schedule': 300.0,  # Every 5 minutes
    },
    'cleanup-old-data': {
        'task': 'ai_integration_system.tasks.scheduled_cleanup',
        'schedule': 86400.0,  # Every 24 hours
    },
    'retry-failed-integrations': {
        'task': 'ai_integration_system.tasks.scheduled_retry_failed',
        'schedule': 3600.0,  # Every hour
    },
}



























