"""
Event-Driven Architecture Processor
Handles events from SQS, EventBridge, and other message brokers
"""

import json
import logging
from typing import Dict, Any, List, Callable, Optional
from datetime import datetime
import asyncio

from aws.aws_services import SQSService, SNSService
from aws.circuit_breaker import CircuitBreaker
from aws.retry_handler import retry_with_backoff
from config.aws_settings import get_aws_settings

logger = logging.getLogger(__name__)
aws_settings = get_aws_settings()


class EventProcessor:
    """
    Event processor for event-driven architecture
    
    Supports:
    - SQS message processing
    - EventBridge events
    - Custom event handlers
    - Dead letter queues
    """
    
    def __init__(self):
        self.sqs = SQSService()
        self.sns = SNSService()
        self.event_handlers: Dict[str, Callable] = {}
        self.circuit_breaker = CircuitBreaker()
    
    def register_handler(self, event_type: str, handler: Callable) -> None:
        """Register event handler for specific event type"""
        self.event_handlers[event_type] = handler
        logger.info(f"Registered handler for event type: {event_type}")
    
    @retry_with_backoff(max_retries=3, backoff_factor=2.0)
    async def process_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single event
        
        Args:
            event: Event data
            
        Returns:
            Processing result
        """
        event_type = event.get("event_type") or event.get("type")
        
        if not event_type:
            raise ValueError("Event type not specified")
        
        handler = self.event_handlers.get(event_type)
        
        if not handler:
            logger.warning(f"No handler registered for event type: {event_type}")
            return {"status": "ignored", "reason": "no_handler"}
        
        try:
            with self.circuit_breaker:
                # Process event
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(event)
                else:
                    result = handler(event)
                
                logger.info(f"Event processed successfully: {event_type}")
                return {
                    "status": "success",
                    "event_type": event_type,
                    "result": result
                }
                
        except Exception as e:
            logger.error(f"Error processing event {event_type}: {str(e)}", exc_info=True)
            raise
    
    async def process_sqs_messages(self, queue_url: Optional[str] = None, max_messages: int = 10) -> List[Dict[str, Any]]:
        """
        Process messages from SQS queue
        
        Args:
            queue_url: SQS queue URL
            max_messages: Maximum messages to process
            
        Returns:
            List of processing results
        """
        results = []
        
        try:
            # Receive messages
            messages = self.sqs.receive_messages(queue_url, max_messages)
            
            for message in messages:
                try:
                    # Parse message body
                    body = json.loads(message.get("Body", "{}"))
                    
                    # Process event
                    result = await self.process_event(body)
                    result["message_id"] = message.get("MessageId")
                    results.append(result)
                    
                    # Delete message after successful processing
                    if result["status"] == "success":
                        self.sqs.delete_message(
                            queue_url or aws_settings.sqs_queue_url,
                            message.get("ReceiptHandle")
                        )
                    
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
                    results.append({
                        "status": "error",
                        "error": str(e),
                        "message_id": message.get("MessageId")
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error receiving messages from SQS: {str(e)}")
            raise
    
    def publish_event(self, event_type: str, event_data: Dict[str, Any], 
                     topic_arn: Optional[str] = None) -> str:
        """
        Publish event to SNS topic
        
        Args:
            event_type: Type of event
            event_data: Event data
            topic_arn: SNS topic ARN
            
        Returns:
            Message ID
        """
        event = {
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "data": event_data
        }
        
        return self.sns.publish(
            message=json.dumps(event),
            subject=f"Event: {event_type}",
            topic_arn=topic_arn
        )
    
    async def process_eventbridge_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process EventBridge event
        
        Args:
            event: EventBridge event structure
            
        Returns:
            Processing result
        """
        # Extract event details
        source = event.get("source")
        detail_type = event.get("detail-type")
        detail = event.get("detail", {})
        
        # Create normalized event
        normalized_event = {
            "event_type": f"{source}.{detail_type}",
            "source": source,
            "detail_type": detail_type,
            "data": detail
        }
        
        return await self.process_event(normalized_event)


# Lambda handler for SQS events
def lambda_sqs_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler for SQS events
    
    This function should be deployed as a separate Lambda function
    that processes SQS messages
    """
    processor = EventProcessor()
    
    # Register handlers
    from services.notification_service import NotificationService
    from services.analytics_service import AnalyticsService
    
    notification_service = NotificationService()
    analytics_service = AnalyticsService()
    
    processor.register_handler("user.milestone", notification_service.send_milestone_notification)
    processor.register_handler("user.progress", analytics_service.update_analytics)
    processor.register_handler("report.generate", lambda e: None)  # Placeholder
    
    results = []
    
    # Process each SQS record
    for record in event.get("Records", []):
        try:
            body = json.loads(record.get("body", "{}"))
            result = asyncio.run(processor.process_event(body))
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing record: {str(e)}")
            results.append({"status": "error", "error": str(e)})
    
    return {
        "statusCode": 200,
        "body": json.dumps({
            "processed": len(results),
            "results": results
        })
    }

