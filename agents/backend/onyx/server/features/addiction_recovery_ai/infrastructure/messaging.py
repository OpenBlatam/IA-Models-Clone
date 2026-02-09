"""
Messaging Service Implementations
Provides implementations for message queues and notifications
"""

import logging
from typing import Dict, Any, List

from core.interfaces import (
    IMessageQueueService, INotificationService, IServiceFactory
)
from config.aws_settings import get_aws_settings

logger = logging.getLogger(__name__)


class SQSMessageQueueService(IMessageQueueService):
    """SQS implementation of message queue service"""
    
    def __init__(self):
        from aws.aws_services import SQSService
        self.service = SQSService()
        self.settings = get_aws_settings()
    
    async def send(self, message: Dict[str, Any], **kwargs) -> str:
        """Send message to SQS"""
        queue_url = kwargs.get("queue_url") or self.settings.sqs_queue_url
        delay_seconds = kwargs.get("delay_seconds", 0)
        message_attributes = kwargs.get("message_attributes")
        
        return self.service.send_message(
            message_body=message,
            queue_url=queue_url,
            delay_seconds=delay_seconds,
            message_attributes=message_attributes
        )
    
    async def receive(self, max_messages: int = 10) -> List[Dict[str, Any]]:
        """Receive messages from SQS"""
        queue_url = get_aws_settings().sqs_queue_url
        messages = self.service.receive_messages(queue_url, max_messages)
        
        # Parse message bodies
        result = []
        for msg in messages:
            import json
            body = json.loads(msg.get("Body", "{}"))
            result.append({
                "id": msg.get("MessageId"),
                "receipt_handle": msg.get("ReceiptHandle"),
                "body": body
            })
        
        return result
    
    async def delete(self, receipt_handle: str) -> None:
        """Delete message from SQS"""
        queue_url = get_aws_settings().sqs_queue_url
        self.service.delete_message(queue_url, receipt_handle)


class SNSNotificationService(INotificationService):
    """SNS implementation of notification service"""
    
    def __init__(self):
        from aws.aws_services import SNSService
        self.service = SNSService()
        self.settings = get_aws_settings()
    
    async def send(self, recipient: str, message: str, **kwargs) -> str:
        """Send notification via SNS"""
        subject = kwargs.get("subject", "Notification")
        topic_arn = kwargs.get("topic_arn") or self.settings.sns_topic_arn
        
        notification_data = {
            "recipient": recipient,
            "message": message,
            **kwargs.get("metadata", {})
        }
        
        import json
        return self.service.publish(
            message=json.dumps(notification_data),
            subject=subject,
            topic_arn=topic_arn
        )


class MessagingServiceFactory(IServiceFactory):
    """Factory for creating messaging services"""
    
    @staticmethod
    def create_queue_service(backend: str = "sqs") -> IMessageQueueService:
        """Create message queue service"""
        if backend == "sqs":
            return SQSMessageQueueService()
        else:
            raise ValueError(f"Unsupported queue backend: {backend}")
    
    @staticmethod
    def create_notification_service(backend: str = "sns") -> INotificationService:
        """Create notification service"""
        if backend == "sns":
            return SNSNotificationService()
        else:
            raise ValueError(f"Unsupported notification backend: {backend}")
    
    def create_message_queue_service(self) -> IMessageQueueService:
        """Create message queue service (factory method)"""
        return self.create_queue_service()
    
    def create_notification_service(self) -> INotificationService:
        """Create notification service (factory method)"""
        return self.create_notification_service()
    
    def create_storage_service(self):
        """Not implemented in messaging factory"""
        raise NotImplementedError
    
    def create_cache_service(self):
        """Not implemented in messaging factory"""
        raise NotImplementedError
    
    def create_file_storage_service(self):
        """Not implemented in messaging factory"""
        raise NotImplementedError
    
    def create_metrics_service(self):
        """Not implemented in messaging factory"""
        raise NotImplementedError
    
    def create_tracing_service(self):
        """Not implemented in messaging factory"""
        raise NotImplementedError
    
    def create_authentication_service(self):
        """Not implemented in messaging factory"""
        raise NotImplementedError















