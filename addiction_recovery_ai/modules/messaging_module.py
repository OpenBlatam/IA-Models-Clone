"""
Messaging Module
Independent messaging module for queues and notifications
"""

from typing import List
from modules.base_module import BaseModule
from infrastructure.messaging import MessagingServiceFactory
from core.service_container import get_container

logger = __import__("logging").getLogger(__name__)


class MessagingModule(BaseModule):
    """Messaging feature module"""
    
    def __init__(self):
        super().__init__("messaging", "1.0.0")
        self._factory = None
    
    def get_dependencies(self) -> List[str]:
        """Messaging module has no dependencies"""
        return []
    
    def _on_initialize(self) -> None:
        """Initialize messaging module"""
        self._factory = MessagingServiceFactory()
        
        # Register services in container
        container = get_container()
        container.register_service(
            "message_queue",
            self._factory.create_message_queue_service()
        )
        container.register_service(
            "notification",
            self._factory.create_notification_service()
        )
        
        logger.info("Messaging module initialized")
    
    def _on_shutdown(self) -> None:
        """Shutdown messaging module"""
        logger.info("Messaging module shut down")
    
    def get_message_queue_service(self):
        """Get message queue service instance"""
        if not self._factory:
            raise RuntimeError("Messaging module not initialized")
        return self._factory.create_message_queue_service()
    
    def get_notification_service(self):
        """Get notification service instance"""
        if not self._factory:
            raise RuntimeError("Messaging module not initialized")
        return self._factory.create_notification_service()















