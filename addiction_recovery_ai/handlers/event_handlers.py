"""
Event Handlers
Modular event handlers for event-driven architecture
"""

import logging
from typing import Dict, Any, Callable, Optional
from core.interfaces import IEventHandler, IEventPublisher
from core.service_container import get_container

logger = logging.getLogger(__name__)


class EventHandlerRegistry:
    """Registry for event handlers"""
    
    def __init__(self):
        self._handlers: Dict[str, IEventHandler] = {}
    
    def register(self, event_type: str, handler: IEventHandler) -> None:
        """Register event handler"""
        self._handlers[event_type] = handler
        logger.info(f"Registered event handler: {event_type}")
    
    def get(self, event_type: str) -> Optional[IEventHandler]:
        """Get event handler by type"""
        return self._handlers.get(event_type)
    
    async def handle(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle event using registered handler"""
        event_type = event.get("event_type") or event.get("type")
        
        if not event_type:
            raise ValueError("Event type not specified")
        
        handler = self.get(event_type)
        
        if not handler:
            logger.warning(f"No handler for event type: {event_type}")
            return {"status": "ignored", "reason": "no_handler"}
        
        try:
            if hasattr(handler, "handle"):
                result = await handler.handle(event)
            elif callable(handler):
                result = await handler(event) if hasattr(handler, "__call__") else handler(event)
            else:
                raise ValueError(f"Invalid handler type: {type(handler)}")
            
            return {"status": "success", "result": result}
        except Exception as e:
            logger.error(f"Error handling event {event_type}: {str(e)}")
            raise


# Event handler implementations
class UserMilestoneEventHandler:
    """Handler for user milestone events"""
    
    async def handle(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle user milestone event"""
        user_id = event.get("data", {}).get("user_id")
        milestone = event.get("data", {}).get("milestone")
        
        # Get services
        container = get_container()
        notification_service = container.get_notification_service()
        metrics_service = container.get_metrics_service()
        
        # Send notification
        await notification_service.send(
            recipient=user_id,
            message=f"Congratulations! You've achieved: {milestone}"
        )
        
        # Record metric
        await metrics_service.increment("milestones_achieved", labels={"milestone_type": milestone})
        
        return {"status": "processed", "user_id": user_id, "milestone": milestone}


class ProgressUpdateEventHandler:
    """Handler for progress update events"""
    
    async def handle(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle progress update event"""
        user_id = event.get("data", {}).get("user_id")
        progress = event.get("data", {}).get("progress")
        
        # Get services
        container = get_container()
        storage_service = container.get_storage_service()
        metrics_service = container.get_metrics_service()
        
        # Update storage
        await storage_service.put({
            "id": f"progress_{user_id}",
            "user_id": user_id,
            "progress": progress,
            "updated_at": event.get("timestamp")
        })
        
        # Record metric
        await metrics_service.record("recovery_progress", progress, labels={"user_id": user_id})
        
        return {"status": "processed", "user_id": user_id}


def register_event_handlers(registry: EventHandlerRegistry) -> None:
    """Register all event handlers"""
    registry.register("user.milestone", UserMilestoneEventHandler())
    registry.register("user.progress", ProgressUpdateEventHandler())
    
    logger.info("Event handlers registered")















