"""
Node Event Handlers
==================

Event handlers for node domain events.
"""

from __future__ import annotations
import logging
from typing import Dict, Any

from ...shared.events.event_bus import EventHandler, DomainEvent
from ...shared.container import DependencyInjectionContainer, get_container
from ...infrastructure.external.notification_service import NotificationService
from ...infrastructure.external.analytics_service import AnalyticsService
from ...infrastructure.external.audit_service import AuditService
from ...infrastructure.external.ai_service import AIService


logger = logging.getLogger(__name__)


class NodeCreatedHandler(EventHandler):
    """Handler for node created events"""
    
    def __init__(self):
        self._container = get_container()
    
    async def handle(self, event: DomainEvent) -> None:
        """Handle node created event"""
        try:
            logger.info(f"Handling node created event: {event.metadata.event_id}")
            
            # Get services
            analytics_service = await self._container.resolve(AnalyticsService)
            audit_service = await self._container.resolve(AuditService)
            ai_service = await self._container.resolve(AIService)
            
            # Extract event data
            node_id = event.data.get("node_id")
            title = event.data.get("title")
            created_at = event.data.get("created_at")
            
            # Track analytics
            await analytics_service.track_node_created(
                node_id=node_id,
                title=title,
                created_at=created_at
            )
            
            # Audit log
            await audit_service.log_node_created(
                node_id=node_id,
                title=title,
                created_at=created_at,
                event_id=event.metadata.event_id
            )
            
            # Trigger AI analysis (async)
            await ai_service.analyze_node_content(node_id)
            
            logger.info(f"Successfully handled node created event: {event.metadata.event_id}")
            
        except Exception as e:
            logger.error(f"Error handling node created event {event.metadata.event_id}: {e}")
            raise
    
    def can_handle(self, event_type: str) -> bool:
        """Check if handler can handle event type"""
        return event_type == "node.created"


class NodeUpdatedHandler(EventHandler):
    """Handler for node updated events"""
    
    def __init__(self):
        self._container = get_container()
    
    async def handle(self, event: DomainEvent) -> None:
        """Handle node updated event"""
        try:
            logger.info(f"Handling node updated event: {event.metadata.event_id}")
            
            # Get services
            analytics_service = await self._container.resolve(AnalyticsService)
            audit_service = await self._container.resolve(AuditService)
            ai_service = await self._container.resolve(AIService)
            
            # Extract event data
            node_id = event.data.get("node_id")
            field = event.data.get("field")
            old_value = event.data.get("old_value")
            new_value = event.data.get("new_value")
            updated_at = event.data.get("updated_at")
            
            # Track analytics
            await analytics_service.track_node_updated(
                node_id=node_id,
                field=field,
                old_value=old_value,
                new_value=new_value,
                updated_at=updated_at
            )
            
            # Audit log
            await audit_service.log_node_updated(
                node_id=node_id,
                field=field,
                old_value=old_value,
                new_value=new_value,
                updated_at=updated_at,
                event_id=event.metadata.event_id
            )
            
            # Trigger AI analysis if content was updated
            if field in ["content", "title"]:
                await ai_service.analyze_node_content(node_id)
            
            logger.info(f"Successfully handled node updated event: {event.metadata.event_id}")
            
        except Exception as e:
            logger.error(f"Error handling node updated event {event.metadata.event_id}: {e}")
            raise
    
    def can_handle(self, event_type: str) -> bool:
        """Check if handler can handle event type"""
        return event_type == "node.updated"


class NodeDeletedHandler(EventHandler):
    """Handler for node deleted events"""
    
    def __init__(self):
        self._container = get_container()
    
    async def handle(self, event: DomainEvent) -> None:
        """Handle node deleted event"""
        try:
            logger.info(f"Handling node deleted event: {event.metadata.event_id}")
            
            # Get services
            analytics_service = await self._container.resolve(AnalyticsService)
            audit_service = await self._container.resolve(AuditService)
            
            # Extract event data
            node_id = event.data.get("node_id")
            deleted_at = event.data.get("deleted_at")
            
            # Track analytics
            await analytics_service.track_node_deleted(
                node_id=node_id,
                deleted_at=deleted_at
            )
            
            # Audit log
            await audit_service.log_node_deleted(
                node_id=node_id,
                deleted_at=deleted_at,
                event_id=event.metadata.event_id
            )
            
            logger.info(f"Successfully handled node deleted event: {event.metadata.event_id}")
            
        except Exception as e:
            logger.error(f"Error handling node deleted event {event.metadata.event_id}: {e}")
            raise
    
    def can_handle(self, event_type: str) -> bool:
        """Check if handler can handle event type"""
        return event_type == "node.deleted"


class NodeContentUpdatedHandler(EventHandler):
    """Handler for node content updated events"""
    
    def __init__(self):
        self._container = get_container()
    
    async def handle(self, event: DomainEvent) -> None:
        """Handle node content updated event"""
        try:
            logger.info(f"Handling node content updated event: {event.metadata.event_id}")
            
            # Get services
            ai_service = await self._container.resolve(AIService)
            analytics_service = await self._container.resolve(AnalyticsService)
            
            # Extract event data
            node_id = event.data.get("node_id")
            old_content = event.data.get("old_content")
            new_content = event.data.get("new_content")
            updated_at = event.data.get("updated_at")
            
            # Track analytics
            await analytics_service.track_content_updated(
                node_id=node_id,
                old_content_length=len(old_content) if old_content else 0,
                new_content_length=len(new_content) if new_content else 0,
                updated_at=updated_at
            )
            
            # Trigger AI analysis
            await ai_service.analyze_node_content(node_id)
            
            # Trigger content quality analysis
            await ai_service.analyze_content_quality(node_id)
            
            logger.info(f"Successfully handled node content updated event: {event.metadata.event_id}")
            
        except Exception as e:
            logger.error(f"Error handling node content updated event {event.metadata.event_id}: {e}")
            raise
    
    def can_handle(self, event_type: str) -> bool:
        """Check if handler can handle event type"""
        return event_type == "node.content_updated"


class NodePriorityChangedHandler(EventHandler):
    """Handler for node priority changed events"""
    
    def __init__(self):
        self._container = get_container()
    
    async def handle(self, event: DomainEvent) -> None:
        """Handle node priority changed event"""
        try:
            logger.info(f"Handling node priority changed event: {event.metadata.event_id}")
            
            # Get services
            analytics_service = await self._container.resolve(AnalyticsService)
            audit_service = await self._container.resolve(AuditService)
            
            # Extract event data
            node_id = event.data.get("node_id")
            old_priority = event.data.get("old_priority")
            new_priority = event.data.get("new_priority")
            changed_at = event.data.get("changed_at")
            
            # Track analytics
            await analytics_service.track_priority_changed(
                node_id=node_id,
                old_priority=old_priority,
                new_priority=new_priority,
                changed_at=changed_at
            )
            
            # Audit log
            await audit_service.log_priority_changed(
                node_id=node_id,
                old_priority=old_priority,
                new_priority=new_priority,
                changed_at=changed_at,
                event_id=event.metadata.event_id
            )
            
            logger.info(f"Successfully handled node priority changed event: {event.metadata.event_id}")
            
        except Exception as e:
            logger.error(f"Error handling node priority changed event {event.metadata.event_id}: {e}")
            raise
    
    def can_handle(self, event_type: str) -> bool:
        """Check if handler can handle event type"""
        return event_type == "node.priority_changed"


class NodeTagAddedHandler(EventHandler):
    """Handler for node tag added events"""
    
    def __init__(self):
        self._container = get_container()
    
    async def handle(self, event: DomainEvent) -> None:
        """Handle node tag added event"""
        try:
            logger.info(f"Handling node tag added event: {event.metadata.event_id}")
            
            # Get services
            analytics_service = await self._container.resolve(AnalyticsService)
            audit_service = await self._container.resolve(AuditService)
            
            # Extract event data
            node_id = event.data.get("node_id")
            tag = event.data.get("tag")
            added_at = event.data.get("added_at")
            
            # Track analytics
            await analytics_service.track_tag_added(
                node_id=node_id,
                tag=tag,
                added_at=added_at
            )
            
            # Audit log
            await audit_service.log_tag_added(
                node_id=node_id,
                tag=tag,
                added_at=added_at,
                event_id=event.metadata.event_id
            )
            
            logger.info(f"Successfully handled node tag added event: {event.metadata.event_id}")
            
        except Exception as e:
            logger.error(f"Error handling node tag added event {event.metadata.event_id}: {e}")
            raise
    
    def can_handle(self, event_type: str) -> bool:
        """Check if handler can handle event type"""
        return event_type == "node.tag_added"


class NodeTagRemovedHandler(EventHandler):
    """Handler for node tag removed events"""
    
    def __init__(self):
        self._container = get_container()
    
    async def handle(self, event: DomainEvent) -> None:
        """Handle node tag removed event"""
        try:
            logger.info(f"Handling node tag removed event: {event.metadata.event_id}")
            
            # Get services
            analytics_service = await self._container.resolve(AnalyticsService)
            audit_service = await self._container.resolve(AuditService)
            
            # Extract event data
            node_id = event.data.get("node_id")
            tag = event.data.get("tag")
            removed_at = event.data.get("removed_at")
            
            # Track analytics
            await analytics_service.track_tag_removed(
                node_id=node_id,
                tag=tag,
                removed_at=removed_at
            )
            
            # Audit log
            await audit_service.log_tag_removed(
                node_id=node_id,
                tag=tag,
                removed_at=removed_at,
                event_id=event.metadata.event_id
            )
            
            logger.info(f"Successfully handled node tag removed event: {event.metadata.event_id}")
            
        except Exception as e:
            logger.error(f"Error handling node tag removed event {event.metadata.event_id}: {e}")
            raise
    
    def can_handle(self, event_type: str) -> bool:
        """Check if handler can handle event type"""
        return event_type == "node.tag_removed"




