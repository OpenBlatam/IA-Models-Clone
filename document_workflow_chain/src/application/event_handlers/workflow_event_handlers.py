"""
Workflow Event Handlers
======================

Event handlers for workflow domain events.
"""

from __future__ import annotations
import logging
from typing import Dict, Any

from ...shared.events.event_bus import EventHandler, DomainEvent
from ...shared.container import DependencyInjectionContainer, get_container
from ...infrastructure.external.notification_service import NotificationService
from ...infrastructure.external.analytics_service import AnalyticsService
from ...infrastructure.external.audit_service import AuditService


logger = logging.getLogger(__name__)


class WorkflowCreatedHandler(EventHandler):
    """Handler for workflow created events"""
    
    def __init__(self):
        self._container = get_container()
    
    async def handle(self, event: DomainEvent) -> None:
        """Handle workflow created event"""
        try:
            logger.info(f"Handling workflow created event: {event.metadata.event_id}")
            
            # Get services
            notification_service = await self._container.resolve(NotificationService)
            analytics_service = await self._container.resolve(AnalyticsService)
            audit_service = await self._container.resolve(AuditService)
            
            # Extract event data
            workflow_id = event.data.get("workflow_id")
            name = event.data.get("name")
            created_at = event.data.get("created_at")
            
            # Send notification
            await notification_service.send_workflow_created_notification(
                workflow_id=workflow_id,
                name=name,
                created_at=created_at
            )
            
            # Track analytics
            await analytics_service.track_workflow_created(
                workflow_id=workflow_id,
                name=name,
                created_at=created_at
            )
            
            # Audit log
            await audit_service.log_workflow_created(
                workflow_id=workflow_id,
                name=name,
                created_at=created_at,
                event_id=event.metadata.event_id
            )
            
            logger.info(f"Successfully handled workflow created event: {event.metadata.event_id}")
            
        except Exception as e:
            logger.error(f"Error handling workflow created event {event.metadata.event_id}: {e}")
            raise
    
    def can_handle(self, event_type: str) -> bool:
        """Check if handler can handle event type"""
        return event_type == "workflow.created"


class WorkflowUpdatedHandler(EventHandler):
    """Handler for workflow updated events"""
    
    def __init__(self):
        self._container = get_container()
    
    async def handle(self, event: DomainEvent) -> None:
        """Handle workflow updated event"""
        try:
            logger.info(f"Handling workflow updated event: {event.metadata.event_id}")
            
            # Get services
            analytics_service = await self._container.resolve(AnalyticsService)
            audit_service = await self._container.resolve(AuditService)
            
            # Extract event data
            workflow_id = event.data.get("workflow_id")
            field = event.data.get("field")
            old_value = event.data.get("old_value")
            new_value = event.data.get("new_value")
            updated_at = event.data.get("updated_at")
            
            # Track analytics
            await analytics_service.track_workflow_updated(
                workflow_id=workflow_id,
                field=field,
                old_value=old_value,
                new_value=new_value,
                updated_at=updated_at
            )
            
            # Audit log
            await audit_service.log_workflow_updated(
                workflow_id=workflow_id,
                field=field,
                old_value=old_value,
                new_value=new_value,
                updated_at=updated_at,
                event_id=event.metadata.event_id
            )
            
            logger.info(f"Successfully handled workflow updated event: {event.metadata.event_id}")
            
        except Exception as e:
            logger.error(f"Error handling workflow updated event {event.metadata.event_id}: {e}")
            raise
    
    def can_handle(self, event_type: str) -> bool:
        """Check if handler can handle event type"""
        return event_type == "workflow.updated"


class WorkflowDeletedHandler(EventHandler):
    """Handler for workflow deleted events"""
    
    def __init__(self):
        self._container = get_container()
    
    async def handle(self, event: DomainEvent) -> None:
        """Handle workflow deleted event"""
        try:
            logger.info(f"Handling workflow deleted event: {event.metadata.event_id}")
            
            # Get services
            notification_service = await self._container.resolve(NotificationService)
            analytics_service = await self._container.resolve(AnalyticsService)
            audit_service = await self._container.resolve(AuditService)
            
            # Extract event data
            workflow_id = event.data.get("workflow_id")
            deleted_at = event.data.get("deleted_at")
            
            # Send notification
            await notification_service.send_workflow_deleted_notification(
                workflow_id=workflow_id,
                deleted_at=deleted_at
            )
            
            # Track analytics
            await analytics_service.track_workflow_deleted(
                workflow_id=workflow_id,
                deleted_at=deleted_at
            )
            
            # Audit log
            await audit_service.log_workflow_deleted(
                workflow_id=workflow_id,
                deleted_at=deleted_at,
                event_id=event.metadata.event_id
            )
            
            logger.info(f"Successfully handled workflow deleted event: {event.metadata.event_id}")
            
        except Exception as e:
            logger.error(f"Error handling workflow deleted event {event.metadata.event_id}: {e}")
            raise
    
    def can_handle(self, event_type: str) -> bool:
        """Check if handler can handle event type"""
        return event_type == "workflow.deleted"


class WorkflowStatusChangedHandler(EventHandler):
    """Handler for workflow status changed events"""
    
    def __init__(self):
        self._container = get_container()
    
    async def handle(self, event: DomainEvent) -> None:
        """Handle workflow status changed event"""
        try:
            logger.info(f"Handling workflow status changed event: {event.metadata.event_id}")
            
            # Get services
            notification_service = await self._container.resolve(NotificationService)
            analytics_service = await self._container.resolve(AnalyticsService)
            audit_service = await self._container.resolve(AuditService)
            
            # Extract event data
            workflow_id = event.data.get("workflow_id")
            old_status = event.data.get("old_status")
            new_status = event.data.get("new_status")
            changed_at = event.data.get("changed_at")
            
            # Send notification for important status changes
            if new_status in ["completed", "error", "cancelled"]:
                await notification_service.send_workflow_status_changed_notification(
                    workflow_id=workflow_id,
                    old_status=old_status,
                    new_status=new_status,
                    changed_at=changed_at
                )
            
            # Track analytics
            await analytics_service.track_workflow_status_changed(
                workflow_id=workflow_id,
                old_status=old_status,
                new_status=new_status,
                changed_at=changed_at
            )
            
            # Audit log
            await audit_service.log_workflow_status_changed(
                workflow_id=workflow_id,
                old_status=old_status,
                new_status=new_status,
                changed_at=changed_at,
                event_id=event.metadata.event_id
            )
            
            logger.info(f"Successfully handled workflow status changed event: {event.metadata.event_id}")
            
        except Exception as e:
            logger.error(f"Error handling workflow status changed event {event.metadata.event_id}: {e}")
            raise
    
    def can_handle(self, event_type: str) -> bool:
        """Check if handler can handle event type"""
        return event_type == "workflow.status_changed"


class WorkflowNodeAddedHandler(EventHandler):
    """Handler for workflow node added events"""
    
    def __init__(self):
        self._container = get_container()
    
    async def handle(self, event: DomainEvent) -> None:
        """Handle workflow node added event"""
        try:
            logger.info(f"Handling workflow node added event: {event.metadata.event_id}")
            
            # Get services
            analytics_service = await self._container.resolve(AnalyticsService)
            audit_service = await self._container.resolve(AuditService)
            
            # Extract event data
            workflow_id = event.data.get("workflow_id")
            node_id = event.data.get("node_id")
            added_at = event.data.get("added_at")
            
            # Track analytics
            await analytics_service.track_node_added(
                workflow_id=workflow_id,
                node_id=node_id,
                added_at=added_at
            )
            
            # Audit log
            await audit_service.log_node_added(
                workflow_id=workflow_id,
                node_id=node_id,
                added_at=added_at,
                event_id=event.metadata.event_id
            )
            
            logger.info(f"Successfully handled workflow node added event: {event.metadata.event_id}")
            
        except Exception as e:
            logger.error(f"Error handling workflow node added event {event.metadata.event_id}: {e}")
            raise
    
    def can_handle(self, event_type: str) -> bool:
        """Check if handler can handle event type"""
        return event_type == "workflow.node_added"


class WorkflowNodeRemovedHandler(EventHandler):
    """Handler for workflow node removed events"""
    
    def __init__(self):
        self._container = get_container()
    
    async def handle(self, event: DomainEvent) -> None:
        """Handle workflow node removed event"""
        try:
            logger.info(f"Handling workflow node removed event: {event.metadata.event_id}")
            
            # Get services
            analytics_service = await self._container.resolve(AnalyticsService)
            audit_service = await self._container.resolve(AuditService)
            
            # Extract event data
            workflow_id = event.data.get("workflow_id")
            node_id = event.data.get("node_id")
            removed_at = event.data.get("removed_at")
            
            # Track analytics
            await analytics_service.track_node_removed(
                workflow_id=workflow_id,
                node_id=node_id,
                removed_at=removed_at
            )
            
            # Audit log
            await audit_service.log_node_removed(
                workflow_id=workflow_id,
                node_id=node_id,
                removed_at=removed_at,
                event_id=event.metadata.event_id
            )
            
            logger.info(f"Successfully handled workflow node removed event: {event.metadata.event_id}")
            
        except Exception as e:
            logger.error(f"Error handling workflow node removed event {event.metadata.event_id}: {e}")
            raise
    
    def can_handle(self, event_type: str) -> bool:
        """Check if handler can handle event type"""
        return event_type == "workflow.node_removed"




