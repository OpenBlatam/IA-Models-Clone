"""
Event Handlers
==============

Event handlers for different types of events in the Business Agents System.
"""

import logging
from typing import Dict, Any
from datetime import datetime

from .types import Event, EventType, EventHandler

logger = logging.getLogger(__name__)

class AgentEventHandler(EventHandler):
    """Handler for agent-related events."""
    
    def __init__(self):
        super().__init__("agent_event_handler")
    
    async def handle(self, event: Event) -> bool:
        """Handle agent events."""
        try:
            if event.type == EventType.AGENT_CREATED:
                await self._handle_agent_created(event)
            elif event.type == EventType.AGENT_UPDATED:
                await self._handle_agent_updated(event)
            elif event.type == EventType.AGENT_DELETED:
                await self._handle_agent_deleted(event)
            elif event.type == EventType.AGENT_EXECUTION_STARTED:
                await self._handle_agent_execution_started(event)
            elif event.type == EventType.AGENT_EXECUTION_COMPLETED:
                await self._handle_agent_execution_completed(event)
            elif event.type == EventType.AGENT_EXECUTION_FAILED:
                await self._handle_agent_execution_failed(event)
            
            logger.debug(f"Handled agent event: {event.type.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to handle agent event {event.id}: {str(e)}")
            return False
    
    def can_handle(self, event: Event) -> bool:
        """Check if this handler can handle the event."""
        return event.type in [
            EventType.AGENT_CREATED,
            EventType.AGENT_UPDATED,
            EventType.AGENT_DELETED,
            EventType.AGENT_EXECUTION_STARTED,
            EventType.AGENT_EXECUTION_COMPLETED,
            EventType.AGENT_EXECUTION_FAILED
        ]
    
    async def _handle_agent_created(self, event: Event):
        """Handle agent created event."""
        agent_id = event.data.get("agent_id")
        agent_name = event.data.get("agent_name")
        
        # Update metrics, send notifications, etc.
        logger.info(f"Agent created: {agent_name} (ID: {agent_id})")
        
        # Broadcast to WebSocket subscribers
        await self._broadcast_websocket_event(event)
    
    async def _handle_agent_updated(self, event: Event):
        """Handle agent updated event."""
        agent_id = event.data.get("agent_id")
        changes = event.data.get("changes", {})
        
        logger.info(f"Agent updated: {agent_id}, changes: {changes}")
        await self._broadcast_websocket_event(event)
    
    async def _handle_agent_deleted(self, event: Event):
        """Handle agent deleted event."""
        agent_id = event.data.get("agent_id")
        
        logger.info(f"Agent deleted: {agent_id}")
        await self._broadcast_websocket_event(event)
    
    async def _handle_agent_execution_started(self, event: Event):
        """Handle agent execution started event."""
        agent_id = event.data.get("agent_id")
        capability_name = event.data.get("capability_name")
        
        logger.info(f"Agent execution started: {agent_id} - {capability_name}")
        await self._broadcast_websocket_event(event)
    
    async def _handle_agent_execution_completed(self, event: Event):
        """Handle agent execution completed event."""
        agent_id = event.data.get("agent_id")
        capability_name = event.data.get("capability_name")
        execution_time = event.data.get("execution_time")
        
        logger.info(f"Agent execution completed: {agent_id} - {capability_name} in {execution_time}s")
        await self._broadcast_websocket_event(event)
    
    async def _handle_agent_execution_failed(self, event: Event):
        """Handle agent execution failed event."""
        agent_id = event.data.get("agent_id")
        capability_name = event.data.get("capability_name")
        error = event.data.get("error")
        
        logger.error(f"Agent execution failed: {agent_id} - {capability_name}, error: {error}")
        await self._broadcast_websocket_event(event)
    
    async def _broadcast_websocket_event(self, event: Event):
        """Broadcast event to WebSocket subscribers."""
        try:
            from ..websocket.manager import websocket_manager
            
            if event.type == EventType.AGENT_EXECUTION_STARTED:
                await websocket_manager.broadcast_agent_execution_update(
                    agent_id=event.data.get("agent_id"),
                    capability_name=event.data.get("capability_name"),
                    status="started"
                )
            elif event.type == EventType.AGENT_EXECUTION_COMPLETED:
                await websocket_manager.broadcast_agent_execution_update(
                    agent_id=event.data.get("agent_id"),
                    capability_name=event.data.get("capability_name"),
                    status="completed",
                    result=event.data.get("result")
                )
            elif event.type == EventType.AGENT_EXECUTION_FAILED:
                await websocket_manager.broadcast_agent_execution_update(
                    agent_id=event.data.get("agent_id"),
                    capability_name=event.data.get("capability_name"),
                    status="failed",
                    error=event.data.get("error")
                )
        except Exception as e:
            logger.error(f"Failed to broadcast WebSocket event: {str(e)}")

class WorkflowEventHandler(EventHandler):
    """Handler for workflow-related events."""
    
    def __init__(self):
        super().__init__("workflow_event_handler")
    
    async def handle(self, event: Event) -> bool:
        """Handle workflow events."""
        try:
            if event.type == EventType.WORKFLOW_CREATED:
                await self._handle_workflow_created(event)
            elif event.type == EventType.WORKFLOW_UPDATED:
                await self._handle_workflow_updated(event)
            elif event.type == EventType.WORKFLOW_DELETED:
                await self._handle_workflow_deleted(event)
            elif event.type == EventType.WORKFLOW_EXECUTION_STARTED:
                await self._handle_workflow_execution_started(event)
            elif event.type == EventType.WORKFLOW_EXECUTION_COMPLETED:
                await self._handle_workflow_execution_completed(event)
            elif event.type == EventType.WORKFLOW_EXECUTION_FAILED:
                await self._handle_workflow_execution_failed(event)
            elif event.type == EventType.WORKFLOW_STEP_COMPLETED:
                await self._handle_workflow_step_completed(event)
            
            logger.debug(f"Handled workflow event: {event.type.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to handle workflow event {event.id}: {str(e)}")
            return False
    
    def can_handle(self, event: Event) -> bool:
        """Check if this handler can handle the event."""
        return event.type in [
            EventType.WORKFLOW_CREATED,
            EventType.WORKFLOW_UPDATED,
            EventType.WORKFLOW_DELETED,
            EventType.WORKFLOW_EXECUTION_STARTED,
            EventType.WORKFLOW_EXECUTION_COMPLETED,
            EventType.WORKFLOW_EXECUTION_FAILED,
            EventType.WORKFLOW_STEP_COMPLETED
        ]
    
    async def _handle_workflow_created(self, event: Event):
        """Handle workflow created event."""
        workflow_id = event.data.get("workflow_id")
        workflow_name = event.data.get("workflow_name")
        
        logger.info(f"Workflow created: {workflow_name} (ID: {workflow_id})")
        await self._broadcast_websocket_event(event)
    
    async def _handle_workflow_updated(self, event: Event):
        """Handle workflow updated event."""
        workflow_id = event.data.get("workflow_id")
        changes = event.data.get("changes", {})
        
        logger.info(f"Workflow updated: {workflow_id}, changes: {changes}")
        await self._broadcast_websocket_event(event)
    
    async def _handle_workflow_deleted(self, event: Event):
        """Handle workflow deleted event."""
        workflow_id = event.data.get("workflow_id")
        
        logger.info(f"Workflow deleted: {workflow_id}")
        await self._broadcast_websocket_event(event)
    
    async def _handle_workflow_execution_started(self, event: Event):
        """Handle workflow execution started event."""
        workflow_id = event.data.get("workflow_id")
        
        logger.info(f"Workflow execution started: {workflow_id}")
        await self._broadcast_websocket_event(event)
    
    async def _handle_workflow_execution_completed(self, event: Event):
        """Handle workflow execution completed event."""
        workflow_id = event.data.get("workflow_id")
        execution_time = event.data.get("execution_time")
        
        logger.info(f"Workflow execution completed: {workflow_id} in {execution_time}s")
        await self._broadcast_websocket_event(event)
    
    async def _handle_workflow_execution_failed(self, event: Event):
        """Handle workflow execution failed event."""
        workflow_id = event.data.get("workflow_id")
        error = event.data.get("error")
        
        logger.error(f"Workflow execution failed: {workflow_id}, error: {error}")
        await self._broadcast_websocket_event(event)
    
    async def _handle_workflow_step_completed(self, event: Event):
        """Handle workflow step completed event."""
        workflow_id = event.data.get("workflow_id")
        step_name = event.data.get("step_name")
        
        logger.info(f"Workflow step completed: {workflow_id} - {step_name}")
        await self._broadcast_websocket_event(event)
    
    async def _broadcast_websocket_event(self, event: Event):
        """Broadcast event to WebSocket subscribers."""
        try:
            from ..websocket.manager import websocket_manager
            
            if event.type == EventType.WORKFLOW_EXECUTION_STARTED:
                await websocket_manager.broadcast_workflow_execution_update(
                    workflow_id=event.data.get("workflow_id"),
                    status="started"
                )
            elif event.type == EventType.WORKFLOW_EXECUTION_COMPLETED:
                await websocket_manager.broadcast_workflow_execution_update(
                    workflow_id=event.data.get("workflow_id"),
                    status="completed",
                    result=event.data.get("result")
                )
            elif event.type == EventType.WORKFLOW_EXECUTION_FAILED:
                await websocket_manager.broadcast_workflow_execution_update(
                    workflow_id=event.data.get("workflow_id"),
                    status="failed",
                    error=event.data.get("error")
                )
            elif event.type == EventType.WORKFLOW_STEP_COMPLETED:
                await websocket_manager.broadcast_workflow_execution_update(
                    workflow_id=event.data.get("workflow_id"),
                    status="step_completed",
                    current_step=event.data.get("step_name")
                )
        except Exception as e:
            logger.error(f"Failed to broadcast WebSocket event: {str(e)}")

class DocumentEventHandler(EventHandler):
    """Handler for document-related events."""
    
    def __init__(self):
        super().__init__("document_event_handler")
    
    async def handle(self, event: Event) -> bool:
        """Handle document events."""
        try:
            if event.type == EventType.DOCUMENT_GENERATION_STARTED:
                await self._handle_document_generation_started(event)
            elif event.type == EventType.DOCUMENT_GENERATION_COMPLETED:
                await self._handle_document_generation_completed(event)
            elif event.type == EventType.DOCUMENT_GENERATION_FAILED:
                await self._handle_document_generation_failed(event)
            elif event.type == EventType.DOCUMENT_DOWNLOADED:
                await self._handle_document_downloaded(event)
            
            logger.debug(f"Handled document event: {event.type.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to handle document event {event.id}: {str(e)}")
            return False
    
    def can_handle(self, event: Event) -> bool:
        """Check if this handler can handle the event."""
        return event.type in [
            EventType.DOCUMENT_GENERATION_STARTED,
            EventType.DOCUMENT_GENERATION_COMPLETED,
            EventType.DOCUMENT_GENERATION_FAILED,
            EventType.DOCUMENT_DOWNLOADED
        ]
    
    async def _handle_document_generation_started(self, event: Event):
        """Handle document generation started event."""
        document_id = event.data.get("document_id")
        document_type = event.data.get("document_type")
        
        logger.info(f"Document generation started: {document_id} ({document_type})")
    
    async def _handle_document_generation_completed(self, event: Event):
        """Handle document generation completed event."""
        document_id = event.data.get("document_id")
        document_type = event.data.get("document_type")
        file_size = event.data.get("file_size")
        
        logger.info(f"Document generation completed: {document_id} ({document_type}), size: {file_size} bytes")
    
    async def _handle_document_generation_failed(self, event: Event):
        """Handle document generation failed event."""
        document_id = event.data.get("document_id")
        error = event.data.get("error")
        
        logger.error(f"Document generation failed: {document_id}, error: {error}")
    
    async def _handle_document_downloaded(self, event: Event):
        """Handle document downloaded event."""
        document_id = event.data.get("document_id")
        user_id = event.data.get("user_id")
        
        logger.info(f"Document downloaded: {document_id} by user {user_id}")

class SystemEventHandler(EventHandler):
    """Handler for system-related events."""
    
    def __init__(self):
        super().__init__("system_event_handler")
    
    async def handle(self, event: Event) -> bool:
        """Handle system events."""
        try:
            if event.type == EventType.SYSTEM_STARTUP:
                await self._handle_system_startup(event)
            elif event.type == EventType.SYSTEM_SHUTDOWN:
                await self._handle_system_shutdown(event)
            elif event.type == EventType.SYSTEM_ERROR:
                await self._handle_system_error(event)
            elif event.type == EventType.SYSTEM_ALERT:
                await self._handle_system_alert(event)
            elif event.type == EventType.METRICS_UPDATED:
                await self._handle_metrics_updated(event)
            
            logger.debug(f"Handled system event: {event.type.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to handle system event {event.id}: {str(e)}")
            return False
    
    def can_handle(self, event: Event) -> bool:
        """Check if this handler can handle the event."""
        return event.type in [
            EventType.SYSTEM_STARTUP,
            EventType.SYSTEM_SHUTDOWN,
            EventType.SYSTEM_ERROR,
            EventType.SYSTEM_ALERT,
            EventType.METRICS_UPDATED
        ]
    
    async def _handle_system_startup(self, event: Event):
        """Handle system startup event."""
        startup_time = event.data.get("startup_time")
        
        logger.info(f"System startup completed in {startup_time}s")
        await self._broadcast_websocket_event(event)
    
    async def _handle_system_shutdown(self, event: Event):
        """Handle system shutdown event."""
        shutdown_reason = event.data.get("shutdown_reason", "normal")
        
        logger.info(f"System shutdown: {shutdown_reason}")
        await self._broadcast_websocket_event(event)
    
    async def _handle_system_error(self, event: Event):
        """Handle system error event."""
        error_type = event.data.get("error_type")
        error_message = event.data.get("error_message")
        
        logger.error(f"System error: {error_type} - {error_message}")
        await self._broadcast_websocket_event(event)
    
    async def _handle_system_alert(self, event: Event):
        """Handle system alert event."""
        alert_type = event.data.get("alert_type")
        severity = event.data.get("severity")
        message = event.data.get("message")
        
        logger.warning(f"System alert: {alert_type} ({severity}) - {message}")
        await self._broadcast_websocket_event(event)
    
    async def _handle_metrics_updated(self, event: Event):
        """Handle metrics updated event."""
        metrics_type = event.data.get("metrics_type")
        
        logger.debug(f"Metrics updated: {metrics_type}")
        await self._broadcast_websocket_event(event)
    
    async def _broadcast_websocket_event(self, event: Event):
        """Broadcast event to WebSocket subscribers."""
        try:
            from ..websocket.manager import websocket_manager
            
            if event.type == EventType.SYSTEM_ALERT:
                await websocket_manager.broadcast_system_alert(
                    alert_type=event.data.get("alert_type"),
                    severity=event.data.get("severity"),
                    message_text=event.data.get("message"),
                    details=event.data.get("details")
                )
        except Exception as e:
            logger.error(f"Failed to broadcast WebSocket event: {str(e)}")
