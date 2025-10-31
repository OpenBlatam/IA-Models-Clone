"""
Automation Engine for Email Sequence System

This module provides advanced automation capabilities including workflow orchestration,
event-driven triggers, and intelligent decision making.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime, timedelta
from uuid import UUID
import json
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from .config import get_settings
from .exceptions import AutomationError
from .cache import cache_manager
from .real_time_engine import real_time_engine, EventType, RealTimeEvent

logger = logging.getLogger(__name__)
settings = get_settings()


class TriggerType(str, Enum):
    """Types of automation triggers"""
    TIME_BASED = "time_based"
    EVENT_BASED = "event_based"
    CONDITION_BASED = "condition_based"
    WEBHOOK = "webhook"
    API_CALL = "api_call"
    SCHEDULED = "scheduled"


class ActionType(str, Enum):
    """Types of automation actions"""
    SEND_EMAIL = "send_email"
    ADD_TO_SEQUENCE = "add_to_sequence"
    REMOVE_FROM_SEQUENCE = "remove_from_sequence"
    UPDATE_SUBSCRIBER = "update_subscriber"
    CREATE_SEGMENT = "create_segment"
    TRIGGER_WEBHOOK = "trigger_webhook"
    SEND_NOTIFICATION = "send_notification"
    PAUSE_SEQUENCE = "pause_sequence"
    RESUME_SEQUENCE = "resume_sequence"
    CUSTOM_ACTION = "custom_action"


class WorkflowStatus(str, Enum):
    """Workflow execution status"""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AutomationTrigger:
    """Automation trigger configuration"""
    trigger_type: TriggerType
    conditions: Dict[str, Any]
    schedule: Optional[str] = None  # Cron expression for scheduled triggers
    webhook_url: Optional[str] = None
    event_types: List[EventType] = field(default_factory=list)
    enabled: bool = True


@dataclass
class AutomationAction:
    """Automation action configuration"""
    action_type: ActionType
    parameters: Dict[str, Any]
    delay_seconds: int = 0
    retry_count: int = 3
    retry_delay: int = 60
    enabled: bool = True


@dataclass
class WorkflowStep:
    """Workflow step configuration"""
    step_id: str
    name: str
    trigger: AutomationTrigger
    actions: List[AutomationAction]
    conditions: Dict[str, Any] = field(default_factory=dict)
    on_success: Optional[str] = None  # Next step ID
    on_failure: Optional[str] = None  # Next step ID
    timeout_seconds: int = 300


@dataclass
class WorkflowExecution:
    """Workflow execution instance"""
    execution_id: UUID
    workflow_id: UUID
    status: WorkflowStatus
    current_step: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class AutomationEngine:
    """Advanced automation engine for email sequences"""
    
    def __init__(self):
        """Initialize automation engine"""
        self.workflows: Dict[UUID, List[WorkflowStep]] = {}
        self.executions: Dict[UUID, WorkflowExecution] = {}
        self.action_handlers: Dict[ActionType, Callable] = {}
        self.trigger_handlers: Dict[TriggerType, Callable] = {}
        self.is_running = False
        
        # Performance metrics
        self.workflows_executed = 0
        self.actions_executed = 0
        self.failed_executions = 0
        
        logger.info("Automation Engine initialized")
    
    async def initialize(self) -> None:
        """Initialize the automation engine"""
        try:
            # Register default action handlers
            self._register_default_handlers()
            
            # Start background tasks
            self.is_running = True
            asyncio.create_task(self._process_scheduled_triggers())
            asyncio.create_task(self._monitor_workflow_executions())
            asyncio.create_task(self._cleanup_completed_executions())
            
            # Register with real-time engine for event-based triggers
            real_time_engine.register_event_handler(EventType.EMAIL_OPENED, self._handle_email_event)
            real_time_engine.register_event_handler(EventType.EMAIL_CLICKED, self._handle_email_event)
            real_time_engine.register_event_handler(EventType.EMAIL_BOUNCED, self._handle_email_event)
            
            logger.info("Automation Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing automation engine: {e}")
            raise AutomationError(f"Failed to initialize automation engine: {e}")
    
    async def shutdown(self) -> None:
        """Shutdown the automation engine"""
        try:
            self.is_running = False
            
            # Cancel all running executions
            for execution in self.executions.values():
                if execution.status == WorkflowStatus.ACTIVE:
                    execution.status = WorkflowStatus.CANCELLED
                    execution.completed_at = datetime.utcnow()
            
            logger.info("Automation Engine shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during automation engine shutdown: {e}")
    
    async def create_workflow(
        self,
        workflow_id: UUID,
        name: str,
        description: str,
        steps: List[Dict[str, Any]],
        enabled: bool = True
    ) -> Dict[str, Any]:
        """
        Create a new automation workflow.
        
        Args:
            workflow_id: Unique workflow ID
            name: Workflow name
            description: Workflow description
            steps: List of workflow steps
            enabled: Whether workflow is enabled
            
        Returns:
            Workflow creation result
        """
        try:
            # Parse and validate steps
            workflow_steps = []
            for step_data in steps:
                step = await self._parse_workflow_step(step_data)
                workflow_steps.append(step)
            
            # Store workflow
            self.workflows[workflow_id] = workflow_steps
            
            # Cache workflow configuration
            workflow_config = {
                "workflow_id": str(workflow_id),
                "name": name,
                "description": description,
                "steps": [step.__dict__ for step in workflow_steps],
                "enabled": enabled,
                "created_at": datetime.utcnow().isoformat()
            }
            
            await cache_manager.set(f"workflow:{workflow_id}", workflow_config, 86400)
            
            logger.info(f"Workflow created: {workflow_id} ({name})")
            
            return {
                "status": "success",
                "workflow_id": str(workflow_id),
                "message": "Workflow created successfully"
            }
            
        except Exception as e:
            logger.error(f"Error creating workflow: {e}")
            raise AutomationError(f"Failed to create workflow: {e}")
    
    async def execute_workflow(
        self,
        workflow_id: UUID,
        trigger_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> UUID:
        """
        Execute a workflow.
        
        Args:
            workflow_id: Workflow ID to execute
            trigger_data: Data that triggered the workflow
            context: Additional execution context
            
        Returns:
            Execution ID
        """
        try:
            if workflow_id not in self.workflows:
                raise AutomationError(f"Workflow {workflow_id} not found")
            
            # Create execution instance
            execution_id = UUID()
            execution = WorkflowExecution(
                execution_id=execution_id,
                workflow_id=workflow_id,
                status=WorkflowStatus.ACTIVE,
                context=context or {}
            )
            
            # Add trigger data to context
            execution.context.update(trigger_data)
            
            # Store execution
            self.executions[execution_id] = execution
            
            # Start workflow execution
            asyncio.create_task(self._execute_workflow_steps(execution))
            
            logger.info(f"Workflow execution started: {execution_id} for workflow {workflow_id}")
            
            return execution_id
            
        except Exception as e:
            logger.error(f"Error executing workflow: {e}")
            raise AutomationError(f"Failed to execute workflow: {e}")
    
    async def pause_workflow_execution(self, execution_id: UUID) -> None:
        """
        Pause a workflow execution.
        
        Args:
            execution_id: Execution ID to pause
        """
        try:
            if execution_id in self.executions:
                execution = self.executions[execution_id]
                if execution.status == WorkflowStatus.ACTIVE:
                    execution.status = WorkflowStatus.PAUSED
                    logger.info(f"Workflow execution paused: {execution_id}")
                else:
                    raise AutomationError(f"Cannot pause execution {execution_id} with status {execution.status}")
            else:
                raise AutomationError(f"Execution {execution_id} not found")
                
        except Exception as e:
            logger.error(f"Error pausing workflow execution: {e}")
            raise AutomationError(f"Failed to pause workflow execution: {e}")
    
    async def resume_workflow_execution(self, execution_id: UUID) -> None:
        """
        Resume a paused workflow execution.
        
        Args:
            execution_id: Execution ID to resume
        """
        try:
            if execution_id in self.executions:
                execution = self.executions[execution_id]
                if execution.status == WorkflowStatus.PAUSED:
                    execution.status = WorkflowStatus.ACTIVE
                    # Continue execution
                    asyncio.create_task(self._execute_workflow_steps(execution))
                    logger.info(f"Workflow execution resumed: {execution_id}")
                else:
                    raise AutomationError(f"Cannot resume execution {execution_id} with status {execution.status}")
            else:
                raise AutomationError(f"Execution {execution_id} not found")
                
        except Exception as e:
            logger.error(f"Error resuming workflow execution: {e}")
            raise AutomationError(f"Failed to resume workflow execution: {e}")
    
    async def cancel_workflow_execution(self, execution_id: UUID) -> None:
        """
        Cancel a workflow execution.
        
        Args:
            execution_id: Execution ID to cancel
        """
        try:
            if execution_id in self.executions:
                execution = self.executions[execution_id]
                execution.status = WorkflowStatus.CANCELLED
                execution.completed_at = datetime.utcnow()
                logger.info(f"Workflow execution cancelled: {execution_id}")
            else:
                raise AutomationError(f"Execution {execution_id} not found")
                
        except Exception as e:
            logger.error(f"Error cancelling workflow execution: {e}")
            raise AutomationError(f"Failed to cancel workflow execution: {e}")
    
    async def get_workflow_status(self, workflow_id: UUID) -> Dict[str, Any]:
        """
        Get workflow status and statistics.
        
        Args:
            workflow_id: Workflow ID
            
        Returns:
            Workflow status information
        """
        try:
            # Get workflow configuration
            workflow_config = await cache_manager.get(f"workflow:{workflow_id}")
            if not workflow_config:
                raise AutomationError(f"Workflow {workflow_id} not found")
            
            # Get execution statistics
            executions = [exec for exec in self.executions.values() if exec.workflow_id == workflow_id]
            
            status_counts = {}
            for status in WorkflowStatus:
                status_counts[status.value] = len([exec for exec in executions if exec.status == status])
            
            return {
                "workflow_id": str(workflow_id),
                "name": workflow_config.get("name"),
                "description": workflow_config.get("description"),
                "enabled": workflow_config.get("enabled"),
                "total_executions": len(executions),
                "status_counts": status_counts,
                "active_executions": status_counts.get("active", 0),
                "last_execution": max([exec.started_at for exec in executions]).isoformat() if executions else None,
                "created_at": workflow_config.get("created_at")
            }
            
        except Exception as e:
            logger.error(f"Error getting workflow status: {e}")
            raise AutomationError(f"Failed to get workflow status: {e}")
    
    async def get_execution_details(self, execution_id: UUID) -> Dict[str, Any]:
        """
        Get detailed execution information.
        
        Args:
            execution_id: Execution ID
            
        Returns:
            Execution details
        """
        try:
            if execution_id not in self.executions:
                raise AutomationError(f"Execution {execution_id} not found")
            
            execution = self.executions[execution_id]
            
            return {
                "execution_id": str(execution_id),
                "workflow_id": str(execution.workflow_id),
                "status": execution.status.value,
                "current_step": execution.current_step,
                "context": execution.context,
                "started_at": execution.started_at.isoformat(),
                "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                "error_message": execution.error_message,
                "duration_seconds": (
                    (execution.completed_at or datetime.utcnow()) - execution.started_at
                ).total_seconds()
            }
            
        except Exception as e:
            logger.error(f"Error getting execution details: {e}")
            raise AutomationError(f"Failed to get execution details: {e}")
    
    # Private helper methods
    async def _parse_workflow_step(self, step_data: Dict[str, Any]) -> WorkflowStep:
        """Parse workflow step from configuration"""
        try:
            trigger = AutomationTrigger(
                trigger_type=TriggerType(step_data["trigger"]["type"]),
                conditions=step_data["trigger"].get("conditions", {}),
                schedule=step_data["trigger"].get("schedule"),
                webhook_url=step_data["trigger"].get("webhook_url"),
                event_types=[EventType(et) for et in step_data["trigger"].get("event_types", [])],
                enabled=step_data["trigger"].get("enabled", True)
            )
            
            actions = []
            for action_data in step_data.get("actions", []):
                action = AutomationAction(
                    action_type=ActionType(action_data["type"]),
                    parameters=action_data.get("parameters", {}),
                    delay_seconds=action_data.get("delay_seconds", 0),
                    retry_count=action_data.get("retry_count", 3),
                    retry_delay=action_data.get("retry_delay", 60),
                    enabled=action_data.get("enabled", True)
                )
                actions.append(action)
            
            return WorkflowStep(
                step_id=step_data["step_id"],
                name=step_data["name"],
                trigger=trigger,
                actions=actions,
                conditions=step_data.get("conditions", {}),
                on_success=step_data.get("on_success"),
                on_failure=step_data.get("on_failure"),
                timeout_seconds=step_data.get("timeout_seconds", 300)
            )
            
        except Exception as e:
            raise AutomationError(f"Error parsing workflow step: {e}")
    
    async def _execute_workflow_steps(self, execution: WorkflowExecution) -> None:
        """Execute workflow steps"""
        try:
            workflow_steps = self.workflows[execution.workflow_id]
            
            for step in workflow_steps:
                if execution.status != WorkflowStatus.ACTIVE:
                    break
                
                execution.current_step = step.step_id
                
                # Check step conditions
                if not await self._evaluate_conditions(step.conditions, execution.context):
                    logger.info(f"Skipping step {step.step_id} due to conditions")
                    continue
                
                # Execute step actions
                try:
                    await self._execute_step_actions(step, execution)
                    
                    # Move to next step on success
                    if step.on_success:
                        # Find next step
                        next_step = next((s for s in workflow_steps if s.step_id == step.on_success), None)
                        if next_step:
                            continue
                    
                except Exception as e:
                    logger.error(f"Error executing step {step.step_id}: {e}")
                    execution.error_message = str(e)
                    
                    # Move to failure step
                    if step.on_failure:
                        next_step = next((s for s in workflow_steps if s.step_id == step.on_failure), None)
                        if next_step:
                            continue
                    else:
                        execution.status = WorkflowStatus.FAILED
                        break
            
            # Mark execution as completed
            if execution.status == WorkflowStatus.ACTIVE:
                execution.status = WorkflowStatus.COMPLETED
            
            execution.completed_at = datetime.utcnow()
            self.workflows_executed += 1
            
            logger.info(f"Workflow execution completed: {execution.execution_id} with status {execution.status}")
            
        except Exception as e:
            logger.error(f"Error executing workflow steps: {e}")
            execution.status = WorkflowStatus.FAILED
            execution.error_message = str(e)
            execution.completed_at = datetime.utcnow()
            self.failed_executions += 1
    
    async def _execute_step_actions(self, step: WorkflowStep, execution: WorkflowExecution) -> None:
        """Execute actions for a workflow step"""
        for action in step.actions:
            if not action.enabled:
                continue
            
            # Apply delay if specified
            if action.delay_seconds > 0:
                await asyncio.sleep(action.delay_seconds)
            
            # Execute action with retry logic
            for attempt in range(action.retry_count + 1):
                try:
                    if action.action_type in self.action_handlers:
                        await self.action_handlers[action.action_type](action, execution)
                        self.actions_executed += 1
                        break
                    else:
                        logger.warning(f"No handler for action type: {action.action_type}")
                        break
                        
                except Exception as e:
                    if attempt < action.retry_count:
                        logger.warning(f"Action failed, retrying in {action.retry_delay}s: {e}")
                        await asyncio.sleep(action.retry_delay)
                    else:
                        raise e
    
    async def _evaluate_conditions(self, conditions: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate workflow step conditions"""
        try:
            # Simple condition evaluation
            # In production, implement more sophisticated condition evaluation
            
            for key, expected_value in conditions.items():
                if key not in context:
                    return False
                
                actual_value = context[key]
                if actual_value != expected_value:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error evaluating conditions: {e}")
            return False
    
    async def _handle_email_event(self, event: RealTimeEvent) -> None:
        """Handle email events for event-based triggers"""
        try:
            # Find workflows with matching event-based triggers
            for workflow_id, steps in self.workflows.items():
                for step in steps:
                    if (step.trigger.trigger_type == TriggerType.EVENT_BASED and
                        event.event_type in step.trigger.event_types and
                        step.trigger.enabled):
                        
                        # Execute workflow
                        await self.execute_workflow(
                            workflow_id=workflow_id,
                            trigger_data={
                                "event_type": event.event_type.value,
                                "sequence_id": str(event.sequence_id),
                                "subscriber_id": str(event.subscriber_id) if event.subscriber_id else None,
                                "data": event.data
                            }
                        )
            
        except Exception as e:
            logger.error(f"Error handling email event: {e}")
    
    async def _process_scheduled_triggers(self) -> None:
        """Process scheduled triggers"""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # In production, implement proper cron scheduling
                # For now, this is a placeholder
                
            except Exception as e:
                logger.error(f"Error processing scheduled triggers: {e}")
    
    async def _monitor_workflow_executions(self) -> None:
        """Monitor workflow executions for timeouts"""
        while self.is_running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                current_time = datetime.utcnow()
                
                for execution in self.executions.values():
                    if execution.status == WorkflowStatus.ACTIVE:
                        # Check for timeout
                        if (current_time - execution.started_at).total_seconds() > 3600:  # 1 hour timeout
                            execution.status = WorkflowStatus.FAILED
                            execution.error_message = "Workflow execution timeout"
                            execution.completed_at = current_time
                            self.failed_executions += 1
                            
                            logger.warning(f"Workflow execution timed out: {execution.execution_id}")
                
            except Exception as e:
                logger.error(f"Error monitoring workflow executions: {e}")
    
    async def _cleanup_completed_executions(self) -> None:
        """Clean up completed executions"""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Clean up every 5 minutes
                
                # Remove executions older than 24 hours
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                
                executions_to_remove = []
                for execution_id, execution in self.executions.items():
                    if (execution.completed_at and 
                        execution.completed_at < cutoff_time and
                        execution.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]):
                        executions_to_remove.append(execution_id)
                
                for execution_id in executions_to_remove:
                    del self.executions[execution_id]
                
                if executions_to_remove:
                    logger.info(f"Cleaned up {len(executions_to_remove)} completed executions")
                
            except Exception as e:
                logger.error(f"Error cleaning up executions: {e}")
    
    def _register_default_handlers(self) -> None:
        """Register default action handlers"""
        # Register placeholder handlers
        # In production, implement actual action handlers
        
        self.action_handlers[ActionType.SEND_EMAIL] = self._handle_send_email
        self.action_handlers[ActionType.ADD_TO_SEQUENCE] = self._handle_add_to_sequence
        self.action_handlers[ActionType.REMOVE_FROM_SEQUENCE] = self._handle_remove_from_sequence
        self.action_handlers[ActionType.UPDATE_SUBSCRIBER] = self._handle_update_subscriber
        self.action_handlers[ActionType.TRIGGER_WEBHOOK] = self._handle_trigger_webhook
        self.action_handlers[ActionType.SEND_NOTIFICATION] = self._handle_send_notification
    
    # Default action handlers (placeholder implementations)
    async def _handle_send_email(self, action: AutomationAction, execution: WorkflowExecution) -> None:
        """Handle send email action"""
        logger.info(f"Executing send email action: {action.parameters}")
        # Implement actual email sending logic
    
    async def _handle_add_to_sequence(self, action: AutomationAction, execution: WorkflowExecution) -> None:
        """Handle add to sequence action"""
        logger.info(f"Executing add to sequence action: {action.parameters}")
        # Implement actual sequence addition logic
    
    async def _handle_remove_from_sequence(self, action: AutomationAction, execution: WorkflowExecution) -> None:
        """Handle remove from sequence action"""
        logger.info(f"Executing remove from sequence action: {action.parameters}")
        # Implement actual sequence removal logic
    
    async def _handle_update_subscriber(self, action: AutomationAction, execution: WorkflowExecution) -> None:
        """Handle update subscriber action"""
        logger.info(f"Executing update subscriber action: {action.parameters}")
        # Implement actual subscriber update logic
    
    async def _handle_trigger_webhook(self, action: AutomationAction, execution: WorkflowExecution) -> None:
        """Handle trigger webhook action"""
        logger.info(f"Executing trigger webhook action: {action.parameters}")
        # Implement actual webhook triggering logic
    
    async def _handle_send_notification(self, action: AutomationAction, execution: WorkflowExecution) -> None:
        """Handle send notification action"""
        logger.info(f"Executing send notification action: {action.parameters}")
        # Implement actual notification sending logic


# Global automation engine instance
automation_engine = AutomationEngine()






























