"""
Advanced Workflow Automation System
===================================

This module provides advanced workflow automation capabilities including
scheduling, triggers, conditions, and intelligent workflow orchestration.
"""

import asyncio
import logging
import json
import re
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import uuid
from enum import Enum
import schedule
import croniter
from collections import defaultdict, deque

# Configure logging
logger = logging.getLogger(__name__)

class TriggerType(Enum):
    """Types of workflow triggers"""
    SCHEDULED = "scheduled"
    EVENT_BASED = "event_based"
    CONDITION_BASED = "condition_based"
    MANUAL = "manual"
    API_CALL = "api_call"
    WEBHOOK = "webhook"
    FILE_UPLOAD = "file_upload"
    CONTENT_ANALYSIS = "content_analysis"
    TREND_DETECTION = "trend_detection"
    PERFORMANCE_THRESHOLD = "performance_threshold"

class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    SCHEDULED = "scheduled"

class ConditionOperator(Enum):
    """Condition operators for workflow automation"""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    REGEX_MATCH = "regex_match"
    IN_LIST = "in_list"
    NOT_IN_LIST = "not_in_list"

@dataclass
class WorkflowTrigger:
    """Workflow trigger configuration"""
    id: str
    trigger_type: TriggerType
    name: str
    description: str
    configuration: Dict[str, Any]
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkflowCondition:
    """Workflow condition for conditional execution"""
    id: str
    name: str
    field: str
    operator: ConditionOperator
    value: Any
    description: str = ""
    enabled: bool = True

@dataclass
class WorkflowAction:
    """Workflow action to execute"""
    id: str
    name: str
    action_type: str
    configuration: Dict[str, Any]
    conditions: List[WorkflowCondition] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    timeout: int = 300
    enabled: bool = True

@dataclass
class WorkflowExecution:
    """Workflow execution record"""
    id: str
    workflow_id: str
    trigger_id: str
    status: WorkflowStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    execution_data: Dict[str, Any] = field(default_factory=dict)
    actions_executed: List[str] = field(default_factory=list)
    actions_failed: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AutomatedWorkflow:
    """Automated workflow configuration"""
    id: str
    name: str
    description: str
    workflow_chain_id: str
    triggers: List[WorkflowTrigger]
    actions: List[WorkflowAction]
    conditions: List[WorkflowCondition]
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    execution_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

class WorkflowAutomationEngine:
    """Advanced workflow automation engine"""
    
    def __init__(self):
        self.automated_workflows: Dict[str, AutomatedWorkflow] = {}
        self.execution_history: List[WorkflowExecution] = []
        self.active_executions: Dict[str, WorkflowExecution] = {}
        self.trigger_registry: Dict[str, Callable] = {}
        self.action_registry: Dict[str, Callable] = {}
        self.condition_evaluators: Dict[ConditionOperator, Callable] = {}
        
        # Initialize condition evaluators
        self._initialize_condition_evaluators()
        
        # Initialize default triggers and actions
        self._initialize_default_triggers()
        self._initialize_default_actions()
        
        # Scheduler for time-based triggers
        self.scheduler = schedule
        self.cron_scheduler = {}
        
        # Event system
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Performance monitoring
        self.performance_metrics: Dict[str, Any] = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0,
            "trigger_frequency": defaultdict(int)
        }
    
    def _initialize_condition_evaluators(self):
        """Initialize condition evaluation functions"""
        self.condition_evaluators = {
            ConditionOperator.EQUALS: lambda field_value, condition_value: field_value == condition_value,
            ConditionOperator.NOT_EQUALS: lambda field_value, condition_value: field_value != condition_value,
            ConditionOperator.GREATER_THAN: lambda field_value, condition_value: field_value > condition_value,
            ConditionOperator.LESS_THAN: lambda field_value, condition_value: field_value < condition_value,
            ConditionOperator.CONTAINS: lambda field_value, condition_value: condition_value in str(field_value),
            ConditionOperator.NOT_CONTAINS: lambda field_value, condition_value: condition_value not in str(field_value),
            ConditionOperator.STARTS_WITH: lambda field_value, condition_value: str(field_value).startswith(str(condition_value)),
            ConditionOperator.ENDS_WITH: lambda field_value, condition_value: str(field_value).endswith(str(condition_value)),
            ConditionOperator.REGEX_MATCH: lambda field_value, condition_value: bool(re.search(condition_value, str(field_value))),
            ConditionOperator.IN_LIST: lambda field_value, condition_value: field_value in condition_value,
            ConditionOperator.NOT_IN_LIST: lambda field_value, condition_value: field_value not in condition_value
        }
    
    def _initialize_default_triggers(self):
        """Initialize default trigger types"""
        self.trigger_registry = {
            "scheduled": self._handle_scheduled_trigger,
            "event_based": self._handle_event_trigger,
            "condition_based": self._handle_condition_trigger,
            "api_call": self._handle_api_trigger,
            "webhook": self._handle_webhook_trigger,
            "file_upload": self._handle_file_upload_trigger,
            "content_analysis": self._handle_content_analysis_trigger,
            "trend_detection": self._handle_trend_detection_trigger,
            "performance_threshold": self._handle_performance_threshold_trigger
        }
    
    def _initialize_default_actions(self):
        """Initialize default action types"""
        self.action_registry = {
            "create_workflow": self._execute_create_workflow,
            "continue_workflow": self._execute_continue_workflow,
            "generate_content": self._execute_generate_content,
            "analyze_content": self._execute_analyze_content,
            "optimize_prompt": self._execute_optimize_prompt,
            "send_notification": self._execute_send_notification,
            "publish_content": self._execute_publish_content,
            "schedule_workflow": self._execute_schedule_workflow,
            "pause_workflow": self._execute_pause_workflow,
            "resume_workflow": self._execute_resume_workflow,
            "cancel_workflow": self._execute_cancel_workflow,
            "update_workflow_settings": self._execute_update_workflow_settings,
            "export_workflow_data": self._execute_export_workflow_data,
            "backup_workflow": self._execute_backup_workflow,
            "restore_workflow": self._execute_restore_workflow
        }
    
    async def create_automated_workflow(
        self,
        name: str,
        description: str,
        workflow_chain_id: str,
        triggers: List[Dict[str, Any]],
        actions: List[Dict[str, Any]],
        conditions: List[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new automated workflow
        
        Args:
            name: Name of the automated workflow
            description: Description of the workflow
            workflow_chain_id: ID of the workflow chain to automate
            triggers: List of trigger configurations
            actions: List of action configurations
            conditions: List of condition configurations
            
        Returns:
            str: ID of the created automated workflow
        """
        try:
            workflow_id = str(uuid.uuid4())
            
            # Create triggers
            workflow_triggers = []
            for trigger_config in triggers:
                trigger = WorkflowTrigger(
                    id=str(uuid.uuid4()),
                    trigger_type=TriggerType(trigger_config["type"]),
                    name=trigger_config["name"],
                    description=trigger_config.get("description", ""),
                    configuration=trigger_config.get("configuration", {}),
                    enabled=trigger_config.get("enabled", True)
                )
                workflow_triggers.append(trigger)
            
            # Create actions
            workflow_actions = []
            for action_config in actions:
                action = WorkflowAction(
                    id=str(uuid.uuid4()),
                    name=action_config["name"],
                    action_type=action_config["type"],
                    configuration=action_config.get("configuration", {}),
                    retry_count=0,
                    max_retries=action_config.get("max_retries", 3),
                    timeout=action_config.get("timeout", 300),
                    enabled=action_config.get("enabled", True)
                )
                workflow_actions.append(action)
            
            # Create conditions
            workflow_conditions = []
            if conditions:
                for condition_config in conditions:
                    condition = WorkflowCondition(
                        id=str(uuid.uuid4()),
                        name=condition_config["name"],
                        field=condition_config["field"],
                        operator=ConditionOperator(condition_config["operator"]),
                        value=condition_config["value"],
                        description=condition_config.get("description", ""),
                        enabled=condition_config.get("enabled", True)
                    )
                    workflow_conditions.append(condition)
            
            # Create automated workflow
            automated_workflow = AutomatedWorkflow(
                id=workflow_id,
                name=name,
                description=description,
                workflow_chain_id=workflow_chain_id,
                triggers=workflow_triggers,
                actions=workflow_actions,
                conditions=workflow_conditions
            )
            
            self.automated_workflows[workflow_id] = automated_workflow
            
            # Set up triggers
            await self._setup_workflow_triggers(automated_workflow)
            
            logger.info(f"Created automated workflow: {workflow_id}")
            return workflow_id
            
        except Exception as e:
            logger.error(f"Error creating automated workflow: {str(e)}")
            raise
    
    async def execute_workflow(
        self,
        workflow_id: str,
        trigger_id: str,
        execution_data: Dict[str, Any] = None
    ) -> str:
        """
        Execute an automated workflow
        
        Args:
            workflow_id: ID of the automated workflow
            trigger_id: ID of the trigger that initiated execution
            execution_data: Additional data for execution
            
        Returns:
            str: ID of the execution
        """
        try:
            if workflow_id not in self.automated_workflows:
                raise ValueError(f"Automated workflow {workflow_id} not found")
            
            automated_workflow = self.automated_workflows[workflow_id]
            
            if not automated_workflow.enabled:
                raise ValueError(f"Automated workflow {workflow_id} is disabled")
            
            # Create execution record
            execution_id = str(uuid.uuid4())
            execution = WorkflowExecution(
                id=execution_id,
                workflow_id=workflow_id,
                trigger_id=trigger_id,
                status=WorkflowStatus.RUNNING,
                started_at=datetime.now(),
                execution_data=execution_data or {}
            )
            
            self.active_executions[execution_id] = execution
            
            # Check conditions
            if not await self._evaluate_conditions(automated_workflow.conditions, execution_data):
                execution.status = WorkflowStatus.CANCELLED
                execution.completed_at = datetime.now()
                execution.error_message = "Conditions not met"
                self._finalize_execution(execution)
                return execution_id
            
            # Execute actions
            for action in automated_workflow.actions:
                if not action.enabled:
                    continue
                
                try:
                    await self._execute_action(action, execution)
                    execution.actions_executed.append(action.id)
                except Exception as e:
                    logger.error(f"Error executing action {action.id}: {str(e)}")
                    execution.actions_failed.append(action.id)
                    
                    if action.retry_count < action.max_retries:
                        action.retry_count += 1
                        # Retry action
                        try:
                            await asyncio.sleep(2 ** action.retry_count)  # Exponential backoff
                            await self._execute_action(action, execution)
                            execution.actions_executed.append(action.id)
                        except Exception as retry_error:
                            logger.error(f"Retry failed for action {action.id}: {str(retry_error)}")
                            execution.actions_failed.append(action.id)
                    else:
                        execution.status = WorkflowStatus.FAILED
                        execution.error_message = f"Action {action.id} failed after {action.max_retries} retries"
                        break
            
            # Finalize execution
            if execution.status == WorkflowStatus.RUNNING:
                execution.status = WorkflowStatus.COMPLETED
            
            execution.completed_at = datetime.now()
            self._finalize_execution(execution)
            
            # Update workflow statistics
            automated_workflow.execution_count += 1
            if execution.status == WorkflowStatus.COMPLETED:
                automated_workflow.success_count += 1
            else:
                automated_workflow.failure_count += 1
            
            # Update performance metrics
            self._update_performance_metrics(execution)
            
            logger.info(f"Executed automated workflow: {workflow_id}, execution: {execution_id}")
            return execution_id
            
        except Exception as e:
            logger.error(f"Error executing automated workflow: {str(e)}")
            if execution_id in self.active_executions:
                execution = self.active_executions[execution_id]
                execution.status = WorkflowStatus.FAILED
                execution.error_message = str(e)
                execution.completed_at = datetime.now()
                self._finalize_execution(execution)
            raise
    
    async def _setup_workflow_triggers(self, automated_workflow: AutomatedWorkflow):
        """Set up triggers for an automated workflow"""
        try:
            for trigger in automated_workflow.triggers:
                if not trigger.enabled:
                    continue
                
                if trigger.trigger_type == TriggerType.SCHEDULED:
                    await self._setup_scheduled_trigger(trigger, automated_workflow.id)
                elif trigger.trigger_type == TriggerType.EVENT_BASED:
                    await self._setup_event_trigger(trigger, automated_workflow.id)
                elif trigger.trigger_type == TriggerType.CONDITION_BASED:
                    await self._setup_condition_trigger(trigger, automated_workflow.id)
                elif trigger.trigger_type == TriggerType.WEBHOOK:
                    await self._setup_webhook_trigger(trigger, automated_workflow.id)
                
        except Exception as e:
            logger.error(f"Error setting up triggers: {str(e)}")
    
    async def _setup_scheduled_trigger(self, trigger: WorkflowTrigger, workflow_id: str):
        """Set up a scheduled trigger"""
        try:
            config = trigger.configuration
            schedule_type = config.get("schedule_type", "cron")
            
            if schedule_type == "cron":
                cron_expression = config.get("cron_expression")
                if cron_expression:
                    # Use croniter for cron expressions
                    cron = croniter.croniter(cron_expression)
                    self.cron_scheduler[trigger.id] = {
                        "cron": cron,
                        "workflow_id": workflow_id,
                        "trigger_id": trigger.id,
                        "next_run": cron.get_next(datetime)
                    }
            elif schedule_type == "interval":
                interval = config.get("interval_seconds", 3600)
                # Use schedule library for interval-based scheduling
                job = self.scheduler.every(interval).seconds.do(
                    self._trigger_scheduled_workflow,
                    workflow_id,
                    trigger.id
                )
                self.cron_scheduler[trigger.id] = {
                    "job": job,
                    "workflow_id": workflow_id,
                    "trigger_id": trigger.id
                }
            elif schedule_type == "daily":
                time_str = config.get("time", "09:00")
                job = self.scheduler.every().day.at(time_str).do(
                    self._trigger_scheduled_workflow,
                    workflow_id,
                    trigger.id
                )
                self.cron_scheduler[trigger.id] = {
                    "job": job,
                    "workflow_id": workflow_id,
                    "trigger_id": trigger.id
                }
            elif schedule_type == "weekly":
                day = config.get("day", "monday")
                time_str = config.get("time", "09:00")
                job = getattr(self.scheduler.every(), day.lower()).at(time_str).do(
                    self._trigger_scheduled_workflow,
                    workflow_id,
                    trigger.id
                )
                self.cron_scheduler[trigger.id] = {
                    "job": job,
                    "workflow_id": workflow_id,
                    "trigger_id": trigger.id
                }
            
            logger.info(f"Set up scheduled trigger: {trigger.id}")
            
        except Exception as e:
            logger.error(f"Error setting up scheduled trigger: {str(e)}")
    
    async def _setup_event_trigger(self, trigger: WorkflowTrigger, workflow_id: str):
        """Set up an event-based trigger"""
        try:
            config = trigger.configuration
            event_type = config.get("event_type")
            
            if event_type:
                # Register event handler
                self.event_handlers[event_type].append(
                    lambda event_data: self._trigger_event_workflow(
                        workflow_id, trigger.id, event_data
                    )
                )
            
            logger.info(f"Set up event trigger: {trigger.id}")
            
        except Exception as e:
            logger.error(f"Error setting up event trigger: {str(e)}")
    
    async def _setup_condition_trigger(self, trigger: WorkflowTrigger, workflow_id: str):
        """Set up a condition-based trigger"""
        try:
            config = trigger.configuration
            check_interval = config.get("check_interval_seconds", 60)
            
            # Set up periodic condition checking
            job = self.scheduler.every(check_interval).seconds.do(
                self._check_condition_trigger,
                workflow_id,
                trigger.id
            )
            self.cron_scheduler[trigger.id] = {
                "job": job,
                "workflow_id": workflow_id,
                "trigger_id": trigger.id
            }
            
            logger.info(f"Set up condition trigger: {trigger.id}")
            
        except Exception as e:
            logger.error(f"Error setting up condition trigger: {str(e)}")
    
    async def _setup_webhook_trigger(self, trigger: WorkflowTrigger, workflow_id: str):
        """Set up a webhook trigger"""
        try:
            config = trigger.configuration
            webhook_path = config.get("webhook_path", f"/webhook/{trigger.id}")
            
            # Register webhook endpoint (this would be integrated with the web framework)
            # For now, we'll store the configuration
            trigger.metadata["webhook_path"] = webhook_path
            trigger.metadata["workflow_id"] = workflow_id
            
            logger.info(f"Set up webhook trigger: {trigger.id} at {webhook_path}")
            
        except Exception as e:
            logger.error(f"Error setting up webhook trigger: {str(e)}")
    
    async def _evaluate_conditions(
        self,
        conditions: List[WorkflowCondition],
        execution_data: Dict[str, Any]
    ) -> bool:
        """Evaluate workflow conditions"""
        try:
            if not conditions:
                return True
            
            for condition in conditions:
                if not condition.enabled:
                    continue
                
                field_value = execution_data.get(condition.field)
                evaluator = self.condition_evaluators.get(condition.operator)
                
                if not evaluator:
                    logger.warning(f"Unknown condition operator: {condition.operator}")
                    continue
                
                if not evaluator(field_value, condition.value):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error evaluating conditions: {str(e)}")
            return False
    
    async def _execute_action(self, action: WorkflowAction, execution: WorkflowExecution):
        """Execute a workflow action"""
        try:
            action_handler = self.action_registry.get(action.action_type)
            if not action_handler:
                raise ValueError(f"Unknown action type: {action.action_type}")
            
            # Set timeout for action execution
            await asyncio.wait_for(
                action_handler(action, execution),
                timeout=action.timeout
            )
            
        except asyncio.TimeoutError:
            raise Exception(f"Action {action.id} timed out after {action.timeout} seconds")
        except Exception as e:
            logger.error(f"Error executing action {action.id}: {str(e)}")
            raise
    
    # Action implementations
    async def _execute_create_workflow(self, action: WorkflowAction, execution: WorkflowExecution):
        """Execute create workflow action"""
        config = action.configuration
        # Implementation would depend on the workflow engine
        logger.info(f"Executing create workflow action: {action.id}")
    
    async def _execute_continue_workflow(self, action: WorkflowAction, execution: WorkflowExecution):
        """Execute continue workflow action"""
        config = action.configuration
        # Implementation would depend on the workflow engine
        logger.info(f"Executing continue workflow action: {action.id}")
    
    async def _execute_generate_content(self, action: WorkflowAction, execution: WorkflowExecution):
        """Execute generate content action"""
        config = action.configuration
        # Implementation would depend on the content generation system
        logger.info(f"Executing generate content action: {action.id}")
    
    async def _execute_analyze_content(self, action: WorkflowAction, execution: WorkflowExecution):
        """Execute analyze content action"""
        config = action.configuration
        # Implementation would depend on the content analysis system
        logger.info(f"Executing analyze content action: {action.id}")
    
    async def _execute_optimize_prompt(self, action: WorkflowAction, execution: WorkflowExecution):
        """Execute optimize prompt action"""
        config = action.configuration
        # Implementation would depend on the prompt optimization system
        logger.info(f"Executing optimize prompt action: {action.id}")
    
    async def _execute_send_notification(self, action: WorkflowAction, execution: WorkflowExecution):
        """Execute send notification action"""
        config = action.configuration
        # Implementation would depend on the notification system
        logger.info(f"Executing send notification action: {action.id}")
    
    async def _execute_publish_content(self, action: WorkflowAction, execution: WorkflowExecution):
        """Execute publish content action"""
        config = action.configuration
        # Implementation would depend on the publishing system
        logger.info(f"Executing publish content action: {action.id}")
    
    async def _execute_schedule_workflow(self, action: WorkflowAction, execution: WorkflowExecution):
        """Execute schedule workflow action"""
        config = action.configuration
        # Implementation would depend on the scheduling system
        logger.info(f"Executing schedule workflow action: {action.id}")
    
    async def _execute_pause_workflow(self, action: WorkflowAction, execution: WorkflowExecution):
        """Execute pause workflow action"""
        config = action.configuration
        # Implementation would depend on the workflow engine
        logger.info(f"Executing pause workflow action: {action.id}")
    
    async def _execute_resume_workflow(self, action: WorkflowAction, execution: WorkflowExecution):
        """Execute resume workflow action"""
        config = action.configuration
        # Implementation would depend on the workflow engine
        logger.info(f"Executing resume workflow action: {action.id}")
    
    async def _execute_cancel_workflow(self, action: WorkflowAction, execution: WorkflowExecution):
        """Execute cancel workflow action"""
        config = action.configuration
        # Implementation would depend on the workflow engine
        logger.info(f"Executing cancel workflow action: {action.id}")
    
    async def _execute_update_workflow_settings(self, action: WorkflowAction, execution: WorkflowExecution):
        """Execute update workflow settings action"""
        config = action.configuration
        # Implementation would depend on the workflow engine
        logger.info(f"Executing update workflow settings action: {action.id}")
    
    async def _execute_export_workflow_data(self, action: WorkflowAction, execution: WorkflowExecution):
        """Execute export workflow data action"""
        config = action.configuration
        # Implementation would depend on the data export system
        logger.info(f"Executing export workflow data action: {action.id}")
    
    async def _execute_backup_workflow(self, action: WorkflowAction, execution: WorkflowExecution):
        """Execute backup workflow action"""
        config = action.configuration
        # Implementation would depend on the backup system
        logger.info(f"Executing backup workflow action: {action.id}")
    
    async def _execute_restore_workflow(self, action: WorkflowAction, execution: WorkflowExecution):
        """Execute restore workflow action"""
        config = action.configuration
        # Implementation would depend on the restore system
        logger.info(f"Executing restore workflow action: {action.id}")
    
    # Trigger handlers
    def _trigger_scheduled_workflow(self, workflow_id: str, trigger_id: str):
        """Trigger a scheduled workflow"""
        try:
            asyncio.create_task(self.execute_workflow(workflow_id, trigger_id))
        except Exception as e:
            logger.error(f"Error triggering scheduled workflow: {str(e)}")
    
    def _trigger_event_workflow(self, workflow_id: str, trigger_id: str, event_data: Dict[str, Any]):
        """Trigger an event-based workflow"""
        try:
            asyncio.create_task(self.execute_workflow(workflow_id, trigger_id, event_data))
        except Exception as e:
            logger.error(f"Error triggering event workflow: {str(e)}")
    
    def _check_condition_trigger(self, workflow_id: str, trigger_id: str):
        """Check condition-based trigger"""
        try:
            # This would check the conditions and trigger if met
            asyncio.create_task(self.execute_workflow(workflow_id, trigger_id))
        except Exception as e:
            logger.error(f"Error checking condition trigger: {str(e)}")
    
    def _finalize_execution(self, execution: WorkflowExecution):
        """Finalize workflow execution"""
        try:
            # Move from active to history
            if execution.id in self.active_executions:
                del self.active_executions[execution.id]
            
            self.execution_history.append(execution)
            
            # Keep only recent history (last 1000 executions)
            if len(self.execution_history) > 1000:
                self.execution_history = self.execution_history[-1000:]
            
            logger.info(f"Finalized execution: {execution.id}")
            
        except Exception as e:
            logger.error(f"Error finalizing execution: {str(e)}")
    
    def _update_performance_metrics(self, execution: WorkflowExecution):
        """Update performance metrics"""
        try:
            self.performance_metrics["total_executions"] += 1
            
            if execution.status == WorkflowStatus.COMPLETED:
                self.performance_metrics["successful_executions"] += 1
            else:
                self.performance_metrics["failed_executions"] += 1
            
            # Update average execution time
            if execution.completed_at and execution.started_at:
                execution_time = (execution.completed_at - execution.started_at).total_seconds()
                current_avg = self.performance_metrics["average_execution_time"]
                total_executions = self.performance_metrics["total_executions"]
                self.performance_metrics["average_execution_time"] = (
                    (current_avg * (total_executions - 1) + execution_time) / total_executions
                )
            
            # Update trigger frequency
            self.performance_metrics["trigger_frequency"][execution.trigger_id] += 1
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {str(e)}")
    
    async def get_workflow_execution_history(
        self,
        workflow_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get workflow execution history"""
        try:
            history = self.execution_history
            
            if workflow_id:
                history = [exec for exec in history if exec.workflow_id == workflow_id]
            
            # Sort by started_at descending
            history.sort(key=lambda x: x.started_at, reverse=True)
            
            # Limit results
            history = history[:limit]
            
            return [
                {
                    "id": exec.id,
                    "workflow_id": exec.workflow_id,
                    "trigger_id": exec.trigger_id,
                    "status": exec.status.value,
                    "started_at": exec.started_at.isoformat(),
                    "completed_at": exec.completed_at.isoformat() if exec.completed_at else None,
                    "error_message": exec.error_message,
                    "actions_executed": exec.actions_executed,
                    "actions_failed": exec.actions_failed,
                    "performance_metrics": exec.performance_metrics
                }
                for exec in history
            ]
            
        except Exception as e:
            logger.error(f"Error getting execution history: {str(e)}")
            return []
    
    async def get_automation_statistics(self) -> Dict[str, Any]:
        """Get automation statistics"""
        try:
            total_workflows = len(self.automated_workflows)
            enabled_workflows = sum(1 for wf in self.automated_workflows.values() if wf.enabled)
            
            total_executions = self.performance_metrics["total_executions"]
            successful_executions = self.performance_metrics["successful_executions"]
            failed_executions = self.performance_metrics["failed_executions"]
            
            success_rate = (successful_executions / total_executions * 100) if total_executions > 0 else 0
            
            return {
                "total_workflows": total_workflows,
                "enabled_workflows": enabled_workflows,
                "disabled_workflows": total_workflows - enabled_workflows,
                "total_executions": total_executions,
                "successful_executions": successful_executions,
                "failed_executions": failed_executions,
                "success_rate": success_rate,
                "average_execution_time": self.performance_metrics["average_execution_time"],
                "active_executions": len(self.active_executions),
                "trigger_frequency": dict(self.performance_metrics["trigger_frequency"])
            }
            
        except Exception as e:
            logger.error(f"Error getting automation statistics: {str(e)}")
            return {}
    
    async def enable_workflow(self, workflow_id: str) -> bool:
        """Enable an automated workflow"""
        try:
            if workflow_id in self.automated_workflows:
                self.automated_workflows[workflow_id].enabled = True
                await self._setup_workflow_triggers(self.automated_workflows[workflow_id])
                logger.info(f"Enabled automated workflow: {workflow_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error enabling workflow: {str(e)}")
            return False
    
    async def disable_workflow(self, workflow_id: str) -> bool:
        """Disable an automated workflow"""
        try:
            if workflow_id in self.automated_workflows:
                self.automated_workflows[workflow_id].enabled = False
                # Remove triggers
                for trigger in self.automated_workflows[workflow_id].triggers:
                    if trigger.id in self.cron_scheduler:
                        del self.cron_scheduler[trigger.id]
                logger.info(f"Disabled automated workflow: {workflow_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error disabling workflow: {str(e)}")
            return False
    
    async def delete_workflow(self, workflow_id: str) -> bool:
        """Delete an automated workflow"""
        try:
            if workflow_id in self.automated_workflows:
                # Disable first
                await self.disable_workflow(workflow_id)
                # Remove from registry
                del self.automated_workflows[workflow_id]
                logger.info(f"Deleted automated workflow: {workflow_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting workflow: {str(e)}")
            return False

# Global instance
workflow_automation_engine = WorkflowAutomationEngine()

# Example usage
if __name__ == "__main__":
    async def test_workflow_automation():
        print("🤖 Testing Workflow Automation System")
        print("=" * 50)
        
        # Create a sample automated workflow
        triggers = [
            {
                "type": "scheduled",
                "name": "Daily Content Generation",
                "description": "Generate content every day at 9 AM",
                "configuration": {
                    "schedule_type": "daily",
                    "time": "09:00"
                }
            }
        ]
        
        actions = [
            {
                "type": "generate_content",
                "name": "Generate Blog Post",
                "configuration": {
                    "topic": "AI Trends",
                    "content_type": "blog",
                    "length": "medium"
                }
            },
            {
                "type": "send_notification",
                "name": "Notify Team",
                "configuration": {
                    "message": "New content generated",
                    "recipients": ["team@example.com"]
                }
            }
        ]
        
        workflow_id = await workflow_automation_engine.create_automated_workflow(
            name="Daily Content Automation",
            description="Automatically generate and distribute content daily",
            workflow_chain_id="sample-chain-id",
            triggers=triggers,
            actions=actions
        )
        
        print(f"Created automated workflow: {workflow_id}")
        
        # Get statistics
        stats = await workflow_automation_engine.get_automation_statistics()
        print(f"Automation statistics: {stats}")
        
        # Test execution
        execution_id = await workflow_automation_engine.execute_workflow(
            workflow_id=workflow_id,
            trigger_id=triggers[0]["name"],
            execution_data={"test": "data"}
        )
        
        print(f"Executed workflow: {execution_id}")
        
        # Get execution history
        history = await workflow_automation_engine.get_workflow_execution_history(workflow_id)
        print(f"Execution history: {len(history)} executions")
    
    asyncio.run(test_workflow_automation())

