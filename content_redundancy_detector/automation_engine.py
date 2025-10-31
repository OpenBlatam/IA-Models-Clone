"""
Automation engine for intelligent content processing workflows
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Workflow status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class TaskStatus(Enum):
    """Task status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TriggerType(Enum):
    """Trigger types"""
    SCHEDULED = "scheduled"
    EVENT = "event"
    MANUAL = "manual"
    API = "api"
    WEBHOOK = "webhook"


class ConditionType(Enum):
    """Condition types"""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    REGEX = "regex"
    CUSTOM = "custom"


@dataclass
class WorkflowTask:
    """Workflow task"""
    id: str
    name: str
    task_type: str
    parameters: Dict[str, Any]
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    timeout: int = 300
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Workflow:
    """Workflow definition"""
    id: str
    name: str
    description: str
    tasks: List[WorkflowTask]
    triggers: List[Dict[str, Any]]
    variables: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowExecution:
    """Workflow execution"""
    id: str
    workflow_id: str
    status: WorkflowStatus
    trigger_type: TriggerType
    trigger_data: Dict[str, Any]
    variables: Dict[str, Any]
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    error: Optional[str] = None
    results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AutomationRule:
    """Automation rule"""
    id: str
    name: str
    description: str
    conditions: List[Dict[str, Any]]
    actions: List[Dict[str, Any]]
    is_active: bool = True
    created_at: float = field(default_factory=time.time)
    last_triggered: Optional[float] = None
    trigger_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class AutomationEngine:
    """Automation engine"""
    
    def __init__(self):
        self._workflows: Dict[str, Workflow] = {}
        self._workflow_executions: Dict[str, WorkflowExecution] = {}
        self._automation_rules: Dict[str, AutomationRule] = {}
        self._task_handlers: Dict[str, Callable] = {}
        self._condition_handlers: Dict[str, Callable] = {}
        self._is_running = False
        self._executor = ThreadPoolExecutor(max_workers=10)
        self._scheduler_task: Optional[asyncio.Task] = None
        self._rule_processor_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start automation engine"""
        if self._is_running:
            return
        
        self._is_running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        self._rule_processor_task = asyncio.create_task(self._rule_processor_loop())
        logger.info("Automation engine started")
    
    async def stop(self) -> None:
        """Stop automation engine"""
        self._is_running = False
        
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        
        if self._rule_processor_task:
            self._rule_processor_task.cancel()
            try:
                await self._rule_processor_task
            except asyncio.CancelledError:
                pass
        
        self._executor.shutdown(wait=True)
        logger.info("Automation engine stopped")
    
    def register_task_handler(self, task_type: str, handler: Callable) -> None:
        """Register task handler"""
        self._task_handlers[task_type] = handler
        logger.info(f"Task handler registered: {task_type}")
    
    def register_condition_handler(self, condition_type: str, handler: Callable) -> None:
        """Register condition handler"""
        self._condition_handlers[condition_type] = handler
        logger.info(f"Condition handler registered: {condition_type}")
    
    def create_workflow(self, name: str, description: str, 
                       tasks: List[Dict[str, Any]], 
                       triggers: List[Dict[str, Any]],
                       variables: Optional[Dict[str, Any]] = None) -> Workflow:
        """Create workflow"""
        workflow_id = str(uuid.uuid4())
        
        # Create workflow tasks
        workflow_tasks = []
        for task_data in tasks:
            task = WorkflowTask(
                id=str(uuid.uuid4()),
                name=task_data["name"],
                task_type=task_data["task_type"],
                parameters=task_data.get("parameters", {}),
                conditions=task_data.get("conditions", []),
                max_retries=task_data.get("max_retries", 3),
                timeout=task_data.get("timeout", 300)
            )
            workflow_tasks.append(task)
        
        workflow = Workflow(
            id=workflow_id,
            name=name,
            description=description,
            tasks=workflow_tasks,
            triggers=triggers,
            variables=variables or {}
        )
        
        self._workflows[workflow_id] = workflow
        logger.info(f"Workflow created: {name} ({workflow_id})")
        
        return workflow
    
    async def execute_workflow(self, workflow_id: str, trigger_type: TriggerType,
                              trigger_data: Optional[Dict[str, Any]] = None,
                              variables: Optional[Dict[str, Any]] = None) -> WorkflowExecution:
        """Execute workflow"""
        if workflow_id not in self._workflows:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        workflow = self._workflows[workflow_id]
        if not workflow.is_active:
            raise ValueError(f"Workflow is not active: {workflow_id}")
        
        # Create execution
        execution_id = str(uuid.uuid4())
        execution = WorkflowExecution(
            id=execution_id,
            workflow_id=workflow_id,
            status=WorkflowStatus.RUNNING,
            trigger_type=trigger_type,
            trigger_data=trigger_data or {},
            variables={**workflow.variables, **(variables or {})}
        )
        
        self._workflow_executions[execution_id] = execution
        
        # Execute workflow in background
        asyncio.create_task(self._execute_workflow_tasks(execution))
        
        logger.info(f"Workflow execution started: {workflow.name} ({execution_id})")
        return execution
    
    async def _execute_workflow_tasks(self, execution: WorkflowExecution) -> None:
        """Execute workflow tasks"""
        try:
            workflow = self._workflows[execution.workflow_id]
            
            for task in workflow.tasks:
                if execution.status != WorkflowStatus.RUNNING:
                    break
                
                # Check task conditions
                if not await self._evaluate_task_conditions(task, execution):
                    task.status = TaskStatus.SKIPPED
                    logger.info(f"Task skipped due to conditions: {task.name}")
                    continue
                
                # Execute task
                await self._execute_task(task, execution)
                
                # Update execution results
                execution.results[task.id] = {
                    "status": task.status.value,
                    "result": task.result,
                    "error": task.error,
                    "started_at": task.started_at,
                    "completed_at": task.completed_at
                }
            
            # Update execution status
            if execution.status == WorkflowStatus.RUNNING:
                execution.status = WorkflowStatus.COMPLETED
                execution.completed_at = time.time()
                logger.info(f"Workflow execution completed: {execution.id}")
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.error = str(e)
            execution.completed_at = time.time()
            logger.error(f"Workflow execution failed: {execution.id} - {e}")
    
    async def _execute_task(self, task: WorkflowTask, execution: WorkflowExecution) -> None:
        """Execute individual task"""
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        
        try:
            # Get task handler
            if task.task_type not in self._task_handlers:
                raise ValueError(f"Task handler not found: {task.task_type}")
            
            handler = self._task_handlers[task.task_type]
            
            # Prepare task context
            context = {
                "task": task,
                "execution": execution,
                "variables": execution.variables,
                "trigger_data": execution.trigger_data
            }
            
            # Execute task
            if asyncio.iscoroutinefunction(handler):
                result = await asyncio.wait_for(
                    handler(task.parameters, context), 
                    timeout=task.timeout
                )
            else:
                # Run in thread pool for sync handlers
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(self._executor, handler, task.parameters, context),
                    timeout=task.timeout
                )
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            
            logger.info(f"Task completed: {task.name}")
            
        except asyncio.TimeoutError:
            task.error = f"Task timeout after {task.timeout} seconds"
            task.status = TaskStatus.FAILED
            task.completed_at = time.time()
            logger.error(f"Task timeout: {task.name}")
            
        except Exception as e:
            task.error = str(e)
            task.status = TaskStatus.FAILED
            task.completed_at = time.time()
            logger.error(f"Task failed: {task.name} - {e}")
            
            # Retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.PENDING
                logger.info(f"Retrying task: {task.name} (attempt {task.retry_count + 1})")
                await asyncio.sleep(2 ** task.retry_count)  # Exponential backoff
                await self._execute_task(task, execution)
    
    async def _evaluate_task_conditions(self, task: WorkflowTask, execution: WorkflowExecution) -> bool:
        """Evaluate task conditions"""
        if not task.conditions:
            return True
        
        for condition in task.conditions:
            condition_type = condition.get("type")
            field = condition.get("field")
            value = condition.get("value")
            
            # Get field value from context
            field_value = self._get_field_value(field, execution)
            
            # Evaluate condition
            if condition_type in self._condition_handlers:
                handler = self._condition_handlers[condition_type]
                if not handler(field_value, value):
                    return False
            else:
                # Default condition evaluation
                if not self._evaluate_default_condition(condition_type, field_value, value):
                    return False
        
        return True
    
    def _get_field_value(self, field: str, execution: WorkflowExecution) -> Any:
        """Get field value from execution context"""
        if field.startswith("variables."):
            var_name = field[10:]  # Remove "variables." prefix
            return execution.variables.get(var_name)
        elif field.startswith("trigger_data."):
            data_name = field[13:]  # Remove "trigger_data." prefix
            return execution.trigger_data.get(data_name)
        elif field.startswith("results."):
            result_name = field[8:]  # Remove "results." prefix
            return execution.results.get(result_name)
        else:
            return getattr(execution, field, None)
    
    def _evaluate_default_condition(self, condition_type: str, field_value: Any, expected_value: Any) -> bool:
        """Evaluate default condition"""
        if condition_type == ConditionType.EQUALS.value:
            return field_value == expected_value
        elif condition_type == ConditionType.NOT_EQUALS.value:
            return field_value != expected_value
        elif condition_type == ConditionType.GREATER_THAN.value:
            return field_value > expected_value
        elif condition_type == ConditionType.LESS_THAN.value:
            return field_value < expected_value
        elif condition_type == ConditionType.CONTAINS.value:
            return expected_value in str(field_value)
        elif condition_type == ConditionType.NOT_CONTAINS.value:
            return expected_value not in str(field_value)
        else:
            return True
    
    def create_automation_rule(self, name: str, description: str,
                              conditions: List[Dict[str, Any]],
                              actions: List[Dict[str, Any]]) -> AutomationRule:
        """Create automation rule"""
        rule_id = str(uuid.uuid4())
        
        rule = AutomationRule(
            id=rule_id,
            name=name,
            description=description,
            conditions=conditions,
            actions=actions
        )
        
        self._automation_rules[rule_id] = rule
        logger.info(f"Automation rule created: {name} ({rule_id})")
        
        return rule
    
    async def process_event(self, event_type: str, event_data: Dict[str, Any]) -> List[str]:
        """Process event and trigger automation rules"""
        triggered_rules = []
        
        for rule in self._automation_rules.values():
            if not rule.is_active:
                continue
            
            # Check if rule should be triggered
            if await self._evaluate_rule_conditions(rule, event_type, event_data):
                # Execute rule actions
                await self._execute_rule_actions(rule, event_data)
                
                # Update rule stats
                rule.last_triggered = time.time()
                rule.trigger_count += 1
                
                triggered_rules.append(rule.id)
                logger.info(f"Automation rule triggered: {rule.name}")
        
        return triggered_rules
    
    async def _evaluate_rule_conditions(self, rule: AutomationRule, 
                                       event_type: str, event_data: Dict[str, Any]) -> bool:
        """Evaluate rule conditions"""
        for condition in rule.conditions:
            condition_type = condition.get("type")
            field = condition.get("field")
            value = condition.get("value")
            
            # Get field value
            field_value = event_data.get(field)
            
            # Evaluate condition
            if condition_type in self._condition_handlers:
                handler = self._condition_handlers[condition_type]
                if not handler(field_value, value):
                    return False
            else:
                if not self._evaluate_default_condition(condition_type, field_value, value):
                    return False
        
        return True
    
    async def _execute_rule_actions(self, rule: AutomationRule, event_data: Dict[str, Any]) -> None:
        """Execute rule actions"""
        for action in rule.actions:
            action_type = action.get("type")
            parameters = action.get("parameters", {})
            
            try:
                if action_type == "execute_workflow":
                    workflow_id = parameters.get("workflow_id")
                    if workflow_id in self._workflows:
                        await self.execute_workflow(
                            workflow_id, 
                            TriggerType.EVENT,
                            event_data
                        )
                
                elif action_type == "send_notification":
                    # Simulate notification sending
                    logger.info(f"Notification sent: {parameters.get('message', 'Automation triggered')}")
                
                elif action_type == "update_variable":
                    # Update workflow variables
                    var_name = parameters.get("variable_name")
                    var_value = parameters.get("variable_value")
                    if var_name:
                        # This would update global variables or specific workflow variables
                        logger.info(f"Variable updated: {var_name} = {var_value}")
                
                elif action_type == "custom_action":
                    # Execute custom action
                    action_handler = parameters.get("handler")
                    if action_handler and callable(action_handler):
                        await action_handler(event_data, parameters)
                
            except Exception as e:
                logger.error(f"Error executing rule action: {e}")
    
    async def _scheduler_loop(self) -> None:
        """Scheduler loop for scheduled workflows"""
        while self._is_running:
            try:
                current_time = time.time()
                
                for workflow in self._workflows.values():
                    if not workflow.is_active:
                        continue
                    
                    for trigger in workflow.triggers:
                        if trigger.get("type") == "scheduled":
                            schedule = trigger.get("schedule")
                            if self._should_trigger_scheduled(workflow.id, schedule, current_time):
                                await self.execute_workflow(
                                    workflow.id,
                                    TriggerType.SCHEDULED,
                                    {"schedule": schedule}
                                )
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(5)
    
    async def _rule_processor_loop(self) -> None:
        """Rule processor loop"""
        while self._is_running:
            try:
                # Process any pending rule evaluations
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in rule processor loop: {e}")
                await asyncio.sleep(5)
    
    def _should_trigger_scheduled(self, workflow_id: str, schedule: Dict[str, Any], current_time: float) -> bool:
        """Check if scheduled workflow should trigger"""
        # Simple schedule checking - in production, use proper cron parsing
        schedule_type = schedule.get("type")
        
        if schedule_type == "interval":
            interval = schedule.get("interval", 3600)  # Default 1 hour
            last_execution = schedule.get("last_execution", 0)
            return current_time - last_execution >= interval
        
        elif schedule_type == "cron":
            # Simplified cron checking
            cron_expression = schedule.get("expression", "0 * * * *")
            # In production, use proper cron library
            return True  # Simplified
        
        return False
    
    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Get workflow by ID"""
        return self._workflows.get(workflow_id)
    
    def get_workflow_execution(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get workflow execution by ID"""
        return self._workflow_executions.get(execution_id)
    
    def get_automation_rule(self, rule_id: str) -> Optional[AutomationRule]:
        """Get automation rule by ID"""
        return self._automation_rules.get(rule_id)
    
    def get_automation_stats(self) -> Dict[str, Any]:
        """Get automation statistics"""
        return {
            "workflows": len(self._workflows),
            "active_workflows": len([w for w in self._workflows.values() if w.is_active]),
            "workflow_executions": len(self._workflow_executions),
            "automation_rules": len(self._automation_rules),
            "active_rules": len([r for r in self._automation_rules.values() if r.is_active]),
            "task_handlers": len(self._task_handlers),
            "condition_handlers": len(self._condition_handlers),
            "is_running": self._is_running
        }


# Global automation engine
automation_engine = AutomationEngine()


# Default task handlers
def content_analysis_task(parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Content analysis task handler"""
    content = parameters.get("content", "")
    
    # Simulate content analysis
    word_count = len(content.split())
    char_count = len(content)
    
    return {
        "word_count": word_count,
        "character_count": char_count,
        "analysis_completed": True
    }


def similarity_check_task(parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Similarity check task handler"""
    text1 = parameters.get("text1", "")
    text2 = parameters.get("text2", "")
    threshold = parameters.get("threshold", 0.5)
    
    # Simulate similarity check
    similarity_score = 0.7  # Simplified
    
    return {
        "similarity_score": similarity_score,
        "is_similar": similarity_score > threshold,
        "similarity_check_completed": True
    }


def quality_assessment_task(parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Quality assessment task handler"""
    content = parameters.get("content", "")
    
    # Simulate quality assessment
    quality_score = 0.8  # Simplified
    
    return {
        "quality_score": quality_score,
        "quality_rating": "good" if quality_score > 0.7 else "poor",
        "quality_assessment_completed": True
    }


def notification_task(parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Notification task handler"""
    message = parameters.get("message", "Workflow notification")
    recipient = parameters.get("recipient", "admin")
    
    # Simulate notification sending
    logger.info(f"Notification sent to {recipient}: {message}")
    
    return {
        "notification_sent": True,
        "recipient": recipient,
        "message": message
    }


def data_export_task(parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Data export task handler"""
    data = parameters.get("data", {})
    format_type = parameters.get("format", "json")
    filename = parameters.get("filename", f"export_{int(time.time())}.{format_type}")
    
    # Simulate data export
    logger.info(f"Data exported to {filename} in {format_type} format")
    
    return {
        "export_completed": True,
        "filename": filename,
        "format": format_type,
        "data_size": len(str(data))
    }


# Register default task handlers
automation_engine.register_task_handler("content_analysis", content_analysis_task)
automation_engine.register_task_handler("similarity_check", similarity_check_task)
automation_engine.register_task_handler("quality_assessment", quality_assessment_task)
automation_engine.register_task_handler("notification", notification_task)
automation_engine.register_task_handler("data_export", data_export_task)

# Default condition handlers
def equals_condition(field_value: Any, expected_value: Any) -> bool:
    """Equals condition handler"""
    return field_value == expected_value


def greater_than_condition(field_value: Any, expected_value: Any) -> bool:
    """Greater than condition handler"""
    try:
        return float(field_value) > float(expected_value)
    except (ValueError, TypeError):
        return False


def contains_condition(field_value: Any, expected_value: Any) -> bool:
    """Contains condition handler"""
    return str(expected_value) in str(field_value)


# Register default condition handlers
automation_engine.register_condition_handler("equals", equals_condition)
automation_engine.register_condition_handler("greater_than", greater_than_condition)
automation_engine.register_condition_handler("contains", contains_condition)


