"""
Gamma App - Workflow Automation Service
Advanced workflow automation with triggers, conditions, and actions
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import uuid
from pathlib import Path
import sqlite3
import redis
from collections import defaultdict
import yaml
import jsonschema
from croniter import croniter

logger = logging.getLogger(__name__)

class TriggerType(Enum):
    """Types of workflow triggers"""
    SCHEDULED = "scheduled"
    EVENT = "event"
    WEBHOOK = "webhook"
    MANUAL = "manual"
    CONDITION = "condition"

class ActionType(Enum):
    """Types of workflow actions"""
    GENERATE_CONTENT = "generate_content"
    SEND_EMAIL = "send_email"
    CREATE_TASK = "create_task"
    UPDATE_DATABASE = "update_database"
    CALL_API = "call_api"
    SEND_NOTIFICATION = "send_notification"
    EXPORT_DATA = "export_data"
    BACKUP_DATA = "backup_data"
    CUSTOM_SCRIPT = "custom_script"

class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class ConditionOperator(Enum):
    """Condition operators"""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    REGEX = "regex"
    IN = "in"
    NOT_IN = "not_in"

@dataclass
class WorkflowTrigger:
    """Workflow trigger definition"""
    trigger_type: TriggerType
    config: Dict[str, Any]
    enabled: bool = True
    last_triggered: Optional[datetime] = None

@dataclass
class WorkflowCondition:
    """Workflow condition definition"""
    field: str
    operator: ConditionOperator
    value: Any
    logical_operator: str = "AND"  # AND, OR

@dataclass
class WorkflowAction:
    """Workflow action definition"""
    action_type: ActionType
    config: Dict[str, Any]
    retry_count: int = 0
    max_retries: int = 3
    timeout: int = 300  # seconds
    depends_on: List[str] = None  # Action IDs this action depends on

@dataclass
class WorkflowExecution:
    """Workflow execution record"""
    execution_id: str
    workflow_id: str
    status: WorkflowStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    execution_data: Dict[str, Any] = None
    action_results: Dict[str, Any] = None

@dataclass
class Workflow:
    """Workflow definition"""
    workflow_id: str
    name: str
    description: str
    triggers: List[WorkflowTrigger]
    conditions: List[WorkflowCondition]
    actions: List[WorkflowAction]
    enabled: bool = True
    created_at: datetime = None
    updated_at: datetime = None
    created_by: str = "system"
    tags: List[str] = None

class WorkflowAutomationService:
    """Advanced workflow automation service"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_path = config.get("database_path", "workflows.db")
        self.redis_client = None
        self.workflows = {}
        self.executions = {}
        self.action_handlers = {}
        self.trigger_handlers = {}
        self.running_executions = {}
        
        # Initialize components
        self._init_database()
        self._init_redis()
        self._register_default_handlers()
    
    def _init_database(self):
        """Initialize workflow database"""
        self.db_path = Path(self.db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create workflows table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS workflows (
                    workflow_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    definition TEXT NOT NULL,
                    enabled BOOLEAN DEFAULT 1,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    created_by TEXT DEFAULT 'system'
                )
            """)
            
            # Create executions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS workflow_executions (
                    execution_id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at DATETIME NOT NULL,
                    completed_at DATETIME,
                    error_message TEXT,
                    execution_data TEXT,
                    action_results TEXT,
                    FOREIGN KEY (workflow_id) REFERENCES workflows (workflow_id)
                )
            """)
            
            # Create triggers table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS workflow_triggers (
                    trigger_id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL,
                    trigger_type TEXT NOT NULL,
                    config TEXT NOT NULL,
                    enabled BOOLEAN DEFAULT 1,
                    last_triggered DATETIME,
                    FOREIGN KEY (workflow_id) REFERENCES workflows (workflow_id)
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_workflows_enabled ON workflows(enabled)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_executions_workflow ON workflow_executions(workflow_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_executions_status ON workflow_executions(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_triggers_workflow ON workflow_triggers(workflow_id)")
            
            conn.commit()
        
        logger.info("Workflow database initialized")
    
    def _init_redis(self):
        """Initialize Redis for caching and pub/sub"""
        try:
            redis_url = self.config.get("redis_url", "redis://localhost:6379")
            self.redis_client = redis.from_url(redis_url)
            self.redis_client.ping()
            logger.info("Redis client initialized")
        except Exception as e:
            logger.warning(f"Redis initialization failed: {e}")
    
    def _register_default_handlers(self):
        """Register default action and trigger handlers"""
        
        # Register action handlers
        self.register_action_handler(ActionType.GENERATE_CONTENT, self._handle_generate_content)
        self.register_action_handler(ActionType.SEND_EMAIL, self._handle_send_email)
        self.register_action_handler(ActionType.CREATE_TASK, self._handle_create_task)
        self.register_action_handler(ActionType.UPDATE_DATABASE, self._handle_update_database)
        self.register_action_handler(ActionType.CALL_API, self._handle_call_api)
        self.register_action_handler(ActionType.SEND_NOTIFICATION, self._handle_send_notification)
        self.register_action_handler(ActionType.EXPORT_DATA, self._handle_export_data)
        self.register_action_handler(ActionType.BACKUP_DATA, self._handle_backup_data)
        self.register_action_handler(ActionType.CUSTOM_SCRIPT, self._handle_custom_script)
        
        # Register trigger handlers
        self.register_trigger_handler(TriggerType.SCHEDULED, self._handle_scheduled_trigger)
        self.register_trigger_handler(TriggerType.EVENT, self._handle_event_trigger)
        self.register_trigger_handler(TriggerType.WEBHOOK, self._handle_webhook_trigger)
        self.register_trigger_handler(TriggerType.MANUAL, self._handle_manual_trigger)
        self.register_trigger_handler(TriggerType.CONDITION, self._handle_condition_trigger)
        
        logger.info("Default handlers registered")
    
    def register_action_handler(self, action_type: ActionType, handler: Callable):
        """Register an action handler"""
        self.action_handlers[action_type] = handler
        logger.info(f"Action handler registered: {action_type.value}")
    
    def register_trigger_handler(self, trigger_type: TriggerType, handler: Callable):
        """Register a trigger handler"""
        self.trigger_handlers[trigger_type] = handler
        logger.info(f"Trigger handler registered: {trigger_type.value}")
    
    async def create_workflow(self, workflow: Workflow) -> str:
        """Create a new workflow"""
        
        if not workflow.workflow_id:
            workflow.workflow_id = str(uuid.uuid4())
        
        workflow.created_at = datetime.now()
        workflow.updated_at = datetime.now()
        
        # Validate workflow
        self._validate_workflow(workflow)
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO workflows 
                (workflow_id, name, description, definition, enabled, created_at, updated_at, created_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                workflow.workflow_id,
                workflow.name,
                workflow.description,
                json.dumps(asdict(workflow)),
                workflow.enabled,
                workflow.created_at.isoformat(),
                workflow.updated_at.isoformat(),
                workflow.created_by
            ))
            
            # Store triggers
            for trigger in workflow.triggers:
                trigger_id = str(uuid.uuid4())
                cursor.execute("""
                    INSERT OR REPLACE INTO workflow_triggers
                    (trigger_id, workflow_id, trigger_type, config, enabled, last_triggered)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    trigger_id,
                    workflow.workflow_id,
                    trigger.trigger_type.value,
                    json.dumps(trigger.config),
                    trigger.enabled,
                    trigger.last_triggered.isoformat() if trigger.last_triggered else None
                ))
            
            conn.commit()
        
        # Cache workflow
        self.workflows[workflow.workflow_id] = workflow
        
        # Start trigger monitoring if enabled
        if workflow.enabled:
            await self._start_trigger_monitoring(workflow)
        
        logger.info(f"Workflow created: {workflow.workflow_id}")
        return workflow.workflow_id
    
    def _validate_workflow(self, workflow: Workflow):
        """Validate workflow definition"""
        
        if not workflow.name:
            raise ValueError("Workflow name is required")
        
        if not workflow.triggers:
            raise ValueError("Workflow must have at least one trigger")
        
        if not workflow.actions:
            raise ValueError("Workflow must have at least one action")
        
        # Validate triggers
        for trigger in workflow.triggers:
            if trigger.trigger_type not in self.trigger_handlers:
                raise ValueError(f"Unsupported trigger type: {trigger.trigger_type}")
        
        # Validate actions
        for action in workflow.actions:
            if action.action_type not in self.action_handlers:
                raise ValueError(f"Unsupported action type: {action.action_type}")
    
    async def _start_trigger_monitoring(self, workflow: Workflow):
        """Start monitoring triggers for a workflow"""
        
        for trigger in workflow.triggers:
            if trigger.enabled:
                handler = self.trigger_handlers.get(trigger.trigger_type)
                if handler:
                    # Start monitoring in background
                    asyncio.create_task(handler(workflow.workflow_id, trigger))
    
    async def execute_workflow(
        self,
        workflow_id: str,
        execution_data: Optional[Dict[str, Any]] = None,
        trigger_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Execute a workflow"""
        
        # Get workflow
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            workflow = await self._load_workflow(workflow_id)
        
        if not workflow:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        if not workflow.enabled:
            raise ValueError(f"Workflow is disabled: {workflow_id}")
        
        # Create execution record
        execution_id = str(uuid.uuid4())
        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_id=workflow_id,
            status=WorkflowStatus.PENDING,
            started_at=datetime.now(),
            execution_data=execution_data or {},
            action_results={}
        )
        
        # Store execution
        self.executions[execution_id] = execution
        self.running_executions[execution_id] = execution
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO workflow_executions
                (execution_id, workflow_id, status, started_at, execution_data, action_results)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                execution_id,
                workflow_id,
                execution.status.value,
                execution.started_at.isoformat(),
                json.dumps(execution.execution_data),
                json.dumps(execution.action_results)
            ))
            conn.commit()
        
        # Execute workflow in background
        asyncio.create_task(self._execute_workflow_async(execution, workflow, trigger_data))
        
        logger.info(f"Workflow execution started: {execution_id}")
        return execution_id
    
    async def _execute_workflow_async(
        self,
        execution: WorkflowExecution,
        workflow: Workflow,
        trigger_data: Optional[Dict[str, Any]]
    ):
        """Execute workflow asynchronously"""
        
        try:
            execution.status = WorkflowStatus.RUNNING
            await self._update_execution_status(execution)
            
            # Check conditions
            if not await self._evaluate_conditions(workflow.conditions, execution.execution_data, trigger_data):
                execution.status = WorkflowStatus.COMPLETED
                execution.completed_at = datetime.now()
                await self._update_execution_status(execution)
                logger.info(f"Workflow conditions not met: {execution.execution_id}")
                return
            
            # Execute actions
            action_results = {}
            for action in workflow.actions:
                try:
                    result = await self._execute_action(action, execution.execution_data, action_results, trigger_data)
                    action_results[action.action_type.value] = result
                    execution.action_results[action.action_type.value] = result
                except Exception as e:
                    logger.error(f"Action execution failed: {action.action_type.value} - {e}")
                    if action.retry_count < action.max_retries:
                        action.retry_count += 1
                        await asyncio.sleep(2 ** action.retry_count)  # Exponential backoff
                        result = await self._execute_action(action, execution.execution_data, action_results, trigger_data)
                        action_results[action.action_type.value] = result
                        execution.action_results[action.action_type.value] = result
                    else:
                        raise e
            
            execution.status = WorkflowStatus.COMPLETED
            execution.completed_at = datetime.now()
            await self._update_execution_status(execution)
            
            logger.info(f"Workflow execution completed: {execution.execution_id}")
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.error_message = str(e)
            execution.completed_at = datetime.now()
            await self._update_execution_status(execution)
            
            logger.error(f"Workflow execution failed: {execution.execution_id} - {e}")
        
        finally:
            # Remove from running executions
            self.running_executions.pop(execution.execution_id, None)
    
    async def _evaluate_conditions(
        self,
        conditions: List[WorkflowCondition],
        execution_data: Dict[str, Any],
        trigger_data: Optional[Dict[str, Any]]
    ) -> bool:
        """Evaluate workflow conditions"""
        
        if not conditions:
            return True
        
        # Combine all data for evaluation
        all_data = {**execution_data, **(trigger_data or {})}
        
        results = []
        for condition in conditions:
            result = self._evaluate_condition(condition, all_data)
            results.append((result, condition.logical_operator))
        
        # Evaluate logical operators
        if not results:
            return True
        
        final_result = results[0][0]
        for i in range(1, len(results)):
            result, operator = results[i]
            if operator == "AND":
                final_result = final_result and result
            elif operator == "OR":
                final_result = final_result or result
        
        return final_result
    
    def _evaluate_condition(self, condition: WorkflowCondition, data: Dict[str, Any]) -> bool:
        """Evaluate a single condition"""
        
        field_value = data.get(condition.field)
        
        if condition.operator == ConditionOperator.EQUALS:
            return field_value == condition.value
        elif condition.operator == ConditionOperator.NOT_EQUALS:
            return field_value != condition.value
        elif condition.operator == ConditionOperator.GREATER_THAN:
            return field_value > condition.value
        elif condition.operator == ConditionOperator.LESS_THAN:
            return field_value < condition.value
        elif condition.operator == ConditionOperator.CONTAINS:
            return condition.value in str(field_value)
        elif condition.operator == ConditionOperator.NOT_CONTAINS:
            return condition.value not in str(field_value)
        elif condition.operator == ConditionOperator.REGEX:
            import re
            return bool(re.search(condition.value, str(field_value)))
        elif condition.operator == ConditionOperator.IN:
            return field_value in condition.value
        elif condition.operator == ConditionOperator.NOT_IN:
            return field_value not in condition.value
        else:
            return False
    
    async def _execute_action(
        self,
        action: WorkflowAction,
        execution_data: Dict[str, Any],
        action_results: Dict[str, Any],
        trigger_data: Optional[Dict[str, Any]]
    ) -> Any:
        """Execute a workflow action"""
        
        handler = self.action_handlers.get(action.action_type)
        if not handler:
            raise ValueError(f"No handler for action type: {action.action_type}")
        
        # Prepare action context
        context = {
            "execution_data": execution_data,
            "action_results": action_results,
            "trigger_data": trigger_data or {},
            "action_config": action.config
        }
        
        # Execute with timeout
        try:
            result = await asyncio.wait_for(
                handler(context),
                timeout=action.timeout
            )
            return result
        except asyncio.TimeoutError:
            raise Exception(f"Action timeout: {action.action_type.value}")
    
    async def _update_execution_status(self, execution: WorkflowExecution):
        """Update execution status in database"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE workflow_executions
                SET status = ?, completed_at = ?, error_message = ?, action_results = ?
                WHERE execution_id = ?
            """, (
                execution.status.value,
                execution.completed_at.isoformat() if execution.completed_at else None,
                execution.error_message,
                json.dumps(execution.action_results),
                execution.execution_id
            ))
            conn.commit()
    
    async def _load_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Load workflow from database"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT definition FROM workflows WHERE workflow_id = ?
            """, (workflow_id,))
            row = cursor.fetchone()
            
            if row:
                workflow_data = json.loads(row[0])
                workflow = Workflow(**workflow_data)
                self.workflows[workflow_id] = workflow
                return workflow
        
        return None
    
    # Action Handlers
    async def _handle_generate_content(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle content generation action"""
        
        config = context["action_config"]
        
        # Import content generator
        from core.content_generator import ContentGenerator
        
        generator = ContentGenerator(self.config)
        
        result = await generator.generate_content(
            content_type=config.get("content_type", "document"),
            topic=config.get("topic", ""),
            description=config.get("description", ""),
            style=config.get("style", "professional"),
            length=config.get("length", "medium")
        )
        
        return {
            "success": True,
            "content_id": result.get("content_id"),
            "content": result.get("content"),
            "metadata": result.get("metadata")
        }
    
    async def _handle_send_email(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle email sending action"""
        
        config = context["action_config"]
        
        # Import email service
        from services.notification_service import NotificationService
        
        notification_service = NotificationService(self.config)
        
        result = await notification_service.send_email(
            to=config.get("to"),
            subject=config.get("subject"),
            body=config.get("body"),
            template=config.get("template"),
            template_variables=config.get("template_variables", {})
        )
        
        return {
            "success": True,
            "message_id": result.get("message_id"),
            "status": result.get("status")
        }
    
    async def _handle_create_task(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle task creation action"""
        
        config = context["action_config"]
        
        # Create task record
        task_id = str(uuid.uuid4())
        task = {
            "task_id": task_id,
            "title": config.get("title"),
            "description": config.get("description"),
            "priority": config.get("priority", "medium"),
            "assignee": config.get("assignee"),
            "due_date": config.get("due_date"),
            "created_at": datetime.now().isoformat(),
            "status": "pending"
        }
        
        # Store task (simplified - would use proper task service)
        return {
            "success": True,
            "task_id": task_id,
            "task": task
        }
    
    async def _handle_update_database(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle database update action"""
        
        config = context["action_config"]
        
        # Execute database operation
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(config.get("query"), config.get("params", []))
            conn.commit()
            
            return {
                "success": True,
                "rows_affected": cursor.rowcount
            }
    
    async def _handle_call_api(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle API call action"""
        
        config = context["action_config"]
        
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            method = config.get("method", "GET").upper()
            url = config.get("url")
            headers = config.get("headers", {})
            data = config.get("data")
            
            async with session.request(method, url, headers=headers, json=data) as response:
                result = await response.json()
                
                return {
                    "success": response.status < 400,
                    "status_code": response.status,
                    "response": result
                }
    
    async def _handle_send_notification(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle notification sending action"""
        
        config = context["action_config"]
        
        # Import notification service
        from services.notification_service import NotificationService
        
        notification_service = NotificationService(self.config)
        
        result = await notification_service.send_notification(
            user_id=config.get("user_id"),
            title=config.get("title"),
            message=config.get("message"),
            notification_type=config.get("type", "info"),
            channels=config.get("channels", ["in_app"])
        )
        
        return {
            "success": True,
            "notification_id": result.get("notification_id"),
            "status": result.get("status")
        }
    
    async def _handle_export_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data export action"""
        
        config = context["action_config"]
        
        # Import export service
        from engines.export_engine import AdvancedExportEngine
        
        export_engine = AdvancedExportEngine(self.config)
        
        result = await export_engine.export_data(
            data=config.get("data"),
            format=config.get("format", "json"),
            filename=config.get("filename"),
            options=config.get("options", {})
        )
        
        return {
            "success": True,
            "file_path": result.get("file_path"),
            "file_size": result.get("file_size")
        }
    
    async def _handle_backup_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data backup action"""
        
        config = context["action_config"]
        
        # Import backup service
        from services.backup_service import BackupService
        
        backup_service = BackupService(self.config)
        
        result = await backup_service.create_backup(
            backup_type=config.get("backup_type", "full"),
            include_files=config.get("include_files", True),
            compression=config.get("compression", True),
            encryption=config.get("encryption", False)
        )
        
        return {
            "success": True,
            "backup_id": result.get("backup_id"),
            "backup_path": result.get("backup_path")
        }
    
    async def _handle_custom_script(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle custom script execution action"""
        
        config = context["action_config"]
        
        script_path = config.get("script_path")
        script_args = config.get("script_args", [])
        
        if not script_path:
            raise ValueError("Script path is required")
        
        # Execute script
        import subprocess
        
        result = subprocess.run(
            [script_path] + script_args,
            capture_output=True,
            text=True,
            timeout=config.get("timeout", 300)
        )
        
        return {
            "success": result.returncode == 0,
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    
    # Trigger Handlers
    async def _handle_scheduled_trigger(self, workflow_id: str, trigger: WorkflowTrigger):
        """Handle scheduled trigger"""
        
        cron_expression = trigger.config.get("cron_expression")
        if not cron_expression:
            logger.error(f"No cron expression for scheduled trigger in workflow {workflow_id}")
            return
        
        cron = croniter(cron_expression)
        
        while True:
            try:
                next_run = cron.get_next(datetime)
                now = datetime.now()
                
                if next_run <= now:
                    # Trigger workflow
                    await self.execute_workflow(workflow_id)
                    trigger.last_triggered = now
                    
                    # Update trigger in database
                    with sqlite3.connect(self.db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute("""
                            UPDATE workflow_triggers
                            SET last_triggered = ?
                            WHERE workflow_id = ? AND trigger_type = ?
                        """, (now.isoformat(), workflow_id, trigger.trigger_type.value))
                        conn.commit()
                
                # Sleep until next check
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Scheduled trigger error for workflow {workflow_id}: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _handle_event_trigger(self, workflow_id: str, trigger: WorkflowTrigger):
        """Handle event trigger"""
        
        event_type = trigger.config.get("event_type")
        if not event_type:
            logger.error(f"No event type for event trigger in workflow {workflow_id}")
            return
        
        # Subscribe to Redis events
        if self.redis_client:
            pubsub = self.redis_client.pubsub()
            pubsub.subscribe(f"events:{event_type}")
            
            async for message in pubsub.listen():
                if message["type"] == "message":
                    try:
                        event_data = json.loads(message["data"])
                        await self.execute_workflow(workflow_id, trigger_data=event_data)
                    except Exception as e:
                        logger.error(f"Event trigger error for workflow {workflow_id}: {e}")
    
    async def _handle_webhook_trigger(self, workflow_id: str, trigger: WorkflowTrigger):
        """Handle webhook trigger"""
        
        # This would be handled by the webhook endpoint
        # The webhook endpoint would call execute_workflow
        pass
    
    async def _handle_manual_trigger(self, workflow_id: str, trigger: WorkflowTrigger):
        """Handle manual trigger"""
        
        # Manual triggers are handled by API endpoints
        pass
    
    async def _handle_condition_trigger(self, workflow_id: str, trigger: WorkflowTrigger):
        """Handle condition trigger"""
        
        # Monitor conditions and trigger when met
        while True:
            try:
                # Check conditions (simplified)
                if await self._check_condition_trigger(trigger):
                    await self.execute_workflow(workflow_id)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Condition trigger error for workflow {workflow_id}: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _check_condition_trigger(self, trigger: WorkflowTrigger) -> bool:
        """Check if condition trigger should fire"""
        
        # Simplified condition checking
        # In a real implementation, this would check various data sources
        return False
    
    async def get_workflow_execution(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get workflow execution by ID"""
        
        return self.executions.get(execution_id)
    
    async def list_workflow_executions(
        self,
        workflow_id: Optional[str] = None,
        status: Optional[WorkflowStatus] = None,
        limit: int = 100
    ) -> List[WorkflowExecution]:
        """List workflow executions"""
        
        query = "SELECT * FROM workflow_executions WHERE 1=1"
        params = []
        
        if workflow_id:
            query += " AND workflow_id = ?"
            params.append(workflow_id)
        
        if status:
            query += " AND status = ?"
            params.append(status.value)
        
        query += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)
        
        executions = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            for row in rows:
                execution = WorkflowExecution(
                    execution_id=row[0],
                    workflow_id=row[1],
                    status=WorkflowStatus(row[2]),
                    started_at=datetime.fromisoformat(row[3]),
                    completed_at=datetime.fromisoformat(row[4]) if row[4] else None,
                    error_message=row[5],
                    execution_data=json.loads(row[6]) if row[6] else {},
                    action_results=json.loads(row[7]) if row[7] else {}
                )
                executions.append(execution)
        
        return executions
    
    async def cancel_workflow_execution(self, execution_id: str) -> bool:
        """Cancel a running workflow execution"""
        
        execution = self.running_executions.get(execution_id)
        if execution:
            execution.status = WorkflowStatus.CANCELLED
            execution.completed_at = datetime.now()
            await self._update_execution_status(execution)
            
            # Remove from running executions
            self.running_executions.pop(execution_id, None)
            
            logger.info(f"Workflow execution cancelled: {execution_id}")
            return True
        
        return False
    
    async def pause_workflow(self, workflow_id: str) -> bool:
        """Pause a workflow"""
        
        workflow = self.workflows.get(workflow_id)
        if workflow:
            workflow.enabled = False
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE workflows SET enabled = 0 WHERE workflow_id = ?
                """, (workflow_id,))
                conn.commit()
            
            logger.info(f"Workflow paused: {workflow_id}")
            return True
        
        return False
    
    async def resume_workflow(self, workflow_id: str) -> bool:
        """Resume a workflow"""
        
        workflow = self.workflows.get(workflow_id)
        if workflow:
            workflow.enabled = True
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE workflows SET enabled = 1 WHERE workflow_id = ?
                """, (workflow_id,))
                conn.commit()
            
            # Restart trigger monitoring
            await self._start_trigger_monitoring(workflow)
            
            logger.info(f"Workflow resumed: {workflow_id}")
            return True
        
        return False
    
    async def delete_workflow(self, workflow_id: str) -> bool:
        """Delete a workflow"""
        
        # Pause workflow first
        await self.pause_workflow(workflow_id)
        
        # Delete from database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM workflow_triggers WHERE workflow_id = ?", (workflow_id,))
            cursor.execute("DELETE FROM workflows WHERE workflow_id = ?", (workflow_id,))
            conn.commit()
        
        # Remove from cache
        self.workflows.pop(workflow_id, None)
        
        logger.info(f"Workflow deleted: {workflow_id}")
        return True
    
    async def get_workflow_statistics(self) -> Dict[str, Any]:
        """Get workflow statistics"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total workflows
            cursor.execute("SELECT COUNT(*) FROM workflows")
            total_workflows = cursor.fetchone()[0]
            
            # Active workflows
            cursor.execute("SELECT COUNT(*) FROM workflows WHERE enabled = 1")
            active_workflows = cursor.fetchone()[0]
            
            # Total executions
            cursor.execute("SELECT COUNT(*) FROM workflow_executions")
            total_executions = cursor.fetchone()[0]
            
            # Successful executions
            cursor.execute("SELECT COUNT(*) FROM workflow_executions WHERE status = 'completed'")
            successful_executions = cursor.fetchone()[0]
            
            # Failed executions
            cursor.execute("SELECT COUNT(*) FROM workflow_executions WHERE status = 'failed'")
            failed_executions = cursor.fetchone()[0]
            
            # Running executions
            running_executions = len(self.running_executions)
        
        return {
            "total_workflows": total_workflows,
            "active_workflows": active_workflows,
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "failed_executions": failed_executions,
            "running_executions": running_executions,
            "success_rate": (successful_executions / max(1, total_executions)) * 100
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        
        # Cancel all running executions
        for execution_id in list(self.running_executions.keys()):
            await self.cancel_workflow_execution(execution_id)
        
        if self.redis_client:
            self.redis_client.close()
        
        logger.info("Workflow automation service cleanup completed")

# Global instance
workflow_automation_service = None

async def get_workflow_automation_service() -> WorkflowAutomationService:
    """Get global workflow automation service instance"""
    global workflow_automation_service
    if not workflow_automation_service:
        config = {
            "database_path": "data/workflows.db",
            "redis_url": "redis://localhost:6379"
        }
        workflow_automation_service = WorkflowAutomationService(config)
    return workflow_automation_service



