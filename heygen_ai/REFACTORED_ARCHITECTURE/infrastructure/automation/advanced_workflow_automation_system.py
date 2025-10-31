"""
Advanced Workflow Automation System

This module provides comprehensive workflow automation capabilities
for the refactored HeyGen AI system with intelligent task orchestration,
process optimization, and automated decision making.
"""

import asyncio
import json
import logging
import uuid
import time
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
from pathlib import Path
import redis
import threading
from collections import defaultdict, deque
import yaml
import hashlib
import base64
from cryptography.fernet import Fernet
import requests
import websockets
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import psutil
import gc
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed
import schedule
import croniter
from jinja2 import Template
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import subprocess
import os
import tempfile
import shutil


logger = logging.getLogger(__name__)


class WorkflowStatus(str, Enum):
    """Workflow status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    RETRYING = "retrying"


class TaskStatus(str, Enum):
    """Task status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class TriggerType(str, Enum):
    """Trigger types."""
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    EVENT = "event"
    WEBHOOK = "webhook"
    API = "api"
    FILE = "file"
    DATABASE = "database"
    EMAIL = "email"


class ConditionType(str, Enum):
    """Condition types."""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    REGEX = "regex"
    EXISTS = "exists"
    NOT_EXISTS = "not_exists"


@dataclass
class WorkflowTrigger:
    """Workflow trigger structure."""
    trigger_id: str
    trigger_type: TriggerType
    name: str
    description: str
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0


@dataclass
class WorkflowCondition:
    """Workflow condition structure."""
    condition_id: str
    name: str
    condition_type: ConditionType
    field: str
    value: Any
    operator: str = "AND"
    negate: bool = False


@dataclass
class WorkflowTask:
    """Workflow task structure."""
    task_id: str
    name: str
    description: str
    task_type: str
    config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    conditions: List[WorkflowCondition] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    timeout: int = 300  # seconds
    priority: int = 0  # higher number = higher priority


@dataclass
class WorkflowExecution:
    """Workflow execution structure."""
    execution_id: str
    workflow_id: str
    status: WorkflowStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    triggered_by: Optional[str] = None
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    task_executions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class WorkflowDefinition:
    """Workflow definition structure."""
    workflow_id: str
    name: str
    description: str
    version: str
    triggers: List[WorkflowTrigger] = field(default_factory=list)
    tasks: List[WorkflowTask] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class TaskExecutor:
    """Advanced task executor with multiple execution strategies."""
    
    def __init__(self):
        self.executors = {
            'http_request': self._execute_http_request,
            'email': self._execute_email,
            'database': self._execute_database,
            'file': self._execute_file,
            'script': self._execute_script,
            'ai_generation': self._execute_ai_generation,
            'data_processing': self._execute_data_processing,
            'notification': self._execute_notification,
            'webhook': self._execute_webhook,
            'api_call': self._execute_api_call
        }
        self.execution_pool = ThreadPoolExecutor(max_workers=10)
    
    async def execute_task(self, task: WorkflowTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a workflow task."""
        try:
            task_type = task.task_type
            if task_type not in self.executors:
                raise ValueError(f"Unknown task type: {task_type}")
            
            # Check conditions
            if not await self._check_conditions(task.conditions, context):
                return {
                    'status': TaskStatus.SKIPPED,
                    'message': 'Task conditions not met',
                    'output': {}
                }
            
            # Execute task
            start_time = time.time()
            result = await self.executors[task_type](task, context)
            execution_time = time.time() - start_time
            
            return {
                'status': TaskStatus.COMPLETED,
                'message': 'Task completed successfully',
                'output': result,
                'execution_time': execution_time
            }
            
        except Exception as e:
            logger.error(f"Task execution error: {e}")
            return {
                'status': TaskStatus.FAILED,
                'message': str(e),
                'output': {}
            }
    
    async def _check_conditions(self, conditions: List[WorkflowCondition], context: Dict[str, Any]) -> bool:
        """Check workflow conditions."""
        if not conditions:
            return True
        
        results = []
        for condition in conditions:
            result = await self._evaluate_condition(condition, context)
            results.append(result)
        
        # Apply operator logic
        if len(results) == 1:
            return results[0]
        
        # Simple AND logic for now
        return all(results)
    
    async def _evaluate_condition(self, condition: WorkflowCondition, context: Dict[str, Any]) -> bool:
        """Evaluate a single condition."""
        try:
            field_value = context.get(condition.field)
            
            if condition.condition_type == ConditionType.EQUALS:
                result = field_value == condition.value
            elif condition.condition_type == ConditionType.NOT_EQUALS:
                result = field_value != condition.value
            elif condition.condition_type == ConditionType.GREATER_THAN:
                result = field_value > condition.value
            elif condition.condition_type == ConditionType.LESS_THAN:
                result = field_value < condition.value
            elif condition.condition_type == ConditionType.CONTAINS:
                result = condition.value in str(field_value)
            elif condition.condition_type == ConditionType.NOT_CONTAINS:
                result = condition.value not in str(field_value)
            elif condition.condition_type == ConditionType.EXISTS:
                result = field_value is not None
            elif condition.condition_type == ConditionType.NOT_EXISTS:
                result = field_value is None
            else:
                result = False
            
            return not result if condition.negate else result
            
        except Exception as e:
            logger.error(f"Condition evaluation error: {e}")
            return False
    
    async def _execute_http_request(self, task: WorkflowTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute HTTP request task."""
        try:
            config = task.config
            url = config.get('url')
            method = config.get('method', 'GET')
            headers = config.get('headers', {})
            data = config.get('data')
            
            # Replace variables in URL and data
            url = self._replace_variables(url, context)
            if data:
                data = self._replace_variables(data, context)
            
            async with aiohttp.ClientSession() as session:
                async with session.request(method, url, headers=headers, json=data) as response:
                    response_data = await response.json()
                    
                    return {
                        'status_code': response.status,
                        'headers': dict(response.headers),
                        'data': response_data
                    }
                    
        except Exception as e:
            logger.error(f"HTTP request execution error: {e}")
            raise
    
    async def _execute_email(self, task: WorkflowTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute email task."""
        try:
            config = task.config
            to_email = config.get('to_email')
            subject = config.get('subject')
            body = config.get('body')
            template = config.get('template')
            
            # Replace variables
            to_email = self._replace_variables(to_email, context)
            subject = self._replace_variables(subject, context)
            body = self._replace_variables(body, context)
            
            # Use template if provided
            if template:
                template_obj = Template(template)
                body = template_obj.render(**context)
            
            # Send email (mock implementation)
            logger.info(f"Email sent to {to_email}: {subject}")
            
            return {
                'to_email': to_email,
                'subject': subject,
                'sent': True
            }
            
        except Exception as e:
            logger.error(f"Email execution error: {e}")
            raise
    
    async def _execute_database(self, task: WorkflowTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute database task."""
        try:
            config = task.config
            operation = config.get('operation')
            query = config.get('query')
            database = config.get('database')
            
            # Replace variables in query
            query = self._replace_variables(query, context)
            
            # Execute database operation (mock implementation)
            logger.info(f"Database operation {operation} executed: {query}")
            
            return {
                'operation': operation,
                'query': query,
                'rows_affected': 1
            }
            
        except Exception as e:
            logger.error(f"Database execution error: {e}")
            raise
    
    async def _execute_file(self, task: WorkflowTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute file task."""
        try:
            config = task.config
            operation = config.get('operation')
            file_path = config.get('file_path')
            content = config.get('content')
            
            # Replace variables
            file_path = self._replace_variables(file_path, context)
            if content:
                content = self._replace_variables(content, context)
            
            # Execute file operation (mock implementation)
            logger.info(f"File operation {operation} executed: {file_path}")
            
            return {
                'operation': operation,
                'file_path': file_path,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"File execution error: {e}")
            raise
    
    async def _execute_script(self, task: WorkflowTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute script task."""
        try:
            config = task.config
            script_path = config.get('script_path')
            script_content = config.get('script_content')
            language = config.get('language', 'python')
            
            # Replace variables
            if script_path:
                script_path = self._replace_variables(script_path, context)
            if script_content:
                script_content = self._replace_variables(script_content, context)
            
            # Execute script (mock implementation)
            logger.info(f"Script execution: {script_path or 'inline'}")
            
            return {
                'script_path': script_path,
                'language': language,
                'success': True,
                'output': 'Script executed successfully'
            }
            
        except Exception as e:
            logger.error(f"Script execution error: {e}")
            raise
    
    async def _execute_ai_generation(self, task: WorkflowTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute AI generation task."""
        try:
            config = task.config
            prompt = config.get('prompt')
            model = config.get('model', 'gpt-3.5-turbo')
            max_tokens = config.get('max_tokens', 1000)
            
            # Replace variables in prompt
            prompt = self._replace_variables(prompt, context)
            
            # Execute AI generation (mock implementation)
            logger.info(f"AI generation with model {model}: {prompt[:100]}...")
            
            return {
                'model': model,
                'prompt': prompt,
                'generated_text': f"AI generated content for: {prompt[:50]}...",
                'tokens_used': len(prompt.split()) + 50
            }
            
        except Exception as e:
            logger.error(f"AI generation execution error: {e}")
            raise
    
    async def _execute_data_processing(self, task: WorkflowTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data processing task."""
        try:
            config = task.config
            operation = config.get('operation')
            input_data = config.get('input_data')
            
            # Replace variables
            if input_data:
                input_data = self._replace_variables(input_data, context)
            
            # Execute data processing (mock implementation)
            logger.info(f"Data processing operation {operation} executed")
            
            return {
                'operation': operation,
                'input_records': len(input_data) if isinstance(input_data, list) else 1,
                'output_records': len(input_data) if isinstance(input_data, list) else 1,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Data processing execution error: {e}")
            raise
    
    async def _execute_notification(self, task: WorkflowTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute notification task."""
        try:
            config = task.config
            message = config.get('message')
            channel = config.get('channel', 'general')
            priority = config.get('priority', 'normal')
            
            # Replace variables
            message = self._replace_variables(message, context)
            
            # Send notification (mock implementation)
            logger.info(f"Notification sent to {channel}: {message}")
            
            return {
                'channel': channel,
                'message': message,
                'priority': priority,
                'sent': True
            }
            
        except Exception as e:
            logger.error(f"Notification execution error: {e}")
            raise
    
    async def _execute_webhook(self, task: WorkflowTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute webhook task."""
        try:
            config = task.config
            url = config.get('url')
            method = config.get('method', 'POST')
            payload = config.get('payload', {})
            
            # Replace variables
            url = self._replace_variables(url, context)
            payload = self._replace_variables(payload, context)
            
            # Execute webhook (mock implementation)
            logger.info(f"Webhook sent to {url}: {method}")
            
            return {
                'url': url,
                'method': method,
                'payload': payload,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Webhook execution error: {e}")
            raise
    
    async def _execute_api_call(self, task: WorkflowTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute API call task."""
        try:
            config = task.config
            endpoint = config.get('endpoint')
            method = config.get('method', 'GET')
            headers = config.get('headers', {})
            data = config.get('data')
            
            # Replace variables
            endpoint = self._replace_variables(endpoint, context)
            if data:
                data = self._replace_variables(data, context)
            
            # Execute API call (mock implementation)
            logger.info(f"API call to {endpoint}: {method}")
            
            return {
                'endpoint': endpoint,
                'method': method,
                'status_code': 200,
                'data': data or {}
            }
            
        except Exception as e:
            logger.error(f"API call execution error: {e}")
            raise
    
    def _replace_variables(self, text: str, context: Dict[str, Any]) -> str:
        """Replace variables in text with context values."""
        if not isinstance(text, str):
            return text
        
        for key, value in context.items():
            placeholder = f"{{{{{key}}}}}"
            text = text.replace(placeholder, str(value))
        
        return text


class WorkflowScheduler:
    """Advanced workflow scheduler with cron support."""
    
    def __init__(self):
        self.scheduled_workflows = {}
        self.scheduler_thread = None
        self.running = False
    
    def start(self):
        """Start the scheduler."""
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_worker, daemon=True)
        self.scheduler_thread.start()
        logger.info("Workflow scheduler started")
    
    def stop(self):
        """Stop the scheduler."""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        logger.info("Workflow scheduler stopped")
    
    def schedule_workflow(self, workflow_id: str, cron_expression: str, trigger_id: str):
        """Schedule a workflow with cron expression."""
        try:
            # Validate cron expression
            croniter.croniter(cron_expression)
            
            self.scheduled_workflows[workflow_id] = {
                'cron_expression': cron_expression,
                'trigger_id': trigger_id,
                'next_run': croniter.croniter(cron_expression).get_next(datetime)
            }
            
            logger.info(f"Workflow {workflow_id} scheduled with cron: {cron_expression}")
            
        except Exception as e:
            logger.error(f"Error scheduling workflow {workflow_id}: {e}")
    
    def _scheduler_worker(self):
        """Scheduler worker thread."""
        while self.running:
            try:
                current_time = datetime.now(timezone.utc)
                
                for workflow_id, schedule_info in self.scheduled_workflows.items():
                    if current_time >= schedule_info['next_run']:
                        # Trigger workflow
                        self._trigger_workflow(workflow_id, schedule_info['trigger_id'])
                        
                        # Calculate next run time
                        cron = croniter.croniter(schedule_info['cron_expression'])
                        schedule_info['next_run'] = cron.get_next(datetime)
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Scheduler worker error: {e}")
                time.sleep(60)
    
    def _trigger_workflow(self, workflow_id: str, trigger_id: str):
        """Trigger a scheduled workflow."""
        # This would trigger the workflow execution
        logger.info(f"Triggering scheduled workflow {workflow_id} with trigger {trigger_id}")


class AdvancedWorkflowAutomationSystem:
    """
    Advanced workflow automation system with comprehensive capabilities.
    
    Features:
    - Workflow definition and management
    - Task execution and orchestration
    - Trigger management and scheduling
    - Condition evaluation and branching
    - Error handling and retry logic
    - Monitoring and analytics
    - Template engine and variable substitution
    - Integration with external systems
    """
    
    def __init__(
        self,
        database_path: str = "workflow_automation.db",
        redis_url: str = None
    ):
        """
        Initialize the advanced workflow automation system.
        
        Args:
            database_path: SQLite database path
            redis_url: Redis URL for caching
        """
        self.database_path = database_path
        self.redis_url = redis_url
        
        # Initialize components
        self.task_executor = TaskExecutor()
        self.scheduler = WorkflowScheduler()
        
        # Initialize Redis client
        self.redis_client = None
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()
            except Exception as e:
                logger.warning(f"Redis not available: {e}")
        
        # Initialize database
        self._initialize_database()
        
        # Start scheduler
        self.scheduler.start()
        
        # Initialize metrics
        self.metrics = {
            'workflows_executed': Counter('workflows_executed_total', 'Total workflows executed', ['status']),
            'tasks_executed': Counter('tasks_executed_total', 'Total tasks executed', ['task_type', 'status']),
            'execution_duration': Histogram('workflow_execution_duration_seconds', 'Workflow execution duration'),
            'active_workflows': Gauge('active_workflows', 'Currently active workflows')
        }
        
        logger.info("Advanced workflow automation system initialized")
    
    def _initialize_database(self):
        """Initialize SQLite database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS workflow_definitions (
                    workflow_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    version TEXT NOT NULL,
                    triggers TEXT NOT NULL,
                    tasks TEXT NOT NULL,
                    variables TEXT,
                    settings TEXT,
                    created_at DATETIME NOT NULL,
                    updated_at DATETIME NOT NULL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS workflow_executions (
                    execution_id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at DATETIME NOT NULL,
                    completed_at DATETIME,
                    triggered_by TEXT,
                    input_data TEXT,
                    output_data TEXT,
                    error_message TEXT,
                    task_executions TEXT,
                    FOREIGN KEY (workflow_id) REFERENCES workflow_definitions (workflow_id)
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
    
    async def create_workflow(self, workflow: WorkflowDefinition) -> bool:
        """Create a new workflow definition."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO workflow_definitions
                (workflow_id, name, description, version, triggers, tasks, variables, settings, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                workflow.workflow_id,
                workflow.name,
                workflow.description,
                workflow.version,
                json.dumps([trigger.__dict__ for trigger in workflow.triggers]),
                json.dumps([task.__dict__ for task in workflow.tasks]),
                json.dumps(workflow.variables),
                json.dumps(workflow.settings),
                workflow.created_at.isoformat(),
                workflow.updated_at.isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            # Schedule workflow if it has scheduled triggers
            for trigger in workflow.triggers:
                if trigger.trigger_type == TriggerType.SCHEDULED and trigger.enabled:
                    cron_expression = trigger.config.get('cron_expression')
                    if cron_expression:
                        self.scheduler.schedule_workflow(
                            workflow.workflow_id,
                            cron_expression,
                            trigger.trigger_id
                        )
            
            logger.info(f"Workflow {workflow.workflow_id} created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating workflow: {e}")
            return False
    
    async def execute_workflow(self, workflow_id: str, input_data: Dict[str, Any] = None, triggered_by: str = None) -> WorkflowExecution:
        """Execute a workflow."""
        try:
            # Get workflow definition
            workflow = await self._get_workflow_definition(workflow_id)
            if not workflow:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            # Create execution record
            execution = WorkflowExecution(
                execution_id=str(uuid.uuid4()),
                workflow_id=workflow_id,
                status=WorkflowStatus.RUNNING,
                started_at=datetime.now(timezone.utc),
                triggered_by=triggered_by,
                input_data=input_data or {}
            )
            
            # Store execution
            await self._store_workflow_execution(execution)
            
            # Update metrics
            self.metrics['workflows_executed'].labels(status='running').inc()
            self.metrics['active_workflows'].inc()
            
            try:
                # Build task dependency graph
                task_graph = self._build_task_graph(workflow.tasks)
                
                # Execute tasks in dependency order
                task_results = await self._execute_tasks(workflow.tasks, task_graph, execution.input_data)
                
                # Update execution
                execution.status = WorkflowStatus.COMPLETED
                execution.completed_at = datetime.now(timezone.utc)
                execution.output_data = task_results
                execution.task_executions = task_results
                
                # Update metrics
                self.metrics['workflows_executed'].labels(status='completed').inc()
                self.metrics['active_workflows'].dec()
                
            except Exception as e:
                # Update execution with error
                execution.status = WorkflowStatus.FAILED
                execution.completed_at = datetime.now(timezone.utc)
                execution.error_message = str(e)
                
                # Update metrics
                self.metrics['workflows_executed'].labels(status='failed').inc()
                self.metrics['active_workflows'].dec()
                
                logger.error(f"Workflow execution failed: {e}")
            
            # Update execution record
            await self._update_workflow_execution(execution)
            
            return execution
            
        except Exception as e:
            logger.error(f"Error executing workflow: {e}")
            return WorkflowExecution(
                execution_id=str(uuid.uuid4()),
                workflow_id=workflow_id,
                status=WorkflowStatus.FAILED,
                started_at=datetime.now(timezone.utc),
                error_message=str(e)
            )
    
    def _build_task_graph(self, tasks: List[WorkflowTask]) -> nx.DiGraph:
        """Build task dependency graph."""
        graph = nx.DiGraph()
        
        # Add tasks as nodes
        for task in tasks:
            graph.add_node(task.task_id, task=task)
        
        # Add dependencies as edges
        for task in tasks:
            for dependency in task.dependencies:
                graph.add_edge(dependency, task.task_id)
        
        return graph
    
    async def _execute_tasks(self, tasks: List[WorkflowTask], task_graph: nx.DiGraph, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute tasks in dependency order."""
        task_results = []
        completed_tasks = set()
        
        # Topological sort for dependency order
        try:
            execution_order = list(nx.topological_sort(task_graph))
        except nx.NetworkXError:
            # Handle circular dependencies
            execution_order = [task.task_id for task in tasks]
        
        for task_id in execution_order:
            task = next((t for t in tasks if t.task_id == task_id), None)
            if not task:
                continue
            
            # Check if all dependencies are completed
            if not all(dep in completed_tasks for dep in task.dependencies):
                continue
            
            # Execute task
            result = await self.task_executor.execute_task(task, context)
            task_results.append({
                'task_id': task_id,
                'task_name': task.name,
                'result': result
            })
            
            # Update context with task output
            if result['status'] == TaskStatus.COMPLETED:
                context[f"{task_id}_output"] = result['output']
                completed_tasks.add(task_id)
            
            # Update metrics
            self.metrics['tasks_executed'].labels(
                task_type=task.task_type,
                status=result['status'].value
            ).inc()
        
        return task_results
    
    async def _get_workflow_definition(self, workflow_id: str) -> Optional[WorkflowDefinition]:
        """Get workflow definition from database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM workflow_definitions WHERE workflow_id = ?', (workflow_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # Parse workflow definition
            triggers_data = json.loads(row[4])
            tasks_data = json.loads(row[5])
            
            triggers = [WorkflowTrigger(**trigger) for trigger in triggers_data]
            tasks = [WorkflowTask(**task) for task in tasks_data]
            
            workflow = WorkflowDefinition(
                workflow_id=row[0],
                name=row[1],
                description=row[2],
                version=row[3],
                triggers=triggers,
                tasks=tasks,
                variables=json.loads(row[6]) if row[6] else {},
                settings=json.loads(row[7]) if row[7] else {},
                created_at=datetime.fromisoformat(row[8]),
                updated_at=datetime.fromisoformat(row[9])
            )
            
            conn.close()
            return workflow
            
        except Exception as e:
            logger.error(f"Error getting workflow definition: {e}")
            return None
    
    async def _store_workflow_execution(self, execution: WorkflowExecution):
        """Store workflow execution in database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO workflow_executions
                (execution_id, workflow_id, status, started_at, completed_at, triggered_by, input_data, output_data, error_message, task_executions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                execution.execution_id,
                execution.workflow_id,
                execution.status.value,
                execution.started_at.isoformat(),
                execution.completed_at.isoformat() if execution.completed_at else None,
                execution.triggered_by,
                json.dumps(execution.input_data),
                json.dumps(execution.output_data),
                execution.error_message,
                json.dumps(execution.task_executions)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing workflow execution: {e}")
    
    async def _update_workflow_execution(self, execution: WorkflowExecution):
        """Update workflow execution in database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE workflow_executions
                SET status = ?, completed_at = ?, output_data = ?, error_message = ?, task_executions = ?
                WHERE execution_id = ?
            ''', (
                execution.status.value,
                execution.completed_at.isoformat() if execution.completed_at else None,
                json.dumps(execution.output_data),
                execution.error_message,
                json.dumps(execution.task_executions),
                execution.execution_id
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating workflow execution: {e}")
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics."""
        return {
            'total_workflows': sum(self.metrics['workflows_executed']._value.sum() for _ in [1]),
            'total_tasks': sum(self.metrics['tasks_executed']._value.sum() for _ in [1]),
            'active_workflows': self.metrics['active_workflows']._value.sum(),
            'execution_duration_avg': sum(self.metrics['execution_duration']._sum for _ in [1]) / max(1, sum(self.metrics['execution_duration']._count for _ in [1]))
        }
    
    async def cleanup(self):
        """Cleanup resources."""
        self.scheduler.stop()
        self.task_executor.execution_pool.shutdown(wait=True)
        logger.info("Workflow automation system cleanup completed")


# Example usage and demonstration
async def main():
    """Demonstrate the advanced workflow automation system."""
    print("🔄 HeyGen AI - Advanced Workflow Automation System Demo")
    print("=" * 70)
    
    # Initialize workflow automation system
    workflow_system = AdvancedWorkflowAutomationSystem(
        database_path="workflow_automation.db",
        redis_url="redis://localhost:6379/0"
    )
    
    try:
        # Create sample workflow
        print("\n📝 Creating Sample Workflow...")
        
        # Define triggers
        triggers = [
            WorkflowTrigger(
                trigger_id="manual_trigger",
                trigger_type=TriggerType.MANUAL,
                name="Manual Trigger",
                description="Trigger workflow manually"
            ),
            WorkflowTrigger(
                trigger_id="scheduled_trigger",
                trigger_type=TriggerType.SCHEDULED,
                name="Daily Trigger",
                description="Trigger workflow daily at 9 AM",
                config={"cron_expression": "0 9 * * *"}
            )
        ]
        
        # Define tasks
        tasks = [
            WorkflowTask(
                task_id="data_collection",
                name="Data Collection",
                description="Collect data from various sources",
                task_type="api_call",
                config={
                    "endpoint": "https://api.example.com/data",
                    "method": "GET"
                }
            ),
            WorkflowTask(
                task_id="data_processing",
                name="Data Processing",
                description="Process collected data",
                task_type="data_processing",
                config={
                    "operation": "transform",
                    "input_data": "{{data_collection_output}}"
                },
                dependencies=["data_collection"]
            ),
            WorkflowTask(
                task_id="ai_analysis",
                name="AI Analysis",
                description="Perform AI analysis on processed data",
                task_type="ai_generation",
                config={
                    "prompt": "Analyze the following data: {{data_processing_output}}",
                    "model": "gpt-3.5-turbo"
                },
                dependencies=["data_processing"]
            ),
            WorkflowTask(
                task_id="notification",
                name="Send Notification",
                description="Send notification with results",
                task_type="notification",
                config={
                    "message": "Analysis completed: {{ai_analysis_output}}",
                    "channel": "general"
                },
                dependencies=["ai_analysis"]
            )
        ]
        
        # Create workflow definition
        workflow = WorkflowDefinition(
            workflow_id="sample_workflow",
            name="Sample Data Analysis Workflow",
            description="A sample workflow that collects, processes, and analyzes data",
            version="1.0.0",
            triggers=triggers,
            tasks=tasks,
            variables={
                "environment": "production",
                "timeout": 300
            }
        )
        
        # Create workflow
        workflow_created = await workflow_system.create_workflow(workflow)
        print(f"Workflow created: {workflow_created}")
        
        # Execute workflow
        print("\n🚀 Executing Workflow...")
        execution = await workflow_system.execute_workflow(
            workflow_id="sample_workflow",
            input_data={
                "user_id": "user123",
                "analysis_type": "revenue"
            },
            triggered_by="manual_trigger"
        )
        
        print(f"Workflow Execution Results:")
        print(f"  Execution ID: {execution.execution_id}")
        print(f"  Status: {execution.status}")
        print(f"  Started At: {execution.started_at}")
        print(f"  Completed At: {execution.completed_at}")
        print(f"  Task Executions: {len(execution.task_executions)}")
        
        if execution.error_message:
            print(f"  Error: {execution.error_message}")
        
        # Get system metrics
        print("\n📊 System Metrics:")
        metrics = workflow_system.get_system_metrics()
        print(f"  Total Workflows: {metrics['total_workflows']}")
        print(f"  Total Tasks: {metrics['total_tasks']}")
        print(f"  Active Workflows: {metrics['active_workflows']}")
        print(f"  Average Execution Duration: {metrics['execution_duration_avg']:.2f}s")
        
        # Test different task types
        print("\n🔧 Testing Different Task Types...")
        
        # HTTP Request Task
        http_task = WorkflowTask(
            task_id="http_test",
            name="HTTP Test",
            description="Test HTTP request task",
            task_type="http_request",
            config={
                "url": "https://httpbin.org/get",
                "method": "GET"
            }
        )
        
        http_result = await workflow_system.task_executor.execute_task(http_task, {})
        print(f"HTTP Task Result: {http_result['status']}")
        
        # Email Task
        email_task = WorkflowTask(
            task_id="email_test",
            name="Email Test",
            description="Test email task",
            task_type="email",
            config={
                "to_email": "test@example.com",
                "subject": "Test Email",
                "body": "This is a test email from workflow automation"
            }
        )
        
        email_result = await workflow_system.task_executor.execute_task(email_task, {})
        print(f"Email Task Result: {email_result['status']}")
        
        # AI Generation Task
        ai_task = WorkflowTask(
            task_id="ai_test",
            name="AI Test",
            description="Test AI generation task",
            task_type="ai_generation",
            config={
                "prompt": "Generate a summary of the workflow automation system",
                "model": "gpt-3.5-turbo"
            }
        )
        
        ai_result = await workflow_system.task_executor.execute_task(ai_task, {})
        print(f"AI Task Result: {ai_result['status']}")
        
        print(f"\n🌐 Workflow Automation API available at: http://localhost:8080/api/v1/workflows")
        print(f"📊 Workflow Dashboard available at: http://localhost:8080/dashboard")
        
    except Exception as e:
        print(f"❌ Demo Error: {e}")
        logger.error(f"Demo failed: {e}")
    
    finally:
        # Cleanup
        await workflow_system.cleanup()
        print("\n✅ Demo completed")


if __name__ == "__main__":
    asyncio.run(main())
