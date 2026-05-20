"""
Workflow Engine
Ported from backend/bulk/workflow_automation.py and adapted for Unified AI Model.
"""

import asyncio
import logging
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import httpx

from .llm_service import LLMService
from ..config import get_config

logger = logging.getLogger(__name__)

class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskType(Enum):
    """Task types in workflows."""
    LLM_GENERATION = "llm_generation"
    API_CALL = "api_call"
    FILE_OPERATION = "file_operation"
    NOTIFICATION = "notification"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    PARALLEL = "parallel"

@dataclass
class WorkflowTask:
    """Individual task in a workflow."""
    id: str
    name: str
    task_type: TaskType
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    timeout: int = 300  # 5 minutes
    status: WorkflowStatus = WorkflowStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

@dataclass
class Workflow:
    """Workflow definition."""
    id: str
    name: str
    description: str
    tasks: List[WorkflowTask] = field(default_factory=list)
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_run: Optional[datetime] = None

class WorkflowEngine:
    """Advanced workflow automation engine."""
    
    def __init__(self, llm_service: Optional[LLMService] = None):
        self.config = get_config()
        self.llm_service = llm_service or LLMService()
        self.workflows: Dict[str, Workflow] = {}
        self.running_workflows: Dict[str, Dict[str, Any]] = {}
        self.workflow_history: List[Dict[str, Any]] = []
        
        # Task handlers
        self.task_handlers = {
            TaskType.LLM_GENERATION: self._handle_llm_generation,
            TaskType.API_CALL: self._handle_api_call,
            TaskType.FILE_OPERATION: self._handle_file_operation,
            TaskType.NOTIFICATION: self._handle_notification,
            TaskType.CONDITIONAL: self._handle_conditional,
            TaskType.LOOP: self._handle_loop,
            TaskType.PARALLEL: self._handle_parallel
        }
    
    def create_workflow(self, workflow_id: str, name: str, description: str, 
                       tasks: List[Dict[str, Any]]) -> Workflow:
        """Create a new workflow."""
        workflow_tasks = []
        
        for task_data in tasks:
            task = WorkflowTask(
                id=task_data['id'],
                name=task_data['name'],
                task_type=TaskType(task_data['type']),
                parameters=task_data.get('parameters', {}),
                dependencies=task_data.get('dependencies', []),
                max_retries=task_data.get('max_retries', 3),
                timeout=task_data.get('timeout', 300)
            )
            workflow_tasks.append(task)
        
        workflow = Workflow(
            id=workflow_id,
            name=name,
            description=description,
            tasks=workflow_tasks,
            enabled=True
        )
        
        self.workflows[workflow_id] = workflow
        logger.info(f"Created workflow: {name} ({workflow_id})")
        return workflow
    
    async def execute_workflow(self, workflow_id: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a workflow."""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        
        if workflow_id in self.running_workflows:
            raise ValueError(f"Workflow {workflow_id} is already running")
        
        logger.info(f"Starting workflow: {workflow.name}")
        
        # Initialize workflow execution
        execution_id = f"{workflow_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.running_workflows[workflow_id] = {
            'execution_id': execution_id,
            'started_at': datetime.now(),
            'parameters': parameters or {},
            'status': WorkflowStatus.RUNNING
        }
        
        try:
            # Reset task statuses
            for task in workflow.tasks:
                task.status = WorkflowStatus.PENDING
                task.result = None
                task.error = None
                task.started_at = None
                task.completed_at = None
            
            # Execute tasks in dependency order
            completed_tasks = set()
            failed_tasks = set()
            
            while len(completed_tasks) + len(failed_tasks) < len(workflow.tasks):
                # Find tasks ready to execute
                ready_tasks = []
                for task in workflow.tasks:
                    if (task.status == WorkflowStatus.PENDING and 
                        all(dep in completed_tasks for dep in task.dependencies)):
                        ready_tasks.append(task)
                
                if not ready_tasks:
                    # No ready tasks, check for failed dependencies
                    remaining_tasks = [t for t in workflow.tasks 
                                     if t.status == WorkflowStatus.PENDING]
                    if remaining_tasks:
                        failed_deps = [t for t in remaining_tasks 
                                     if any(dep in failed_tasks for dep in t.dependencies)]
                        if failed_deps:
                            for task in failed_deps:
                                task.status = WorkflowStatus.FAILED
                                task.error = "Dependency failed"
                                failed_tasks.add(task.id)
                            continue
                    
                    # Deadlock or all tasks completed
                    break
                
                # Execute ready tasks (sequentially for now, could be parallelized)
                for task in ready_tasks:
                    try:
                        await self._execute_task(task, parameters or {})
                        if task.status == WorkflowStatus.COMPLETED:
                            completed_tasks.add(task.id)
                        else:
                            failed_tasks.add(task.id)
                    except Exception as e:
                        task.status = WorkflowStatus.FAILED
                        task.error = str(e)
                        failed_tasks.add(task.id)
                        logger.error(f"Task {task.id} failed: {e}")
            
            # Determine overall workflow status
            if len(failed_tasks) == 0:
                workflow_status = WorkflowStatus.COMPLETED
            elif len(completed_tasks) == 0:
                workflow_status = WorkflowStatus.FAILED
            else:
                workflow_status = WorkflowStatus.COMPLETED  # Partial success
            
            # Update workflow
            workflow.last_run = datetime.now()
            
            # Record execution history
            execution_result = {
                'execution_id': execution_id,
                'workflow_id': workflow_id,
                'status': workflow_status.value,
                'started_at': self.running_workflows[workflow_id]['started_at'],
                'completed_at': datetime.now(),
                'completed_tasks': len(completed_tasks),
                'failed_tasks': len(failed_tasks),
                'total_tasks': len(workflow.tasks)
            }
            
            self.workflow_history.append(execution_result)
            
            # Clean up
            del self.running_workflows[workflow_id]
            
            logger.info(f"Workflow completed: {workflow.name} ({workflow_status.value})")
            return execution_result
            
        except Exception as e:
            # Clean up on error
            if workflow_id in self.running_workflows:
                del self.running_workflows[workflow_id]
            
            logger.error(f"Workflow {workflow_id} failed: {e}")
            raise
    
    async def _execute_task(self, task: WorkflowTask, workflow_parameters: Dict[str, Any]):
        """Execute a single task."""
        task.started_at = datetime.now()
        task.status = WorkflowStatus.RUNNING
        
        logger.info(f"Executing task: {task.name}")
        
        try:
            # Merge workflow parameters with task parameters
            task_params = {**workflow_parameters, **task.parameters}
            
            # Get task handler
            handler = self.task_handlers.get(task.task_type)
            if not handler:
                raise ValueError(f"No handler for task type: {task.task_type}")
            
            # Execute task with timeout
            result = await asyncio.wait_for(
                handler(task, task_params),
                timeout=task.timeout
            )
            
            task.result = result
            task.status = WorkflowStatus.COMPLETED
            task.completed_at = datetime.now()
            
            logger.info(f"Task completed: {task.name}")
            
        except asyncio.TimeoutError:
            task.status = WorkflowStatus.FAILED
            task.error = f"Task timeout after {task.timeout} seconds"
            task.completed_at = datetime.now()
            logger.error(f"Task timeout: {task.name}")
            
        except Exception as e:
            task.status = WorkflowStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()
            logger.error(f"Task failed: {task.name} - {e}")
    
    async def _handle_llm_generation(self, task: WorkflowTask, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle LLM generation task."""
        prompt = parameters.get('prompt', '')
        system_prompt = parameters.get('system_prompt', 'You are a helpful assistant.')
        
        result = await self.llm_service.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            model=self.config.default_model
        )
        
        return {
            'content': result.content,
            'usage': result.usage
        }
    
    async def _handle_api_call(self, task: WorkflowTask, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle API call task."""
        url = parameters.get('url', '')
        method = parameters.get('method', 'GET')
        headers = parameters.get('headers', {})
        data = parameters.get('data', {})
        
        async with httpx.AsyncClient() as client:
            if method.upper() == 'GET':
                response = await client.get(url, headers=headers)
            elif method.upper() == 'POST':
                response = await client.post(url, headers=headers, json=data)
            elif method.upper() == 'PUT':
                response = await client.put(url, headers=headers, json=data)
            elif method.upper() == 'DELETE':
                response = await client.delete(url, headers=headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            return {
                'status_code': response.status_code,
                'response': response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
            }
    
    async def _handle_file_operation(self, task: WorkflowTask, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle file operation task."""
        operation = parameters.get('operation', 'read')
        file_path = parameters.get('file_path', '')
        content = parameters.get('content', '')
        
        if operation == 'read':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return {'content': content}
        
        elif operation == 'write':
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return {'bytes_written': len(content.encode('utf-8'))}
        
        else:
            raise ValueError(f"Unknown file operation: {operation}")
    
    async def _handle_notification(self, task: WorkflowTask, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle notification task."""
        message = parameters.get('message', '')
        logger.info(f"WORKFLOW NOTIFICATION: {message}")
        return {'sent': True}
    
    async def _handle_conditional(self, task: WorkflowTask, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle conditional task."""
        condition = parameters.get('condition', '')
        # Simplified condition evaluation logic
        # In a real system, this would be more robust/safe
        result = bool(parameters.get(condition, False))
        return {'result': result}
    
    async def _handle_loop(self, task: WorkflowTask, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle loop task."""
        # Simplified loop implementation
        iterations = parameters.get('iterations', 1)
        return {'iterations': iterations, 'status': 'completed'}
    
    async def _handle_parallel(self, task: WorkflowTask, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle parallel task."""
        # Simplified parallel implementation
        return {'status': 'completed'}
