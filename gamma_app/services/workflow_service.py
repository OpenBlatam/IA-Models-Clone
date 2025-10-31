"""
Gamma App - Workflow Service
Advanced workflow automation and orchestration
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from datetime import datetime, timedelta
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue, Empty
import networkx as nx
from pathlib import Path

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
    RETRYING = "retrying"

class TaskType(Enum):
    """Task types"""
    CONTENT_GENERATION = "content_generation"
    DOCUMENT_PROCESSING = "document_processing"
    IMAGE_PROCESSING = "image_processing"
    VIDEO_PROCESSING = "video_processing"
    AUDIO_PROCESSING = "audio_processing"
    EMAIL_SENDING = "email_sending"
    WEBHOOK_CALL = "webhook_call"
    DATABASE_OPERATION = "database_operation"
    API_CALL = "api_call"
    CUSTOM = "custom"

@dataclass
class TaskDefinition:
    """Task definition"""
    id: str
    name: str
    task_type: TaskType
    function: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 3
    retry_delay: int = 5
    timeout: int = 300
    dependencies: List[str] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    parallel: bool = False

@dataclass
class TaskExecution:
    """Task execution"""
    task_id: str
    workflow_id: str
    status: TaskStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    execution_time: float = 0.0

@dataclass
class WorkflowDefinition:
    """Workflow definition"""
    id: str
    name: str
    description: str
    tasks: List[TaskDefinition]
    triggers: List[Dict[str, Any]] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    timeout: int = 3600
    max_parallel_tasks: int = 5
    retry_policy: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkflowExecution:
    """Workflow execution"""
    id: str
    workflow_id: str
    status: WorkflowStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    task_executions: List[TaskExecution] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0

class WorkflowEngine:
    """Advanced workflow engine"""
    
    def __init__(self):
        self.workflows = {}
        self.executions = {}
        self.task_functions = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.running = False
        self.worker_thread = None
        self.task_queue = Queue()
        self.lock = threading.Lock()
    
    def register_task_function(self, name: str, function: Callable):
        """Register a task function"""
        self.task_functions[name] = function
        logger.info(f"Registered task function: {name}")
    
    def create_workflow(self, definition: WorkflowDefinition) -> str:
        """Create a new workflow"""
        try:
            # Validate workflow
            self._validate_workflow(definition)
            
            # Store workflow
            self.workflows[definition.id] = definition
            logger.info(f"Created workflow: {definition.name} ({definition.id})")
            
            return definition.id
            
        except Exception as e:
            logger.error(f"Error creating workflow: {e}")
            raise
    
    def _validate_workflow(self, definition: WorkflowDefinition):
        """Validate workflow definition"""
        # Check for circular dependencies
        graph = nx.DiGraph()
        for task in definition.tasks:
            graph.add_node(task.id)
            for dep in task.dependencies:
                graph.add_edge(dep, task.id)
        
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("Workflow contains circular dependencies")
        
        # Validate task functions
        for task in definition.tasks:
            if task.function not in self.task_functions:
                raise ValueError(f"Task function not found: {task.function}")
    
    async def execute_workflow(
        self,
        workflow_id: str,
        variables: Dict[str, Any] = None
    ) -> str:
        """Execute a workflow"""
        try:
            if workflow_id not in self.workflows:
                raise ValueError(f"Workflow not found: {workflow_id}")
            
            # Create execution
            execution_id = str(uuid.uuid4())
            execution = WorkflowExecution(
                id=execution_id,
                workflow_id=workflow_id,
                status=WorkflowStatus.PENDING,
                start_time=datetime.now(),
                variables=variables or {}
            )
            
            # Store execution
            self.executions[execution_id] = execution
            
            # Start execution
            asyncio.create_task(self._execute_workflow_async(execution))
            
            logger.info(f"Started workflow execution: {execution_id}")
            return execution_id
            
        except Exception as e:
            logger.error(f"Error executing workflow: {e}")
            raise
    
    async def _execute_workflow_async(self, execution: WorkflowExecution):
        """Execute workflow asynchronously"""
        try:
            execution.status = WorkflowStatus.RUNNING
            workflow = self.workflows[execution.workflow_id]
            
            # Create task executions
            task_executions = {}
            for task_def in workflow.tasks:
                task_exec = TaskExecution(
                    task_id=task_def.id,
                    workflow_id=execution.workflow_id,
                    status=TaskStatus.PENDING
                )
                execution.task_executions.append(task_exec)
                task_executions[task_def.id] = task_exec
            
            # Execute tasks
            await self._execute_tasks(execution, workflow, task_executions)
            
            # Check final status
            if all(task.status == TaskStatus.COMPLETED for task in execution.task_executions):
                execution.status = WorkflowStatus.COMPLETED
            elif any(task.status == TaskStatus.FAILED for task in execution.task_executions):
                execution.status = WorkflowStatus.FAILED
            else:
                execution.status = WorkflowStatus.FAILED
            
            execution.end_time = datetime.now()
            execution.execution_time = (execution.end_time - execution.start_time).total_seconds()
            
            logger.info(f"Workflow execution completed: {execution.id} - {execution.status.value}")
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.error = str(e)
            execution.end_time = datetime.now()
            logger.error(f"Error executing workflow: {e}")
    
    async def _execute_tasks(
        self,
        execution: WorkflowExecution,
        workflow: WorkflowDefinition,
        task_executions: Dict[str, TaskExecution]
    ):
        """Execute workflow tasks"""
        try:
            completed_tasks = set()
            running_tasks = set()
            
            while len(completed_tasks) < len(workflow.tasks):
                # Find ready tasks
                ready_tasks = []
                for task_def in workflow.tasks:
                    if (task_def.id not in completed_tasks and 
                        task_def.id not in running_tasks and
                        all(dep in completed_tasks for dep in task_def.dependencies)):
                        ready_tasks.append(task_def)
                
                if not ready_tasks:
                    # Check for failed tasks
                    failed_tasks = [t for t in execution.task_executions if t.status == TaskStatus.FAILED]
                    if failed_tasks:
                        break
                    await asyncio.sleep(0.1)
                    continue
                
                # Execute ready tasks
                if workflow.max_parallel_tasks == 1:
                    # Sequential execution
                    for task_def in ready_tasks:
                        await self._execute_task(execution, task_def, task_executions[task_def.id])
                        completed_tasks.add(task_def.id)
                else:
                    # Parallel execution
                    parallel_tasks = ready_tasks[:workflow.max_parallel_tasks]
                    running_tasks.update(task.id for task in parallel_tasks)
                    
                    # Execute tasks in parallel
                    tasks = [
                        self._execute_task(execution, task_def, task_executions[task_def.id])
                        for task_def in parallel_tasks
                    ]
                    
                    await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Update completed tasks
                    for task_def in parallel_tasks:
                        task_exec = task_executions[task_def.id]
                        if task_exec.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED]:
                            completed_tasks.add(task_def.id)
                            running_tasks.discard(task_def.id)
                
        except Exception as e:
            logger.error(f"Error executing tasks: {e}")
            raise
    
    async def _execute_task(
        self,
        execution: WorkflowExecution,
        task_def: TaskDefinition,
        task_exec: TaskExecution
    ):
        """Execute a single task"""
        try:
            task_exec.status = TaskStatus.RUNNING
            task_exec.start_time = datetime.now()
            
            # Check conditions
            if not self._check_task_conditions(task_def, execution.variables):
                task_exec.status = TaskStatus.SKIPPED
                task_exec.end_time = datetime.now()
                return
            
            # Execute task with retries
            for attempt in range(task_def.retry_count + 1):
                try:
                    # Get task function
                    if task_def.function not in self.task_functions:
                        raise ValueError(f"Task function not found: {task_def.function}")
                    
                    func = self.task_functions[task_def.function]
                    
                    # Prepare parameters
                    params = self._prepare_task_parameters(task_def, execution.variables)
                    
                    # Execute task
                    if asyncio.iscoroutinefunction(func):
                        result = await asyncio.wait_for(func(**params), timeout=task_def.timeout)
                    else:
                        result = await asyncio.get_event_loop().run_in_executor(
                            self.executor, 
                            lambda: func(**params)
                        )
                    
                    # Task completed successfully
                    task_exec.status = TaskStatus.COMPLETED
                    task_exec.result = result
                    task_exec.end_time = datetime.now()
                    task_exec.execution_time = (task_exec.end_time - task_exec.start_time).total_seconds()
                    
                    logger.info(f"Task completed: {task_def.name} ({task_def.id})")
                    return
                    
                except asyncio.TimeoutError:
                    error_msg = f"Task timeout: {task_def.name}"
                    logger.warning(error_msg)
                    if attempt < task_def.retry_count:
                        task_exec.retry_count += 1
                        await asyncio.sleep(task_def.retry_delay)
                        continue
                    else:
                        task_exec.status = TaskStatus.FAILED
                        task_exec.error = error_msg
                        break
                        
                except Exception as e:
                    error_msg = f"Task error: {str(e)}"
                    logger.error(error_msg)
                    if attempt < task_def.retry_count:
                        task_exec.retry_count += 1
                        await asyncio.sleep(task_def.retry_delay)
                        continue
                    else:
                        task_exec.status = TaskStatus.FAILED
                        task_exec.error = error_msg
                        break
            
            task_exec.end_time = datetime.now()
            task_exec.execution_time = (task_exec.end_time - task_exec.start_time).total_seconds()
            
        except Exception as e:
            task_exec.status = TaskStatus.FAILED
            task_exec.error = str(e)
            task_exec.end_time = datetime.now()
            logger.error(f"Error executing task: {e}")
    
    def _check_task_conditions(self, task_def: TaskDefinition, variables: Dict[str, Any]) -> bool:
        """Check if task conditions are met"""
        try:
            if not task_def.conditions:
                return True
            
            for condition, expected_value in task_def.conditions.items():
                if condition not in variables:
                    return False
                
                actual_value = variables[condition]
                if actual_value != expected_value:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking task conditions: {e}")
            return False
    
    def _prepare_task_parameters(self, task_def: TaskDefinition, variables: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare task parameters"""
        try:
            params = task_def.parameters.copy()
            
            # Replace variables in parameters
            for key, value in params.items():
                if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                    var_name = value[2:-1]
                    if var_name in variables:
                        params[key] = variables[var_name]
            
            return params
            
        except Exception as e:
            logger.error(f"Error preparing task parameters: {e}")
            return task_def.parameters
    
    def get_workflow_execution(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get workflow execution by ID"""
        return self.executions.get(execution_id)
    
    def get_workflow_executions(
        self,
        workflow_id: Optional[str] = None,
        status: Optional[WorkflowStatus] = None,
        limit: int = 100
    ) -> List[WorkflowExecution]:
        """Get workflow executions with filters"""
        try:
            executions = list(self.executions.values())
            
            if workflow_id:
                executions = [e for e in executions if e.workflow_id == workflow_id]
            
            if status:
                executions = [e for e in executions if e.status == status]
            
            # Sort by start time (newest first)
            executions.sort(key=lambda x: x.start_time, reverse=True)
            
            return executions[:limit]
            
        except Exception as e:
            logger.error(f"Error getting workflow executions: {e}")
            return []
    
    def cancel_workflow_execution(self, execution_id: str) -> bool:
        """Cancel a workflow execution"""
        try:
            if execution_id not in self.executions:
                return False
            
            execution = self.executions[execution_id]
            if execution.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]:
                return False
            
            execution.status = WorkflowStatus.CANCELLED
            execution.end_time = datetime.now()
            
            # Cancel running tasks
            for task_exec in execution.task_executions:
                if task_exec.status == TaskStatus.RUNNING:
                    task_exec.status = TaskStatus.FAILED
                    task_exec.error = "Workflow cancelled"
                    task_exec.end_time = datetime.now()
            
            logger.info(f"Cancelled workflow execution: {execution_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling workflow execution: {e}")
            return False
    
    def pause_workflow_execution(self, execution_id: str) -> bool:
        """Pause a workflow execution"""
        try:
            if execution_id not in self.executions:
                return False
            
            execution = self.executions[execution_id]
            if execution.status != WorkflowStatus.RUNNING:
                return False
            
            execution.status = WorkflowStatus.PAUSED
            logger.info(f"Paused workflow execution: {execution_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error pausing workflow execution: {e}")
            return False
    
    def resume_workflow_execution(self, execution_id: str) -> bool:
        """Resume a paused workflow execution"""
        try:
            if execution_id not in self.executions:
                return False
            
            execution = self.executions[execution_id]
            if execution.status != WorkflowStatus.PAUSED:
                return False
            
            execution.status = WorkflowStatus.RUNNING
            logger.info(f"Resumed workflow execution: {execution_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error resuming workflow execution: {e}")
            return False
    
    def get_workflow_statistics(self) -> Dict[str, Any]:
        """Get workflow statistics"""
        try:
            executions = list(self.executions.values())
            
            stats = {
                'total_executions': len(executions),
                'completed_executions': len([e for e in executions if e.status == WorkflowStatus.COMPLETED]),
                'failed_executions': len([e for e in executions if e.status == WorkflowStatus.FAILED]),
                'running_executions': len([e for e in executions if e.status == WorkflowStatus.RUNNING]),
                'cancelled_executions': len([e for e in executions if e.status == WorkflowStatus.CANCELLED]),
                'paused_executions': len([e for e in executions if e.status == WorkflowStatus.PAUSED]),
                'average_execution_time': 0.0,
                'total_workflows': len(self.workflows),
                'total_task_functions': len(self.task_functions)
            }
            
            # Calculate average execution time
            completed_executions = [e for e in executions if e.status == WorkflowStatus.COMPLETED and e.execution_time > 0]
            if completed_executions:
                stats['average_execution_time'] = sum(e.execution_time for e in completed_executions) / len(completed_executions)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting workflow statistics: {e}")
            return {}
    
    def export_workflow(self, workflow_id: str, output_path: str) -> bool:
        """Export workflow definition"""
        try:
            if workflow_id not in self.workflows:
                return False
            
            workflow = self.workflows[workflow_id]
            
            # Convert to serializable format
            workflow_data = {
                'id': workflow.id,
                'name': workflow.name,
                'description': workflow.description,
                'tasks': [
                    {
                        'id': task.id,
                        'name': task.name,
                        'task_type': task.task_type.value,
                        'function': task.function,
                        'parameters': task.parameters,
                        'retry_count': task.retry_count,
                        'retry_delay': task.retry_delay,
                        'timeout': task.timeout,
                        'dependencies': task.dependencies,
                        'conditions': task.conditions,
                        'parallel': task.parallel
                    }
                    for task in workflow.tasks
                ],
                'triggers': workflow.triggers,
                'variables': workflow.variables,
                'timeout': workflow.timeout,
                'max_parallel_tasks': workflow.max_parallel_tasks,
                'retry_policy': workflow.retry_policy
            }
            
            # Save to file
            with open(output_path, 'w') as f:
                json.dump(workflow_data, f, indent=2)
            
            logger.info(f"Exported workflow: {workflow_id} to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting workflow: {e}")
            return False
    
    def import_workflow(self, file_path: str) -> str:
        """Import workflow definition"""
        try:
            with open(file_path, 'r') as f:
                workflow_data = json.load(f)
            
            # Create workflow definition
            tasks = []
            for task_data in workflow_data['tasks']:
                task = TaskDefinition(
                    id=task_data['id'],
                    name=task_data['name'],
                    task_type=TaskType(task_data['task_type']),
                    function=task_data['function'],
                    parameters=task_data.get('parameters', {}),
                    retry_count=task_data.get('retry_count', 3),
                    retry_delay=task_data.get('retry_delay', 5),
                    timeout=task_data.get('timeout', 300),
                    dependencies=task_data.get('dependencies', []),
                    conditions=task_data.get('conditions', {}),
                    parallel=task_data.get('parallel', False)
                )
                tasks.append(task)
            
            workflow = WorkflowDefinition(
                id=workflow_data['id'],
                name=workflow_data['name'],
                description=workflow_data['description'],
                tasks=tasks,
                triggers=workflow_data.get('triggers', []),
                variables=workflow_data.get('variables', {}),
                timeout=workflow_data.get('timeout', 3600),
                max_parallel_tasks=workflow_data.get('max_parallel_tasks', 5),
                retry_policy=workflow_data.get('retry_policy', {})
            )
            
            # Create workflow
            workflow_id = self.create_workflow(workflow)
            logger.info(f"Imported workflow: {workflow_id} from {file_path}")
            return workflow_id
            
        except Exception as e:
            logger.error(f"Error importing workflow: {e}")
            raise
    
    def cleanup_old_executions(self, days: int = 30):
        """Cleanup old workflow executions"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            executions_to_remove = []
            
            for execution_id, execution in self.executions.items():
                if execution.start_time < cutoff_date:
                    executions_to_remove.append(execution_id)
            
            for execution_id in executions_to_remove:
                del self.executions[execution_id]
            
            logger.info(f"Cleaned up {len(executions_to_remove)} old workflow executions")
            
        except Exception as e:
            logger.error(f"Error cleaning up old executions: {e}")

# Global workflow engine instance
workflow_engine = WorkflowEngine()

def register_workflow_task(name: str, function: Callable):
    """Register workflow task function"""
    workflow_engine.register_task_function(name, function)

def create_workflow(definition: WorkflowDefinition) -> str:
    """Create workflow using global engine"""
    return workflow_engine.create_workflow(definition)

async def execute_workflow(workflow_id: str, variables: Dict[str, Any] = None) -> str:
    """Execute workflow using global engine"""
    return await workflow_engine.execute_workflow(workflow_id, variables)

def get_workflow_execution(execution_id: str) -> Optional[WorkflowExecution]:
    """Get workflow execution using global engine"""
    return workflow_engine.get_workflow_execution(execution_id)

def get_workflow_executions(workflow_id: str = None, status: WorkflowStatus = None, limit: int = 100) -> List[WorkflowExecution]:
    """Get workflow executions using global engine"""
    return workflow_engine.get_workflow_executions(workflow_id, status, limit)

def cancel_workflow_execution(execution_id: str) -> bool:
    """Cancel workflow execution using global engine"""
    return workflow_engine.cancel_workflow_execution(execution_id)

def get_workflow_statistics() -> Dict[str, Any]:
    """Get workflow statistics using global engine"""
    return workflow_engine.get_workflow_statistics()

def export_workflow(workflow_id: str, output_path: str) -> bool:
    """Export workflow using global engine"""
    return workflow_engine.export_workflow(workflow_id, output_path)

def import_workflow(file_path: str) -> str:
    """Import workflow using global engine"""
    return workflow_engine.import_workflow(file_path)

























