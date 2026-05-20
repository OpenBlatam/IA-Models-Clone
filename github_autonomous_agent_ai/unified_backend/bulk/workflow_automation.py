"""
BUL Workflow Automation
=======================

Advanced workflow automation for the BUL system.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import yaml
import schedule
import threading
import time

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from modules.api_handler import APIHandler
from modules.document_processor import DocumentProcessor
from modules.query_analyzer import QueryAnalyzer
from modules.business_agents import BusinessAgentManager
from config_optimized import BULConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
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
    DOCUMENT_GENERATION = "document_generation"
    DATA_PROCESSING = "data_processing"
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
    error: str = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

@dataclass
class Workflow:
    """Workflow definition."""
    id: str
    name: str
    description: str
    tasks: List[WorkflowTask] = field(default_factory=list)
    schedule: Optional[str] = None
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None

class WorkflowAutomation:
    """Advanced workflow automation system."""
    
    def __init__(self, config: Optional[BULConfig] = None):
        self.config = config or BULConfig()
        self.workflows = {}
        self.running_workflows = {}
        self.workflow_history = []
        self.scheduler_thread = None
        self.scheduler_running = False
        
        # Initialize BUL components
        self.processor = DocumentProcessor(self.config.to_dict())
        self.analyzer = QueryAnalyzer()
        self.agent_manager = BusinessAgentManager(self.config.to_dict())
        self.api_handler = APIHandler(
            self.processor, self.analyzer, self.agent_manager
        )
        
        # Task handlers
        self.task_handlers = {
            TaskType.DOCUMENT_GENERATION: self._handle_document_generation,
            TaskType.DATA_PROCESSING: self._handle_data_processing,
            TaskType.API_CALL: self._handle_api_call,
            TaskType.FILE_OPERATION: self._handle_file_operation,
            TaskType.NOTIFICATION: self._handle_notification,
            TaskType.CONDITIONAL: self._handle_conditional,
            TaskType.LOOP: self._handle_loop,
            TaskType.PARALLEL: self._handle_parallel
        }
    
    def create_workflow(self, workflow_id: str, name: str, description: str, 
                       tasks: List[Dict[str, Any]], schedule: str = None) -> Workflow:
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
            schedule=schedule,
            enabled=True
        )
        
        self.workflows[workflow_id] = workflow
        
        if schedule:
            self._schedule_workflow(workflow)
        
        print(f"✅ Created workflow: {name} ({workflow_id})")
        return workflow
    
    def _schedule_workflow(self, workflow: Workflow):
        """Schedule a workflow for execution."""
        if not workflow.schedule:
            return
        
        try:
            # Parse schedule and add to scheduler
            if workflow.schedule.startswith('every '):
                # Handle "every X minutes/hours/days"
                parts = workflow.schedule.split()
                if len(parts) >= 3:
                    interval = int(parts[1])
                    unit = parts[2]
                    
                    if unit == 'minutes':
                        schedule.every(interval).minutes.do(self._run_scheduled_workflow, workflow.id)
                    elif unit == 'hours':
                        schedule.every(interval).hours.do(self._run_scheduled_workflow, workflow.id)
                    elif unit == 'days':
                        schedule.every(interval).days.do(self._run_scheduled_workflow, workflow.id)
            
            elif workflow.schedule.startswith('daily at '):
                # Handle "daily at HH:MM"
                time_str = workflow.schedule.replace('daily at ', '')
                schedule.every().day.at(time_str).do(self._run_scheduled_workflow, workflow.id)
            
            elif workflow.schedule.startswith('weekly on '):
                # Handle "weekly on Monday at HH:MM"
                parts = workflow.schedule.replace('weekly on ', '').split(' at ')
                if len(parts) == 2:
                    day = parts[0].lower()
                    time_str = parts[1]
                    
                    day_map = {
                        'monday': schedule.every().monday,
                        'tuesday': schedule.every().tuesday,
                        'wednesday': schedule.every().wednesday,
                        'thursday': schedule.every().thursday,
                        'friday': schedule.every().friday,
                        'saturday': schedule.every().saturday,
                        'sunday': schedule.every().sunday
                    }
                    
                    if day in day_map:
                        day_map[day].at(time_str).do(self._run_scheduled_workflow, workflow.id)
            
            print(f"📅 Scheduled workflow: {workflow.name} ({workflow.schedule})")
            
        except Exception as e:
            logger.error(f"Error scheduling workflow {workflow.id}: {e}")
    
    def _run_scheduled_workflow(self, workflow_id: str):
        """Run a scheduled workflow."""
        if workflow_id in self.workflows:
            workflow = self.workflows[workflow_id]
            if workflow.enabled:
                print(f"⏰ Running scheduled workflow: {workflow.name}")
                asyncio.create_task(self.execute_workflow(workflow_id))
    
    async def execute_workflow(self, workflow_id: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a workflow."""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        
        if workflow_id in self.running_workflows:
            raise ValueError(f"Workflow {workflow_id} is already running")
        
        print(f"🚀 Starting workflow: {workflow.name}")
        
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
                
                # Execute ready tasks
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
            if workflow.schedule:
                workflow.next_run = self._calculate_next_run(workflow.schedule)
            
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
            
            print(f"✅ Workflow completed: {workflow.name} ({workflow_status.value})")
            
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
        
        print(f"  🔄 Executing task: {task.name}")
        
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
            
            print(f"  ✅ Task completed: {task.name}")
            
        except asyncio.TimeoutError:
            task.status = WorkflowStatus.FAILED
            task.error = f"Task timeout after {task.timeout} seconds"
            task.completed_at = datetime.now()
            print(f"  ⏰ Task timeout: {task.name}")
            
        except Exception as e:
            task.status = WorkflowStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()
            print(f"  ❌ Task failed: {task.name} - {e}")
    
    async def _handle_document_generation(self, task: WorkflowTask, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle document generation task."""
        query = parameters.get('query', '')
        business_area = parameters.get('business_area', 'general')
        document_type = parameters.get('document_type', 'document')
        
        # Generate document
        result = await self.api_handler.generate_document(
            query=query,
            business_area=business_area,
            document_type=document_type,
            priority=parameters.get('priority', 1)
        )
        
        return {
            'task_type': 'document_generation',
            'result': result,
            'query': query,
            'business_area': business_area,
            'document_type': document_type
        }
    
    async def _handle_data_processing(self, task: WorkflowTask, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data processing task."""
        operation = parameters.get('operation', 'analyze')
        data = parameters.get('data', {})
        
        if operation == 'analyze':
            # Analyze data using query analyzer
            analysis = self.analyzer.analyze_query(data.get('query', ''))
            return {
                'task_type': 'data_processing',
                'operation': operation,
                'result': analysis
            }
        
        elif operation == 'transform':
            # Transform data
            transformed_data = self._transform_data(data)
            return {
                'task_type': 'data_processing',
                'operation': operation,
                'result': transformed_data
            }
        
        else:
            raise ValueError(f"Unknown data processing operation: {operation}")
    
    async def _handle_api_call(self, task: WorkflowTask, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle API call task."""
        url = parameters.get('url', '')
        method = parameters.get('method', 'GET')
        headers = parameters.get('headers', {})
        data = parameters.get('data', {})
        
        import httpx
        
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
                'task_type': 'api_call',
                'url': url,
                'method': method,
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
            return {
                'task_type': 'file_operation',
                'operation': operation,
                'file_path': file_path,
                'content': content
            }
        
        elif operation == 'write':
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return {
                'task_type': 'file_operation',
                'operation': operation,
                'file_path': file_path,
                'bytes_written': len(content.encode('utf-8'))
            }
        
        elif operation == 'copy':
            import shutil
            dest_path = parameters.get('dest_path', '')
            shutil.copy2(file_path, dest_path)
            return {
                'task_type': 'file_operation',
                'operation': operation,
                'source_path': file_path,
                'dest_path': dest_path
            }
        
        else:
            raise ValueError(f"Unknown file operation: {operation}")
    
    async def _handle_notification(self, task: WorkflowTask, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle notification task."""
        notification_type = parameters.get('type', 'log')
        message = parameters.get('message', '')
        recipient = parameters.get('recipient', '')
        
        if notification_type == 'log':
            logger.info(f"Workflow notification: {message}")
            return {
                'task_type': 'notification',
                'type': notification_type,
                'message': message,
                'sent': True
            }
        
        elif notification_type == 'email':
            # Placeholder for email notification
            print(f"📧 Email notification to {recipient}: {message}")
            return {
                'task_type': 'notification',
                'type': notification_type,
                'message': message,
                'recipient': recipient,
                'sent': True
            }
        
        else:
            raise ValueError(f"Unknown notification type: {notification_type}")
    
    async def _handle_conditional(self, task: WorkflowTask, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle conditional task."""
        condition = parameters.get('condition', '')
        true_action = parameters.get('true_action', {})
        false_action = parameters.get('false_action', {})
        
        # Evaluate condition (simplified)
        condition_result = self._evaluate_condition(condition, parameters)
        
        if condition_result:
            action = true_action
        else:
            action = false_action
        
        return {
            'task_type': 'conditional',
            'condition': condition,
            'condition_result': condition_result,
            'action_taken': action
        }
    
    async def _handle_loop(self, task: WorkflowTask, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle loop task."""
        loop_type = parameters.get('loop_type', 'for')
        iterations = parameters.get('iterations', 1)
        loop_tasks = parameters.get('tasks', [])
        
        results = []
        
        if loop_type == 'for':
            for i in range(iterations):
                iteration_result = {
                    'iteration': i + 1,
                    'results': []
                }
                
                for loop_task_data in loop_tasks:
                    loop_task = WorkflowTask(
                        id=f"{task.id}_loop_{i}_{loop_task_data['id']}",
                        name=loop_task_data['name'],
                        task_type=TaskType(loop_task_data['type']),
                        parameters=loop_task_data.get('parameters', {})
                    )
                    
                    try:
                        result = await self._execute_task(loop_task, parameters)
                        iteration_result['results'].append(result)
                    except Exception as e:
                        iteration_result['results'].append({
                            'error': str(e),
                            'status': 'failed'
                        })
                
                results.append(iteration_result)
        
        return {
            'task_type': 'loop',
            'loop_type': loop_type,
            'iterations': iterations,
            'results': results
        }
    
    async def _handle_parallel(self, task: WorkflowTask, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle parallel task execution."""
        parallel_tasks = parameters.get('tasks', [])
        
        # Create task objects
        tasks_to_execute = []
        for task_data in parallel_tasks:
            parallel_task = WorkflowTask(
                id=f"{task.id}_parallel_{task_data['id']}",
                name=task_data['name'],
                task_type=TaskType(task_data['type']),
                parameters=task_data.get('parameters', {})
            )
            tasks_to_execute.append(parallel_task)
        
        # Execute tasks in parallel
        results = await asyncio.gather(
            *[self._execute_task(t, parameters) for t in tasks_to_execute],
            return_exceptions=True
        )
        
        return {
            'task_type': 'parallel',
            'results': results
        }
    
    def _transform_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform data (placeholder implementation)."""
        # Simple data transformation
        transformed = {}
        for key, value in data.items():
            if isinstance(value, str):
                transformed[key] = value.upper()
            else:
                transformed[key] = value
        return transformed
    
    def _evaluate_condition(self, condition: str, parameters: Dict[str, Any]) -> bool:
        """Evaluate a condition (simplified implementation)."""
        # Simple condition evaluation
        if '==' in condition:
            left, right = condition.split('==', 1)
            left = left.strip()
            right = right.strip().strip('"\'')
            return str(parameters.get(left, '')) == right
        elif '>' in condition:
            left, right = condition.split('>', 1)
            left = left.strip()
            right = right.strip()
            return float(parameters.get(left, 0)) > float(right)
        elif '<' in condition:
            left, right = condition.split('<', 1)
            left = left.strip()
            right = right.strip()
            return float(parameters.get(left, 0)) < float(right)
        else:
            return bool(parameters.get(condition, False))
    
    def _calculate_next_run(self, schedule: str) -> datetime:
        """Calculate next run time for scheduled workflow."""
        # Simplified implementation
        now = datetime.now()
        if 'daily' in schedule:
            return now + timedelta(days=1)
        elif 'hourly' in schedule:
            return now + timedelta(hours=1)
        else:
            return now + timedelta(minutes=30)
    
    def start_scheduler(self):
        """Start the workflow scheduler."""
        if self.scheduler_running:
            return
        
        self.scheduler_running = True
        
        def run_scheduler():
            while self.scheduler_running:
                schedule.run_pending()
                time.sleep(1)
        
        self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        print("⏰ Workflow scheduler started")
    
    def stop_scheduler(self):
        """Stop the workflow scheduler."""
        self.scheduler_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join()
        
        print("⏹️ Workflow scheduler stopped")
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow status."""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        is_running = workflow_id in self.running_workflows
        
        return {
            'workflow_id': workflow_id,
            'name': workflow.name,
            'enabled': workflow.enabled,
            'is_running': is_running,
            'last_run': workflow.last_run,
            'next_run': workflow.next_run,
            'schedule': workflow.schedule,
            'task_count': len(workflow.tasks),
            'running_info': self.running_workflows.get(workflow_id)
        }
    
    def get_workflow_history(self, workflow_id: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get workflow execution history."""
        history = self.workflow_history
        
        if workflow_id:
            history = [h for h in history if h['workflow_id'] == workflow_id]
        
        return history[-limit:] if limit else history
    
    def save_workflows(self, file_path: str = "workflows.yaml"):
        """Save workflows to file."""
        workflows_data = {}
        
        for workflow_id, workflow in self.workflows.items():
            workflows_data[workflow_id] = {
                'name': workflow.name,
                'description': workflow.description,
                'schedule': workflow.schedule,
                'enabled': workflow.enabled,
                'tasks': [
                    {
                        'id': task.id,
                        'name': task.name,
                        'type': task.task_type.value,
                        'parameters': task.parameters,
                        'dependencies': task.dependencies,
                        'max_retries': task.max_retries,
                        'timeout': task.timeout
                    }
                    for task in workflow.tasks
                ]
            }
        
        with open(file_path, 'w') as f:
            yaml.dump(workflows_data, f, default_flow_style=False)
        
        print(f"💾 Workflows saved to: {file_path}")
    
    def load_workflows(self, file_path: str = "workflows.yaml"):
        """Load workflows from file."""
        if not Path(file_path).exists():
            print(f"⚠️ Workflow file not found: {file_path}")
            return
        
        with open(file_path, 'r') as f:
            workflows_data = yaml.safe_load(f)
        
        for workflow_id, workflow_data in workflows_data.items():
            self.create_workflow(
                workflow_id=workflow_id,
                name=workflow_data['name'],
                description=workflow_data['description'],
                tasks=workflow_data['tasks'],
                schedule=workflow_data.get('schedule')
            )
        
        print(f"📂 Workflows loaded from: {file_path}")
    
    def generate_workflow_report(self) -> str:
        """Generate workflow automation report."""
        report = f"""
BUL Workflow Automation Report
=============================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

WORKFLOWS
---------
Total Workflows: {len(self.workflows)}
Running Workflows: {len(self.running_workflows)}
Scheduled Workflows: {len([w for w in self.workflows.values() if w.schedule])}

WORKFLOW DETAILS
----------------
"""
        
        for workflow_id, workflow in self.workflows.items():
            status = "Running" if workflow_id in self.running_workflows else "Idle"
            report += f"""
{workflow.name} ({workflow_id}):
  Status: {status}
  Enabled: {workflow.enabled}
  Schedule: {workflow.schedule or 'Manual'}
  Tasks: {len(workflow.tasks)}
  Last Run: {workflow.last_run or 'Never'}
  Next Run: {workflow.next_run or 'N/A'}
"""
        
        # Execution history
        recent_executions = self.get_workflow_history(limit=5)
        if recent_executions:
            report += f"""
RECENT EXECUTIONS
-----------------
"""
            for execution in recent_executions:
                report += f"""
{execution['workflow_id']} - {execution['started_at'].strftime('%Y-%m-%d %H:%M:%S')}:
  Status: {execution['status']}
  Duration: {(execution['completed_at'] - execution['started_at']).total_seconds():.2f}s
  Completed Tasks: {execution['completed_tasks']}/{execution['total_tasks']}
"""
        
        return report

def main():
    """Main workflow automation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BUL Workflow Automation")
    parser.add_argument("--create", help="Create a new workflow")
    parser.add_argument("--execute", help="Execute a workflow")
    parser.add_argument("--list", action="store_true", help="List all workflows")
    parser.add_argument("--status", help="Get workflow status")
    parser.add_argument("--history", help="Get workflow history")
    parser.add_argument("--start-scheduler", action="store_true", help="Start workflow scheduler")
    parser.add_argument("--stop-scheduler", action="store_true", help="Stop workflow scheduler")
    parser.add_argument("--save", help="Save workflows to file")
    parser.add_argument("--load", help="Load workflows from file")
    parser.add_argument("--report", action="store_true", help="Generate workflow report")
    
    args = parser.parse_args()
    
    automation = WorkflowAutomation()
    
    print("🔄 BUL Workflow Automation")
    print("=" * 40)
    
    if args.create:
        # Create sample workflow
        sample_workflow = automation.create_workflow(
            workflow_id=args.create,
            name=f"Sample Workflow {args.create}",
            description="A sample workflow for demonstration",
            tasks=[
                {
                    'id': 'task1',
                    'name': 'Generate Document',
                    'type': 'document_generation',
                    'parameters': {
                        'query': 'Create a business strategy document',
                        'business_area': 'marketing',
                        'document_type': 'strategy'
                    }
                },
                {
                    'id': 'task2',
                    'name': 'Send Notification',
                    'type': 'notification',
                    'parameters': {
                        'type': 'log',
                        'message': 'Document generation completed'
                    },
                    'dependencies': ['task1']
                }
            ],
            schedule="every 1 hours"
        )
        print(f"✅ Created workflow: {sample_workflow.name}")
    
    elif args.execute:
        async def execute_workflow():
            try:
                result = await automation.execute_workflow(args.execute)
                print(f"✅ Workflow executed: {result['status']}")
                print(f"   Execution ID: {result['execution_id']}")
                print(f"   Completed Tasks: {result['completed_tasks']}")
                print(f"   Failed Tasks: {result['failed_tasks']}")
            except Exception as e:
                print(f"❌ Workflow execution failed: {e}")
        
        asyncio.run(execute_workflow())
    
    elif args.list:
        workflows = automation.workflows
        if workflows:
            print(f"\n📋 Available Workflows ({len(workflows)}):")
            print("-" * 50)
            for workflow_id, workflow in workflows.items():
                status = "Running" if workflow_id in automation.running_workflows else "Idle"
                print(f"{workflow.name} ({workflow_id}):")
                print(f"  Status: {status}")
                print(f"  Schedule: {workflow.schedule or 'Manual'}")
                print(f"  Tasks: {len(workflow.tasks)}")
                print()
        else:
            print("No workflows found.")
    
    elif args.status:
        try:
            status = automation.get_workflow_status(args.status)
            print(f"\n📊 Workflow Status: {status['name']}")
            print(f"   ID: {status['workflow_id']}")
            print(f"   Enabled: {status['enabled']}")
            print(f"   Running: {status['is_running']}")
            print(f"   Schedule: {status['schedule'] or 'Manual'}")
            print(f"   Last Run: {status['last_run'] or 'Never'}")
            print(f"   Next Run: {status['next_run'] or 'N/A'}")
        except Exception as e:
            print(f"❌ Error getting workflow status: {e}")
    
    elif args.history:
        history = automation.get_workflow_history(args.history, limit=10)
        if history:
            print(f"\n📜 Workflow History: {args.history}")
            print("-" * 50)
            for execution in history:
                print(f"{execution['started_at'].strftime('%Y-%m-%d %H:%M:%S')}: {execution['status']}")
                print(f"  Duration: {(execution['completed_at'] - execution['started_at']).total_seconds():.2f}s")
                print(f"  Tasks: {execution['completed_tasks']}/{execution['total_tasks']}")
                print()
        else:
            print("No execution history found.")
    
    elif args.start_scheduler:
        automation.start_scheduler()
        print("⏰ Workflow scheduler started")
        print("💡 Press Ctrl+C to stop")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            automation.stop_scheduler()
    
    elif args.stop_scheduler:
        automation.stop_scheduler()
    
    elif args.save:
        automation.save_workflows(args.save)
    
    elif args.load:
        automation.load_workflows(args.load)
    
    elif args.report:
        report = automation.generate_workflow_report()
        print(report)
        
        # Save report
        report_file = f"workflow_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"📄 Report saved to: {report_file}")
    
    else:
        # Show quick overview
        print(f"📋 Workflows: {len(automation.workflows)}")
        print(f"🔄 Running: {len(automation.running_workflows)}")
        print(f"📜 History: {len(automation.workflow_history)}")
        print(f"\n💡 Use --list to see all workflows")
        print(f"💡 Use --create <id> to create a sample workflow")
        print(f"💡 Use --execute <id> to run a workflow")
        print(f"💡 Use --start-scheduler to start the scheduler")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
