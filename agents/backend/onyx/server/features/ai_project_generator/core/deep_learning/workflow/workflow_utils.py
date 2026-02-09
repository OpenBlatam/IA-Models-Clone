"""
Workflow Utilities
==================

Workflow orchestration and automation utilities.
"""

import logging
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from enum import Enum
import time

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Task:
    """
    Workflow task.
    """
    name: str
    function: Callable
    dependencies: List[str] = None
    retries: int = 0
    timeout: Optional[float] = None
    
    def __post_init__(self):
        """Initialize task."""
        if self.dependencies is None:
            self.dependencies = []
        self.status = TaskStatus.PENDING
        self.result = None
        self.error = None
        self.start_time = None
        self.end_time = None


class Pipeline:
    """
    Pipeline for workflow execution.
    """
    
    def __init__(self, name: str):
        """
        Initialize pipeline.
        
        Args:
            name: Pipeline name
        """
        self.name = name
        self.tasks: Dict[str, Task] = {}
        self.execution_order: List[str] = []
    
    def add_task(
        self,
        name: str,
        function: Callable,
        dependencies: Optional[List[str]] = None,
        retries: int = 0,
        timeout: Optional[float] = None
    ) -> 'Pipeline':
        """
        Add task to pipeline.
        
        Args:
            name: Task name
            function: Task function
            dependencies: Task dependencies
            retries: Number of retries
            timeout: Task timeout
            
        Returns:
            Self for method chaining
        """
        task = Task(
            name=name,
            function=function,
            dependencies=dependencies or [],
            retries=retries,
            timeout=timeout
        )
        self.tasks[name] = task
        return self
    
    def _resolve_order(self) -> List[str]:
        """
        Resolve task execution order based on dependencies.
        
        Returns:
            Ordered list of task names
        """
        # Topological sort
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(task_name: str):
            if task_name in temp_visited:
                raise ValueError(f"Circular dependency detected involving {task_name}")
            if task_name in visited:
                return
            
            temp_visited.add(task_name)
            
            if task_name in self.tasks:
                for dep in self.tasks[task_name].dependencies:
                    visit(dep)
            
            temp_visited.remove(task_name)
            visited.add(task_name)
            order.append(task_name)
        
        for task_name in self.tasks:
            if task_name not in visited:
                visit(task_name)
        
        return order
    
    def execute(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute pipeline.
        
        Args:
            context: Execution context
            
        Returns:
            Execution results
        """
        if context is None:
            context = {}
        
        execution_order = self._resolve_order()
        results = {}
        
        for task_name in execution_order:
            task = self.tasks[task_name]
            
            # Check dependencies
            if not all(dep in results for dep in task.dependencies):
                task.status = TaskStatus.SKIPPED
                logger.warning(f"Task {task_name} skipped due to missing dependencies")
                continue
            
            # Execute task
            task.status = TaskStatus.RUNNING
            task.start_time = time.time()
            
            try:
                # Prepare task arguments from context and dependencies
                task_args = {**context}
                for dep in task.dependencies:
                    task_args[dep] = results[dep]
                
                # Execute with retries
                for attempt in range(task.retries + 1):
                    try:
                        if task.timeout:
                            # Simple timeout (could be improved with threading)
                            result = task.function(**task_args)
                        else:
                            result = task.function(**task_args)
                        
                        task.status = TaskStatus.COMPLETED
                        task.result = result
                        results[task_name] = result
                        break
                    
                    except Exception as e:
                        if attempt < task.retries:
                            logger.warning(f"Task {task_name} failed, retrying... ({attempt + 1}/{task.retries})")
                            continue
                        else:
                            raise e
                
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                logger.error(f"Task {task_name} failed: {e}")
                results[task_name] = None
            
            finally:
                task.end_time = time.time()
        
        return results


class Workflow:
    """
    Comprehensive workflow manager.
    """
    
    def __init__(self, name: str):
        """
        Initialize workflow.
        
        Args:
            name: Workflow name
        """
        self.name = name
        self.pipelines: Dict[str, Pipeline] = {}
        self.context: Dict[str, Any] = {}
    
    def add_pipeline(self, pipeline: Pipeline) -> 'Workflow':
        """
        Add pipeline to workflow.
        
        Args:
            pipeline: Pipeline instance
            
        Returns:
            Self for method chaining
        """
        self.pipelines[pipeline.name] = pipeline
        return self
    
    def set_context(self, key: str, value: Any) -> 'Workflow':
        """
        Set workflow context.
        
        Args:
            key: Context key
            value: Context value
            
        Returns:
            Self for method chaining
        """
        self.context[key] = value
        return self
    
    def execute(self, pipeline_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute workflow.
        
        Args:
            pipeline_name: Pipeline name (None for all)
            
        Returns:
            Execution results
        """
        if pipeline_name:
            if pipeline_name not in self.pipelines:
                raise ValueError(f"Pipeline {pipeline_name} not found")
            return self.pipelines[pipeline_name].execute(self.context)
        else:
            results = {}
            for name, pipeline in self.pipelines.items():
                results[name] = pipeline.execute(self.context)
            return results


class WorkflowExecutor:
    """
    Workflow executor with monitoring.
    """
    
    def __init__(self):
        """Initialize executor."""
        self.workflows: Dict[str, Workflow] = {}
        self.execution_history: List[Dict[str, Any]] = []
    
    def register_workflow(self, workflow: Workflow) -> None:
        """
        Register workflow.
        
        Args:
            workflow: Workflow instance
        """
        self.workflows[workflow.name] = workflow
    
    def execute_workflow(
        self,
        workflow_name: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute workflow.
        
        Args:
            workflow_name: Workflow name
            context: Execution context
            
        Returns:
            Execution results
        """
        if workflow_name not in self.workflows:
            raise ValueError(f"Workflow {workflow_name} not found")
        
        workflow = self.workflows[workflow_name]
        
        if context:
            for key, value in context.items():
                workflow.set_context(key, value)
        
        start_time = time.time()
        results = workflow.execute()
        end_time = time.time()
        
        execution_record = {
            'workflow': workflow_name,
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time,
            'results': results
        }
        
        self.execution_history.append(execution_record)
        return results
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get execution history."""
        return self.execution_history



