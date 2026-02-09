"""
Task Orchestrator

Utilities for task orchestration and DAG execution.
"""

import logging
from typing import Dict, List, Any, Callable, Set
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class TaskOrchestrator:
    """Orchestrate tasks in DAG (Directed Acyclic Graph)."""
    
    def __init__(self):
        """Initialize task orchestrator."""
        self.tasks: Dict[str, Callable] = {}
        self.dependencies: Dict[str, List[str]] = defaultdict(list)
        self.execution_order: List[str] = []
    
    def add_task(
        self,
        task_name: str,
        task_func: Callable,
        depends_on: List[str] = None
    ) -> None:
        """
        Add task to orchestrator.
        
        Args:
            task_name: Task name
            task_func: Task function
            depends_on: Task dependencies
        """
        self.tasks[task_name] = task_func
        
        if depends_on:
            self.dependencies[task_name] = depends_on
        
        self._compute_execution_order()
    
    def _compute_execution_order(self) -> None:
        """Compute task execution order using topological sort."""
        # Build dependency graph
        in_degree = {task: 0 for task in self.tasks}
        
        for task, deps in self.dependencies.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[task] += 1
        
        # Topological sort
        queue = deque([task for task, degree in in_degree.items() if degree == 0])
        order = []
        
        while queue:
            task = queue.popleft()
            order.append(task)
            
            # Reduce in-degree for dependent tasks
            for dependent_task, deps in self.dependencies.items():
                if task in deps:
                    in_degree[dependent_task] -= 1
                    if in_degree[dependent_task] == 0:
                        queue.append(dependent_task)
        
        if len(order) != len(self.tasks):
            raise ValueError("Circular dependency detected in task graph")
        
        self.execution_order = order
    
    def execute(
        self,
        initial_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute all tasks in order.
        
        Args:
            initial_context: Initial execution context
            
        Returns:
            Task results
        """
        if initial_context is None:
            initial_context = {}
        
        results = {}
        
        for task_name in self.execution_order:
            task_func = self.tasks[task_name]
            
            # Prepare inputs from dependencies
            inputs = {}
            for dep in self.dependencies.get(task_name, []):
                if dep in results:
                    inputs[dep] = results[dep]
            
            # Execute task
            logger.info(f"Executing task: {task_name}")
            result = task_func(inputs, initial_context)
            results[task_name] = result
        
        return results


def orchestrate_tasks(
    tasks: Dict[str, Callable],
    dependencies: Dict[str, List[str]] = None
) -> Dict[str, Any]:
    """
    Orchestrate and execute tasks.
    
    Args:
        tasks: Dictionary of task names and functions
        dependencies: Task dependencies
        
    Returns:
        Task results
    """
    orchestrator = TaskOrchestrator()
    
    for task_name, task_func in tasks.items():
        deps = dependencies.get(task_name, []) if dependencies else []
        orchestrator.add_task(task_name, task_func, deps)
    
    return orchestrator.execute()


def create_task_dag() -> TaskOrchestrator:
    """Create task DAG orchestrator."""
    return TaskOrchestrator()



