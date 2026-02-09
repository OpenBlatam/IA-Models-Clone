"""
Workflow Engine

Utilities for workflow management and execution.
"""

import logging
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class WorkflowStep:
    """Workflow step definition."""
    name: str
    func: Callable
    dependencies: List[str] = None
    retry: bool = False
    timeout: Optional[float] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class WorkflowEngine:
    """Workflow execution engine."""
    
    def __init__(self):
        """Initialize workflow engine."""
        self.steps: Dict[str, WorkflowStep] = {}
        self.execution_order: List[str] = []
    
    def add_step(
        self,
        name: str,
        func: Callable,
        dependencies: List[str] = None,
        **kwargs
    ) -> None:
        """
        Add workflow step.
        
        Args:
            name: Step name
            func: Step function
            dependencies: Step dependencies
            **kwargs: Additional step parameters
        """
        step = WorkflowStep(
            name=name,
            func=func,
            dependencies=dependencies or [],
            **kwargs
        )
        
        self.steps[name] = step
        self._update_execution_order()
        
        logger.info(f"Added workflow step: {name}")
    
    def _update_execution_order(self) -> None:
        """Update execution order based on dependencies."""
        # Topological sort
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(node: str) -> None:
            if node in temp_visited:
                raise ValueError(f"Circular dependency detected: {node}")
            
            if node in visited:
                return
            
            temp_visited.add(node)
            
            if node in self.steps:
                for dep in self.steps[node].dependencies:
                    visit(dep)
            
            temp_visited.remove(node)
            visited.add(node)
            order.append(node)
        
        for step_name in self.steps:
            if step_name not in visited:
                visit(step_name)
        
        self.execution_order = order
    
    def execute(
        self,
        initial_data: Any = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute workflow.
        
        Args:
            initial_data: Initial data
            context: Execution context
            
        Returns:
            Workflow results
        """
        if context is None:
            context = {}
        
        results = {}
        context['initial_data'] = initial_data
        
        for step_name in self.execution_order:
            step = self.steps[step_name]
            
            logger.info(f"Executing step: {step_name}")
            
            try:
                # Prepare inputs from dependencies
                inputs = {}
                for dep in step.dependencies:
                    if dep in results:
                        inputs[dep] = results[dep]
                
                # Execute step
                result = step.func(inputs, context)
                results[step_name] = result
                
            except Exception as e:
                logger.error(f"Step {step_name} failed: {e}")
                
                if step.retry:
                    # Retry logic
                    logger.info(f"Retrying step: {step_name}")
                    result = step.func(inputs, context)
                    results[step_name] = result
                else:
                    raise
        
        return results


def create_workflow() -> WorkflowEngine:
    """Create workflow engine."""
    return WorkflowEngine()


def execute_workflow(
    workflow: WorkflowEngine,
    initial_data: Any = None,
    **kwargs
) -> Dict[str, Any]:
    """Execute workflow."""
    return workflow.execute(initial_data, **kwargs)


def add_workflow_step(
    workflow: WorkflowEngine,
    name: str,
    func: Callable,
    **kwargs
) -> None:
    """Add step to workflow."""
    workflow.add_step(name, func, **kwargs)



