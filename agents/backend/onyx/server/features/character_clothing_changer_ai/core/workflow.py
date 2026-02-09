"""
Workflow System
===============

System for defining and executing workflows.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Workflow status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowStep:
    """Workflow step definition."""
    name: str
    func: Callable[[Dict[str, Any]], Awaitable[Any]]
    dependencies: List[str] = field(default_factory=list)
    retry_on_failure: bool = False
    timeout: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowResult:
    """Workflow execution result."""
    workflow_id: str
    status: WorkflowStatus
    results: Dict[str, Any] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)
    duration: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class Workflow:
    """Workflow definition and executor."""
    
    def __init__(self, name: str):
        """
        Initialize workflow.
        
        Args:
            name: Workflow name
        """
        self.name = name
        self.steps: Dict[str, WorkflowStep] = {}
        self.execution_order: List[str] = []
    
    def add_step(
        self,
        name: str,
        func: Callable[[Dict[str, Any]], Awaitable[Any]],
        dependencies: Optional[List[str]] = None,
        retry_on_failure: bool = False,
        timeout: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add workflow step.
        
        Args:
            name: Step name
            func: Step function
            dependencies: Optional step dependencies
            retry_on_failure: Whether to retry on failure
            timeout: Optional timeout
            metadata: Optional metadata
        """
        step = WorkflowStep(
            name=name,
            func=func,
            dependencies=dependencies or [],
            retry_on_failure=retry_on_failure,
            timeout=timeout,
            metadata=metadata or {}
        )
        self.steps[name] = step
        logger.debug(f"Added step {name} to workflow {self.name}")
    
    def _calculate_execution_order(self) -> List[str]:
        """Calculate execution order based on dependencies."""
        # Topological sort
        order = []
        visited = set()
        temp_visited = set()
        
        def visit(name: str):
            if name in temp_visited:
                raise ValueError(f"Circular dependency detected involving {name}")
            if name in visited:
                return
            
            temp_visited.add(name)
            
            if name in self.steps:
                step = self.steps[name]
                for dep in step.dependencies:
                    visit(dep)
            
            temp_visited.remove(name)
            visited.add(name)
            order.append(name)
        
        for name in self.steps.keys():
            if name not in visited:
                visit(name)
        
        return order
    
    async def execute(self, initial_context: Optional[Dict[str, Any]] = None) -> WorkflowResult:
        """
        Execute workflow.
        
        Args:
            initial_context: Optional initial context
            
        Returns:
            Workflow result
        """
        workflow_id = f"{self.name}_{datetime.now().timestamp()}"
        start = datetime.now()
        context = initial_context or {}
        results = {}
        errors = {}
        
        self.execution_order = self._calculate_execution_order()
        
        try:
            for step_name in self.execution_order:
                step = self.steps[step_name]
                
                # Check dependencies
                for dep in step.dependencies:
                    if dep not in results:
                        raise ValueError(f"Dependency {dep} not found for step {step_name}")
                
                # Execute step
                try:
                    if step.timeout:
                        result = await asyncio.wait_for(
                            step.func(context),
                            timeout=step.timeout
                        )
                    else:
                        result = await step.func(context)
                    
                    results[step_name] = result
                    context[step_name] = result
                    logger.info(f"Step {step_name} completed")
                    
                except Exception as e:
                    if step.retry_on_failure:
                        logger.warning(f"Step {step_name} failed, retrying: {e}")
                        try:
                            result = await step.func(context)
                            results[step_name] = result
                            context[step_name] = result
                        except Exception as retry_error:
                            errors[step_name] = str(retry_error)
                            raise
                    else:
                        errors[step_name] = str(e)
                        raise
            
            duration = (datetime.now() - start).total_seconds()
            return WorkflowResult(
                workflow_id=workflow_id,
                status=WorkflowStatus.COMPLETED,
                results=results,
                duration=duration
            )
            
        except Exception as e:
            duration = (datetime.now() - start).total_seconds()
            return WorkflowResult(
                workflow_id=workflow_id,
                status=WorkflowStatus.FAILED,
                results=results,
                errors=errors,
                duration=duration
            )


class WorkflowManager:
    """Manager for multiple workflows."""
    
    def __init__(self):
        """Initialize workflow manager."""
        self.workflows: Dict[str, Workflow] = {}
        self.execution_history: List[WorkflowResult] = []
        self.max_history = 1000
    
    def register(self, workflow: Workflow):
        """
        Register a workflow.
        
        Args:
            workflow: Workflow instance
        """
        self.workflows[workflow.name] = workflow
        logger.debug(f"Registered workflow: {workflow.name}")
    
    async def execute(self, workflow_name: str, context: Optional[Dict[str, Any]] = None) -> WorkflowResult:
        """
        Execute a workflow.
        
        Args:
            workflow_name: Workflow name
            context: Optional initial context
            
        Returns:
            Workflow result
        """
        if workflow_name not in self.workflows:
            raise ValueError(f"Workflow {workflow_name} not found")
        
        workflow = self.workflows[workflow_name]
        result = await workflow.execute(context)
        
        # Save to history
        self.execution_history.append(result)
        if len(self.execution_history) > self.max_history:
            self.execution_history = self.execution_history[-self.max_history:]
        
        return result
    
    def get_workflow(self, name: str) -> Optional[Workflow]:
        """Get workflow by name."""
        return self.workflows.get(name)
    
    def get_history(self, limit: int = 100) -> List[WorkflowResult]:
        """Get execution history."""
        return self.execution_history[-limit:]

