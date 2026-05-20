"""
Workflow Engine for Document Analyzer
======================================

Advanced workflow orchestration for complex document processing pipelines.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class WorkflowStatus(Enum):
    """Workflow status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class WorkflowStep:
    """Workflow step definition"""
    name: str
    func: Callable
    depends_on: List[str] = field(default_factory=list)
    retry_on_failure: bool = True
    max_retries: int = 3
    timeout: Optional[float] = None

@dataclass
class WorkflowResult:
    """Workflow execution result"""
    workflow_id: str
    status: WorkflowStatus
    steps_completed: List[str] = field(default_factory=list)
    steps_failed: List[str] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: float = 0.0
    error: Optional[str] = None

class WorkflowEngine:
    """Advanced workflow engine"""
    
    def __init__(self):
        self.workflows: Dict[str, List[WorkflowStep]] = {}
        self.executions: Dict[str, WorkflowResult] = {}
        logger.info("WorkflowEngine initialized")
    
    def register_workflow(
        self,
        workflow_name: str,
        steps: List[WorkflowStep]
    ):
        """Register a workflow"""
        self.workflows[workflow_name] = steps
        logger.info(f"Registered workflow: {workflow_name} with {len(steps)} steps")
    
    async def execute_workflow(
        self,
        workflow_name: str,
        initial_context: Dict[str, Any] = None,
        workflow_id: Optional[str] = None
    ) -> WorkflowResult:
        """Execute a workflow"""
        if workflow_name not in self.workflows:
            raise ValueError(f"Workflow '{workflow_name}' not found")
        
        workflow_id = workflow_id or f"{workflow_name}_{int(datetime.now().timestamp())}"
        steps = self.workflows[workflow_name]
        context = initial_context or {}
        
        result = WorkflowResult(
            workflow_id=workflow_id,
            status=WorkflowStatus.RUNNING,
            start_time=datetime.now()
        )
        
        self.executions[workflow_id] = result
        
        try:
            # Build dependency graph
            completed_steps = set()
            step_results = {}
            
            while len(completed_steps) < len(steps):
                # Find steps ready to run
                ready_steps = [
                    step for step in steps
                    if step.name not in completed_steps
                    and all(dep in completed_steps for dep in step.depends_on)
                ]
                
                if not ready_steps:
                    # Circular dependency or missing dependency
                    result.status = WorkflowStatus.FAILED
                    result.error = "Circular dependency or missing dependency detected"
                    break
                
                # Execute ready steps in parallel
                tasks = []
                for step in ready_steps:
                    tasks.append(self._execute_step(step, context, step_results))
                
                step_outputs = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for step, output in zip(ready_steps, step_outputs):
                    if isinstance(output, Exception):
                        result.steps_failed.append(step.name)
                        result.error = str(output)
                        if step.name in [s.name for s in steps if not s.retry_on_failure]:
                            result.status = WorkflowStatus.FAILED
                            break
                    else:
                        result.steps_completed.append(step.name)
                        step_results[step.name] = output
                        completed_steps.add(step.name)
                        result.results[step.name] = output
            
            if result.status == WorkflowStatus.RUNNING:
                if result.steps_failed:
                    result.status = WorkflowStatus.FAILED
                else:
                    result.status = WorkflowStatus.COMPLETED
            
        except Exception as e:
            result.status = WorkflowStatus.FAILED
            result.error = str(e)
            logger.error(f"Workflow execution failed: {e}")
        
        finally:
            result.end_time = datetime.now()
            if result.start_time:
                result.duration = (result.end_time - result.start_time).total_seconds()
        
        return result
    
    async def _execute_step(
        self,
        step: WorkflowStep,
        context: Dict[str, Any],
        step_results: Dict[str, Any]
    ) -> Any:
        """Execute a workflow step"""
        for attempt in range(step.max_retries + 1):
            try:
                # Prepare context
                step_context = {**context, **step_results}
                
                # Execute with timeout if specified
                if step.timeout:
                    output = await asyncio.wait_for(
                        step.func(step_context) if asyncio.iscoroutinefunction(step.func)
                        else asyncio.to_thread(step.func, step_context),
                        timeout=step.timeout
                    )
                else:
                    if asyncio.iscoroutinefunction(step.func):
                        output = await step.func(step_context)
                    else:
                        output = await asyncio.to_thread(step.func, step_context)
                
                return output
            
            except Exception as e:
                if attempt < step.max_retries and step.retry_on_failure:
                    wait_time = 2 ** attempt
                    logger.warning(f"Step '{step.name}' failed (attempt {attempt + 1}/{step.max_retries}). Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise
    
    def get_workflow_status(self, workflow_id: str) -> Optional[WorkflowResult]:
        """Get workflow execution status"""
        return self.executions.get(workflow_id)

# Global instance
workflow_engine = WorkflowEngine()
















