from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
import logging
from typing import Union, List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from datetime import datetime
import uuid
from refactored_math_system import MathService, MathOperation, MathResult, OperationType, CalculationMethod
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Math Workflow Engine for OS Content
Advanced workflow orchestration for complex mathematical operations.
"""



logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowStepType(Enum):
    """Types of workflow steps."""
    MATH_OPERATION = "math_operation"
    CONDITION = "condition"
    LOOP = "loop"
    PARALLEL = "parallel"
    CUSTOM = "custom"


@dataclass
class WorkflowStep:
    """Individual step in a workflow."""
    step_id: str
    step_type: WorkflowStepType
    name: str
    config: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    timeout: Optional[float] = None
    retries: int = 0
    retry_delay: float = 1.0
    
    def __post_init__(self) -> Any:
        if not self.step_id:
            self.step_id = str(uuid.uuid4())


@dataclass
class WorkflowStepResult:
    """Result of a workflow step execution."""
    step_id: str
    step_name: str
    success: bool
    result: Any
    execution_time: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class WorkflowExecution:
    """Workflow execution context."""
    workflow_id: str
    workflow_name: str
    status: WorkflowStatus
    steps: List[WorkflowStep]
    results: Dict[str, WorkflowStepResult] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class MathWorkflowEngine:
    """Advanced workflow engine for mathematical operations."""
    
    def __init__(self, math_service: MathService, max_concurrent_workflows: int = 10):
        
    """__init__ function."""
self.math_service = math_service
        self.max_concurrent_workflows = max_concurrent_workflows
        self.active_workflows: Dict[str, WorkflowExecution] = {}
        self.workflow_history: List[WorkflowExecution] = []
        self.step_handlers: Dict[WorkflowStepType, Callable] = {
            WorkflowStepType.MATH_OPERATION: self._execute_math_operation,
            WorkflowStepType.CONDITION: self._execute_condition,
            WorkflowStepType.LOOP: self._execute_loop,
            WorkflowStepType.PARALLEL: self._execute_parallel,
            WorkflowStepType.CUSTOM: self._execute_custom
        }
        
        logger.info(f"MathWorkflowEngine initialized with {max_concurrent_workflows} max concurrent workflows")
    
    async def execute_workflow(self, workflow_name: str, steps: List[WorkflowStep], 
                              initial_variables: Dict[str, Any] = None) -> WorkflowExecution:
        """Execute a complete workflow."""
        workflow_id = str(uuid.uuid4())
        
        execution = WorkflowExecution(
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            status=WorkflowStatus.PENDING,
            steps=steps,
            variables=initial_variables or {}
        )
        
        # Check if we can start new workflow
        if len(self.active_workflows) >= self.max_concurrent_workflows:
            raise RuntimeError("Maximum number of concurrent workflows reached")
        
        try:
            self.active_workflows[workflow_id] = execution
            execution.status = WorkflowStatus.RUNNING
            execution.start_time = datetime.now()
            
            logger.info(f"Starting workflow {workflow_name} (ID: {workflow_id})")
            
            # Execute workflow steps
            await self._execute_workflow_steps(execution)
            
            execution.status = WorkflowStatus.COMPLETED
            execution.end_time = datetime.now()
            execution.total_execution_time = (execution.end_time - execution.start_time).total_seconds()
            
            logger.info(f"Workflow {workflow_name} completed successfully in {execution.total_execution_time:.2f}s")
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.end_time = datetime.now()
            execution.total_execution_time = (execution.end_time - execution.start_time).total_seconds()
            execution.metadata["error"] = str(e)
            
            logger.error(f"Workflow {workflow_name} failed: {e}")
            
        finally:
            # Move to history and remove from active
            self.workflow_history.append(execution)
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]
        
        return execution
    
    async def _execute_workflow_steps(self, execution: WorkflowExecution):
        """Execute all steps in a workflow."""
        # Build dependency graph
        step_dependencies = self._build_dependency_graph(execution.steps)
        
        # Execute steps in dependency order
        executed_steps = set()
        
        while len(executed_steps) < len(execution.steps):
            # Find steps that can be executed (all dependencies satisfied)
            ready_steps = [
                step for step in execution.steps
                if step.step_id not in executed_steps and
                all(dep in executed_steps for dep in step.dependencies)
            ]
            
            if not ready_steps:
                raise RuntimeError("Circular dependency detected in workflow")
            
            # Execute ready steps in parallel
            tasks = []
            for step in ready_steps:
                task = self._execute_step_with_retry(execution, step)
                tasks.append(task)
            
            step_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for step, result in zip(ready_steps, step_results):
                if isinstance(result, Exception):
                    # Step failed
                    step_result = WorkflowStepResult(
                        step_id=step.step_id,
                        step_name=step.name,
                        success=False,
                        result=None,
                        execution_time=0.0,
                        error_message=str(result)
                    )
                    execution.results[step.step_id] = step_result
                    raise result  # Fail the entire workflow
                else:
                    execution.results[step.step_id] = result
                    executed_steps.add(step.step_id)
    
    def _build_dependency_graph(self, steps: List[WorkflowStep]) -> Dict[str, List[str]]:
        """Build dependency graph for workflow steps."""
        dependencies = {}
        for step in steps:
            dependencies[step.step_id] = step.dependencies
        return dependencies
    
    async def _execute_step_with_retry(self, execution: WorkflowExecution, 
                                     step: WorkflowStep) -> WorkflowStepResult:
        """Execute a single step with retry logic."""
        start_time = time.time()
        last_exception = None
        
        for attempt in range(step.retries + 1):
            try:
                if step.timeout:
                    result = await asyncio.wait_for(
                        self._execute_single_step(execution, step),
                        timeout=step.timeout
                    )
                else:
                    result = await self._execute_single_step(execution, step)
                
                execution_time = time.time() - start_time
                result.execution_time = execution_time
                return result
                
            except Exception as e:
                last_exception = e
                if attempt < step.retries:
                    logger.warning(f"Step {step.name} failed (attempt {attempt + 1}/{step.retries + 1}): {e}")
                    await asyncio.sleep(step.retry_delay)
                else:
                    logger.error(f"Step {step.name} failed after {step.retries + 1} attempts: {e}")
        
        # All retries exhausted
        execution_time = time.time() - start_time
        return WorkflowStepResult(
            step_id=step.step_id,
            step_name=step.name,
            success=False,
            result=None,
            execution_time=execution_time,
            error_message=str(last_exception)
        )
    
    async def _execute_single_step(self, execution: WorkflowExecution, 
                                 step: WorkflowStep) -> WorkflowStepResult:
        """Execute a single workflow step."""
        start_time = time.time()
        
        try:
            handler = self.step_handlers.get(step.step_type)
            if not handler:
                raise ValueError(f"No handler for step type: {step.step_type}")
            
            result = await handler(execution, step)
            
            execution_time = time.time() - start_time
            return WorkflowStepResult(
                step_id=step.step_id,
                step_name=step.name,
                success=True,
                result=result,
                execution_time=execution_time,
                metadata={"step_type": step.step_type.value}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return WorkflowStepResult(
                step_id=step.step_id,
                step_name=step.name,
                success=False,
                result=None,
                execution_time=execution_time,
                error_message=str(e)
            )
    
    async def _execute_math_operation(self, execution: WorkflowExecution, 
                                    step: WorkflowStep) -> Any:
        """Execute a mathematical operation step."""
        config = step.config
        
        # Resolve variables in operands
        operands = []
        for operand in config.get("operands", []):
            if isinstance(operand, str) and operand.startswith("$"):
                # Variable reference
                var_name = operand[1:]
                if var_name in execution.variables:
                    operands.append(execution.variables[var_name])
                else:
                    raise ValueError(f"Variable {var_name} not found")
            else:
                operands.append(operand)
        
        # Create math operation
        operation = MathOperation(
            operation_type=OperationType(config["operation"].upper()),
            operands=operands,
            method=CalculationMethod(config.get("method", "basic"))
        )
        
        # Execute operation
        result = await self.math_service.processor.process_operation(operation)
        
        # Store result in variables if specified
        output_var = config.get("output_variable")
        if output_var:
            execution.variables[output_var] = result.value
        
        return result.value
    
    async def _execute_condition(self, execution: WorkflowExecution, 
                               step: WorkflowStep) -> Any:
        """Execute a conditional step."""
        config = step.config
        condition = config.get("condition", "")
        
        # Simple condition evaluation (can be extended)
        if condition == "true":
            return True
        elif condition == "false":
            return False
        else:
            # Evaluate condition using variables
            try:
                # Simple expression evaluation (can be made more sophisticated)
                return eval(condition, {"__builtins__": {}}, execution.variables)
            except Exception as e:
                raise ValueError(f"Invalid condition: {condition} - {e}")
    
    async def _execute_loop(self, execution: WorkflowExecution, 
                          step: WorkflowStep) -> List[Any]:
        """Execute a loop step."""
        config = step.config
        iterations = config.get("iterations", 1)
        loop_steps = config.get("steps", [])
        
        results = []
        for i in range(iterations):
            # Execute loop steps
            loop_execution = WorkflowExecution(
                workflow_id=f"{execution.workflow_id}_loop_{i}",
                workflow_name=f"Loop {i}",
                status=WorkflowStatus.RUNNING,
                steps=loop_steps,
                variables=execution.variables.copy()
            )
            
            await self._execute_workflow_steps(loop_execution)
            results.append(loop_execution.results)
        
        return results
    
    async def _execute_parallel(self, execution: WorkflowExecution, 
                              step: WorkflowStep) -> List[Any]:
        """Execute parallel steps."""
        config = step.config
        parallel_steps = config.get("steps", [])
        
        # Execute all steps in parallel
        tasks = []
        for parallel_step in parallel_steps:
            task = self._execute_single_step(execution, parallel_step)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
    
    async def _execute_custom(self, execution: WorkflowExecution, 
                            step: WorkflowStep) -> Any:
        """Execute a custom step."""
        config = step.config
        custom_function = config.get("function")
        
        if custom_function:
            # Execute custom function with variables
            try:
                return custom_function(execution.variables)
            except Exception as e:
                raise ValueError(f"Custom function failed: {e}")
        else:
            raise ValueError("No custom function specified")
    
    def get_workflow_status(self, workflow_id: str) -> Optional[WorkflowExecution]:
        """Get status of a specific workflow."""
        if workflow_id in self.active_workflows:
            return self.active_workflows[workflow_id]
        
        # Check history
        for workflow in self.workflow_history:
            if workflow.workflow_id == workflow_id:
                return workflow
        
        return None
    
    def get_all_workflows(self) -> Dict[str, WorkflowExecution]:
        """Get all workflows (active and completed)."""
        all_workflows = self.active_workflows.copy()
        
        # Add completed workflows
        for workflow in self.workflow_history:
            all_workflows[workflow.workflow_id] = workflow
        
        return all_workflows
    
    def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow."""
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            workflow.status = WorkflowStatus.CANCELLED
            workflow.end_time = datetime.now()
            workflow.total_execution_time = (workflow.end_time - workflow.start_time).total_seconds()
            
            # Move to history
            self.workflow_history.append(workflow)
            del self.active_workflows[workflow_id]
            
            logger.info(f"Workflow {workflow_id} cancelled")
            return True
        
        return False


# Example workflow definitions
def create_simple_calculation_workflow() -> List[WorkflowStep]:
    """Create a simple calculation workflow."""
    return [
        WorkflowStep(
            step_id="step1",
            step_type=WorkflowStepType.MATH_OPERATION,
            name="Add numbers",
            config={
                "operation": "add",
                "operands": [1, 2, 3],
                "method": "basic",
                "output_variable": "sum"
            }
        ),
        WorkflowStep(
            step_id="step2",
            step_type=WorkflowStepType.MATH_OPERATION,
            name="Multiply by 2",
            config={
                "operation": "multiply",
                "operands": ["$sum", 2],
                "method": "basic",
                "output_variable": "result"
            },
            dependencies=["step1"]
        )
    ]


def create_complex_workflow() -> List[WorkflowStep]:
    """Create a more complex workflow with conditions and loops."""
    return [
        WorkflowStep(
            step_id="init",
            step_type=WorkflowStepType.MATH_OPERATION,
            name="Initialize counter",
            config={
                "operation": "add",
                "operands": [0],
                "method": "basic",
                "output_variable": "counter"
            }
        ),
        WorkflowStep(
            step_id="loop",
            step_type=WorkflowStepType.LOOP,
            name="Process loop",
            config={
                "iterations": 3,
                "steps": [
                    WorkflowStep(
                        step_id="increment",
                        step_type=WorkflowStepType.MATH_OPERATION,
                        name="Increment counter",
                        config={
                            "operation": "add",
                            "operands": ["$counter", 1],
                            "method": "basic",
                            "output_variable": "counter"
                        }
                    )
                ]
            },
            dependencies=["init"]
        )
    ]


# Example usage
async def main():
    """Example usage of the math workflow engine."""
    # Create math service
    math_service = create_math_service()
    
    # Create workflow engine
    workflow_engine = MathWorkflowEngine(math_service)
    
    # Create and execute simple workflow
    simple_workflow = create_simple_calculation_workflow()
    result = await workflow_engine.execute_workflow("Simple Calculation", simple_workflow)
    
    print(f"Workflow completed: {result.status}")
    print(f"Variables: {result.variables}")
    print(f"Execution time: {result.total_execution_time:.2f}s")
    
    # Create and execute complex workflow
    complex_workflow = create_complex_workflow()
    result2 = await workflow_engine.execute_workflow("Complex Workflow", complex_workflow)
    
    print(f"Complex workflow completed: {result2.status}")
    print(f"Variables: {result2.variables}")
    print(f"Execution time: {result2.total_execution_time:.2f}s")


match __name__:
    case "__main__":
    asyncio.run(main()) 