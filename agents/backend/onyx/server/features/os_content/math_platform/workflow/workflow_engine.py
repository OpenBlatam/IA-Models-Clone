from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import logging
from typing import Union, List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from datetime import datetime
import uuid
import traceback
from collections import defaultdict
from ..core.math_service import MathService, MathOperation, MathResult, OperationType, CalculationMethod
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Math Workflow Engine
Advanced workflow orchestration for complex mathematical operations with production features.
"""



logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class WorkflowStepType(Enum):
    """Types of workflow steps."""
    MATH_OPERATION = "math_operation"
    CONDITION = "condition"
    LOOP = "loop"
    PARALLEL = "parallel"
    CUSTOM = "custom"
    DELAY = "delay"
    LOG = "log"


class WorkflowError(Exception):
    """Custom workflow error."""
    pass


@dataclass
class WorkflowStep:
    """Individual step in a workflow with enhanced configuration."""
    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    step_type: WorkflowStepType = WorkflowStepType.MATH_OPERATION
    name: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    timeout: Optional[float] = None
    retries: int = 3
    retry_delay: float = 1.0
    max_retry_delay: float = 60.0
    backoff_multiplier: float = 2.0
    critical: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> Any:
        """Validate step configuration."""
        if not self.name:
            raise ValueError("Step name is required")
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("Timeout must be positive")
        if self.retries < 0:
            raise ValueError("Retries cannot be negative")
        if self.retry_delay < 0:
            raise ValueError("Retry delay cannot be negative")


@dataclass
class WorkflowStepResult:
    """Result of a workflow step execution with enhanced metadata."""
    step_id: str
    step_name: str
    success: bool
    result: Any
    execution_time: float
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    cache_hit: bool = False


@dataclass
class WorkflowExecution:
    """Workflow execution context with enhanced tracking."""
    workflow_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    workflow_name: str = ""
    status: WorkflowStatus = WorkflowStatus.PENDING
    steps: List[WorkflowStep] = field(default_factory=list)
    results: Dict[str, WorkflowStepResult] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_log: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowMetrics:
    """Workflow performance metrics."""
    total_workflows: int = 0
    successful_workflows: int = 0
    failed_workflows: int = 0
    average_execution_time: float = 0.0
    total_execution_time: float = 0.0
    active_workflows: int = 0
    max_concurrent_workflows: int = 0


class MathWorkflowEngine:
    """Advanced workflow engine for mathematical operations with production features."""
    
    def __init__(self, math_service: Optional[MathService] = None, 
                 max_concurrent_workflows: int = 10,
                 max_workflow_history: int = 1000):
        
    """__init__ function."""
self.math_service = math_service
        self.max_concurrent_workflows = max_concurrent_workflows
        self.max_workflow_history = max_workflow_history
        
        # Workflow management
        self.active_workflows: Dict[str, WorkflowExecution] = {}
        self.workflow_history: List[WorkflowExecution] = []
        self.workflow_queue: asyncio.Queue = asyncio.Queue()
        
        # Step handlers
        self.step_handlers: Dict[WorkflowStepType, Callable] = {
            WorkflowStepType.MATH_OPERATION: self._execute_math_operation,
            WorkflowStepType.CONDITION: self._execute_condition,
            WorkflowStepType.LOOP: self._execute_loop,
            WorkflowStepType.PARALLEL: self._execute_parallel,
            WorkflowStepType.CUSTOM: self._execute_custom,
            WorkflowStepType.DELAY: self._execute_delay,
            WorkflowStepType.LOG: self._execute_log
        }
        
        # Metrics and monitoring
        self.metrics = WorkflowMetrics()
        self.metrics.max_concurrent_workflows = max_concurrent_workflows
        
        # Performance tracking
        self.execution_times: List[float] = []
        self.error_counts: Dict[str, int] = defaultdict(int)
        
        # Shutdown flag
        self._shutdown_event = asyncio.Event()
        
        logger.info(f"MathWorkflowEngine initialized with max {max_concurrent_workflows} concurrent workflows")
    
    async def initialize(self) -> Any:
        """Initialize the workflow engine."""
        logger.info("Initializing workflow engine...")
        
        # Start workflow processor
        asyncio.create_task(self._workflow_processor_loop())
        
        logger.info("Workflow engine initialized")
    
    async def shutdown(self) -> Any:
        """Shutdown the workflow engine gracefully."""
        logger.info("Shutting down workflow engine...")
        
        self._shutdown_event.set()
        
        # Cancel all active workflows
        for workflow_id in list(self.active_workflows.keys()):
            await self.cancel_workflow(workflow_id)
        
        # Wait for workflow processor to finish
        await asyncio.sleep(1)
        
        logger.info("Workflow engine shutdown completed")
    
    async def _workflow_processor_loop(self) -> Any:
        """Main workflow processing loop."""
        while not self._shutdown_event.is_set():
            try:
                # Process workflow from queue
                workflow_data = await asyncio.wait_for(
                    self.workflow_queue.get(), 
                    timeout=1.0
                )
                
                await self._process_workflow(workflow_data)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Workflow processor error: {e}")
                await asyncio.sleep(1)
    
    async def _process_workflow(self, workflow_data: Dict[str, Any]):
        """Process a single workflow."""
        workflow_name = workflow_data["name"]
        steps = workflow_data["steps"]
        initial_variables = workflow_data.get("variables", {})
        
        await self.execute_workflow(workflow_name, steps, initial_variables)
    
    async def execute_workflow(self, workflow_name: str, steps: List[WorkflowStep], 
                              initial_variables: Dict[str, Any] = None) -> WorkflowExecution:
        """Execute a workflow with enhanced error handling and monitoring."""
        if len(self.active_workflows) >= self.max_concurrent_workflows:
            raise WorkflowError(f"Maximum concurrent workflows ({self.max_concurrent_workflows}) exceeded")
        
        # Create execution context
        execution = WorkflowExecution(
            workflow_name=workflow_name,
            steps=steps,
            variables=initial_variables or {},
            start_time=datetime.now()
        )
        
        self.active_workflows[execution.workflow_id] = execution
        self.metrics.active_workflows += 1
        self.metrics.total_workflows += 1
        
        logger.info(f"Starting workflow {workflow_name} (ID: {execution.workflow_id})")
        
        try:
            execution.status = WorkflowStatus.RUNNING
            
            # Execute workflow steps
            await self._execute_workflow_steps(execution)
            
            # Mark as completed
            execution.status = WorkflowStatus.COMPLETED
            execution.end_time = datetime.now()
            execution.total_execution_time = (
                execution.end_time - execution.start_time
            ).total_seconds()
            
            self.metrics.successful_workflows += 1
            self._update_execution_metrics(execution.total_execution_time)
            
            logger.info(f"Workflow {workflow_name} completed successfully in {execution.total_execution_time:.2f}s")
            
        except asyncio.CancelledError:
            execution.status = WorkflowStatus.CANCELLED
            execution.end_time = datetime.now()
            logger.info(f"Workflow {workflow_name} was cancelled")
            
        except asyncio.TimeoutError:
            execution.status = WorkflowStatus.TIMEOUT
            execution.end_time = datetime.now()
            self._log_workflow_error(execution, "Workflow execution timed out")
            logger.error(f"Workflow {workflow_name} timed out")
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.end_time = datetime.now()
            self._log_workflow_error(execution, str(e), traceback.format_exc())
            self.metrics.failed_workflows += 1
            logger.error(f"Workflow {workflow_name} failed: {e}")
            
        finally:
            # Cleanup
            self._add_to_history(execution)
            if execution.workflow_id in self.active_workflows:
                del self.active_workflows[execution.workflow_id]
            self.metrics.active_workflows -= 1
        
        return execution
    
    async def _execute_workflow_steps(self, execution: WorkflowExecution):
        """Execute workflow steps with dependency resolution."""
        # Build dependency graph
        dependency_graph = self._build_dependency_graph(execution.steps)
        
        # Create step lookup
        step_lookup = {step.step_id: step for step in execution.steps}
        
        # Track completed steps
        completed_steps = set()
        running_steps = set()
        
        while len(completed_steps) < len(execution.steps):
            # Find ready steps
            ready_steps = []
            for step in execution.steps:
                if (step.step_id not in completed_steps and 
                    step.step_id not in running_steps and
                    all(dep in completed_steps for dep in step.dependencies)):
                    ready_steps.append(step)
            
            if not ready_steps and not running_steps:
                # Deadlock detection
                raise WorkflowError("Workflow deadlock detected - circular dependencies")
            
            # Execute ready steps
            tasks = []
            for step in ready_steps:
                running_steps.add(step.step_id)
                task = asyncio.create_task(
                    self._execute_step_with_retry(execution, step)
                )
                tasks.append((step.step_id, task))
            
            # Wait for at least one step to complete
            if tasks:
                done, pending = await asyncio.wait(
                    [task for _, task in tasks],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Process completed tasks
                for step_id, task in tasks:
                    if task in done:
                        try:
                            result = await task
                            execution.results[step_id] = result
                            completed_steps.add(step_id)
                            running_steps.remove(step_id)
                            
                            # Update variables if step was successful
                            if result.success and hasattr(result.result, 'get'):
                                execution.variables.update(result.result)
                                
                        except Exception as e:
                            logger.error(f"Step {step_id} failed: {e}")
                            # Handle step failure based on critical flag
                            step = step_lookup[step_id]
                            if step.critical:
                                raise WorkflowError(f"Critical step {step.name} failed: {e}")
                            else:
                                # Mark as completed but failed
                                completed_steps.add(step_id)
                                running_steps.remove(step_id)
                
                # Cancel pending tasks
                for task in pending:
                    task.cancel()
    
    def _build_dependency_graph(self, steps: List[WorkflowStep]) -> Dict[str, List[str]]:
        """Build dependency graph for workflow steps."""
        graph = {}
        for step in steps:
            graph[step.step_id] = step.dependencies.copy()
        return graph
    
    async def _execute_step_with_retry(self, execution: WorkflowExecution, 
                                     step: WorkflowStep) -> WorkflowStepResult:
        """Execute a step with retry logic and exponential backoff."""
        start_time = time.time()
        last_error = None
        
        for attempt in range(step.retries + 1):
            try:
                # Execute step
                result = await asyncio.wait_for(
                    self._execute_single_step(execution, step),
                    timeout=step.timeout or 30.0
                )
                
                execution_time = time.time() - start_time
                
                return WorkflowStepResult(
                    step_id=step.step_id,
                    step_name=step.name,
                    success=True,
                    result=result,
                    execution_time=execution_time,
                    retry_count=attempt,
                    metadata=step.metadata
                )
                
            except asyncio.TimeoutError:
                last_error = f"Step timeout after {step.timeout}s"
                logger.warning(f"Step {step.name} timeout on attempt {attempt + 1}")
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Step {step.name} failed on attempt {attempt + 1}: {e}")
                
                if attempt < step.retries:
                    # Calculate delay with exponential backoff
                    delay = min(
                        step.retry_delay * (step.backoff_multiplier ** attempt),
                        step.max_retry_delay
                    )
                    await asyncio.sleep(delay)
        
        # All retries exhausted
        execution_time = time.time() - start_time
        
        return WorkflowStepResult(
            step_id=step.step_id,
            step_name=step.name,
            success=False,
            result=None,
            execution_time=execution_time,
            error_message=last_error,
            error_type=type(last_error).__name__ if last_error else None,
            retry_count=step.retries,
            metadata=step.metadata
        )
    
    async def _execute_single_step(self, execution: WorkflowExecution, 
                                 step: WorkflowStep) -> Any:
        """Execute a single workflow step."""
        if step.step_type not in self.step_handlers:
            raise WorkflowError(f"Unknown step type: {step.step_type}")
        
        handler = self.step_handlers[step.step_type]
        
        try:
            result = await handler(execution, step)
            return result
        except Exception as e:
            logger.error(f"Step execution error: {e}")
            raise
    
    async def _execute_math_operation(self, execution: WorkflowExecution, 
                                    step: WorkflowStep) -> Any:
        """Execute a mathematical operation step."""
        if not self.math_service:
            raise WorkflowError("Math service not available")
        
        config = step.config
        operation_type = config.get("operation")
        operands = config.get("operands", [])
        method = config.get("method", "basic")
        output_variable = config.get("output_variable")
        
        if not operation_type:
            raise WorkflowError("Operation type not specified")
        
        # Execute operation
        operation = MathOperation(
            operation_type=OperationType(operation_type),
            operands=operands,
            method=CalculationMethod(method)
        )
        
        result = await self.math_service.process_operation(operation)
        
        # Store result in variables if output variable specified
        if output_variable:
            execution.variables[output_variable] = result.value
        
        return {
            "value": result.value,
            "execution_time": result.execution_time,
            "success": result.success,
            "cache_hit": result.cache_hit
        }
    
    async def _execute_condition(self, execution: WorkflowExecution, 
                               step: WorkflowStep) -> Any:
        """Execute a conditional step."""
        config = step.config
        condition = config.get("condition")
        true_branch = config.get("true_branch", [])
        false_branch = config.get("false_branch", [])
        
        # Evaluate condition
        condition_result = self._evaluate_condition(condition, execution.variables)
        
        # Execute appropriate branch
        if condition_result:
            return await self._execute_branch(execution, true_branch)
        else:
            return await self._execute_branch(execution, false_branch)
    
    async def _execute_loop(self, execution: WorkflowExecution, 
                          step: WorkflowStep) -> List[Any]:
        """Execute a loop step."""
        config = step.config
        iterations = config.get("iterations", 1)
        loop_steps = config.get("steps", [])
        
        results = []
        for i in range(iterations):
            execution.variables["loop_index"] = i
            result = await self._execute_branch(execution, loop_steps)
            results.append(result)
        
        return results
    
    async def _execute_parallel(self, execution: WorkflowExecution, 
                              step: WorkflowStep) -> List[Any]:
        """Execute parallel steps."""
        config = step.config
        parallel_steps = config.get("steps", [])
        
        tasks = []
        for step_config in parallel_steps:
            task = asyncio.create_task(
                self._execute_branch(execution, [step_config])
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
    
    async def _execute_custom(self, execution: WorkflowExecution, 
                            step: WorkflowStep) -> Any:
        """Execute a custom step."""
        config = step.config
        function_name = config.get("function")
        
        if not function_name:
            raise WorkflowError("Custom function name not specified")
        
        # This would typically call a registered custom function
        # For now, return the function name
        return {"function": function_name, "variables": execution.variables}
    
    async def _execute_delay(self, execution: WorkflowExecution, 
                           step: WorkflowStep) -> None:
        """Execute a delay step."""
        config = step.config
        delay_seconds = config.get("seconds", 1.0)
        
        await asyncio.sleep(delay_seconds)
        return {"delay_seconds": delay_seconds}
    
    async def _execute_log(self, execution: WorkflowExecution, 
                          step: WorkflowStep) -> Any:
        """Execute a logging step."""
        config = step.config
        message = config.get("message", "")
        level = config.get("level", "info"f")
        
        # Format message with variables
        formatted_message = message"
        
        if level == "error":
            logger.error(formatted_message)
        elif level == "warning":
            logger.warning(formatted_message)
        elif level == "debug":
            logger.debug(formatted_message)
        else:
            logger.info(formatted_message)
        
        return {"message": formatted_message, "level": level}
    
    def _evaluate_condition(self, condition: str, variables: Dict[str, Any]) -> bool:
        """Evaluate a condition expression."""
        try:
            # Simple condition evaluation - could be enhanced with a proper expression parser
            return bool(eval(condition, {"__builtins__": {}}, variables))
        except Exception as e:
            logger.error(f"Condition evaluation error: {e}")
            return False
    
    async def _execute_branch(self, execution: WorkflowExecution, 
                            steps: List[Dict[str, Any]]) -> List[Any]:
        """Execute a branch of steps."""
        results = []
        for step_config in steps:
            step = WorkflowStep(
                name=step_config.get("name", "unnamed"),
                step_type=WorkflowStepType(step_config.get("type", "custom")),
                config=step_config.get("config", {})
            )
            
            result = await self._execute_step_with_retry(execution, step)
            results.append(result.result)
        
        return results
    
    def _log_workflow_error(self, execution: WorkflowExecution, 
                          error_message: str, traceback_str: str = None):
        """Log workflow error."""
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "workflow_id": execution.workflow_id,
            "workflow_name": execution.workflow_name,
            "error_message": error_message,
            "traceback": traceback_str
        }
        
        execution.error_log.append(error_entry)
        self.error_counts[execution.workflow_name] += 1
    
    def _update_execution_metrics(self, execution_time: float):
        """Update execution metrics."""
        self.execution_times.append(execution_time)
        if len(self.execution_times) > 1000:
            self.execution_times.pop(0)
        
        self.metrics.total_execution_time += execution_time
        self.metrics.average_execution_time = (
            self.metrics.total_execution_time / self.metrics.total_workflows
        )
    
    def _add_to_history(self, execution: WorkflowExecution):
        """Add execution to history."""
        self.workflow_history.append(execution)
        
        # Maintain history size
        if len(self.workflow_history) > self.max_workflow_history:
            self.workflow_history.pop(0)
    
    def get_workflow_status(self, workflow_id: str) -> Optional[WorkflowExecution]:
        """Get workflow status by ID."""
        # Check active workflows
        if workflow_id in self.active_workflows:
            return self.active_workflows[workflow_id]
        
        # Check history
        for execution in self.workflow_history:
            if execution.workflow_id == workflow_id:
                return execution
        
        return None
    
    def get_all_workflows(self) -> Dict[str, WorkflowExecution]:
        """Get all active workflows."""
        return self.active_workflows.copy()
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow."""
        if workflow_id in self.active_workflows:
            execution = self.active_workflows[workflow_id]
            execution.status = WorkflowStatus.CANCELLED
            execution.end_time = datetime.now()
            
            logger.info(f"Cancelled workflow {execution.workflow_name}")
            return True
        
        return False
    
    def get_workflow_metrics(self) -> Dict[str, Any]:
        """Get workflow performance metrics."""
        return {
            "total_workflows": self.metrics.total_workflows,
            "successful_workflows": self.metrics.successful_workflows,
            "failed_workflows": self.metrics.failed_workflows,
            "active_workflows": self.metrics.active_workflows,
            "average_execution_time": self.metrics.average_execution_time,
            "total_execution_time": self.metrics.total_execution_time,
            "success_rate": (
                self.metrics.successful_workflows / self.metrics.total_workflows
                if self.metrics.total_workflows > 0 else 0
            ),
            "error_counts": dict(self.error_counts)
        }


def create_simple_calculation_workflow() -> List[WorkflowStep]:
    """Create a simple calculation workflow example."""
    return [
        WorkflowStep(
            name="Add numbers",
            step_type=WorkflowStepType.MATH_OPERATION,
            config={
                "operation": "add",
                "operands": [1, 2, 3, 4, 5],
                "method": "basic",
                "output_variable": "sum"
            }
        ),
        WorkflowStep(
            name="Multiply result",
            step_type=WorkflowStepType.MATH_OPERATION,
            config={
                "operation": "multiply",
                "operands": ["{{sum}}", 2],
                "method": "numpy",
                "output_variable": "result"
            },
            dependencies=["Add numbers"]
        )
    ]


def create_complex_workflow() -> List[WorkflowStep]:
    """Create a complex workflow example."""
    return [
        WorkflowStep(
            name="Initial calculation",
            step_type=WorkflowStepType.MATH_OPERATION,
            config={
                "operation": "add",
                "operands": [10, 20, 30],
                "method": "basic",
                "output_variable": "initial_sum"
            }
        ),
        WorkflowStep(
            name="Conditional check",
            step_type=WorkflowStepType.CONDITION,
            config={
                "condition": "initial_sum > 50",
                "true_branch": [
                    {
                        "name": "High value calculation",
                        "type": "math_operation",
                        "config": {
                            "operation": "multiply",
                            "operands": ["{{initial_sum}}", 2],
                            "output_variable": "final_result"
                        }
                    }
                ],
                "false_branch": [
                    {
                        "name": "Low value calculation",
                        "type": "math_operation",
                        "config": {
                            "operation": "add",
                            "operands": ["{{initial_sum}}", 100],
                            "output_variable": "final_result"
                        }
                    }
                ]
            },
            dependencies=["Initial calculation"]
        ),
        WorkflowStep(
            name="Log result",
            step_type=WorkflowStepType.LOG,
            config={
                "message": "Final result: {{final_result}}",
                "level": "info"
            },
            dependencies=["Conditional check"]
        )
    ]


async def main():
    """Main function for testing."""
    # Create math service
    math_service = MathService()
    
    # Create workflow engine
    engine = MathWorkflowEngine(math_service)
    await engine.initialize()
    
    try:
        # Create and execute simple workflow
        steps = create_simple_calculation_workflow()
        result = await engine.execute_workflow("Simple Test", steps)
        
        print(f"Workflow result: {result}")
        print(f"Workflow status: {result.status}")
        print(f"Execution time: {result.total_execution_time:.2f}s")
        
        # Get metrics
        metrics = engine.get_workflow_metrics()
        print(f"Workflow metrics: {metrics}")
        
    finally:
        await engine.shutdown()


match __name__:
    case "__main__":
    asyncio.run(main()) 