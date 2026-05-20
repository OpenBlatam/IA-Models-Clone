"""
Data Pipeline for Color Grading AI
===================================

Advanced data processing pipeline system.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Pipeline stages."""
    INPUT = "input"
    TRANSFORM = "transform"
    PROCESS = "process"
    VALIDATE = "validate"
    OUTPUT = "output"


@dataclass
class PipelineStep:
    """Pipeline step definition."""
    name: str
    stage: PipelineStage
    processor: Callable
    dependencies: List[str] = field(default_factory=list)
    error_handler: Optional[Callable] = None
    retry_count: int = 0
    timeout: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Pipeline execution result."""
    success: bool
    output: Any
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class DataPipeline:
    """
    Data processing pipeline.
    
    Features:
    - Multi-stage processing
    - Dependency management
    - Error handling
    - Retry logic
    - Parallel processing
    - Result aggregation
    """
    
    def __init__(self, name: str):
        """
        Initialize data pipeline.
        
        Args:
            name: Pipeline name
        """
        self.name = name
        self._steps: List[PipelineStep] = []
        self._execution_history: List[PipelineResult] = []
    
    def add_step(
        self,
        name: str,
        stage: PipelineStage,
        processor: Callable,
        dependencies: Optional[List[str]] = None,
        error_handler: Optional[Callable] = None,
        retry_count: int = 0,
        timeout: Optional[float] = None
    ):
        """
        Add pipeline step.
        
        Args:
            name: Step name
            stage: Pipeline stage
            processor: Processing function
            dependencies: Optional dependencies
            error_handler: Optional error handler
            retry_count: Retry count
            timeout: Optional timeout
        """
        step = PipelineStep(
            name=name,
            stage=stage,
            processor=processor,
            dependencies=dependencies or [],
            error_handler=error_handler,
            retry_count=retry_count,
            timeout=timeout
        )
        
        self._steps.append(step)
        logger.debug(f"Added pipeline step: {name} ({stage.value})")
    
    async def execute(
        self,
        input_data: Any,
        parallel: bool = False
    ) -> PipelineResult:
        """
        Execute pipeline.
        
        Args:
            input_data: Input data
            parallel: Whether to execute in parallel where possible
            
        Returns:
            Pipeline result
        """
        start_time = datetime.now()
        intermediate_results = {}
        errors = []
        
        try:
            # Group steps by stage
            stages = self._group_by_stage()
            
            # Execute stages in order
            current_data = input_data
            
            for stage_name, steps in stages.items():
                if parallel and len(steps) > 1:
                    # Execute steps in parallel
                    results = await self._execute_parallel(steps, current_data, intermediate_results)
                else:
                    # Execute steps sequentially
                    results = await self._execute_sequential(steps, current_data, intermediate_results)
                
                intermediate_results.update(results)
                
                # Use last result as input for next stage
                if results:
                    current_data = list(results.values())[-1]
            
            success = len(errors) == 0
            
        except Exception as e:
            logger.error(f"Pipeline execution error: {e}")
            success = False
            errors.append(str(e))
            current_data = None
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        result = PipelineResult(
            success=success,
            output=current_data,
            intermediate_results=intermediate_results,
            errors=errors,
            execution_time=execution_time
        )
        
        self._execution_history.append(result)
        return result
    
    def _group_by_stage(self) -> Dict[PipelineStage, List[PipelineStep]]:
        """Group steps by stage."""
        stages = {}
        for step in self._steps:
            if step.stage not in stages:
                stages[step.stage] = []
            stages[step.stage].append(step)
        return stages
    
    async def _execute_sequential(
        self,
        steps: List[PipelineStep],
        input_data: Any,
        previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute steps sequentially."""
        results = {}
        current_data = input_data
        
        for step in steps:
            # Check dependencies
            if not all(dep in previous_results for dep in step.dependencies):
                continue
            
            try:
                result = await self._execute_step(step, current_data, previous_results)
                results[step.name] = result
                current_data = result
            except Exception as e:
                if step.error_handler:
                    try:
                        result = await step.error_handler(e, current_data)
                        results[step.name] = result
                    except Exception as handler_error:
                        logger.error(f"Error handler failed: {handler_error}")
                else:
                    raise
        
        return results
    
    async def _execute_parallel(
        self,
        steps: List[PipelineStep],
        input_data: Any,
        previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute steps in parallel."""
        # Only execute steps without dependencies in parallel
        ready_steps = [s for s in steps if not s.dependencies]
        pending_steps = [s for s in steps if s.dependencies]
        
        results = {}
        
        # Execute ready steps in parallel
        if ready_steps:
            tasks = [
                self._execute_step(step, input_data, previous_results)
                for step in ready_steps
            ]
            step_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for step, result in zip(ready_steps, step_results):
                if isinstance(result, Exception):
                    logger.error(f"Step {step.name} failed: {result}")
                else:
                    results[step.name] = result
        
        # Execute pending steps sequentially
        for step in pending_steps:
            if all(dep in results for dep in step.dependencies):
                try:
                    result = await self._execute_step(step, input_data, {**previous_results, **results})
                    results[step.name] = result
                except Exception as e:
                    logger.error(f"Step {step.name} failed: {e}")
        
        return results
    
    async def _execute_step(
        self,
        step: PipelineStep,
        input_data: Any,
        previous_results: Dict[str, Any]
    ) -> Any:
        """Execute a single pipeline step."""
        # Prepare parameters
        params = {"input": input_data}
        params.update({k: v for k, v in previous_results.items() if k in step.dependencies})
        
        # Execute with retry
        last_error = None
        for attempt in range(step.retry_count + 1):
            try:
                if step.timeout:
                    return await asyncio.wait_for(
                        self._call_processor(step.processor, params),
                        timeout=step.timeout
                    )
                else:
                    return await self._call_processor(step.processor, params)
            except Exception as e:
                last_error = e
                if attempt < step.retry_count:
                    logger.warning(f"Step {step.name} failed, retrying: {e}")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
        
        if last_error:
            raise last_error
    
    async def _call_processor(self, processor: Callable, params: Dict[str, Any]) -> Any:
        """Call processor function."""
        if asyncio.iscoroutinefunction(processor):
            return await processor(**params)
        else:
            return processor(**params)
    
    def get_execution_history(self, limit: int = 100) -> List[PipelineResult]:
        """Get execution history."""
        return self._execution_history[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        if not self._execution_history:
            return {
                "total_executions": 0,
                "success_rate": 0.0,
            }
        
        total = len(self._execution_history)
        successful = sum(1 for r in self._execution_history if r.success)
        
        return {
            "name": self.name,
            "steps_count": len(self._steps),
            "total_executions": total,
            "successful_executions": successful,
            "failed_executions": total - successful,
            "success_rate": successful / total if total > 0 else 0.0,
            "avg_execution_time": sum(r.execution_time for r in self._execution_history) / total,
        }


