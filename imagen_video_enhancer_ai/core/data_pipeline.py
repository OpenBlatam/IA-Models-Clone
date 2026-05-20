"""
Data Pipeline
==============

Advanced data transformation pipeline system.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable, Awaitable, TypeVar
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


@dataclass
class TransformStep:
    """Data transformation step."""
    name: str
    transformer: Callable[[T], Awaitable[R]]
    parallel: bool = False
    timeout: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TransformResult:
    """Transformation result."""
    success: bool
    input_data: Any
    output_data: Any
    step_results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    duration: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class DataPipeline:
    """Data transformation pipeline."""
    
    def __init__(self, name: str):
        """
        Initialize data pipeline.
        
        Args:
            name: Pipeline name
        """
        self.name = name
        self.steps: List[TransformStep] = []
    
    def add_step(
        self,
        name: str,
        transformer: Callable[[T], Awaitable[R]],
        parallel: bool = False,
        timeout: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add transformation step.
        
        Args:
            name: Step name
            transformer: Transformation function
            parallel: Whether to run in parallel
            timeout: Optional timeout
            metadata: Optional metadata
        """
        step = TransformStep(
            name=name,
            transformer=transformer,
            parallel=parallel,
            timeout=timeout,
            metadata=metadata or {}
        )
        self.steps.append(step)
        logger.debug(f"Added step {name} to pipeline {self.name}")
    
    async def transform(self, data: T) -> TransformResult:
        """
        Transform data through pipeline.
        
        Args:
            data: Input data
            
        Returns:
            Transformation result
        """
        start = datetime.now()
        current_data = data
        step_results = {}
        errors = []
        
        try:
            for step in self.steps:
                try:
                    if step.timeout:
                        result = await asyncio.wait_for(
                            step.transformer(current_data),
                            timeout=step.timeout
                        )
                    else:
                        result = await step.transformer(current_data)
                    
                    current_data = result
                    step_results[step.name] = result
                    logger.debug(f"Step {step.name} completed")
                    
                except Exception as e:
                    error_msg = f"Step {step.name} failed: {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)
                    raise
            
            duration = (datetime.now() - start).total_seconds()
            return TransformResult(
                success=True,
                input_data=data,
                output_data=current_data,
                step_results=step_results,
                duration=duration
            )
            
        except Exception as e:
            duration = (datetime.now() - start).total_seconds()
            return TransformResult(
                success=False,
                input_data=data,
                output_data=current_data,
                step_results=step_results,
                errors=errors,
                duration=duration
            )
    
    async def transform_batch(self, items: List[T]) -> List[TransformResult]:
        """
        Transform batch of items.
        
        Args:
            items: List of input items
            
        Returns:
            List of transformation results
        """
        tasks = [self.transform(item) for item in items]
        return await asyncio.gather(*tasks)


class DataPipelineManager:
    """Manager for multiple data pipelines."""
    
    def __init__(self):
        """Initialize pipeline manager."""
        self.pipelines: Dict[str, DataPipeline] = {}
        self.execution_history: List[TransformResult] = []
        self.max_history = 1000
    
    def register(self, pipeline: DataPipeline):
        """
        Register a pipeline.
        
        Args:
            pipeline: Pipeline instance
        """
        self.pipelines[pipeline.name] = pipeline
        logger.debug(f"Registered pipeline: {pipeline.name}")
    
    async def transform(self, pipeline_name: str, data: Any) -> TransformResult:
        """
        Transform data through pipeline.
        
        Args:
            pipeline_name: Pipeline name
            data: Input data
            
        Returns:
            Transformation result
        """
        if pipeline_name not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_name} not found")
        
        pipeline = self.pipelines[pipeline_name]
        result = await pipeline.transform(data)
        
        # Save to history
        self.execution_history.append(result)
        if len(self.execution_history) > self.max_history:
            self.execution_history = self.execution_history[-self.max_history:]
        
        return result
    
    def get_pipeline(self, name: str) -> Optional[DataPipeline]:
        """Get pipeline by name."""
        return self.pipelines.get(name)
    
    def get_history(self, limit: int = 100) -> List[TransformResult]:
        """Get execution history."""
        return self.execution_history[-limit:]




