"""
Pipeline System
===============

System for creating and executing data processing pipelines.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable, Awaitable, TypeVar
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')


class PipelineStage(Enum):
    """Pipeline stage."""
    INPUT = "input"
    PROCESSING = "processing"
    OUTPUT = "output"


@dataclass
class PipelineStep:
    """Pipeline step definition."""
    name: str
    stage: PipelineStage
    processor: Callable[[T], Awaitable[T]]
    parallel: bool = False
    timeout: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Pipeline execution result."""
    pipeline_id: str
    input_data: Any
    output_data: Any
    stage_results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    duration: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class Pipeline:
    """Data processing pipeline."""
    
    def __init__(self, name: str):
        """
        Initialize pipeline.
        
        Args:
            name: Pipeline name
        """
        self.name = name
        self.steps: List[PipelineStep] = []
    
    def add_step(
        self,
        name: str,
        processor: Callable[[T], Awaitable[T]],
        stage: PipelineStage = PipelineStage.PROCESSING,
        parallel: bool = False,
        timeout: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add pipeline step.
        
        Args:
            name: Step name
            processor: Processing function
            stage: Pipeline stage
            parallel: Whether to run in parallel
            timeout: Optional timeout
            metadata: Optional metadata
        """
        step = PipelineStep(
            name=name,
            stage=stage,
            processor=processor,
            parallel=parallel,
            timeout=timeout,
            metadata=metadata or {}
        )
        self.steps.append(step)
        logger.debug(f"Added step {name} to pipeline {self.name}")
    
    async def process(self, data: T) -> PipelineResult:
        """
        Process data through pipeline.
        
        Args:
            data: Input data
            
        Returns:
            Pipeline result
        """
        pipeline_id = f"{self.name}_{datetime.now().timestamp()}"
        start = datetime.now()
        current_data = data
        stage_results = {}
        errors = []
        
        try:
            for step in self.steps:
                try:
                    if step.timeout:
                        result = await asyncio.wait_for(
                            step.processor(current_data),
                            timeout=step.timeout
                        )
                    else:
                        result = await step.processor(current_data)
                    
                    current_data = result
                    stage_results[step.name] = result
                    logger.debug(f"Step {step.name} completed")
                    
                except Exception as e:
                    error_msg = f"Step {step.name} failed: {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)
                    raise
            
            duration = (datetime.now() - start).total_seconds()
            return PipelineResult(
                pipeline_id=pipeline_id,
                input_data=data,
                output_data=current_data,
                stage_results=stage_results,
                duration=duration
            )
            
        except Exception as e:
            duration = (datetime.now() - start).total_seconds()
            return PipelineResult(
                pipeline_id=pipeline_id,
                input_data=data,
                output_data=current_data,
                stage_results=stage_results,
                errors=errors,
                duration=duration
            )
    
    async def process_batch(self, items: List[T]) -> List[PipelineResult]:
        """
        Process batch of items.
        
        Args:
            items: List of input items
            
        Returns:
            List of pipeline results
        """
        tasks = [self.process(item) for item in items]
        return await asyncio.gather(*tasks)


class PipelineManager:
    """Manager for multiple pipelines."""
    
    def __init__(self):
        """Initialize pipeline manager."""
        self.pipelines: Dict[str, Pipeline] = {}
        self.execution_history: List[PipelineResult] = []
        self.max_history = 1000
    
    def register(self, pipeline: Pipeline):
        """
        Register a pipeline.
        
        Args:
            pipeline: Pipeline instance
        """
        self.pipelines[pipeline.name] = pipeline
        logger.debug(f"Registered pipeline: {pipeline.name}")
    
    async def process(self, pipeline_name: str, data: Any) -> PipelineResult:
        """
        Process data through pipeline.
        
        Args:
            pipeline_name: Pipeline name
            data: Input data
            
        Returns:
            Pipeline result
        """
        if pipeline_name not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_name} not found")
        
        pipeline = self.pipelines[pipeline_name]
        result = await pipeline.process(data)
        
        # Save to history
        self.execution_history.append(result)
        if len(self.execution_history) > self.max_history:
            self.execution_history = self.execution_history[-self.max_history:]
        
        return result
    
    def get_pipeline(self, name: str) -> Optional[Pipeline]:
        """Get pipeline by name."""
        return self.pipelines.get(name)
    
    def get_history(self, limit: int = 100) -> List[PipelineResult]:
        """Get execution history."""
        return self.execution_history[-limit:]




