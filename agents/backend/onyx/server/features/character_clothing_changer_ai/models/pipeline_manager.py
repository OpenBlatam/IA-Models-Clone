"""
Pipeline Manager for Flux2 Clothing Changer
============================================

Advanced pipeline management and orchestration.
"""

import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PipelineStatus(Enum):
    """Pipeline status."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineStep:
    """Pipeline step."""
    step_id: str
    step_name: str
    processor: Callable
    config: Dict[str, Any] = None
    timeout: Optional[float] = None
    
    def __post_init__(self):
        if self.config is None:
            self.config = {}


@dataclass
class Pipeline:
    """Pipeline definition."""
    pipeline_id: str
    name: str
    steps: List[PipelineStep]
    status: PipelineStatus = PipelineStatus.IDLE
    created_at: float = time.time()
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class PipelineManager:
    """Advanced pipeline management system."""
    
    def __init__(self):
        """Initialize pipeline manager."""
        self.pipelines: Dict[str, Pipeline] = {}
        self.execution_history: List[Dict[str, Any]] = []
    
    def create_pipeline(
        self,
        pipeline_id: str,
        name: str,
        steps: List[Dict[str, Any]],
    ) -> Pipeline:
        """
        Create pipeline.
        
        Args:
            pipeline_id: Pipeline identifier
            name: Pipeline name
            steps: List of step definitions
            
        Returns:
            Created pipeline
        """
        pipeline_steps = []
        
        for step_def in steps:
            step = PipelineStep(
                step_id=step_def.get("id", f"step_{len(pipeline_steps)}"),
                step_name=step_def.get("name", ""),
                processor=step_def["processor"],
                config=step_def.get("config", {}),
                timeout=step_def.get("timeout"),
            )
            pipeline_steps.append(step)
        
        pipeline = Pipeline(
            pipeline_id=pipeline_id,
            name=name,
            steps=pipeline_steps,
        )
        
        self.pipelines[pipeline_id] = pipeline
        logger.info(f"Created pipeline: {pipeline_id}")
        return pipeline
    
    def execute_pipeline(
        self,
        pipeline_id: str,
        input_data: Any,
    ) -> Any:
        """
        Execute pipeline.
        
        Args:
            pipeline_id: Pipeline identifier
            input_data: Input data
            
        Returns:
            Pipeline output
        """
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline not found: {pipeline_id}")
        
        pipeline = self.pipelines[pipeline_id]
        pipeline.status = PipelineStatus.RUNNING
        pipeline.started_at = time.time()
        
        result = input_data
        
        try:
            for step in pipeline.steps:
                logger.debug(f"Executing step: {step.step_name}")
                result = step.processor(result, **step.config)
            
            pipeline.status = PipelineStatus.COMPLETED
            pipeline.completed_at = time.time()
            
            self.execution_history.append({
                "pipeline_id": pipeline_id,
                "status": "completed",
                "duration": pipeline.completed_at - pipeline.started_at,
                "timestamp": time.time(),
            })
            
            logger.info(f"Pipeline {pipeline_id} completed")
            return result
        except Exception as e:
            pipeline.status = PipelineStatus.FAILED
            logger.error(f"Pipeline {pipeline_id} failed: {e}")
            raise
    
    def get_pipeline(self, pipeline_id: str) -> Optional[Pipeline]:
        """Get pipeline by ID."""
        return self.pipelines.get(pipeline_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline manager statistics."""
        return {
            "total_pipelines": len(self.pipelines),
            "completed_pipelines": len([
                p for p in self.pipelines.values()
                if p.status == PipelineStatus.COMPLETED
            ]),
            "failed_pipelines": len([
                p for p in self.pipelines.values()
                if p.status == PipelineStatus.FAILED
            ]),
            "execution_history_size": len(self.execution_history),
        }


