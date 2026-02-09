"""
Pipeline Transformer for Flux2 Clothing Changer
================================================

Advanced pipeline transformation system.
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PipelineStage:
    """Pipeline stage."""
    stage_id: str
    stage_type: str
    processor: Callable
    config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.config is None:
            self.config = {}


@dataclass
class Pipeline:
    """Pipeline definition."""
    pipeline_id: str
    name: str
    stages: List[PipelineStage]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class PipelineTransformer:
    """Advanced pipeline transformation system."""
    
    def __init__(self):
        """Initialize pipeline transformer."""
        self.pipelines: Dict[str, Pipeline] = {}
        self.stage_processors: Dict[str, Callable] = {}
    
    def register_stage_processor(
        self,
        stage_type: str,
        processor: Callable[[Any, Dict[str, Any]], Any],
    ) -> None:
        """
        Register stage processor.
        
        Args:
            stage_type: Stage type
            processor: Processor function
        """
        self.stage_processors[stage_type] = processor
        logger.info(f"Registered stage processor: {stage_type}")
    
    def create_pipeline(
        self,
        pipeline_id: str,
        name: str,
        stages: List[Dict[str, Any]],
    ) -> Pipeline:
        """
        Create pipeline.
        
        Args:
            pipeline_id: Pipeline identifier
            name: Pipeline name
            stages: List of stage definitions
            
        Returns:
            Created pipeline
        """
        pipeline_stages = []
        
        for stage_def in stages:
            stage_type = stage_def["type"]
            stage_id = stage_def.get("id", f"{stage_type}_{len(pipeline_stages)}")
            
            # Get processor
            if stage_type in self.stage_processors:
                processor = self.stage_processors[stage_type]
            else:
                # Default pass-through processor
                processor = lambda data, config: data
            
            stage = PipelineStage(
                stage_id=stage_id,
                stage_type=stage_type,
                processor=processor,
                config=stage_def.get("config", {}),
            )
            
            pipeline_stages.append(stage)
        
        pipeline = Pipeline(
            pipeline_id=pipeline_id,
            name=name,
            stages=pipeline_stages,
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
            Transformed data
        """
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline not found: {pipeline_id}")
        
        pipeline = self.pipelines[pipeline_id]
        result = input_data
        
        for stage in pipeline.stages:
            try:
                result = stage.processor(result, stage.config)
                logger.debug(f"Stage {stage.stage_id} completed")
            except Exception as e:
                logger.error(f"Stage {stage.stage_id} failed: {e}")
                raise
        
        return result
    
    def get_pipeline(self, pipeline_id: str) -> Optional[Pipeline]:
        """Get pipeline by ID."""
        return self.pipelines.get(pipeline_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline transformer statistics."""
        return {
            "total_pipelines": len(self.pipelines),
            "stage_processors": len(self.stage_processors),
        }


