"""Data transformation pipelines"""
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TransformType(Enum):
    """Transform types"""
    FILTER = "filter"
    MAP = "map"
    REDUCE = "reduce"
    SORT = "sort"
    GROUP = "group"
    MERGE = "merge"
    SPLIT = "split"


@dataclass
class PipelineStep:
    """Pipeline step definition"""
    name: str
    transform_type: TransformType
    function: Callable
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


class DataPipeline:
    """Data transformation pipeline"""
    
    def __init__(self, name: str):
        """
        Initialize pipeline
        
        Args:
            name: Pipeline name
        """
        self.name = name
        self.steps: List[PipelineStep] = []
    
    def add_step(
        self,
        name: str,
        transform_type: TransformType,
        function: Callable,
        parameters: Optional[Dict[str, Any]] = None
    ):
        """
        Add step to pipeline
        
        Args:
            name: Step name
            transform_type: Transform type
            function: Transform function
            parameters: Optional parameters
        """
        step = PipelineStep(
            name=name,
            transform_type=transform_type,
            function=function,
            parameters=parameters or {}
        )
        self.steps.append(step)
    
    def execute(self, data: Any) -> Any:
        """
        Execute pipeline
        
        Args:
            data: Input data
            
        Returns:
            Transformed data
        """
        result = data
        
        for step in self.steps:
            try:
                if step.transform_type == TransformType.FILTER:
                    result = [item for item in result if step.function(item, **step.parameters)]
                elif step.transform_type == TransformType.MAP:
                    result = [step.function(item, **step.parameters) for item in result]
                elif step.transform_type == TransformType.REDUCE:
                    result = step.function(result, **step.parameters)
                elif step.transform_type == TransformType.SORT:
                    result = sorted(result, key=lambda x: step.function(x, **step.parameters))
                elif step.transform_type == TransformType.GROUP:
                    groups = {}
                    for item in result:
                        key = step.function(item, **step.parameters)
                        if key not in groups:
                            groups[key] = []
                        groups[key].append(item)
                    result = groups
                elif step.transform_type == TransformType.MERGE:
                    result = step.function(result, **step.parameters)
                elif step.transform_type == TransformType.SPLIT:
                    result = step.function(result, **step.parameters)
                else:
                    result = step.function(result, **step.parameters)
            except Exception as e:
                logger.error(f"Error in pipeline step {step.name}: {e}")
                raise
        
        return result


class PipelineManager:
    """Manage data pipelines"""
    
    def __init__(self):
        self.pipelines: Dict[str, DataPipeline] = {}
        self._register_default_pipelines()
    
    def _register_default_pipelines(self):
        """Register default pipelines"""
        # Table processing pipeline
        table_pipeline = DataPipeline("table_processing")
        table_pipeline.add_step(
            "filter_empty",
            TransformType.FILTER,
            lambda table: len(table.get("rows", [])) > 0
        )
        table_pipeline.add_step(
            "add_index",
            TransformType.MAP,
            lambda table, index: {**table, "index": index},
            {"index": 0}
        )
        self.pipelines["table_processing"] = table_pipeline
        
        # Content enhancement pipeline
        content_pipeline = DataPipeline("content_enhancement")
        content_pipeline.add_step(
            "normalize_headings",
            TransformType.MAP,
            lambda heading: {**heading, "level": min(heading.get("level", 1), 6)}
        )
        self.pipelines["content_enhancement"] = content_pipeline
    
    def create_pipeline(self, name: str) -> DataPipeline:
        """
        Create a new pipeline
        
        Args:
            name: Pipeline name
            
        Returns:
            Pipeline instance
        """
        pipeline = DataPipeline(name)
        self.pipelines[name] = pipeline
        return pipeline
    
    def get_pipeline(self, name: str) -> Optional[DataPipeline]:
        """Get pipeline by name"""
        return self.pipelines.get(name)
    
    def list_pipelines(self) -> List[str]:
        """List all pipelines"""
        return list(self.pipelines.keys())
    
    def execute_pipeline(self, name: str, data: Any) -> Any:
        """
        Execute a pipeline
        
        Args:
            name: Pipeline name
            data: Input data
            
        Returns:
            Transformed data
        """
        pipeline = self.get_pipeline(name)
        if not pipeline:
            raise ValueError(f"Pipeline not found: {name}")
        
        return pipeline.execute(data)


# Global pipeline manager
_pipeline_manager: Optional[PipelineManager] = None


def get_pipeline_manager() -> PipelineManager:
    """Get global pipeline manager"""
    global _pipeline_manager
    if _pipeline_manager is None:
        _pipeline_manager = PipelineManager()
    return _pipeline_manager

