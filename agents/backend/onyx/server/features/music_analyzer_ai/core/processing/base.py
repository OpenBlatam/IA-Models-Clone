"""
Processing Base Module

Base classes and types for processing layers.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging
import time

logger = logging.getLogger(__name__)


class ProcessingStage(Enum):
    """Processing stages in the pipeline."""
    PREPROCESSING = "preprocessing"
    FEATURE_EXTRACTION = "feature_extraction"
    ML_INFERENCE = "ml_inference"
    POSTPROCESSING = "postprocessing"
    VALIDATION = "validation"


@dataclass
class ProcessingResult:
    """Result from a processing layer."""
    stage: ProcessingStage
    data: Any
    metadata: Dict[str, Any]
    processing_time: float
    success: bool = True
    error: Optional[str] = None


class ProcessingLayer:
    """Base class for processing layers."""
    
    def __init__(self, name: str, stage: ProcessingStage):
        self.name = name
        self.stage = stage
    
    def process(self, input_data: Any, **kwargs) -> ProcessingResult:
        """
        Process input data.
        
        Args:
            input_data: Input data to process.
            **kwargs: Additional arguments.
        
        Returns:
            ProcessingResult with processed data.
        """
        start_time = time.time()
        try:
            result = self._process(input_data, **kwargs)
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                stage=self.stage,
                data=result,
                metadata={"layer": self.name},
                processing_time=processing_time,
                success=True
            )
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error in layer {self.name}: {str(e)}")
            return ProcessingResult(
                stage=self.stage,
                data=None,
                metadata={"layer": self.name},
                processing_time=processing_time,
                success=False,
                error=str(e)
            )
    
    def _process(self, input_data: Any, **kwargs) -> Any:
        """Override this method in subclasses."""
        raise NotImplementedError



