"""
Processing Pipeline Module

Multi-layer processing pipeline implementation.
"""

from typing import Dict, Any, List
import logging
import time

logger = logging.getLogger(__name__)

from .base import ProcessingResult, ProcessingLayer
from .layers import (
    PreprocessingLayer,
    FeatureExtractionLayer,
    MLInferenceLayer,
    PostprocessingLayer,
    ValidationLayer
)


class ProcessingPipeline:
    """
    Multi-layer processing pipeline.
    Chains multiple processing layers together.
    """
    
    def __init__(self):
        self.layers: List[ProcessingLayer] = []
        self.results: List[ProcessingResult] = []
    
    def add_layer(self, layer: ProcessingLayer):
        """
        Add a processing layer to the pipeline.
        
        Args:
            layer: Processing layer to add.
        """
        self.layers.append(layer)
        logger.info(f"Added layer: {layer.name} ({layer.stage.value})")
    
    def process(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """
        Process data through all layers.
        
        Args:
            input_data: Input data to process.
            **kwargs: Additional arguments.
        
        Returns:
            Dictionary with processing results.
        """
        start_time = time.time()
        self.results = []
        
        current_data = input_data
        
        for layer in self.layers:
            result = layer.process(current_data, **kwargs)
            self.results.append(result)
            
            if not result.success:
                logger.error(f"Layer {layer.name} failed: {result.error}")
                return {
                    "success": False,
                    "error": result.error,
                    "failed_layer": layer.name,
                    "results": [r.__dict__ for r in self.results]
                }
            
            current_data = result.data
        
        total_time = time.time() - start_time
        
        return {
            "success": True,
            "data": current_data,
            "results": [r.__dict__ for r in self.results],
            "total_time": total_time,
            "layers_processed": len(self.results)
        }
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline."""
        return {
            "num_layers": len(self.layers),
            "layers": [
                {
                    "name": layer.name,
                    "stage": layer.stage.value
                }
                for layer in self.layers
            ]
        }


def create_default_pipeline() -> ProcessingPipeline:
    """Create default processing pipeline."""
    pipeline = ProcessingPipeline()
    
    # Add layers in order
    pipeline.add_layer(PreprocessingLayer())
    pipeline.add_layer(FeatureExtractionLayer())
    pipeline.add_layer(MLInferenceLayer(model_name="multi_task"))
    pipeline.add_layer(PostprocessingLayer())
    pipeline.add_layer(ValidationLayer())
    
    return pipeline



