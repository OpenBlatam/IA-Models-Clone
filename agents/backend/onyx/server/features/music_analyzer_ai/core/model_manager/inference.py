"""
Model Inference Module

Inference pipeline creation and execution.
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

from .manager import ModelManager
from ...inference.pipelines import StandardInferencePipeline


class ModelInferenceMixin:
    """Model inference mixin."""
    
    def create_inference_pipeline(
        self: ModelManager,
        pipeline_name: str,
        model_name: str,
        preprocess_fn: Optional[Any] = None,
        postprocess_fn: Optional[Any] = None
    ) -> StandardInferencePipeline:
        """
        Create inference pipeline for a model.
        
        Args:
            pipeline_name: Name for the pipeline.
            model_name: Name of registered model.
            preprocess_fn: Preprocessing function.
            postprocess_fn: Postprocessing function.
        
        Returns:
            Inference pipeline instance.
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        pipeline = self.factory.create_inference_pipeline(
            model=model,
            config={
                "device": self.device_manager.get_device(),
                "use_mixed_precision": self.device_manager.enable_mixed_precision(),
                "preprocess_fn": preprocess_fn,
                "postprocess_fn": postprocess_fn
            }
        )
        
        self.inference_pipelines[pipeline_name] = pipeline
        
        logger.info(f"Created inference pipeline: {pipeline_name}")
        return pipeline
    
    def predict(
        self: ModelManager,
        pipeline_name: str,
        input_data: Any
    ) -> Dict[str, Any]:
        """
        Run inference using a pipeline.
        
        Args:
            pipeline_name: Name of inference pipeline.
            input_data: Input data.
        
        Returns:
            Prediction results.
        """
        if pipeline_name not in self.inference_pipelines:
            raise ValueError(f"Pipeline {pipeline_name} not found")
        
        pipeline = self.inference_pipelines[pipeline_name]
        return pipeline.predict(input_data)



