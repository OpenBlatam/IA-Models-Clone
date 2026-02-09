"""
Inference Factory Module

Inference pipeline creation functionality.
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ...inference.pipelines import StandardInferencePipeline


class InferenceFactoryMixin:
    """Inference factory mixin."""
    
    def create_inference_pipeline(
        self,
        model: nn.Module,
        config: Optional[Dict[str, Any]] = None
    ) -> StandardInferencePipeline:
        """
        Create inference pipeline.
        
        Args:
            model: Model instance
            config: Pipeline configuration
        
        Returns:
            Inference pipeline instance
        """
        config = config or {}
        
        return StandardInferencePipeline(
            model=model,
            preprocess_fn=config.get("preprocess_fn"),
            postprocess_fn=config.get("postprocess_fn"),
            device=config.get("device", "cuda"),
            use_mixed_precision=config.get("use_mixed_precision", True)
        )



