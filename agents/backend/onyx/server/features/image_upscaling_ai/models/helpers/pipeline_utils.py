"""
Pipeline Utilities
==================

Utilities for processing pipelines and workflows.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from pathlib import Path
from PIL import Image

from .metrics_utils import UpscalingMetrics
from .quality_calculator_utils import QualityCalculator
from .upscaling_algorithms import UpscalingAlgorithms
from .image_processing_utils import ImageProcessingUtils

logger = logging.getLogger(__name__)


class PipelineUtils:
    """Utilities for processing pipelines."""
    
    # Predefined pipelines
    PIPELINES = {
        "standard": [
            {"type": "upscale", "method": "lanczos"},
            {"type": "enhance_edges", "strength": 1.1},
        ],
        "quality": [
            {"type": "upscale", "method": "multi_scale", "passes": 2},
            {"type": "reduce_artifacts", "method": "bilateral"},
            {"type": "enhance_edges", "strength": 1.2},
        ],
        "speed": [
            {"type": "upscale", "method": "bicubic"},
        ],
        "balanced": [
            {"type": "upscale", "method": "opencv"},
            {"type": "enhance_edges", "strength": 1.15},
        ],
        "ultra_quality": [
            {"type": "upscale", "method": "real_esrgan_like"},
        ],
    }
    
    @staticmethod
    def execute_pipeline(
        image: Image.Image,
        scale_factor: float,
        pipeline: Union[str, List[Dict[str, Any]]],
        upscale_func: Optional[Callable] = None
    ) -> Image.Image:
        """
        Execute a processing pipeline.
        
        Args:
            image: Input image
            scale_factor: Scale factor
            pipeline: Pipeline name or list of steps
            upscale_func: Optional custom upscale function
            
        Returns:
            Processed image
        """
        # Get pipeline steps
        if isinstance(pipeline, str):
            if pipeline not in PipelineUtils.PIPELINES:
                logger.warning(f"Unknown pipeline '{pipeline}', using 'standard'")
                pipeline = "standard"
            steps = PipelineUtils.PIPELINES[pipeline]
        else:
            steps = pipeline
        
        result = image
        
        for step in steps:
            step_type = step.get("type")
            params = step.get("params", {})
            
            if step_type == "upscale":
                method = step.get("method", "lanczos")
                if upscale_func:
                    result = upscale_func(result, scale_factor, method)
                else:
                    if method == "lanczos":
                        result = UpscalingAlgorithms.upscale_lanczos(result, scale_factor)
                    elif method == "bicubic":
                        result = UpscalingAlgorithms.upscale_bicubic_enhanced(result, scale_factor)
                    elif method == "opencv":
                        result = UpscalingAlgorithms.upscale_opencv_edsr(result, scale_factor)
                    elif method == "multi_scale":
                        passes = step.get("passes", 2)
                        result = UpscalingAlgorithms.multi_scale_upscale(result, scale_factor, passes)
                    elif method == "real_esrgan_like":
                        result = UpscalingAlgorithms.upscale_real_esrgan_like(result, scale_factor)
                    else:
                        result = UpscalingAlgorithms.upscale_lanczos(result, scale_factor)
                        
            elif step_type == "enhance_edges":
                strength = step.get("strength", 1.1)
                result = ImageProcessingUtils.enhance_edges(result, strength)
                
            elif step_type == "reduce_artifacts":
                method = step.get("method", "bilateral")
                result = ImageProcessingUtils.reduce_artifacts(result, method)
                
            elif step_type == "apply_anti_aliasing":
                strength = step.get("strength", 0.5)
                result = ImageProcessingUtils.apply_anti_aliasing(result, strength)
                
            elif step_type == "adaptive_contrast":
                result = ImageProcessingUtils.adaptive_contrast_enhancement(result)
                
            elif step_type == "texture_enhancement":
                strength = step.get("strength", 0.3)
                result = ImageProcessingUtils.texture_enhancement(result, strength)
                
            elif step_type == "color_enhancement":
                saturation = step.get("saturation", 1.1)
                vibrance = step.get("vibrance", 1.05)
                result = ImageProcessingUtils.color_enhancement(result, saturation, vibrance)
        
        return result
    
    @staticmethod
    def get_pipeline_info(pipeline_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a pipeline.
        
        Args:
            pipeline_name: Name of the pipeline
            
        Returns:
            Pipeline information or None if not found
        """
        if pipeline_name not in PipelineUtils.PIPELINES:
            return None
        
        return {
            "name": pipeline_name,
            "steps": PipelineUtils.PIPELINES[pipeline_name],
            "step_count": len(PipelineUtils.PIPELINES[pipeline_name]),
        }
    
    @staticmethod
    def list_pipelines() -> List[str]:
        """List all available pipelines."""
        return list(PipelineUtils.PIPELINES.keys())


