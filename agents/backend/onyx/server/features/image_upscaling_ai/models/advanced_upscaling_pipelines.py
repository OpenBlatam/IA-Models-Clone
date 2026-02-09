"""
Advanced Upscaling Pipelines and Workflows
==========================================

Pipeline and workflow management for upscaling operations.
"""

import logging
import time
from typing import Tuple, Optional, Dict, Any, List, Union, Callable
from pathlib import Path
from PIL import Image

from .helpers import (
    UpscalingMetrics,
    PipelineUtils,
    ConfigUtils,
)

logger = logging.getLogger(__name__)


class PipelineMethods:
    """Pipeline and workflow management methods."""
    
    def __init__(self, base_upscaler):
        """Initialize with base upscaler."""
        self.base_upscaler = base_upscaler
        self._custom_pipelines = {}
        self._workflows = {}
    
    def upscale_with_pipeline(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        pipeline: str = "standard",
        return_metrics: bool = False
    ) -> Union[Image.Image, Tuple[Image.Image, UpscalingMetrics]]:
        """Upscale using predefined processing pipelines."""
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        
        start_time = time.time()
        original_size = pil_image.size
        
        # Select pipeline
        if pipeline == "standard":
            result = self.base_upscaler.upscale_lanczos(pil_image, scale_factor)
            result = self.base_upscaler.enhance_edges(result, strength=1.1)
        elif pipeline == "quality":
            result = self.base_upscaler.multi_scale_upscale(pil_image, scale_factor, passes=2)
            result = self.base_upscaler.reduce_artifacts(result, method="bilateral")
            result = self.base_upscaler.enhance_edges(result, strength=1.2)
        elif pipeline == "speed":
            result = self.base_upscaler.upscale_bicubic_enhanced(pil_image, scale_factor)
        elif pipeline == "balanced":
            result = self.base_upscaler.upscale_opencv_edsr(pil_image, scale_factor)
            result = self.base_upscaler.enhance_edges(result, strength=1.15)
        elif pipeline == "ultra_quality":
            result = self.base_upscaler.upscale(pil_image, scale_factor, "real_esrgan_like", return_metrics=False)
        elif pipeline in self._custom_pipelines:
            result = self._execute_custom_pipeline(pil_image, scale_factor, pipeline)
        else:
            result = self.base_upscaler.upscale_lanczos(pil_image, scale_factor)
        
        processing_time = time.time() - start_time
        quality_metrics = self.base_upscaler.calculate_quality_metrics(result)
        
        metrics = UpscalingMetrics(
            original_size=original_size,
            upscaled_size=result.size,
            scale_factor=scale_factor,
            processing_time=processing_time,
            quality_score=quality_metrics.overall_quality,
            method_used=f"pipeline_{pipeline}",
            success=True,
        )
        
        if return_metrics:
            return result, metrics
        return result
    
    def create_custom_pipeline(
        self,
        name: str,
        steps: List[Dict[str, Any]]
    ) -> None:
        """Create a custom upscaling pipeline."""
        self._custom_pipelines[name] = {
            "steps": steps,
            "created_at": time.time(),
        }
        logger.info(f"Custom pipeline '{name}' created with {len(steps)} steps")
    
    def _execute_custom_pipeline(
        self,
        image: Image.Image,
        scale_factor: float,
        pipeline_name: str
    ) -> Image.Image:
        """Execute a custom pipeline."""
        if pipeline_name not in self._custom_pipelines:
            raise ValueError(f"Pipeline '{pipeline_name}' not found")
        
        pipeline = self._custom_pipelines[pipeline_name]
        result = image
        
        for step in pipeline["steps"]:
            step_type = step.get("type")
            step_params = step.get("params", {})
            
            if step_type == "upscale":
                method = step_params.get("method", "lanczos")
                result = self.base_upscaler.upscale(result, scale_factor, method, return_metrics=False)
            elif step_type == "enhance_edges":
                strength = step_params.get("strength", 1.2)
                result = self.base_upscaler.enhance_edges(result, strength=strength)
            elif step_type == "reduce_artifacts":
                method = step_params.get("method", "bilateral")
                result = self.base_upscaler.reduce_artifacts(result, method=method)
            elif step_type == "apply_anti_aliasing":
                strength = step_params.get("strength", 0.5)
                result = self.base_upscaler.apply_anti_aliasing(result, strength=strength)
        
        return result
    
    def list_custom_pipelines(self) -> List[str]:
        """List all custom pipelines."""
        return list(self._custom_pipelines.keys())
    
    def get_pipeline_info(self, pipeline_name: str) -> Dict[str, Any]:
        """Get information about a custom pipeline."""
        if pipeline_name not in self._custom_pipelines:
            raise ValueError(f"Pipeline '{pipeline_name}' not found")
        
        return self._custom_pipelines[pipeline_name]
    
    def create_workflow_preset(
        self,
        name: str,
        workflow: Dict[str, Any]
    ) -> None:
        """Create a workflow preset."""
        self._workflows[name] = {
            **workflow,
            "created_at": time.time(),
        }
        logger.info(f"Workflow preset '{name}' created")
    
    def upscale_with_workflow(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        workflow_name: str,
        return_metrics: bool = False
    ) -> Union[Image.Image, Tuple[Image.Image, UpscalingMetrics]]:
        """Upscale using a workflow preset."""
        if workflow_name not in self._workflows:
            raise ValueError(f"Workflow '{workflow_name}' not found")
        
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        
        start_time = time.time()
        original_size = pil_image.size
        
        workflow = self._workflows[workflow_name]
        result = PipelineUtils.execute_workflow(
            pil_image,
            scale_factor,
            workflow,
            self.base_upscaler
        )
        
        processing_time = time.time() - start_time
        quality_metrics = self.base_upscaler.calculate_quality_metrics(result)
        
        metrics = UpscalingMetrics(
            original_size=original_size,
            upscaled_size=result.size,
            scale_factor=scale_factor,
            processing_time=processing_time,
            quality_score=quality_metrics.overall_quality,
            method_used=f"workflow_{workflow_name}",
            success=True,
        )
        
        if return_metrics:
            return result, metrics
        return result
    
    def list_workflows(self) -> List[str]:
        """List all workflow presets."""
        return list(self._workflows.keys())
    
    def get_workflow_info(self, workflow_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a workflow."""
        return self._workflows.get(workflow_name)
    
    def export_upscaling_config(
        self,
        output_path: str,
        include_pipelines: bool = True,
        include_workflows: bool = True
    ) -> Dict[str, Any]:
        """Export upscaling configuration."""
        config = {
            "version": "1.0",
            "exported_at": time.time(),
            "settings": {
                "enable_cache": self.base_upscaler.enable_cache,
                "cache_size": self.base_upscaler.cache.max_size if self.base_upscaler.cache else 0,
                "max_workers": self.base_upscaler.executor._max_workers,
            }
        }
        
        if include_pipelines:
            config["custom_pipelines"] = self._custom_pipelines
        
        if include_workflows:
            config["workflows"] = self._workflows
        
        import json
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Configuration exported to {output_path}")
        return config
    
    def load_and_apply_config(
        self,
        config_path: str
    ) -> None:
        """Load and apply configuration."""
        import json
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Apply settings
        if "settings" in config:
            settings = config["settings"]
            # Note: Some settings may require reinitialization
        
        # Load pipelines
        if "custom_pipelines" in config:
            self._custom_pipelines = config["custom_pipelines"]
            logger.info(f"Loaded {len(self._custom_pipelines)} custom pipelines")
        
        # Load workflows
        if "workflows" in config:
            self._workflows = config["workflows"]
            logger.info(f"Loaded {len(self._workflows)} workflows")
        
        logger.info(f"Configuration loaded from {config_path}")


