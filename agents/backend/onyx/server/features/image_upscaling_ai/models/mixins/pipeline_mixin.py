"""
Pipeline Mixin

Contains pipeline and workflow management methods.
"""

import logging
from typing import Union, Dict, Any, List, Optional
from pathlib import Path
from PIL import Image

from ..helpers import (
    UpscalingMetrics,
    PipelineUtils,
)

logger = logging.getLogger(__name__)


class PipelineMixin:
    """
    Mixin providing pipeline and workflow functionality.
    
    This mixin contains:
    - Pipeline creation and execution
    - Workflow management
    - Custom pipeline support
    - Pipeline comparison
    """
    
    def create_workflow_preset(
        self,
        workflow_name: str,
        steps: List[Dict[str, Any]],
        description: str = ""
    ) -> Dict[str, Any]:
        """Create a workflow preset."""
        return PipelineUtils.create_workflow_preset(
            workflow_name=workflow_name,
            steps=steps,
            description=description
        )
    
    def upscale_with_workflow(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        workflow_name: str,
        return_metrics: bool = False
    ) -> Union[Image.Image, Tuple[Image.Image, UpscalingMetrics]]:
        """
        Upscale using a workflow preset.
        
        This method should be implemented in the main class.
        """
        raise NotImplementedError("This method should be implemented in the main class")
    
    def list_workflows(self) -> List[str]:
        """List available workflows."""
        return PipelineUtils.list_workflows()
    
    def get_workflow_info(self, workflow_name: str) -> Dict[str, Any]:
        """Get information about a workflow."""
        return PipelineUtils.get_workflow_info(workflow_name)


