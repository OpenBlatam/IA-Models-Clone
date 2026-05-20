"""
Workflow Preparer
================
Prepares workflows with parameters for execution
"""

import copy
import logging
import random
from typing import Dict, Any, Optional

from .workflow_manager import WorkflowManager
from .workflow_nodes import WorkflowNodeManager
from .workflow_validator import WorkflowValidator

logger = logging.getLogger(__name__)


class WorkflowPreparer:
    """
    Prepares workflows with parameters for execution.
    """
    
    def __init__(
        self,
        workflow_manager: WorkflowManager,
        node_manager: WorkflowNodeManager,
        validator: WorkflowValidator
    ):
        """
        Initialize workflow preparer.
        
        Args:
            workflow_manager: Workflow manager instance
            node_manager: Node manager instance
            validator: Workflow validator instance
        """
        self.workflow_manager = workflow_manager
        self.node_manager = node_manager
        self.validator = validator
    
    def prepare_workflow(
        self,
        image_url: str,
        mask_url: Optional[str] = None,
        prompt: str = "best quality",
        negative_prompt: str = "",
        guidance_scale: float = 50.0,
        num_steps: int = 12,
        seed: Optional[int] = None,
        face_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Prepare workflow with parameters.
        
        Args:
            image_url: URL or path to input image
            mask_url: URL or path to mask image (optional)
            prompt: Positive prompt
            negative_prompt: Negative prompt
            guidance_scale: Guidance scale (default 50 for Flux)
            num_steps: Number of inference steps
            seed: Random seed (optional, will generate if not provided)
            face_url: URL or path to face image for face swap (optional)
            
        Returns:
            Prepared workflow dictionary ready for ComfyUI
            
        Raises:
            ValueError: If required parameters are invalid
        """
        # Validate inputs
        self.validator.validate_parameters(
            image_url, prompt, guidance_scale, num_steps
        )
        
        # Deep copy workflow to avoid modifying template
        workflow = copy.deepcopy(self.workflow_manager.get_template())
        
        # Validate workflow structure
        self.validator.validate_workflow(workflow)
        
        # Generate seed if not provided
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        
        # Update nodes with parameters
        self._update_image_node(workflow, image_url)
        self._update_prompt_node(workflow, prompt)
        self._update_negative_prompt_node(workflow, negative_prompt)
        self._update_guidance_node(workflow, guidance_scale)
        self._update_sampler_node(workflow, num_steps, seed)
        
        if mask_url:
            self._update_mask_node(workflow, mask_url)
        
        if face_url:
            self._update_face_node(workflow, face_url)
        
        return workflow
    
    def _update_image_node(self, workflow: Dict[str, Any], image_url: str) -> None:
        """Update image load node"""
        self.node_manager.update_node_by_name(
            workflow, "LOAD_IMAGE", 0, image_url
        )
    
    def _update_prompt_node(self, workflow: Dict[str, Any], prompt: str) -> None:
        """Update prompt encoding node"""
        self.node_manager.update_node_by_name(
            workflow, "CLIP_TEXT_ENCODE", 0, prompt
        )
    
    def _update_negative_prompt_node(
        self,
        workflow: Dict[str, Any],
        negative_prompt: str
    ) -> None:
        """Update negative prompt node"""
        if negative_prompt:
            self.node_manager.update_node_by_name(
                workflow, "CONDITIONING_ZERO_OUT", 0, negative_prompt
            )
    
    def _update_guidance_node(
        self,
        workflow: Dict[str, Any],
        guidance_scale: float
    ) -> None:
        """Update guidance scale node"""
        self.node_manager.update_node_by_name(
            workflow, "FLUX_GUIDANCE", 0, guidance_scale
        )
    
    def _update_sampler_node(
        self,
        workflow: Dict[str, Any],
        num_steps: int,
        seed: int
    ) -> None:
        """Update sampler node"""
        # Update steps (typically index 0)
        self.node_manager.update_node_by_name(
            workflow, "KSAMPLER", 0, num_steps
        )
        # Update seed (typically index 1)
        self.node_manager.update_node_by_name(
            workflow, "KSAMPLER", 1, seed
        )
    
    def _update_mask_node(self, workflow: Dict[str, Any], mask_url: str) -> None:
        """Update mask node if mask is provided"""
        # This would depend on your specific workflow structure
        logger.debug(f"Mask URL provided: {mask_url}")
        # Add mask node update logic here based on your workflow
    
    def _update_face_node(self, workflow: Dict[str, Any], face_url: str) -> None:
        """Update face load node for face swap"""
        self.node_manager.update_node_by_name(
            workflow, "LOAD_NEW_FACE", 0, face_url
        )

