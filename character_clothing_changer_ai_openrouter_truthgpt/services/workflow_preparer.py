"""
Workflow Preparer
=================

Prepares workflows with parameters, validation, and node updates.
"""

import logging
import copy
import random
from typing import Dict, Any, Optional

from .workflow_template_manager import WorkflowTemplateManager
from .workflow_node_manager import WorkflowNodeManager

logger = logging.getLogger(__name__)


class WorkflowPreparer:
    """Prepares workflows with parameters."""
    
    def __init__(
        self,
        template_manager: WorkflowTemplateManager,
        node_manager: WorkflowNodeManager,
        required_node_ids: list
    ):
        """
        Initialize Workflow Preparer.
        
        Args:
            template_manager: Workflow template manager
            node_manager: Workflow node manager
            required_node_ids: List of required node IDs
        """
        self.template_manager = template_manager
        self.node_manager = node_manager
        self.required_node_ids = required_node_ids
    
    def validate_parameters(
        self,
        image_url: str,
        prompt: str,
        guidance_scale: float,
        num_steps: int
    ) -> None:
        """
        Validate workflow parameters.
        
        Args:
            image_url: Image URL to validate
            prompt: Prompt to validate
            guidance_scale: Guidance scale to validate
            num_steps: Number of steps to validate
            
        Raises:
            ValueError: If any parameter is invalid
        """
        if not image_url or not image_url.strip():
            raise ValueError("image_url is required and cannot be empty")
        
        if not prompt or not prompt.strip():
            raise ValueError("prompt is required and cannot be empty")
        
        if not (1.0 <= guidance_scale <= 100.0):
            raise ValueError(
                f"guidance_scale must be between 1.0 and 100.0, got {guidance_scale}"
            )
        
        if not (1 <= num_steps <= 100):
            raise ValueError(
                f"num_steps must be between 1 and 100, got {num_steps}"
            )
    
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
            guidance_scale: Guidance scale
            num_steps: Number of inference steps
            seed: Random seed (optional)
            face_url: URL or path to face image (optional)
            
        Returns:
            Prepared workflow dictionary ready for ComfyUI
            
        Raises:
            ValueError: If required parameters are invalid
        """
        # Validate inputs
        self.validate_parameters(image_url, prompt, guidance_scale, num_steps)
        
        # Deep copy workflow to avoid modifying template
        workflow = copy.deepcopy(self.template_manager.load_template())
        
        # Validate workflow structure
        is_valid, error_msg = self.template_manager.validate_structure(
            workflow,
            self.required_node_ids
        )
        if not is_valid:
            raise ValueError(f"Invalid workflow structure: {error_msg}")
        
        # Generate seed if not provided
        if seed is None:
            seed = random.randint(1, 2**31 - 1)
            logger.debug(f"Generated random seed: {seed}")
        
        # Update workflow nodes
        self.node_manager.update_image_node(workflow, image_url)
        self.node_manager.update_prompt_node(workflow, prompt)
        self.node_manager.update_sampler_node(workflow, seed, num_steps)
        self.node_manager.update_guidance_node(workflow, guidance_scale)
        
        if negative_prompt:
            self._update_negative_prompt_node(workflow, negative_prompt)
        
        if mask_url:
            self._update_mask_node(workflow, mask_url)
        
        if face_url:
            self.node_manager.update_face_node(workflow, face_url)
        
        logger.info(
            f"Workflow prepared: image={image_url[:50]}..., "
            f"prompt={prompt[:50]}..., steps={num_steps}, "
            f"guidance={guidance_scale}, seed={seed}, "
            f"face_swap={face_url is not None}, mask={mask_url is not None}"
        )
        
        return workflow
    
    def _update_negative_prompt_node(
        self,
        workflow: Dict[str, Any],
        negative_prompt: str
    ) -> bool:
        """
        Update negative prompt node if available.
        
        Args:
            workflow: Workflow dictionary
            negative_prompt: Negative prompt text
            
        Returns:
            True if update was successful
        """
        # Try to find ConditioningZeroOut node
        node = self.node_manager.find_node(
            workflow,
            self.node_manager.node_ids.get("CONDITIONING_ZERO_OUT", -1)
        )
        if node:
            return self.node_manager.update_node_widget(node, 0, negative_prompt)
        return False
    
    def _update_mask_node(
        self,
        workflow: Dict[str, Any],
        mask_url: str
    ) -> bool:
        """
        Update mask node if mask URL is provided.
        
        Args:
            workflow: Workflow dictionary
            mask_url: URL or path to mask image
            
        Returns:
            True if mask was updated, False otherwise
        """
        if not mask_url:
            return False
        
        # The mask is typically loaded with the main image in LoadImage node
        load_image_node = self.node_manager.find_node(
            workflow,
            self.node_manager.node_ids["LOAD_IMAGE"]
        )
        if load_image_node:
            logger.debug(f"Mask URL provided: {mask_url[:50]}...")
            return True
        
        return False

