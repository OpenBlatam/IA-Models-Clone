"""
Workflow Validator
=================
Validates workflow structure and parameters
"""

import logging
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)


class WorkflowValidator:
    """
    Validates workflow structure and parameters.
    """
    
    @staticmethod
    def validate_workflow_structure(workflow: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate workflow structure.
        
        Args:
            workflow: Workflow dictionary to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(workflow, dict):
            return False, "Workflow must be a dictionary"
        
        if "nodes" not in workflow:
            return False, "Workflow missing 'nodes' key"
        
        if not isinstance(workflow["nodes"], list):
            return False, "Workflow 'nodes' must be a list"
        
        if len(workflow["nodes"]) == 0:
            return False, "Workflow must have at least one node"
        
        # Validate each node has required fields
        for i, node in enumerate(workflow["nodes"]):
            if not isinstance(node, dict):
                return False, f"Node {i} must be a dictionary"
            
            if "id" not in node:
                return False, f"Node {i} missing 'id' field"
            
            if not isinstance(node["id"], int):
                return False, f"Node {i} 'id' must be an integer"
        
        return True, None
    
    @staticmethod
    def validate_parameters(
        image_url: str,
        prompt: str,
        guidance_scale: float,
        num_steps: int
    ) -> None:
        """
        Validate workflow parameters.
        
        Args:
            image_url: Image URL or path
            prompt: Prompt text
            guidance_scale: Guidance scale value
            num_steps: Number of inference steps
            
        Raises:
            ValueError: If any parameter is invalid
        """
        if not image_url or not image_url.strip():
            raise ValueError("image_url is required")
        
        if not prompt or not prompt.strip():
            raise ValueError("prompt is required")
        
        if not (1.0 <= guidance_scale <= 100.0):
            raise ValueError(
                f"guidance_scale must be between 1.0 and 100.0, got {guidance_scale}"
            )
        
        if not (1 <= num_steps <= 100):
            raise ValueError(
                f"num_steps must be between 1 and 100, got {num_steps}"
            )
    
    @staticmethod
    def validate_workflow(workflow: Dict[str, Any]) -> None:
        """
        Validate workflow and raise exception if invalid.
        
        Args:
            workflow: Workflow dictionary to validate
            
        Raises:
            ValueError: If workflow is invalid
        """
        is_valid, error_msg = WorkflowValidator.validate_workflow_structure(workflow)
        if not is_valid:
            raise ValueError(f"Invalid workflow structure: {error_msg}")

