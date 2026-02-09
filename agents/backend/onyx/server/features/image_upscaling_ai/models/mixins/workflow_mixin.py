"""
Workflow Mixin

Contains workflow orchestration and automation functionality.
"""

import logging
from typing import Union, Dict, Any, List, Optional, Callable
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)


class WorkflowMixin:
    """
    Mixin providing workflow orchestration functionality.
    
    This mixin contains:
    - Workflow creation
    - Workflow execution
    - Workflow scheduling
    - Workflow monitoring
    - Automated workflows
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize workflow mixin."""
        super().__init__(*args, **kwargs)
        if not hasattr(self, '_workflows'):
            self._workflows = {}
    
    def create_workflow(
        self,
        workflow_name: str,
        steps: List[Dict[str, Any]],
        description: str = ""
    ) -> bool:
        """
        Create a new workflow.
        
        Args:
            workflow_name: Name of the workflow
            steps: List of workflow steps
            description: Optional description
            
        Returns:
            True if successful
        """
        if not hasattr(self, '_workflows'):
            self._workflows = {}
        
        self._workflows[workflow_name] = {
            "steps": steps,
            "description": description,
            "created_at": str(Path().cwd()),  # Placeholder
        }
        
        logger.info(f"Workflow '{workflow_name}' created with {len(steps)} steps")
        return True
    
    def execute_workflow(
        self,
        workflow_name: str,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Execute a workflow.
        
        Args:
            workflow_name: Name of workflow to execute
            image: Input image
            scale_factor: Scale factor
            progress_callback: Optional progress callback
            
        Returns:
            Dictionary with execution results
        """
        if not hasattr(self, '_workflows') or workflow_name not in self._workflows:
            raise ValueError(f"Workflow '{workflow_name}' not found")
        
        workflow = self._workflows[workflow_name]
        steps = workflow["steps"]
        
        if isinstance(image, (str, Path)):
            current_image = Image.open(image).convert("RGB")
        else:
            current_image = image.convert("RGB")
        
        results = []
        total_steps = len(steps)
        
        for i, step in enumerate(steps):
            if progress_callback:
                progress_callback(i, total_steps)
            
            step_type = step.get("type")
            step_params = step.get("params", {})
            
            try:
                if step_type == "upscale":
                    method = step_params.get("method", "lanczos")
                    current_image = self.upscale(
                        current_image, scale_factor, method, return_metrics=False
                    )
                elif step_type == "enhance":
                    enhancement = step_params.get("enhancement", "edges")
                    if enhancement == "edges":
                        current_image = self.enhance_edges(
                            current_image, strength=step_params.get("strength", 1.2)
                        )
                    elif enhancement == "contrast":
                        current_image = self.adaptive_contrast_enhancement(current_image)
                elif step_type == "validate":
                    validation = self.validate_image(current_image)
                    if not validation.get("is_valid", True):
                        raise ValueError(f"Validation failed: {validation.get('errors', [])}")
                elif step_type == "compress":
                    current_image = self.compress_image(
                        current_image, quality=step_params.get("quality", 85)
                    )
                
                results.append({
                    "step": i + 1,
                    "type": step_type,
                    "success": True,
                })
            except Exception as e:
                results.append({
                    "step": i + 1,
                    "type": step_type,
                    "success": False,
                    "error": str(e),
                })
                logger.error(f"Workflow step {i+1} failed: {e}")
                break
        
        if progress_callback:
            progress_callback(total_steps, total_steps)
        
        return {
            "workflow": workflow_name,
            "success": all(r["success"] for r in results),
            "steps_executed": len(results),
            "total_steps": total_steps,
            "results": results,
            "final_image": current_image,
        }
    
    def list_workflows(self) -> List[str]:
        """List available workflows."""
        if not hasattr(self, '_workflows'):
            return []
        return list(self._workflows.keys())
    
    def get_workflow_info(self, workflow_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a workflow."""
        if not hasattr(self, '_workflows') or workflow_name not in self._workflows:
            return None
        return self._workflows[workflow_name]
    
    def delete_workflow(self, workflow_name: str) -> bool:
        """Delete a workflow."""
        if not hasattr(self, '_workflows') or workflow_name not in self._workflows:
            return False
        
        del self._workflows[workflow_name]
        logger.info(f"Workflow '{workflow_name}' deleted")
        return True
    
    def create_standard_workflow(
        self,
        workflow_type: str = "quality"
    ) -> str:
        """
        Create a standard workflow.
        
        Args:
            workflow_type: Type of workflow ('quality', 'speed', 'balanced')
            
        Returns:
            Workflow name
        """
        if workflow_type == "quality":
            steps = [
                {"type": "upscale", "params": {"method": "real_esrgan_like"}},
                {"type": "enhance", "params": {"enhancement": "edges", "strength": 1.2}},
                {"type": "enhance", "params": {"enhancement": "contrast"}},
                {"type": "validate", "params": {}},
            ]
            name = "quality_workflow"
        elif workflow_type == "speed":
            steps = [
                {"type": "upscale", "params": {"method": "lanczos"}},
            ]
            name = "speed_workflow"
        else:  # balanced
            steps = [
                {"type": "upscale", "params": {"method": "real_esrgan_like"}},
                {"type": "enhance", "params": {"enhancement": "edges", "strength": 1.1}},
            ]
            name = "balanced_workflow"
        
        self.create_workflow(name, steps, f"Standard {workflow_type} workflow")
        return name


