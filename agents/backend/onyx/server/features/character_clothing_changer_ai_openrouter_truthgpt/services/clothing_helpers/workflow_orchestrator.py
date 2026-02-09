"""
Workflow Orchestrator
=====================
Orchestrates the complete clothing change workflow
"""

import logging
from typing import Dict, Any, Optional

from services.comfyui_service import ComfyUIService

logger = logging.getLogger(__name__)


class WorkflowOrchestrator:
    """
    Orchestrates the complete clothing change workflow.
    """
    
    def __init__(self, comfyui_service: ComfyUIService):
        """
        Initialize workflow orchestrator.
        
        Args:
            comfyui_service: ComfyUI service instance
        """
        self.comfyui_service = comfyui_service
    
    async def execute_clothing_change(
        self,
        image_url: str,
        prompt: str,
        mask_url: Optional[str] = None,
        negative_prompt: str = "",
        guidance_scale: float = 50.0,
        num_steps: int = 12,
        seed: Optional[int] = None,
        wait_for_completion: bool = False,
        timeout: float = 300.0
    ) -> Dict[str, Any]:
        """
        Execute clothing change workflow.
        
        Args:
            image_url: URL or path to input image
            prompt: Optimized prompt
            mask_url: Optional mask image URL
            negative_prompt: Negative prompt
            guidance_scale: Guidance scale
            num_steps: Number of inference steps
            seed: Random seed
            wait_for_completion: Whether to wait for completion
            timeout: Timeout for waiting
            
        Returns:
            Workflow execution result
        """
        try:
            # Prepare workflow
            workflow = self.comfyui_service.prepare_workflow(
                image_url=image_url,
                mask_url=mask_url,
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_steps=num_steps,
                seed=seed
            )
            
            # Queue prompt
            queue_result = await self.comfyui_service.queue_prompt(workflow)
            prompt_id = queue_result.get("prompt_id")
            
            if not prompt_id:
                return {
                    "success": False,
                    "error": "Failed to queue prompt",
                    "queue_result": queue_result
                }
            
            result = {
                "success": True,
                "prompt_id": prompt_id,
                "status": "queued",
                "workflow_status": "queued"
            }
            
            # Wait for completion if requested
            if wait_for_completion:
                completion_result = await self.comfyui_service.wait_for_completion(
                    prompt_id, timeout
                )
                result.update(completion_result)
                
                if completion_result.get("status") == "completed":
                    result["success"] = True
                    result["workflow_status"] = "completed"
                    result["images"] = completion_result.get("images", [])
                else:
                    result["success"] = False
                    result["workflow_status"] = completion_result.get("status", "failed")
                    result["error"] = completion_result.get("error") or completion_result.get("message")
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing workflow: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "workflow_status": "failed"
            }
    
    async def execute_face_swap(
        self,
        image_url: str,
        face_url: str,
        prompt: str = "best quality",
        negative_prompt: str = "",
        guidance_scale: float = 50.0,
        num_steps: int = 12,
        seed: Optional[int] = None,
        wait_for_completion: bool = False,
        timeout: float = 300.0
    ) -> Dict[str, Any]:
        """
        Execute face swap workflow.
        
        Args:
            image_url: URL or path to input image
            face_url: URL or path to face image
            prompt: Prompt text
            negative_prompt: Negative prompt
            guidance_scale: Guidance scale
            num_steps: Number of inference steps
            seed: Random seed
            wait_for_completion: Whether to wait for completion
            timeout: Timeout for waiting
            
        Returns:
            Workflow execution result
        """
        try:
            # Prepare workflow with face URL
            workflow = self.comfyui_service.prepare_workflow(
                image_url=image_url,
                face_url=face_url,
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_steps=num_steps,
                seed=seed
            )
            
            # Queue prompt
            queue_result = await self.comfyui_service.queue_prompt(workflow)
            prompt_id = queue_result.get("prompt_id")
            
            if not prompt_id:
                return {
                    "success": False,
                    "error": "Failed to queue prompt",
                    "queue_result": queue_result
                }
            
            result = {
                "success": True,
                "prompt_id": prompt_id,
                "status": "queued",
                "workflow_status": "queued"
            }
            
            # Wait for completion if requested
            if wait_for_completion:
                completion_result = await self.comfyui_service.wait_for_completion(
                    prompt_id, timeout
                )
                result.update(completion_result)
                
                if completion_result.get("status") == "completed":
                    result["success"] = True
                    result["workflow_status"] = "completed"
                    result["images"] = completion_result.get("images", [])
                else:
                    result["success"] = False
                    result["workflow_status"] = completion_result.get("status", "failed")
                    result["error"] = completion_result.get("error") or completion_result.get("message")
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing face swap workflow: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "workflow_status": "failed"
            }

