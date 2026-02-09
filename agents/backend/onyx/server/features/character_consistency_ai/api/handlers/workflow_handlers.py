"""
Workflow Handlers
=================

Request handlers for workflow-related endpoints.
"""

import logging
from typing import Optional, Dict, Any
from fastapi.responses import JSONResponse

from ..utils.error_handlers import handle_api_error

logger = logging.getLogger(__name__)


class WorkflowHandlers:
    """Handlers for workflow-related operations."""
    
    def __init__(self, service):
        """
        Initialize handlers.
        
        Args:
            service: CharacterConsistencyService instance
        """
        self.service = service
    
    async def create_workflow_tensor(
        self,
        embedding_path: str,
        prompt_template: str,
        negative_prompt: Optional[str],
        num_inference_steps: int,
        guidance_scale: float,
    ) -> JSONResponse:
        """
        Handle create workflow tensor request.
        
        Args:
            embedding_path: Path to embedding
            prompt_template: Prompt template
            negative_prompt: Optional negative prompt
            num_inference_steps: Inference steps
            guidance_scale: Guidance scale
            
        Returns:
            JSONResponse with workflow tensor info
        """
        try:
            result = self.service.create_workflow_tensor(
                embedding_path=embedding_path,
                prompt_template=prompt_template,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
            
            return JSONResponse(content=result)
        
        except Exception as e:
            raise handle_api_error("create_workflow_tensor", e)

