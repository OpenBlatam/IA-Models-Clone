"""
Response Builder
================
Builds response dictionaries for clothing change operations
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ResponseBuilder:
    """
    Builds response dictionaries for clothing change operations.
    """
    
    @staticmethod
    def build_success_response(
        prompt_id: str,
        original_prompt: str,
        optimized_prompt: str,
        openrouter_used: bool,
        truthgpt_used: bool,
        workflow_status: str,
        settings: Optional[Dict[str, Any]] = None,
        images: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Build success response dictionary.
        
        Args:
            prompt_id: ComfyUI prompt ID
            original_prompt: Original user prompt
            optimized_prompt: Final optimized prompt
            openrouter_used: Whether OpenRouter was used
            truthgpt_used: Whether TruthGPT was used
            workflow_status: ComfyUI workflow status
            settings: Optional generation settings
            images: Optional list of result images
            
        Returns:
            Success response dictionary
        """
        response = {
            "success": True,
            "prompt_id": prompt_id,
            "original_prompt": original_prompt,
            "optimized_prompt": optimized_prompt,
            "openrouter_used": openrouter_used,
            "truthgpt_used": truthgpt_used,
            "workflow_status": workflow_status
        }
        
        if settings:
            response["settings"] = settings
        
        if images:
            response["images"] = images
        
        return response
    
    @staticmethod
    def build_error_response(
        error: str,
        original_prompt: Optional[str] = None,
        optimized_prompt: Optional[str] = None,
        openrouter_used: bool = False,
        truthgpt_used: bool = False
    ) -> Dict[str, Any]:
        """
        Build error response dictionary.
        
        Args:
            error: Error message
            original_prompt: Optional original prompt
            optimized_prompt: Optional optimized prompt
            openrouter_used: Whether OpenRouter was used
            truthgpt_used: Whether TruthGPT was used
            
        Returns:
            Error response dictionary
        """
        response = {
            "success": False,
            "error": error,
            "openrouter_used": openrouter_used,
            "truthgpt_used": truthgpt_used
        }
        
        if original_prompt:
            response["original_prompt"] = original_prompt
        
        if optimized_prompt:
            response["optimized_prompt"] = optimized_prompt
        
        return response
    
    @staticmethod
    def build_status_response(
        prompt_id: str,
        status: str,
        workflow_status: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Build status response dictionary.
        
        Args:
            prompt_id: ComfyUI prompt ID
            status: Current status
            workflow_status: Optional workflow status
            additional_data: Optional additional data
            
        Returns:
            Status response dictionary
        """
        response = {
            "prompt_id": prompt_id,
            "status": status
        }
        
        if workflow_status:
            response["workflow_status"] = workflow_status
        
        if additional_data:
            response.update(additional_data)
        
        return response

