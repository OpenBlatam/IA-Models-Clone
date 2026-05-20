"""
Workflow Status Manager
=======================

Manages workflow status checking and completion waiting.
"""

import logging
import asyncio
import time
from typing import Dict, Any, Optional
from enum import Enum

from .comfyui_api_client import ComfyUIAPIClient

logger = logging.getLogger(__name__)


class WorkflowStatus(str, Enum):
    """Workflow execution status"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowStatusManager:
    """Manages workflow status operations."""
    
    def __init__(self, api_client: ComfyUIAPIClient):
        """
        Initialize Workflow Status Manager.
        
        Args:
            api_client: ComfyUI API client instance
        """
        self.api_client = api_client
    
    async def get_prompt_status(self, prompt_id: str) -> Dict[str, Any]:
        """
        Get status of a specific prompt.
        
        Args:
            prompt_id: The prompt ID to check
            
        Returns:
            Dictionary with prompt status information
        """
        try:
            # Check queue first
            queue_status = await self.api_client.get_queue_status()
            
            # Check if in running queue
            for item in queue_status.get("queue_running", []):
                if item.get("prompt_id") == prompt_id:
                    return {
                        "prompt_id": prompt_id,
                        "status": WorkflowStatus.PROCESSING.value,
                        "position": "running"
                    }
            
            # Check if in pending queue
            for idx, item in enumerate(queue_status.get("queue_pending", [])):
                if item.get("prompt_id") == prompt_id:
                    return {
                        "prompt_id": prompt_id,
                        "status": WorkflowStatus.QUEUED.value,
                        "position": idx + 1
                    }
            
            # Check history
            history = await self.api_client.get_history(max_items=50)
            for prompt_data in history.values():
                if prompt_data.get("prompt_id") == prompt_id:
                    return {
                        "prompt_id": prompt_id,
                        "status": WorkflowStatus.COMPLETED.value,
                        "position": "completed"
                    }
            
            return {
                "prompt_id": prompt_id,
                "status": "unknown",
                "message": "Prompt not found in queue or recent history"
            }
            
        except Exception as e:
            logger.error(f"Error getting prompt status: {e}")
            return {
                "prompt_id": prompt_id,
                "status": "error",
                "error": str(e)
            }
    
    async def wait_for_completion(
        self,
        prompt_id: str,
        timeout: float = 300.0,
        check_interval: float = 2.0
    ) -> Dict[str, Any]:
        """
        Wait for a prompt to complete.
        
        Args:
            prompt_id: Prompt ID to wait for
            timeout: Maximum time to wait in seconds
            check_interval: Interval between status checks in seconds
            
        Returns:
            Dictionary with completion status and results
        """
        start_time = time.time()
        elapsed_time = 0.0
        
        logger.info(f"Waiting for prompt {prompt_id} to complete (timeout: {timeout}s)")
        
        while elapsed_time < timeout:
            status = await self.get_prompt_status(prompt_id)
            current_status = status.get("status")
            
            if current_status == WorkflowStatus.COMPLETED.value:
                # Get output images
                images = await self.api_client.get_output_images(prompt_id)
                
                return {
                    "prompt_id": prompt_id,
                    "status": WorkflowStatus.COMPLETED.value,
                    "elapsed_time": elapsed_time,
                    "images": images,
                    "image_count": len(images)
                }
            
            elif current_status == WorkflowStatus.FAILED.value:
                return {
                    "prompt_id": prompt_id,
                    "status": WorkflowStatus.FAILED.value,
                    "elapsed_time": elapsed_time,
                    "error": status.get("error", "Unknown error")
                }
            
            # Wait before next check
            await asyncio.sleep(check_interval)
            elapsed_time = time.time() - start_time
        
        # Timeout
        logger.warning(f"Timeout waiting for prompt {prompt_id} after {elapsed_time}s")
        return {
            "prompt_id": prompt_id,
            "status": "timeout",
            "elapsed_time": elapsed_time,
            "message": f"Prompt did not complete within {timeout}s"
        }

