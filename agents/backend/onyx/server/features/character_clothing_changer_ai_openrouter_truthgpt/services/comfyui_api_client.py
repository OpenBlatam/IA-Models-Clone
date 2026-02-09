"""
ComfyUI API Client
==================

Handles direct API interactions with ComfyUI including queueing,
history retrieval, and status checking.
"""

import logging
from typing import Dict, Any, Optional, List
import httpx

from .retry_handler import RetryHandler

logger = logging.getLogger(__name__)

DEFAULT_CLIENT_ID = "character_clothing_changer"
DEFAULT_HISTORY_ITEMS = 10
MIN_HISTORY_ITEMS = 1
MAX_HISTORY_ITEMS = 100


class ComfyUIAPIClient:
    """Handles ComfyUI API interactions."""
    
    def __init__(
        self,
        api_url: str,
        http_client: httpx.AsyncClient,
        retry_handler: RetryHandler
    ):
        """
        Initialize ComfyUI API Client.
        
        Args:
            api_url: ComfyUI API URL
            http_client: HTTP client instance
            retry_handler: Retry handler instance
        """
        self.api_url = api_url.rstrip('/')
        self.http_client = http_client
        self.retry_handler = retry_handler
    
    async def queue_prompt(
        self,
        workflow: Dict[str, Any],
        client_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Queue a prompt in ComfyUI.
        
        Args:
            workflow: Workflow dictionary to queue
            client_id: Optional client ID for tracking
            
        Returns:
            Dictionary with prompt_id and execution details
        """
        if not workflow:
            raise ValueError("Workflow cannot be empty")
        
        payload = {
            "prompt": workflow,
            "client_id": client_id or DEFAULT_CLIENT_ID
        }
        
        async def _queue():
            response = await self.http_client.post(
                f"{self.api_url}/prompt",
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            
            prompt_id = result.get("prompt_id")
            if prompt_id:
                logger.info(f"Prompt queued successfully: {prompt_id}")
            else:
                logger.warning("ComfyUI response missing prompt_id")
            
            # Check for node errors
            node_errors = result.get("node_errors", [])
            if node_errors:
                logger.warning(f"Workflow has node errors: {node_errors}")
            
            return result
        
        return await self.retry_handler.execute_with_retry(_queue)
    
    async def get_history(self, max_items: int = DEFAULT_HISTORY_ITEMS) -> Dict[str, Any]:
        """
        Get workflow execution history from ComfyUI.
        
        Args:
            max_items: Maximum number of history items to retrieve (1-100)
            
        Returns:
            Dictionary with history data or empty dict on error
        """
        max_items = max(MIN_HISTORY_ITEMS, min(max_items, MAX_HISTORY_ITEMS))
        
        try:
            response = await self.http_client.get(
                f"{self.api_url}/history/{max_items}",
                timeout=30.0
            )
            response.raise_for_status()
            history = response.json()
            logger.debug(f"Retrieved {len(history)} history items")
            return history
        except Exception as e:
            logger.error(f"Error getting ComfyUI history: {e}")
            return {}
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """
        Get current queue status from ComfyUI.
        
        Returns:
            Dictionary with queue status
        """
        try:
            response = await self.http_client.get(
                f"{self.api_url}/queue",
                timeout=30.0
            )
            response.raise_for_status()
            status = response.json()
            
            running_count = len(status.get("queue_running", []))
            pending_count = len(status.get("queue_pending", []))
            logger.debug(f"Queue status: {running_count} running, {pending_count} pending")
            
            return status
        except Exception as e:
            logger.error(f"Error getting ComfyUI queue status: {e}")
            return {"queue_running": [], "queue_pending": []}
    
    async def cancel_prompt(self, prompt_id: str) -> Dict[str, Any]:
        """
        Cancel a queued prompt.
        
        Args:
            prompt_id: Prompt ID to cancel
            
        Returns:
            Cancellation result
        """
        try:
            response = await self.http_client.post(
                f"{self.api_url}/queue",
                json={"delete": [prompt_id]}
            )
            response.raise_for_status()
            logger.info(f"Prompt {prompt_id} cancelled")
            return {"cancelled": True, "prompt_id": prompt_id}
        except Exception as e:
            logger.error(f"Error cancelling prompt {prompt_id}: {e}")
            return {"cancelled": False, "error": str(e)}
    
    async def get_output_images(self, prompt_id: str) -> List[Dict[str, Any]]:
        """
        Get output images for a completed prompt.
        
        Args:
            prompt_id: Prompt ID
            
        Returns:
            List of output image dictionaries
        """
        try:
            history = await self.get_history(max_items=50)
            for prompt_data in history.values():
                if prompt_data.get("prompt_id") == prompt_id:
                    outputs = prompt_data.get("outputs", {})
                    images = []
                    for node_id, node_outputs in outputs.items():
                        if "images" in node_outputs:
                            for img in node_outputs["images"]:
                                images.append({
                                    "node_id": node_id,
                                    "filename": img.get("filename"),
                                    "subfolder": img.get("subfolder"),
                                    "type": img.get("type")
                                })
                    return images
            return []
        except Exception as e:
            logger.error(f"Error getting output images for {prompt_id}: {e}")
            return []

