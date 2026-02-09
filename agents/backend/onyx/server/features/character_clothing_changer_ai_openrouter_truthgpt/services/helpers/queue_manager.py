"""
Queue Manager
=============
Manages ComfyUI queue operations
"""

import logging
import httpx
from typing import Dict, Any, Optional
from .http_client_manager import HTTPClientManager

logger = logging.getLogger(__name__)

# Constants
DEFAULT_CLIENT_ID = "character_clothing_changer"
MAX_HISTORY_ITEMS = 100
MIN_HISTORY_ITEMS = 1
DEFAULT_HISTORY_ITEMS = 10


class QueueManager:
    """
    Manages ComfyUI queue operations.
    """
    
    def __init__(self, api_url: str, http_client_manager: HTTPClientManager):
        """
        Initialize queue manager.
        
        Args:
            api_url: ComfyUI API URL
            http_client_manager: HTTP client manager instance
        """
        self.api_url = api_url.rstrip('/')
        self.http_client_manager = http_client_manager
    
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
            
        Raises:
            ValueError: If workflow is invalid
            httpx.HTTPStatusError: If API request fails
        """
        if not workflow:
            raise ValueError("Workflow cannot be empty")
        
        client = await self.http_client_manager.get_client()
        payload = {
            "prompt": workflow,
            "client_id": client_id or DEFAULT_CLIENT_ID
        }
        
        logger.debug("Queueing prompt in ComfyUI")
        
        response = await client.post(
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
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """
        Get current queue status.
        
        Returns:
            Dictionary with queue information
        """
        client = await self.http_client_manager.get_client()
        
        response = await client.get(f"{self.api_url}/queue")
        response.raise_for_status()
        return response.json()
    
    async def get_history(
        self,
        max_items: int = DEFAULT_HISTORY_ITEMS
    ) -> Dict[str, Any]:
        """
        Get workflow execution history.
        
        Args:
            max_items: Maximum number of history items to retrieve
            
        Returns:
            Dictionary with history information
        """
        if not (MIN_HISTORY_ITEMS <= max_items <= MAX_HISTORY_ITEMS):
            max_items = DEFAULT_HISTORY_ITEMS
        
        client = await self.http_client_manager.get_client()
        
        response = await client.get(
            f"{self.api_url}/history/{max_items}"
        )
        response.raise_for_status()
        return response.json()
    
    async def get_prompt_status(self, prompt_id: str) -> Dict[str, Any]:
        """
        Get status of a specific prompt.
        
        Args:
            prompt_id: Prompt ID to check
            
        Returns:
            Dictionary with prompt status
        """
        client = await self.http_client_manager.get_client()
        
        # Check queue first
        queue_status = await self.get_queue_status()
        
        # Check running queue
        running = queue_status.get("queue_running", [])
        for item in running:
            if item[1] == prompt_id:
                return {
                    "status": "processing",
                    "prompt_id": prompt_id,
                    "position": 0
                }
        
        # Check pending queue
        pending = queue_status.get("queue_pending", [])
        for i, item in enumerate(pending):
            if item[1] == prompt_id:
                return {
                    "status": "queued",
                    "prompt_id": prompt_id,
                    "position": i + 1
                }
        
        # Check history
        history = await self.get_history(MAX_HISTORY_ITEMS)
        for prompt_id_key, prompt_data in history.items():
            if prompt_id_key == prompt_id:
                return {
                    "status": "completed",
                    "prompt_id": prompt_id,
                    "data": prompt_data
                }
        
        return {
            "status": "unknown",
            "prompt_id": prompt_id
        }
    
    async def cancel_prompt(self, prompt_id: str) -> Dict[str, Any]:
        """
        Cancel a queued prompt.
        
        Args:
            prompt_id: Prompt ID to cancel
            
        Returns:
            Cancellation result
        """
        client = await self.http_client_manager.get_client()
        
        response = await client.post(
            f"{self.api_url}/queue",
            json={"delete": [prompt_id]}
        )
        response.raise_for_status()
        return response.json()

