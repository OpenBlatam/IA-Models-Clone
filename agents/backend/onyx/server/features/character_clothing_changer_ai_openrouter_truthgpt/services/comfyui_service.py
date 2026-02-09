"""
ComfyUI Service for Character Clothing Changer
==============================================

Service for executing ComfyUI workflows for character clothing changes.

Features:
- Workflow template management and caching
- Node mapping and parameter updates
- Queue management and status tracking
- History retrieval
- Retry logic with exponential backoff
- Comprehensive error handling
"""

import json
import logging
import httpx
import asyncio
import copy
import random
import time
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

from config.settings import get_settings

logger = logging.getLogger(__name__)

# Constants
DEFAULT_CLIENT_ID = "character_clothing_changer"
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0
DEFAULT_TIMEOUT = 300.0
DEFAULT_CONNECT_TIMEOUT = 10.0
MAX_CONNECTIONS = 50
MAX_KEEPALIVE_CONNECTIONS = 10
KEEPALIVE_EXPIRY = 30.0
MAX_HISTORY_ITEMS = 100
MIN_HISTORY_ITEMS = 1
DEFAULT_HISTORY_ITEMS = 10


# WorkflowStatus is now imported from workflow_status_manager


@dataclass
class WorkflowNode:
    """Represents a workflow node"""
    node_id: int
    node_type: str
    position: List[float]
    widgets_values: List[Any]
    
    def update_widget(self, index: int, value: Any) -> bool:
        """Update widget value at index"""
        if 0 <= index < len(self.widgets_values):
            self.widgets_values[index] = value
            return True
        return False


# Node IDs from the workflow JSON
NODE_IDS = {
    "LOAD_IMAGE": 239,
    "LOAD_NEW_FACE": 240,
    "CLIP_TEXT_ENCODE": 343,
    "FLUX_GUIDANCE": 345,
    "KSAMPLER": 346,
    "VAE_DECODE": 214,
    "IMAGE_CROP": 228,
    "INPAINT_CROP": 411,
    "INPAINT_STITCH": 412,
    "CONDITIONING_ZERO_OUT": 404,  # For negative prompt
}


@dataclass
class ComfyUIConfig:
    """Configuration for ComfyUI service"""
    api_url: str
    workflow_path: str
    max_retries: int = DEFAULT_MAX_RETRIES
    retry_delay: float = DEFAULT_RETRY_DELAY
    timeout: float = DEFAULT_TIMEOUT
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT
    max_connections: int = MAX_CONNECTIONS
    max_keepalive: int = MAX_KEEPALIVE_CONNECTIONS
    keepalive_expiry: float = KEEPALIVE_EXPIRY


class ComfyUIService:
    """
    Service for interacting with ComfyUI API.
    
    This service handles:
    - Workflow template loading and caching
    - Node parameter updates
    - Prompt queueing with retry logic
    - Queue and history status checking
    - HTTP client management with connection pooling
    
    Attributes:
        config: ComfyUI configuration
        _workflow_template: Cached workflow template
        _http_client: HTTP client instance (lazy initialized)
        _client_lock: Lock for thread-safe client initialization
    """
    
    def __init__(self, config: Optional[ComfyUIConfig] = None):
        """
        Initialize ComfyUI service.
        
        Args:
            config: Optional configuration. If not provided, loads from settings.
        """
        self.settings = get_settings()
        self.config = config or ComfyUIConfig(
            api_url=self.settings.comfyui_api_url.rstrip('/'),
            workflow_path=self.settings.comfyui_workflow_path
        )
        
        # Initialize managers
        self.http_client_manager = HTTPClientManager(
            max_connections=self.config.max_connections,
            max_keepalive=self.config.max_keepalive,
            keepalive_expiry=self.config.keepalive_expiry,
            timeout=self.config.timeout,
            connect_timeout=self.config.connect_timeout
        )
        
        self.template_manager = WorkflowTemplateManager(self.config.workflow_path)
        
        self.node_manager = WorkflowNodeManager(NODE_IDS)
        
        self.retry_handler = RetryHandler(
            max_retries=self.config.max_retries,
            retry_delay=self.config.retry_delay
        )
        
        # Initialize API client (will be set after HTTP client is ready)
        self.api_client: Optional[ComfyUIAPIClient] = None
        self.status_manager: Optional[WorkflowStatusManager] = None
        
        self.workflow_preparer = WorkflowPreparer(
            template_manager=self.template_manager,
            node_manager=self.node_manager,
            required_node_ids=[
                NODE_IDS["LOAD_IMAGE"],
                NODE_IDS["CLIP_TEXT_ENCODE"],
                NODE_IDS["KSAMPLER"]
            ]
        )
    
    async def _get_api_client(self) -> ComfyUIAPIClient:
        """
        Get or create API client.
        
        Returns:
            ComfyUIAPIClient instance
        """
        if self.api_client is None:
            http_client = await self.http_client_manager.get_client()
            self.api_client = ComfyUIAPIClient(
                api_url=self.config.api_url,
                http_client=http_client,
                retry_handler=self.retry_handler
            )
            self.status_manager = WorkflowStatusManager(self.api_client)
        return self.api_client
    
    async def close(self) -> None:
        """Close HTTP client and cleanup resources"""
        await self.http_client_manager.close()
    
    
    async def queue_prompt(
        self,
        workflow: Dict[str, Any],
        client_id: Optional[str] = None,
        retry: bool = True
    ) -> Dict[str, Any]:
        """
        Queue a prompt in ComfyUI with retry logic and exponential backoff.
        
        Args:
            workflow: Workflow dictionary to queue
            client_id: Optional client ID for tracking
            retry: Whether to retry on failure (default: True)
            
        Returns:
            Dictionary with prompt_id and execution details
            
        Raises:
            ValueError: If workflow is invalid
            httpx.HTTPStatusError: If API request fails after retries
            httpx.TimeoutException: If request times out after retries
            Exception: For other unexpected errors
        """
        api_client = await self._get_api_client()
        return await api_client.queue_prompt(workflow, client_id)
    
    async def get_history(self, max_items: int = 10) -> Dict[str, Any]:
        """
        Get workflow execution history from ComfyUI.
        
        Args:
            max_items: Maximum number of history items to retrieve (1-100)
            
        Returns:
            Dictionary with history data or empty dict on error
        """
        api_client = await self._get_api_client()
        return await api_client.get_history(max_items)
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """
        Get current queue status from ComfyUI.
        
        Returns:
            Dictionary with queue status
        """
        api_client = await self._get_api_client()
        return await api_client.get_queue_status()
    
    async def get_prompt_status(self, prompt_id: str) -> Dict[str, Any]:
        """
        Get status of a specific prompt.
        
        Args:
            prompt_id: The prompt ID to check
            
        Returns:
            Dictionary with prompt status information
        """
        if self.status_manager is None:
            await self._get_api_client()
        return await self.status_manager.get_prompt_status(prompt_id)
    
    
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
            seed: Random seed (optional)
            face_url: URL or path to face image (optional)
            
        Returns:
            Prepared workflow dictionary ready for ComfyUI
            
        Raises:
            ValueError: If required parameters are invalid
        """
        return self.workflow_preparer.prepare_workflow(
            image_url=image_url,
            mask_url=mask_url,
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            seed=seed,
            face_url=face_url
        )
    
    def get_workflow_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded workflow template.
        
        Returns:
            Dictionary with workflow information
        """
        info = self.template_manager.get_info()
        
        # Check for face swap support
        workflow = self.template_manager.load_template()
        nodes = workflow.get("nodes", [])
        has_face_swap = any(
            node.get("id") == NODE_IDS["LOAD_NEW_FACE"]
            for node in nodes
        )
        
        info["has_face_swap"] = has_face_swap
        return info
    
    async def cancel_prompt(self, prompt_id: str) -> Dict[str, Any]:
        """
        Cancel a queued prompt.
        
        Args:
            prompt_id: Prompt ID to cancel
            
        Returns:
            Dictionary with cancellation result
        """
        if not prompt_id or not prompt_id.strip():
            raise ValueError("prompt_id is required")
        
        api_client = await self._get_api_client()
        return await api_client.cancel_prompt(prompt_id)
    
    async def get_output_images(self, prompt_id: str) -> List[Dict[str, Any]]:
        """
        Get output images for a completed prompt.
        
        Args:
            prompt_id: Prompt ID to get images for
            
        Returns:
            List of image dictionaries
        """
        api_client = await self._get_api_client()
        images = await api_client.get_output_images(prompt_id)
        
        # Add full URLs
        for img in images:
            if img.get("filename"):
                base_url = self.config.api_url.rstrip('/')
                subfolder = img.get("subfolder", "")
                subfolder_path = f"/{subfolder}" if subfolder else ""
                img["url"] = f"{base_url}/view{subfolder_path}/{img['filename']}"
        
        return images
    
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
        if self.status_manager is None:
            await self._get_api_client()
        return await self.status_manager.wait_for_completion(prompt_id, timeout, check_interval)
    
    def _update_negative_prompt_node(
        self,
        workflow: Dict[str, Any],
        negative_prompt: str
    ) -> bool:
        """
        Update negative prompt node if it exists.
        
        Args:
            workflow: Workflow dictionary
            negative_prompt: Negative prompt text
            
        Returns:
            True if node was updated, False otherwise
        """
        # Find ConditioningZeroOut node (node 404)
        negative_node = self._find_node(workflow, NODE_IDS["CONDITIONING_ZERO_OUT"])
        if negative_node:
            logger.debug(f"Found ConditioningZeroOut node, negative prompt: {negative_prompt[:50]}...")
            # The ConditioningZeroOut node typically has the negative prompt in its widgets
            # Update widget index 0 if it exists
            if "widgets_values" in negative_node and len(negative_node["widgets_values"]) > 0:
                self._update_node_widget(negative_node, 0, negative_prompt)
                return True
        
        # Try to find a CLIPTextEncode node for negative prompt
        # Look for nodes connected to ConditioningZeroOut or with specific naming
        nodes = workflow.get("nodes", [])
        for node in nodes:
            if node.get("type") == "CLIPTextEncode":
                # Check if this might be the negative prompt node
                # Look for nodes with specific titles or positions that suggest negative prompt
                node_title = node.get("title", "").lower()
                if "negative" in node_title or node.get("id") != NODE_IDS["CLIP_TEXT_ENCODE"]:
                    success = self._update_node_widget(node, 0, negative_prompt)
                    if success:
                        logger.debug(f"Updated negative prompt in CLIPTextEncode node: {negative_prompt[:50]}...")
                        return True
        
        logger.warning("Negative prompt node not found, skipping negative prompt update")
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
        
        # The mask can be in different places depending on workflow structure:
        # 1. As a second input to LoadImage node (widget index 1)
        # 2. In a separate LoadImage node specifically for masks
        # 3. In an ImageCrop or InpaintCrop node
        
        # Try updating the main LoadImage node with mask as second widget
        load_image_node = self._find_node(workflow, NODE_IDS["LOAD_IMAGE"])
        if load_image_node:
            # Check if LoadImage supports multiple inputs (mask as widget index 1)
            success = self._update_node_widget(load_image_node, 1, mask_url)
            if success:
                logger.debug(f"Updated LoadImage node with mask: {mask_url[:50]}...")
                return True
        
        # Try ImageCrop node (node 228)
        image_crop_node = self._find_node(workflow, NODE_IDS["IMAGE_CROP"])
        if image_crop_node:
            success = self._update_node_widget(image_crop_node, 0, mask_url)
            if success:
                logger.debug(f"Updated ImageCrop node with mask: {mask_url[:50]}...")
                return True
        
        # Try InpaintCrop node (node 411)
        inpaint_crop_node = self._find_node(workflow, NODE_IDS["INPAINT_CROP"])
        if inpaint_crop_node:
            success = self._update_node_widget(inpaint_crop_node, 0, mask_url)
            if success:
                logger.debug(f"Updated InpaintCrop node with mask: {mask_url[:50]}...")
                return True
        
        # Look for any other LoadImage nodes that might be for masks
        nodes = workflow.get("nodes", [])
        for node in nodes:
            if node.get("type") == "LoadImage" and node.get("id") != NODE_IDS["LOAD_IMAGE"]:
                node_title = node.get("title", "").lower()
                if "mask" in node_title:
                    success = self._update_node_widget(node, 0, mask_url)
                    if success:
                        logger.debug(f"Updated mask LoadImage node with mask: {mask_url[:50]}...")
                        return True
        
        logger.warning("Mask node not found, skipping mask update")
        return False
    
    
    async def execute_clothing_change(
        self,
        image_url: str,
        mask_url: Optional[str] = None,
        clothing_description: str = "",
        prompt: Optional[str] = None,
        negative_prompt: str = "",
        guidance_scale: float = 50.0,
        num_steps: int = 12,
        seed: Optional[int] = None,
        face_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute clothing change workflow with optional face swap.
        
        Args:
            image_url: URL or path to input image
            mask_url: URL or path to mask image
            clothing_description: Description of desired clothing
            prompt: Full prompt (if not provided, will be generated)
            negative_prompt: Negative prompt
            guidance_scale: Guidance scale
            num_steps: Number of inference steps
            seed: Random seed
            face_url: URL or path to face image for face swap (optional)
            
        Returns:
            Execution result
        """
        # Use provided prompt or default
        final_prompt = prompt or f"best quality, {clothing_description}"
        
        # Prepare workflow
        workflow = self.prepare_workflow(
            image_url=image_url,
            mask_url=mask_url,
            prompt=final_prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            seed=seed,
            face_url=face_url
        )
        
        # Queue prompt
        result = await self.queue_prompt(workflow)
        
        return {
            "prompt_id": result.get("prompt_id"),
            "workflow": workflow,
            "status": "queued",
            "face_swap": face_url is not None,
            "workflow_info": self.get_workflow_info()
        }
    
    async def execute_face_swap(
        self,
        image_url: str,
        face_url: str,
        mask_url: Optional[str] = None,
        prompt: str = "best quality, face swap, high quality portrait",
        negative_prompt: str = "",
        guidance_scale: float = 50.0,
        num_steps: int = 12,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute face swap workflow on an image that's being inpainting.
        
        This method swaps the face in the input image with the face from face_url.
        The workflow uses the "Load New Face" node to load the replacement face.
        
        Args:
            image_url: URL or path to input image (the image in inpainting)
            face_url: URL or path to face image to swap in
            mask_url: URL or path to mask image (optional, for inpainting area)
            prompt: Positive prompt for generation
            negative_prompt: Negative prompt
            guidance_scale: Guidance scale (default: 50.0)
            num_steps: Number of inference steps (default: 12)
            seed: Random seed (optional)
            
        Returns:
            Execution result with prompt_id and status
            
        Raises:
            ValueError: If required parameters are invalid
        """
        # Validate inputs
        if not image_url or not image_url.strip():
            raise ValueError("image_url is required")
        if not face_url or not face_url.strip():
            raise ValueError("face_url is required for face swap")
        
        logger.info(f"Executing face swap: image={image_url[:50]}..., face={face_url[:50]}...")
        
        # Prepare workflow with face swap
        workflow = self.prepare_workflow(
            image_url=image_url,
            mask_url=mask_url,
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            seed=seed,
            face_url=face_url
        )
        
        # Queue prompt
        result = await self.queue_prompt(workflow)
        
        return {
            "prompt_id": result.get("prompt_id"),
            "workflow": workflow,
            "status": "queued",
            "face_swap": True,
            "face_url": face_url,
            "workflow_info": self.get_workflow_info()
        }

