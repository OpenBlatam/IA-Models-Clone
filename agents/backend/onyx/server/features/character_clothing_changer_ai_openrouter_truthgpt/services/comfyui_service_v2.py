"""
ComfyUI Service V2 - Refactored
================================
Refactored ComfyUI service using modular helpers

This version uses helper modules for better separation of concerns:
- HTTPClientManager: HTTP client lifecycle
- WorkflowManager: Workflow template management
- QueueManager: Queue operations
- RetryHandler: Retry logic
- WorkflowNodeManager: Node operations
- WorkflowValidator: Validation
- WorkflowPreparer: Workflow preparation
- ImageRetriever: Image retrieval
- ExecutionMonitor: Execution monitoring
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from config.settings import get_settings
from services.helpers import (
    HTTPClientManager,
    HTTPClientConfig,
    WorkflowManager,
    QueueManager,
    RetryHandler,
    WorkflowNodeManager,
    WorkflowValidator,
    WorkflowPreparer,
    ImageRetriever,
    ExecutionMonitor,
    WorkflowStatus,
)

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


class ComfyUIServiceV2:
    """
    Refactored ComfyUI service using modular helpers.
    
    This service orchestrates the helper modules to provide
    a clean interface for ComfyUI operations.
    """
    
    def __init__(self, config: Optional[ComfyUIConfig] = None):
        """
        Initialize ComfyUI service V2.
        
        Args:
            config: Optional configuration. If not provided, loads from settings.
        """
        self.settings = get_settings()
        self.config = config or ComfyUIConfig(
            api_url=self.settings.comfyui_api_url.rstrip('/'),
            workflow_path=self.settings.comfyui_workflow_path
        )
        
        # Initialize helper modules
        http_config = HTTPClientConfig(
            max_connections=self.config.max_connections,
            max_keepalive=self.config.max_keepalive,
            keepalive_expiry=self.config.keepalive_expiry,
            timeout=self.config.timeout,
            connect_timeout=self.config.connect_timeout
        )
        
        self.http_client_manager = HTTPClientManager(http_config)
        self.workflow_manager = WorkflowManager(self.config.workflow_path)
        self.node_manager = WorkflowNodeManager()
        self.validator = WorkflowValidator()
        self.retry_handler = RetryHandler(
            max_retries=self.config.max_retries,
            base_delay=self.config.retry_delay
        )
        self.queue_manager = QueueManager(
            self.config.api_url,
            self.http_client_manager
        )
        self.image_retriever = ImageRetriever(
            self.config.api_url,
            self.http_client_manager
        )
        self.execution_monitor = ExecutionMonitor(
            self.queue_manager,
            self.image_retriever
        )
        self.workflow_preparer = WorkflowPreparer(
            self.workflow_manager,
            self.node_manager,
            self.validator
        )
        
        logger.info("ComfyUI Service V2 initialized")
    
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
        Queue a prompt in ComfyUI with retry logic.
        
        Args:
            workflow: Workflow dictionary to queue
            client_id: Optional client ID for tracking
            retry: Whether to retry on failure
            
        Returns:
            Dictionary with prompt_id and execution details
        """
        async def _queue():
            return await self.queue_manager.queue_prompt(workflow, client_id)
        
        if retry:
            return await self.retry_handler.execute_with_retry(_queue)
        else:
            return await _queue()
    
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
            Prepared workflow dictionary
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
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        return await self.queue_manager.get_queue_status()
    
    async def get_history(self, max_items: int = 10) -> Dict[str, Any]:
        """Get workflow execution history"""
        return await self.queue_manager.get_history(max_items)
    
    async def get_prompt_status(self, prompt_id: str) -> Dict[str, Any]:
        """Get status of a specific prompt"""
        return await self.queue_manager.get_prompt_status(prompt_id)
    
    async def get_output_images(self, prompt_id: str) -> List[Dict[str, Any]]:
        """Get output images for a completed prompt"""
        return await self.image_retriever.get_output_images(prompt_id)
    
    async def wait_for_completion(
        self,
        prompt_id: str,
        timeout: float = 300.0,
        check_interval: float = 2.0
    ) -> Dict[str, Any]:
        """Wait for a prompt to complete"""
        return await self.execution_monitor.wait_for_completion(
            prompt_id, timeout, check_interval
        )
    
    async def cancel_prompt(self, prompt_id: str) -> Dict[str, Any]:
        """Cancel a queued prompt"""
        return await self.queue_manager.cancel_prompt(prompt_id)
    
    def get_workflow_info(self) -> Dict[str, Any]:
        """
        Get information about the workflow template.
        
        Returns:
            Dictionary with workflow information
        """
        workflow = self.workflow_manager.get_template()
        nodes = workflow.get("nodes", [])
        
        return {
            "node_count": len(nodes),
            "version": workflow.get("version", "unknown"),
            "last_node_id": workflow.get("last_node_id"),
            "last_link_id": workflow.get("last_link_id"),
            "nodes": [
                {
                    "id": node.get("id"),
                    "type": node.get("type"),
                    "title": node.get("title", "")
                }
                for node in nodes[:10]  # First 10 nodes as sample
            ]
        }
    
    async def execute_clothing_change(
        self,
        image_url: str,
        clothing_description: str,
        mask_url: Optional[str] = None,
        negative_prompt: str = "",
        guidance_scale: float = 50.0,
        num_steps: int = 12,
        seed: Optional[int] = None,
        wait_for_completion: bool = True,
        timeout: float = 300.0
    ) -> Dict[str, Any]:
        """
        Execute a complete clothing change workflow.
        
        Args:
            image_url: URL or path to input image
            clothing_description: Description of desired clothing
            mask_url: Optional mask image URL
            negative_prompt: Negative prompt
            guidance_scale: Guidance scale
            num_steps: Number of inference steps
            seed: Random seed
            wait_for_completion: Whether to wait for completion
            timeout: Timeout for waiting
            
        Returns:
            Dictionary with execution results
        """
        # Prepare workflow
        workflow = self.prepare_workflow(
            image_url=image_url,
            mask_url=mask_url,
            prompt=clothing_description,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            seed=seed
        )
        
        # Queue prompt
        queue_result = await self.queue_prompt(workflow)
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
            "status": "queued"
        }
        
        # Wait for completion if requested
        if wait_for_completion:
            completion_result = await self.wait_for_completion(prompt_id, timeout)
            result.update(completion_result)
            
            if completion_result.get("status") == WorkflowStatus.COMPLETED.value:
                result["success"] = True
                result["images"] = completion_result.get("images", [])
            else:
                result["success"] = False
                result["error"] = completion_result.get("error") or completion_result.get("message")
        
        return result

