"""
Execution Monitor
================
Monitors workflow execution and waits for completion
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional
from enum import Enum

from .queue_manager import QueueManager
from .image_retriever import ImageRetriever

logger = logging.getLogger(__name__)


class WorkflowStatus(str, Enum):
    """Workflow execution status"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class ExecutionMonitor:
    """
    Monitors workflow execution and waits for completion.
    """
    
    def __init__(
        self,
        queue_manager: QueueManager,
        image_retriever: ImageRetriever
    ):
        """
        Initialize execution monitor.
        
        Args:
            queue_manager: Queue manager instance
            image_retriever: Image retriever instance
        """
        self.queue_manager = queue_manager
        self.image_retriever = image_retriever
    
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
            timeout: Maximum time to wait in seconds (default: 300)
            check_interval: Interval between status checks in seconds (default: 2.0)
            
        Returns:
            Dictionary with completion status and results
        """
        start_time = time.time()
        elapsed_time = 0.0
        
        logger.info(f"Waiting for prompt {prompt_id} to complete (timeout: {timeout}s)")
        
        while elapsed_time < timeout:
            status = await self.queue_manager.get_prompt_status(prompt_id)
            current_status = status.get("status")
            
            if current_status == WorkflowStatus.COMPLETED.value:
                # Get output images
                images = await self.image_retriever.get_output_images(prompt_id)
                
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
            "status": WorkflowStatus.TIMEOUT.value,
            "elapsed_time": elapsed_time,
            "message": f"Prompt did not complete within {timeout}s"
        }
    
    async def monitor_execution(
        self,
        prompt_id: str,
        on_status_change: Optional[callable] = None,
        timeout: float = 300.0,
        check_interval: float = 2.0
    ) -> Dict[str, Any]:
        """
        Monitor execution with status change callbacks.
        
        Args:
            prompt_id: Prompt ID to monitor
            on_status_change: Optional callback for status changes
            timeout: Maximum time to wait
            check_interval: Interval between checks
            
        Returns:
            Final execution result
        """
        last_status = None
        
        async def check_status():
            nonlocal last_status
            status = await self.queue_manager.get_prompt_status(prompt_id)
            current_status = status.get("status")
            
            if current_status != last_status:
                last_status = current_status
                if on_status_change:
                    await on_status_change(prompt_id, current_status, status)
            
            return current_status
        
        start_time = time.time()
        elapsed_time = 0.0
        
        while elapsed_time < timeout:
            current_status = await check_status()
            
            if current_status in [WorkflowStatus.COMPLETED.value, WorkflowStatus.FAILED.value]:
                if current_status == WorkflowStatus.COMPLETED.value:
                    images = await self.image_retriever.get_output_images(prompt_id)
                    return {
                        "prompt_id": prompt_id,
                        "status": current_status,
                        "elapsed_time": elapsed_time,
                        "images": images,
                        "image_count": len(images)
                    }
                else:
                    return {
                        "prompt_id": prompt_id,
                        "status": current_status,
                        "elapsed_time": elapsed_time,
                        "error": "Workflow execution failed"
                    }
            
            await asyncio.sleep(check_interval)
            elapsed_time = time.time() - start_time
        
        return {
            "prompt_id": prompt_id,
            "status": WorkflowStatus.TIMEOUT.value,
            "elapsed_time": elapsed_time,
            "message": f"Timeout after {timeout}s"
        }

