"""
Batch Item Processor
===================
Processes individual batch items
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum

from services.clothing_service import ClothingChangeService

logger = logging.getLogger(__name__)


class ItemStatus(str, Enum):
    """Item processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BatchItemProcessor:
    """
    Processes individual batch items.
    """
    
    def __init__(self, clothing_service: ClothingChangeService):
        """
        Initialize batch item processor.
        
        Args:
            clothing_service: Clothing change service instance
        """
        self.clothing_service = clothing_service
    
    async def process_clothing_change_item(
        self,
        item: Dict[str, Any],
        item_id: str
    ) -> Dict[str, Any]:
        """
        Process a single clothing change item.
        
        Args:
            item: Item data dictionary
            item_id: Unique item identifier
            
        Returns:
            Processing result dictionary
        """
        try:
            result = await self.clothing_service.change_clothing(
                image_url=item["image_url"],
                clothing_description=item.get("clothing_description", ""),
                mask_url=item.get("mask_url"),
                character_name=item.get("character_name"),
                negative_prompt=item.get("negative_prompt", ""),
                guidance_scale=item.get("guidance_scale", 50.0),
                num_steps=item.get("num_steps", 12),
                seed=item.get("seed"),
                optimize_prompt=item.get("optimize_prompt", True)
            )
            
            return {
                "item_id": item_id,
                "success": True,
                "result": result,
                "prompt_id": result.get("prompt_id")
            }
        except Exception as e:
            logger.error(f"Error processing clothing change item {item_id}: {e}", exc_info=True)
            return {
                "item_id": item_id,
                "success": False,
                "error": str(e)
            }
    
    async def process_face_swap_item(
        self,
        item: Dict[str, Any],
        item_id: str
    ) -> Dict[str, Any]:
        """
        Process a single face swap item.
        
        Args:
            item: Item data dictionary
            item_id: Unique item identifier
            
        Returns:
            Processing result dictionary
        """
        try:
            result = await self.clothing_service.change_clothing(
                image_url=item["image_url"],
                clothing_description="",  # Not used for face swap
                face_url=item.get("face_url"),
                negative_prompt=item.get("negative_prompt", ""),
                guidance_scale=item.get("guidance_scale", 50.0),
                num_steps=item.get("num_steps", 12),
                seed=item.get("seed"),
                optimize_prompt=item.get("optimize_prompt", True)
            )
            
            return {
                "item_id": item_id,
                "success": True,
                "result": result,
                "prompt_id": result.get("prompt_id")
            }
        except Exception as e:
            logger.error(f"Error processing face swap item {item_id}: {e}", exc_info=True)
            return {
                "item_id": item_id,
                "success": False,
                "error": str(e)
            }
    
    def build_item_result(
        self,
        item_id: str,
        result: Dict[str, Any],
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Build result dictionary for a processed item.
        
        Args:
            item_id: Item identifier
            result: Processing result
            started_at: Processing start time
            completed_at: Processing completion time
            
        Returns:
            Formatted item result dictionary
        """
        item_result = {
            "item_id": item_id,
            "status": ItemStatus.COMPLETED.value if result.get("success") else ItemStatus.FAILED.value,
            "success": result.get("success", False)
        }
        
        if result.get("success"):
            item_result["result"] = result.get("result", {})
            item_result["prompt_id"] = result.get("prompt_id")
        else:
            item_result["error"] = result.get("error", "Unknown error")
        
        if started_at:
            item_result["started_at"] = started_at.isoformat()
        if completed_at:
            item_result["completed_at"] = completed_at.isoformat()
        
        return item_result
    
    def build_item_error(
        self,
        item_id: str,
        error_message: str,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Build error result dictionary for a failed item.
        
        Args:
            item_id: Item identifier
            error_message: Error message
            started_at: Processing start time
            completed_at: Processing completion time
            
        Returns:
            Formatted error result dictionary
        """
        error_result = {
            "item_id": item_id,
            "status": ItemStatus.FAILED.value,
            "success": False,
            "error": error_message
        }
        
        if started_at:
            error_result["started_at"] = started_at.isoformat()
        if completed_at:
            error_result["completed_at"] = completed_at.isoformat()
        
        return error_result

