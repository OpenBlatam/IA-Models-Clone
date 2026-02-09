"""
Batch Routes
============

API routes for batch processing.
"""

import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from ..dependencies import get_agent
from ..models import BatchProcessRequest
from ...core.batch_processor import BatchItem

logger = logging.getLogger(__name__)

router = APIRouter(tags=["batch"])


@router.post("/batch-process")
async def batch_process(request: BatchProcessRequest):
    """Process multiple files in batch."""
    agent = get_agent()
    
    try:
        # Convert to BatchItem objects
        batch_items = []
        for item in request.items:
            batch_items.append(BatchItem(
                file_path=item["file_path"],
                service_type=item.get("service_type", "enhance_image"),
                enhancement_type=item.get("enhancement_type", "general"),
                options=item.get("options", {}),
                priority=item.get("priority", 0),
                metadata=item.get("metadata", {})
            ))
        
        # Process batch
        result = await agent.process_batch(batch_items)
        
        return JSONResponse(result.to_dict())
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))




