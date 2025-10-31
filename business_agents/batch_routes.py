"""Batch processing API endpoints."""
from fastapi import APIRouter, HTTPException
from typing import List, Any, Dict, Optional
from pydantic import BaseModel
from .batch_processing import BatchProcessor

router = APIRouter(prefix="/batch", tags=["Batch Processing"])

# Global batch processor instance
batch_processor = BatchProcessor(max_concurrent=10)


class BatchRequest(BaseModel):
    """Request model for batch processing."""
    items: List[Any]
    max_concurrent: int = 10


class BatchStatusResponse(BaseModel):
    """Response model for batch status."""
    batch_id: str
    status: str
    total: int
    completed: int
    result: Optional[Dict[str, Any]] = None


@router.post("/process", response_model=Dict[str, str])
async def create_batch(request: BatchRequest):
    """
    Create a batch processing job.
    
    Note: This is a simplified example. In production, you'd:
    - Validate items
    - Define specific processors
    - Use Celery/RQ for background processing
    """
    # Example processor (replace with your actual logic)
    async def example_processor(item: Any) -> Any:
        # Simulate processing
        import asyncio
        await asyncio.sleep(0.1)
        return {"processed": item, "result": f"Processed {item}"}
    
    batch_id = await batch_processor.create_batch(
        items=request.items,
        processor=example_processor,
        max_concurrent=request.max_concurrent
    )
    
    return {"batch_id": batch_id, "status": "created"}


@router.get("/{batch_id}", response_model=Dict[str, Any])
async def get_batch_status(batch_id: str):
    """Get batch processing status."""
    status = batch_processor.get_batch_status(batch_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")
    return status


@router.get("/", response_model=List[Dict[str, Any]])
async def list_batches():
    """List all active batches."""
    return list(batch_processor.active_batches.values())

