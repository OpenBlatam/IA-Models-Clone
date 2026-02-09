"""Batch processing endpoints."""

from fastapi import APIRouter, HTTPException, Depends
from typing import List
from datetime import datetime
import asyncio

from api.schemas.comparison import BatchVisualizationRequest, BatchVisualizationResponse
from api.schemas.visualization import VisualizationRequest
from services.visualization_service import VisualizationService
from core.dependencies import get_visualization_service
from utils.decorators_advanced import track_metrics
from utils.metrics import metrics_collector
from utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/batch", response_model=BatchVisualizationResponse)
@track_metrics("api.batch")
async def process_batch(
    request: BatchVisualizationRequest,
    service: VisualizationService = Depends(get_visualization_service)
) -> BatchVisualizationResponse:
    """
    Process multiple visualizations in batch.
    
    Args:
        request: Batch processing request
        service: Visualization service instance
        
    Returns:
        BatchVisualizationResponse with results
    """
    start_time = datetime.utcnow()
    results = []
    processed = 0
    failed = 0
    
    # Process with concurrency limit
    semaphore = asyncio.Semaphore(request.max_concurrent)
    
    async def process_single(req_data: dict):
        """Process a single visualization request."""
        async with semaphore:
            try:
                viz_request = VisualizationRequest(**req_data)
                result = await service.create_visualization(viz_request)
                return {"success": True, "result": result.dict()}
            except Exception as e:
                logger.error(f"Error processing batch item: {e}")
                return {"success": False, "error": str(e)}
    
    # Process all requests
    tasks = [process_single(req) for req in request.requests]
    batch_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    for result in batch_results:
        if isinstance(result, Exception):
            failed += 1
            results.append({"success": False, "error": str(result)})
        elif result.get("success"):
            processed += 1
            results.append(result)
        else:
            failed += 1
            results.append(result)
    
    processing_time = (datetime.utcnow() - start_time).total_seconds()
    
    metrics_collector.increment("api.batch.processed", processed)
    metrics_collector.increment("api.batch.failed", failed)
    metrics_collector.record_timing("api.batch.processing_time", processing_time)
    
    return BatchVisualizationResponse(
        total=len(request.requests),
        processed=processed,
        failed=failed,
        results=results,
        processing_time=processing_time
    )

