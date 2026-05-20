"""
Batch execution router for Multi-Model API
Handles batch execution endpoints
"""

import asyncio
import logging
import uuid
import time
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, Request, BackgroundTasks, HTTPException

from ...api.schemas import (
    BatchMultiModelRequest,
    BatchMultiModelResponse,
    MultiModelRequest,
    MultiModelResponse
)
from ...api.dependencies import get_execution_service, check_rate_limit
from ...api.helpers import validate_rate_limit
from ...core.services import ExecutionService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/multi-model", tags=["Batch"])


@router.post("/execute/batch", response_model=BatchMultiModelResponse)
async def execute_batch_multi_model(
    batch_request: BatchMultiModelRequest,
    http_request: Request,
    background_tasks: BackgroundTasks,
    execution_service: ExecutionService = Depends(get_execution_service)
):
    """
    Execute multiple multi-model requests in batch
    
    Processes up to 10 requests in parallel with rate limiting protection
    
    Args:
        batch_request: Batch request containing multiple MultiModelRequest objects
        http_request: FastAPI Request object
        background_tasks: Background tasks for async operations
        execution_service: Execution service instance
        
    Returns:
        BatchMultiModelResponse with results for all requests
    """
    batch_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Rate limiting
    rate_limit_info = await check_rate_limit(http_request, "batch")
    validate_rate_limit(rate_limit_info)
    
    # Validate batch size
    if not batch_request.requests:
        raise HTTPException(
            status_code=400,
            detail="At least one request must be provided in batch"
        )
    
    if len(batch_request.requests) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 requests allowed per batch"
        )
    
    # Validate each request
    from ...core.services import ValidationService
    for i, req in enumerate(batch_request.requests):
        try:
            ValidationService.validate_request(req)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid request at index {i}: {str(e)}"
            )
    
    responses: List[MultiModelResponse] = []
    successful_count = 0
    failed_count = 0
    
    async def process_single_request(req: MultiModelRequest, index: int) -> Optional[MultiModelResponse]:
        """Process a single request in the batch"""
        try:
            return await execution_service.execute(
                request=req,
                background_task=background_tasks
            )
        except Exception as e:
            logger.error(
                f"Batch request {index} failed: {e}",
                exc_info=True,
                extra={
                    "batch_id": batch_id,
                    "request_index": index,
                    "stop_on_first_error": batch_request.stop_on_first_error
                }
            )
            if batch_request.stop_on_first_error:
                raise
            return None
    
    # Process all requests in parallel
    tasks = [
        process_single_request(req, i)
        for i, req in enumerate(batch_request.requests)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    for result in results:
        if isinstance(result, Exception):
            failed_count += 1
            if batch_request.stop_on_first_error:
                raise HTTPException(
                    status_code=500,
                    detail=f"Batch processing stopped due to error: {str(result)}"
                )
        elif result is not None:
            responses.append(result)
            successful_count += 1
        else:
            failed_count += 1
    
    total_latency_ms = (time.time() - start_time) * 1000
    
    return BatchMultiModelResponse(
        request_id=batch_id,
        total_requests=len(batch_request.requests),
        successful_requests=successful_count,
        failed_requests=failed_count,
        responses=responses,
        timestamp=datetime.now().isoformat()
    )

