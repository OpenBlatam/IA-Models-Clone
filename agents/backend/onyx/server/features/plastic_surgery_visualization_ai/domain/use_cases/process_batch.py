"""Use case for batch processing."""

import asyncio
from datetime import datetime
from typing import List, Dict, Any

from api.schemas.comparison import BatchVisualizationRequest, BatchVisualizationResponse
from api.schemas.visualization import VisualizationRequest
from domain.use_cases.create_visualization import CreateVisualizationUseCase
from core.interfaces import IMetricsCollector
from utils.logger import get_logger

logger = get_logger(__name__)


class ProcessBatchUseCase:
    """Use case for processing batch visualizations."""
    
    def __init__(
        self,
        create_visualization_use_case: CreateVisualizationUseCase,
        metrics_collector: IMetricsCollector
    ):
        self.create_visualization_use_case = create_visualization_use_case
        self.metrics_collector = metrics_collector
    
    async def execute(
        self,
        request: BatchVisualizationRequest
    ) -> BatchVisualizationResponse:
        """
        Execute the use case.
        
        Args:
            request: Batch processing request
            
        Returns:
            BatchVisualizationResponse with results
        """
        start_time = datetime.utcnow()
        results = []
        processed = 0
        failed = 0
        
        # Process with concurrency limit
        semaphore = asyncio.Semaphore(request.max_concurrent)
        
        async def process_single(req_data: Dict[str, Any]):
            """Process a single visualization request."""
            async with semaphore:
                try:
                    viz_request = VisualizationRequest(**req_data)
                    result = await self.create_visualization_use_case.execute(viz_request)
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
        
        self.metrics_collector.increment("api.batch.processed", processed)
        self.metrics_collector.increment("api.batch.failed", failed)
        self.metrics_collector.record_timing("api.batch.processing_time", processing_time)
        
        return BatchVisualizationResponse(
            total=len(request.requests),
            processed=processed,
            failed=failed,
            results=results,
            processing_time=processing_time
        )

