"""
Batch API Handler - Manejador de APIs batch
===========================================
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class BatchRequest:
    """Request batch"""
    id: str
    method: str
    path: str
    headers: Optional[Dict[str, str]] = None
    body: Optional[Any] = None


@dataclass
class BatchResponse:
    """Response batch"""
    request_id: str
    status_code: int
    headers: Dict[str, str]
    body: Any
    duration: float


@dataclass
class BatchResult:
    """Resultado de batch"""
    total_requests: int
    successful: int
    failed: int
    responses: List[BatchResponse]
    total_duration: float


class BatchAPIHandler:
    """Manejador de APIs batch"""
    
    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
    
    async def execute_batch(
        self,
        requests: List[BatchRequest],
        handler: Callable
    ) -> BatchResult:
        """Ejecuta un batch de requests"""
        start_time = datetime.now()
        semaphore = asyncio.Semaphore(self.max_concurrent)
        responses = []
        successful = 0
        failed = 0
        
        async def process_request(req: BatchRequest) -> BatchResponse:
            async with semaphore:
                req_start = datetime.now()
                try:
                    if asyncio.iscoroutinefunction(handler):
                        result = await handler(req.method, req.path, req.headers, req.body)
                    else:
                        result = handler(req.method, req.path, req.headers, req.body)
                    
                    duration = (datetime.now() - req_start).total_seconds()
                    
                    return BatchResponse(
                        request_id=req.id,
                        status_code=result.get("status_code", 200),
                        headers=result.get("headers", {}),
                        body=result.get("body"),
                        duration=duration
                    )
                except Exception as e:
                    duration = (datetime.now() - req_start).total_seconds()
                    logger.error(f"Error procesando request {req.id}: {e}")
                    return BatchResponse(
                        request_id=req.id,
                        status_code=500,
                        headers={},
                        body={"error": str(e)},
                        duration=duration
                    )
        
        tasks = [process_request(req) for req in requests]
        responses = await asyncio.gather(*tasks)
        
        for resp in responses:
            if 200 <= resp.status_code < 300:
                successful += 1
            else:
                failed += 1
        
        total_duration = (datetime.now() - start_time).total_seconds()
        
        return BatchResult(
            total_requests=len(requests),
            successful=successful,
            failed=failed,
            responses=responses,
            total_duration=total_duration
        )




