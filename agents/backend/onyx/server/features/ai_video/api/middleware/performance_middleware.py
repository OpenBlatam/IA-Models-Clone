from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import time
from typing import Callable
from fastapi import Request, Response
from ..utils.metrics import record_metric
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Performance Middleware - Request timing and optimization
"""



def create_performance_middleware() -> Callable:
    """Create performance monitoring middleware."""
    
    async def performance_middleware(request: Request, call_next: Callable) -> Response:
        # Record start time
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Add timing header
        response.headers["X-Process-Time"] = str(process_time)
        
        # Record metrics
        await record_metric("request_duration", process_time, {
            "method": request.method,
            "endpoint": request.url.path,
            "status_code": response.status_code,
        })
        
        return response
    
    return performance_middleware 