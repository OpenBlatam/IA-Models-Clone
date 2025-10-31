from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import time
import uuid
from typing import Callable
from fastapi import Request, Response
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Logging Middleware - Structured logging
"""



def create_logging_middleware() -> Callable:
    """Create logging middleware."""
    
    async def logging_middleware(request: Request, call_next: Callable) -> Response:
        # Generate correlation ID
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        
        # Add to request state
        request.state.correlation_id = correlation_id
        
        # Process request
        response = await call_next(request)
        
        # Add correlation ID to response
        response.headers["X-Correlation-ID"] = correlation_id
        
        return response
    
    return logging_middleware 