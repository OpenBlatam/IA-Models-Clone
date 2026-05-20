from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import time
import logging
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from typing import Any, List, Dict, Optional
import asyncio
logger = logging.getLogger("blog_system")

# Métrica global de throughput
request_count = 0

async def performance_middleware(request: Request, call_next):
    
    """performance_middleware function."""
global request_count
    start = time.time()
    response: Response = await call_next(request)
    process_time = time.time() - start
    response.headers["X-Process-Time"] = str(process_time)
    request_count += 1
    response.headers["X-Request-Count"] = str(request_count)
    logger.info(f"[PERF] {request.method} {request.url.path} - {response.status_code} - {process_time:.4f}s - throughput={request_count}")
    return response

async def error_handling_middleware(request: Request, call_next):
    
    """error_handling_middleware function."""
try:
        return await call_next(request)
    except ValidationError as exc:
        logger.error(f"Validation error: {exc}")
        return JSONResponse(status_code=422, content={"detail": str(exc)})
    except Exception as exc:
        logger.critical(f"Unhandled server error: {exc}", exc_info=True)
        return JSONResponse(status_code=500, content={"detail": "Internal server error. Please try again later."})

async def logging_and_timing_middleware(request: Request, call_next):
    
    """logging_and_timing_middleware function."""
logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response 