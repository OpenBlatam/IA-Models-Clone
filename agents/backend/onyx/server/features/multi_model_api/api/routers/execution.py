"""
Execution router for Multi-Model API
Handles all execution-related endpoints
"""

import logging
import uuid
from typing import Dict, Any
from fastapi import APIRouter, Depends, Request, BackgroundTasks, HTTPException
from fastapi.responses import ORJSONResponse, Response

from ...api.schemas import MultiModelRequest, MultiModelResponse
from ...api.dependencies import get_execution_service
from ...api.exceptions import (
    ValidationException,
    TimeoutException,
    ModelExecutionException,
    RateLimitExceededException
)
from ...core.services import ExecutionService
from ...core.performance import fast_json_dumps
from ...core.response_optimizer import ResponseOptimizer
from ...api.helpers import validate_rate_limit
from ...api.dependencies import check_rate_limit

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/multi-model", tags=["Execution"])


@router.post("/execute", response_model=MultiModelResponse)
async def execute_multi_model(
    request: MultiModelRequest,
    http_request: Request,
    background_tasks: BackgroundTasks,
    execution_service: ExecutionService = Depends(get_execution_service)
) -> Response:
    """
    Execute multiple AI models with optimized parallel processing
    
    Features:
    - Rate limiting protection
    - Multi-tier caching
    - Parallel, sequential, or consensus strategies
    - Partial success handling
    - Configurable timeouts
    """
    request_id = str(uuid.uuid4())
    
    try:
        # Rate limiting
        rate_limit_info = await check_rate_limit(http_request, "execute")
        validate_rate_limit(rate_limit_info)
        
        # Execute using service
        response = await execution_service.execute(
            request=request,
            request_id=request_id,
            background_task=background_tasks
        )
        
        # Optimize response if large
        return _optimize_response(response)
        
    except ValidationException as e:
        return _handle_validation_error(e, request_id)
    except RateLimitExceededException as e:
        return _handle_rate_limit_error(e, request_id)
    except TimeoutException as e:
        return _handle_timeout_error(e, request_id)
    except ModelExecutionException as e:
        return _handle_execution_error(e, request_id)
    except Exception as e:
        return _handle_unexpected_error(e, request_id)


def _handle_validation_error(e: ValidationException, request_id: str) -> HTTPException:
    """Handle validation errors"""
    logger.warning(
        f"Validation error for request {request_id}: {e}",
        extra={"request_id": request_id, "error": str(e)}
    )
    raise HTTPException(status_code=400, detail=str(e))


def _handle_rate_limit_error(e: RateLimitExceededException, request_id: str) -> HTTPException:
    """Handle rate limit errors"""
    logger.warning(
        f"Rate limit exceeded for request {request_id}",
        extra={"request_id": request_id}
    )
    raise HTTPException(status_code=429, detail=str(e))


def _handle_timeout_error(e: TimeoutException, request_id: str) -> HTTPException:
    """Handle timeout errors"""
    logger.error(
        f"Timeout for request {request_id}: {e}",
        extra={"request_id": request_id, "timeout": getattr(e, "timeout", None)}
    )
    raise HTTPException(status_code=504, detail=str(e))


def _handle_execution_error(e: ModelExecutionException, request_id: str) -> HTTPException:
    """Handle execution errors"""
    logger.error(
        f"Execution error for request {request_id}: {e}",
        extra={"request_id": request_id},
        exc_info=True
    )
    raise HTTPException(status_code=500, detail=str(e))


def _handle_unexpected_error(e: Exception, request_id: str) -> HTTPException:
    """Handle unexpected errors"""
    logger.error(
        f"Unexpected error for request {request_id}: {e}",
        extra={"request_id": request_id},
        exc_info=True
    )
    raise HTTPException(status_code=500, detail="Internal server error")


def _optimize_response(response: MultiModelResponse) -> Response:
    """Optimize response with compression if needed"""
    try:
        # Use model_dump() if available (Pydantic v2), fallback to dict()
        _dict_method = getattr(response, 'model_dump', None)
        response_dict: Dict[str, Any] = _dict_method() if _dict_method else response.dict()
        
        # Use fast JSON serialization to check size
        if callable(fast_json_dumps):
            response_json = fast_json_dumps(response_dict)
            response_size = len(response_json)
            
            # Compress if larger than threshold
            if response_size > 1024:
                compressed, is_compressed = ResponseOptimizer.compress_response(
                    response_dict,
                    threshold=1024
                )
                if is_compressed:
                    return Response(
                        content=compressed,
                        media_type="application/json",
                        headers={
                            "Content-Encoding": "gzip",
                            "Content-Length": str(len(compressed))
                        }
                    )
            
            # Return optimized JSON response
            return ORJSONResponse(content=response_dict)
        else:
            # Fallback to standard response
            return ORJSONResponse(content=response_dict)
            
    except Exception as e:
        logger.warning(
            f"Failed to optimize response, using fallback: {e}",
            exc_info=True
        )
        # Fallback to standard response
        _dict_method = getattr(response, 'model_dump', None)
        fallback_dict = _dict_method() if _dict_method else response.dict()
        return ORJSONResponse(content=fallback_dict)

