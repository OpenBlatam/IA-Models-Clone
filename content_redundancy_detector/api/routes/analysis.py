"""
Analysis Routes
API endpoints for content analysis operations
"""

import logging
import time
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

try:
    from ...application.services import AnalysisService, QualityService
    from ...application.dependencies import get_analysis_service, get_quality_service
    from ...application.dtos import AnalysisRequest, SimilarityRequest, QualityRequest
    from ...domain.value_objects import AnalysisResult, SimilarityResult, QualityResult
    APPLICATION_LAYER_AVAILABLE = True
except ImportError:
    APPLICATION_LAYER_AVAILABLE = False
    # Fallback if application layer not available
    AnalysisService = None
    QualityService = None

from ...utils.validation import ContentValidator, ValidationError as CustomValidationError
from ...utils.error_codes import ErrorCode, format_error_response, get_status_code_for_error
from ...utils.response_helpers import (
    get_request_id, create_success_response, create_error_response, json_response
)
from ...utils.structured_logging import (
    set_request_context, clear_request_context, log_performance
)
from ...services import analyze_content, detect_similarity, assess_quality

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyze", tags=["Analysis"])


@router.post("", response_model=Dict[str, Any])
async def analyze_content_endpoint(
    http_request: Request,
    request: Dict[str, Any]
) -> JSONResponse:
    """
    Analyze content for redundancy with enhanced validation and error handling
    """
    start_time = time.perf_counter()
    request_id = get_request_id(http_request)
    set_request_context(request_id=request_id)
    
    try:
        # Extract content from request
        content = request.get("content", "")
        if not isinstance(content, str):
            error_response = format_error_response(
                ErrorCode.INVALID_TYPE,
                "Content must be a string",
                {"received_type": type(content).__name__},
                request_id
            )
            return json_response(
                error_response,
                status_code=get_status_code_for_error(ErrorCode.INVALID_TYPE),
                request_id=request_id
            )
        
        # Validate content
        is_valid, error_msg, error_code = ContentValidator.validate_analysis_input(content)
        if not is_valid:
            error_response = format_error_response(
                ErrorCode(error_code) if error_code in [e.value for e in ErrorCode] else ErrorCode.VALIDATION_ERROR,
                error_msg or "Content validation failed",
                None,
                request_id
            )
            return json_response(
                error_response,
                status_code=400,
                request_id=request_id
            )
        
        # Process analysis
        user_id = http_request.headers.get("X-User-Id") or http_request.headers.get("User-Id")
        
        try:
            result = analyze_content(content, request_id=request_id, user_id=user_id)
            
            # Create success response
            response_data = create_success_response(
                data=result,
                message="Content analyzed successfully",
                metadata={
                    "word_count": result.get("word_count"),
                    "character_count": result.get("character_count"),
                    "redundancy_score": result.get("redundancy_score")
                },
                request_id=request_id
            )
            
            duration = time.perf_counter() - start_time
            log_performance("analyze_content", duration, logger, status_code=200)
            
            return json_response(response_data, status_code=200, request_id=request_id)
            
        except Exception as e:
            logger.error(f"Analysis processing error: {e}", exc_info=True, extra={"request_id": request_id})
            error_response = format_error_response(
                ErrorCode.PROCESSING_ERROR,
                f"Failed to analyze content: {str(e)}",
                None,
                request_id
            )
            return json_response(
                error_response,
                status_code=500,
                request_id=request_id
            )
    
    except Exception as e:
        duration = time.perf_counter() - start_time
        logger.error(f"Unexpected error in analyze_content: {e}", exc_info=True, extra={"request_id": request_id})
        error_response = format_error_response(
            ErrorCode.INTERNAL_ERROR,
            "An unexpected error occurred",
            None,
            request_id
        )
        return json_response(error_response, status_code=500, request_id=request_id)
    
    finally:
        clear_request_context()


@router.post("/similarity", response_model=Dict[str, Any])
async def check_similarity(
    request: SimilarityRequest,
    service: AnalysisService = Depends(get_analysis_service)
) -> Dict[str, Any]:
    """
    Check similarity between two texts
    """
    try:
        result = await service.detect_similarity(request)
        
        return {
            "success": True,
            "data": result.to_dict(),
            "error": None,
            "timestamp": result.timestamp
        }
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Similarity check error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/quality", response_model=Dict[str, Any])
async def assess_quality(
    request: QualityRequest,
    service: QualityService = Depends(get_quality_service)
) -> Dict[str, Any]:
    """
    Assess content quality
    """
    try:
        result = await service.assess_quality(request)
        
        return {
            "success": True,
            "data": result.to_dict(),
            "error": None,
            "timestamp": result.timestamp
        }
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Quality assessment error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
