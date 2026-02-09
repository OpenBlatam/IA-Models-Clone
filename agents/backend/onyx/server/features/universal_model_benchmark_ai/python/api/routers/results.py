"""
Results Router - Endpoints for benchmark results.

This module provides REST API endpoints for managing
and querying benchmark results.
"""

import logging
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPAuthorizationCredentials

from ..models import BenchmarkRequest, BenchmarkResponse, ErrorResponse
from ..auth import verify_token
from core.results import ResultsManager, BenchmarkResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/results", tags=["results"])

# Initialize manager (should be injected in production)
results_manager = ResultsManager()


@router.get("", response_model=dict)
async def get_results(
    model_name: Optional[str] = None,
    benchmark_name: Optional[str] = None,
    limit: int = 100,
    token: str = Depends(verify_token),
):
    """
    Get benchmark results.
    
    Args:
        model_name: Filter by model name
        benchmark_name: Filter by benchmark name
        limit: Maximum number of results to return
        token: Authentication token
    
    Returns:
        Dictionary with results list
    """
    try:
        results = results_manager.get_results(
            model_name=model_name,
            benchmark_name=benchmark_name,
            limit=limit,
        )
        return {"results": [r.to_dict() if hasattr(r, 'to_dict') else r for r in results]}
    except Exception as e:
        logger.exception("Error getting results")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{result_id}", response_model=dict)
async def get_result(
    result_id: str,
    token: str = Depends(verify_token),
):
    """
    Get specific result by ID.
    
    Args:
        result_id: Result identifier
        token: Authentication token
    
    Returns:
        Result dictionary
    
    Raises:
        HTTPException: If result not found
    """
    try:
        results = results_manager.get_results()
        result = next((r for r in results if r.benchmark_name == result_id), None)
        if not result:
            raise HTTPException(status_code=404, detail="Result not found")
        return result.to_dict() if hasattr(result, 'to_dict') else result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error getting result")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("", response_model=dict)
async def create_result(
    result: BenchmarkResult,
    token: str = Depends(verify_token),
):
    """
    Create a new result.
    
    Args:
        result: Benchmark result to create
        token: Authentication token
    
    Returns:
        Dictionary with result ID and status
    """
    try:
        results_manager.save_result(result)
        return {"id": result.benchmark_name, "status": "created"}
    except Exception as e:
        logger.exception("Error creating result")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/comparison/{benchmark_name}", response_model=dict)
async def get_comparison(
    benchmark_name: str,
    model_names: Optional[List[str]] = None,
    token: str = Depends(verify_token),
):
    """
    Get comparison results for a benchmark.
    
    Args:
        benchmark_name: Name of the benchmark
        model_names: Optional list of model names to compare
        token: Authentication token
    
    Returns:
        Comparison results dictionary
    """
    try:
        comparison = results_manager.get_comparison(
            benchmark_name=benchmark_name,
            model_names=model_names,
        )
        return comparison.__dict__ if hasattr(comparison, '__dict__') else comparison
    except Exception as e:
        logger.exception("Error getting comparison")
        raise HTTPException(status_code=500, detail=str(e))












