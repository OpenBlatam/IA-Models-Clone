"""
Comparison API Endpoints

This module provides API endpoints for content comparison functionality.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging

from ...core.config import get_config, SystemConfig
from ...core.exceptions import ComparisonError, ValidationError

logger = logging.getLogger(__name__)

router = APIRouter()


class ComparisonRequest(BaseModel):
    """Request model for content comparison"""
    content1: str = Field(..., description="First content to compare", min_length=1)
    content2: str = Field(..., description="Second content to compare", min_length=1)
    comparison_type: str = Field(default="similarity", description="Type of comparison to perform")
    options: Dict[str, Any] = Field(default_factory=dict, description="Comparison options")


class ComparisonResponse(BaseModel):
    """Response model for content comparison"""
    comparison_id: str
    content1_hash: str
    content2_hash: str
    comparison_type: str
    similarity_score: float
    results: Dict[str, Any]
    metadata: Dict[str, Any]
    processing_time: float


class ModelComparisonRequest(BaseModel):
    """Request model for model comparison"""
    model1_results: Dict[str, Any] = Field(..., description="Results from first model")
    model2_results: Dict[str, Any] = Field(..., description="Results from second model")
    comparison_metrics: List[str] = Field(default_factory=list, description="Metrics to compare")
    options: Dict[str, Any] = Field(default_factory=dict, description="Comparison options")


class ModelComparisonResponse(BaseModel):
    """Response model for model comparison"""
    comparison_id: str
    model1_name: str
    model2_name: str
    comparison_metrics: List[str]
    results: Dict[str, Any]
    winner: Optional[str]
    confidence: float
    processing_time: float


class SimilaritySearchRequest(BaseModel):
    """Request model for similarity search"""
    content: str = Field(..., description="Content to find similar items for", min_length=1)
    threshold: float = Field(default=0.8, description="Similarity threshold", ge=0.0, le=1.0)
    limit: int = Field(default=10, description="Maximum number of results", ge=1, le=100)
    options: Dict[str, Any] = Field(default_factory=dict, description="Search options")


class SimilaritySearchResponse(BaseModel):
    """Response model for similarity search"""
    search_id: str
    query_hash: str
    threshold: float
    total_matches: int
    results: List[Dict[str, Any]]
    processing_time: float


@router.post("/content", response_model=ComparisonResponse)
async def compare_content(
    request: ComparisonRequest,
    background_tasks: BackgroundTasks,
    config: SystemConfig = Depends(get_config)
):
    """
    Compare two pieces of content
    
    This endpoint performs comprehensive comparison between two content pieces
    including similarity analysis, quality differences, and trend analysis.
    """
    try:
        if not config.features.get("comparison_engine", False):
            raise HTTPException(
                status_code=403,
                detail="Comparison engine feature is not enabled"
            )
        
        # Import here to avoid circular imports
        from ...engines.comparison_engine import ComparisonEngine
        
        engine = ComparisonEngine(config)
        await engine.initialize()
        
        # Perform comparison
        import time
        start_time = time.time()
        
        results = await engine.compare_content(request.content1, request.content2)
        
        processing_time = time.time() - start_time
        
        # Generate comparison ID and content hashes
        import hashlib
        content1_hash = hashlib.md5(request.content1.encode()).hexdigest()
        content2_hash = hashlib.md5(request.content2.encode()).hexdigest()
        comparison_id = f"comparison_{content1_hash}_{content2_hash}_{int(time.time())}"
        
        response = ComparisonResponse(
            comparison_id=comparison_id,
            content1_hash=content1_hash,
            content2_hash=content2_hash,
            comparison_type=request.comparison_type,
            similarity_score=results.get("similarity_score", 0.0),
            results=results,
            metadata={
                "engine_version": "1.0.0",
                "timestamp": time.time(),
                "options": request.options
            },
            processing_time=processing_time
        )
        
        # Clean up
        await engine.shutdown()
        
        return response
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ComparisonError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Content comparison failed: {e}")
        raise HTTPException(status_code=500, detail="Content comparison failed")


@router.post("/models", response_model=ModelComparisonResponse)
async def compare_models(
    request: ModelComparisonRequest,
    background_tasks: BackgroundTasks,
    config: SystemConfig = Depends(get_config)
):
    """
    Compare two model results
    
    This endpoint compares the results from two different AI models
    to determine which performs better on specific metrics.
    """
    try:
        if not config.features.get("comparison_engine", False):
            raise HTTPException(
                status_code=403,
                detail="Comparison engine feature is not enabled"
            )
        
        # Import here to avoid circular imports
        from ...engines.comparison_engine import ComparisonEngine
        
        engine = ComparisonEngine(config)
        await engine.initialize()
        
        # Perform model comparison
        import time
        start_time = time.time()
        
        results = await engine.compare_models(request.model1_results, request.model2_results)
        
        processing_time = time.time() - start_time
        
        # Generate comparison ID
        comparison_id = f"model_comparison_{int(time.time())}"
        
        # Determine winner
        winner = None
        confidence = 0.0
        if "winner" in results:
            winner = results["winner"]
            confidence = results.get("confidence", 0.0)
        
        response = ModelComparisonResponse(
            comparison_id=comparison_id,
            model1_name=request.model1_results.get("model_name", "Model 1"),
            model2_name=request.model2_results.get("model_name", "Model 2"),
            comparison_metrics=request.comparison_metrics,
            results=results,
            winner=winner,
            confidence=confidence,
            processing_time=processing_time
        )
        
        # Clean up
        await engine.shutdown()
        
        return response
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ComparisonError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Model comparison failed: {e}")
        raise HTTPException(status_code=500, detail="Model comparison failed")


@router.post("/similarity", response_model=SimilaritySearchResponse)
async def find_similar_content(
    request: SimilaritySearchRequest,
    background_tasks: BackgroundTasks,
    config: SystemConfig = Depends(get_config)
):
    """
    Find similar content pieces
    
    This endpoint searches for content pieces similar to the provided content
    based on various similarity metrics.
    """
    try:
        if not config.features.get("comparison_engine", False):
            raise HTTPException(
                status_code=403,
                detail="Comparison engine feature is not enabled"
            )
        
        # Import here to avoid circular imports
        from ...engines.comparison_engine import ComparisonEngine
        
        engine = ComparisonEngine(config)
        await engine.initialize()
        
        # Perform similarity search
        import time
        start_time = time.time()
        
        results = await engine.find_similar_content(
            request.content, 
            threshold=request.threshold
        )
        
        processing_time = time.time() - start_time
        
        # Generate search ID and query hash
        import hashlib
        query_hash = hashlib.md5(request.content.encode()).hexdigest()
        search_id = f"similarity_search_{query_hash}_{int(time.time())}"
        
        # Limit results
        limited_results = results[:request.limit]
        
        response = SimilaritySearchResponse(
            search_id=search_id,
            query_hash=query_hash,
            threshold=request.threshold,
            total_matches=len(results),
            results=limited_results,
            processing_time=processing_time
        )
        
        # Clean up
        await engine.shutdown()
        
        return response
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ComparisonError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Similarity search failed: {e}")
        raise HTTPException(status_code=500, detail="Similarity search failed")


@router.get("/history/{comparison_id}")
async def get_comparison_history(
    comparison_id: str,
    config: SystemConfig = Depends(get_config)
):
    """
    Get comparison history by ID
    
    Retrieves the results of a previous comparison.
    """
    try:
        if not config.features.get("comparison_engine", False):
            raise HTTPException(
                status_code=403,
                detail="Comparison engine feature is not enabled"
            )
        
        # This would typically query a database or cache
        # For now, return a placeholder response
        return {
            "comparison_id": comparison_id,
            "status": "not_found",
            "message": "Comparison history retrieval not yet implemented"
        }
        
    except Exception as e:
        logger.error(f"Failed to get comparison history: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve comparison history")


@router.get("/metrics")
async def get_comparison_metrics(config: SystemConfig = Depends(get_config)):
    """
    Get available comparison metrics
    
    Returns a list of all metrics that can be used for comparison.
    """
    try:
        if not config.features.get("comparison_engine", False):
            raise HTTPException(
                status_code=403,
                detail="Comparison engine feature is not enabled"
            )
        
        # Import here to avoid circular imports
        from ...engines.comparison_engine import ComparisonEngine
        
        engine = ComparisonEngine(config)
        capabilities = engine.get_capabilities()
        
        return {
            "available_metrics": [
                "similarity_score",
                "quality_difference",
                "trend_direction",
                "significant_changes",
                "recommendations",
                "confidence_score"
            ],
            "capabilities": capabilities,
            "description": "List of all available comparison metrics and capabilities"
        }
        
    except Exception as e:
        logger.error(f"Failed to get comparison metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")





















