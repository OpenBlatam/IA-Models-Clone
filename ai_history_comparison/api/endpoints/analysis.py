"""
Analysis API Endpoints

This module provides API endpoints for content analysis functionality.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging

from ...core.config import get_config, SystemConfig
from ...core.exceptions import AnalysisError, ValidationError

logger = logging.getLogger(__name__)

router = APIRouter()


class AnalysisRequest(BaseModel):
    """Request model for content analysis"""
    content: str = Field(..., description="Content to analyze", min_length=1)
    analysis_type: str = Field(default="comprehensive", description="Type of analysis to perform")
    options: Dict[str, Any] = Field(default_factory=dict, description="Analysis options")


class AnalysisResponse(BaseModel):
    """Response model for content analysis"""
    analysis_id: str
    content_hash: str
    analysis_type: str
    results: Dict[str, Any]
    metadata: Dict[str, Any]
    processing_time: float


class BatchAnalysisRequest(BaseModel):
    """Request model for batch analysis"""
    contents: List[str] = Field(..., description="List of contents to analyze", min_items=1, max_items=100)
    analysis_type: str = Field(default="comprehensive", description="Type of analysis to perform")
    options: Dict[str, Any] = Field(default_factory=dict, description="Analysis options")


class BatchAnalysisResponse(BaseModel):
    """Response model for batch analysis"""
    batch_id: str
    total_items: int
    processed_items: int
    results: List[Dict[str, Any]]
    errors: List[Dict[str, Any]]
    processing_time: float


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_content(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    config: SystemConfig = Depends(get_config)
):
    """
    Analyze a single piece of content
    
    This endpoint performs comprehensive analysis on the provided content
    including quality, sentiment, complexity, and readability metrics.
    """
    try:
        if not config.features.get("content_analysis", False):
            raise HTTPException(
                status_code=403,
                detail="Content analysis feature is not enabled"
            )
        
        # Import here to avoid circular imports
        from ...analyzers.content_analyzer import ContentAnalyzer
        
        analyzer = ContentAnalyzer(config)
        await analyzer.initialize()
        
        # Perform analysis
        import time
        start_time = time.time()
        
        results = await analyzer.analyze(request.content, **request.options)
        
        processing_time = time.time() - start_time
        
        # Generate analysis ID and content hash
        import hashlib
        content_hash = hashlib.md5(request.content.encode()).hexdigest()
        analysis_id = f"analysis_{content_hash}_{int(time.time())}"
        
        response = AnalysisResponse(
            analysis_id=analysis_id,
            content_hash=content_hash,
            analysis_type=request.analysis_type,
            results=results,
            metadata={
                "analyzer_version": "1.0.0",
                "timestamp": time.time(),
                "options": request.options
            },
            processing_time=processing_time
        )
        
        # Clean up
        await analyzer.shutdown()
        
        return response
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except AnalysisError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Analysis failed")


@router.post("/analyze/batch", response_model=BatchAnalysisResponse)
async def analyze_content_batch(
    request: BatchAnalysisRequest,
    background_tasks: BackgroundTasks,
    config: SystemConfig = Depends(get_config)
):
    """
    Analyze multiple pieces of content in batch
    
    This endpoint performs analysis on multiple content pieces efficiently.
    Maximum 100 items per batch.
    """
    try:
        if not config.features.get("content_analysis", False):
            raise HTTPException(
                status_code=403,
                detail="Content analysis feature is not enabled"
            )
        
        # Import here to avoid circular imports
        from ...analyzers.content_analyzer import ContentAnalyzer
        
        analyzer = ContentAnalyzer(config)
        await analyzer.initialize()
        
        # Perform batch analysis
        import time
        start_time = time.time()
        
        results = await analyzer.batch_analyze(request.contents, **request.options)
        
        processing_time = time.time() - start_time
        
        # Generate batch ID
        batch_id = f"batch_{int(time.time())}"
        
        # Separate results and errors
        successful_results = []
        errors = []
        
        for i, result in enumerate(results):
            if "error" in result:
                errors.append({
                    "index": i,
                    "content": request.contents[i][:100] + "..." if len(request.contents[i]) > 100 else request.contents[i],
                    "error": result["error"]
                })
            else:
                successful_results.append({
                    "index": i,
                    "content_hash": hashlib.md5(request.contents[i].encode()).hexdigest(),
                    "results": result
                })
        
        response = BatchAnalysisResponse(
            batch_id=batch_id,
            total_items=len(request.contents),
            processed_items=len(successful_results),
            results=successful_results,
            errors=errors,
            processing_time=processing_time
        )
        
        # Clean up
        await analyzer.shutdown()
        
        return response
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except AnalysisError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Batch analysis failed")


@router.get("/metrics")
async def get_analysis_metrics(config: SystemConfig = Depends(get_config)):
    """
    Get available analysis metrics
    
    Returns a list of all metrics that can be calculated during analysis.
    """
    try:
        if not config.features.get("content_analysis", False):
            raise HTTPException(
                status_code=403,
                detail="Content analysis feature is not enabled"
            )
        
        # Import here to avoid circular imports
        from ...analyzers.content_analyzer import ContentAnalyzer
        
        analyzer = ContentAnalyzer(config)
        metrics = analyzer.get_analysis_metrics()
        
        return {
            "available_metrics": metrics,
            "total_metrics": len(metrics),
            "description": "List of all available analysis metrics"
        }
        
    except Exception as e:
        logger.error(f"Failed to get analysis metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")


@router.get("/history/{analysis_id}")
async def get_analysis_history(
    analysis_id: str,
    config: SystemConfig = Depends(get_config)
):
    """
    Get analysis history by ID
    
    Retrieves the results of a previous analysis.
    """
    try:
        if not config.features.get("content_analysis", False):
            raise HTTPException(
                status_code=403,
                detail="Content analysis feature is not enabled"
            )
        
        # This would typically query a database or cache
        # For now, return a placeholder response
        return {
            "analysis_id": analysis_id,
            "status": "not_found",
            "message": "Analysis history retrieval not yet implemented"
        }
        
    except Exception as e:
        logger.error(f"Failed to get analysis history: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve analysis history")


@router.delete("/history/{analysis_id}")
async def delete_analysis_history(
    analysis_id: str,
    config: SystemConfig = Depends(get_config)
):
    """
    Delete analysis history by ID
    
    Removes the results of a previous analysis from storage.
    """
    try:
        if not config.features.get("content_analysis", False):
            raise HTTPException(
                status_code=403,
                detail="Content analysis feature is not enabled"
            )
        
        # This would typically delete from a database or cache
        # For now, return a placeholder response
        return {
            "analysis_id": analysis_id,
            "status": "deleted",
            "message": "Analysis history deletion not yet implemented"
        }
        
    except Exception as e:
        logger.error(f"Failed to delete analysis history: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete analysis history")





















