"""
Optimized FastAPI Application
============================

Single file containing the complete API.
Optimized for maximum efficiency and minimal complexity.
"""

from fastapi import FastAPI, HTTPException, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any, List
import logging

from .core import AIHistorySystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize system
system = AIHistorySystem()

# Create FastAPI app
app = FastAPI(
    title="AI History Comparison API",
    description="Optimized API for AI history comparison",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "AI History Comparison API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/analyze")
async def analyze_content(
    content: str,
    model: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Analyze content and create history entry.
    
    Args:
        content: Content to analyze
        model: AI model version
        metadata: Optional metadata
        
    Returns:
        Analysis result with entry details
    """
    try:
        if not content or not content.strip():
            raise HTTPException(status_code=400, detail="Content cannot be empty")
        
        if not model or not model.strip():
            raise HTTPException(status_code=400, detail="Model cannot be empty")
        
        # Analyze content
        entry = system.analyze_content(content, model, metadata)
        
        return {
            "success": True,
            "entry": entry.to_dict(),
            "message": "Content analyzed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error analyzing content: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/entries/{entry_id}")
async def get_entry(entry_id: str = Path(..., description="Entry ID")):
    """
    Get entry by ID.
    
    Args:
        entry_id: Entry ID
        
    Returns:
        Entry details
    """
    try:
        entry = system.get_entry(entry_id)
        
        if not entry:
            raise HTTPException(status_code=404, detail="Entry not found")
        
        return {
            "entry": entry.to_dict(),
            "message": "Entry retrieved successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting entry {entry_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/entries")
async def get_entries(
    model: Optional[str] = Query(None, description="Filter by model"),
    days: Optional[int] = Query(7, description="Number of recent days"),
    limit: Optional[int] = Query(100, description="Maximum number of entries")
):
    """
    Get entries with optional filtering.
    
    Args:
        model: Filter by model
        days: Number of recent days
        limit: Maximum number of entries
        
    Returns:
        List of entries
    """
    try:
        entries = system.get_entries(model, days, limit)
        
        return {
            "entries": [entry.to_dict() for entry in entries],
            "total": len(entries),
            "message": f"Retrieved {len(entries)} entries"
        }
        
    except Exception as e:
        logger.error(f"Error getting entries: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare")
async def compare_models(
    entry1_id: str,
    entry2_id: str
):
    """
    Compare two model entries.
    
    Args:
        entry1_id: First entry ID
        entry2_id: Second entry ID
        
    Returns:
        Comparison result
    """
    try:
        if not entry1_id or not entry2_id:
            raise HTTPException(status_code=400, detail="Both entry IDs are required")
        
        # Compare models
        result = system.compare_models(entry1_id, entry2_id)
        
        return {
            "success": True,
            "comparison": result.to_dict(),
            "message": "Models compared successfully"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error comparing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/comparisons")
async def get_comparisons(
    model_a: Optional[str] = Query(None, description="Filter by model A"),
    model_b: Optional[str] = Query(None, description="Filter by model B"),
    days: Optional[int] = Query(7, description="Number of recent days"),
    limit: Optional[int] = Query(50, description="Maximum number of comparisons")
):
    """
    Get comparisons with optional filtering.
    
    Args:
        model_a: Filter by model A
        model_b: Filter by model B
        days: Number of recent days
        limit: Maximum number of comparisons
        
    Returns:
        List of comparisons
    """
    try:
        comparisons = system.get_comparisons(model_a, model_b, days, limit)
        
        return {
            "comparisons": [comp.to_dict() for comp in comparisons],
            "total": len(comparisons),
            "message": f"Retrieved {len(comparisons)} comparisons"
        }
        
    except Exception as e:
        logger.error(f"Error getting comparisons: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """
    Get system statistics.
    
    Returns:
        System statistics
    """
    try:
        stats = system.get_stats()
        
        return {
            "stats": stats,
            "message": "Statistics retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/entries/{entry_id}")
async def delete_entry(entry_id: str = Path(..., description="Entry ID")):
    """
    Delete entry by ID.
    
    Args:
        entry_id: Entry ID
        
    Returns:
        Deletion result
    """
    try:
        deleted = system.delete_entry(entry_id)
        
        if not deleted:
            raise HTTPException(status_code=404, detail="Entry not found")
        
        return {
            "success": True,
            "message": "Entry deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting entry {entry_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/quality/{entry_id}")
async def get_quality_assessment(entry_id: str = Path(..., description="Entry ID")):
    """
    Get quality assessment for entry.
    
    Args:
        entry_id: Entry ID
        
    Returns:
        Quality assessment
    """
    try:
        entry = system.get_entry(entry_id)
        
        if not entry:
            raise HTTPException(status_code=404, detail="Entry not found")
        
        # Simple quality assessment
        quality_score = entry.quality
        
        if quality_score >= 0.8:
            level = "excellent"
        elif quality_score >= 0.6:
            level = "good"
        elif quality_score >= 0.4:
            level = "fair"
        else:
            level = "poor"
        
        # Identify strengths and weaknesses
        strengths = []
        weaknesses = []
        
        if entry.readability >= 0.7:
            strengths.append("high_readability")
        else:
            weaknesses.append("low_readability")
        
        if entry.sentiment >= 0.6:
            strengths.append("positive_sentiment")
        elif entry.sentiment <= 0.4:
            weaknesses.append("negative_sentiment")
        
        if entry.words >= 50:
            strengths.append("adequate_length")
        else:
            weaknesses.append("too_short")
        
        # Get recommendations
        recommendations = []
        if "low_readability" in weaknesses:
            recommendations.append("Improve sentence structure and word choice")
        if "negative_sentiment" in weaknesses:
            recommendations.append("Consider more positive language")
        if "too_short" in weaknesses:
            recommendations.append("Add more detail and context")
        
        assessment = {
            "entry_id": entry_id,
            "quality_score": quality_score,
            "quality_level": level,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "recommendations": recommendations
        }
        
        return {
            "assessment": assessment,
            "message": "Quality assessment completed"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error assessing quality for entry {entry_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)




