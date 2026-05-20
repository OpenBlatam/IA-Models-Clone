"""
Advanced Real AI Document Processor Routes
More real, working API endpoints for document processing
"""

import logging
from fastapi import APIRouter, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import Optional, List
import asyncio
from advanced_real_processor import advanced_real_processor

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/advanced-real", tags=["Advanced Real Document Processing"])

@router.post("/analyze-text-advanced")
async def analyze_text_advanced(
    text: str = Form(...),
    use_cache: bool = Form(True)
):
    """Analyze text with advanced real AI capabilities"""
    try:
        result = await advanced_real_processor.process_text_advanced(text, "analyze", use_cache)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error in advanced text analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze-sentiment-advanced")
async def analyze_sentiment_advanced(
    text: str = Form(...),
    use_cache: bool = Form(True)
):
    """Analyze sentiment with advanced features"""
    try:
        result = await advanced_real_processor.process_text_advanced(text, "sentiment", use_cache)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error in advanced sentiment analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/classify-text-advanced")
async def classify_text_advanced(
    text: str = Form(...),
    use_cache: bool = Form(True)
):
    """Classify text with advanced features"""
    try:
        result = await advanced_real_processor.process_text_advanced(text, "classify", use_cache)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error in advanced text classification: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/summarize-text-advanced")
async def summarize_text_advanced(
    text: str = Form(...),
    use_cache: bool = Form(True)
):
    """Summarize text with advanced features"""
    try:
        result = await advanced_real_processor.process_text_advanced(text, "summarize", use_cache)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error in advanced text summarization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze-complexity")
async def analyze_complexity(
    text: str = Form(...)
):
    """Analyze text complexity"""
    try:
        result = await advanced_real_processor.process_text_advanced(text, "analyze")
        complexity = result.get("advanced_analysis", {}).get("complexity", {})
        return JSONResponse(content=complexity)
    except Exception as e:
        logger.error(f"Error analyzing complexity: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze-readability")
async def analyze_readability(
    text: str = Form(...)
):
    """Analyze text readability"""
    try:
        result = await advanced_real_processor.process_text_advanced(text, "analyze")
        readability = result.get("advanced_analysis", {}).get("readability", {})
        return JSONResponse(content=readability)
    except Exception as e:
        logger.error(f"Error analyzing readability: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze-language-patterns")
async def analyze_language_patterns(
    text: str = Form(...)
):
    """Analyze language patterns"""
    try:
        result = await advanced_real_processor.process_text_advanced(text, "analyze")
        patterns = result.get("advanced_analysis", {}).get("language_patterns", {})
        return JSONResponse(content=patterns)
    except Exception as e:
        logger.error(f"Error analyzing language patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze-quality-metrics")
async def analyze_quality_metrics(
    text: str = Form(...)
):
    """Analyze text quality metrics"""
    try:
        result = await advanced_real_processor.process_text_advanced(text, "analyze")
        quality = result.get("advanced_analysis", {}).get("quality_metrics", {})
        return JSONResponse(content=quality)
    except Exception as e:
        logger.error(f"Error analyzing quality metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze-keywords-advanced")
async def analyze_keywords_advanced(
    text: str = Form(...)
):
    """Analyze keywords with advanced features"""
    try:
        result = await advanced_real_processor.process_text_advanced(text, "analyze")
        keywords = result.get("advanced_analysis", {}).get("keyword_analysis", {})
        return JSONResponse(content=keywords)
    except Exception as e:
        logger.error(f"Error analyzing keywords: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze-similarity")
async def analyze_similarity(
    text: str = Form(...)
):
    """Analyze text similarity"""
    try:
        result = await advanced_real_processor.process_text_advanced(text, "similarity")
        similarity = result.get("similarity_analysis", {})
        return JSONResponse(content=similarity)
    except Exception as e:
        logger.error(f"Error analyzing similarity: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze-topics")
async def analyze_topics(
    text: str = Form(...)
):
    """Analyze text topics"""
    try:
        result = await advanced_real_processor.process_text_advanced(text, "topics")
        topics = result.get("topic_analysis", {})
        return JSONResponse(content=topics)
    except Exception as e:
        logger.error(f"Error analyzing topics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch-process-advanced")
async def batch_process_advanced(
    texts: List[str] = Form(...),
    task: str = Form("analyze"),
    use_cache: bool = Form(True)
):
    """Process multiple texts with advanced features"""
    try:
        results = []
        for i, text in enumerate(texts):
            try:
                result = await advanced_real_processor.process_text_advanced(text, task, use_cache)
                result["batch_index"] = i
                results.append(result)
            except Exception as e:
                results.append({
                    "batch_index": i,
                    "error": str(e),
                    "status": "error"
                })
        
        return JSONResponse(content={
            "batch_results": results,
            "total_processed": len(results),
            "successful": len([r for r in results if r.get("status") != "error"])
        })
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/processing-stats")
async def get_processing_stats():
    """Get processing statistics"""
    try:
        stats = advanced_real_processor.get_stats()
        return JSONResponse(content=stats)
    except Exception as e:
        logger.error(f"Error getting processing stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clear-cache")
async def clear_cache():
    """Clear processing cache"""
    try:
        await advanced_real_processor.clear_cache()
        return JSONResponse(content={"message": "Cache cleared successfully"})
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health-advanced")
async def health_check_advanced():
    """Advanced health check"""
    try:
        stats = advanced_real_processor.get_stats()
        capabilities = advanced_real_processor.get_capabilities()
        
        return JSONResponse(content={
            "status": "healthy",
            "service": "Advanced Real AI Document Processor",
            "version": "2.0.0",
            "features": {
                "basic_ai": True,
                "advanced_analysis": True,
                "complexity_analysis": True,
                "readability_analysis": True,
                "language_patterns": True,
                "quality_metrics": True,
                "keyword_analysis": True,
                "similarity_analysis": capabilities["similarity_analysis"],
                "topic_analysis": capabilities["topic_analysis"],
                "caching": True,
                "batch_processing": True
            },
            "processing_stats": stats["stats"],
            "models_loaded": stats["models_loaded"]
        })
    except Exception as e:
        logger.error(f"Error in advanced health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/capabilities-advanced")
async def get_advanced_capabilities():
    """Get advanced capabilities"""
    try:
        capabilities = advanced_real_processor.get_capabilities()
        stats = advanced_real_processor.get_stats()
        
        return JSONResponse(content={
            "basic_capabilities": [
                "text_analysis",
                "sentiment_analysis",
                "text_classification",
                "text_summarization",
                "keyword_extraction",
                "language_detection",
                "question_answering",
                "named_entity_recognition",
                "part_of_speech_tagging"
            ],
            "advanced_capabilities": [
                "complexity_analysis",
                "readability_analysis",
                "language_pattern_analysis",
                "quality_metrics",
                "advanced_keyword_analysis",
                "similarity_analysis",
                "topic_analysis",
                "batch_processing",
                "caching",
                "performance_monitoring"
            ],
            "models_loaded": capabilities["models_loaded"],
            "performance": {
                "cache_enabled": True,
                "average_processing_time": stats["stats"]["average_processing_time"],
                "total_requests": stats["stats"]["total_requests"],
                "success_rate": stats["success_rate"],
                "cache_hit_rate": stats["cache_hit_rate"]
            }
        })
    except Exception as e:
        logger.error(f"Error getting advanced capabilities: {e}")
        raise HTTPException(status_code=500, detail=str(e))













