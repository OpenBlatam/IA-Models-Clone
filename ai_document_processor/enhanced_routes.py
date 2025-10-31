"""
Enhanced AI Document Processor Routes
Advanced API endpoints with real, working features
"""

import logging
from fastapi import APIRouter, HTTPException, Form, File, UploadFile
from fastapi.responses import JSONResponse
from typing import Optional, List
import asyncio
from enhanced_ai_processor import enhanced_ai_processor

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/enhanced", tags=["Enhanced Document Processing"])

@router.post("/analyze-text-enhanced")
async def analyze_text_enhanced(
    text: str = Form(...),
    use_cache: bool = Form(True)
):
    """Analyze text with enhanced AI capabilities"""
    try:
        result = await enhanced_ai_processor.process_text_enhanced(text, "analyze", use_cache)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error in enhanced text analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze-sentiment-enhanced")
async def analyze_sentiment_enhanced(
    text: str = Form(...),
    use_cache: bool = Form(True)
):
    """Analyze sentiment with enhanced features"""
    try:
        result = await enhanced_ai_processor.process_text_enhanced(text, "sentiment", use_cache)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error in enhanced sentiment analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/classify-text-enhanced")
async def classify_text_enhanced(
    text: str = Form(...),
    use_cache: bool = Form(True)
):
    """Classify text with enhanced features"""
    try:
        result = await enhanced_ai_processor.process_text_enhanced(text, "classify", use_cache)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error in enhanced text classification: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/summarize-text-enhanced")
async def summarize_text_enhanced(
    text: str = Form(...),
    use_cache: bool = Form(True)
):
    """Summarize text with enhanced features"""
    try:
        result = await enhanced_ai_processor.process_text_enhanced(text, "summarize", use_cache)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error in enhanced text summarization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze-complexity")
async def analyze_complexity(
    text: str = Form(...)
):
    """Analyze text complexity"""
    try:
        result = await enhanced_ai_processor.process_text_enhanced(text, "analyze")
        complexity = result.get("enhanced_analysis", {}).get("complexity", {})
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
        result = await enhanced_ai_processor.process_text_enhanced(text, "analyze")
        readability = result.get("enhanced_analysis", {}).get("readability", {})
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
        result = await enhanced_ai_processor.process_text_enhanced(text, "analyze")
        patterns = result.get("enhanced_analysis", {}).get("language_patterns", {})
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
        result = await enhanced_ai_processor.process_text_enhanced(text, "analyze")
        quality = result.get("enhanced_analysis", {}).get("quality_metrics", {})
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
        result = await enhanced_ai_processor.process_text_enhanced(text, "analyze")
        keywords = result.get("enhanced_analysis", {}).get("keyword_analysis", {})
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
        result = await enhanced_ai_processor.process_text_enhanced(text, "similarity")
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
        result = await enhanced_ai_processor.process_text_enhanced(text, "topics")
        topics = result.get("topic_analysis", {})
        return JSONResponse(content=topics)
    except Exception as e:
        logger.error(f"Error analyzing topics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch-process-enhanced")
async def batch_process_enhanced(
    texts: List[str] = Form(...),
    task: str = Form("analyze"),
    use_cache: bool = Form(True)
):
    """Process multiple texts with enhanced features"""
    try:
        results = []
        for i, text in enumerate(texts):
            try:
                result = await enhanced_ai_processor.process_text_enhanced(text, task, use_cache)
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
        stats = enhanced_ai_processor.get_processing_stats()
        return JSONResponse(content=stats)
    except Exception as e:
        logger.error(f"Error getting processing stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clear-cache")
async def clear_cache():
    """Clear processing cache"""
    try:
        await enhanced_ai_processor.clear_cache()
        return JSONResponse(content={"message": "Cache cleared successfully"})
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health-enhanced")
async def health_check_enhanced():
    """Enhanced health check"""
    try:
        stats = enhanced_ai_processor.get_processing_stats()
        
        return JSONResponse(content={
            "status": "healthy",
            "service": "Enhanced AI Document Processor",
            "version": "2.0.0",
            "features": {
                "basic_ai": True,
                "enhanced_analysis": True,
                "complexity_analysis": True,
                "readability_analysis": True,
                "language_patterns": True,
                "quality_metrics": True,
                "keyword_analysis": True,
                "similarity_analysis": stats["models_loaded"]["tfidf_vectorizer"],
                "topic_analysis": stats["models_loaded"]["tfidf_vectorizer"],
                "caching": stats["redis_available"] or stats["cache_size"] > 0,
                "batch_processing": True
            },
            "processing_stats": stats["stats"],
            "models_loaded": stats["models_loaded"]
        })
    except Exception as e:
        logger.error(f"Error in enhanced health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/capabilities-enhanced")
async def get_enhanced_capabilities():
    """Get enhanced capabilities"""
    try:
        stats = enhanced_ai_processor.get_processing_stats()
        
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
            "enhanced_capabilities": [
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
            "models_loaded": stats["models_loaded"],
            "performance": {
                "cache_enabled": stats["redis_available"] or stats["cache_size"] > 0,
                "average_processing_time": stats["stats"]["average_processing_time"],
                "total_requests": stats["stats"]["total_requests"],
                "cache_hit_rate": stats["stats"]["cache_hits"] / max(1, stats["stats"]["total_requests"])
            }
        })
    except Exception as e:
        logger.error(f"Error getting enhanced capabilities: {e}")
        raise HTTPException(status_code=500, detail=str(e))













