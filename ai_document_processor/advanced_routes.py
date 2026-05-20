"""
Advanced Real AI Document Processor Routes
Enhanced API endpoints with advanced features
"""

import logging
from fastapi import APIRouter, HTTPException, Form, File, UploadFile, Query
from fastapi.responses import JSONResponse
from typing import Optional, List
import asyncio
from advanced_ai_processor import advanced_ai_processor
from document_parser import document_parser

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/advanced-documents", tags=["Advanced Document Processing"])

@router.post("/process-text-advanced")
async def process_text_advanced(
    text: str = Form(...),
    task: str = Form("analyze"),
    use_cache: bool = Form(True)
):
    """Process text with advanced AI capabilities"""
    try:
        result = await advanced_ai_processor.process_document_advanced(text, task, use_cache)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error in advanced text processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload-document")
async def upload_document(
    file: UploadFile = File(...),
    task: str = Form("analyze"),
    use_cache: bool = Form(True)
):
    """Upload and process document file"""
    try:
        # Check if file format is supported
        if not document_parser.is_format_supported(file.filename):
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format: {file.filename}"
            )
        
        # Read file content
        file_content = await file.read()
        
        # Parse document
        parse_result = await document_parser.parse_document(file_content, file.filename)
        
        if not parse_result["parsing_successful"]:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to parse document: {parse_result.get('error', 'Unknown error')}"
            )
        
        # Process with AI
        ai_result = await advanced_ai_processor.process_document_advanced(
            parse_result["text_content"], 
            task, 
            use_cache
        )
        
        # Combine results
        result = {
            "document_info": {
                "filename": parse_result["filename"],
                "file_type": parse_result["file_type"],
                "file_size": parse_result["file_size"],
                "metadata": parse_result["metadata"]
            },
            "ai_analysis": ai_result
        }
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing uploaded document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze-complexity")
async def analyze_complexity(
    text: str = Form(...)
):
    """Analyze text complexity"""
    try:
        result = await advanced_ai_processor.process_document_advanced(text, "analyze")
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
        result = await advanced_ai_processor.process_document_advanced(text, "analyze")
        readability = result.get("advanced_analysis", {}).get("readability", {})
        return JSONResponse(content=readability)
    except Exception as e:
        logger.error(f"Error analyzing readability: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze-similarity")
async def analyze_similarity(
    text: str = Form(...)
):
    """Analyze text similarity"""
    try:
        result = await advanced_ai_processor.process_document_advanced(text, "similarity")
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
        result = await advanced_ai_processor.process_document_advanced(text, "topics")
        topics = result.get("topic_analysis", {})
        return JSONResponse(content=topics)
    except Exception as e:
        logger.error(f"Error analyzing topics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch-process")
async def batch_process(
    texts: List[str] = Form(...),
    task: str = Form("analyze"),
    use_cache: bool = Form(True)
):
    """Process multiple texts in batch"""
    try:
        results = []
        for i, text in enumerate(texts):
            try:
                result = await advanced_ai_processor.process_document_advanced(text, task, use_cache)
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

@router.get("/supported-formats")
async def get_supported_formats():
    """Get list of supported file formats"""
    try:
        formats = document_parser.get_supported_formats()
        return JSONResponse(content={
            "supported_formats": formats,
            "total_formats": len([f for f in formats.values() if f])
        })
    except Exception as e:
        logger.error(f"Error getting supported formats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/processing-stats")
async def get_processing_stats():
    """Get processing statistics"""
    try:
        stats = advanced_ai_processor.get_processing_stats()
        return JSONResponse(content=stats)
    except Exception as e:
        logger.error(f"Error getting processing stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clear-cache")
async def clear_cache():
    """Clear processing cache"""
    try:
        await advanced_ai_processor.clear_cache()
        return JSONResponse(content={"message": "Cache cleared successfully"})
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health-advanced")
async def health_check_advanced():
    """Advanced health check"""
    try:
        stats = advanced_ai_processor.get_processing_stats()
        formats = document_parser.get_supported_formats()
        
        return JSONResponse(content={
            "status": "healthy",
            "service": "Advanced AI Document Processor",
            "version": "2.0.0",
            "features": {
                "basic_ai": advanced_ai_processor.initialized,
                "advanced_analysis": True,
                "caching": stats["redis_available"] or len(stats["cache_size"]) > 0,
                "similarity_analysis": stats["sentence_transformer_available"],
                "topic_analysis": stats["vectorizer_available"],
                "document_parsing": any(formats.values())
            },
            "processing_stats": stats["stats"],
            "supported_formats": formats
        })
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/capabilities-advanced")
async def get_advanced_capabilities():
    """Get advanced capabilities"""
    try:
        stats = advanced_ai_processor.get_processing_stats()
        formats = document_parser.get_supported_formats()
        
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
                "similarity_analysis",
                "topic_analysis",
                "language_pattern_analysis",
                "quality_metrics",
                "batch_processing",
                "document_upload",
                "caching",
                "performance_monitoring"
            ],
            "document_formats": {
                "supported": [fmt for fmt, supported in formats.items() if supported],
                "unsupported": [fmt for fmt, supported in formats.items() if not supported]
            },
            "models_loaded": {
                "spacy": advanced_ai_processor.nlp_model is not None,
                "nltk_sentiment": advanced_ai_processor.sentiment_analyzer is not None,
                "transformers_classifier": advanced_ai_processor.classifier is not None,
                "transformers_summarizer": advanced_ai_processor.summarizer is not None,
                "transformers_qa": advanced_ai_processor.qa_pipeline is not None,
                "sentence_transformer": stats["sentence_transformer_available"],
                "tfidf_vectorizer": stats["vectorizer_available"]
            },
            "performance": {
                "cache_enabled": stats["redis_available"] or stats["cache_size"] > 0,
                "average_processing_time": stats["stats"]["average_processing_time"],
                "total_requests": stats["stats"]["total_requests"],
                "cache_hit_rate": stats["stats"]["cache_hits"] / max(1, stats["stats"]["total_requests"])
            }
        })
    except Exception as e:
        logger.error(f"Error getting capabilities: {e}")
        raise HTTPException(status_code=500, detail=str(e))













