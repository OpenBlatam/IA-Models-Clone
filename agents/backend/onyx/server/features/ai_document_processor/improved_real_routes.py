"""
Improved Real AI Document Processor Routes
Real, working API endpoints for document processing
"""

import logging
from fastapi import APIRouter, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import Optional
import asyncio
from real_working_processor import real_working_processor

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/real", tags=["Real Document Processing"])

@router.post("/analyze-text")
async def analyze_text(
    text: str = Form(...)
):
    """Analyze text with real AI capabilities"""
    try:
        result = await real_working_processor.process_text(text, "analyze")
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error analyzing text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze-sentiment")
async def analyze_sentiment(
    text: str = Form(...)
):
    """Analyze sentiment of text"""
    try:
        result = await real_working_processor.process_text(text, "sentiment")
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/classify-text")
async def classify_text(
    text: str = Form(...)
):
    """Classify text"""
    try:
        result = await real_working_processor.process_text(text, "classify")
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error classifying text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/summarize-text")
async def summarize_text(
    text: str = Form(...)
):
    """Summarize text"""
    try:
        result = await real_working_processor.process_text(text, "summarize")
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error summarizing text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/extract-keywords")
async def extract_keywords(
    text: str = Form(...),
    top_n: int = Form(10)
):
    """Extract keywords from text"""
    try:
        result = real_working_processor.extract_keywords(text, top_n)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error extracting keywords: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/detect-language")
async def detect_language(
    text: str = Form(...)
):
    """Detect language of text"""
    try:
        result = real_working_processor.detect_language(text)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error detecting language: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        capabilities = real_working_processor.get_capabilities()
        stats = real_working_processor.get_stats()
        
        return JSONResponse(content={
            "status": "healthy",
            "service": "Real Working AI Document Processor",
            "version": "1.0.0",
            "capabilities": capabilities,
            "stats": stats,
            "timestamp": real_working_processor._generate_text_id("health_check")
        })
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/capabilities")
async def get_capabilities():
    """Get available capabilities"""
    try:
        capabilities = real_working_processor.get_capabilities()
        return JSONResponse(content=capabilities)
    except Exception as e:
        logger.error(f"Error getting capabilities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_stats():
    """Get processing statistics"""
    try:
        stats = real_working_processor.get_stats()
        return JSONResponse(content=stats)
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))













