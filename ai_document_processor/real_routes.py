"""
Real AI Document Processor Routes
Functional API endpoints for document processing
"""

import logging
from fastapi import APIRouter, HTTPException, Form, File, UploadFile
from fastapi.responses import JSONResponse
from typing import Optional, List
import asyncio
from real_ai_processor import real_ai_processor

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/real-documents", tags=["Real Document Processing"])

@router.post("/process-text")
async def process_text(
    text: str = Form(...),
    task: str = Form("analyze")
):
    """Process text with real AI capabilities"""
    try:
        result = await real_ai_processor.process_document(text, task)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error processing text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze-sentiment")
async def analyze_sentiment(
    text: str = Form(...)
):
    """Analyze sentiment of text"""
    try:
        result = await real_ai_processor.process_document(text, "sentiment")
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
        result = await real_ai_processor.process_document(text, "classify")
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
        result = await real_ai_processor.process_document(text, "summarize")
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
        result = await real_ai_processor.extract_keywords(text, top_n)
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
        result = await real_ai_processor.detect_language(text)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error detecting language: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/answer-question")
async def answer_question(
    context: str = Form(...),
    question: str = Form(...)
):
    """Answer a question about the document"""
    try:
        result = await real_ai_processor.answer_question(context, question)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Real AI Document Processor",
        "version": "1.0.0",
        "features": {
            "spacy": real_ai_processor.nlp_model is not None,
            "nltk": real_ai_processor.sentiment_analyzer is not None,
            "transformers": real_ai_processor.classifier is not None,
            "summarizer": real_ai_processor.summarizer is not None,
            "qa_pipeline": real_ai_processor.qa_pipeline is not None
        }
    }

@router.get("/capabilities")
async def get_capabilities():
    """Get available capabilities"""
    return {
        "capabilities": [
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
        "models_loaded": {
            "spacy": real_ai_processor.nlp_model is not None,
            "nltk_sentiment": real_ai_processor.sentiment_analyzer is not None,
            "transformers_classifier": real_ai_processor.classifier is not None,
            "transformers_summarizer": real_ai_processor.summarizer is not None,
            "transformers_qa": real_ai_processor.qa_pipeline is not None
        }
    }













