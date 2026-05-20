"""
NLP API Endpoints
================

REST API endpoints for the NLP system integrated with the Business Agents platform.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime
import asyncio
import logging

from .nlp_system import nlp_system, TextAnalysisResult, Language, SentimentType, EntityType
from .exceptions import NLPProcessingError, ModelLoadError
from .config import config

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/nlp", tags=["NLP"])

# Pydantic models for API
class TextAnalysisRequest(BaseModel):
    """Request model for text analysis."""
    text: str = Field(..., description="Text to analyze", min_length=1, max_length=10000)
    language: Optional[str] = Field(default="en", description="Language code")
    include_sentiment: bool = Field(default=True, description="Include sentiment analysis")
    include_entities: bool = Field(default=True, description="Include entity extraction")
    include_keywords: bool = Field(default=True, description="Include keyword extraction")
    include_topics: bool = Field(default=False, description="Include topic modeling")
    include_readability: bool = Field(default=True, description="Include readability score")

class TextAnalysisResponse(BaseModel):
    """Response model for text analysis."""
    text: str
    language: str
    sentiment: Optional[Dict[str, Any]] = None
    entities: Optional[List[Dict[str, Any]]] = None
    keywords: Optional[List[str]] = None
    topics: Optional[List[Dict[str, Any]]] = None
    readability_score: Optional[float] = None
    word_count: int
    sentence_count: int
    processing_time: float
    timestamp: datetime

class SentimentAnalysisRequest(BaseModel):
    """Request model for sentiment analysis."""
    text: str = Field(..., description="Text to analyze", min_length=1, max_length=5000)
    language: Optional[str] = Field(default="en", description="Language code")

class SentimentAnalysisResponse(BaseModel):
    """Response model for sentiment analysis."""
    text: str
    sentiment: Dict[str, Any]
    processing_time: float
    timestamp: datetime

class EntityExtractionRequest(BaseModel):
    """Request model for entity extraction."""
    text: str = Field(..., description="Text to analyze", min_length=1, max_length=5000)
    language: Optional[str] = Field(default="en", description="Language code")
    entity_types: Optional[List[str]] = Field(default=None, description="Specific entity types to extract")

class EntityExtractionResponse(BaseModel):
    """Response model for entity extraction."""
    text: str
    entities: List[Dict[str, Any]]
    processing_time: float
    timestamp: datetime

class KeywordExtractionRequest(BaseModel):
    """Request model for keyword extraction."""
    text: str = Field(..., description="Text to analyze", min_length=1, max_length=5000)
    language: Optional[str] = Field(default="en", description="Language code")
    top_k: int = Field(default=10, description="Number of top keywords to return", ge=1, le=50)

class KeywordExtractionResponse(BaseModel):
    """Response model for keyword extraction."""
    text: str
    keywords: List[str]
    processing_time: float
    timestamp: datetime

class TopicModelingRequest(BaseModel):
    """Request model for topic modeling."""
    texts: List[str] = Field(..., description="List of texts to analyze", min_items=1, max_items=100)
    language: Optional[str] = Field(default="en", description="Language code")
    n_topics: int = Field(default=5, description="Number of topics to extract", ge=2, le=20)

class TopicModelingResponse(BaseModel):
    """Response model for topic modeling."""
    topics: List[Dict[str, Any]]
    processing_time: float
    timestamp: datetime

class TextClassificationRequest(BaseModel):
    """Request model for text classification."""
    text: str = Field(..., description="Text to classify", min_length=1, max_length=5000)
    categories: List[str] = Field(..., description="Categories to classify into", min_items=1, max_items=10)
    language: Optional[str] = Field(default="en", description="Language code")

class TextClassificationResponse(BaseModel):
    """Response model for text classification."""
    text: str
    classifications: Dict[str, float]
    processing_time: float
    timestamp: datetime

class TextSummarizationRequest(BaseModel):
    """Request model for text summarization."""
    text: str = Field(..., description="Text to summarize", min_length=1, max_length=10000)
    max_length: int = Field(default=150, description="Maximum length of summary", ge=50, le=500)
    language: Optional[str] = Field(default="en", description="Language code")

class TextSummarizationResponse(BaseModel):
    """Response model for text summarization."""
    original_text: str
    summary: str
    compression_ratio: float
    processing_time: float
    timestamp: datetime

class TextTranslationRequest(BaseModel):
    """Request model for text translation."""
    text: str = Field(..., description="Text to translate", min_length=1, max_length=5000)
    target_language: str = Field(..., description="Target language code")
    source_language: Optional[str] = Field(default="en", description="Source language code")

class TextTranslationResponse(BaseModel):
    """Response model for text translation."""
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    processing_time: float
    timestamp: datetime

class BatchAnalysisRequest(BaseModel):
    """Request model for batch text analysis."""
    texts: List[str] = Field(..., description="List of texts to analyze", min_items=1, max_items=50)
    language: Optional[str] = Field(default="en", description="Language code")
    include_sentiment: bool = Field(default=True, description="Include sentiment analysis")
    include_entities: bool = Field(default=True, description="Include entity extraction")
    include_keywords: bool = Field(default=True, description="Include keyword extraction")

class BatchAnalysisResponse(BaseModel):
    """Response model for batch text analysis."""
    results: List[TextAnalysisResponse]
    total_processing_time: float
    timestamp: datetime

# Dependency to ensure NLP system is initialized
async def get_nlp_system():
    """Get initialized NLP system."""
    if not nlp_system.is_initialized:
        await nlp_system.initialize()
    return nlp_system

# API Endpoints

@router.get("/health", response_model=Dict[str, Any])
async def get_nlp_health():
    """Get NLP system health status."""
    try:
        health = await nlp_system.get_system_health()
        return health
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {e}")

@router.post("/analyze", response_model=TextAnalysisResponse)
async def analyze_text(
    request: TextAnalysisRequest,
    nlp: NLPSystem = Depends(get_nlp_system)
):
    """Perform comprehensive text analysis."""
    try:
        start_time = datetime.now()
        
        # Perform text analysis
        result = await nlp.analyze_text(
            text=request.text,
            language=request.language or "en"
        )
        
        # Filter results based on request parameters
        response_data = {
            "text": result.text,
            "language": result.language,
            "word_count": result.word_count,
            "sentence_count": result.sentence_count,
            "processing_time": result.processing_time,
            "timestamp": result.timestamp
        }
        
        if request.include_sentiment:
            response_data["sentiment"] = result.sentiment
        if request.include_entities:
            response_data["entities"] = result.entities
        if request.include_keywords:
            response_data["keywords"] = result.keywords
        if request.include_topics:
            response_data["topics"] = result.topics
        if request.include_readability:
            response_data["readability_score"] = result.readability_score
        
        return TextAnalysisResponse(**response_data)
        
    except NLPProcessingError as e:
        logger.error(f"NLP processing error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Text analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Text analysis failed: {e}")

@router.post("/sentiment", response_model=SentimentAnalysisResponse)
async def analyze_sentiment(
    request: SentimentAnalysisRequest,
    nlp: NLPSystem = Depends(get_nlp_system)
):
    """Analyze sentiment of text."""
    try:
        start_time = datetime.now()
        
        sentiment = await nlp.analyze_sentiment(
            text=request.text,
            language=request.language or "en"
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return SentimentAnalysisResponse(
            text=request.text,
            sentiment=sentiment,
            processing_time=processing_time,
            timestamp=start_time
        )
        
    except NLPProcessingError as e:
        logger.error(f"Sentiment analysis error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {e}")

@router.post("/entities", response_model=EntityExtractionResponse)
async def extract_entities(
    request: EntityExtractionRequest,
    nlp: NLPSystem = Depends(get_nlp_system)
):
    """Extract named entities from text."""
    try:
        start_time = datetime.now()
        
        entities = await nlp.extract_entities(
            text=request.text,
            language=request.language or "en"
        )
        
        # Filter entities by type if specified
        if request.entity_types:
            entities = [
                entity for entity in entities 
                if entity.get('label') in request.entity_types
            ]
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return EntityExtractionResponse(
            text=request.text,
            entities=entities,
            processing_time=processing_time,
            timestamp=start_time
        )
        
    except NLPProcessingError as e:
        logger.error(f"Entity extraction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Entity extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Entity extraction failed: {e}")

@router.post("/keywords", response_model=KeywordExtractionResponse)
async def extract_keywords(
    request: KeywordExtractionRequest,
    nlp: NLPSystem = Depends(get_nlp_system)
):
    """Extract keywords from text."""
    try:
        start_time = datetime.now()
        
        keywords = await nlp.extract_keywords(
            text=request.text,
            language=request.language or "en",
            top_k=request.top_k
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return KeywordExtractionResponse(
            text=request.text,
            keywords=keywords,
            processing_time=processing_time,
            timestamp=start_time
        )
        
    except NLPProcessingError as e:
        logger.error(f"Keyword extraction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Keyword extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Keyword extraction failed: {e}")

@router.post("/topics", response_model=TopicModelingResponse)
async def extract_topics(
    request: TopicModelingRequest,
    nlp: NLPSystem = Depends(get_nlp_system)
):
    """Extract topics from a collection of texts."""
    try:
        start_time = datetime.now()
        
        topics = await nlp.extract_topics(
            texts=request.texts,
            language=request.language or "en",
            n_topics=request.n_topics
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return TopicModelingResponse(
            topics=topics,
            processing_time=processing_time,
            timestamp=start_time
        )
        
    except NLPProcessingError as e:
        logger.error(f"Topic modeling error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Topic modeling failed: {e}")
        raise HTTPException(status_code=500, detail=f"Topic modeling failed: {e}")

@router.post("/classify", response_model=TextClassificationResponse)
async def classify_text(
    request: TextClassificationRequest,
    nlp: NLPSystem = Depends(get_nlp_system)
):
    """Classify text into predefined categories."""
    try:
        start_time = datetime.now()
        
        classifications = await nlp.classify_text(
            text=request.text,
            categories=request.categories,
            language=request.language or "en"
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return TextClassificationResponse(
            text=request.text,
            classifications=classifications,
            processing_time=processing_time,
            timestamp=start_time
        )
        
    except NLPProcessingError as e:
        logger.error(f"Text classification error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Text classification failed: {e}")
        raise HTTPException(status_code=500, detail=f"Text classification failed: {e}")

@router.post("/summarize", response_model=TextSummarizationResponse)
async def summarize_text(
    request: TextSummarizationRequest,
    nlp: NLPSystem = Depends(get_nlp_system)
):
    """Summarize text."""
    try:
        start_time = datetime.now()
        
        summary = await nlp.summarize_text(
            text=request.text,
            max_length=request.max_length,
            language=request.language or "en"
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        compression_ratio = len(summary) / len(request.text) if request.text else 0
        
        return TextSummarizationResponse(
            original_text=request.text,
            summary=summary,
            compression_ratio=compression_ratio,
            processing_time=processing_time,
            timestamp=start_time
        )
        
    except NLPProcessingError as e:
        logger.error(f"Text summarization error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Text summarization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Text summarization failed: {e}")

@router.post("/translate", response_model=TextTranslationResponse)
async def translate_text(
    request: TextTranslationRequest,
    nlp: NLPSystem = Depends(get_nlp_system)
):
    """Translate text to target language."""
    try:
        start_time = datetime.now()
        
        translated_text = await nlp.translate_text(
            text=request.text,
            target_language=request.target_language,
            source_language=request.source_language or "en"
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return TextTranslationResponse(
            original_text=request.text,
            translated_text=translated_text,
            source_language=request.source_language or "en",
            target_language=request.target_language,
            processing_time=processing_time,
            timestamp=start_time
        )
        
    except NLPProcessingError as e:
        logger.error(f"Text translation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Text translation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Text translation failed: {e}")

@router.post("/batch", response_model=BatchAnalysisResponse)
async def batch_analyze_texts(
    request: BatchAnalysisRequest,
    nlp: NLPSystem = Depends(get_nlp_system)
):
    """Perform batch text analysis."""
    try:
        start_time = datetime.now()
        results = []
        
        # Process texts in parallel
        tasks = []
        for text in request.texts:
            task = nlp.analyze_text(
                text=text,
                language=request.language or "en"
            )
            tasks.append(task)
        
        # Wait for all analyses to complete
        analysis_results = await asyncio.gather(*tasks)
        
        # Convert to response format
        for result in analysis_results:
            response_data = {
                "text": result.text,
                "language": result.language,
                "word_count": result.word_count,
                "sentence_count": result.sentence_count,
                "processing_time": result.processing_time,
                "timestamp": result.timestamp
            }
            
            if request.include_sentiment:
                response_data["sentiment"] = result.sentiment
            if request.include_entities:
                response_data["entities"] = result.entities
            if request.include_keywords:
                response_data["keywords"] = result.keywords
            
            results.append(TextAnalysisResponse(**response_data))
        
        total_processing_time = (datetime.now() - start_time).total_seconds()
        
        return BatchAnalysisResponse(
            results=results,
            total_processing_time=total_processing_time,
            timestamp=start_time
        )
        
    except NLPProcessingError as e:
        logger.error(f"Batch analysis error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {e}")

@router.get("/languages", response_model=List[str])
async def get_supported_languages():
    """Get list of supported languages."""
    return [lang.value for lang in Language]

@router.get("/entity-types", response_model=List[str])
async def get_entity_types():
    """Get list of supported entity types."""
    return [entity_type.value for entity_type in EntityType]

@router.get("/sentiment-types", response_model=List[str])
async def get_sentiment_types():
    """Get list of supported sentiment types."""
    return [sentiment.value for sentiment in SentimentType]

# Utility endpoints
@router.post("/detect-language")
async def detect_language(
    text: str = Field(..., description="Text to analyze", min_length=1, max_length=1000),
    nlp: NLPSystem = Depends(get_nlp_system)
):
    """Detect the language of text."""
    try:
        language = await nlp.detect_language(text)
        return {"text": text, "language": language, "timestamp": datetime.now()}
    except Exception as e:
        logger.error(f"Language detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Language detection failed: {e}")

@router.post("/readability")
async def calculate_readability(
    text: str = Field(..., description="Text to analyze", min_length=1, max_length=10000),
    language: str = Field(default="en", description="Language code"),
    nlp: NLPSystem = Depends(get_nlp_system)
):
    """Calculate readability score for text."""
    try:
        readability_score = await nlp.calculate_readability(text, language)
        return {
            "text": text,
            "readability_score": readability_score,
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Readability calculation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Readability calculation failed: {e}")












