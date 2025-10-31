"""
Super Advanced NLP Routes for AI Document Processor
API routes for super advanced Natural Language Processing features
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
from super_advanced_nlp import super_advanced_nlp_system

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/super-advanced-nlp", tags=["Super Advanced NLP"])

# Pydantic models
class TextInput(BaseModel):
    text: str = Field(..., description="Text to process")
    method: Optional[str] = Field("transformer", description="Processing method")

class ClassificationInput(BaseModel):
    text: str = Field(..., description="Text to classify")
    categories: List[str] = Field(..., description="Available categories")
    method: Optional[str] = Field("transformer", description="Classification method")
    include_confidence: Optional[bool] = Field(True, description="Include confidence scores")

class SentimentInput(BaseModel):
    text: str = Field(..., description="Text to analyze")
    method: Optional[str] = Field("transformer", description="Analysis method")
    include_emotions: Optional[bool] = Field(True, description="Include emotions")
    include_aspects: Optional[bool] = Field(True, description="Include aspect analysis")

class GenerationInput(BaseModel):
    prompt: str = Field(..., description="Prompt for generation")
    method: Optional[str] = Field("transformer", description="Generation method")
    max_length: Optional[int] = Field(100, description="Maximum length")
    temperature: Optional[float] = Field(0.7, description="Temperature for generation")

class QAInput(BaseModel):
    question: str = Field(..., description="Question to answer")
    context: Optional[str] = Field("", description="Context for answering")
    method: Optional[str] = Field("transformer", description="QA method")

class NERInput(BaseModel):
    text: str = Field(..., description="Text to extract entities from")
    method: Optional[str] = Field("transformer", description="NER method")

class SummarizationInput(BaseModel):
    text: str = Field(..., description="Text to summarize")
    method: Optional[str] = Field("transformer", description="Summarization method")
    max_length: Optional[int] = Field(100, description="Maximum summary length")
    include_highlights: Optional[bool] = Field(True, description="Include highlights")

# Model management endpoints
@router.post("/models/transformer/{model_name}")
async def load_transformer_model(model_name: str):
    """Load transformer model"""
    try:
        result = await super_advanced_nlp_system.load_transformer_model(model_name)
        return result
    except Exception as e:
        logger.error(f"Error loading transformer model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/embedding/{model_name}")
async def load_embedding_model(model_name: str):
    """Load embedding model"""
    try:
        result = await super_advanced_nlp_system.load_embedding_model(model_name)
        return result
    except Exception as e:
        logger.error(f"Error loading embedding model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Classification endpoints
@router.post("/classify")
async def super_advanced_text_classification(input_data: ClassificationInput):
    """Super advanced text classification"""
    try:
        result = await super_advanced_nlp_system.super_advanced_text_classification(
            text=input_data.text,
            categories=input_data.categories,
            method=input_data.method,
            include_confidence=input_data.include_confidence
        )
        return result
    except Exception as e:
        logger.error(f"Error in super advanced text classification: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Sentiment analysis endpoints
@router.post("/sentiment")
async def super_advanced_sentiment_analysis(input_data: SentimentInput):
    """Super advanced sentiment analysis"""
    try:
        result = await super_advanced_nlp_system.super_advanced_sentiment_analysis(
            text=input_data.text,
            method=input_data.method,
            include_emotions=input_data.include_emotions,
            include_aspects=input_data.include_aspects
        )
        return result
    except Exception as e:
        logger.error(f"Error in super advanced sentiment analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Text generation endpoints
@router.post("/generate")
async def super_advanced_text_generation(input_data: GenerationInput):
    """Super advanced text generation"""
    try:
        result = await super_advanced_nlp_system.super_advanced_text_generation(
            prompt=input_data.prompt,
            method=input_data.method,
            max_length=input_data.max_length,
            temperature=input_data.temperature
        )
        return result
    except Exception as e:
        logger.error(f"Error in super advanced text generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Question answering endpoints
@router.post("/qa")
async def super_advanced_question_answering(input_data: QAInput):
    """Super advanced question answering"""
    try:
        result = await super_advanced_nlp_system.super_advanced_question_answering(
            question=input_data.question,
            context=input_data.context,
            method=input_data.method
        )
        return result
    except Exception as e:
        logger.error(f"Error in super advanced question answering: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Named entity recognition endpoints
@router.post("/ner")
async def super_advanced_entity_recognition(input_data: NERInput):
    """Super advanced named entity recognition"""
    try:
        result = await super_advanced_nlp_system.super_advanced_entity_recognition(
            text=input_data.text,
            method=input_data.method
        )
        return result
    except Exception as e:
        logger.error(f"Error in super advanced entity recognition: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Text summarization endpoints
@router.post("/summarize")
async def super_advanced_text_summarization(input_data: SummarizationInput):
    """Super advanced text summarization"""
    try:
        result = await super_advanced_nlp_system.super_advanced_text_summarization(
            text=input_data.text,
            method=input_data.method,
            max_length=input_data.max_length,
            include_highlights=input_data.include_highlights
        )
        return result
    except Exception as e:
        logger.error(f"Error in super advanced text summarization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Advanced analysis endpoints
@router.post("/analyze/comprehensive")
async def comprehensive_super_advanced_analysis(text: str):
    """Comprehensive super advanced NLP analysis"""
    try:
        results = {}
        
        # Text classification
        classification_result = await super_advanced_nlp_system.super_advanced_text_classification(
            text=text,
            categories=["technology", "business", "science", "health", "education"],
            method="transformer",
            include_confidence=True
        )
        results["classification"] = classification_result
        
        # Sentiment analysis
        sentiment_result = await super_advanced_nlp_system.super_advanced_sentiment_analysis(
            text=text,
            method="transformer",
            include_emotions=True,
            include_aspects=True
        )
        results["sentiment"] = sentiment_result
        
        # Named entity recognition
        ner_result = await super_advanced_nlp_system.super_advanced_entity_recognition(
            text=text,
            method="transformer"
        )
        results["entities"] = ner_result
        
        # Text summarization
        summarization_result = await super_advanced_nlp_system.super_advanced_text_summarization(
            text=text,
            method="transformer",
            max_length=100,
            include_highlights=True
        )
        results["summarization"] = summarization_result
        
        return {
            "status": "success",
            "comprehensive_analysis": results,
            "text_length": len(text)
        }
        
    except Exception as e:
        logger.error(f"Error in comprehensive super advanced analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Batch processing endpoints
@router.post("/batch/classify")
async def batch_super_advanced_classification(texts: List[str], categories: List[str], method: str = "transformer"):
    """Batch super advanced text classification"""
    try:
        results = []
        for text in texts:
            result = await super_advanced_nlp_system.super_advanced_text_classification(
                text=text,
                categories=categories,
                method=method,
                include_confidence=True
            )
            results.append(result)
        return {
            "status": "success",
            "results": results,
            "total_texts": len(texts)
        }
    except Exception as e:
        logger.error(f"Error in batch super advanced classification: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch/sentiment")
async def batch_super_advanced_sentiment_analysis(texts: List[str], method: str = "transformer"):
    """Batch super advanced sentiment analysis"""
    try:
        results = []
        for text in texts:
            result = await super_advanced_nlp_system.super_advanced_sentiment_analysis(
                text=text,
                method=method,
                include_emotions=True,
                include_aspects=True
            )
            results.append(result)
        return {
            "status": "success",
            "results": results,
            "total_texts": len(texts)
        }
    except Exception as e:
        logger.error(f"Error in batch super advanced sentiment analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch/generate")
async def batch_super_advanced_text_generation(prompts: List[str], method: str = "transformer", max_length: int = 100):
    """Batch super advanced text generation"""
    try:
        results = []
        for prompt in prompts:
            result = await super_advanced_nlp_system.super_advanced_text_generation(
                prompt=prompt,
                method=method,
                max_length=max_length,
                temperature=0.7
            )
            results.append(result)
        return {
            "status": "success",
            "results": results,
            "total_prompts": len(prompts)
        }
    except Exception as e:
        logger.error(f"Error in batch super advanced text generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch/qa")
async def batch_super_advanced_question_answering(questions: List[str], context: str = "", method: str = "transformer"):
    """Batch super advanced question answering"""
    try:
        results = []
        for question in questions:
            result = await super_advanced_nlp_system.super_advanced_question_answering(
                question=question,
                context=context,
                method=method
            )
            results.append(result)
        return {
            "status": "success",
            "results": results,
            "total_questions": len(questions)
        }
    except Exception as e:
        logger.error(f"Error in batch super advanced question answering: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch/ner")
async def batch_super_advanced_entity_recognition(texts: List[str], method: str = "transformer"):
    """Batch super advanced entity recognition"""
    try:
        results = []
        for text in texts:
            result = await super_advanced_nlp_system.super_advanced_entity_recognition(
                text=text,
                method=method
            )
            results.append(result)
        return {
            "status": "success",
            "results": results,
            "total_texts": len(texts)
        }
    except Exception as e:
        logger.error(f"Error in batch super advanced entity recognition: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch/summarize")
async def batch_super_advanced_text_summarization(texts: List[str], method: str = "transformer", max_length: int = 100):
    """Batch super advanced text summarization"""
    try:
        results = []
        for text in texts:
            result = await super_advanced_nlp_system.super_advanced_text_summarization(
                text=text,
                method=method,
                max_length=max_length,
                include_highlights=True
            )
            results.append(result)
        return {
            "status": "success",
            "results": results,
            "total_texts": len(texts)
        }
    except Exception as e:
        logger.error(f"Error in batch super advanced text summarization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Statistics and monitoring endpoints
@router.get("/stats")
async def get_super_advanced_nlp_stats():
    """Get super advanced NLP processing statistics"""
    try:
        result = super_advanced_nlp_system.get_super_advanced_nlp_stats()
        return result
    except Exception as e:
        logger.error(f"Error getting super advanced NLP stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def super_advanced_nlp_health():
    """Super advanced NLP system health check"""
    try:
        stats = super_advanced_nlp_system.get_super_advanced_nlp_stats()
        return {
            "status": "healthy",
            "uptime_seconds": stats["uptime_seconds"],
            "success_rate": stats["success_rate"],
            "total_requests": stats["stats"]["total_super_advanced_requests"],
            "successful_requests": stats["stats"]["successful_super_advanced_requests"],
            "failed_requests": stats["stats"]["failed_super_advanced_requests"],
            "transformer_requests": stats["transformer_requests"],
            "embedding_requests": stats["embedding_requests"],
            "classification_requests": stats["classification_requests"],
            "generation_requests": stats["generation_requests"],
            "qa_requests": stats["qa_requests"],
            "ner_requests": stats["ner_requests"],
            "sentiment_requests": stats["sentiment_requests"]
        }
    except Exception as e:
        logger.error(f"Error in super advanced NLP health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Utility endpoints
@router.get("/methods")
async def get_available_methods():
    """Get available processing methods"""
    return {
        "classification_methods": ["transformer", "ensemble"],
        "sentiment_methods": ["transformer", "rule_based"],
        "generation_methods": ["transformer", "creative"],
        "qa_methods": ["transformer", "retrieval"],
        "ner_methods": ["transformer", "rule_based"],
        "summarization_methods": ["transformer", "extractive"],
        "transformer_models": [
            "bert-base-uncased", "roberta-base", "distilbert-base-uncased",
            "albert-base-v2", "xlnet-base-cased", "electra-base",
            "deberta-base", "bart-base", "t5-base", "gpt2"
        ],
        "embedding_models": [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ]
    }

@router.get("/models/status")
async def get_models_status():
    """Get status of loaded models"""
    try:
        return {
            "transformer_models": {
                model: "loaded" if super_advanced_nlp_system.transformer_models.get(model) is not None else "not_loaded"
                for model in super_advanced_nlp_system.transformer_models
            },
            "embedding_models": {
                model: "loaded" if super_advanced_nlp_system.embedding_models.get(model) is not None else "not_loaded"
                for model in super_advanced_nlp_system.embedding_models
            },
            "classification_models": {
                model: "loaded" if super_advanced_nlp_system.classification_models.get(model) is not None else "not_loaded"
                for model in super_advanced_nlp_system.classification_models
            },
            "generation_models": {
                model: "loaded" if super_advanced_nlp_system.generation_models.get(model) is not None else "not_loaded"
                for model in super_advanced_nlp_system.generation_models
            },
            "qa_models": {
                model: "loaded" if super_advanced_nlp_system.qa_models.get(model) is not None else "not_loaded"
                for model in super_advanced_nlp_system.qa_models
            },
            "ner_models": {
                model: "loaded" if super_advanced_nlp_system.ner_models.get(model) is not None else "not_loaded"
                for model in super_advanced_nlp_system.ner_models
            }
        }
    except Exception as e:
        logger.error(f"Error getting models status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Creative and analytical endpoints
@router.post("/creative/write")
async def creative_writing(prompt: str, style: str = "general", max_length: int = 200):
    """Creative writing generation"""
    try:
        result = await super_advanced_nlp_system.super_advanced_text_generation(
            prompt=f"Write in {style} style: {prompt}",
            method="creative",
            max_length=max_length,
            temperature=0.8
        )
        return result
    except Exception as e:
        logger.error(f"Error in creative writing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analytical/analyze")
async def analytical_analysis(text: str, analysis_type: str = "comprehensive"):
    """Analytical text analysis"""
    try:
        if analysis_type == "comprehensive":
            result = await comprehensive_super_advanced_analysis(text)
        else:
            # Specific analysis types
            if analysis_type == "sentiment":
                result = await super_advanced_nlp_system.super_advanced_sentiment_analysis(
                    text=text,
                    method="transformer",
                    include_emotions=True,
                    include_aspects=True
                )
            elif analysis_type == "entities":
                result = await super_advanced_nlp_system.super_advanced_entity_recognition(
                    text=text,
                    method="transformer"
                )
            elif analysis_type == "classification":
                result = await super_advanced_nlp_system.super_advanced_text_classification(
                    text=text,
                    categories=["technology", "business", "science", "health", "education"],
                    method="transformer",
                    include_confidence=True
                )
            else:
                result = {"error": f"Unknown analysis type: {analysis_type}"}
        
        return result
    except Exception as e:
        logger.error(f"Error in analytical analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))












