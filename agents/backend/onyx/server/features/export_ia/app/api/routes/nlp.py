"""
NLP API Routes - Rutas de la API NLP
"""

import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ...nlp.core import get_nlp_engine
from ...nlp.models import (
    NLPAnalysisRequest, Language, SentimentType, TextType,
    TextAnalysisResult, SentimentResult, LanguageDetectionResult
)

logger = logging.getLogger(__name__)
router = APIRouter()


class TextAnalysisRequest(BaseModel):
    """Solicitud de análisis de texto."""
    text: str = Field(..., description="Texto a analizar", min_length=1, max_length=10000)
    analysis_types: List[str] = Field(
        default=["sentiment", "language", "entities"],
        description="Tipos de análisis a realizar"
    )
    language: Optional[str] = Field(None, description="Idioma del texto")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parámetros adicionales")


class SentimentAnalysisRequest(BaseModel):
    """Solicitud de análisis de sentimiento."""
    text: str = Field(..., description="Texto a analizar", min_length=1, max_length=10000)


class LanguageDetectionRequest(BaseModel):
    """Solicitud de detección de idioma."""
    text: str = Field(..., description="Texto para detectar idioma", min_length=1, max_length=10000)


class TranslationRequest(BaseModel):
    """Solicitud de traducción."""
    text: str = Field(..., description="Texto a traducir", min_length=1, max_length=10000)
    source_language: str = Field(..., description="Idioma origen")
    target_language: str = Field(..., description="Idioma destino")


class SummarizationRequest(BaseModel):
    """Solicitud de resumen."""
    text: str = Field(..., description="Texto a resumir", min_length=1, max_length=10000)
    max_sentences: int = Field(default=3, description="Máximo número de oraciones", ge=1, le=10)


class TextGenerationRequest(BaseModel):
    """Solicitud de generación de texto."""
    prompt: str = Field(..., description="Prompt para generar texto", min_length=1, max_length=1000)
    template: str = Field(default="summary", description="Template a usar")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parámetros adicionales")


@router.post("/nlp/analyze")
async def analyze_text(request: TextAnalysisRequest):
    """Analizar texto con múltiples técnicas NLP."""
    try:
        nlp_engine = get_nlp_engine()
        
        # Crear solicitud NLP
        nlp_request = NLPAnalysisRequest(
            text=request.text,
            analysis_types=request.analysis_types,
            language=Language(request.language) if request.language else None,
            parameters=request.parameters
        )
        
        # Procesar análisis
        result = await nlp_engine.analyze_text(nlp_request)
        
        return {
            "request_id": result.request_id,
            "results": result.results,
            "processing_time": result.processing_time,
            "success": result.success,
            "error_message": result.error_message,
            "timestamp": result.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error en análisis de texto: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/nlp/sentiment")
async def analyze_sentiment(request: SentimentAnalysisRequest):
    """Analizar sentimiento del texto."""
    try:
        nlp_engine = get_nlp_engine()
        
        # Analizar sentimiento
        result = await nlp_engine.sentiment_analyzer.analyze(request.text)
        
        return {
            "text": result.text,
            "sentiment": result.sentiment.value,
            "confidence": result.confidence,
            "positive_score": result.positive_score,
            "negative_score": result.negative_score,
            "neutral_score": result.neutral_score,
            "emotional_intensity": result.emotional_intensity,
            "timestamp": result.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error en análisis de sentimiento: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/nlp/language")
async def detect_language(request: LanguageDetectionRequest):
    """Detectar idioma del texto."""
    try:
        nlp_engine = get_nlp_engine()
        
        # Detectar idioma
        result = await nlp_engine.language_detector.detect(request.text)
        
        return {
            "text": result.text,
            "detected_language": result.detected_language.value,
            "confidence": result.confidence,
            "alternative_languages": result.alternative_languages,
            "timestamp": result.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error en detección de idioma: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/nlp/translate")
async def translate_text(request: TranslationRequest):
    """Traducir texto."""
    try:
        nlp_engine = get_nlp_engine()
        
        # Traducir texto
        result = await nlp_engine.translator.translate(
            request.text,
            Language(request.source_language),
            Language(request.target_language)
        )
        
        return {
            "original_text": result.original_text,
            "translated_text": result.translated_text,
            "source_language": result.source_language.value,
            "target_language": result.target_language.value,
            "confidence": result.confidence,
            "timestamp": result.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error en traducción: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/nlp/summarize")
async def summarize_text(request: SummarizationRequest):
    """Resumir texto."""
    try:
        nlp_engine = get_nlp_engine()
        
        # Resumir texto
        result = await nlp_engine.summarizer.summarize(request.text, request.max_sentences)
        
        return {
            "original_text": result.original_text,
            "summary": result.summary,
            "compression_ratio": result.compression_ratio,
            "key_points": result.key_points,
            "word_count_original": result.word_count_original,
            "word_count_summary": result.word_count_summary,
            "timestamp": result.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error en resumen: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/nlp/generate")
async def generate_text(request: TextGenerationRequest):
    """Generar texto."""
    try:
        nlp_engine = get_nlp_engine()
        
        # Generar texto
        result = await nlp_engine.text_generator.generate(
            request.prompt,
            request.template,
            **request.parameters
        )
        
        return {
            "prompt": result.prompt,
            "generated_text": result.generated_text,
            "model_used": result.model_used,
            "parameters": result.parameters,
            "confidence": result.confidence,
            "timestamp": result.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error en generación de texto: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/nlp/health")
async def nlp_health_check():
    """Verificar salud del sistema NLP."""
    try:
        nlp_engine = get_nlp_engine()
        health_status = await nlp_engine.health_check()
        
        return health_status
        
    except Exception as e:
        logger.error(f"Error en health check NLP: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/nlp/metrics")
async def get_nlp_metrics():
    """Obtener métricas del sistema NLP."""
    try:
        nlp_engine = get_nlp_engine()
        metrics = await nlp_engine.get_metrics()
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error al obtener métricas NLP: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/nlp/cache/clear")
async def clear_nlp_cache():
    """Limpiar cache del sistema NLP."""
    try:
        nlp_engine = get_nlp_engine()
        await nlp_engine.clear_cache()
        
        return {"message": "Cache NLP limpiado exitosamente"}
        
    except Exception as e:
        logger.error(f"Error al limpiar cache NLP: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/nlp/supported-languages")
async def get_supported_languages():
    """Obtener idiomas soportados."""
    try:
        languages = [lang.value for lang in Language]
        
        return {
            "supported_languages": languages,
            "count": len(languages)
        }
        
    except Exception as e:
        logger.error(f"Error al obtener idiomas soportados: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/nlp/analysis-types")
async def get_analysis_types():
    """Obtener tipos de análisis disponibles."""
    try:
        analysis_types = [
            "sentiment",
            "language",
            "entities",
            "keywords",
            "topics",
            "classification",
            "text_analysis"
        ]
        
        return {
            "analysis_types": analysis_types,
            "count": len(analysis_types)
        }
        
    except Exception as e:
        logger.error(f"Error al obtener tipos de análisis: {e}")
        raise HTTPException(status_code=500, detail=str(e))




