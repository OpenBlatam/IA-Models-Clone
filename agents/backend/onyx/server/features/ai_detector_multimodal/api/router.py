"""
Router simple (peon router) para el detector multimodal de IA
Endpoints principales para detección de contenido generado por IA
"""

import logging
import time
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from ..schemas import (
    AIDetectionInput,
    AIDetectionResult,
    BatchDetectionInput,
    BatchDetectionResult,
    HealthResponse
)
from ..core.detector import MultimodalAIDetector

logger = logging.getLogger(__name__)

# Inicializar detector
detector = MultimodalAIDetector()

# Crear router
router = APIRouter(prefix="/ai-detector", tags=["AI Detector Multimodal"])


@router.get("/", response_model=Dict[str, Any])
async def root():
    """Endpoint raíz con información de la API"""
    return {
        "name": "AI Detector Multimodal",
        "version": "1.0.0",
        "description": "Detector de contenido generado por IA con análisis forense",
        "endpoints": {
            "detect": "/ai-detector/detect",
            "batch": "/ai-detector/batch",
            "health": "/ai-detector/health"
        },
        "features": [
            "Detección de texto generado por IA",
            "Detección de imágenes generadas por IA",
            "Detección de audio generado por IA",
            "Detección de video generado por IA",
            "Análisis forense de prompts",
            "Identificación de modelos de IA",
            "Porcentaje de contenido generado por IA"
        ]
    }


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check del servicio"""
    try:
        health_data = detector.get_health()
        return HealthResponse(**health_data)
    except Exception as e:
        logger.error(f"Error en health check: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@router.post("/detect", response_model=AIDetectionResult)
async def detect_ai_content(input_data: AIDetectionInput):
    """
    Detecta si el contenido fue generado por IA
    
    Analiza el contenido y retorna:
    - Si fue generado por IA (boolean)
    - Porcentaje de contenido generado por IA (0-100)
    - Modelos de IA detectados
    - Análisis forense del posible prompt usado
    """
    logger.info(f"Detección solicitada - Tipo: {input_data.content_type}")
    
    try:
        # Convertir contenido si es necesario
        content = input_data.content
        if isinstance(content, bytes):
            content = content.decode('utf-8', errors='ignore')
        
        # Realizar detección
        result = detector.detect(
            content=content,
            content_type=input_data.content_type.value,
            metadata=input_data.metadata
        )
        
        # Convertir a schema de respuesta
        detection_result = AIDetectionResult(**result)
        
        logger.info(
            f"Detección completada - "
            f"Es IA: {detection_result.is_ai_generated}, "
            f"Porcentaje: {detection_result.ai_percentage:.2f}%, "
            f"Confianza: {detection_result.confidence_score:.2f}"
        )
        
        return detection_result
        
    except ValueError as e:
        logger.warning(f"Error de validación: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error en detección: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


@router.post("/batch", response_model=BatchDetectionResult)
async def batch_detect_ai_content(
    input_data: BatchDetectionInput,
    background_tasks: BackgroundTasks
):
    """
    Detecta múltiples contenidos en batch
    
    Procesa una lista de contenidos y retorna resultados para cada uno
    """
    logger.info(f"Detección batch solicitada - Items: {len(input_data.items)}")
    
    try:
        results = []
        successful = 0
        failed = 0
        start_time = time.time()
        
        # Procesar cada item
        for item in input_data.items:
            try:
                # Convertir contenido si es necesario
                content = item.content
                if isinstance(content, bytes):
                    content = content.decode('utf-8', errors='ignore')
                
                # Realizar detección
                result = detector.detect(
                    content=content,
                    content_type=item.content_type.value,
                    metadata=item.metadata
                )
                
                detection_result = AIDetectionResult(**result)
                results.append(detection_result)
                successful += 1
                
            except Exception as e:
                logger.error(f"Error procesando item en batch: {e}")
                failed += 1
                # Agregar resultado de error
                error_result = AIDetectionResult(
                    is_ai_generated=False,
                    ai_percentage=0.0,
                    detected_models=[],
                    primary_model=None,
                    forensic_analysis=None,
                    confidence_score=0.0,
                    detection_methods=["error"],
                    processing_time=0.0,
                    timestamp=time.time()
                )
                results.append(error_result)
        
        processing_time = time.time() - start_time
        
        batch_result = BatchDetectionResult(
            results=results,
            total_processed=len(input_data.items),
            successful=successful,
            failed=failed,
            processing_time=processing_time
        )
        
        logger.info(
            f"Batch completado - "
            f"Total: {batch_result.total_processed}, "
            f"Exitosos: {batch_result.successful}, "
            f"Fallidos: {batch_result.failed}"
        )
        
        return batch_result
        
    except Exception as e:
        logger.error(f"Error en batch: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


@router.post("/detect/text", response_model=AIDetectionResult)
async def detect_text_ai(text: str, metadata: Dict[str, Any] = None):
    """
    Endpoint simplificado para detección de texto
    """
    input_data = AIDetectionInput(
        content=text,
        content_type="text",
        metadata=metadata
    )
    return await detect_ai_content(input_data)


@router.get("/models", response_model=Dict[str, Any])
async def list_detected_models():
    """
    Lista los modelos de IA que el detector puede identificar
    """
    return {
        "models": [
            {
                "name": "gpt-3.5",
                "provider": "OpenAI",
                "description": "GPT-3.5 Turbo"
            },
            {
                "name": "gpt-4",
                "provider": "OpenAI",
                "description": "GPT-4"
            },
            {
                "name": "claude",
                "provider": "Anthropic",
                "description": "Claude"
            },
            {
                "name": "gemini",
                "provider": "Google",
                "description": "Google Gemini"
            },
            {
                "name": "llama",
                "provider": "Meta",
                "description": "LLaMA"
            },
            {
                "name": "mistral",
                "provider": "Mistral AI",
                "description": "Mistral"
            },
            {
                "name": "cohere",
                "provider": "Cohere",
                "description": "Cohere Command"
            },
            {
                "name": "palm",
                "provider": "Google",
                "description": "PaLM"
            },
            {
                "name": "jurassic",
                "provider": "AI21 Labs",
                "description": "Jurassic"
            },
            {
                "name": "groq",
                "provider": "Groq",
                "description": "Groq"
            },
            {
                "name": "openrouter",
                "provider": "OpenRouter",
                "description": "OpenRouter"
            }
        ],
        "total": 11
    }


@router.post("/cache/clear")
async def clear_cache():
    """
    Limpia el cache de detecciones
    """
    logger.info("Limpieza de cache solicitada")
    
    try:
        detector.clear_cache()
        return {
            "success": True,
            "message": "Cache limpiado exitosamente",
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error limpiando cache: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


@router.get("/cache/stats")
async def get_cache_stats():
    """
    Obtiene estadísticas del cache
    """
    logger.info("Estadísticas de cache solicitadas")
    
    try:
        cache_size = len(detector.detection_cache)
        cache_max = detector.cache_max_size
        
        return {
            "success": True,
            "data": {
                "cache_size": cache_size,
                "cache_max_size": cache_max,
                "cache_usage_percent": (cache_size / cache_max * 100) if cache_max > 0 else 0,
                "cache_available": cache_max - cache_size
            },
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas de cache: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


@router.get("/stats", response_model=Dict[str, Any])
async def get_detection_stats():
    """
    Obtiene estadísticas generales del detector
    """
    logger.info("Estadísticas del detector solicitadas")
    
    try:
        health_data = detector.get_health()
        cache_size = len(detector.detection_cache)
        statistics = detector.get_statistics()
        
        return {
            "success": True,
            "data": {
                "health": health_data,
                "cache": {
                    "size": cache_size,
                    "max_size": detector.cache_max_size,
                    "usage_percent": (cache_size / detector.cache_max_size * 100) if detector.cache_max_size > 0 else 0
                },
                "statistics": statistics,
                "capabilities": {
                    "models_supported": 12,
                    "detection_methods": 146,
                    "content_types": ["text", "image", "audio", "video", "multimodal"],
                    "features": [
                        "Pattern Matching",
                        "Statistical Analysis",
                        "Structure Analysis",
                        "Style Analysis",
                        "Entropy Analysis",
                        "Semantic Coherence",
                        "Syntactic Complexity",
                        "Citation Analysis",
                        "Temporal Analysis",
                        "Watermark Detection",
                        "Edit Detection",
                        "Sentiment Analysis",
                        "Contextual Analysis",
                        "Translation Detection",
                        "Generation Patterns",
                        "Writing Quality",
                        "Paraphrase Detection",
                        "Risk Analysis",
                        "Metadata Analysis",
                        "Language Pattern Analysis",
                        "Semantic Similarity Analysis",
                        "Keyword Frequency Analysis",
                        "Response Pattern Detection",
                        "Narrative Coherence Analysis",
                        "Adaptive Weighting System",
                        "Historical Context Analysis",
                        "Advanced N-gram Analysis",
                        "Comparative Similarity Analysis",
                        "Machine Learning Pattern Analysis",
                        "Model Signature Analysis",
                        "Semantic Embedding Analysis",
                        "Alert System",
                        "Temporal Pattern Analysis",
                        "Hybrid Model Detection",
                        "Advanced Frequency Analysis",
                        "Advanced Contextual Coherence",
                        "Text Deepfake Detection",
                        "Advanced Writing Quality Analysis",
                        "Deepfake Pattern Detection",
                        "Enhanced Scoring System",
                        "Advanced Repetition Pattern Analysis",
                        "AI Paraphrasing Advanced Detection",
                        "Style Mixture Detection",
                        "Generation Sophistication Analysis",
                        "Advanced Lexical Diversity Analysis",
                        "AI Hedging Pattern Detection",
                        "Sentence Complexity Distribution Analysis",
                        "AI Verbosity Pattern Detection",
                        "Pronoun Usage Pattern Analysis",
                        "AI Question Pattern Detection",
                        "AI Closure Pattern Analysis",
                        "AI Enumeration Pattern Detection",
                        "AI Metaphor Pattern Analysis",
                        "AI Emphasis Pattern Detection",
                        "AI Modifier Pattern Analysis",
                        "AI Conditional Pattern Detection",
                        "AI Passive Voice Pattern Analysis",
                        "AI Connector Pattern Detection",
                        "AI Quantifier Pattern Analysis",
                        "AI Assertion Pattern Detection",
                        "AI Negation Pattern Analysis",
                        "AI Comparison Pattern Detection",
                        "AI Temporal Marker Pattern Analysis",
                        "AI Causality Pattern Detection",
                        "AI Modal Verb Pattern Analysis",
                        "AI Hedge Phrase Pattern Detection",
                        "AI Relative Clause Pattern Analysis",
                        "AI Infinitive Pattern Detection",
                        "AI Gerund Pattern Analysis",
                        "AI Participle Pattern Detection",
                        "AI Subjunctive Pattern Analysis",
                        "AI Article Pattern Detection",
                        "AI Preposition Pattern Analysis",
                        "AI Conjunction Pattern Detection",
                        "AI Determiner Pattern Analysis",
                        "AI Pronoun Reference Pattern Detection",
                        "AI Adverb Pattern Analysis",
                        "AI Adjective Pattern Detection",
                        "AI Noun Pattern Analysis",
                        "AI Verb Pattern Detection",
                        "AI Sentence Length Pattern Analysis",
                        "AI Paragraph Structure Pattern Detection",
                        "AI Punctuation Pattern Analysis",
                        "AI Capitalization Pattern Detection",
                        "AI Word Frequency Pattern Analysis",
                        "AI Phrase Repetition Pattern Detection",
                        "AI Semantic Density Pattern Analysis",
                        "AI Coherence Markers Pattern Detection",
                        "AI Lexical Sophistication Pattern Analysis",
                        "AI Formality Pattern Detection",
                        "AI Register Pattern Analysis",
                        "AI Discourse Markers Pattern Detection",
                        "AI Textual Cohesion Pattern Analysis",
                        "AI Information Density Pattern Detection",
                        "AI Hedging Density Pattern Analysis",
                        "AI Authorial Voice Pattern Detection",
                        "AI Textual Variety Pattern Analysis",
                        "AI Lexical Repetition Pattern Detection",
                        "AI Syntactic Uniformity Pattern Analysis",
                        "AI Emotional Expression Pattern Detection",
                        "AI Contextual Ambiguity Pattern Analysis",
                        "AI Lexical Richness Pattern Detection",
                        "AI Syntactic Variation Pattern Analysis",
                        "AI Discourse Coherence Pattern Detection",
                        "AI Textual Rhythm Pattern Analysis",
                        "AI Semantic Redundancy Pattern Detection",
                        "AI Lexical Sophistication Advanced Analysis",
                        "AI Pragmatic Markers Pattern Detection",
                        "AI Conversational Pattern Analysis",
                        "AI Metadiscourse Pattern Detection",
                        "AI Evidentiality Pattern Analysis",
                        "AI Engagement Pattern Detection",
                        "AI Politeness Pattern Analysis",
                        "AI Formality Markers Pattern Detection",
                        "AI Hedging Advanced Pattern Analysis",
                        "AI Assertiveness Pattern Detection",
                        "AI Intertextuality Pattern Analysis",
                        "AI Citation Density Pattern Detection",
                        "AI Authority Claims Pattern Analysis",
                        "AI Expertise Markers Pattern Detection",
                        "AI Temporal Coherence Pattern Analysis",
                        "AI Causal Chain Pattern Detection",
                        "AI Narrative Structure Pattern Analysis",
                        "AI Argumentation Pattern Detection",
                        "AI Lexical Consistency Pattern Analysis",
                        "AI Semantic Field Pattern Detection",
                        "AI Register Consistency Pattern Analysis",
                        "AI Stylistic Uniformity Pattern Detection",
                        "AI Phraseology Pattern Analysis",
                        "AI Collocation Pattern Detection",
                        "AI Idiomatic Pattern Analysis",
                        "AI Cultural References Pattern Detection",
                        "AI Metaphorical Pattern Analysis",
                        "AI Analogical Pattern Detection",
                        "AI Irony Pattern Analysis",
                        "AI Humor Pattern Detection",
                        "AI Sarcasm Pattern Analysis",
                        "AI Hyperbole Pattern Detection",
                        "AI Euphemism Pattern Analysis",
                        "AI Understatement Pattern Detection",
                        "AI Alliteration Pattern Analysis",
                        "AI Assonance Pattern Detection",
                        "AI Rhythm Pattern Analysis",
                        "AI Poetic Pattern Detection",
                        "AI Lexical Density Pattern Analysis",
                        "AI Semantic Network Pattern Detection",
                        "AI Conceptual Coherence Pattern Analysis",
                        "AI Knowledge Graph Pattern Detection"
                    ]
                }
            },
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


@router.get("/history", response_model=Dict[str, Any])
async def get_detection_history(limit: int = 50):
    """
    Obtiene el historial de detecciones recientes
    """
    logger.info(f"Historial de detecciones solicitado - Límite: {limit}")
    
    try:
        history = detector.get_detection_history(limit=limit)
        
        return {
            "success": True,
            "data": {
                "history": history,
                "total_entries": len(history),
                "limit": limit
            },
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error obteniendo historial: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


@router.get("/statistics", response_model=Dict[str, Any])
async def get_statistics():
    """
    Obtiene estadísticas detalladas de las detecciones
    """
    logger.info("Estadísticas detalladas solicitadas")
    
    try:
        stats = detector.get_statistics()
        
        return {
            "success": True,
            "data": stats,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


@router.post("/learn")
async def add_known_text(text: str = None, is_ai: bool = None, 
                         data: Dict[str, Any] = None):
    """
    Añade un texto conocido al sistema de aprendizaje
    
    Puede recibir los parámetros directamente o en un JSON body
    """
    # Manejar ambos formatos: query params o body
    if data:
        text = data.get("text", text)
        is_ai = data.get("is_ai", is_ai)
    
    if text is None or is_ai is None:
        raise HTTPException(
            status_code=400, 
            detail="Se requieren los parámetros 'text' y 'is_ai'"
        )
    
    logger.info(f"Añadiendo texto conocido: {'IA' if is_ai else 'Humano'}")
    
    try:
        detector.add_known_text(text, is_ai)
        
        return {
            "success": True,
            "message": f"Texto {'IA' if is_ai else 'humano'} añadido al sistema de aprendizaje",
            "known_ai_texts": len(detector.known_ai_texts),
            "known_human_texts": len(detector.known_human_texts),
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error añadiendo texto conocido: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

