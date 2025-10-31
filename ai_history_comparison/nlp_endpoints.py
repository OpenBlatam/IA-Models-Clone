"""
NLP API Endpoints for AI History Comparison System
Endpoints de API NLP para el sistema de análisis de historial de IA
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router for NLP endpoints
router = APIRouter(prefix="/nlp", tags=["NLP Analysis"])

# Pydantic models for API
class TextAnalysisRequest(BaseModel):
    text: str = Field(..., description="Texto a analizar")
    document_id: str = Field(..., description="ID del documento")
    analysis_types: Optional[List[str]] = Field(None, description="Tipos de análisis a realizar")
    language: Optional[str] = Field("auto", description="Idioma del texto")

class BatchAnalysisRequest(BaseModel):
    texts: List[Dict[str, str]] = Field(..., description="Lista de textos con IDs")
    analysis_types: Optional[List[str]] = Field(None, description="Tipos de análisis")
    priority: Optional[str] = Field("normal", description="Prioridad de procesamiento")

class PatternAnalysisRequest(BaseModel):
    documents: List[Dict[str, Any]] = Field(..., description="Documentos a analizar")
    pattern_types: Optional[List[str]] = Field(None, description="Tipos de patrones")
    enable_clustering: Optional[bool] = Field(True, description="Habilitar clustering")
    enable_evolution: Optional[bool] = Field(True, description="Habilitar análisis de evolución")

class TextComparisonRequest(BaseModel):
    text1: str = Field(..., description="Primer texto")
    text2: str = Field(..., description="Segundo texto")
    document_id1: str = Field(..., description="ID del primer documento")
    document_id2: str = Field(..., description="ID del segundo documento")

# ============================================================================
# NLP ENGINE ENDPOINTS
# ============================================================================

@router.post("/analyze")
async def analyze_text(request: TextAnalysisRequest):
    """Analizar texto individual"""
    try:
        from nlp_engine import AdvancedNLPEngine, AnalysisType, LanguageType
        
        # Inicializar motor NLP
        language = LanguageType.AUTO if request.language == "auto" else LanguageType(request.language)
        nlp_engine = AdvancedNLPEngine(language=language)
        
        # Convertir tipos de análisis
        analysis_types = None
        if request.analysis_types:
            analysis_types = [AnalysisType(at) for at in request.analysis_types]
        
        # Analizar texto
        analysis = await nlp_engine.analyze_text(
            text=request.text,
            document_id=request.document_id,
            analysis_types=analysis_types
        )
        
        return {
            "success": True,
            "analysis": {
                "document_id": analysis.document_id,
                "language": analysis.language,
                "tokens": [
                    {
                        "text": token.text,
                        "pos": token.pos,
                        "lemma": token.lemma,
                        "is_stop": token.is_stop,
                        "is_punct": token.is_punct
                    }
                    for token in analysis.tokens
                ],
                "entities": [
                    {
                        "text": entity.text,
                        "label": entity.label,
                        "start": entity.start,
                        "end": entity.end,
                        "confidence": entity.confidence
                    }
                    for entity in analysis.entities
                ],
                "sentiment": {
                    "polarity": analysis.sentiment.polarity,
                    "subjectivity": analysis.sentiment.subjectivity,
                    "sentiment_type": analysis.sentiment.sentiment_type.value,
                    "confidence": analysis.sentiment.confidence,
                    "emotional_tone": analysis.sentiment.emotional_tone,
                    "intensity": analysis.sentiment.intensity
                } if analysis.sentiment else None,
                "keywords": [
                    {
                        "text": kw.text,
                        "score": kw.score,
                        "frequency": kw.frequency,
                        "tfidf_score": kw.tfidf_score
                    }
                    for kw in analysis.keywords
                ],
                "metrics": {
                    "word_count": analysis.metrics.word_count,
                    "sentence_count": analysis.metrics.sentence_count,
                    "character_count": analysis.metrics.character_count,
                    "avg_word_length": analysis.metrics.avg_word_length,
                    "avg_sentence_length": analysis.metrics.avg_sentence_length,
                    "lexical_diversity": analysis.metrics.lexical_diversity,
                    "readability_score": analysis.metrics.readability_score,
                    "complexity_score": analysis.metrics.complexity_score,
                    "formality_score": analysis.metrics.formality_score
                },
                "topics": [
                    {
                        "id": topic.id,
                        "name": topic.name,
                        "keywords": topic.keywords,
                        "weight": topic.weight,
                        "coherence": topic.coherence
                    }
                    for topic in analysis.topics
                ],
                "analyzed_at": analysis.analyzed_at.isoformat()
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error analyzing text: {e}")
        return {"error": str(e)}

@router.post("/analyze/batch")
async def analyze_texts_batch(request: BatchAnalysisRequest):
    """Analizar múltiples textos en lote"""
    try:
        from text_processor import RealTimeTextProcessor, ProcessingPriority
        
        # Inicializar procesador
        processor = RealTimeTextProcessor()
        await processor.start()
        
        # Convertir prioridad
        priority = ProcessingPriority.NORMAL
        if request.priority == "high":
            priority = ProcessingPriority.HIGH
        elif request.priority == "critical":
            priority = ProcessingPriority.CRITICAL
        elif request.priority == "low":
            priority = ProcessingPriority.LOW
        
        # Enviar tareas
        task_ids = []
        for text_data in request.texts:
            text = text_data.get("text", "")
            doc_id = text_data.get("document_id", f"doc_{len(task_ids)}")
            
            task_id = await processor.submit_task(
                text=text,
                document_id=doc_id,
                priority=priority
            )
            task_ids.append(task_id)
        
        # Esperar completación
        results = await processor.wait_for_completion(task_ids, timeout=300)
        
        # Detener procesador
        await processor.stop()
        
        return {
            "success": True,
            "task_ids": task_ids,
            "results": {
                task_id: {
                    "status": task.status.value,
                    "result": task.result,
                    "error": task.error,
                    "processing_time": (task.completed_at - task.started_at).total_seconds() if task.completed_at and task.started_at else None
                }
                for task_id, task in results.items()
            },
            "total_tasks": len(task_ids),
            "completed_tasks": len(results),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in batch analysis: {e}")
        return {"error": str(e)}

@router.post("/compare")
async def compare_texts(request: TextComparisonRequest):
    """Comparar dos textos"""
    try:
        from nlp_engine import AdvancedNLPEngine, LanguageType
        
        # Inicializar motor NLP
        nlp_engine = AdvancedNLPEngine(language=LanguageType.AUTO)
        
        # Comparar textos
        comparison = await nlp_engine.compare_texts(
            text1=request.text1,
            text2=request.text2
        )
        
        return {
            "success": True,
            "comparison": comparison,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error comparing texts: {e}")
        return {"error": str(e)}

@router.get("/analysis/{document_id}")
async def get_analysis(document_id: str):
    """Obtener análisis de un documento"""
    try:
        from nlp_engine import AdvancedNLPEngine
        
        nlp_engine = AdvancedNLPEngine()
        
        # Buscar análisis
        if document_id in nlp_engine.analyses:
            analysis = nlp_engine.analyses[document_id]
            return {
                "success": True,
                "analysis": {
                    "document_id": analysis.document_id,
                    "language": analysis.language,
                    "sentiment": {
                        "polarity": analysis.sentiment.polarity,
                        "sentiment_type": analysis.sentiment.sentiment_type.value,
                        "confidence": analysis.sentiment.confidence
                    } if analysis.sentiment else None,
                    "metrics": {
                        "word_count": analysis.metrics.word_count,
                        "readability_score": analysis.metrics.readability_score,
                        "complexity_score": analysis.metrics.complexity_score
                    },
                    "analyzed_at": analysis.analyzed_at.isoformat()
                },
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "success": False,
                "message": f"Analysis for document {document_id} not found"
            }
        
    except Exception as e:
        logger.error(f"Error getting analysis: {e}")
        return {"error": str(e)}

@router.get("/summary")
async def get_nlp_summary():
    """Obtener resumen de análisis NLP"""
    try:
        from nlp_engine import AdvancedNLPEngine
        
        nlp_engine = AdvancedNLPEngine()
        summary = await nlp_engine.get_analysis_summary()
        
        return {
            "success": True,
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting NLP summary: {e}")
        return {"error": str(e)}

# ============================================================================
# PATTERN ANALYSIS ENDPOINTS
# ============================================================================

@router.post("/patterns/analyze")
async def analyze_patterns(request: PatternAnalysisRequest):
    """Analizar patrones en documentos"""
    try:
        from pattern_analyzer import TextPatternAnalyzer, PatternType
        
        # Inicializar analizador
        analyzer = TextPatternAnalyzer()
        
        # Convertir tipos de patrones
        pattern_types = None
        if request.pattern_types:
            pattern_types = [PatternType(pt) for pt in request.pattern_types]
        
        # Analizar patrones
        results = await analyzer.analyze_documents(
            documents=request.documents,
            pattern_types=pattern_types
        )
        
        return {
            "success": True,
            "pattern_analysis": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error analyzing patterns: {e}")
        return {"error": str(e)}

@router.get("/patterns/summary")
async def get_patterns_summary():
    """Obtener resumen de patrones"""
    try:
        from pattern_analyzer import TextPatternAnalyzer
        
        analyzer = TextPatternAnalyzer()
        summary = await analyzer.get_pattern_summary()
        
        return {
            "success": True,
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting patterns summary: {e}")
        return {"error": str(e)}

@router.get("/patterns/similar/{pattern_id}")
async def get_similar_patterns(
    pattern_id: str,
    limit: int = Query(5, description="Número máximo de patrones similares")
):
    """Obtener patrones similares"""
    try:
        from pattern_analyzer import TextPatternAnalyzer
        
        analyzer = TextPatternAnalyzer()
        similar_patterns = await analyzer.find_similar_patterns(pattern_id, limit)
        
        return {
            "success": True,
            "similar_patterns": [
                {
                    "id": pattern.id,
                    "name": pattern.name,
                    "type": pattern.type.value,
                    "category": pattern.category.value,
                    "frequency": pattern.frequency,
                    "confidence": pattern.confidence
                }
                for pattern in similar_patterns
            ],
            "total_similar": len(similar_patterns),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting similar patterns: {e}")
        return {"error": str(e)}

@router.get("/patterns/export")
async def export_patterns():
    """Exportar patrones a archivo"""
    try:
        from pattern_analyzer import TextPatternAnalyzer
        
        analyzer = TextPatternAnalyzer()
        filepath = await analyzer.export_patterns()
        
        return FileResponse(
            path=filepath,
            filename=f"patterns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            media_type="application/json"
        )
        
    except Exception as e:
        logger.error(f"Error exporting patterns: {e}")
        return {"error": str(e)}

# ============================================================================
# TEXT PROCESSING ENDPOINTS
# ============================================================================

@router.get("/processor/status")
async def get_processor_status():
    """Obtener estado del procesador de texto"""
    try:
        from text_processor import RealTimeTextProcessor
        
        processor = RealTimeTextProcessor()
        stats = await processor.get_processing_statistics()
        
        return {
            "success": True,
            "status": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting processor status: {e}")
        return {"error": str(e)}

@router.get("/processor/queue")
async def get_queue_status():
    """Obtener estado de las colas de procesamiento"""
    try:
        from text_processor import RealTimeTextProcessor
        
        processor = RealTimeTextProcessor()
        queue_status = await processor.get_queue_status()
        
        return {
            "success": True,
            "queue_status": queue_status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting queue status: {e}")
        return {"error": str(e)}

@router.post("/processor/start")
async def start_processor():
    """Iniciar procesador de texto"""
    try:
        from text_processor import RealTimeTextProcessor
        
        processor = RealTimeTextProcessor()
        await processor.start()
        
        return {
            "success": True,
            "message": "Text processor started successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error starting processor: {e}")
        return {"error": str(e)}

@router.post("/processor/stop")
async def stop_processor():
    """Detener procesador de texto"""
    try:
        from text_processor import RealTimeTextProcessor
        
        processor = RealTimeTextProcessor()
        await processor.stop()
        
        return {
            "success": True,
            "message": "Text processor stopped successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error stopping processor: {e}")
        return {"error": str(e)}

# ============================================================================
# LANGUAGE DETECTION ENDPOINTS
# ============================================================================

@router.post("/language/detect")
async def detect_language(text: str):
    """Detectar idioma del texto"""
    try:
        from nlp_engine import AdvancedNLPEngine
        
        nlp_engine = AdvancedNLPEngine()
        language = nlp_engine._detect_language(text)
        
        return {
            "success": True,
            "language": language,
            "confidence": 0.8,  # Placeholder
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error detecting language: {e}")
        return {"error": str(e)}

@router.post("/language/detect/batch")
async def detect_languages_batch(texts: List[str]):
    """Detectar idiomas de múltiples textos"""
    try:
        from nlp_engine import AdvancedNLPEngine
        
        nlp_engine = AdvancedNLPEngine()
        results = []
        
        for i, text in enumerate(texts):
            language = nlp_engine._detect_language(text)
            results.append({
                "index": i,
                "text": text[:100] + "..." if len(text) > 100 else text,
                "language": language,
                "confidence": 0.8
            })
        
        return {
            "success": True,
            "results": results,
            "total_texts": len(texts),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error detecting languages: {e}")
        return {"error": str(e)}

# ============================================================================
# SENTIMENT ANALYSIS ENDPOINTS
# ============================================================================

@router.post("/sentiment/analyze")
async def analyze_sentiment(text: str):
    """Analizar sentimiento del texto"""
    try:
        from nlp_engine import AdvancedNLPEngine
        
        nlp_engine = AdvancedNLPEngine()
        
        # Crear análisis temporal
        temp_analysis = await nlp_engine.analyze_text(text, "temp_sentiment")
        
        if temp_analysis.sentiment:
            return {
                "success": True,
                "sentiment": {
                    "polarity": temp_analysis.sentiment.polarity,
                    "subjectivity": temp_analysis.sentiment.subjectivity,
                    "sentiment_type": temp_analysis.sentiment.sentiment_type.value,
                    "confidence": temp_analysis.sentiment.confidence,
                    "emotional_tone": temp_analysis.sentiment.emotional_tone,
                    "intensity": temp_analysis.sentiment.intensity
                },
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "success": False,
                "message": "Sentiment analysis not available"
            }
        
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}")
        return {"error": str(e)}

@router.post("/sentiment/analyze/batch")
async def analyze_sentiments_batch(texts: List[str]):
    """Analizar sentimientos de múltiples textos"""
    try:
        from nlp_engine import AdvancedNLPEngine
        
        nlp_engine = AdvancedNLPEngine()
        results = []
        
        for i, text in enumerate(texts):
            temp_analysis = await nlp_engine.analyze_text(text, f"temp_sentiment_{i}")
            
            if temp_analysis.sentiment:
                results.append({
                    "index": i,
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "sentiment": {
                        "polarity": temp_analysis.sentiment.polarity,
                        "sentiment_type": temp_analysis.sentiment.sentiment_type.value,
                        "confidence": temp_analysis.sentiment.confidence,
                        "emotional_tone": temp_analysis.sentiment.emotional_tone
                    }
                })
            else:
                results.append({
                    "index": i,
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "sentiment": None
                })
        
        return {
            "success": True,
            "results": results,
            "total_texts": len(texts),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error analyzing sentiments: {e}")
        return {"error": str(e)}

# ============================================================================
# KEYWORD EXTRACTION ENDPOINTS
# ============================================================================

@router.post("/keywords/extract")
async def extract_keywords(
    text: str,
    max_keywords: int = Query(10, description="Número máximo de palabras clave")
):
    """Extraer palabras clave del texto"""
    try:
        from nlp_engine import AdvancedNLPEngine
        
        nlp_engine = AdvancedNLPEngine()
        
        # Crear análisis temporal
        temp_analysis = await nlp_engine.analyze_text(text, "temp_keywords")
        
        # Limitar palabras clave
        keywords = temp_analysis.keywords[:max_keywords]
        
        return {
            "success": True,
            "keywords": [
                {
                    "text": kw.text,
                    "score": kw.score,
                    "frequency": kw.frequency,
                    "tfidf_score": kw.tfidf_score
                }
                for kw in keywords
            ],
            "total_keywords": len(keywords),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error extracting keywords: {e}")
        return {"error": str(e)}

@router.post("/keywords/extract/batch")
async def extract_keywords_batch(
    texts: List[str],
    max_keywords: int = Query(10, description="Número máximo de palabras clave por texto")
):
    """Extraer palabras clave de múltiples textos"""
    try:
        from nlp_engine import AdvancedNLPEngine
        
        nlp_engine = AdvancedNLPEngine()
        results = []
        
        for i, text in enumerate(texts):
            temp_analysis = await nlp_engine.analyze_text(text, f"temp_keywords_{i}")
            
            keywords = temp_analysis.keywords[:max_keywords]
            
            results.append({
                "index": i,
                "text": text[:100] + "..." if len(text) > 100 else text,
                "keywords": [
                    {
                        "text": kw.text,
                        "score": kw.score,
                        "frequency": kw.frequency
                    }
                    for kw in keywords
                ],
                "total_keywords": len(keywords)
            })
        
        return {
            "success": True,
            "results": results,
            "total_texts": len(texts),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error extracting keywords: {e}")
        return {"error": str(e)}

# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

@router.get("/health")
async def nlp_health_check():
    """Health check para el sistema NLP"""
    try:
        # Verificar que los módulos se pueden importar
        from nlp_engine import AdvancedNLPEngine
        from text_processor import RealTimeTextProcessor
        from pattern_analyzer import TextPatternAnalyzer
        
        return {
            "status": "healthy",
            "modules": {
                "nlp_engine": "available",
                "text_processor": "available",
                "pattern_analyzer": "available"
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"NLP health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@router.get("/capabilities")
async def get_nlp_capabilities():
    """Obtener capacidades del sistema NLP"""
    return {
        "capabilities": {
            "text_analysis": [
                "tokenization",
                "pos_tagging",
                "named_entity_recognition",
                "sentiment_analysis",
                "keyword_extraction",
                "topic_modeling",
                "text_metrics"
            ],
            "pattern_analysis": [
                "linguistic_patterns",
                "structural_patterns",
                "semantic_patterns",
                "quality_patterns",
                "style_patterns",
                "pattern_clustering",
                "pattern_evolution"
            ],
            "text_processing": [
                "real_time_processing",
                "batch_processing",
                "priority_queues",
                "error_handling",
                "retry_mechanisms"
            ],
            "language_support": [
                "spanish",
                "english",
                "auto_detection"
            ]
        },
        "supported_formats": [
            "plain_text",
            "json",
            "batch_processing"
        ],
        "timestamp": datetime.now().isoformat()
    }



























