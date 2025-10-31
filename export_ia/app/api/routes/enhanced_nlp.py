"""
Enhanced NLP API Routes - Rutas de la API NLP mejorada
"""

import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ...nlp.enhanced_engine import get_enhanced_nlp_engine
from ...nlp.models import NLPAnalysisRequest, Language

logger = logging.getLogger(__name__)
router = APIRouter()


class EnhancedTextAnalysisRequest(BaseModel):
    """Solicitud de análisis de texto mejorado."""
    text: str = Field(..., description="Texto a analizar", min_length=1, max_length=50000)
    analysis_types: List[str] = Field(
        default=["sentiment", "language", "entities", "classification"],
        description="Tipos de análisis a realizar"
    )
    use_advanced_models: bool = Field(default=True, description="Usar modelos avanzados")
    use_ai_integration: bool = Field(default=False, description="Usar integración con IA externa")
    use_embeddings: bool = Field(default=True, description="Usar embeddings")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parámetros adicionales")


class EnhancedSummarizationRequest(BaseModel):
    """Solicitud de resumen mejorado."""
    text: str = Field(..., description="Texto a resumir", min_length=1, max_length=50000)
    max_length: int = Field(default=150, description="Longitud máxima del resumen", ge=50, le=500)
    use_ai: bool = Field(default=False, description="Usar IA externa para resumen")
    provider: str = Field(default="openai", description="Proveedor de IA")


class EnhancedTextGenerationRequest(BaseModel):
    """Solicitud de generación de texto mejorada."""
    prompt: str = Field(..., description="Prompt para generar texto", min_length=1, max_length=2000)
    template: str = Field(default="summary", description="Template a usar")
    use_ai: bool = Field(default=False, description="Usar IA externa para generación")
    provider: str = Field(default="openai", description="Proveedor de IA")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parámetros adicionales")


class SimilarityAnalysisRequest(BaseModel):
    """Solicitud de análisis de similitud."""
    query_text: str = Field(..., description="Texto de consulta", min_length=1, max_length=10000)
    candidate_texts: List[str] = Field(..., description="Textos candidatos", min_items=1, max_items=100)
    top_k: int = Field(default=5, description="Número de resultados", ge=1, le=20)


class ClusteringRequest(BaseModel):
    """Solicitud de clustering de textos."""
    texts: List[str] = Field(..., description="Textos a agrupar", min_items=2, max_items=1000)
    n_clusters: int = Field(default=3, description="Número de clusters", ge=2, le=20)


class AIAnalysisRequest(BaseModel):
    """Solicitud de análisis con IA externa."""
    text: str = Field(..., description="Texto a analizar", min_length=1, max_length=10000)
    analysis_type: str = Field(..., description="Tipo de análisis", regex="^(sentiment|summarization|translation)$")
    provider: str = Field(default="openai", description="Proveedor de IA")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parámetros adicionales")


@router.post("/nlp/enhanced/analyze")
async def analyze_text_enhanced(request: EnhancedTextAnalysisRequest):
    """Análisis de texto mejorado con funcionalidades avanzadas."""
    try:
        nlp_engine = get_enhanced_nlp_engine()
        
        # Configurar motor según parámetros
        nlp_engine.use_advanced_models = request.use_advanced_models
        nlp_engine.use_ai_integration = request.use_ai_integration
        nlp_engine.use_embeddings = request.use_embeddings
        
        # Crear solicitud NLP
        nlp_request = NLPAnalysisRequest(
            text=request.text,
            analysis_types=request.analysis_types,
            parameters=request.parameters
        )
        
        # Procesar análisis
        result = await nlp_engine.analyze_text_enhanced(nlp_request)
        
        return {
            "request_id": result.request_id,
            "results": result.results,
            "processing_time": result.processing_time,
            "success": result.success,
            "error_message": result.error_message,
            "configuration": {
                "use_advanced_models": request.use_advanced_models,
                "use_ai_integration": request.use_ai_integration,
                "use_embeddings": request.use_embeddings
            },
            "timestamp": result.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error en análisis de texto mejorado: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/nlp/enhanced/summarize")
async def summarize_text_enhanced(request: EnhancedSummarizationRequest):
    """Resumen de texto mejorado."""
    try:
        nlp_engine = get_enhanced_nlp_engine()
        
        # Procesar resumen
        result = await nlp_engine.summarize_enhanced(
            request.text,
            request.max_length,
            request.use_ai,
            request.provider
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error en resumen mejorado: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/nlp/enhanced/generate")
async def generate_text_enhanced(request: EnhancedTextGenerationRequest):
    """Generación de texto mejorada."""
    try:
        nlp_engine = get_enhanced_nlp_engine()
        
        # Procesar generación
        result = await nlp_engine.generate_text_enhanced(
            request.prompt,
            request.template,
            request.use_ai,
            request.provider
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error en generación de texto mejorada: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/nlp/enhanced/similarity")
async def find_similar_texts(request: SimilarityAnalysisRequest):
    """Encontrar textos similares usando embeddings."""
    try:
        nlp_engine = get_enhanced_nlp_engine()
        
        # Encontrar textos similares
        result = await nlp_engine.find_similar_texts(
            request.query_text,
            request.candidate_texts,
            request.top_k
        )
        
        return {
            "query_text": request.query_text,
            "similar_texts": result,
            "total_candidates": len(request.candidate_texts),
            "top_k": request.top_k,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error en análisis de similitud: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/nlp/enhanced/cluster")
async def cluster_texts(request: ClusteringRequest):
    """Agrupar textos por similitud semántica."""
    try:
        nlp_engine = get_enhanced_nlp_engine()
        
        # Agrupar textos
        result = await nlp_engine.cluster_texts(
            request.texts,
            request.n_clusters
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error en clustering de textos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/nlp/enhanced/ai-analysis")
async def ai_analysis(request: AIAnalysisRequest):
    """Análisis con IA externa."""
    try:
        nlp_engine = get_enhanced_nlp_engine()
        
        # Realizar análisis con IA
        if request.analysis_type == "sentiment":
            result = await nlp_engine.ai_integration.analyze_sentiment_ai(
                request.text, request.provider
            )
        elif request.analysis_type == "summarization":
            max_length = request.parameters.get("max_length", 150)
            result = await nlp_engine.ai_integration.summarize_text_ai(
                request.text, max_length, request.provider
            )
        elif request.analysis_type == "translation":
            target_language = request.parameters.get("target_language", "es")
            result = await nlp_engine.ai_integration.translate_text_ai(
                request.text, target_language, request.provider
            )
        else:
            raise ValueError(f"Tipo de análisis no soportado: {request.analysis_type}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error en análisis con IA: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/nlp/enhanced/health")
async def enhanced_nlp_health_check():
    """Verificar salud del sistema NLP mejorado."""
    try:
        nlp_engine = get_enhanced_nlp_engine()
        health_status = await nlp_engine.health_check_enhanced()
        
        return health_status
        
    except Exception as e:
        logger.error(f"Error en health check NLP mejorado: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/nlp/enhanced/metrics")
async def get_enhanced_nlp_metrics():
    """Obtener métricas del sistema NLP mejorado."""
    try:
        nlp_engine = get_enhanced_nlp_engine()
        metrics = await nlp_engine.get_enhanced_metrics()
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error al obtener métricas NLP mejoradas: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/nlp/enhanced/optimize")
async def optimize_enhanced_nlp():
    """Optimizar sistema NLP mejorado."""
    try:
        nlp_engine = get_enhanced_nlp_engine()
        optimization_result = await nlp_engine.optimize_performance()
        
        return optimization_result
        
    except Exception as e:
        logger.error(f"Error al optimizar NLP mejorado: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/nlp/enhanced/providers")
async def get_available_providers():
    """Obtener proveedores de IA disponibles."""
    try:
        nlp_engine = get_enhanced_nlp_engine()
        providers = await nlp_engine.ai_integration.get_available_providers()
        
        return providers
        
    except Exception as e:
        logger.error(f"Error al obtener proveedores: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/nlp/enhanced/models")
async def get_loaded_models():
    """Obtener modelos cargados."""
    try:
        nlp_engine = get_enhanced_nlp_engine()
        model_info = await nlp_engine.transformer_manager.get_model_info()
        
        return model_info
        
    except Exception as e:
        logger.error(f"Error al obtener modelos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/nlp/enhanced/models/{model_type}/load")
async def load_model(model_type: str):
    """Cargar un modelo específico."""
    try:
        nlp_engine = get_enhanced_nlp_engine()
        success = await nlp_engine.transformer_manager.load_model(model_type)
        
        if success:
            return {"message": f"Modelo {model_type} cargado exitosamente"}
        else:
            raise HTTPException(status_code=400, detail=f"No se pudo cargar el modelo {model_type}")
        
    except Exception as e:
        logger.error(f"Error al cargar modelo {model_type}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/nlp/enhanced/models/{model_type}")
async def unload_model(model_type: str):
    """Descargar un modelo específico."""
    try:
        nlp_engine = get_enhanced_nlp_engine()
        success = await nlp_engine.transformer_manager.unload_model(model_type)
        
        if success:
            return {"message": f"Modelo {model_type} descargado exitosamente"}
        else:
            raise HTTPException(status_code=400, detail=f"No se pudo descargar el modelo {model_type}")
        
    except Exception as e:
        logger.error(f"Error al descargar modelo {model_type}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/nlp/enhanced/embeddings/stats")
async def get_embedding_stats():
    """Obtener estadísticas de embeddings."""
    try:
        nlp_engine = get_enhanced_nlp_engine()
        stats = await nlp_engine.embedding_manager.get_cache_stats()
        
        return stats
        
    except Exception as e:
        logger.error(f"Error al obtener estadísticas de embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/nlp/enhanced/embeddings/clear-cache")
async def clear_embedding_cache():
    """Limpiar cache de embeddings."""
    try:
        nlp_engine = get_enhanced_nlp_engine()
        await nlp_engine.embedding_manager.clear_cache()
        
        return {"message": "Cache de embeddings limpiado exitosamente"}
        
    except Exception as e:
        logger.error(f"Error al limpiar cache de embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))




