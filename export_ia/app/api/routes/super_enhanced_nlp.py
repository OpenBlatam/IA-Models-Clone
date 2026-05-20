"""
Super Enhanced NLP API Routes - Rutas API para el motor NLP super mejorado
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field

from ...nlp.super_enhanced_engine import get_super_enhanced_nlp_engine, SuperEnhancedNLPEngine
from ...nlp.models import NLPAnalysisRequest, Language, SentimentType

logger = logging.getLogger(__name__)

# Crear router
router = APIRouter(prefix="/nlp/super-enhanced", tags=["Super Enhanced NLP"])


# Modelos Pydantic para requests
class SuperEnhancedAnalysisRequest(BaseModel):
    """Solicitud de análisis super mejorado."""
    text: str = Field(..., description="Texto a analizar")
    analysis_types: List[str] = Field(default=["sentiment", "language", "entities"], description="Tipos de análisis")
    language: Optional[str] = Field(None, description="Idioma del texto")
    use_advanced_models: bool = Field(True, description="Usar modelos avanzados")
    use_ai_integration: bool = Field(True, description="Usar integración con IA")
    use_embeddings: bool = Field(True, description="Usar embeddings")
    use_analytics: bool = Field(True, description="Usar analíticas avanzadas")
    reference_texts: Optional[List[str]] = Field(None, description="Textos de referencia para similitud")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Parámetros adicionales")


class ConversationStartRequest(BaseModel):
    """Solicitud para iniciar conversación."""
    user_id: Optional[str] = Field(None, description="ID del usuario")
    conversation_type: str = Field("casual", description="Tipo de conversación")
    language: str = Field("english", description="Idioma de la conversación")
    initial_context: Optional[Dict[str, Any]] = Field(None, description="Contexto inicial")


class MessageSendRequest(BaseModel):
    """Solicitud para enviar mensaje."""
    message: str = Field(..., description="Mensaje del usuario")
    user_id: Optional[str] = Field(None, description="ID del usuario")


class DocumentAnalysisRequest(BaseModel):
    """Solicitud de análisis de documento."""
    content: str = Field(..., description="Contenido del documento")
    title: str = Field("Untitled Document", description="Título del documento")
    document_type: Optional[str] = Field(None, description="Tipo de documento")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadatos adicionales")


class ContentOptimizationRequest(BaseModel):
    """Solicitud de optimización de contenido."""
    content: str = Field(..., description="Contenido a optimizar")
    optimization_goal: str = Field(..., description="Objetivo de optimización")
    content_type: str = Field(..., description="Tipo de contenido")
    target_keywords: Optional[List[str]] = Field(None, description="Palabras clave objetivo")
    custom_rules: Optional[List[Dict[str, Any]]] = Field(None, description="Reglas personalizadas")


class EmbeddingRequest(BaseModel):
    """Solicitud de embeddings."""
    texts: Union[str, List[str]] = Field(..., description="Texto(s) para generar embeddings")
    use_cache: bool = Field(True, description="Usar cache si está disponible")


class SimilarityRequest(BaseModel):
    """Solicitud de similitud."""
    text1: str = Field(..., description="Primer texto")
    text2: str = Field(..., description="Segundo texto")


class SimilarTextsRequest(BaseModel):
    """Solicitud de textos similares."""
    query_text: str = Field(..., description="Texto de consulta")
    candidate_texts: List[str] = Field(..., description="Textos candidatos")
    top_k: int = Field(5, description="Número de resultados a retornar")


class ClusteringRequest(BaseModel):
    """Solicitud de clustering."""
    texts: List[str] = Field(..., description="Textos a agrupar")
    n_clusters: int = Field(3, description="Número de clusters")


class ComponentConfigurationRequest(BaseModel):
    """Solicitud de configuración de componentes."""
    use_advanced_models: Optional[bool] = Field(None, description="Usar modelos avanzados")
    use_ai_integration: Optional[bool] = Field(None, description="Usar integración con IA")
    use_embeddings: Optional[bool] = Field(None, description="Usar embeddings")
    use_analytics: Optional[bool] = Field(None, description="Usar analíticas avanzadas")
    use_conversation_ai: Optional[bool] = Field(None, description="Usar IA conversacional")
    use_document_analysis: Optional[bool] = Field(None, description="Usar análisis de documentos")
    use_content_optimization: Optional[bool] = Field(None, description="Usar optimización de contenido")


# Endpoints principales
@router.post("/analyze")
async def analyze_text_super_enhanced(request: SuperEnhancedAnalysisRequest):
    """
    Análisis de texto super mejorado con todas las funcionalidades.
    """
    try:
        nlp_engine = await get_super_enhanced_nlp_engine()
        
        # Configurar componentes según la solicitud
        await nlp_engine.configure_components(
            use_advanced_models=request.use_advanced_models,
            use_ai_integration=request.use_ai_integration,
            use_embeddings=request.use_embeddings,
            use_analytics=request.use_analytics
        )
        
        # Crear solicitud de análisis
        nlp_request = NLPAnalysisRequest(
            text=request.text,
            analysis_types=request.analysis_types,
            language=Language(request.language) if request.language else None,
            parameters=request.parameters
        )
        
        # Agregar textos de referencia si se proporcionan
        if request.reference_texts:
            nlp_request.reference_texts = request.reference_texts
        
        # Realizar análisis super mejorado
        result = await nlp_engine.analyze_text_super_enhanced(nlp_request)
        
        return {
            "request_id": result.request_id,
            "results": result.results,
            "processing_time": result.processing_time,
            "success": result.success,
            "error_message": result.error_message,
            "configuration": {
                "use_advanced_models": request.use_advanced_models,
                "use_ai_integration": request.use_ai_integration,
                "use_embeddings": request.use_embeddings,
                "use_analytics": request.use_analytics
            },
            "timestamp": result.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error en análisis super mejorado: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/conversation/start")
async def start_conversation(request: ConversationStartRequest):
    """
    Iniciar una conversación con IA.
    """
    try:
        nlp_engine = await get_super_enhanced_nlp_engine()
        
        result = await nlp_engine.start_conversation(
            user_id=request.user_id,
            conversation_type=request.conversation_type,
            language=request.language,
            initial_context=request.initial_context
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error al iniciar conversación: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/conversation/{conversation_id}/message")
async def send_message(conversation_id: str, request: MessageSendRequest):
    """
    Enviar mensaje a una conversación.
    """
    try:
        nlp_engine = await get_super_enhanced_nlp_engine()
        
        result = await nlp_engine.send_message(
            conversation_id=conversation_id,
            message=request.message,
            user_id=request.user_id
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error al enviar mensaje: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversation/{conversation_id}/history")
async def get_conversation_history(conversation_id: str):
    """
    Obtener historial de conversación.
    """
    try:
        nlp_engine = await get_super_enhanced_nlp_engine()
        
        history = await nlp_engine.conversation_ai.get_conversation_history(conversation_id)
        
        return {
            "conversation_id": conversation_id,
            "history": history,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al obtener historial de conversación: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversation/{conversation_id}/analytics")
async def get_conversation_analytics(conversation_id: str):
    """
    Obtener analíticas de conversación.
    """
    try:
        nlp_engine = await get_super_enhanced_nlp_engine()
        
        analytics = await nlp_engine.conversation_ai.get_conversation_analytics(conversation_id)
        
        return analytics
        
    except Exception as e:
        logger.error(f"Error al obtener analíticas de conversación: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/document/analyze")
async def analyze_document(request: DocumentAnalysisRequest):
    """
    Analizar un documento completo.
    """
    try:
        nlp_engine = await get_super_enhanced_nlp_engine()
        
        result = await nlp_engine.analyze_document(
            content=request.content,
            title=request.title,
            document_type=request.document_type,
            metadata=request.metadata
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error al analizar documento: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/content/optimize")
async def optimize_content(request: ContentOptimizationRequest):
    """
    Optimizar contenido según objetivos específicos.
    """
    try:
        nlp_engine = await get_super_enhanced_nlp_engine()
        
        result = await nlp_engine.optimize_content(
            content=request.content,
            optimization_goal=request.optimization_goal,
            content_type=request.content_type,
            target_keywords=request.target_keywords,
            custom_rules=request.custom_rules
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error al optimizar contenido: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embeddings")
async def get_embeddings(request: EmbeddingRequest):
    """
    Obtener embeddings de texto(s).
    """
    try:
        nlp_engine = await get_super_enhanced_nlp_engine()
        
        embeddings = await nlp_engine.get_embeddings(
            texts=request.texts,
            use_cache=request.use_cache
        )
        
        return {
            "embeddings": embeddings,
            "text_count": len(request.texts) if isinstance(request.texts, list) else 1,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al obtener embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/similarity")
async def calculate_similarity(request: SimilarityRequest):
    """
    Calcular similitud entre dos textos.
    """
    try:
        nlp_engine = await get_super_enhanced_nlp_engine()
        
        similarity = await nlp_engine.calculate_similarity(
            text1=request.text1,
            text2=request.text2
        )
        
        return {
            "text1": request.text1,
            "text2": request.text2,
            "similarity": similarity,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al calcular similitud: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/similar-texts")
async def find_similar_texts(request: SimilarTextsRequest):
    """
    Encontrar textos similares a una consulta.
    """
    try:
        nlp_engine = await get_super_enhanced_nlp_engine()
        
        similar_texts = await nlp_engine.find_similar_texts(
            query_text=request.query_text,
            candidate_texts=request.candidate_texts,
            top_k=request.top_k
        )
        
        return {
            "query_text": request.query_text,
            "similar_texts": similar_texts,
            "total_candidates": len(request.candidate_texts),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al encontrar textos similares: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cluster")
async def cluster_texts(request: ClusteringRequest):
    """
    Agrupar textos en clusters.
    """
    try:
        nlp_engine = await get_super_enhanced_nlp_engine()
        
        clusters = await nlp_engine.cluster_texts(
            texts=request.texts,
            n_clusters=request.n_clusters
        )
        
        return {
            "clusters": clusters,
            "total_texts": len(request.texts),
            "n_clusters": request.n_clusters,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al agrupar textos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/configure")
async def configure_components(request: ComponentConfigurationRequest):
    """
    Configurar componentes del motor super mejorado.
    """
    try:
        nlp_engine = await get_super_enhanced_nlp_engine()
        
        configuration = await nlp_engine.configure_components(
            use_advanced_models=request.use_advanced_models,
            use_ai_integration=request.use_ai_integration,
            use_embeddings=request.use_embeddings,
            use_analytics=request.use_analytics,
            use_conversation_ai=request.use_conversation_ai,
            use_document_analysis=request.use_document_analysis,
            use_content_optimization=request.use_content_optimization
        )
        
        return configuration
        
    except Exception as e:
        logger.error(f"Error al configurar componentes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check_super_enhanced():
    """
    Verificar salud del motor super mejorado.
    """
    try:
        nlp_engine = await get_super_enhanced_nlp_engine()
        
        health = await nlp_engine.health_check_super_enhanced()
        
        return health
        
    except Exception as e:
        logger.error(f"Error en health check super mejorado: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_super_metrics():
    """
    Obtener métricas super mejoradas.
    """
    try:
        nlp_engine = await get_super_enhanced_nlp_engine()
        
        metrics = await nlp_engine.get_super_metrics()
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error al obtener métricas super mejoradas: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/capabilities")
async def get_capabilities():
    """
    Obtener capacidades del motor super mejorado.
    """
    try:
        return {
            "basic_nlp": {
                "sentiment_analysis": True,
                "language_detection": True,
                "entity_extraction": True,
                "text_generation": True,
                "summarization": True,
                "translation": True
            },
            "advanced_features": {
                "transformer_models": True,
                "ai_integration": True,
                "embeddings": True,
                "advanced_analytics": True,
                "conversation_ai": True,
                "document_analysis": True,
                "content_optimization": True
            },
            "ai_providers": {
                "openai": True,
                "anthropic": True,
                "cohere": True,
                "hugging_face": True
            },
            "optimization_goals": [
                "seo",
                "readability", 
                "engagement",
                "conversion",
                "accessibility",
                "clarity",
                "persuasion",
                "technical"
            ],
            "content_types": [
                "blog_post",
                "article",
                "product_description",
                "email",
                "social_media",
                "landing_page",
                "news_letter",
                "technical_document",
                "marketing_copy",
                "educational_content"
            ],
            "conversation_types": [
                "customer_service",
                "sales",
                "technical_support",
                "educational",
                "casual",
                "professional"
            ],
            "document_types": [
                "text",
                "pdf",
                "word",
                "html",
                "markdown",
                "email",
                "report",
                "article",
                "blog_post",
                "news",
                "academic_paper",
                "legal_document",
                "technical_document"
            ],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al obtener capacidades: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def get_available_models():
    """
    Obtener modelos disponibles.
    """
    try:
        nlp_engine = await get_super_enhanced_nlp_engine()
        
        models = await nlp_engine.transformer_manager.get_model_info()
        
        return {
            "transformer_models": models,
            "ai_providers": {
                "openai": ["gpt-3.5-turbo", "gpt-4", "text-embedding-ada-002"],
                "anthropic": ["claude-3-sonnet", "claude-3-opus"],
                "cohere": ["command", "embed-english-v2.0"],
                "hugging_face": ["roberta-base", "bert-base-uncased", "bart-large-cnn"]
            },
            "embedding_models": ["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al obtener modelos disponibles: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/optimization-rules")
async def get_optimization_rules():
    """
    Obtener reglas de optimización disponibles.
    """
    try:
        nlp_engine = await get_super_enhanced_nlp_engine()
        
        rules = await nlp_engine.content_optimizer.get_optimization_rules()
        
        return {
            "optimization_rules": rules,
            "total_rules": len(rules),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al obtener reglas de optimización: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/conversation/{conversation_id}")
async def end_conversation(conversation_id: str):
    """
    Finalizar una conversación.
    """
    try:
        nlp_engine = await get_super_enhanced_nlp_engine()
        
        success = await nlp_engine.conversation_ai.end_conversation(conversation_id)
        
        return {
            "conversation_id": conversation_id,
            "ended": success,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al finalizar conversación: {e}")
        raise HTTPException(status_code=500, detail=str(e))




