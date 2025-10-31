"""
Enhanced NLP Engine - Motor NLP mejorado con funcionalidades avanzadas
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import uuid

from .models import (
    TextAnalysisResult, SentimentResult, LanguageDetectionResult,
    TranslationResult, SummarizationResult, TextGenerationResult,
    EntityRecognitionResult, KeywordExtractionResult, TopicModelingResult,
    TextSimilarityResult, TextClassificationResult, NLPAnalysisRequest,
    NLPAnalysisResponse, Language, SentimentType, TextType
)
from .core import NLPEngine
from .advanced.transformer_models import TransformerModelManager
from .advanced.embeddings import EmbeddingManager
from .advanced.ai_integration import AIIntegrationManager

logger = logging.getLogger(__name__)


class EnhancedNLPEngine(NLPEngine):
    """
    Motor NLP mejorado con funcionalidades avanzadas.
    """
    
    def __init__(self):
        """Inicializar el motor NLP mejorado."""
        super().__init__()
        
        # Componentes avanzados
        self.transformer_manager = TransformerModelManager()
        self.embedding_manager = EmbeddingManager()
        self.ai_integration = AIIntegrationManager()
        
        # Configuración avanzada
        self.use_advanced_models = True
        self.use_ai_integration = True
        self.use_embeddings = True
        
        # Métricas avanzadas
        self.advanced_metrics = {
            "transformer_requests": 0,
            "embedding_requests": 0,
            "ai_integration_requests": 0,
            "advanced_processing_time": 0.0
        }
        
        logger.info("Enhanced NLP Engine inicializado")
    
    async def initialize(self):
        """Inicializar el motor NLP mejorado."""
        if not self._initialized:
            try:
                # Inicializar motor base
                await super().initialize()
                
                # Inicializar componentes avanzados
                if self.use_advanced_models:
                    await self.transformer_manager.initialize()
                
                if self.use_embeddings:
                    await self.embedding_manager.initialize()
                
                if self.use_ai_integration:
                    await self.ai_integration.initialize()
                
                self._initialized = True
                logger.info("Enhanced NLP Engine completamente inicializado")
                
            except Exception as e:
                logger.error(f"Error al inicializar Enhanced NLP Engine: {e}")
                raise
    
    async def shutdown(self):
        """Cerrar el motor NLP mejorado."""
        if self._initialized:
            try:
                # Cerrar componentes avanzados
                if self.use_ai_integration:
                    await self.ai_integration.shutdown()
                
                if self.use_embeddings:
                    await self.embedding_manager.shutdown()
                
                if self.use_advanced_models:
                    await self.transformer_manager.shutdown()
                
                # Cerrar motor base
                await super().shutdown()
                
                self._initialized = False
                logger.info("Enhanced NLP Engine cerrado")
                
            except Exception as e:
                logger.error(f"Error al cerrar Enhanced NLP Engine: {e}")
    
    async def analyze_text_enhanced(self, request: NLPAnalysisRequest) -> NLPAnalysisResponse:
        """
        Análisis de texto mejorado con funcionalidades avanzadas.
        
        Args:
            request: Solicitud de análisis
            
        Returns:
            Respuesta con resultados del análisis mejorado
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            # Verificar cache
            cache_key = self._generate_cache_key(request)
            if cache_key in self.cache:
                cached_result = self.cache[cache_key]
                if datetime.now() - cached_result['timestamp'] < timedelta(seconds=self.cache_ttl):
                    self.metrics["cache_hits"] += 1
                    logger.info(f"Resultado obtenido del cache: {request_id}")
                    return cached_result['result']
            
            self.metrics["cache_misses"] += 1
            
            # Procesar análisis con componentes avanzados
            results = {}
            
            # Análisis básico mejorado
            if "text_analysis" in request.analysis_types:
                results["text_analysis"] = await self._analyze_text_enhanced(request.text)
            
            # Análisis de sentimiento avanzado
            if "sentiment" in request.analysis_types:
                if self.use_advanced_models:
                    results["sentiment"] = await self.transformer_manager.analyze_sentiment_advanced(request.text)
                else:
                    results["sentiment"] = await self.sentiment_analyzer.analyze(request.text)
            
            # Detección de idioma
            if "language" in request.analysis_types:
                results["language"] = await self.language_detector.detect(request.text)
            
            # Reconocimiento de entidades avanzado
            if "entities" in request.analysis_types:
                if self.use_advanced_models:
                    results["entities"] = await self.transformer_manager.extract_entities_advanced(request.text)
                else:
                    results["entities"] = await self._extract_entities(request.text)
            
            # Extracción de palabras clave
            if "keywords" in request.analysis_types:
                results["keywords"] = await self._extract_keywords(request.text)
            
            # Modelado de temas
            if "topics" in request.analysis_types:
                results["topics"] = await self._extract_topics(request.text)
            
            # Clasificación de texto avanzada
            if "classification" in request.analysis_types:
                if self.use_advanced_models:
                    results["classification"] = await self.transformer_manager.classify_text_advanced(request.text)
                else:
                    results["classification"] = await self._classify_text(request.text)
            
            # Análisis de similitud semántica
            if "similarity" in request.analysis_types and self.use_embeddings:
                results["similarity"] = await self._analyze_similarity(request.text, request.parameters)
            
            # Análisis con IA externa
            if "ai_analysis" in request.analysis_types and self.use_ai_integration:
                results["ai_analysis"] = await self._ai_analysis(request.text, request.parameters)
            
            # Crear respuesta
            processing_time = time.time() - start_time
            response = NLPAnalysisResponse(
                request_id=request_id,
                results=results,
                processing_time=processing_time,
                success=True
            )
            
            # Guardar en cache
            self.cache[cache_key] = {
                'result': response,
                'timestamp': datetime.now()
            }
            
            # Actualizar métricas
            self._update_metrics(processing_time, True)
            
            logger.info(f"Análisis NLP mejorado completado: {request_id} (tiempo: {processing_time:.2f}s)")
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_metrics(processing_time, False)
            
            logger.error(f"Error en análisis NLP mejorado: {e}")
            
            return NLPAnalysisResponse(
                request_id=request_id,
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
    
    async def _analyze_text_enhanced(self, text: str) -> TextAnalysisResult:
        """Análisis básico de texto mejorado."""
        # Usar análisis básico del motor padre
        basic_analysis = await self._analyze_text_basic(text)
        
        # Agregar análisis avanzado si está disponible
        if self.use_embeddings:
            try:
                embedding_info = await self.embedding_manager.get_text_representation(text)
                basic_analysis.embedding_stats = embedding_info.get("embedding_stats", {})
            except Exception as e:
                logger.warning(f"Error al obtener embedding: {e}")
        
        return basic_analysis
    
    async def _analyze_similarity(self, text: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Análisis de similitud semántica."""
        try:
            # Obtener texto de comparación
            compare_text = parameters.get("compare_text", "")
            if not compare_text:
                return {"error": "No se proporcionó texto para comparar"}
            
            # Calcular similitud
            similarity = await self.embedding_manager.calculate_similarity(text, compare_text)
            
            return {
                "text1": text,
                "text2": compare_text,
                "similarity_score": similarity,
                "similarity_type": "cosine_similarity",
                "model_used": self.embedding_manager.model_name,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error en análisis de similitud: {e}")
            return {"error": str(e)}
    
    async def _ai_analysis(self, text: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Análisis con IA externa."""
        try:
            analysis_type = parameters.get("ai_analysis_type", "sentiment")
            provider = parameters.get("provider", "openai")
            
            if analysis_type == "sentiment":
                return await self.ai_integration.analyze_sentiment_ai(text, provider)
            elif analysis_type == "summarization":
                max_length = parameters.get("max_length", 150)
                return await self.ai_integration.summarize_text_ai(text, max_length, provider)
            elif analysis_type == "translation":
                target_language = parameters.get("target_language", "es")
                return await self.ai_integration.translate_text_ai(text, target_language, provider)
            else:
                return {"error": f"Tipo de análisis IA no soportado: {analysis_type}"}
                
        except Exception as e:
            logger.error(f"Error en análisis con IA: {e}")
            return {"error": str(e)}
    
    async def summarize_enhanced(self, text: str, max_length: int = 150, use_ai: bool = False, provider: str = "openai") -> Dict[str, Any]:
        """Resumen mejorado con opciones avanzadas."""
        try:
            if use_ai and self.use_ai_integration:
                # Usar IA externa para resumen
                return await self.ai_integration.summarize_text_ai(text, max_length, provider)
            elif self.use_advanced_models:
                # Usar modelo transformer
                return await self.transformer_manager.summarize_advanced(text, max_length)
            else:
                # Usar resumidor básico
                result = await self.summarizer.summarize(text, max_length // 50)  # Aproximar oraciones
                return {
                    "original_text": result.original_text,
                    "summary": result.summary,
                    "compression_ratio": result.compression_ratio,
                    "key_points": result.key_points,
                    "word_count_original": result.word_count_original,
                    "word_count_summary": result.word_count_summary,
                    "method": "basic",
                    "timestamp": result.timestamp.isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error en resumen mejorado: {e}")
            raise
    
    async def generate_text_enhanced(self, prompt: str, template: str = "summary", use_ai: bool = False, provider: str = "openai") -> Dict[str, Any]:
        """Generación de texto mejorada."""
        try:
            if use_ai and self.use_ai_integration:
                # Usar IA externa para generación
                messages = [
                    {"role": "system", "content": f"Eres un asistente experto en {template}."},
                    {"role": "user", "content": prompt}
                ]
                result = await self.ai_integration.chat_completion(messages, provider=provider)
                return {
                    "prompt": prompt,
                    "generated_text": result["response"],
                    "method": "ai_external",
                    "provider": provider,
                    "model": result.get("model", "unknown"),
                    "timestamp": datetime.now().isoformat()
                }
            elif self.use_advanced_models:
                # Usar modelo transformer
                return await self.transformer_manager.generate_text_advanced(prompt)
            else:
                # Usar generador básico
                result = await self.text_generator.generate(prompt, template)
                return {
                    "prompt": result.prompt,
                    "generated_text": result.generated_text,
                    "method": "basic",
                    "model_used": result.model_used,
                    "timestamp": result.timestamp.isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error en generación de texto mejorada: {e}")
            raise
    
    async def find_similar_texts(self, query: str, texts: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """Encontrar textos similares usando embeddings."""
        if not self.use_embeddings:
            raise ValueError("Embeddings no están habilitados")
        
        try:
            return await self.embedding_manager.find_most_similar(query, texts, top_k)
        except Exception as e:
            logger.error(f"Error al encontrar textos similares: {e}")
            raise
    
    async def cluster_texts(self, texts: List[str], n_clusters: int = 3) -> Dict[str, Any]:
        """Agrupar textos por similitud semántica."""
        if not self.use_embeddings:
            raise ValueError("Embeddings no están habilitados")
        
        try:
            return await self.embedding_manager.cluster_texts(texts, n_clusters)
        except Exception as e:
            logger.error(f"Error en clustering de textos: {e}")
            raise
    
    async def get_enhanced_metrics(self) -> Dict[str, Any]:
        """Obtener métricas mejoradas del motor NLP."""
        base_metrics = await self.get_metrics()
        
        # Agregar métricas avanzadas
        enhanced_metrics = {
            **base_metrics,
            "advanced_metrics": self.advanced_metrics,
            "components_status": {
                "transformer_manager": await self.transformer_manager.health_check() if self.use_advanced_models else {"status": "disabled"},
                "embedding_manager": await self.embedding_manager.health_check() if self.use_embeddings else {"status": "disabled"},
                "ai_integration": await self.ai_integration.health_check() if self.use_ai_integration else {"status": "disabled"}
            },
            "configuration": {
                "use_advanced_models": self.use_advanced_models,
                "use_ai_integration": self.use_ai_integration,
                "use_embeddings": self.use_embeddings
            }
        }
        
        return enhanced_metrics
    
    async def optimize_performance(self) -> Dict[str, Any]:
        """Optimizar rendimiento del motor NLP mejorado."""
        optimizations = []
        
        # Optimizar motor base
        base_optimizations = await super().optimize_performance()
        optimizations.extend(base_optimizations.get("optimizations_applied", []))
        
        # Optimizar embedding manager
        if self.use_embeddings:
            try:
                await self.embedding_manager.clear_cache()
                optimizations.append("Cache de embeddings limpiado")
            except Exception as e:
                logger.warning(f"Error al limpiar cache de embeddings: {e}")
        
        # Optimizar transformer manager
        if self.use_advanced_models:
            try:
                # Limpiar modelos no utilizados
                model_info = await self.transformer_manager.get_model_info()
                optimizations.append(f"Modelos cargados: {len(model_info['loaded_models'])}")
            except Exception as e:
                logger.warning(f"Error al obtener info de modelos: {e}")
        
        return {
            "optimizations_applied": optimizations,
            "timestamp": datetime.now().isoformat(),
            "enhanced_metrics": await self.get_enhanced_metrics()
        }
    
    async def health_check_enhanced(self) -> Dict[str, Any]:
        """Verificar salud del motor NLP mejorado."""
        base_health = await self.health_check()
        
        # Verificar componentes avanzados
        advanced_health = {
            "transformer_manager": await self.transformer_manager.health_check() if self.use_advanced_models else {"status": "disabled"},
            "embedding_manager": await self.embedding_manager.health_check() if self.use_embeddings else {"status": "disabled"},
            "ai_integration": await self.ai_integration.health_check() if self.use_ai_integration else {"status": "disabled"}
        }
        
        # Determinar estado general
        all_healthy = (
            base_health["status"] == "healthy" and
            all(comp["status"] in ["healthy", "disabled"] for comp in advanced_health.values())
        )
        
        return {
            "status": "healthy" if all_healthy else "unhealthy",
            "base_engine": base_health,
            "advanced_components": advanced_health,
            "configuration": {
                "use_advanced_models": self.use_advanced_models,
                "use_ai_integration": self.use_ai_integration,
                "use_embeddings": self.use_embeddings
            },
            "timestamp": datetime.now().isoformat()
        }


# Instancia global del motor NLP mejorado
_enhanced_nlp_engine: Optional[EnhancedNLPEngine] = None


def get_enhanced_nlp_engine() -> EnhancedNLPEngine:
    """Obtener la instancia global del motor NLP mejorado."""
    global _enhanced_nlp_engine
    if _enhanced_nlp_engine is None:
        _enhanced_nlp_engine = EnhancedNLPEngine()
    return _enhanced_nlp_engine




