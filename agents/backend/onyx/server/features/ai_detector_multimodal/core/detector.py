"""
Core detector de IA multimodal
Implementa la lógica principal de detección de contenido generado por IA
"""

import logging
import time
import hashlib
import re
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class MultimodalAIDetector:
    """Detector multimodal de contenido generado por IA"""
    
    # Patrones comunes de modelos de IA - MEJORADO
    AI_MODEL_PATTERNS = {
        "gpt-3.5": {
            "patterns": [
                r"\b(?:as an ai|i'm an ai|i am an ai|as a language model|as an artificial intelligence)",
                r"\b(?:i cannot|i don't have|i'm not able|i don't have access|i'm unable)",
                r"\b(?:i apologize|i'm sorry|unfortunately|i cannot provide)",
                r"\b(?:please note that|it's important to|keep in mind)",
                r"\b(?:i hope this helps|let me know if|feel free to ask)"
            ],
            "provider": "OpenAI",
            "confidence_base": 0.75,
            "version_patterns": [r"gpt-3\.5", r"gpt-3\.5-turbo", r"text-davinci"]
        },
        "gpt-4": {
            "patterns": [
                r"\b(?:comprehensive|thorough|detailed analysis|in-depth)",
                r"\b(?:let me|i'll|i will provide|allow me to)",
                r"\b(?:here's|here is a|below is|following is)",
                r"\b(?:to summarize|in summary|in conclusion|to conclude)",
                r"\b(?:it's worth noting|it should be noted|important to understand)"
            ],
            "provider": "OpenAI",
            "confidence_base": 0.82,
            "version_patterns": [r"gpt-4", r"gpt-4-turbo", r"gpt-4o"]
        },
        "claude": {
            "patterns": [
                r"\b(?:i'd be happy|i can help|let me assist|i'd be glad)",
                r"\b(?:here's how|here are|i'll explain|let me break this down)",
                r"\b(?:based on|considering|taking into account|given that)",
                r"\b(?:to clarify|to elaborate|in other words|put another way)",
                r"\b(?:i should mention|it's also worth|additionally|furthermore)"
            ],
            "provider": "Anthropic",
            "confidence_base": 0.78,
            "version_patterns": [r"claude-3", r"claude-2", r"claude-instant"]
        },
        "gemini": {
            "patterns": [
                r"\b(?:here's|here are|let me share|i can share)",
                r"\b(?:i can|i'll|i will|i'm able to)",
                r"\b(?:based on|according to|in terms of|with regard to)",
                r"\b(?:to help you|to assist you|for your reference)",
                r"\b(?:it appears|it seems|this suggests|this indicates)"
            ],
            "provider": "Google",
            "confidence_base": 0.72,
            "version_patterns": [r"gemini-pro", r"gemini-ultra", r"gemini-1"]
        },
        "llama": {
            "patterns": [
                r"\b(?:let me|i'll|i will|allow me)",
                r"\b(?:here's|here is|below|following)",
                r"\b(?:i can|i'm able|i have the ability|i'm capable)",
                r"\b(?:to answer|to respond|to help|to assist)",
                r"\b(?:it's important|it should be|one should|we should)"
            ],
            "provider": "Meta",
            "confidence_base": 0.70,
            "version_patterns": [r"llama-2", r"llama-3", r"llama-70b"]
        },
        "mistral": {
            "patterns": [
                r"\b(?:let me|i'll|i can|i will)",
                r"\b(?:here's|here are|below|following)",
                r"\b(?:to help|to assist|to provide|to offer)",
                r"\b(?:it's worth|it should be|one must|we must)"
            ],
            "provider": "Mistral AI",
            "confidence_base": 0.68,
            "version_patterns": [r"mistral-7b", r"mistral-medium", r"mixtral"]
        },
        "cohere": {
            "patterns": [
                r"\b(?:let me|i'll|i can|i will)",
                r"\b(?:here's|here are|below|following)",
                r"\b(?:to help|to assist|to provide|to offer)",
                r"\b(?:based on|according to|in terms of)"
            ],
            "provider": "Cohere",
            "confidence_base": 0.65,
            "version_patterns": [r"command", r"command-light", r"command-nightly"]
        },
        "palm": {
            "patterns": [
                r"\b(?:here's|here are|let me|i can)",
                r"\b(?:based on|according to|in terms of)",
                r"\b(?:to help|to assist|to provide)"
            ],
            "provider": "Google",
            "confidence_base": 0.63,
            "version_patterns": [r"palm-2", r"palm", r"text-bison"]
        },
        "jurassic": {
            "patterns": [
                r"\b(?:let me|i'll|i can|i will)",
                r"\b(?:here's|here are|below|following)",
                r"\b(?:to help|to assist|to provide)"
            ],
            "provider": "AI21 Labs",
            "confidence_base": 0.60,
            "version_patterns": [r"j2", r"jurassic-2", r"jamba"]
        },
        "groq": {
            "patterns": [
                r"\b(?:let me|i'll|i can|i will)",
                r"\b(?:here's|here are|below|following)",
                r"\b(?:to help|to assist|to provide)"
            ],
            "provider": "Groq",
            "confidence_base": 0.62,
            "version_patterns": [r"mixtral", r"llama-3", r"gemma"]
        },
        "openrouter": {
            "patterns": [
                r"\b(?:let me|i'll|i can|i will)",
                r"\b(?:here's|here are|below|following)",
                r"\b(?:to help|to assist|to provide)"
            ],
            "provider": "OpenRouter",
            "confidence_base": 0.60,
            "version_patterns": []
        }
    }
    
    # Características de texto generado por IA
    AI_TEXT_FEATURES = {
        "perplexity_threshold": 50.0,  # Texto de IA suele tener menor perplexity
        "burstiness_threshold": 0.5,   # Menos variación en longitud de oraciones
        "repetition_threshold": 0.3,   # Menos repetición de palabras
        "coherence_score": 0.8,        # Alta coherencia
        "formality_score": 0.7         # Texto más formal
    }
    
    def __init__(self):
        self.start_time = time.time()
        self.models_loaded = 0
        self.detection_cache = {}  # Cache para resultados
        self.cache_max_size = 1000  # Tamaño máximo del cache
        self.detection_history = []  # Historial de detecciones
        self.max_history_size = 100  # Tamaño máximo del historial
        self.adaptive_weights = {}  # Pesos adaptativos basados en historial
        self.model_performance = {}  # Rendimiento de cada método
        self.known_ai_texts = []  # Textos conocidos de IA para comparación
        self.known_human_texts = []  # Textos conocidos humanos para comparación
        self.max_known_texts = 50  # Máximo de textos conocidos
        self.alert_threshold = 0.8  # Umbral para alertas de alta confianza
        self.model_signatures = {}  # Firmas características de cada modelo
        logger.info("MultimodalAIDetector inicializado")
    
    def detect(self, content: str, content_type: str = "text", metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Detecta si el contenido fue generado por IA
        
        Args:
            content: Contenido a analizar
            content_type: Tipo de contenido (text, image, audio, video)
            metadata: Metadatos adicionales
            
        Returns:
            Diccionario con resultados de la detección
        """
        start_time = time.time()
        
        # Verificar cache
        cache_key = self._generate_cache_key(content, content_type)
        if cache_key in self.detection_cache:
            cached_result = self.detection_cache[cache_key].copy()
            cached_result["from_cache"] = True
            cached_result["processing_time"] = time.time() - start_time
            return cached_result
        
        try:
            if content_type == "text":
                result = self._detect_text_ai(content, metadata)
            elif content_type == "image":
                result = self._detect_image_ai(content, metadata)
            elif content_type == "audio":
                result = self._detect_audio_ai(content, metadata)
            elif content_type == "video":
                result = self._detect_video_ai(content, metadata)
            else:
                result = self._detect_multimodal(content, metadata)
            
            result["processing_time"] = time.time() - start_time
            result["timestamp"] = time.time()
            result["from_cache"] = False
            
            # Guardar en cache
            self._add_to_cache(cache_key, result)
            
            # Guardar en historial
            self._add_to_history(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error en detección: {e}", exc_info=True)
            raise
    
    def _generate_cache_key(self, content: str, content_type: str) -> str:
        """Genera una clave de cache para el contenido"""
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        return f"{content_type}:{content_hash}"
    
    def _add_to_cache(self, cache_key: str, result: Dict[str, Any]):
        """Añade resultado al cache"""
        if len(self.detection_cache) >= self.cache_max_size:
            # Eliminar el más antiguo (FIFO)
            oldest_key = next(iter(self.detection_cache))
            del self.detection_cache[oldest_key]
        
        self.detection_cache[cache_key] = result.copy()
    
    def clear_cache(self):
        """Limpia el cache de detecciones"""
        self.detection_cache.clear()
        logger.info("Cache de detecciones limpiado")
    
    def _add_to_history(self, result: Dict[str, Any]):
        """Añade resultado al historial"""
        if len(self.detection_history) >= self.max_history_size:
            # Eliminar el más antiguo (FIFO)
            self.detection_history.pop(0)
        
        # Guardar solo información esencial
        history_entry = {
            "timestamp": result.get("timestamp", time.time()),
            "is_ai_generated": result.get("is_ai_generated", False),
            "ai_percentage": result.get("ai_percentage", 0.0),
            "primary_model": result.get("primary_model", {}).get("model_name") if result.get("primary_model") else None,
            "confidence_score": result.get("confidence_score", 0.0)
        }
        self.detection_history.append(history_entry)
    
    def get_detection_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Obtiene el historial de detecciones"""
        return self.detection_history[-limit:] if limit > 0 else self.detection_history
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas del detector"""
        if not self.detection_history:
            return {
                "total_detections": 0,
                "ai_detections": 0,
                "human_detections": 0,
                "avg_ai_percentage": 0.0,
                "avg_confidence": 0.0,
                "most_common_model": None
            }
        
        total = len(self.detection_history)
        ai_detections = sum(1 for entry in self.detection_history if entry.get("is_ai_generated", False))
        human_detections = total - ai_detections
        
        avg_ai_percentage = np.mean([entry.get("ai_percentage", 0.0) for entry in self.detection_history])
        avg_confidence = np.mean([entry.get("confidence_score", 0.0) for entry in self.detection_history])
        
        # Modelo más común
        models = [entry.get("primary_model") for entry in self.detection_history if entry.get("primary_model")]
        most_common_model = max(set(models), key=models.count) if models else None
        
        return {
            "total_detections": total,
            "ai_detections": ai_detections,
            "human_detections": human_detections,
            "ai_detection_rate": (ai_detections / total * 100) if total > 0 else 0.0,
            "avg_ai_percentage": avg_ai_percentage,
            "avg_confidence": avg_confidence,
            "most_common_model": most_common_model,
            "cache_size": len(self.detection_cache),
            "cache_usage_percent": (len(self.detection_cache) / self.cache_max_size * 100) if self.cache_max_size > 0 else 0
        }
    
    def _detect_text_ai(self, text: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Detecta si un texto fue generado por IA - MEJORADO"""
        detection_methods = []
        scores = []
        detected_models = []
        method_weights = {}  # Pesos para cada método
        
        # Método 1: Análisis de patrones de modelos (peso alto)
        model_detections = self._detect_model_patterns(text)
        if model_detections:
            detected_models.extend(model_detections)
            detection_methods.append("pattern_matching")
            pattern_score = max([m["confidence"] for m in model_detections])
            scores.append(pattern_score)
            method_weights["pattern_matching"] = 0.35  # Peso alto
        
        # Método 2: Análisis estadístico de texto (peso medio-alto)
        text_stats = self._analyze_text_statistics(text)
        if text_stats["ai_likelihood"] > 0.3:
            detection_methods.append("statistical_analysis")
            scores.append(text_stats["ai_likelihood"])
            method_weights["statistical_analysis"] = 0.25
        
        # Método 3: Análisis de estructura y coherencia (peso medio)
        structure_score = self._analyze_text_structure(text)
        if structure_score > 0.4:
            detection_methods.append("structure_analysis")
            scores.append(structure_score)
            method_weights["structure_analysis"] = 0.15
        
        # Método 4: Análisis de estilo (peso medio)
        style_score = self._analyze_text_style(text)
        if style_score > 0.3:
            detection_methods.append("style_analysis")
            scores.append(style_score)
            method_weights["style_analysis"] = 0.15
        
        # Método 5: Análisis de entropía y n-gramas (NUEVO)
        entropy_score = self._analyze_entropy_and_ngrams(text)
        if entropy_score > 0.3:
            detection_methods.append("entropy_analysis")
            scores.append(entropy_score)
            method_weights["entropy_analysis"] = 0.10
        
        # Método 6: Análisis de coherencia semántica (NUEVO)
        semantic_score = self._analyze_semantic_coherence(text)
        if semantic_score > 0.3:
            detection_methods.append("semantic_coherence")
            scores.append(semantic_score)
            method_weights["semantic_coherence"] = 0.08
        
        # Método 7: Análisis de complejidad sintáctica (NUEVO)
        syntactic_score = self._analyze_syntactic_complexity(text)
        if syntactic_score > 0.3:
            detection_methods.append("syntactic_complexity")
            scores.append(syntactic_score)
            method_weights["syntactic_complexity"] = 0.07
        
        # Método 8: Análisis de citas y referencias (NUEVO)
        citation_score = self._analyze_citations_and_references(text)
        if citation_score > 0.3:
            detection_methods.append("citation_analysis")
            scores.append(citation_score)
            method_weights["citation_analysis"] = 0.05
        
        # Método 9: Análisis temporal (NUEVO)
        temporal_score = self._analyze_temporal_consistency(text)
        if temporal_score > 0.3:
            detection_methods.append("temporal_analysis")
            scores.append(temporal_score)
            method_weights["temporal_analysis"] = 0.04
        
        # Método 10: Detección de watermarks (NUEVO)
        watermark_score = self._detect_watermarks(text)
        if watermark_score > 0.2:
            detection_methods.append("watermark_detection")
            scores.append(watermark_score)
            method_weights["watermark_detection"] = 0.03
        
        # Método 11: Análisis de ediciones/parches (NUEVO)
        edit_score = self._detect_edits_and_patches(text)
        if edit_score > 0.2:
            detection_methods.append("edit_detection")
            scores.append(edit_score)
            method_weights["edit_detection"] = 0.02
        
        # Método 12: Análisis de sentimientos (NUEVO)
        sentiment_score = self._analyze_sentiment_patterns(text)
        if sentiment_score > 0.3:
            detection_methods.append("sentiment_analysis")
            scores.append(sentiment_score)
            method_weights["sentiment_analysis"] = 0.02
        
        # Método 13: Análisis de contexto y coherencia temática (NUEVO)
        context_score = self._analyze_contextual_coherence(text)
        if context_score > 0.3:
            detection_methods.append("contextual_analysis")
            scores.append(context_score)
            method_weights["contextual_analysis"] = 0.03
        
        # Método 14: Detección de traducción automática (NUEVO)
        translation_score = self._detect_machine_translation(text)
        if translation_score > 0.2:
            detection_methods.append("translation_detection")
            scores.append(translation_score)
            method_weights["translation_detection"] = 0.02
        
        # Método 15: Análisis de patrones de generación (NUEVO)
        generation_score = self._analyze_generation_patterns(text)
        if generation_score > 0.3:
            detection_methods.append("generation_patterns")
            scores.append(generation_score)
            method_weights["generation_patterns"] = 0.02
        
        # Método 16: Análisis de calidad de escritura (NUEVO)
        quality_score = self._analyze_writing_quality(text)
        if quality_score > 0.3:
            detection_methods.append("writing_quality")
            scores.append(quality_score)
            method_weights["writing_quality"] = 0.02
        
        # Método 17: Detección de parafraseo (NUEVO)
        paraphrase_score = self._detect_paraphrasing(text)
        if paraphrase_score > 0.2:
            detection_methods.append("paraphrase_detection")
            scores.append(paraphrase_score)
            method_weights["paraphrase_detection"] = 0.02
        
        # Método 18: Análisis de riesgo y confiabilidad (NUEVO)
        risk_score = self._analyze_risk_and_reliability(text, detected_models, scores)
        if risk_score > 0.2:
            detection_methods.append("risk_analysis")
            scores.append(risk_score)
            method_weights["risk_analysis"] = 0.01
        
        # Método 19: Análisis de metadatos y contexto (NUEVO)
        if metadata:
            metadata_score = self._analyze_metadata_and_context(text, metadata)
            if metadata_score > 0.2:
                detection_methods.append("metadata_analysis")
                scores.append(metadata_score)
                method_weights["metadata_analysis"] = 0.01
        
        # Método 20: Análisis de idioma y localización (NUEVO)
        language_score = self._analyze_language_patterns(text)
        if language_score > 0.2:
            detection_methods.append("language_analysis")
            scores.append(language_score)
            method_weights["language_analysis"] = 0.01
        
        # Método 21: Análisis de similitud semántica (NUEVO)
        semantic_similarity_score = self._analyze_semantic_similarity(text)
        if semantic_similarity_score > 0.2:
            detection_methods.append("semantic_similarity")
            scores.append(semantic_similarity_score)
            method_weights["semantic_similarity"] = 0.01
        
        # Método 22: Análisis de frecuencia de palabras clave (NUEVO)
        keyword_frequency_score = self._analyze_keyword_frequency(text)
        if keyword_frequency_score > 0.2:
            detection_methods.append("keyword_frequency")
            scores.append(keyword_frequency_score)
            method_weights["keyword_frequency"] = 0.01
        
        # Método 23: Detección de patrones de respuesta típicos (NUEVO)
        response_pattern_score = self._detect_response_patterns(text)
        if response_pattern_score > 0.2:
            detection_methods.append("response_patterns")
            scores.append(response_pattern_score)
            method_weights["response_patterns"] = 0.01
        
        # Método 24: Análisis de coherencia narrativa (NUEVO)
        narrative_coherence_score = self._analyze_narrative_coherence(text)
        if narrative_coherence_score > 0.2:
            detection_methods.append("narrative_coherence")
            scores.append(narrative_coherence_score)
            method_weights["narrative_coherence"] = 0.01
        
        # Sistema de scoring adaptativo (NUEVO)
        method_weights = self._apply_adaptive_weights(method_weights, detection_methods, text)
        
        # Ajustar pesos si hay nuevos métodos
        total_current_weight = sum(method_weights.values())
        if total_current_weight > 1.0:
            # Normalizar pesos si exceden 1.0
            method_weights = {k: v / total_current_weight for k, v in method_weights.items()}
        
        # Análisis de contexto histórico (NUEVO)
        historical_context_score = self._analyze_historical_context(text, detected_models)
        if historical_context_score > 0.2:
            detection_methods.append("historical_context")
            scores.append(historical_context_score)
            method_weights["historical_context"] = 0.01
        
        # Análisis de n-gramas avanzado (NUEVO)
        advanced_ngram_score = self._analyze_advanced_ngrams(text)
        if advanced_ngram_score > 0.2:
            detection_methods.append("advanced_ngrams")
            scores.append(advanced_ngram_score)
            method_weights["advanced_ngrams"] = 0.01
        
        # Análisis comparativo con textos conocidos (NUEVO)
        comparative_score = self._analyze_comparative_similarity(text)
        if comparative_score > 0.2:
            detection_methods.append("comparative_analysis")
            scores.append(comparative_score)
            method_weights["comparative_analysis"] = 0.01
        
        # Análisis de aprendizaje automático básico (NUEVO)
        ml_score = self._analyze_with_ml_patterns(text, detected_models, scores)
        if ml_score > 0.2:
            detection_methods.append("ml_patterns")
            scores.append(ml_score)
            method_weights["ml_patterns"] = 0.01
        
        # Análisis de firmas de modelos específicos (NUEVO)
        signature_score = self._analyze_model_signatures(text, detected_models)
        if signature_score > 0.2:
            detection_methods.append("model_signatures")
            scores.append(signature_score)
            method_weights["model_signatures"] = 0.01
        
        # Análisis de embeddings semánticos básico (NUEVO)
        embedding_score = self._analyze_semantic_embeddings(text)
        if embedding_score > 0.2:
            detection_methods.append("semantic_embeddings")
            scores.append(embedding_score)
            method_weights["semantic_embeddings"] = 0.01
        
        # Análisis de patrones temporales (NUEVO)
        temporal_pattern_score = self._analyze_temporal_patterns(text, metadata)
        if temporal_pattern_score > 0.2:
            detection_methods.append("temporal_patterns")
            scores.append(temporal_pattern_score)
            method_weights["temporal_patterns"] = 0.01
        
        # Detección de modelos híbridos (NUEVO)
        hybrid_score = self._detect_hybrid_models(text, detected_models)
        if hybrid_score > 0.2:
            detection_methods.append("hybrid_detection")
            scores.append(hybrid_score)
            method_weights["hybrid_detection"] = 0.01
        
        # Análisis de frecuencia avanzado (NUEVO)
        frequency_score = self._analyze_advanced_frequency(text)
        if frequency_score > 0.2:
            detection_methods.append("advanced_frequency")
            scores.append(frequency_score)
            method_weights["advanced_frequency"] = 0.01
        
        # Análisis de coherencia contextual avanzado (NUEVO)
        advanced_context_score = self._analyze_advanced_contextual_coherence(text)
        if advanced_context_score > 0.2:
            detection_methods.append("advanced_contextual")
            scores.append(advanced_context_score)
            method_weights["advanced_contextual"] = 0.01
        
        # Detección de deepfake de texto (NUEVO)
        deepfake_score = self._detect_text_deepfake(text, detected_models)
        if deepfake_score > 0.2:
            detection_methods.append("text_deepfake")
            scores.append(deepfake_score)
            method_weights["text_deepfake"] = 0.01
        
        # Análisis de calidad de escritura avanzado (NUEVO)
        advanced_quality_score = self._analyze_advanced_writing_quality(text)
        if advanced_quality_score > 0.2:
            detection_methods.append("advanced_writing_quality")
            scores.append(advanced_quality_score)
            method_weights["advanced_writing_quality"] = 0.01
        
        # Detección de patrones de deepfake (NUEVO)
        deepfake_pattern_score = self._detect_deepfake_patterns(text)
        if deepfake_pattern_score > 0.2:
            detection_methods.append("deepfake_patterns")
            scores.append(deepfake_pattern_score)
            method_weights["deepfake_patterns"] = 0.01
        
        # Sistema de scoring mejorado (NUEVO)
        enhanced_score = self._enhanced_scoring_system(scores, detection_methods, detected_models)
        if enhanced_score > 0.2:
            detection_methods.append("enhanced_scoring")
            scores.append(enhanced_score)
            method_weights["enhanced_scoring"] = 0.02
        
        # Análisis avanzado de patrones de repetición (NUEVO)
        repetition_score = self._analyze_advanced_repetition_patterns(text)
        if repetition_score > 0.2:
            detection_methods.append("advanced_repetition")
            scores.append(repetition_score)
            method_weights["advanced_repetition"] = 0.01
        
        # Detección avanzada de parafraseo con IA (NUEVO)
        ai_paraphrase_score = self._detect_ai_paraphrasing_advanced(text)
        if ai_paraphrase_score > 0.2:
            detection_methods.append("ai_paraphrasing_advanced")
            scores.append(ai_paraphrase_score)
            method_weights["ai_paraphrasing_advanced"] = 0.01
        
        # Detección de mezclas de estilos (NUEVO)
        style_mixture_score = self._analyze_style_mixture(text)
        if style_mixture_score > 0.2:
            detection_methods.append("style_mixture")
            scores.append(style_mixture_score)
            method_weights["style_mixture"] = 0.01
        
        # Análisis sofisticado de patrones de generación (NUEVO)
        generation_sophistication_score = self._analyze_generation_sophistication(text)
        if generation_sophistication_score > 0.2:
            detection_methods.append("generation_sophistication")
            scores.append(generation_sophistication_score)
            method_weights["generation_sophistication"] = 0.01
        
        # Análisis avanzado de diversidad léxica (NUEVO)
        lexical_diversity_score = self._analyze_lexical_diversity_advanced(text)
        if lexical_diversity_score > 0.2:
            detection_methods.append("lexical_diversity_advanced")
            scores.append(lexical_diversity_score)
            method_weights["lexical_diversity_advanced"] = 0.01
        
        # Detección de patrones de hedging (NUEVO)
        hedging_score = self._detect_ai_hedging_patterns(text)
        if hedging_score > 0.2:
            detection_methods.append("ai_hedging_patterns")
            scores.append(hedging_score)
            method_weights["ai_hedging_patterns"] = 0.01
        
        # Análisis de distribución de complejidad de oraciones (NUEVO)
        complexity_distribution_score = self._analyze_sentence_complexity_distribution(text)
        if complexity_distribution_score > 0.2:
            detection_methods.append("complexity_distribution")
            scores.append(complexity_distribution_score)
            method_weights["complexity_distribution"] = 0.01
        
        # Detección de patrones de verbosidad (NUEVO)
        verbosity_score = self._detect_ai_verbosity_patterns(text)
        if verbosity_score > 0.2:
            detection_methods.append("ai_verbosity_patterns")
            scores.append(verbosity_score)
            method_weights["ai_verbosity_patterns"] = 0.01
        
        # Análisis de patrones de uso de pronombres (NUEVO)
        pronoun_score = self._analyze_pronoun_usage_patterns(text)
        if pronoun_score > 0.2:
            detection_methods.append("pronoun_usage_patterns")
            scores.append(pronoun_score)
            method_weights["pronoun_usage_patterns"] = 0.01
        
        # Detección de patrones de preguntas (NUEVO)
        question_score = self._detect_ai_question_patterns(text)
        if question_score > 0.2:
            detection_methods.append("ai_question_patterns")
            scores.append(question_score)
            method_weights["ai_question_patterns"] = 0.01
        
        # Análisis de patrones de cierre (NUEVO)
        closure_score = self._analyze_ai_closure_patterns(text)
        if closure_score > 0.2:
            detection_methods.append("ai_closure_patterns")
            scores.append(closure_score)
            method_weights["ai_closure_patterns"] = 0.01
        
        # Detección de patrones de enumeración (NUEVO)
        enumeration_score = self._detect_ai_enumeration_patterns(text)
        if enumeration_score > 0.2:
            detection_methods.append("ai_enumeration_patterns")
            scores.append(enumeration_score)
            method_weights["ai_enumeration_patterns"] = 0.01
        
        # Análisis de patrones de metáforas (NUEVO)
        metaphor_score = self._analyze_ai_metaphor_patterns(text)
        if metaphor_score > 0.2:
            detection_methods.append("ai_metaphor_patterns")
            scores.append(metaphor_score)
            method_weights["ai_metaphor_patterns"] = 0.01
        
        # Detección de patrones de énfasis (NUEVO)
        emphasis_score = self._detect_ai_emphasis_patterns(text)
        if emphasis_score > 0.2:
            detection_methods.append("ai_emphasis_patterns")
            scores.append(emphasis_score)
            method_weights["ai_emphasis_patterns"] = 0.01
        
        # Análisis de patrones de modificadores (NUEVO)
        modifier_score = self._analyze_ai_modifier_patterns(text)
        if modifier_score > 0.2:
            detection_methods.append("ai_modifier_patterns")
            scores.append(modifier_score)
            method_weights["ai_modifier_patterns"] = 0.01
        
        # Detección de patrones condicionales (NUEVO)
        conditional_score = self._detect_ai_conditional_patterns(text)
        if conditional_score > 0.2:
            detection_methods.append("ai_conditional_patterns")
            scores.append(conditional_score)
            method_weights["ai_conditional_patterns"] = 0.01
        
        # Análisis de patrones de voz pasiva (NUEVO)
        passive_voice_score = self._analyze_ai_passive_voice_patterns(text)
        if passive_voice_score > 0.2:
            detection_methods.append("ai_passive_voice_patterns")
            scores.append(passive_voice_score)
            method_weights["ai_passive_voice_patterns"] = 0.01
        
        # Detección de patrones de conectores (NUEVO)
        connector_score = self._detect_ai_connector_patterns(text)
        if connector_score > 0.2:
            detection_methods.append("ai_connector_patterns")
            scores.append(connector_score)
            method_weights["ai_connector_patterns"] = 0.01
        
        # Análisis de patrones de cuantificadores (NUEVO)
        quantifier_score = self._analyze_ai_quantifier_patterns(text)
        if quantifier_score > 0.2:
            detection_methods.append("ai_quantifier_patterns")
            scores.append(quantifier_score)
            method_weights["ai_quantifier_patterns"] = 0.01
        
        # Detección de patrones de aserciones (NUEVO)
        assertion_score = self._detect_ai_assertion_patterns(text)
        if assertion_score > 0.2:
            detection_methods.append("ai_assertion_patterns")
            scores.append(assertion_score)
            method_weights["ai_assertion_patterns"] = 0.01
        
        # Análisis de patrones de negación (NUEVO)
        negation_score = self._analyze_ai_negation_patterns(text)
        if negation_score > 0.2:
            detection_methods.append("ai_negation_patterns")
            scores.append(negation_score)
            method_weights["ai_negation_patterns"] = 0.01
        
        # Detección de patrones de comparación (NUEVO)
        comparison_score = self._detect_ai_comparison_patterns(text)
        if comparison_score > 0.2:
            detection_methods.append("ai_comparison_patterns")
            scores.append(comparison_score)
            method_weights["ai_comparison_patterns"] = 0.01
        
        # Análisis de patrones de marcadores temporales (NUEVO)
        temporal_marker_score = self._analyze_ai_temporal_marker_patterns(text)
        if temporal_marker_score > 0.2:
            detection_methods.append("ai_temporal_marker_patterns")
            scores.append(temporal_marker_score)
            method_weights["ai_temporal_marker_patterns"] = 0.01
        
        # Detección de patrones de causalidad (NUEVO)
        causality_score = self._detect_ai_causality_patterns(text)
        if causality_score > 0.2:
            detection_methods.append("ai_causality_patterns")
            scores.append(causality_score)
            method_weights["ai_causality_patterns"] = 0.01
        
        # Análisis de patrones de verbos modales (NUEVO)
        modal_verb_score = self._analyze_ai_modal_verb_patterns(text)
        if modal_verb_score > 0.2:
            detection_methods.append("ai_modal_verb_patterns")
            scores.append(modal_verb_score)
            method_weights["ai_modal_verb_patterns"] = 0.01
        
        # Detección de patrones de frases de hedging (NUEVO)
        hedge_phrase_score = self._detect_ai_hedge_phrase_patterns(text)
        if hedge_phrase_score > 0.2:
            detection_methods.append("ai_hedge_phrase_patterns")
            scores.append(hedge_phrase_score)
            method_weights["ai_hedge_phrase_patterns"] = 0.01
        
        # Análisis de patrones de cláusulas relativas (NUEVO)
        relative_clause_score = self._analyze_ai_relative_clause_patterns(text)
        if relative_clause_score > 0.2:
            detection_methods.append("ai_relative_clause_patterns")
            scores.append(relative_clause_score)
            method_weights["ai_relative_clause_patterns"] = 0.01
        
        # Detección de patrones de infinitivos (NUEVO)
        infinitive_score = self._detect_ai_infinitive_patterns(text)
        if infinitive_score > 0.2:
            detection_methods.append("ai_infinitive_patterns")
            scores.append(infinitive_score)
            method_weights["ai_infinitive_patterns"] = 0.01
        
        # Análisis de patrones de gerundios (NUEVO)
        gerund_score = self._analyze_ai_gerund_patterns(text)
        if gerund_score > 0.2:
            detection_methods.append("ai_gerund_patterns")
            scores.append(gerund_score)
            method_weights["ai_gerund_patterns"] = 0.01
        
        # Detección de patrones de participios (NUEVO)
        participle_score = self._detect_ai_participle_patterns(text)
        if participle_score > 0.2:
            detection_methods.append("ai_participle_patterns")
            scores.append(participle_score)
            method_weights["ai_participle_patterns"] = 0.01
        
        # Análisis de patrones de subjuntivo (NUEVO)
        subjunctive_score = self._analyze_ai_subjunctive_patterns(text)
        if subjunctive_score > 0.2:
            detection_methods.append("ai_subjunctive_patterns")
            scores.append(subjunctive_score)
            method_weights["ai_subjunctive_patterns"] = 0.01
        
        # Detección de patrones de artículos (NUEVO)
        article_score = self._detect_ai_article_patterns(text)
        if article_score > 0.2:
            detection_methods.append("ai_article_patterns")
            scores.append(article_score)
            method_weights["ai_article_patterns"] = 0.01
        
        # Análisis de patrones de preposiciones (NUEVO)
        preposition_score = self._analyze_ai_preposition_patterns(text)
        if preposition_score > 0.2:
            detection_methods.append("ai_preposition_patterns")
            scores.append(preposition_score)
            method_weights["ai_preposition_patterns"] = 0.01
        
        # Detección de patrones de conjunciones (NUEVO)
        conjunction_score = self._detect_ai_conjunction_patterns(text)
        if conjunction_score > 0.2:
            detection_methods.append("ai_conjunction_patterns")
            scores.append(conjunction_score)
            method_weights["ai_conjunction_patterns"] = 0.01
        
        # Análisis de patrones de determinantes (NUEVO)
        determiner_score = self._analyze_ai_determiner_patterns(text)
        if determiner_score > 0.2:
            detection_methods.append("ai_determiner_patterns")
            scores.append(determiner_score)
            method_weights["ai_determiner_patterns"] = 0.01
        
        # Detección de patrones de referencia pronominal (NUEVO)
        pronoun_ref_score = self._detect_ai_pronoun_reference_patterns(text)
        if pronoun_ref_score > 0.2:
            detection_methods.append("ai_pronoun_reference_patterns")
            scores.append(pronoun_ref_score)
            method_weights["ai_pronoun_reference_patterns"] = 0.01
        
        # Análisis de patrones de adverbios (NUEVO)
        adverb_score = self._analyze_ai_adverb_patterns(text)
        if adverb_score > 0.2:
            detection_methods.append("ai_adverb_patterns")
            scores.append(adverb_score)
            method_weights["ai_adverb_patterns"] = 0.01
        
        # Detección de patrones de adjetivos (NUEVO)
        adjective_score = self._detect_ai_adjective_patterns(text)
        if adjective_score > 0.2:
            detection_methods.append("ai_adjective_patterns")
            scores.append(adjective_score)
            method_weights["ai_adjective_patterns"] = 0.01
        
        # Análisis de patrones de sustantivos (NUEVO)
        noun_score = self._analyze_ai_noun_patterns(text)
        if noun_score > 0.2:
            detection_methods.append("ai_noun_patterns")
            scores.append(noun_score)
            method_weights["ai_noun_patterns"] = 0.01
        
        # Detección de patrones de verbos (NUEVO)
        verb_score = self._detect_ai_verb_patterns(text)
        if verb_score > 0.2:
            detection_methods.append("ai_verb_patterns")
            scores.append(verb_score)
            method_weights["ai_verb_patterns"] = 0.01
        
        # Análisis de patrones de longitud de oraciones (NUEVO)
        sentence_length_score = self._analyze_ai_sentence_length_patterns(text)
        if sentence_length_score > 0.2:
            detection_methods.append("ai_sentence_length_patterns")
            scores.append(sentence_length_score)
            method_weights["ai_sentence_length_patterns"] = 0.01
        
        # Detección de patrones de estructura de párrafos (NUEVO)
        paragraph_structure_score = self._detect_ai_paragraph_structure_patterns(text)
        if paragraph_structure_score > 0.2:
            detection_methods.append("ai_paragraph_structure_patterns")
            scores.append(paragraph_structure_score)
            method_weights["ai_paragraph_structure_patterns"] = 0.01
        
        # Análisis de patrones de puntuación (NUEVO)
        punctuation_score = self._analyze_ai_punctuation_patterns(text)
        if punctuation_score > 0.2:
            detection_methods.append("ai_punctuation_patterns")
            scores.append(punctuation_score)
            method_weights["ai_punctuation_patterns"] = 0.01
        
        # Detección de patrones de capitalización (NUEVO)
        capitalization_score = self._detect_ai_capitalization_patterns(text)
        if capitalization_score > 0.2:
            detection_methods.append("ai_capitalization_patterns")
            scores.append(capitalization_score)
            method_weights["ai_capitalization_patterns"] = 0.01
        
        # Análisis de patrones de frecuencia de palabras (NUEVO)
        word_frequency_score = self._analyze_ai_word_frequency_patterns(text)
        if word_frequency_score > 0.2:
            detection_methods.append("ai_word_frequency_patterns")
            scores.append(word_frequency_score)
            method_weights["ai_word_frequency_patterns"] = 0.01
        
        # Detección de patrones de repetición de frases (NUEVO)
        phrase_repetition_score = self._detect_ai_phrase_repetition_patterns(text)
        if phrase_repetition_score > 0.2:
            detection_methods.append("ai_phrase_repetition_patterns")
            scores.append(phrase_repetition_score)
            method_weights["ai_phrase_repetition_patterns"] = 0.01
        
        # Análisis de patrones de densidad semántica (NUEVO)
        semantic_density_score = self._analyze_ai_semantic_density_patterns(text)
        if semantic_density_score > 0.2:
            detection_methods.append("ai_semantic_density_patterns")
            scores.append(semantic_density_score)
            method_weights["ai_semantic_density_patterns"] = 0.01
        
        # Detección de patrones de marcadores de coherencia (NUEVO)
        coherence_markers_score = self._detect_ai_coherence_markers_patterns(text)
        if coherence_markers_score > 0.2:
            detection_methods.append("ai_coherence_markers_patterns")
            scores.append(coherence_markers_score)
            method_weights["ai_coherence_markers_patterns"] = 0.01
        
        # Análisis de patrones de sofisticación léxica (NUEVO)
        lexical_sophistication_score = self._analyze_ai_lexical_sophistication_patterns(text)
        if lexical_sophistication_score > 0.2:
            detection_methods.append("ai_lexical_sophistication_patterns")
            scores.append(lexical_sophistication_score)
            method_weights["ai_lexical_sophistication_patterns"] = 0.01
        
        # Detección de patrones de formalidad (NUEVO)
        formality_score = self._detect_ai_formality_patterns(text)
        if formality_score > 0.2:
            detection_methods.append("ai_formality_patterns")
            scores.append(formality_score)
            method_weights["ai_formality_patterns"] = 0.01
        
        # Análisis de patrones de registro (NUEVO)
        register_score = self._analyze_ai_register_patterns(text)
        if register_score > 0.2:
            detection_methods.append("ai_register_patterns")
            scores.append(register_score)
            method_weights["ai_register_patterns"] = 0.01
        
        # Detección de patrones de marcadores discursivos (NUEVO)
        discourse_markers_score = self._detect_ai_discourse_markers_patterns(text)
        if discourse_markers_score > 0.2:
            detection_methods.append("ai_discourse_markers_patterns")
            scores.append(discourse_markers_score)
            method_weights["ai_discourse_markers_patterns"] = 0.01
        
        # Análisis de patrones de cohesión textual (NUEVO)
        textual_cohesion_score = self._analyze_ai_textual_cohesion_patterns(text)
        if textual_cohesion_score > 0.2:
            detection_methods.append("ai_textual_cohesion_patterns")
            scores.append(textual_cohesion_score)
            method_weights["ai_textual_cohesion_patterns"] = 0.01
        
        # Detección de patrones de densidad de información (NUEVO)
        information_density_score = self._detect_ai_information_density_patterns(text)
        if information_density_score > 0.2:
            detection_methods.append("ai_information_density_patterns")
            scores.append(information_density_score)
            method_weights["ai_information_density_patterns"] = 0.01
        
        # Análisis de patrones de densidad de hedging (NUEVO)
        hedging_density_score = self._analyze_ai_hedging_density_patterns(text)
        if hedging_density_score > 0.2:
            detection_methods.append("ai_hedging_density_patterns")
            scores.append(hedging_density_score)
            method_weights["ai_hedging_density_patterns"] = 0.01
        
        # Detección de patrones de voz autorial (NUEVO)
        authorial_voice_score = self._detect_ai_authorial_voice_patterns(text)
        if authorial_voice_score > 0.2:
            detection_methods.append("ai_authorial_voice_patterns")
            scores.append(authorial_voice_score)
            method_weights["ai_authorial_voice_patterns"] = 0.01
        
        # Análisis de patrones de variedad textual (NUEVO)
        textual_variety_score = self._analyze_ai_textual_variety_patterns(text)
        if textual_variety_score > 0.2:
            detection_methods.append("ai_textual_variety_patterns")
            scores.append(textual_variety_score)
            method_weights["ai_textual_variety_patterns"] = 0.01
        
        # Detección de patrones de repetición léxica (NUEVO)
        lexical_repetition_score = self._detect_ai_lexical_repetition_patterns(text)
        if lexical_repetition_score > 0.2:
            detection_methods.append("ai_lexical_repetition_patterns")
            scores.append(lexical_repetition_score)
            method_weights["ai_lexical_repetition_patterns"] = 0.01
        
        # Análisis de patrones de uniformidad sintáctica (NUEVO)
        syntactic_uniformity_score = self._analyze_ai_syntactic_uniformity_patterns(text)
        if syntactic_uniformity_score > 0.2:
            detection_methods.append("ai_syntactic_uniformity_patterns")
            scores.append(syntactic_uniformity_score)
            method_weights["ai_syntactic_uniformity_patterns"] = 0.01
        
        # Detección de patrones de expresión emocional (NUEVO)
        emotional_expression_score = self._detect_ai_emotional_expression_patterns(text)
        if emotional_expression_score > 0.2:
            detection_methods.append("ai_emotional_expression_patterns")
            scores.append(emotional_expression_score)
            method_weights["ai_emotional_expression_patterns"] = 0.01
        
        # Análisis de patrones de ambigüedad contextual (NUEVO V28)
        contextual_ambiguity_score = self._analyze_ai_contextual_ambiguity_patterns(text)
        if contextual_ambiguity_score > 0.2:
            detection_methods.append("ai_contextual_ambiguity_patterns")
            scores.append(contextual_ambiguity_score)
            method_weights["ai_contextual_ambiguity_patterns"] = 0.01
        
        # Detección de patrones de riqueza léxica (NUEVO V28)
        lexical_richness_score = self._detect_ai_lexical_richness_patterns(text)
        if lexical_richness_score > 0.2:
            detection_methods.append("ai_lexical_richness_patterns")
            scores.append(lexical_richness_score)
            method_weights["ai_lexical_richness_patterns"] = 0.01
        
        # Análisis de patrones de variación sintáctica (NUEVO V28)
        syntactic_variation_score = self._analyze_ai_syntactic_variation_patterns(text)
        if syntactic_variation_score > 0.2:
            detection_methods.append("ai_syntactic_variation_patterns")
            scores.append(syntactic_variation_score)
            method_weights["ai_syntactic_variation_patterns"] = 0.01
        
        # Detección de patrones de coherencia discursiva (NUEVO V28)
        discourse_coherence_score = self._detect_ai_discourse_coherence_patterns(text)
        if discourse_coherence_score > 0.2:
            detection_methods.append("ai_discourse_coherence_patterns")
            scores.append(discourse_coherence_score)
            method_weights["ai_discourse_coherence_patterns"] = 0.01
        
        # Análisis de patrones de ritmo textual (NUEVO V29)
        textual_rhythm_score = self._analyze_ai_textual_rhythm_patterns(text)
        if textual_rhythm_score > 0.2:
            detection_methods.append("ai_textual_rhythm_patterns")
            scores.append(textual_rhythm_score)
            method_weights["ai_textual_rhythm_patterns"] = 0.01
        
        # Detección de patrones de redundancia semántica (NUEVO V29)
        semantic_redundancy_score = self._detect_ai_semantic_redundancy_patterns(text)
        if semantic_redundancy_score > 0.2:
            detection_methods.append("ai_semantic_redundancy_patterns")
            scores.append(semantic_redundancy_score)
            method_weights["ai_semantic_redundancy_patterns"] = 0.01
        
        # Análisis avanzado de sofisticación léxica (NUEVO V29)
        lexical_sophistication_advanced_score = self._analyze_ai_lexical_sophistication_advanced(text)
        if lexical_sophistication_advanced_score > 0.2:
            detection_methods.append("ai_lexical_sophistication_advanced")
            scores.append(lexical_sophistication_advanced_score)
            method_weights["ai_lexical_sophistication_advanced"] = 0.01
        
        # Detección de patrones de marcadores pragmáticos (NUEVO V29)
        pragmatic_markers_score = self._detect_ai_pragmatic_markers_patterns(text)
        if pragmatic_markers_score > 0.2:
            detection_methods.append("ai_pragmatic_markers_patterns")
            scores.append(pragmatic_markers_score)
            method_weights["ai_pragmatic_markers_patterns"] = 0.01
        
        # Análisis de patrones conversacionales (NUEVO V30)
        conversational_score = self._analyze_ai_conversational_patterns(text)
        if conversational_score > 0.2:
            detection_methods.append("ai_conversational_patterns")
            scores.append(conversational_score)
            method_weights["ai_conversational_patterns"] = 0.01
        
        # Detección de patrones de metadiscurso (NUEVO V30)
        metadiscourse_score = self._detect_ai_metadiscourse_patterns(text)
        if metadiscourse_score > 0.2:
            detection_methods.append("ai_metadiscourse_patterns")
            scores.append(metadiscourse_score)
            method_weights["ai_metadiscourse_patterns"] = 0.01
        
        # Análisis de patrones de evidencialidad (NUEVO V30)
        evidentiality_score = self._analyze_ai_evidentiality_patterns(text)
        if evidentiality_score > 0.2:
            detection_methods.append("ai_evidentiality_patterns")
            scores.append(evidentiality_score)
            method_weights["ai_evidentiality_patterns"] = 0.01
        
        # Detección de patrones de engagement (NUEVO V30)
        engagement_score = self._detect_ai_engagement_patterns(text)
        if engagement_score > 0.2:
            detection_methods.append("ai_engagement_patterns")
            scores.append(engagement_score)
            method_weights["ai_engagement_patterns"] = 0.01
        
        # Análisis de patrones de cortesía (NUEVO V31)
        politeness_score = self._analyze_ai_politeness_patterns(text)
        if politeness_score > 0.2:
            detection_methods.append("ai_politeness_patterns")
            scores.append(politeness_score)
            method_weights["ai_politeness_patterns"] = 0.01
        
        # Detección de patrones de marcadores de formalidad (NUEVO V31)
        formality_markers_score = self._detect_ai_formality_markers_patterns(text)
        if formality_markers_score > 0.2:
            detection_methods.append("ai_formality_markers_patterns")
            scores.append(formality_markers_score)
            method_weights["ai_formality_markers_patterns"] = 0.01
        
        # Análisis avanzado de patrones de hedging (NUEVO V31)
        hedging_advanced_score = self._analyze_ai_hedging_advanced_patterns(text)
        if hedging_advanced_score > 0.2:
            detection_methods.append("ai_hedging_advanced_patterns")
            scores.append(hedging_advanced_score)
            method_weights["ai_hedging_advanced_patterns"] = 0.01
        
        # Detección de patrones de asertividad (NUEVO V31)
        assertiveness_score = self._detect_ai_assertiveness_patterns(text)
        if assertiveness_score > 0.2:
            detection_methods.append("ai_assertiveness_patterns")
            scores.append(assertiveness_score)
            method_weights["ai_assertiveness_patterns"] = 0.01
        
        # Análisis de patrones de intertextualidad (NUEVO V32)
        intertextuality_score = self._analyze_ai_intertextuality_patterns(text)
        if intertextuality_score > 0.2:
            detection_methods.append("ai_intertextuality_patterns")
            scores.append(intertextuality_score)
            method_weights["ai_intertextuality_patterns"] = 0.01
        
        # Detección de patrones de densidad de citas (NUEVO V32)
        citation_density_score = self._detect_ai_citation_density_patterns(text)
        if citation_density_score > 0.2:
            detection_methods.append("ai_citation_density_patterns")
            scores.append(citation_density_score)
            method_weights["ai_citation_density_patterns"] = 0.01
        
        # Análisis de patrones de afirmaciones de autoridad (NUEVO V32)
        authority_claims_score = self._analyze_ai_authority_claims_patterns(text)
        if authority_claims_score > 0.2:
            detection_methods.append("ai_authority_claims_patterns")
            scores.append(authority_claims_score)
            method_weights["ai_authority_claims_patterns"] = 0.01
        
        # Detección de patrones de marcadores de expertise (NUEVO V32)
        expertise_markers_score = self._detect_ai_expertise_markers_patterns(text)
        if expertise_markers_score > 0.2:
            detection_methods.append("ai_expertise_markers_patterns")
            scores.append(expertise_markers_score)
            method_weights["ai_expertise_markers_patterns"] = 0.01
        
        # Análisis de patrones de coherencia temporal (NUEVO V33)
        temporal_coherence_score = self._analyze_ai_temporal_coherence_patterns(text)
        if temporal_coherence_score > 0.2:
            detection_methods.append("ai_temporal_coherence_patterns")
            scores.append(temporal_coherence_score)
            method_weights["ai_temporal_coherence_patterns"] = 0.01
        
        # Detección de patrones de cadenas causales (NUEVO V33)
        causal_chain_score = self._detect_ai_causal_chain_patterns(text)
        if causal_chain_score > 0.2:
            detection_methods.append("ai_causal_chain_patterns")
            scores.append(causal_chain_score)
            method_weights["ai_causal_chain_patterns"] = 0.01
        
        # Análisis de patrones de estructura narrativa (NUEVO V33)
        narrative_structure_score = self._analyze_ai_narrative_structure_patterns(text)
        if narrative_structure_score > 0.2:
            detection_methods.append("ai_narrative_structure_patterns")
            scores.append(narrative_structure_score)
            method_weights["ai_narrative_structure_patterns"] = 0.01
        
        # Detección de patrones de argumentación (NUEVO V33)
        argumentation_score = self._detect_ai_argumentation_patterns(text)
        if argumentation_score > 0.2:
            detection_methods.append("ai_argumentation_patterns")
            scores.append(argumentation_score)
            method_weights["ai_argumentation_patterns"] = 0.01
        
        # Análisis de patrones de consistencia léxica (NUEVO V34)
        lexical_consistency_score = self._analyze_ai_lexical_consistency_patterns(text)
        if lexical_consistency_score > 0.2:
            detection_methods.append("ai_lexical_consistency_patterns")
            scores.append(lexical_consistency_score)
            method_weights["ai_lexical_consistency_patterns"] = 0.01
        
        # Detección de patrones de campos semánticos (NUEVO V34)
        semantic_field_score = self._detect_ai_semantic_field_patterns(text)
        if semantic_field_score > 0.2:
            detection_methods.append("ai_semantic_field_patterns")
            scores.append(semantic_field_score)
            method_weights["ai_semantic_field_patterns"] = 0.01
        
        # Análisis de patrones de consistencia de registro (NUEVO V34)
        register_consistency_score = self._analyze_ai_register_consistency_patterns(text)
        if register_consistency_score > 0.2:
            detection_methods.append("ai_register_consistency_patterns")
            scores.append(register_consistency_score)
            method_weights["ai_register_consistency_patterns"] = 0.01
        
        # Detección de patrones de uniformidad estilística (NUEVO V34)
        stylistic_uniformity_score = self._detect_ai_stylistic_uniformity_patterns(text)
        if stylistic_uniformity_score > 0.2:
            detection_methods.append("ai_stylistic_uniformity_patterns")
            scores.append(stylistic_uniformity_score)
            method_weights["ai_stylistic_uniformity_patterns"] = 0.01
        
        # Análisis de patrones de fraseología (NUEVO V35)
        phraseology_score = self._analyze_ai_phraseology_patterns(text)
        if phraseology_score > 0.2:
            detection_methods.append("ai_phraseology_patterns")
            scores.append(phraseology_score)
            method_weights["ai_phraseology_patterns"] = 0.01
        
        # Detección de patrones de colocaciones (NUEVO V35)
        collocation_score = self._detect_ai_collocation_patterns(text)
        if collocation_score > 0.2:
            detection_methods.append("ai_collocation_patterns")
            scores.append(collocation_score)
            method_weights["ai_collocation_patterns"] = 0.01
        
        # Análisis de patrones idiomáticos (NUEVO V35)
        idiomatic_score = self._analyze_ai_idiomatic_patterns(text)
        if idiomatic_score > 0.2:
            detection_methods.append("ai_idiomatic_patterns")
            scores.append(idiomatic_score)
            method_weights["ai_idiomatic_patterns"] = 0.01
        
        # Detección de patrones de referencias culturales (NUEVO V35)
        cultural_references_score = self._detect_ai_cultural_references_patterns(text)
        if cultural_references_score > 0.2:
            detection_methods.append("ai_cultural_references_patterns")
            scores.append(cultural_references_score)
            method_weights["ai_cultural_references_patterns"] = 0.01
        
        # Análisis de patrones metafóricos (NUEVO V36)
        metaphorical_score = self._analyze_ai_metaphorical_patterns(text)
        if metaphorical_score > 0.2:
            detection_methods.append("ai_metaphorical_patterns")
            scores.append(metaphorical_score)
            method_weights["ai_metaphorical_patterns"] = 0.01
        
        # Detección de patrones analógicos (NUEVO V36)
        analogical_score = self._detect_ai_analogical_patterns(text)
        if analogical_score > 0.2:
            detection_methods.append("ai_analogical_patterns")
            scores.append(analogical_score)
            method_weights["ai_analogical_patterns"] = 0.01
        
        # Análisis de patrones de ironía (NUEVO V36)
        irony_score = self._analyze_ai_irony_patterns(text)
        if irony_score > 0.2:
            detection_methods.append("ai_irony_patterns")
            scores.append(irony_score)
            method_weights["ai_irony_patterns"] = 0.01
        
        # Detección de patrones de humor (NUEVO V36)
        humor_score = self._detect_ai_humor_patterns(text)
        if humor_score > 0.2:
            detection_methods.append("ai_humor_patterns")
            scores.append(humor_score)
            method_weights["ai_humor_patterns"] = 0.01
        
        # Análisis de patrones de sarcasmo (NUEVO V37)
        sarcasm_score = self._analyze_ai_sarcasm_patterns(text)
        if sarcasm_score > 0.2:
            detection_methods.append("ai_sarcasm_patterns")
            scores.append(sarcasm_score)
            method_weights["ai_sarcasm_patterns"] = 0.01
        
        # Detección de patrones de hipérbole (NUEVO V37)
        hyperbole_score = self._detect_ai_hyperbole_patterns(text)
        if hyperbole_score > 0.2:
            detection_methods.append("ai_hyperbole_patterns")
            scores.append(hyperbole_score)
            method_weights["ai_hyperbole_patterns"] = 0.01
        
        # Análisis de patrones de eufemismo (NUEVO V37)
        euphemism_score = self._analyze_ai_euphemism_patterns(text)
        if euphemism_score > 0.2:
            detection_methods.append("ai_euphemism_patterns")
            scores.append(euphemism_score)
            method_weights["ai_euphemism_patterns"] = 0.01
        
        # Detección de patrones de lítote (NUEVO V37)
        understatement_score = self._detect_ai_understatement_patterns(text)
        if understatement_score > 0.2:
            detection_methods.append("ai_understatement_patterns")
            scores.append(understatement_score)
            method_weights["ai_understatement_patterns"] = 0.01
        
        # Análisis de patrones de aliteración (NUEVO V38)
        alliteration_score = self._analyze_ai_alliteration_patterns(text)
        if alliteration_score > 0.2:
            detection_methods.append("ai_alliteration_patterns")
            scores.append(alliteration_score)
            method_weights["ai_alliteration_patterns"] = 0.01
        
        # Detección de patrones de asonancia (NUEVO V38)
        assonance_score = self._detect_ai_assonance_patterns(text)
        if assonance_score > 0.2:
            detection_methods.append("ai_assonance_patterns")
            scores.append(assonance_score)
            method_weights["ai_assonance_patterns"] = 0.01
        
        # Análisis de patrones de ritmo poético (NUEVO V38)
        rhythm_score = self._analyze_ai_rhythm_patterns(text)
        if rhythm_score > 0.2:
            detection_methods.append("ai_rhythm_patterns")
            scores.append(rhythm_score)
            method_weights["ai_rhythm_patterns"] = 0.01
        
        # Detección de patrones poéticos (NUEVO V38)
        poetic_score = self._detect_ai_poetic_patterns(text)
        if poetic_score > 0.2:
            detection_methods.append("ai_poetic_patterns")
            scores.append(poetic_score)
            method_weights["ai_poetic_patterns"] = 0.01
        
        # Análisis de patrones de densidad léxica (NUEVO V39)
        lexical_density_score = self._analyze_ai_lexical_density_patterns(text)
        if lexical_density_score > 0.2:
            detection_methods.append("ai_lexical_density_patterns")
            scores.append(lexical_density_score)
            method_weights["ai_lexical_density_patterns"] = 0.01
        
        # Detección de patrones de redes semánticas (NUEVO V39)
        semantic_network_score = self._detect_ai_semantic_network_patterns(text)
        if semantic_network_score > 0.2:
            detection_methods.append("ai_semantic_network_patterns")
            scores.append(semantic_network_score)
            method_weights["ai_semantic_network_patterns"] = 0.01
        
        # Análisis de patrones de coherencia conceptual (NUEVO V39)
        conceptual_coherence_score = self._analyze_ai_conceptual_coherence_patterns(text)
        if conceptual_coherence_score > 0.2:
            detection_methods.append("ai_conceptual_coherence_patterns")
            scores.append(conceptual_coherence_score)
            method_weights["ai_conceptual_coherence_patterns"] = 0.01
        
        # Detección de patrones de grafos de conocimiento (NUEVO V39)
        knowledge_graph_score = self._detect_ai_knowledge_graph_patterns(text)
        if knowledge_graph_score > 0.2:
            detection_methods.append("ai_knowledge_graph_patterns")
            scores.append(knowledge_graph_score)
            method_weights["ai_knowledge_graph_patterns"] = 0.01
        
        # Calcular porcentaje de IA con pesos ponderados
        if scores:
            # Si hay pesos, usar promedio ponderado
            if method_weights and len(scores) == len(method_weights):
                total_weight = sum(method_weights.values())
                weighted_sum = sum(score * method_weights.get(method, 0.1) 
                                 for score, method in zip(scores, detection_methods))
                avg_score = weighted_sum / total_weight if total_weight > 0 else np.mean(scores)
            else:
                avg_score = np.mean(scores)
            
            ai_percentage = avg_score * 100
            is_ai = ai_percentage > 50.0
            confidence = max(scores) if scores else 0.0
        else:
            ai_percentage = 0.0
            is_ai = False
            confidence = 0.0
        
        # Análisis forense mejorado
        forensic = self._forensic_analysis(text, detected_models, text_stats)
        
        # Modelo principal
        primary_model = None
        if detected_models:
            primary_model = max(detected_models, key=lambda x: x["confidence"])
        
        # Calcular score de riesgo y confiabilidad
        risk_score = self._analyze_risk_and_reliability(text, detected_models, scores) if scores else 0.0
        
        # Información adicional de calidad
        quality_info = {
            "writing_quality": self._analyze_writing_quality(text) if len(text.split()) > 10 else 0.0,
            "paraphrase_likelihood": self._detect_paraphrasing(text) if len(text.split()) > 20 else 0.0,
            "risk_score": risk_score,
            "reliability": "high" if risk_score > 0.7 else "medium" if risk_score > 0.4 else "low"
        }
        
        # Sistema de alertas (NUEVO)
        alerts = self._generate_alerts(ai_percentage, confidence, detected_models, primary_model)
        
        return {
            "is_ai_generated": is_ai,
            "ai_percentage": ai_percentage,
            "detected_models": detected_models,
            "primary_model": primary_model,
            "forensic_analysis": forensic,
            "confidence_score": confidence,
            "detection_methods": detection_methods,
            "quality_info": quality_info,
            "alerts": alerts
        }
    
    def _detect_model_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Detecta patrones específicos de modelos de IA - MEJORADO"""
        detections = []
        text_lower = text.lower()
        
        for model_name, model_info in self.AI_MODEL_PATTERNS.items():
            matches = 0
            total_patterns = len(model_info["patterns"])
            pattern_matches = []
            
            # Contar matches de patrones
            for pattern in model_info["patterns"]:
                pattern_matches_found = len(re.findall(pattern, text_lower, re.IGNORECASE))
                if pattern_matches_found > 0:
                    matches += 1
                    pattern_matches.append(pattern_matches_found)
            
            # Detectar versión si hay version_patterns
            detected_version = None
            if "version_patterns" in model_info:
                for version_pattern in model_info["version_patterns"]:
                    if re.search(version_pattern, text_lower, re.IGNORECASE):
                        detected_version = version_pattern.replace(r"\.", ".").replace(r"\-", "-")
                        break
            
            if matches > 0:
                # Calcular confianza mejorada
                base_confidence = model_info["confidence_base"]
                match_ratio = matches / total_patterns
                
                # Bonus por múltiples matches del mismo patrón
                total_occurrences = sum(pattern_matches) if pattern_matches else matches
                occurrence_bonus = min(0.1 * (total_occurrences - matches), 0.15)
                
                confidence = base_confidence * match_ratio + occurrence_bonus
                confidence = min(confidence, 0.95)
                
                detections.append({
                    "model_name": model_name,
                    "confidence": confidence,
                    "provider": model_info["provider"],
                    "version": detected_version,
                    "matches": matches,
                    "total_occurrences": total_occurrences
                })
        
        return sorted(detections, key=lambda x: x["confidence"], reverse=True)
    
    def _analyze_text_statistics(self, text: str) -> Dict[str, Any]:
        """Analiza estadísticas del texto para detectar IA - MEJORADO"""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) == 0 or len(sentences) == 0:
            return {"ai_likelihood": 0.0}
        
        # Longitud promedio de oraciones
        sentence_lengths = [len(s.split()) for s in sentences]
        avg_sentence_length = np.mean(sentence_lengths)
        
        # Variación en longitud de oraciones (burstiness)
        if len(sentence_lengths) > 1:
            burstiness = np.std(sentence_lengths) / np.mean(sentence_lengths) if np.mean(sentence_lengths) > 0 else 0
        else:
            burstiness = 0
        
        # Repetición de palabras
        word_freq = {}
        for word in words:
            word_lower = word.lower().strip('.,!?;:()[]{}"\'')
            if word_lower:
                word_freq[word_lower] = word_freq.get(word_lower, 0) + 1
        
        max_freq = max(word_freq.values()) if word_freq else 0
        repetition_ratio = max_freq / len(words) if len(words) > 0 else 0
        
        # Diversidad de vocabulario (type-token ratio)
        unique_words = len(word_freq)
        vocab_diversity = unique_words / len(words) if len(words) > 0 else 0
        
        # Análisis de palabras funcionales vs contenido
        functional_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                          'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 
                          'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 
                          'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
        functional_count = sum(1 for w in words if w.lower() in functional_words)
        functional_ratio = functional_count / len(words) if len(words) > 0 else 0
        
        # Calcular likelihood de IA mejorado
        ai_score = 0.0
        
        # 1. Longitud de oraciones (IA suele tener oraciones de longitud similar)
        if 10 <= avg_sentence_length <= 25:
            ai_score += 0.15
        elif 8 <= avg_sentence_length <= 30:
            ai_score += 0.10
        
        # 2. Burstiness (IA tiene menos variación)
        if burstiness < 0.3:
            ai_score += 0.25
        elif burstiness < 0.5:
            ai_score += 0.15
        
        # 3. Repetición (IA evita repetición excesiva)
        if repetition_ratio < 0.2:
            ai_score += 0.15
        elif repetition_ratio < 0.3:
            ai_score += 0.10
        
        # 4. Diversidad de vocabulario (IA tiene buena diversidad)
        if 0.4 <= vocab_diversity <= 0.7:
            ai_score += 0.15
        elif vocab_diversity > 0.7:
            ai_score += 0.05  # Demasiada diversidad puede ser humano
        
        # 5. Ratio de palabras funcionales (IA usa más palabras funcionales)
        if 0.3 <= functional_ratio <= 0.5:
            ai_score += 0.15
        
        # 6. Longitud del texto (textos largos de IA son más coherentes)
        if len(words) > 100:
            ai_score += 0.10
        elif len(words) > 50:
            ai_score += 0.05
        
        # 7. Consistencia en puntuación
        punctuation_consistency = self._check_punctuation_consistency(text)
        if punctuation_consistency > 0.7:
            ai_score += 0.05
        
        # Normalizar a 0-1
        ai_likelihood = min(ai_score, 1.0)
        
        return {
            "ai_likelihood": ai_likelihood,
            "avg_sentence_length": avg_sentence_length,
            "burstiness": burstiness,
            "repetition_ratio": repetition_ratio,
            "vocab_diversity": vocab_diversity,
            "functional_ratio": functional_ratio
        }
    
    def _check_punctuation_consistency(self, text: str) -> float:
        """Verifica consistencia en el uso de puntuación"""
        # Contar diferentes tipos de puntuación
        periods = text.count('.')
        commas = text.count(',')
        exclamations = text.count('!')
        questions = text.count('?')
        semicolons = text.count(';')
        colons = text.count(':')
        
        total_punct = periods + commas + exclamations + questions + semicolons + colons
        if total_punct == 0:
            return 0.5  # Sin puntuación, neutral
        
        # IA suele usar más puntos y comas de forma consistente
        if periods > 0 and commas > 0:
            consistency = 0.8
        elif periods > 0:
            consistency = 0.6
        else:
            consistency = 0.4
        
        return consistency
    
    def _analyze_text_structure(self, text: str) -> float:
        """Analiza la estructura del texto"""
        score = 0.0
        
        # Presencia de estructura organizada
        if re.search(r'\b(first|second|third|finally|in conclusion|to summarize)', text, re.IGNORECASE):
            score += 0.3
        
        # Uso de listas o enumeraciones
        if re.search(r'\d+\.\s|\-\s|\*\s', text):
            score += 0.2
        
        # Párrafos bien formados
        paragraphs = text.split('\n\n')
        if len(paragraphs) > 1:
            avg_para_length = np.mean([len(p.split()) for p in paragraphs if p.strip()])
            if 50 <= avg_para_length <= 200:
                score += 0.3
        
        # Coherencia temática
        if len(text.split()) > 100:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_text_style(self, text: str) -> float:
        """Analiza el estilo del texto"""
        score = 0.0
        
        # Formalidad
        formal_words = ['therefore', 'however', 'furthermore', 'moreover', 'consequently', 'additionally']
        formal_count = sum(1 for word in formal_words if word.lower() in text.lower())
        if formal_count > 0:
            score += 0.3
        
        # Evita contracciones (más formal)
        contractions = ["don't", "won't", "can't", "it's", "that's", "there's"]
        contraction_count = sum(1 for c in contractions if c.lower() in text.lower())
        if contraction_count == 0 and len(text.split()) > 50:
            score += 0.2
        
        # Uso de vocabulario sofisticado
        sophisticated_patterns = [
            r'\b(?:utilize|facilitate|implement|optimize|enhance)\b',
            r'\b(?:comprehensive|systematic|methodical|strategic)\b'
        ]
        sophisticated_count = sum(1 for pattern in sophisticated_patterns if re.search(pattern, text, re.IGNORECASE))
        if sophisticated_count > 0:
            score += 0.3
        
        # Evita errores comunes
        common_errors = ['teh', 'adn', 'taht', 'recieve']
        error_count = sum(1 for error in common_errors if error in text.lower())
        if error_count == 0:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_entropy_and_ngrams(self, text: str) -> float:
        """Analiza entropía y n-gramas para detectar IA - NUEVO"""
        words = text.lower().split()
        if len(words) < 10:
            return 0.0
        
        # Calcular entropía de caracteres
        char_freq = {}
        for char in text.lower():
            if char.isalnum() or char.isspace():
                char_freq[char] = char_freq.get(char, 0) + 1
        
        total_chars = sum(char_freq.values())
        if total_chars == 0:
            return 0.0
        
        entropy = 0.0
        for count in char_freq.values():
            prob = count / total_chars
            if prob > 0:
                entropy -= prob * np.log2(prob)
        
        # Texto de IA suele tener entropía más baja (más predecible)
        # Entropía normal típica: 4-5 bits, IA: 3.5-4.5 bits
        if 3.5 <= entropy <= 4.5:
            entropy_score = 0.4
        elif entropy < 3.5:
            entropy_score = 0.6
        else:
            entropy_score = 0.2
        
        # Análisis de bigramas comunes de IA
        bigrams = {}
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            bigrams[bigram] = bigrams.get(bigram, 0) + 1
        
        # Bigramas muy comunes sugieren texto de IA
        if bigrams:
            max_bigram_freq = max(bigrams.values())
            bigram_diversity = len(bigrams) / len(words) if len(words) > 0 else 0
            
            if bigram_diversity < 0.3:  # Poca diversidad
                bigram_score = 0.3
            else:
                bigram_score = 0.1
        
        return min((entropy_score + bigram_score) / 2, 1.0)
    
    def _analyze_semantic_coherence(self, text: str) -> float:
        """Analiza coherencia semántica del texto - NUEVO"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 3:
            return 0.0
        
        score = 0.0
        
        # 1. Palabras de transición (indican coherencia)
        transition_words = [
            'however', 'therefore', 'furthermore', 'moreover', 'consequently',
            'additionally', 'meanwhile', 'subsequently', 'nevertheless', 'thus',
            'hence', 'accordingly', 'indeed', 'specifically', 'particularly'
        ]
        transition_count = sum(1 for word in transition_words if word.lower() in text.lower())
        if transition_count > 0:
            score += min(transition_count * 0.1, 0.3)
        
        # 2. Referencias pronominales consistentes
        pronouns = ['it', 'this', 'that', 'these', 'those', 'they', 'he', 'she']
        pronoun_count = sum(1 for p in pronouns if re.search(rf'\b{p}\b', text.lower()))
        if pronoun_count > 2:
            score += 0.2
        
        # 3. Repetición de conceptos clave (coherencia temática)
        words = text.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 4:  # Solo palabras significativas
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Conceptos que aparecen múltiples veces
        key_concepts = [w for w, count in word_freq.items() if count > 2]
        if len(key_concepts) > 0:
            score += min(len(key_concepts) * 0.05, 0.3)
        
        # 4. Estructura lógica (si...entonces, porque...por lo tanto)
        logical_patterns = [
            r'\b(?:if|when|because|since|as)\b.*\b(?:then|therefore|thus|so|hence)\b',
            r'\b(?:first|second|third|finally|in conclusion)\b',
            r'\b(?:on one hand|on the other hand)\b'
        ]
        logical_count = sum(1 for pattern in logical_patterns if re.search(pattern, text, re.IGNORECASE))
        if logical_count > 0:
            score += min(logical_count * 0.15, 0.2)
        
        return min(score, 1.0)
    
    def _analyze_syntactic_complexity(self, text: str) -> float:
        """Analiza complejidad sintáctica del texto - NUEVO"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) == 0:
            return 0.0
        
        score = 0.0
        
        # 1. Oraciones compuestas y complejas
        complex_markers = [
            r'\b(?:although|though|while|whereas|despite|in spite of)\b',
            r'\b(?:which|that|who|whom|whose)\b',  # Cláusulas relativas
            r'\b(?:because|since|as|due to|owing to)\b',  # Cláusulas causales
            r'\b(?:if|unless|provided that|as long as)\b'  # Cláusulas condicionales
        ]
        
        complex_sentences = 0
        for sentence in sentences:
            complex_count = sum(1 for marker in complex_markers if re.search(marker, sentence, re.IGNORECASE))
            if complex_count > 0:
                complex_sentences += 1
        
        if len(sentences) > 0:
            complex_ratio = complex_sentences / len(sentences)
            if 0.3 <= complex_ratio <= 0.7:  # Balance ideal
                score += 0.3
            elif complex_ratio > 0.7:
                score += 0.2
        
        # 2. Uso de gerundios y participios
        gerund_patterns = [
            r'\b\w+ing\b',  # Gerundios
            r'\b\w+ed\b',   # Participios pasados
            r'\b\w+en\b'    # Participios irregulares
        ]
        gerund_count = sum(len(re.findall(pattern, text)) for pattern in gerund_patterns)
        if gerund_count > len(sentences) * 0.5:
            score += 0.2
        
        # 3. Uso de preposiciones complejas
        complex_prepositions = [
            'according to', 'in addition to', 'in spite of', 'on behalf of',
            'with regard to', 'in terms of', 'in case of', 'by means of'
        ]
        prep_count = sum(1 for prep in complex_prepositions if prep in text.lower())
        if prep_count > 0:
            score += min(prep_count * 0.1, 0.2)
        
        # 4. Uso de voz pasiva (más compleja)
        passive_patterns = [
            r'\b(?:is|are|was|were|been|being)\s+\w+ed\b',
            r'\b(?:is|are|was|were)\s+\w+en\b'
        ]
        passive_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in passive_patterns)
        if passive_count > 0:
            score += min(passive_count * 0.05, 0.15)
        
        # 5. Longitud promedio de oraciones (oraciones más largas = más complejas)
        avg_length = np.mean([len(s.split()) for s in sentences])
        if 15 <= avg_length <= 25:
            score += 0.15
        
        return min(score, 1.0)
    
    def _analyze_citations_and_references(self, text: str) -> float:
        """Analiza citas y referencias en el texto - NUEVO"""
        score = 0.0
        
        # 1. Detectar citas directas
        direct_quotes = len(re.findall(r'["""].*?["""]', text))
        if direct_quotes > 0:
            score += 0.2
        
        # 2. Detectar referencias académicas
        academic_patterns = [
            r'\([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?,\s*\d{4}\)',  # (Author, 2024)
            r'\[.*?\]',  # [1], [Author 2024]
            r'\d+\.\s*[A-Z][a-z]+.*?\d{4}',  # 1. Author (2024)
            r'\(see\s+.*?\)',  # (see Author, 2024)
            r'according to\s+[A-Z][a-z]+',  # according to Author
            r'as\s+[A-Z][a-z]+\s+notes?',  # as Author notes
            r'[A-Z][a-z]+\s+\(\d{4}\)'  # Author (2024)
        ]
        
        citation_count = sum(len(re.findall(pattern, text)) for pattern in academic_patterns)
        if citation_count > 0:
            score += min(citation_count * 0.15, 0.4)
        
        # 3. Detectar números de página o secciones
        page_refs = len(re.findall(r'\b(?:p\.|pp\.|page|pages)\s+\d+', text, re.IGNORECASE))
        if page_refs > 0:
            score += 0.2
        
        # 4. Detectar URLs o enlaces
        url_pattern = r'https?://\S+|www\.\S+'
        urls = len(re.findall(url_pattern, text))
        if urls > 0:
            score += min(urls * 0.1, 0.2)
        
        # 5. Detectar notas al pie
        footnote_patterns = [
            r'\d+\s*\(footnote|note\)',
            r'\[note\s+\d+\]',
            r'\(see note \d+\)'
        ]
        footnotes = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in footnote_patterns)
        if footnotes > 0:
            score += 0.1
        
        # Texto con muchas citas suele ser más humano (académico)
        # Pero texto de IA puede imitar citas, así que score moderado
        return min(score, 1.0)
    
    def _analyze_temporal_consistency(self, text: str) -> float:
        """Analiza consistencia temporal del texto - NUEVO"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 3:
            return 0.0
        
        score = 0.0
        
        # 1. Análisis de cambios de estilo a lo largo del texto
        # Dividir texto en tercios
        third = len(sentences) // 3
        first_third = sentences[:third] if third > 0 else sentences[:1]
        middle_third = sentences[third:2*third] if third > 0 else sentences[1:2]
        last_third = sentences[2*third:] if third > 0 else sentences[2:]
        
        # Calcular longitud promedio de oraciones en cada tercio
        if first_third and middle_third and last_third:
            avg_first = np.mean([len(s.split()) for s in first_third])
            avg_middle = np.mean([len(s.split()) for s in middle_third])
            avg_last = np.mean([len(s.split()) for s in last_third])
            
            # Texto de IA suele tener consistencia (poca variación)
            variation = np.std([avg_first, avg_middle, avg_last])
            if variation < 3.0:  # Poca variación = más consistente = posible IA
                score += 0.3
        
        # 2. Análisis de vocabulario a lo largo del texto
        first_words = set(word.lower() for s in first_third for word in s.split() if len(word) > 4)
        last_words = set(word.lower() for s in last_third for word in s.split() if len(word) > 4)
        
        if first_words and last_words:
            overlap = len(first_words & last_words) / len(first_words | last_words) if (first_words | last_words) else 0
            if overlap > 0.3:  # Alto solapamiento = consistencia temática
                score += 0.2
        
        # 3. Detección de cambios abruptos de tono
        formal_words = ['therefore', 'however', 'furthermore', 'moreover', 'consequently']
        informal_words = ["don't", "won't", "can't", "it's", "that's", "gonna", "wanna"]
        
        formal_count_first = sum(1 for word in formal_words if any(word in s.lower() for s in first_third))
        informal_count_first = sum(1 for word in informal_words if any(word in s.lower() for s in first_third))
        formal_count_last = sum(1 for word in formal_words if any(word in s.lower() for s in last_third))
        informal_count_last = sum(1 for word in informal_words if any(word in s.lower() for s in last_third))
        
        # Cambios abruptos sugieren edición humana
        if abs(formal_count_first - formal_count_last) > 2 or abs(informal_count_first - informal_count_last) > 2:
            score -= 0.2  # Penalizar cambios abruptos (más humano)
        
        return max(min(score, 1.0), 0.0)
    
    def _detect_watermarks(self, text: str) -> float:
        """Detecta posibles watermarks en el texto - NUEVO"""
        score = 0.0
        
        # 1. Patrones de watermark conocidos
        watermark_patterns = [
            r'\b(?:generated by|created by|produced by)\s+(?:ai|artificial intelligence|chatgpt|gpt|claude)',
            r'\b(?:this (?:text|content|document) (?:was|is) (?:generated|created|produced))',
            r'\[ai generated\]|\[generated by ai\]|\[ai content\]',
            r'<!--.*?ai.*?-->',  # Comentarios HTML
            r'\/\*.*?ai.*?\*\/',  # Comentarios de código
        ]
        
        watermark_matches = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in watermark_patterns)
        if watermark_matches > 0:
            score += 0.5
        
        # 2. Patrones de caracteres especiales sospechosos
        # Algunos watermarks usan caracteres Unicode especiales
        special_chars = ['\u200b', '\u200c', '\u200d', '\u2060', '\u2061', '\u2062', '\u2063']
        special_count = sum(text.count(char) for char in special_chars)
        if special_count > 0:
            score += min(special_count * 0.1, 0.3)
        
        # 3. Patrones de espaciado inusual
        # Espacios múltiples o patrones repetitivos
        double_spaces = len(re.findall(r'  +', text))
        if double_spaces > len(text.split()) * 0.1:  # Más del 10% de espacios dobles
            score += 0.1
        
        # 4. Patrones de hash o identificadores ocultos
        hash_patterns = [
            r'\b[a-f0-9]{8,}\b',  # Hashes hexadecimales
            r'[A-Z0-9]{10,}',  # Identificadores alfanuméricos largos
        ]
        hash_matches = sum(len(re.findall(pattern, text)) for pattern in hash_patterns)
        if hash_matches > 0:
            score += min(hash_matches * 0.05, 0.1)
        
        return min(score, 1.0)
    
    def _detect_edits_and_patches(self, text: str) -> float:
        """Detecta ediciones y parches en el texto - NUEVO"""
        score = 0.0
        
        # 1. Detectar correcciones o ediciones explícitas
        edit_markers = [
            r'\[edit\]|\[edited\]|\[correction\]|\[updated\]',
            r'\(edit:.*?\)|\(edited:.*?\)|\(correction:.*?\)',
            r'note:.*?edit|note:.*?correction',
        ]
        
        edit_matches = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in edit_markers)
        if edit_matches > 0:
            score += 0.4
        
        # 2. Detectar cambios de formato abruptos
        # Mezcla de mayúsculas/minúsculas inconsistentes
        words = text.split()
        if len(words) > 10:
            all_caps = sum(1 for w in words if w.isupper() and len(w) > 1)
            all_lower = sum(1 for w in words if w.islower() and len(w) > 1)
            mixed_case = len(words) - all_caps - all_lower
            
            # Mezcla excesiva sugiere edición
            if mixed_case > len(words) * 0.3:
                score += 0.2
        
        # 3. Detectar paréntesis o corchetes de edición
        # Muchos paréntesis pueden indicar aclaraciones añadidas
        paren_ratio = (text.count('(') + text.count('[')) / len(text.split()) if len(text.split()) > 0 else 0
        if paren_ratio > 0.1:  # Más del 10% de palabras tienen paréntesis
            score += 0.2
        
        # 4. Detectar cambios de estilo dentro del texto
        # Texto que empieza formal y termina informal (o viceversa)
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) > 5:
            first_half = ' '.join(sentences[:len(sentences)//2])
            second_half = ' '.join(sentences[len(sentences)//2:])
            
            # Contar palabras formales vs informales
            formal_words = ['therefore', 'however', 'furthermore', 'moreover']
            informal_words = ["don't", "won't", "can't", "it's", "that's"]
            
            formal_first = sum(1 for w in formal_words if w in first_half.lower())
            informal_first = sum(1 for w in informal_words if w in first_half.lower())
            formal_second = sum(1 for w in formal_words if w in second_half.lower())
            informal_second = sum(1 for w in informal_words if w in second_half.lower())
            
            # Cambio significativo de estilo
            if abs(formal_first - formal_second) > 2 or abs(informal_first - informal_second) > 2:
                score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_sentiment_patterns(self, text: str) -> float:
        """Analiza patrones de sentimientos en el texto - NUEVO"""
        score = 0.0
        
        # 1. Detectar emociones típicas de IA
        # Texto de IA suele ser más neutral y menos emocional
        emotional_words = [
            'love', 'hate', 'angry', 'sad', 'happy', 'excited', 'frustrated',
            'amazing', 'terrible', 'awesome', 'awful', 'fantastic', 'horrible'
        ]
        emotional_count = sum(1 for word in emotional_words if word.lower() in text.lower())
        
        # Texto con muy pocas emociones puede ser IA
        word_count = len(text.split())
        if word_count > 50:
            emotional_ratio = emotional_count / word_count
            if emotional_ratio < 0.01:  # Menos del 1% de palabras emocionales
                score += 0.3
            elif emotional_ratio < 0.02:
                score += 0.2
        
        # 2. Detectar uso de emojis o expresiones
        # Texto de IA suele evitar emojis
        emoji_pattern = r'[😀-🙏🌀-🗿]'
        emoji_count = len(re.findall(emoji_pattern, text))
        if emoji_count == 0 and word_count > 50:
            score += 0.1
        
        # 3. Análisis de polaridad
        # Texto de IA suele ser más balanceado
        positive_words = ['good', 'great', 'excellent', 'wonderful', 'fantastic', 'amazing', 'best']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'poor', 'worst']
        
        positive_count = sum(1 for word in positive_words if word.lower() in text.lower())
        negative_count = sum(1 for word in negative_words if word.lower() in text.lower())
        
        # Balance entre positivo y negativo
        if positive_count > 0 and negative_count > 0:
            balance = min(positive_count, negative_count) / max(positive_count, negative_count)
            if balance > 0.5:  # Balanceado
                score += 0.2
        elif positive_count == 0 and negative_count == 0:
            score += 0.2  # Muy neutral
        
        # 4. Detectar expresiones de incertidumbre
        # IA a veces usa expresiones de incertidumbre
        uncertainty_words = [
            'perhaps', 'maybe', 'possibly', 'might', 'could', 'may',
            'uncertain', 'unclear', 'possibly', 'potentially'
        ]
        uncertainty_count = sum(1 for word in uncertainty_words if word.lower() in text.lower())
        if uncertainty_count > word_count * 0.02:  # Más del 2%
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_contextual_coherence(self, text: str) -> float:
        """Analiza coherencia contextual y temática avanzada - NUEVO"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 3:
            return 0.0
        
        score = 0.0
        
        # 1. Análisis de coherencia temática
        # Extraer palabras clave de cada oración
        all_words = []
        for sentence in sentences:
            words = [w.lower() for w in sentence.split() if len(w) > 4 and w.isalpha()]
            all_words.extend(words)
        
        # Contar frecuencia de palabras clave
        word_freq = {}
        for word in all_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Palabras que aparecen múltiples veces (temas centrales)
        key_themes = [w for w, count in word_freq.items() if count > 2]
        if len(key_themes) > 0:
            # Alta coherencia temática
            theme_coherence = len(key_themes) / len(set(all_words)) if len(set(all_words)) > 0 else 0
            if theme_coherence > 0.2:
                score += 0.3
        
        # 2. Análisis de progresión lógica
        # Detectar si el texto sigue una progresión lógica
        progression_markers = [
            r'\b(?:first|initially|to begin|starting with)\b',
            r'\b(?:second|next|then|following|subsequently)\b',
            r'\b(?:third|furthermore|additionally|moreover)\b',
            r'\b(?:finally|lastly|in conclusion|to conclude|ultimately)\b'
        ]
        
        progression_count = 0
        for i, sentence in enumerate(sentences):
            for pattern in progression_markers:
                if re.search(pattern, sentence, re.IGNORECASE):
                    progression_count += 1
                    break
        
        if progression_count >= 2:
            score += 0.25
        
        # 3. Análisis de referencias cruzadas
        # Detectar referencias a conceptos mencionados anteriormente
        reference_words = ['this', 'that', 'these', 'those', 'it', 'they', 'he', 'she']
        reference_count = sum(1 for word in reference_words if word.lower() in text.lower())
        
        if reference_count > len(sentences) * 0.3:  # Más del 30% de oraciones tienen referencias
            score += 0.2
        
        # 4. Análisis de coherencia semántica entre párrafos
        paragraphs = text.split('\n\n')
        if len(paragraphs) > 1:
            para_words = []
            for para in paragraphs:
                words = [w.lower() for w in para.split() if len(w) > 4 and w.isalpha()]
                para_words.append(set(words))
            
            # Calcular solapamiento entre párrafos
            overlaps = []
            for i in range(len(para_words) - 1):
                if para_words[i] and para_words[i+1]:
                    overlap = len(para_words[i] & para_words[i+1]) / len(para_words[i] | para_words[i+1])
                    overlaps.append(overlap)
            
            if overlaps:
                avg_overlap = np.mean(overlaps)
                if avg_overlap > 0.15:  # Solapamiento significativo
                    score += 0.25
        
        return min(score, 1.0)
    
    def _detect_machine_translation(self, text: str) -> float:
        """Detecta si el texto fue traducido automáticamente - NUEVO"""
        score = 0.0
        
        # 1. Patrones típicos de traducción automática
        translation_patterns = [
            r'\b(?:the the|a a|an an)\b',  # Repetición de artículos
            r'\b(?:is is|are are|was was|were were)\b',  # Repetición de verbos
            r'\b(?:of of|in in|on on|at at)\b',  # Repetición de preposiciones
            r'\b(?:and and|or or|but but)\b',  # Repetición de conjunciones
        ]
        
        translation_matches = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in translation_patterns)
        if translation_matches > 0:
            score += 0.4
        
        # 2. Orden de palabras inusual (típico de traducción)
        # Frases que suenan "traducidas"
        unusual_patterns = [
            r'\b(?:very much|so much|too much)\s+\w+',  # Construcciones traducidas
            r'\b(?:the same|the different|the new|the old)\s+\w+',  # Artículos innecesarios
        ]
        
        unusual_matches = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in unusual_patterns)
        if unusual_matches > 2:
            score += 0.3
        
        # 3. Uso excesivo de palabras de traducción literal
        literal_words = ['very', 'much', 'quite', 'rather', 'quite', 'indeed']
        literal_count = sum(1 for word in literal_words if word.lower() in text.lower())
        word_count = len(text.split())
        
        if word_count > 0:
            literal_ratio = literal_count / word_count
            if literal_ratio > 0.05:  # Más del 5%
                score += 0.2
        
        # 4. Falta de modismos o expresiones naturales
        # Texto traducido suele carecer de modismos
        idioms = [
            'break the ice', 'piece of cake', 'once in a blue moon',
            'the ball is in your court', 'barking up the wrong tree'
        ]
        idiom_count = sum(1 for idiom in idioms if idiom.lower() in text.lower())
        
        if idiom_count == 0 and word_count > 100:
            score += 0.1  # Texto largo sin modismos puede ser traducido
        
        return min(score, 1.0)
    
    def _analyze_generation_patterns(self, text: str) -> float:
        """Analiza patrones específicos de generación de IA - NUEVO"""
        score = 0.0
        
        # 1. Patrones de inicio típicos de IA
        start_patterns = [
            r'^(?:let me|i\'ll|i will|i can|i\'d like to|i would like to)',
            r'^(?:here\'s|here are|below is|following is|as follows)',
            r'^(?:to answer|to respond|to help|to assist)',
            r'^(?:based on|according to|in terms of|with regard to)',
        ]
        
        first_sentence = text.split('.')[0] if '.' in text else text[:100]
        start_matches = sum(1 for pattern in start_patterns if re.search(pattern, first_sentence, re.IGNORECASE))
        if start_matches > 0:
            score += 0.3
        
        # 2. Patrones de estructura repetitiva
        # IA suele usar estructuras similares
        structure_patterns = [
            r'\b(?:first|second|third|fourth|fifth)\b.*?\b(?:then|next|after|following)\b',
            r'\b(?:one|two|three)\s+(?:way|method|approach|solution)\b',
            r'\b(?:on one hand|on the other hand)\b',
            r'\b(?:in addition|furthermore|moreover|additionally)\b',
        ]
        
        structure_matches = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in structure_patterns)
        if structure_matches > 1:
            score += 0.25
        
        # 3. Uso de frases de transición excesivo
        transition_phrases = [
            'in conclusion', 'to summarize', 'in summary', 'to conclude',
            'in other words', 'that is to say', 'put another way',
            'for example', 'for instance', 'such as', 'namely'
        ]
        
        transition_count = sum(1 for phrase in transition_phrases if phrase.lower() in text.lower())
        word_count = len(text.split())
        
        if word_count > 0:
            transition_ratio = transition_count / (word_count / 100)  # Por cada 100 palabras
            if transition_ratio > 2:  # Más de 2 transiciones por 100 palabras
                score += 0.2
        
        # 4. Patrones de cierre típicos de IA
        end_patterns = [
            r'\b(?:i hope this helps|i hope this|hope this helps)\b',
            r'\b(?:let me know if|feel free to|if you have any)\b',
            r'\b(?:please let me know|don\'t hesitate to|if you need)\b',
            r'\b(?:in conclusion|to summarize|in summary|to conclude)\b',
        ]
        
        last_sentences = ' '.join(text.split('.')[-3:]) if '.' in text else text[-200:]
        end_matches = sum(1 for pattern in end_patterns if re.search(pattern, last_sentences, re.IGNORECASE))
        if end_matches > 0:
            score += 0.25
        
        return min(score, 1.0)
    
    def _analyze_writing_quality(self, text: str) -> float:
        """Analiza la calidad de escritura del texto - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) == 0 or len(sentences) == 0:
            return 0.0
        
        # 1. Análisis de errores gramaticales comunes
        # Texto de IA suele tener menos errores gramaticales
        common_errors = [
            'teh', 'adn', 'taht', 'recieve', 'seperate', 'occured',
            'definately', 'accomodate', 'begining', 'existance'
        ]
        error_count = sum(1 for error in common_errors if error in text.lower())
        
        if error_count == 0:
            score += 0.3  # Sin errores comunes
        elif error_count <= 2:
            score += 0.1  # Pocos errores
        
        # 2. Análisis de puntuación correcta
        # Texto de IA suele tener puntuación más consistente
        punctuation_errors = [
            r'\s+[.!?]',  # Espacios antes de puntuación
            r'[.!?]\s*[a-z]',  # Falta de mayúscula después de punto
            r',\s*,',  # Comas dobles
            r'\.\s*\.',  # Puntos dobles
        ]
        
        punct_errors = sum(len(re.findall(pattern, text)) for pattern in punctuation_errors)
        if punct_errors == 0:
            score += 0.2
        
        # 3. Análisis de vocabulario variado
        unique_words = len(set(word.lower() for word in words))
        vocab_ratio = unique_words / len(words) if len(words) > 0 else 0
        
        if 0.5 <= vocab_ratio <= 0.8:  # Balance ideal
            score += 0.2
        elif vocab_ratio > 0.8:
            score += 0.1  # Demasiada variedad puede ser sospechoso
        
        # 4. Análisis de longitud de palabras
        # Texto de IA puede tener palabras más largas en promedio
        avg_word_length = np.mean([len(word) for word in words if word.isalpha()])
        if 4.5 <= avg_word_length <= 6.5:
            score += 0.15
        
        # 5. Análisis de estructura de párrafos
        paragraphs = text.split('\n\n')
        if len(paragraphs) > 1:
            para_lengths = [len(p.split()) for p in paragraphs if p.strip()]
            if para_lengths:
                avg_para_length = np.mean(para_lengths)
                if 50 <= avg_para_length <= 200:  # Párrafos bien formados
                    score += 0.15
        
        return min(score, 1.0)
    
    def _detect_paraphrasing(self, text: str) -> float:
        """Detecta si el texto fue parafraseado - NUEVO"""
        score = 0.0
        
        # 1. Detectar sinónimos excesivos
        # Texto parafraseado suele usar muchos sinónimos
        synonym_pairs = [
            ('big', 'large'), ('small', 'little'), ('good', 'great'),
            ('bad', 'terrible'), ('important', 'significant'), ('help', 'assist'),
            ('use', 'utilize'), ('show', 'demonstrate'), ('tell', 'inform'),
            ('get', 'obtain'), ('make', 'create'), ('do', 'perform')
        ]
        
        synonym_count = 0
        for word1, word2 in synonym_pairs:
            if word1 in text.lower() and word2 in text.lower():
                synonym_count += 1
        
        if synonym_count > 3:
            score += 0.3
        
        # 2. Detectar cambios de estructura sin cambios de significado
        # Frases que dicen lo mismo de forma diferente
        similar_phrases = [
            (r'\b(?:in order to|to)\b', r'\b(?:so that|so as to)\b'),
            (r'\b(?:because|since)\b', r'\b(?:due to|owing to)\b'),
            (r'\b(?:although|though)\b', r'\b(?:even though|despite)\b'),
        ]
        
        phrase_variations = 0
        for pattern1, pattern2 in similar_phrases:
            if re.search(pattern1, text, re.IGNORECASE) and re.search(pattern2, text, re.IGNORECASE):
                phrase_variations += 1
        
        if phrase_variations > 1:
            score += 0.25
        
        # 3. Detectar uso de palabras más formales de lo necesario
        # Parafraseo suele usar vocabulario más formal
        formal_replacements = [
            ('use', 'utilize'), ('help', 'assist'), ('show', 'demonstrate'),
            ('tell', 'inform'), ('get', 'obtain'), ('make', 'create'),
            ('start', 'commence'), ('end', 'conclude'), ('try', 'attempt')
        ]
        
        formal_count = sum(1 for informal, formal in formal_replacements 
                          if formal in text.lower() and informal not in text.lower())
        
        if formal_count > 2:
            score += 0.25
        
        # 4. Detectar cambios de voz (activa a pasiva o viceversa)
        # Parafraseo a veces cambia la voz
        active_patterns = [
            r'\b(?:we|they|he|she|it)\s+\w+ed\b',
            r'\b(?:we|they|he|she|it)\s+\w+s\b'
        ]
        passive_patterns = [
            r'\b(?:is|are|was|were)\s+\w+ed\s+by\b',
            r'\b(?:is|are|was|were)\s+\w+en\s+by\b'
        ]
        
        active_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in active_patterns)
        passive_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in passive_patterns)
        
        if active_count > 0 and passive_count > 0:
            score += 0.2  # Mezcla de voces puede indicar parafraseo
        
        return min(score, 1.0)
    
    def _analyze_risk_and_reliability(self, text: str, detected_models: List[Dict], scores: List[float]) -> float:
        """Analiza el riesgo y confiabilidad de la detección - NUEVO"""
        score = 0.0
        
        # 1. Análisis de confianza basado en métodos de detección
        if len(scores) > 5:  # Múltiples métodos coinciden
            score += 0.2
        elif len(scores) > 3:
            score += 0.1
        
        # 2. Análisis de consistencia entre métodos
        if scores:
            score_std = np.std(scores)
            if score_std < 0.2:  # Baja desviación = alta consistencia
                score += 0.3
            elif score_std < 0.3:
                score += 0.15
        
        # 3. Análisis de modelos detectados
        if len(detected_models) > 1:  # Múltiples modelos detectados
            score += 0.2
        elif len(detected_models) == 1:
            model_confidence = detected_models[0].get("confidence", 0)
            if model_confidence > 0.8:
                score += 0.3
            elif model_confidence > 0.6:
                score += 0.15
        
        # 4. Análisis de longitud del texto
        # Textos más largos suelen dar resultados más confiables
        word_count = len(text.split())
        if word_count > 500:
            score += 0.15
        elif word_count > 200:
            score += 0.1
        elif word_count < 50:
            score -= 0.1  # Textos muy cortos son menos confiables
        
        # 5. Análisis de calidad del texto
        # Texto de baja calidad puede dar falsos positivos
        quality_indicators = [
            len(re.findall(r'\b(?:the|a|an|and|or|but|in|on|at|to|for|of|with|by)\b', text.lower())),
            len(re.findall(r'[.!?]', text)),
            len(re.findall(r'[A-Z][a-z]+', text))
        ]
        
        if all(ind > 0 for ind in quality_indicators):
            score += 0.1  # Texto tiene estructura básica
        
        return max(min(score, 1.0), 0.0)
    
    def _analyze_metadata_and_context(self, text: str, metadata: Dict[str, Any]) -> float:
        """Analiza metadatos y contexto para detectar IA - NUEVO"""
        score = 0.0
        
        # 1. Análisis de fuente
        source = metadata.get("source", "").lower()
        ai_sources = ["chatgpt", "gpt", "claude", "gemini", "ai", "openai", "anthropic"]
        if any(ai_source in source for ai_source in ai_sources):
            score += 0.4
        
        # 2. Análisis de timestamp
        # Contenido generado recientemente puede ser más probable que sea IA
        timestamp = metadata.get("timestamp")
        if timestamp:
            import datetime
            try:
                if isinstance(timestamp, (int, float)):
                    content_time = datetime.datetime.fromtimestamp(timestamp)
                else:
                    content_time = datetime.datetime.fromisoformat(str(timestamp))
                
                now = datetime.datetime.now()
                age_hours = (now - content_time).total_seconds() / 3600
                
                # Contenido muy reciente puede ser más sospechoso
                if age_hours < 24:  # Menos de 24 horas
                    score += 0.1
            except:
                pass
        
        # 3. Análisis de user agent o aplicación
        user_agent = metadata.get("user_agent", "").lower()
        app_name = metadata.get("app_name", "").lower()
        
        ai_apps = ["chatgpt", "claude", "bard", "copilot", "ai", "assistant"]
        if any(ai_app in user_agent or ai_app in app_name for ai_app in ai_apps):
            score += 0.3
        
        # 4. Análisis de idioma en metadatos vs texto
        metadata_lang = metadata.get("language", "").lower()
        # Si hay discrepancia puede indicar traducción automática
        if metadata_lang and metadata_lang not in ["en", "english"]:
            # Texto en inglés pero metadata en otro idioma puede indicar traducción
            score += 0.1
        
        # 5. Análisis de referrer o origen
        referrer = metadata.get("referrer", "").lower()
        origin = metadata.get("origin", "").lower()
        
        ai_origins = ["openai", "anthropic", "google ai", "chatgpt", "claude"]
        if any(ai_origin in referrer or ai_origin in origin for ai_origin in ai_origins):
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_language_patterns(self, text: str) -> float:
        """Analiza patrones de idioma y localización - NUEVO"""
        score = 0.0
        
        # 1. Detectar mezcla de idiomas (típico de traducción automática)
        # Patrones de palabras en diferentes idiomas
        english_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with']
        spanish_words = ['el', 'la', 'los', 'las', 'de', 'en', 'y', 'o', 'pero', 'con', 'por']
        french_words = ['le', 'la', 'les', 'de', 'en', 'et', 'ou', 'mais', 'avec', 'pour']
        
        english_count = sum(1 for word in english_words if word.lower() in text.lower())
        spanish_count = sum(1 for word in spanish_words if word.lower() in text.lower())
        french_count = sum(1 for word in french_words if word.lower() in text.lower())
        
        # Mezcla de idiomas puede indicar traducción automática
        languages_detected = sum([
            english_count > 5,
            spanish_count > 5,
            french_count > 5
        ])
        
        if languages_detected > 1:
            score += 0.3
        
        # 2. Análisis de caracteres especiales por idioma
        # Texto de IA traducido puede tener caracteres especiales mal usados
        special_chars = ['á', 'é', 'í', 'ó', 'ú', 'ñ', 'ü', 'ç', 'à', 'è', 'ì', 'ò', 'ù']
        special_count = sum(1 for char in special_chars if char in text.lower())
        
        # Muchos caracteres especiales pueden indicar traducción
        if special_count > len(text) * 0.05:  # Más del 5%
            score += 0.2
        
        # 3. Análisis de orden de palabras (SVO vs otros órdenes)
        # Texto traducido puede tener orden de palabras inusual
        # Detectar patrones de orden inusual
        unusual_patterns = [
            r'\b(?:the|a|an)\s+\w+\s+(?:the|a|an)\b',  # Artículos duplicados
            r'\b\w+\s+is\s+is\b',  # "is is"
            r'\b\w+\s+the\s+the\b',  # "the the"
        ]
        
        unusual_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in unusual_patterns)
        if unusual_count > 0:
            score += 0.3
        
        # 4. Análisis de expresiones idiomáticas
        # Texto traducido suele carecer de expresiones idiomáticas naturales
        idioms = [
            'break the ice', 'piece of cake', 'once in a blue moon',
            'the ball is in your court', 'barking up the wrong tree',
            'cost an arm and a leg', 'hit the nail on the head'
        ]
        
        idiom_count = sum(1 for idiom in idioms if idiom.lower() in text.lower())
        word_count = len(text.split())
        
        # Texto largo sin modismos puede ser traducido
        if idiom_count == 0 and word_count > 200:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_semantic_similarity(self, text: str) -> float:
        """Analiza similitud semántica usando técnicas básicas - NUEVO"""
        score = 0.0
        
        # Dividir texto en oraciones
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.split()) > 3]
        
        if len(sentences) < 2:
            return 0.0
        
        # 1. Análisis de similitud entre oraciones (Jaccard)
        similarities = []
        for i in range(len(sentences) - 1):
            words1 = set(re.findall(r'\b\w+\b', sentences[i].lower()))
            words2 = set(re.findall(r'\b\w+\b', sentences[i+1].lower()))
            
            if len(words1) > 0 and len(words2) > 0:
                intersection = len(words1.intersection(words2))
                union = len(words1.union(words2))
                similarity = intersection / union if union > 0 else 0
                similarities.append(similarity)
        
        if similarities:
            avg_similarity = np.mean(similarities)
            # Alta similitud entre oraciones puede indicar IA (texto muy coherente)
            if avg_similarity > 0.3:
                score += 0.3
        
        # 2. Análisis de repetición de conceptos clave
        words = re.findall(r'\b\w+\b', text.lower())
        word_freq = {}
        for word in words:
            if len(word) > 4:  # Solo palabras significativas
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Conceptos muy repetidos pueden indicar IA
        if word_freq:
            max_freq = max(word_freq.values())
            total_words = len(words)
            if max_freq / total_words > 0.05:  # Más del 5% de repetición
                score += 0.2
        
        # 3. Análisis de coherencia temática (palabras relacionadas)
        # Detectar si hay un tema dominante
        significant_words = [w for w, f in word_freq.items() if f > 2 and len(w) > 4]
        if len(significant_words) < len(word_freq) * 0.3:  # Pocas palabras dominantes
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_keyword_frequency(self, text: str) -> float:
        """Analiza frecuencia de palabras clave típicas de IA - NUEVO"""
        score = 0.0
        text_lower = text.lower()
        
        # Palabras clave típicas de respuestas de IA
        ai_keywords = [
            'however', 'furthermore', 'moreover', 'additionally', 'consequently',
            'therefore', 'thus', 'hence', 'accordingly', 'nevertheless',
            'in conclusion', 'to summarize', 'in summary', 'overall',
            'it is important to note', 'it should be noted', 'it is worth mentioning',
            'as a result', 'in other words', 'for instance', 'for example',
            'specifically', 'particularly', 'especially', 'notably'
        ]
        
        # Contar ocurrencias de palabras clave
        keyword_count = sum(1 for keyword in ai_keywords if keyword in text_lower)
        word_count = len(text.split())
        
        # Ratio de palabras clave
        if word_count > 0:
            keyword_ratio = keyword_count / (word_count / 100)  # Por cada 100 palabras
            
            # Alto ratio indica posible IA
            if keyword_ratio > 2:  # Más de 2 palabras clave por cada 100 palabras
                score += 0.4
            elif keyword_ratio > 1:
                score += 0.2
        
        # Frases típicas de IA
        ai_phrases = [
            r'\bit is (?:important|worth noting|essential|crucial)',
            r'\b(?:in|to) (?:conclusion|summary|conclude)',
            r'\b(?:as|it) (?:can|may) be (?:seen|observed|noted)',
            r'\b(?:this|these) (?:suggests?|indicates?|demonstrates?)',
            r'\b(?:it|this) (?:is|should be) (?:noted|mentioned|understood)',
            r'\b(?:in|for) (?:order|addition) to',
            r'\b(?:on|with) (?:regard|respect) to'
        ]
        
        phrase_count = sum(len(re.findall(phrase, text_lower)) for phrase in ai_phrases)
        if phrase_count > 2:
            score += 0.3
        elif phrase_count > 0:
            score += 0.15
        
        return min(score, 1.0)
    
    def _detect_response_patterns(self, text: str) -> float:
        """Detecta patrones típicos de respuestas de IA - NUEVO"""
        score = 0.0
        text_lower = text.lower()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # 1. Patrones de inicio típicos de IA
        ai_start_patterns = [
            r'^(?:i|we|this|these|it|they) (?:would|will|can|may|might)',
            r'^(?:to|in order to)',
            r'^(?:based on|according to|in accordance with)',
            r'^(?:it is|this is|these are)',
            r'^(?:let me|allow me)'
        ]
        
        start_matches = sum(1 for pattern in ai_start_patterns 
                          if re.search(pattern, sentences[0].lower() if sentences else "", re.IGNORECASE))
        if start_matches > 0:
            score += 0.2
        
        # 2. Estructura de respuesta típica (introducción, desarrollo, conclusión)
        # Detectar si tiene estructura muy organizada
        has_intro = any(word in text_lower[:200] for word in ['introduction', 'overview', 'summary', 'begin', 'start'])
        has_conclusion = any(word in text_lower[-200:] for word in ['conclusion', 'summary', 'conclude', 'finally', 'overall'])
        
        if has_intro and has_conclusion:
            score += 0.3
        
        # 3. Uso excesivo de conectores lógicos
        logical_connectors = ['however', 'therefore', 'thus', 'hence', 'consequently', 
                            'furthermore', 'moreover', 'additionally', 'nevertheless']
        connector_count = sum(1 for connector in logical_connectors if connector in text_lower)
        
        if connector_count > len(sentences) * 0.2:  # Más del 20% de las oraciones
            score += 0.3
        
        # 4. Formato de lista numerada o con viñetas (típico de IA)
        list_patterns = [
            r'\d+\.\s+[A-Z]',  # Numeración
            r'[•\-\*]\s+[A-Z]',  # Viñetas
            r'(?:first|second|third|fourth|fifth|finally|lastly)',
        ]
        
        list_count = sum(len(re.findall(pattern, text)) for pattern in list_patterns)
        if list_count > 2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_narrative_coherence(self, text: str) -> float:
        """Analiza coherencia narrativa del texto - NUEVO"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.split()) > 3]
        
        if len(sentences) < 3:
            return 0.0
        
        # 1. Análisis de referencias pronominales
        # Texto de IA suele tener referencias pronominales muy claras
        pronouns = ['it', 'this', 'that', 'these', 'those', 'he', 'she', 'they', 'we', 'you']
        pronoun_count = sum(len(re.findall(rf'\b{pronoun}\b', text.lower())) for pronoun in pronouns)
        word_count = len(text.split())
        
        if word_count > 0:
            pronoun_ratio = pronoun_count / word_count
            # Ratio muy alto o muy bajo puede indicar IA
            if pronoun_ratio > 0.08 or pronoun_ratio < 0.02:
                score += 0.2
        
        # 2. Análisis de progresión temática
        # Detectar si el tema se mantiene constante
        first_half_words = set(re.findall(r'\b\w{4,}\b', ' '.join(sentences[:len(sentences)//2]).lower()))
        second_half_words = set(re.findall(r'\b\w{4,}\b', ' '.join(sentences[len(sentences)//2:]).lower()))
        
        if len(first_half_words) > 0 and len(second_half_words) > 0:
            overlap = len(first_half_words.intersection(second_half_words))
            total_unique = len(first_half_words.union(second_half_words))
            
            if total_unique > 0:
                overlap_ratio = overlap / total_unique
                # Alta superposición temática puede indicar IA
                if overlap_ratio > 0.4:
                    score += 0.3
        
        # 3. Análisis de variación en longitud de oraciones
        sentence_lengths = [len(s.split()) for s in sentences]
        if len(sentence_lengths) > 1:
            length_variance = np.var(sentence_lengths)
            avg_length = np.mean(sentence_lengths)
            
            # Baja variación puede indicar IA (oraciones muy uniformes)
            if avg_length > 0:
                cv = np.sqrt(length_variance) / avg_length  # Coeficiente de variación
                if cv < 0.3:  # Baja variación
                    score += 0.2
        
        # 4. Análisis de transiciones entre párrafos
        # Texto de IA suele tener transiciones muy explícitas
        transition_words = ['however', 'therefore', 'furthermore', 'moreover', 
                           'additionally', 'consequently', 'meanwhile', 'subsequently']
        transition_count = sum(1 for word in transition_words if word in text.lower())
        
        if len(sentences) > 0:
            transition_ratio = transition_count / len(sentences)
            # Ratio alto de transiciones puede indicar IA
            if transition_ratio > 0.15:
                score += 0.3
        
        return min(score, 1.0)
    
    def _apply_adaptive_weights(self, method_weights: Dict[str, float], 
                                detection_methods: List[str], text: str) -> Dict[str, float]:
        """Aplica pesos adaptativos basados en el historial y contexto - NUEVO"""
        # Si no hay historial suficiente, usar pesos por defecto
        if len(self.detection_history) < 10:
            return method_weights
        
        # Calcular rendimiento de cada método basado en historial
        if not self.model_performance:
            # Inicializar rendimiento
            for method in detection_methods:
                self.model_performance[method] = {
                    "success_count": 0,
                    "total_count": 0,
                    "avg_score": 0.5
                }
        
        # Ajustar pesos basado en longitud del texto
        text_length = len(text.split())
        if text_length < 50:
            # Textos cortos: más peso a pattern matching
            if "pattern_matching" in method_weights:
                method_weights["pattern_matching"] *= 1.2
        elif text_length > 500:
            # Textos largos: más peso a análisis estructural
            if "structure_analysis" in method_weights:
                method_weights["structure_analysis"] *= 1.15
            if "semantic_coherence" in method_weights:
                method_weights["semantic_coherence"] *= 1.1
        
        # Ajustar pesos basado en modelos detectados
        if detection_methods and "pattern_matching" in detection_methods:
            # Si hay pattern matching, aumentar su peso ligeramente
            if "pattern_matching" in method_weights:
                method_weights["pattern_matching"] *= 1.1
        
        # Normalizar para mantener suma <= 1.0
        total = sum(method_weights.values())
        if total > 1.0:
            method_weights = {k: v / total for k, v in method_weights.items()}
        
        return method_weights
    
    def _analyze_historical_context(self, text: str, detected_models: List[Dict]) -> float:
        """Analiza contexto histórico comparando con detecciones anteriores - NUEVO"""
        score = 0.0
        
        if len(self.detection_history) < 5:
            return 0.0
        
        # 1. Comparar con detecciones recientes similares
        recent_detections = self.detection_history[-10:]
        
        # Si hay modelos detectados, verificar si son consistentes con el historial
        if detected_models:
            model_names = [m["model_name"] for m in detected_models]
            recent_models = [entry.get("primary_model") for entry in recent_detections 
                           if entry.get("primary_model")]
            
            # Si el modelo detectado aparece frecuentemente en el historial
            if recent_models:
                model_frequency = sum(1 for m in recent_models if m in model_names)
                if model_frequency > len(recent_models) * 0.3:  # Más del 30%
                    score += 0.3
        
        # 2. Análisis de tendencias
        # Si hay muchas detecciones de IA recientes, puede indicar patrón
        recent_ai_count = sum(1 for entry in recent_detections 
                             if entry.get("is_ai_generated", False))
        recent_ai_ratio = recent_ai_count / len(recent_detections) if recent_detections else 0
        
        # Si hay un patrón consistente de detecciones de IA
        if recent_ai_ratio > 0.7:  # Más del 70% son IA
            score += 0.2
        elif recent_ai_ratio < 0.3:  # Menos del 30% son IA
            score -= 0.1  # Penalizar si el patrón es inconsistente
        
        # 3. Análisis de confianza histórica
        recent_confidences = [entry.get("confidence_score", 0.0) 
                            for entry in recent_detections]
        if recent_confidences:
            avg_confidence = np.mean(recent_confidences)
            # Si la confianza promedio es alta, puede indicar patrón
            if avg_confidence > 0.7:
                score += 0.2
        
        return max(min(score, 1.0), 0.0)
    
    def _analyze_advanced_ngrams(self, text: str) -> float:
        """Análisis avanzado de n-gramas (trigramas, 4-gramas) - NUEVO"""
        score = 0.0
        words = text.lower().split()
        
        if len(words) < 10:
            return 0.0
        
        # 1. Análisis de trigramas
        trigrams = {}
        for i in range(len(words) - 2):
            trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
            trigrams[trigram] = trigrams.get(trigram, 0) + 1
        
        if trigrams:
            # Calcular diversidad de trigramas
            trigram_diversity = len(trigrams) / len(words) if len(words) > 0 else 0
            
            # Baja diversidad de trigramas puede indicar IA
            if trigram_diversity < 0.2:
                score += 0.3
            elif trigram_diversity < 0.3:
                score += 0.15
            
            # Trigramas muy repetidos
            max_trigram_freq = max(trigrams.values())
            if max_trigram_freq > 3:  # Aparece más de 3 veces
                score += 0.2
        
        # 2. Análisis de 4-gramas (frases comunes de IA)
        fourgrams = {}
        for i in range(len(words) - 3):
            fourgram = f"{words[i]} {words[i+1]} {words[i+2]} {words[i+3]}"
            fourgrams[fourgram] = fourgrams.get(fourgram, 0) + 1
        
        # Frases típicas de IA (4-gramas comunes)
        ai_fourgrams = [
            'it is important to', 'it should be noted', 'as a result of',
            'in order to be', 'it is worth noting', 'in the case of',
            'on the other hand', 'in addition to the', 'as well as the'
        ]
        
        ai_fourgram_count = sum(1 for fg in ai_fourgrams if fg in ' '.join(words))
        if ai_fourgram_count > 0:
            score += min(ai_fourgram_count * 0.15, 0.3)
        
        # 3. Análisis de secuencias repetitivas
        # Detectar patrones como "A, B, and C" repetidos
        repetitive_patterns = [
            r'\w+,\s+\w+,\s+and\s+\w+',  # A, B, and C
            r'\w+\s+and\s+\w+\s+and\s+\w+',  # A and B and C
        ]
        
        repetitive_count = sum(len(re.findall(pattern, text.lower())) 
                              for pattern in repetitive_patterns)
        if repetitive_count > 2:
            score += 0.2
        
        # 4. Análisis de distribución de n-gramas
        # Texto de IA suele tener distribución más uniforme
        if trigrams:
            trigram_freqs = list(trigrams.values())
            if len(trigram_freqs) > 1:
                freq_variance = np.var(trigram_freqs)
                avg_freq = np.mean(trigram_freqs)
                
                # Baja varianza indica distribución uniforme (típico de IA)
                if avg_freq > 0:
                    cv = np.sqrt(freq_variance) / avg_freq
                    if cv < 0.5:  # Baja variación
                        score += 0.15
        
        return min(score, 1.0)
    
    def _analyze_comparative_similarity(self, text: str) -> float:
        """Analiza similitud comparativa con textos conocidos - NUEVO"""
        score = 0.0
        
        if not self.known_ai_texts and not self.known_human_texts:
            return 0.0
        
        text_words = set(re.findall(r'\b\w+\b', text.lower()))
        
        # Comparar con textos conocidos de IA
        if self.known_ai_texts:
            ai_similarities = []
            for known_text in self.known_ai_texts[:10]:  # Limitar a 10 para rendimiento
                known_words = set(re.findall(r'\b\w+\b', known_text.lower()))
                
                if len(text_words) > 0 and len(known_words) > 0:
                    intersection = len(text_words.intersection(known_words))
                    union = len(text_words.union(known_words))
                    similarity = intersection / union if union > 0 else 0
                    ai_similarities.append(similarity)
            
            if ai_similarities:
                max_ai_similarity = max(ai_similarities)
                avg_ai_similarity = np.mean(ai_similarities)
                
                # Alta similitud con textos de IA conocidos
                if max_ai_similarity > 0.4:
                    score += 0.4
                elif avg_ai_similarity > 0.3:
                    score += 0.2
        
        # Comparar con textos conocidos humanos
        if self.known_human_texts:
            human_similarities = []
            for known_text in self.known_human_texts[:10]:  # Limitar a 10
                known_words = set(re.findall(r'\b\w+\b', known_text.lower()))
                
                if len(text_words) > 0 and len(known_words) > 0:
                    intersection = len(text_words.intersection(known_words))
                    union = len(text_words.union(known_words))
                    similarity = intersection / union if union > 0 else 0
                    human_similarities.append(similarity)
            
            if human_similarities:
                max_human_similarity = max(human_similarities)
                avg_human_similarity = np.mean(human_similarities)
                
                # Alta similitud con textos humanos conocidos
                if max_human_similarity > 0.4:
                    score -= 0.2  # Penalizar si es similar a texto humano
                elif avg_human_similarity > 0.3:
                    score -= 0.1
        
        # Análisis de patrones comunes
        # Si el texto tiene patrones similares a textos de IA conocidos
        if self.known_ai_texts:
            # Analizar longitud promedio
            known_lengths = [len(t.split()) for t in self.known_ai_texts[:10]]
            text_length = len(text.split())
            
            if known_lengths:
                avg_known_length = np.mean(known_lengths)
                if abs(text_length - avg_known_length) < avg_known_length * 0.3:
                    score += 0.1
        
        return max(min(score, 1.0), 0.0)
    
    def _analyze_with_ml_patterns(self, text: str, detected_models: List[Dict], 
                                  scores: List[float]) -> float:
        """Análisis con patrones de machine learning básico - NUEVO"""
        score = 0.0
        
        # 1. Análisis de combinación de características
        # Si múltiples métodos detectan IA, aumenta la confianza
        if len(scores) > 5:
            high_scores = [s for s in scores if s > 0.6]
            if len(high_scores) > 3:  # Más de 3 métodos con score alto
                score += 0.3
        
        # 2. Análisis de consistencia entre métodos
        if len(scores) > 1:
            score_variance = np.var(scores)
            avg_score = np.mean(scores)
            
            # Baja varianza indica consistencia (típico de IA)
            if avg_score > 0:
                cv = np.sqrt(score_variance) / avg_score
                if cv < 0.3:  # Baja variación
                    score += 0.2
        
        # 3. Análisis de modelos detectados
        if detected_models:
            # Si hay modelos detectados con alta confianza
            high_confidence_models = [m for m in detected_models if m.get("confidence", 0) > 0.7]
            if len(high_confidence_models) > 0:
                score += 0.2
            
            # Si múltiples modelos detectados (puede indicar texto mixto o parafraseo)
            if len(detected_models) > 1:
                score += 0.1
        
        # 4. Análisis de longitud y complejidad
        word_count = len(text.split())
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Textos de IA suelen tener características específicas según longitud
        if word_count > 500:
            # Textos largos de IA: alta coherencia, estructura clara
            if len(sentences) > 0:
                avg_sentence_length = np.mean([len(s.split()) for s in sentences])
                if 12 <= avg_sentence_length <= 22:
                    score += 0.1
        elif word_count < 100:
            # Textos cortos de IA: muy estructurados, sin errores
            if len(sentences) > 0:
                avg_sentence_length = np.mean([len(s.split()) for s in sentences])
                if 8 <= avg_sentence_length <= 18:
                    score += 0.1
        
        # 5. Análisis de características combinadas
        # Combinar múltiples señales para decisión final
        feature_count = 0
        
        # Señal 1: Múltiples métodos activos
        if len(scores) > 8:
            feature_count += 1
        
        # Señal 2: Modelos detectados
        if detected_models:
            feature_count += 1
        
        # Señal 3: Scores consistentes
        if len(scores) > 1 and np.std(scores) < 0.2:
            feature_count += 1
        
        # Señal 4: Texto estructurado
        if re.search(r'\b(first|second|third|finally|in conclusion)', text, re.IGNORECASE):
            feature_count += 1
        
        # Si hay múltiples señales, aumenta score
        if feature_count >= 3:
            score += 0.2
        elif feature_count >= 2:
            score += 0.1
        
        return min(score, 1.0)
    
    def add_known_text(self, text: str, is_ai: bool):
        """Añade un texto conocido al sistema de aprendizaje - NUEVO"""
        if is_ai:
            if len(self.known_ai_texts) >= self.max_known_texts:
                self.known_ai_texts.pop(0)  # FIFO
            self.known_ai_texts.append(text)
        else:
            if len(self.known_human_texts) >= self.max_known_texts:
                self.known_human_texts.pop(0)  # FIFO
            self.known_human_texts.append(text)
        
        logger.info(f"Texto conocido añadido: {'IA' if is_ai else 'Humano'}")
    
    def _analyze_model_signatures(self, text: str, detected_models: List[Dict]) -> float:
        """Analiza firmas características específicas de cada modelo - NUEVO"""
        score = 0.0
        
        if not detected_models:
            return 0.0
        
        # Firmas específicas por modelo
        model_signatures = {
            "gpt-4": {
                "phrases": ["it's important to note", "it's worth mentioning", "it should be emphasized"],
                "structure": "highly_structured",
                "formality": "very_formal"
            },
            "gpt-3.5": {
                "phrases": ["let me", "i'd like to", "i think", "in my opinion"],
                "structure": "moderately_structured",
                "formality": "moderate"
            },
            "claude": {
                "phrases": ["i understand", "to clarify", "it's worth considering", "let's explore"],
                "structure": "very_structured",
                "formality": "very_formal"
            },
            "gemini": {
                "phrases": ["here's", "let's", "i'll", "you can"],
                "structure": "moderately_structured",
                "formality": "moderate"
            }
        }
        
        text_lower = text.lower()
        primary_model = max(detected_models, key=lambda x: x["confidence"]) if detected_models else None
        
        if primary_model:
            model_name = primary_model.get("model_name", "").lower()
            
            # Verificar firmas del modelo detectado
            if model_name in model_signatures:
                signatures = model_signatures[model_name]
                
                # Verificar frases características
                phrase_matches = sum(1 for phrase in signatures["phrases"] if phrase in text_lower)
                if phrase_matches > 0:
                    score += min(phrase_matches * 0.15, 0.4)
                
                # Verificar estructura
                if signatures["structure"] == "highly_structured":
                    if re.search(r'\b(first|second|third|finally|in conclusion)', text_lower):
                        score += 0.2
                elif signatures["structure"] == "very_structured":
                    if re.search(r'\b(?:introduction|overview|summary|conclusion)', text_lower):
                        score += 0.2
                
                # Verificar formalidad
                if signatures["formality"] == "very_formal":
                    formal_words = ['therefore', 'however', 'furthermore', 'moreover', 'consequently']
                    formal_count = sum(1 for word in formal_words if word in text_lower)
                    if formal_count > 2:
                        score += 0.2
        
        # Análisis de combinación de características del modelo
        # Si múltiples características coinciden, aumenta confianza
        if score > 0.3:
            score += 0.1  # Bonus por múltiples coincidencias
        
        return min(score, 1.0)
    
    def _analyze_semantic_embeddings(self, text: str) -> float:
        """Análisis básico de embeddings semánticos sin modelos externos - NUEVO"""
        score = 0.0
        words = text.lower().split()
        
        if len(words) < 10:
            return 0.0
        
        # 1. Análisis de clusters semánticos básico
        # Agrupar palabras por similitud de longitud y frecuencia
        word_lengths = [len(w) for w in words if len(w) > 3]
        if len(word_lengths) > 0:
            avg_length = np.mean(word_lengths)
            length_std = np.std(word_lengths)
            
            # Texto de IA suele tener distribución más uniforme de longitudes
            if length_std < 2.0:  # Baja desviación
                score += 0.2
        
        # 2. Análisis de co-ocurrencia de palabras
        # Palabras que aparecen juntas frecuentemente
        word_pairs = {}
        for i in range(len(words) - 1):
            if len(words[i]) > 3 and len(words[i+1]) > 3:
                pair = f"{words[i]}_{words[i+1]}"
                word_pairs[pair] = word_pairs.get(pair, 0) + 1
        
        # Muchas co-ocurrencias repetidas pueden indicar IA
        if word_pairs:
            max_cooccurrence = max(word_pairs.values())
            if max_cooccurrence > 2:  # Aparece más de 2 veces
                score += 0.2
        
        # 3. Análisis de densidad semántica
        # Palabras significativas vs palabras funcionales
        significant_words = [w for w in words if len(w) > 4]
        functional_words = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with']
        functional_count = sum(1 for w in words if w in functional_words)
        
        if len(words) > 0:
            significant_ratio = len(significant_words) / len(words)
            functional_ratio = functional_count / len(words)
            
            # Texto de IA suele tener balance específico
            if 0.4 <= significant_ratio <= 0.6 and 0.3 <= functional_ratio <= 0.5:
                score += 0.2
        
        # 4. Análisis de distribución de palabras
        # Texto de IA tiene distribución más predecible
        word_freq = {}
        for word in words:
            if len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        if word_freq:
            freqs = list(word_freq.values())
            if len(freqs) > 1:
                freq_variance = np.var(freqs)
                avg_freq = np.mean(freqs)
                
                # Baja varianza relativa indica distribución uniforme
                if avg_freq > 0:
                    cv = np.sqrt(freq_variance) / avg_freq
                    if cv < 0.5:  # Baja variación
                        score += 0.2
        
        # 5. Análisis de contexto semántico
        # Palabras relacionadas semánticamente (sinónimos, antónimos básicos)
        semantic_groups = [
            ['good', 'great', 'excellent', 'wonderful', 'fantastic'],
            ['important', 'significant', 'crucial', 'essential', 'vital'],
            ['problem', 'issue', 'challenge', 'difficulty', 'obstacle'],
            ['solution', 'answer', 'resolution', 'fix', 'remedy']
        ]
        
        semantic_matches = 0
        for group in semantic_groups:
            group_matches = sum(1 for word in group if word in words)
            if group_matches > 1:  # Múltiples palabras del mismo grupo
                semantic_matches += 1
        
        if semantic_matches > 1:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_temporal_patterns(self, text: str, metadata: Optional[Dict]) -> float:
        """Analiza patrones temporales de generación - NUEVO"""
        score = 0.0
        
        # 1. Análisis de timestamp en metadatos
        if metadata and metadata.get("timestamp"):
            import datetime
            try:
                if isinstance(metadata["timestamp"], (int, float)):
                    content_time = datetime.datetime.fromtimestamp(metadata["timestamp"])
                else:
                    content_time = datetime.datetime.fromisoformat(str(metadata["timestamp"]))
                
                now = datetime.datetime.now()
                age_hours = (now - content_time).total_seconds() / 3600
                
                # Contenido muy reciente puede ser más sospechoso
                if age_hours < 1:  # Menos de 1 hora
                    score += 0.3
                elif age_hours < 24:  # Menos de 24 horas
                    score += 0.1
            except:
                pass
        
        # 2. Análisis de patrones de tiempo en el texto
        # Referencias temporales específicas pueden indicar generación reciente
        time_patterns = [
            r'\b(?:today|yesterday|recently|lately|now|currently)\b',
            r'\b(?:in \d{4}|this year|this month|this week)\b',
            r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b'
        ]
        
        time_matches = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in time_patterns)
        if time_matches > 0:
            score += min(time_matches * 0.1, 0.2)
        
        # 3. Análisis de referencias a eventos recientes
        # Texto de IA puede tener referencias a eventos muy recientes
        recent_event_patterns = [
            r'\b(?:latest|newest|most recent|current)\b',
            r'\b(?:as of|up to date|updated)\b'
        ]
        
        recent_matches = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in recent_event_patterns)
        if recent_matches > 1:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_hybrid_models(self, text: str, detected_models: List[Dict]) -> float:
        """Detecta uso de modelos híbridos o combinados - NUEVO"""
        score = 0.0
        
        if len(detected_models) < 2:
            return 0.0
        
        # 1. Múltiples modelos con confianza similar
        confidences = [m.get("confidence", 0) for m in detected_models]
        if len(confidences) > 1:
            conf_std = np.std(confidences)
            conf_mean = np.mean(confidences)
            
            # Si las confianzas son similares, puede indicar uso híbrido
            if conf_mean > 0 and conf_std / conf_mean < 0.3:  # Baja variación relativa
                score += 0.3
        
        # 2. Modelos de diferentes proveedores
        providers = [m.get("provider") for m in detected_models if m.get("provider")]
        unique_providers = len(set(providers))
        
        if unique_providers > 1:
            score += 0.3  # Múltiples proveedores = posible uso híbrido
        
        # 3. Patrones de diferentes modelos en diferentes partes del texto
        # Dividir texto en partes y analizar cada una
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) > 5:
            # Analizar primera mitad vs segunda mitad
            first_half = ' '.join(sentences[:len(sentences)//2])
            second_half = ' '.join(sentences[len(sentences)//2:])
            
            # Detectar modelos en cada mitad
            first_models = self._detect_model_patterns(first_half)
            second_models = self._detect_model_patterns(second_half)
            
            if first_models and second_models:
                first_model_names = {m.get("model_name") for m in first_models}
                second_model_names = {m.get("model_name") for m in second_models}
                
                # Si hay modelos diferentes en cada mitad
                if not first_model_names.intersection(second_model_names):
                    score += 0.4  # Modelos completamente diferentes = híbrido
        
        # 4. Análisis de estilo mixto
        # Texto que combina características de diferentes modelos
        text_lower = text.lower()
        
        # Características de GPT
        gpt_features = sum(1 for phrase in ["it's important to note", "it's worth mentioning"] if phrase in text_lower)
        # Características de Claude
        claude_features = sum(1 for phrase in ["i understand", "to clarify", "let's explore"] if phrase in text_lower)
        # Características de Gemini
        gemini_features = sum(1 for phrase in ["here's", "let's", "you can"] if phrase in text_lower)
        
        # Si hay características de múltiples modelos
        model_features = sum([gpt_features > 0, claude_features > 0, gemini_features > 0])
        if model_features > 1:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_advanced_frequency(self, text: str) -> float:
        """Análisis avanzado de frecuencia de palabras y patrones - NUEVO"""
        score = 0.0
        words = text.lower().split()
        
        if len(words) < 20:
            return 0.0
        
        # 1. Análisis de distribución de Zipf
        # Texto natural sigue distribución de Zipf aproximadamente
        word_freq = {}
        for word in words:
            word_clean = word.strip('.,!?;:()[]{}"\'')
            if word_clean and len(word_clean) > 2:
                word_freq[word_clean] = word_freq.get(word_clean, 0) + 1
        
        if word_freq:
            freqs = sorted(word_freq.values(), reverse=True)
            
            # Calcular si sigue distribución de Zipf
            # En Zipf, la frecuencia del rango n es aproximadamente 1/n
            if len(freqs) > 5:
                # Comparar con distribución esperada
                expected_ratio = freqs[0] / freqs[1] if freqs[1] > 0 else 0
                actual_ratio = freqs[1] / freqs[2] if len(freqs) > 2 and freqs[2] > 0 else 0
                
                # Texto de IA puede desviarse de Zipf
                if abs(expected_ratio - actual_ratio) > 0.5:
                    score += 0.2
        
        # 2. Análisis de palabras raras vs comunes
        # Texto de IA suele tener menos palabras muy raras
        common_words = {'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 
                       'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at'}
        rare_words = [w for w, f in word_freq.items() if f == 1 and w not in common_words]
        
        if len(words) > 0:
            rare_ratio = len(rare_words) / len(words)
            # Texto de IA tiene menos palabras raras (hapax legomena)
            if rare_ratio < 0.3:  # Menos del 30% son palabras raras
                score += 0.2
        
        # 3. Análisis de frecuencia de palabras funcionales específicas
        # Texto de IA usa ciertas palabras funcionales más frecuentemente
        ai_functional_words = ['the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                             'have', 'has', 'had', 'will', 'would', 'could', 'should']
        
        ai_functional_count = sum(1 for w in words if w in ai_functional_words)
        if len(words) > 0:
            ai_functional_ratio = ai_functional_count / len(words)
            # Ratio muy alto puede indicar IA
            if ai_functional_ratio > 0.15:  # Más del 15%
                score += 0.2
        
        # 4. Análisis de frecuencia de palabras de contenido
        # Texto de IA tiene distribución más uniforme de palabras de contenido
        content_words = [w for w in words if len(w) > 4 and w not in common_words]
        content_freq = {}
        for word in content_words:
            content_freq[word] = content_freq.get(word, 0) + 1
        
        if content_freq:
            content_freqs = list(content_freq.values())
            if len(content_freqs) > 1:
                freq_variance = np.var(content_freqs)
                avg_freq = np.mean(content_freqs)
                
                # Baja varianza indica distribución uniforme (típico de IA)
                if avg_freq > 0:
                    cv = np.sqrt(freq_variance) / avg_freq
                    if cv < 0.4:  # Baja variación
                        score += 0.2
        
        # 5. Análisis de palabras de alta frecuencia
        # Texto de IA puede tener palabras que aparecen con frecuencia inusual
        if word_freq:
            max_freq = max(word_freq.values())
            total_words = len(words)
            
            # Si una palabra aparece más del 5% del tiempo
            if max_freq / total_words > 0.05:
                # Pero no es una palabra funcional común
                most_common = [w for w, f in word_freq.items() if f == max_freq][0]
                if most_common not in common_words:
                    score += 0.2  # Palabra de contenido muy repetida
        
        return min(score, 1.0)
    
    def _analyze_advanced_contextual_coherence(self, text: str) -> float:
        """Análisis avanzado de coherencia contextual - NUEVO"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 5:
            return 0.0
        
        # 1. Análisis de progresión temática
        words_per_sentence = [set(s.lower().split()) for s in sentences]
        overlaps = []
        for i in range(len(words_per_sentence) - 1):
            if words_per_sentence[i] and words_per_sentence[i+1]:
                overlap = len(words_per_sentence[i] & words_per_sentence[i+1])
                union = len(words_per_sentence[i] | words_per_sentence[i+1])
                jaccard = overlap / union if union > 0 else 0
                overlaps.append(jaccard)
        
        if overlaps:
            avg_overlap = np.mean(overlaps)
            if avg_overlap > 0.15:
                score += 0.3
            elif avg_overlap > 0.10:
                score += 0.2
        
        # 2. Análisis de referencias cruzadas
        pronoun_patterns = [
            r'\b(this|that|these|those|it|they|he|she)\s+\w+',
            r'\b(such|same|similar|different)\s+\w+',
            r'\b(above|below|previously|earlier|later)\s+\w+'
        ]
        cross_ref_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in pronoun_patterns)
        if cross_ref_count > len(sentences) * 0.3:
            score += 0.25
        
        # 3. Análisis de coherencia lógica
        logical_connectors = [
            r'\b(?:because|since|as)\s+\w+.*\b(?:therefore|thus|hence|so)\s+\w+',
            r'\b(?:if|when)\s+\w+.*\b(?:then|consequently|as a result)\s+\w+',
            r'\b(?:although|though|while)\s+\w+.*\b(?:however|nevertheless|yet)\s+\w+'
        ]
        logical_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in logical_connectors)
        if logical_count > 0:
            score += min(logical_count * 0.1, 0.25)
        
        # 4. Análisis de consistencia de perspectiva
        first_person = len(re.findall(r'\b(?:i|me|my|mine|we|us|our|ours)\b', text, re.IGNORECASE))
        second_person = len(re.findall(r'\b(?:you|your|yours)\b', text, re.IGNORECASE))
        third_person = len(re.findall(r'\b(?:he|she|him|her|his|hers|they|them|their|theirs|it|its)\b', text, re.IGNORECASE))
        total_perspective = first_person + second_person + third_person
        if total_perspective > 0:
            max_perspective = max(first_person, second_person, third_person)
            consistency = max_perspective / total_perspective
            if consistency > 0.6:
                score += 0.2
        
        return min(score, 1.0)
    
    def _detect_text_deepfake(self, text: str, detected_models: List[Dict]) -> float:
        """Detección de deepfake de texto - manipulación o combinación artificial - NUEVO"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 6:
            return 0.0
        
        # 1. Detección de cambios abruptos de estilo
        segment_size = len(sentences) // 3
        segments = [' '.join(sentences[i:i+segment_size]) for i in range(0, len(sentences), segment_size)]
        segment_features = []
        for segment in segments:
            words = segment.lower().split()
            if len(words) > 0:
                avg_length = np.mean([len(w) for w in words])
                formal_words = sum(1 for w in ['therefore', 'however', 'furthermore', 'moreover', 'consequently'] if w in segment.lower())
                segment_features.append({'avg_word_length': avg_length, 'formal_count': formal_words, 'length': len(words)})
        
        if len(segment_features) > 1:
            word_length_vars = [f['avg_word_length'] for f in segment_features]
            if len(word_length_vars) > 1:
                word_length_cv = np.std(word_length_vars) / np.mean(word_length_vars) if np.mean(word_length_vars) > 0 else 0
                if word_length_cv > 0.3:
                    score += 0.3
        
        # 2. Detección de modelos múltiples
        if len(detected_models) > 1:
            model_names = [m['model_name'] for m in detected_models]
            unique_models = len(set(model_names))
            if unique_models > 1:
                score += 0.25
        
        # 3. Detección de parches o ediciones
        edit_markers = [r'\[.*?\]', r'\(.*?edited.*?\)', r'\[edit\]', r'\.\.\.']
        edit_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in edit_markers)
        if edit_count > 2:
            score += 0.2
        
        # 4. Detección de inconsistencias temporales
        past_tense = len(re.findall(r'\b(?:was|were|had|did|went|came|said|told)\b', text, re.IGNORECASE))
        present_tense = len(re.findall(r'\b(?:is|are|am|do|does|go|come|say|tell)\b', text, re.IGNORECASE))
        future_tense = len(re.findall(r'\b(?:will|shall|going to|would|could|should)\b', text, re.IGNORECASE))
        total_tense = past_tense + present_tense + future_tense
        if total_tense > 0:
            max_tense = max(past_tense, present_tense, future_tense)
            tense_consistency = max_tense / total_tense
            if tense_consistency < 0.5:
                score += 0.15
        
        # 5. Detección de vocabulario mixto
        formal_words = ['therefore', 'however', 'furthermore', 'moreover', 'consequently', 'additionally', 'nevertheless', 'accordingly', 'subsequently']
        informal_words = ["don't", "won't", "can't", "it's", "that's", "there's", "gonna", "wanna", "yeah", "ok", "okay"]
        formal_count = sum(1 for w in formal_words if w in text.lower())
        informal_count = sum(1 for w in informal_words if w in text.lower())
        if formal_count > 2 and informal_count > 2:
            score += 0.15
        
        return min(score, 1.0)
    
    def _detect_deepfake_patterns(self, text: str) -> float:
        """Detecta patrones de deepfake o contenido sintético - NUEVO"""
        score = 0.0
        
        synthetic_patterns = [
            r'\b(?:synthetic|artificial|generated|created by ai|ai-generated)\b',
            r'\b(?:deepfake|fake|fabricated|manufactured)\b',
            r'\b(?:this (?:content|text|document) (?:was|is) (?:generated|created|produced) (?:by|using))\b'
        ]
        
        synthetic_matches = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in synthetic_patterns)
        if synthetic_matches > 0:
            score += 0.4
        
        manipulation_patterns = [
            r'\b(?:edited|modified|altered|manipulated|processed)\b',
            r'\b(?:original (?:text|content|version))\b',
            r'\b(?:before|after|original|modified version)\b'
        ]
        
        manipulation_matches = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in manipulation_patterns)
        if manipulation_matches > 1:
            score += 0.2
        
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) > 5:
            sentence_lengths = [len(s.split()) for s in sentences]
            if len(sentence_lengths) > 1:
                length_cv = np.std(sentence_lengths) / np.mean(sentence_lengths) if np.mean(sentence_lengths) > 0 else 0
                if length_cv > 0.8:
                    score += 0.2
        
        watermark_indicators = [
            r'\[.*?watermark.*?\]',
            r'<!--.*?generated.*?-->',
            r'\/\*.*?ai.*?\*\/',
            r'generated.*?by.*?ai',
            r'created.*?using.*?artificial.*?intelligence'
        ]
        
        watermark_matches = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in watermark_indicators)
        if watermark_matches > 0:
            score += 0.3
        
        hidden_patterns = [r'[^\x00-\x7F]', r'\s{3,}', r'\t+']
        hidden_count = sum(len(re.findall(pattern, text)) for pattern in hidden_patterns)
        if hidden_count > len(text) * 0.01:
            score += 0.1
        
        return min(score, 1.0)
    
    def _enhanced_scoring_system(self, scores: List[float], detection_methods: List[str], 
                                detected_models: List[Dict]) -> float:
        """Sistema de scoring mejorado con machine learning básico - NUEVO"""
        score = 0.0
        
        if not scores or len(scores) == 0:
            return 0.0
        
        high_scores = [s for s in scores if s > 0.6]
        medium_scores = [s for s in scores if 0.4 <= s <= 0.6]
        total_methods = len(scores)
        
        if total_methods > 0:
            high_ratio = len(high_scores) / total_methods
            medium_ratio = len(medium_scores) / total_methods
            
            if high_ratio > 0.4:
                score += 0.3
            elif high_ratio > 0.25:
                score += 0.2
            
            if 0.3 <= high_ratio <= 0.6 and 0.2 <= medium_ratio <= 0.5:
                score += 0.2
        
        critical_methods = ['pattern_matching', 'statistical_analysis', 'model_signatures']
        critical_scores = [s for m, s in zip(detection_methods, scores) if m in critical_methods]
        
        if critical_scores:
            avg_critical = np.mean(critical_scores)
            if avg_critical > 0.7:
                score += 0.3
            elif avg_critical > 0.5:
                score += 0.2
        
        if detected_models:
            high_confidence_models = [m for m in detected_models if m.get("confidence", 0) > 0.75]
            if len(high_confidence_models) > 0:
                score += 0.2
            
            if len(detected_models) > 1:
                confidences = [m.get("confidence", 0) for m in detected_models]
                if len(confidences) > 1:
                    conf_std = np.std(confidences)
                    if conf_std < 0.15:
                        score += 0.1
        
        if len(scores) > 3:
            score_mean = np.mean(scores)
            score_std = np.std(scores)
            if score_mean > 0 and score_std / score_mean < 0.3:
                score += 0.2
        
        complementary_pairs = [
            ('pattern_matching', 'model_signatures'),
            ('statistical_analysis', 'advanced_frequency'),
            ('semantic_coherence', 'narrative_coherence'),
            ('structure_analysis', 'syntactic_complexity')
        ]
        
        for method1, method2 in complementary_pairs:
            if method1 in detection_methods and method2 in detection_methods:
                idx1 = detection_methods.index(method1)
                idx2 = detection_methods.index(method2)
                score1 = scores[idx1]
                score2 = scores[idx2]
                if score1 > 0.5 and score2 > 0.5:
                    score += 0.1
        
        return min(score, 1.0)
    
    def _analyze_advanced_repetition_patterns(self, text: str) -> float:
        """Análisis avanzado de patrones de repetición - NUEVO"""
        score = 0.0
        words = text.lower().split()
        
        if len(words) < 30:
            return 0.0
        
        word_freq = {}
        for word in words:
            word_clean = word.strip('.,!?;:()[]{}"\'')
            if word_clean and len(word_clean) > 3:
                word_freq[word_clean] = word_freq.get(word_clean, 0) + 1
        
        if not word_freq:
            return 0.0
        
        total_words = len(words)
        max_freq = max(word_freq.values())
        max_freq_ratio = max_freq / total_words if total_words > 0 else 0
        
        if max_freq_ratio > 0.08:
            score += 0.3
        elif max_freq_ratio > 0.05:
            score += 0.2
        
        repeated_words = [w for w, f in word_freq.items() if f > 3]
        if len(repeated_words) > len(word_freq) * 0.15:
            score += 0.25
        
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) > 3:
            sentence_starts = [s.split()[0].lower() if s.split() else '' for s in sentences]
            start_repetition = len(sentence_starts) - len(set(sentence_starts))
            if start_repetition > len(sentences) * 0.3:
                score += 0.2
        
        phrase_patterns = {}
        for i in range(len(words) - 2):
            phrase = ' '.join(words[i:i+3])
            phrase_patterns[phrase] = phrase_patterns.get(phrase, 0) + 1
        
        repeated_phrases = [p for p, f in phrase_patterns.items() if f > 1]
        if len(repeated_phrases) > 0 and len(repeated_phrases) / len(phrase_patterns) > 0.1:
            score += 0.25
        
        return min(score, 1.0)
    
    def _detect_ai_paraphrasing_advanced(self, text: str) -> float:
        """Detección avanzada de parafraseo con IA - NUEVO"""
        score = 0.0
        words = text.split()
        
        if len(words) < 50:
            return 0.0
        
        synonym_patterns = [
            (r'\b(?:important|significant|crucial|vital|essential)\b', 0.1),
            (r'\b(?:help|assist|aid|support|facilitate)\b', 0.1),
            (r'\b(?:good|excellent|outstanding|remarkable|exceptional)\b', 0.1),
            (r'\b(?:bad|poor|terrible|awful|dreadful)\b', 0.1),
            (r'\b(?:big|large|huge|enormous|massive)\b', 0.1),
            (r'\b(?:small|tiny|little|miniature|minuscule)\b', 0.1)
        ]
        
        synonym_count = 0
        for pattern, weight in synonym_patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            if matches > 2:
                synonym_count += matches
                score += min(matches * weight, 0.15)
        
        structural_variations = [
            r'\b(?:in order to|so as to|for the purpose of)\b',
            r'\b(?:due to the fact that|because|since|as)\b',
            r'\b(?:in the event that|if|should|when)\b',
            r'\b(?:with regard to|regarding|concerning|about)\b'
        ]
        
        variation_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in structural_variations)
        if variation_count > 3:
            score += 0.2
        
        formal_alternatives = [
            (r'\b(?:utilize|use|employ)\b', 0.1),
            (r'\b(?:facilitate|help|make easier)\b', 0.1),
            (r'\b(?:implement|put into effect|carry out)\b', 0.1),
            (r'\b(?:optimize|improve|make better)\b', 0.1)
        ]
        
        for pattern, weight in formal_alternatives:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            if matches > 1:
                score += min(matches * weight, 0.1)
        
        if synonym_count > 5:
            score += 0.15
        
        return min(score, 1.0)
    
    def _analyze_style_mixture(self, text: str) -> float:
        """Detecta mezclas de estilos (formal/informal) - NUEVO"""
        score = 0.0
        words = text.split()
        
        if len(words) < 40:
            return 0.0
        
        formal_indicators = [
            'therefore', 'however', 'furthermore', 'moreover', 'consequently',
            'additionally', 'nevertheless', 'accordingly', 'subsequently',
            'utilize', 'facilitate', 'implement', 'optimize', 'enhance'
        ]
        
        informal_indicators = [
            "don't", "won't", "can't", "it's", "that's", "there's",
            "gonna", "wanna", "yeah", "ok", "okay", "lol", "btw",
            "imo", "tbh", "nvm", "idk"
        ]
        
        formal_count = sum(1 for word in formal_indicators if word.lower() in text.lower())
        informal_count = sum(1 for word in informal_indicators if word.lower() in text.lower())
        
        if formal_count > 3 and informal_count > 2:
            score += 0.4
        elif formal_count > 2 and informal_count > 1:
            score += 0.3
        
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) > 5:
            formal_sentences = sum(1 for s in sentences if any(fi in s.lower() for fi in formal_indicators))
            informal_sentences = sum(1 for s in sentences if any(ii in s.lower() for ii in informal_indicators))
            
            if formal_sentences > 0 and informal_sentences > 0:
                mixture_ratio = min(formal_sentences, informal_sentences) / max(formal_sentences, informal_sentences)
                if mixture_ratio > 0.3:
                    score += 0.3
        
        academic_words = ['furthermore', 'moreover', 'consequently', 'therefore', 'nevertheless']
        casual_words = ["don't", "won't", "can't", "gonna", "wanna"]
        
        academic_count = sum(1 for w in academic_words if w in text.lower())
        casual_count = sum(1 for w in casual_words if w in text.lower())
        
        if academic_count > 2 and casual_count > 1:
            score += 0.3
        
        return min(score, 1.0)
    
    def _analyze_generation_sophistication(self, text: str) -> float:
        """Análisis sofisticado de patrones de generación - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 50 or len(sentences) < 5:
            return 0.0
        
        sophisticated_starters = [
            r'^(?:it is|it\'s) (?:important|worth noting|crucial|essential) (?:to note|to mention|to point out)',
            r'^(?:in|when) (?:considering|analyzing|examining|evaluating)',
            r'^(?:to|in order to) (?:better|more effectively|more accurately)',
            r'^(?:this|these) (?:findings|results|observations|insights)',
            r'^(?:it|this) (?:should|must|needs to) (?:be noted|be emphasized|be highlighted)'
        ]
        
        sophisticated_count = 0
        for pattern in sophisticated_starters:
            matches = sum(1 for s in sentences if re.search(pattern, s, re.IGNORECASE))
            sophisticated_count += matches
        
        if sophisticated_count > 2:
            score += 0.3
        elif sophisticated_count > 0:
            score += 0.15
        
        transition_overuse = [
            'however', 'therefore', 'furthermore', 'moreover', 'consequently',
            'additionally', 'meanwhile', 'subsequently', 'nevertheless', 'thus'
        ]
        
        transition_count = sum(1 for word in transition_overuse if word.lower() in text.lower())
        if transition_count > len(sentences) * 0.2:
            score += 0.25
        
        hedging_phrases = [
            r'\b(?:it (?:is|may be|might be|could be) (?:possible|likely|probable|plausible))',
            r'\b(?:there (?:is|may be|might be|could be) (?:a possibility|a chance|a likelihood))',
            r'\b(?:it (?:seems|appears|looks) (?:that|as if|as though))',
            r'\b(?:one (?:might|could|may) (?:argue|suggest|propose|consider))'
        ]
        
        hedging_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in hedging_phrases)
        if hedging_count > 2:
            score += 0.2
        
        qualifier_overuse = [
            'somewhat', 'rather', 'quite', 'fairly', 'relatively', 'comparatively',
            'generally', 'typically', 'usually', 'often', 'frequently'
        ]
        
        qualifier_count = sum(1 for q in qualifier_overuse if q.lower() in text.lower())
        if qualifier_count > len(words) * 0.02:
            score += 0.15
        
        return min(score, 1.0)
    
    def _analyze_lexical_diversity_advanced(self, text: str) -> float:
        """Análisis avanzado de diversidad léxica - NUEVO"""
        score = 0.0
        words = text.lower().split()
        
        if len(words) < 50:
            return 0.0
        
        word_freq = {}
        for word in words:
            word_clean = word.strip('.,!?;:()[]{}"\'')
            if word_clean and len(word_clean) > 2:
                word_freq[word_clean] = word_freq.get(word_clean, 0) + 1
        
        if not word_freq:
            return 0.0
        
        unique_words = len(word_freq)
        total_words = len(words)
        type_token_ratio = unique_words / total_words if total_words > 0 else 0
        
        if 0.3 <= type_token_ratio <= 0.5:
            score += 0.3
        elif type_token_ratio < 0.3:
            score += 0.2
        
        hapax_legomena = [w for w, f in word_freq.items() if f == 1]
        hapax_ratio = len(hapax_legomena) / unique_words if unique_words > 0 else 0
        
        if hapax_ratio < 0.4:
            score += 0.25
        
        word_lengths = [len(w) for w in word_freq.keys()]
        if word_lengths:
            avg_word_length = np.mean(word_lengths)
            if 4.5 <= avg_word_length <= 6.5:
                score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_hedging_patterns(self, text: str) -> float:
        """Detecta patrones de hedging (cautela) típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        
        if len(words) < 30:
            return 0.0
        
        hedging_phrases = [
            r'\b(?:it (?:is|may be|might be|could be|seems|appears) (?:possible|likely|probable|plausible|important|worth noting))',
            r'\b(?:there (?:is|may be|might be|could be) (?:a possibility|a chance|a likelihood|evidence))',
            r'\b(?:it (?:seems|appears|looks) (?:that|as if|as though))',
            r'\b(?:one (?:might|could|may|should) (?:argue|suggest|propose|consider|note))',
            r'\b(?:it (?:would|could|might) (?:be (?:argued|suggested|noted|considered))|seem)',
            r'\b(?:to (?:some|a certain) (?:extent|degree))',
            r'\b(?:in (?:some|many|most) (?:cases|instances|situations))',
            r'\b(?:generally|typically|usually|often|frequently|commonly)'
        ]
        
        hedging_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in hedging_phrases)
        
        if hedging_count > len(words) * 0.03:
            score += 0.4
        elif hedging_count > len(words) * 0.02:
            score += 0.3
        elif hedging_count > 0:
            score += 0.15
        
        uncertainty_markers = [
            'perhaps', 'maybe', 'possibly', 'potentially', 'presumably',
            'supposedly', 'allegedly', 'reportedly', 'apparently', 'seemingly'
        ]
        
        uncertainty_count = sum(1 for marker in uncertainty_markers if marker.lower() in text.lower())
        if uncertainty_count > 2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_sentence_complexity_distribution(self, text: str) -> float:
        """Analiza la distribución de complejidad de oraciones - NUEVO"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 10:
            return 0.0
        
        sentence_lengths = [len(s.split()) for s in sentences]
        if len(sentence_lengths) < 2:
            return 0.0
        
        mean_length = np.mean(sentence_lengths)
        std_length = np.std(sentence_lengths)
        cv = std_length / mean_length if mean_length > 0 else 0
        
        if cv < 0.3:
            score += 0.3
        elif cv < 0.4:
            score += 0.2
        
        complex_sentences = 0
        for sentence in sentences:
            clause_count = len(re.findall(r'\b(?:and|or|but|because|since|although|while|if|when)\b', sentence, re.IGNORECASE))
            if clause_count > 1:
                complex_sentences += 1
        
        complex_ratio = complex_sentences / len(sentences) if len(sentences) > 0 else 0
        
        if 0.4 <= complex_ratio <= 0.7:
            score += 0.25
        elif complex_ratio > 0.7:
            score += 0.15
        
        simple_sentences = sum(1 for s in sentences if len(s.split()) < 10)
        medium_sentences = sum(1 for s in sentences if 10 <= len(s.split()) <= 20)
        long_sentences = sum(1 for s in sentences if len(s.split()) > 20)
        
        if len(sentences) > 0:
            simple_ratio = simple_sentences / len(sentences)
            medium_ratio = medium_sentences / len(sentences)
            long_ratio = long_sentences / len(sentences)
            
            if 0.3 <= medium_ratio <= 0.6 and simple_ratio < 0.3 and long_ratio < 0.3:
                score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_verbosity_patterns(self, text: str) -> float:
        """Detecta patrones de verbosidad típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 40:
            return 0.0
        
        verbose_phrases = [
            r'\b(?:it is important to note that|it should be noted that|it is worth mentioning that)',
            r'\b(?:in order to|so as to|for the purpose of)',
            r'\b(?:due to the fact that|because of the fact that)',
            r'\b(?:with regard to|in regard to|concerning the matter of)',
            r'\b(?:in the context of|within the framework of|in the realm of)',
            r'\b(?:it can be observed that|it may be seen that|it is evident that)',
            r'\b(?:as a result of the fact that|owing to the fact that)',
            r'\b(?:in the event that|in case that|should it be the case that)'
        ]
        
        verbose_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in verbose_phrases)
        
        if verbose_count > 3:
            score += 0.4
        elif verbose_count > 1:
            score += 0.25
        
        redundant_phrases = [
            r'\b(?:each and every|first and foremost|any and all)',
            r'\b(?:various different|many different|several different)',
            r'\b(?:completely finished|totally complete|absolutely certain)',
            r'\b(?:free gift|new innovation|past history)'
        ]
        
        redundant_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in redundant_phrases)
        if redundant_count > 1:
            score += 0.2
        
        avg_words_per_sentence = len(words) / len(sentences) if len(sentences) > 0 else 0
        
        if 15 <= avg_words_per_sentence <= 25:
            score += 0.15
        
        return min(score, 1.0)
    
    def _analyze_pronoun_usage_patterns(self, text: str) -> float:
        """Analiza patrones de uso de pronombres - NUEVO"""
        score = 0.0
        words = text.split()
        
        if len(words) < 30:
            return 0.0
        
        first_person = len(re.findall(r'\b(?:i|me|my|mine|we|us|our|ours)\b', text, re.IGNORECASE))
        second_person = len(re.findall(r'\b(?:you|your|yours)\b', text, re.IGNORECASE))
        third_person = len(re.findall(r'\b(?:he|she|him|her|his|hers|they|them|their|theirs|it|its)\b', text, re.IGNORECASE))
        
        total_pronouns = first_person + second_person + third_person
        
        if total_pronouns > 0:
            first_ratio = first_person / total_pronouns
            second_ratio = second_person / total_pronouns
            third_ratio = third_person / total_pronouns
            
            if third_ratio > 0.6:
                score += 0.3
            elif third_ratio > 0.5:
                score += 0.2
            
            if first_ratio < 0.1 and second_ratio < 0.1:
                score += 0.2
        
        pronoun_repetition = len(re.findall(r'\b(it|this|that|these|those)\s+\w+\s+\1\b', text, re.IGNORECASE))
        if pronoun_repetition > 2:
            score += 0.15
        
        return min(score, 1.0)
    
    def _detect_ai_question_patterns(self, text: str) -> float:
        """Detecta patrones de preguntas típicos de IA - NUEVO"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 5:
            return 0.0
        
        question_count = text.count('?')
        question_ratio = question_count / len(sentences) if len(sentences) > 0 else 0
        
        if question_ratio > 0.3:
            score += 0.3
        elif question_ratio > 0.2:
            score += 0.2
        
        rhetorical_questions = [
            r'\b(?:have you ever|did you know|are you aware|do you realize)',
            r'\b(?:what if|what would happen if|imagine if)',
            r'\b(?:isn\'t it|don\'t you think|wouldn\'t you agree)'
        ]
        
        rhetorical_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in rhetorical_questions)
        if rhetorical_count > 1:
            score += 0.25
        
        question_starters = [
            r'^(?:what|how|why|when|where|who|which|can|could|would|should|may|might)',
            r'^(?:is|are|was|were|do|does|did|have|has|had)'
        ]
        
        question_starter_count = sum(1 for s in sentences 
                                    if any(re.search(pattern, s, re.IGNORECASE) for pattern in question_starters))
        
        if question_starter_count > len(sentences) * 0.2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_closure_patterns(self, text: str) -> float:
        """Analiza patrones de cierre típicos de IA - NUEVO"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 5:
            return 0.0
        
        last_sentences = ' '.join(sentences[-3:]) if len(sentences) >= 3 else ' '.join(sentences)
        
        closure_patterns = [
            r'\b(?:i hope this helps|hope this helps|i hope this|hope this)',
            r'\b(?:let me know if|feel free to|don\'t hesitate to|if you have any)',
            r'\b(?:please let me know|please feel free|if you need|if you require)',
            r'\b(?:in conclusion|to summarize|to conclude|in summary|to sum up)',
            r'\b(?:thank you|thanks|appreciate|grateful)',
            r'\b(?:if you have any questions|if you need clarification|if you\'d like to know more)'
        ]
        
        closure_count = sum(len(re.findall(pattern, last_sentences, re.IGNORECASE)) for pattern in closure_patterns)
        
        if closure_count > 1:
            score += 0.4
        elif closure_count > 0:
            score += 0.25
        
        ending_phrases = [
            'best regards', 'sincerely', 'yours truly', 'take care',
            'have a great day', 'good luck', 'all the best'
        ]
        
        ending_count = sum(1 for phrase in ending_phrases if phrase.lower() in last_sentences.lower())
        if ending_count > 0:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_enumeration_patterns(self, text: str) -> float:
        """Detecta patrones de enumeración típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        
        if len(words) < 40:
            return 0.0
        
        enumeration_markers = [
            r'\b(?:first|second|third|fourth|fifth|sixth|finally|lastly|last)',
            r'\b(?:one|two|three|four|five|six|seven|eight|nine|ten)',
            r'\b(?:firstly|secondly|thirdly|fourthly|fifthly)',
            r'\b(?:initially|subsequently|then|next|afterward|finally)'
        ]
        
        enum_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in enumeration_markers)
        
        if enum_count > 5:
            score += 0.4
        elif enum_count > 3:
            score += 0.3
        elif enum_count > 1:
            score += 0.15
        
        numbered_lists = len(re.findall(r'\d+\.\s+[A-Z]', text))
        bullet_lists = len(re.findall(r'[•\-\*]\s+[A-Z]', text))
        
        if numbered_lists > 3 or bullet_lists > 3:
            score += 0.3
        elif numbered_lists > 1 or bullet_lists > 1:
            score += 0.2
        
        sequential_patterns = [
            r'\b(?:step \d+|stage \d+|phase \d+|part \d+)',
            r'\b(?:point \d+|item \d+|element \d+)'
        ]
        
        sequential_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in sequential_patterns)
        if sequential_count > 2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_metaphor_patterns(self, text: str) -> float:
        """Analiza patrones de metáforas y lenguaje figurado - NUEVO"""
        score = 0.0
        words = text.split()
        
        if len(words) < 50:
            return 0.0
        
        metaphor_indicators = [
            r'\b(?:like|as|similar to|comparable to|akin to)\s+\w+',
            r'\b(?:metaphor|analogy|simile|comparison)',
            r'\b(?:is like|is as|resembles|mirrors|reflects)',
            r'\b(?:think of|imagine|picture|visualize)'
        ]
        
        metaphor_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in metaphor_indicators)
        
        if metaphor_count == 0 and len(words) > 100:
            score += 0.3
        
        figurative_language = [
            r'\b(?:it\'s like|it\'s as if|it\'s similar to)',
            r'\b(?:think of it as|imagine it as|picture it as)',
            r'\b(?:just like|much like|very much like)'
        ]
        
        figurative_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in figurative_language)
        if figurative_count > 2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_emphasis_patterns(self, text: str) -> float:
        """Detecta patrones de énfasis típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        
        if len(words) < 30:
            return 0.0
        
        emphasis_markers = [
            r'\b(?:important|crucial|essential|vital|critical|significant)',
            r'\b(?:very|extremely|highly|particularly|especially|notably)',
            r'\b(?:indeed|certainly|definitely|absolutely|undoubtedly)',
            r'\b(?:it is important|it is crucial|it is essential|it is vital)',
            r'\b(?:must|should|need to|have to|ought to)'
        ]
        
        emphasis_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in emphasis_markers)
        
        if emphasis_count > len(words) * 0.03:
            score += 0.4
        elif emphasis_count > len(words) * 0.02:
            score += 0.3
        elif emphasis_count > 3:
            score += 0.15
        
        intensifiers = ['very', 'extremely', 'highly', 'particularly', 'especially', 'notably', 'remarkably']
        intensifier_count = sum(1 for word in intensifiers if word.lower() in text.lower())
        
        if intensifier_count > len(words) * 0.02:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_modifier_patterns(self, text: str) -> float:
        """Analiza patrones de modificadores típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        
        if len(words) < 40:
            return 0.0
        
        modifier_patterns = [
            r'\b(?:quite|rather|somewhat|fairly|relatively|comparatively)\s+\w+',
            r'\b(?:very|extremely|highly|particularly|especially)\s+\w+',
            r'\b(?:more|less|most|least)\s+\w+',
            r'\b(?:incredibly|remarkably|exceptionally|notably)\s+\w+'
        ]
        
        modifier_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in modifier_patterns)
        
        if modifier_count > len(words) * 0.04:
            score += 0.3
        elif modifier_count > len(words) * 0.02:
            score += 0.2
        
        redundant_modifiers = [
            r'\b(?:very|extremely|incredibly)\s+(?:important|crucial|essential|significant)',
            r'\b(?:quite|rather|somewhat)\s+(?:interesting|notable|remarkable)',
            r'\b(?:highly|extremely|very)\s+(?:effective|efficient|successful)'
        ]
        
        redundant_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in redundant_modifiers)
        if redundant_count > 1:
            score += 0.25
        
        return min(score, 1.0)
    
    def _detect_ai_conditional_patterns(self, text: str) -> float:
        """Detecta patrones condicionales típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        conditional_markers = [
            r'\b(?:if|when|unless|provided that|as long as|in case)',
            r'\b(?:would|could|should|might|may)\s+\w+',
            r'\b(?:if\s+\w+\s+then|if\s+\w+\s+would|if\s+\w+\s+could)',
            r'\b(?:assuming|supposing|given that|considering that)'
        ]
        
        conditional_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in conditional_markers)
        
        if conditional_count > len(sentences) * 0.3:
            score += 0.3
        elif conditional_count > len(sentences) * 0.2:
            score += 0.2
        
        hypothetical_phrases = [
            r'\b(?:if\s+you|if\s+one|if\s+someone|if\s+we)',
            r'\b(?:suppose|imagine|assume|presume)',
            r'\b(?:what if|what would happen if|what might happen if)'
        ]
        
        hypothetical_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in hypothetical_phrases)
        if hypothetical_count > 2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_passive_voice_patterns(self, text: str) -> float:
        """Analiza patrones de voz pasiva típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        passive_voice_patterns = [
            r'\b(?:is|are|was|were|been|being)\s+\w+ed\b',
            r'\b(?:is|are|was|were|been|being)\s+\w+en\b',
            r'\b(?:is|are|was|were|been|being)\s+\w+ed\s+by\b',
            r'\b(?:has|have|had)\s+been\s+\w+ed\b',
            r'\b(?:will|would|can|could|should|must)\s+be\s+\w+ed\b'
        ]
        
        passive_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in passive_voice_patterns)
        
        if passive_count > len(sentences) * 0.4:
            score += 0.4
        elif passive_count > len(sentences) * 0.25:
            score += 0.3
        elif passive_count > len(sentences) * 0.15:
            score += 0.15
        
        passive_indicators = ['by', 'was', 'were', 'been', 'being']
        passive_indicator_count = sum(1 for word in words if word.lower() in passive_indicators)
        
        if passive_indicator_count > len(words) * 0.05:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_connector_patterns(self, text: str) -> float:
        """Detecta patrones de conectores típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 40:
            return 0.0
        
        connector_patterns = [
            r'\b(?:furthermore|moreover|additionally|in addition|also|besides)',
            r'\b(?:however|nevertheless|nonetheless|on the other hand|conversely)',
            r'\b(?:therefore|thus|hence|consequently|as a result|accordingly)',
            r'\b(?:for instance|for example|such as|namely|specifically)',
            r'\b(?:in conclusion|to summarize|in summary|overall|ultimately)'
        ]
        
        connector_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in connector_patterns)
        
        if connector_count > len(sentences) * 0.3:
            score += 0.4
        elif connector_count > len(sentences) * 0.2:
            score += 0.3
        elif connector_count > 3:
            score += 0.15
        
        transition_phrases = [
            r'\b(?:first of all|secondly|thirdly|finally|lastly)',
            r'\b(?:on the one hand|on the other hand)',
            r'\b(?:in other words|that is to say|to put it differently)'
        ]
        
        transition_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in transition_phrases)
        if transition_count > 2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_quantifier_patterns(self, text: str) -> float:
        """Analiza patrones de cuantificadores típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        
        if len(words) < 30:
            return 0.0
        
        quantifier_patterns = [
            r'\b(?:many|most|several|various|numerous|multiple|various)',
            r'\b(?:some|few|many|most|all|every|each)',
            r'\b(?:a number of|a variety of|a range of|a series of)',
            r'\b(?:the majority of|the majority|most of|many of)'
        ]
        
        quantifier_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in quantifier_patterns)
        
        if quantifier_count > len(words) * 0.03:
            score += 0.3
        elif quantifier_count > len(words) * 0.02:
            score += 0.2
        elif quantifier_count > 2:
            score += 0.1
        
        vague_quantifiers = ['many', 'some', 'several', 'various', 'numerous', 'multiple']
        vague_count = sum(1 for word in words if word.lower() in vague_quantifiers)
        
        if vague_count > len(words) * 0.02:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_assertion_patterns(self, text: str) -> float:
        """Detecta patrones de aserciones típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        assertion_patterns = [
            r'\b(?:it is clear that|it is evident that|it is obvious that)',
            r'\b(?:there is no doubt that|undoubtedly|certainly|definitely)',
            r'\b(?:it can be seen that|it can be observed that|it can be noted that)',
            r'\b(?:it should be noted that|it is important to note that)',
            r'\b(?:it is worth noting that|it is worth mentioning that)'
        ]
        
        assertion_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in assertion_patterns)
        
        if assertion_count > len(sentences) * 0.2:
            score += 0.4
        elif assertion_count > len(sentences) * 0.1:
            score += 0.3
        elif assertion_count > 1:
            score += 0.15
        
        definitive_statements = [
            r'\b(?:always|never|all|every|none|no one|nothing)',
            r'\b(?:must|should|ought to|have to|need to)',
            r'\b(?:cannot|can never|will never|must not)'
        ]
        
        definitive_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in definitive_statements)
        if definitive_count > len(sentences) * 0.3:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_negation_patterns(self, text: str) -> float:
        """Analiza patrones de negación típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        negation_patterns = [
            r'\b(?:not|no|never|neither|nor|none|nothing|nobody|nowhere)',
            r'\b(?:cannot|cannot|could not|would not|should not|must not)',
            r'\b(?:is not|are not|was not|were not|has not|have not)',
            r'\b(?:does not|do not|did not|will not|would not)'
        ]
        
        negation_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in negation_patterns)
        
        if negation_count > len(sentences) * 0.3:
            score += 0.3
        elif negation_count > len(sentences) * 0.2:
            score += 0.2
        elif negation_count > 2:
            score += 0.1
        
        double_negation = len(re.findall(r'\b(?:not|no|never)\s+\w+\s+(?:not|no|never)', text, re.IGNORECASE))
        if double_negation > 0:
            score += 0.15
        
        return min(score, 1.0)
    
    def _detect_ai_comparison_patterns(self, text: str) -> float:
        """Detecta patrones de comparación típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        comparison_patterns = [
            r'\b(?:compared to|compared with|in comparison to|in comparison with)',
            r'\b(?:similar to|similar as|as similar as|like|as)',
            r'\b(?:different from|different than|unlike|contrary to)',
            r'\b(?:more than|less than|greater than|smaller than)',
            r'\b(?:as\s+\w+\s+as|so\s+\w+\s+as|such\s+as)'
        ]
        
        comparison_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in comparison_patterns)
        
        if comparison_count > len(sentences) * 0.25:
            score += 0.3
        elif comparison_count > len(sentences) * 0.15:
            score += 0.2
        elif comparison_count > 2:
            score += 0.1
        
        superlative_patterns = [
            r'\b(?:the most|the least|the best|the worst|the greatest|the smallest)',
            r'\b(?:more\s+\w+|\w+er\s+than)',
            r'\b(?:most\s+\w+|least\s+\w+)'
        ]
        
        superlative_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in superlative_patterns)
        if superlative_count > len(sentences) * 0.2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_temporal_marker_patterns(self, text: str) -> float:
        """Analiza patrones de marcadores temporales típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        temporal_markers = [
            r'\b(?:first|initially|originally|at first|in the beginning)',
            r'\b(?:then|next|subsequently|afterward|after that|following that)',
            r'\b(?:finally|ultimately|eventually|in the end|at last)',
            r'\b(?:meanwhile|simultaneously|at the same time|concurrently)',
            r'\b(?:previously|before|earlier|prior to|in the past)',
            r'\b(?:later|afterwards|subsequently|then|following)'
        ]
        
        temporal_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in temporal_markers)
        
        if temporal_count > len(sentences) * 0.3:
            score += 0.3
        elif temporal_count > len(sentences) * 0.2:
            score += 0.2
        elif temporal_count > 2:
            score += 0.1
        
        sequential_markers = ['first', 'second', 'third', 'fourth', 'fifth', 'then', 'next', 'finally']
        sequential_count = sum(1 for word in words if word.lower() in sequential_markers)
        
        if sequential_count > len(words) * 0.02:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_causality_patterns(self, text: str) -> float:
        """Detecta patrones de causalidad típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        causality_patterns = [
            r'\b(?:because|since|as|due to|owing to|as a result of)',
            r'\b(?:therefore|thus|hence|consequently|as a result|accordingly)',
            r'\b(?:causes|caused by|leads to|results in|brings about)',
            r'\b(?:if\s+\w+\s+then|when\s+\w+\s+then|whenever\s+\w+\s+then)',
            r'\b(?:the reason why|the reason that|why\s+\w+\s+is)'
        ]
        
        causality_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in causality_patterns)
        
        if causality_count > len(sentences) * 0.3:
            score += 0.3
        elif causality_count > len(sentences) * 0.2:
            score += 0.2
        elif causality_count > 2:
            score += 0.1
        
        effect_markers = ['therefore', 'thus', 'hence', 'consequently', 'as a result', 'accordingly']
        effect_count = sum(1 for word in words if word.lower() in effect_markers)
        
        if effect_count > len(sentences) * 0.15:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_modal_verb_patterns(self, text: str) -> float:
        """Analiza patrones de verbos modales típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        modal_verbs = ['can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would']
        modal_count = sum(1 for word in words if word.lower() in modal_verbs)
        
        if modal_count > len(sentences) * 0.4:
            score += 0.3
        elif modal_count > len(sentences) * 0.3:
            score += 0.2
        elif modal_count > len(sentences) * 0.2:
            score += 0.1
        
        modal_patterns = [
            r'\b(?:can|could|may|might|must|should|would)\s+be',
            r'\b(?:can|could|may|might|must|should|would)\s+have',
            r'\b(?:can|could|may|might|must|should|would)\s+not',
            r'\b(?:it\s+(?:can|could|may|might|must|should|would))'
        ]
        
        modal_pattern_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in modal_patterns)
        if modal_pattern_count > len(sentences) * 0.25:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_hedge_phrase_patterns(self, text: str) -> float:
        """Detecta patrones de frases de hedging típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        hedge_phrases = [
            r'\b(?:it seems|it appears|it would seem|it might seem)',
            r'\b(?:perhaps|maybe|possibly|probably|likely|unlikely)',
            r'\b(?:to some extent|in some way|in a sense|to a certain degree)',
            r'\b(?:might be|could be|may be|seems to be|appears to be)',
            r'\b(?:generally|usually|typically|often|sometimes|occasionally)'
        ]
        
        hedge_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in hedge_phrases)
        
        if hedge_count > len(sentences) * 0.3:
            score += 0.3
        elif hedge_count > len(sentences) * 0.2:
            score += 0.2
        elif hedge_count > 2:
            score += 0.1
        
        uncertainty_markers = ['perhaps', 'maybe', 'possibly', 'probably', 'likely', 'unlikely', 'might', 'could']
        uncertainty_count = sum(1 for word in words if word.lower() in uncertainty_markers)
        
        if uncertainty_count > len(words) * 0.02:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_relative_clause_patterns(self, text: str) -> float:
        """Analiza patrones de cláusulas relativas típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        relative_clause_patterns = [
            r'\b(?:which|that|who|whom|whose|where|when)\s+\w+',
            r'\b(?:,\s+which|,\s+that|,\s+who|,\s+whom|,\s+whose)',
            r'\b(?:of which|of whom|of whose|in which|at which|on which)',
            r'\b(?:the\s+\w+\s+(?:which|that|who|whom|whose))'
        ]
        
        relative_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in relative_clause_patterns)
        
        if relative_count > len(sentences) * 0.4:
            score += 0.3
        elif relative_count > len(sentences) * 0.3:
            score += 0.2
        elif relative_count > len(sentences) * 0.2:
            score += 0.1
        
        relative_pronouns = ['which', 'that', 'who', 'whom', 'whose', 'where', 'when']
        relative_pronoun_count = sum(1 for word in words if word.lower() in relative_pronouns)
        
        if relative_pronoun_count > len(words) * 0.03:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_infinitive_patterns(self, text: str) -> float:
        """Detecta patrones de infinitivos típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        infinitive_patterns = [
            r'\b(?:to\s+\w+\s+the|to\s+\w+\s+a|to\s+\w+\s+an)',
            r'\b(?:in order to|so as to|in an attempt to|with the aim to)',
            r'\b(?:it is\s+\w+\s+to|it was\s+\w+\s+to|it would be\s+\w+\s+to)',
            r'\b(?:the\s+\w+\s+to\s+\w+)',
            r'\b(?:to\s+\w+\s+and\s+to\s+\w+)'
        ]
        
        infinitive_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in infinitive_patterns)
        
        if infinitive_count > len(sentences) * 0.3:
            score += 0.3
        elif infinitive_count > len(sentences) * 0.2:
            score += 0.2
        elif infinitive_count > 2:
            score += 0.1
        
        to_infinitive = len(re.findall(r'\bto\s+\w+', text, re.IGNORECASE))
        if to_infinitive > len(words) * 0.05:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_gerund_patterns(self, text: str) -> float:
        """Analiza patrones de gerundios típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        gerund_patterns = [
            r'\b\w+ing\s+(?:the|a|an|this|that|these|those)',
            r'\b(?:by|while|when|after|before|upon)\s+\w+ing',
            r'\b(?:is|are|was|were|being)\s+\w+ing',
            r'\b(?:start|begin|continue|keep|stop|finish)\s+\w+ing',
            r'\b(?:without|instead of|in addition to)\s+\w+ing'
        ]
        
        gerund_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in gerund_patterns)
        
        if gerund_count > len(sentences) * 0.3:
            score += 0.3
        elif gerund_count > len(sentences) * 0.2:
            score += 0.2
        elif gerund_count > 2:
            score += 0.1
        
        ing_words = len(re.findall(r'\b\w+ing\b', text, re.IGNORECASE))
        if ing_words > len(words) * 0.04:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_participle_patterns(self, text: str) -> float:
        """Detecta patrones de participios típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        participle_patterns = [
            r'\b\w+ed\s+(?:by|with|in|on|at|from|to)',
            r'\b\w+en\s+(?:by|with|in|on|at|from|to)',
            r'\b(?:having|being|having been)\s+\w+ed',
            r'\b(?:having|being|having been)\s+\w+en',
            r'\b(?:the|a|an)\s+\w+ed\s+\w+',
            r'\b(?:the|a|an)\s+\w+en\s+\w+'
        ]
        
        participle_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in participle_patterns)
        
        if participle_count > len(sentences) * 0.3:
            score += 0.3
        elif participle_count > len(sentences) * 0.2:
            score += 0.2
        elif participle_count > 2:
            score += 0.1
        
        past_participle = len(re.findall(r'\b\w+ed\b', text, re.IGNORECASE))
        past_participle_en = len(re.findall(r'\b\w+en\b', text, re.IGNORECASE))
        
        if (past_participle + past_participle_en) > len(words) * 0.05:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_subjunctive_patterns(self, text: str) -> float:
        """Analiza patrones de subjuntivo típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        subjunctive_patterns = [
            r'\b(?:if\s+\w+\s+were|if\s+\w+\s+had|if\s+\w+\s+could)',
            r'\b(?:it\s+is\s+important\s+that|it\s+is\s+crucial\s+that|it\s+is\s+essential\s+that)',
            r'\b(?:suggest|recommend|insist|demand|require)\s+that',
            r'\b(?:wish|hope|prefer|desire)\s+(?:that|to)',
            r'\b(?:as\s+if|as\s+though|even\s+if|even\s+though)'
        ]
        
        subjunctive_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in subjunctive_patterns)
        
        if subjunctive_count > len(sentences) * 0.2:
            score += 0.3
        elif subjunctive_count > len(sentences) * 0.1:
            score += 0.2
        elif subjunctive_count > 1:
            score += 0.1
        
        conditional_subjunctive = len(re.findall(r'\bif\s+\w+\s+(?:were|had|could|would|should)', text, re.IGNORECASE))
        if conditional_subjunctive > len(sentences) * 0.15:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_article_patterns(self, text: str) -> float:
        """Detecta patrones de uso de artículos típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        
        if len(words) < 30:
            return 0.0
        
        articles = ['a', 'an', 'the']
        article_count = sum(1 for word in words if word.lower() in articles)
        
        if article_count > len(words) * 0.12:
            score += 0.2
        elif article_count > len(words) * 0.10:
            score += 0.15
        elif article_count > len(words) * 0.08:
            score += 0.1
        
        article_patterns = [
            r'\b(?:the|a|an)\s+\w+\s+(?:the|a|an)\s+\w+',
            r'\b(?:the|a|an)\s+\w+\s+(?:the|a|an)\s+\w+\s+(?:the|a|an)',
            r'\b(?:the|a|an)\s+\w+\s+and\s+(?:the|a|an)\s+\w+'
        ]
        
        article_pattern_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in article_patterns)
        if article_pattern_count > len(words) * 0.02:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_preposition_patterns(self, text: str) -> float:
        """Analiza patrones de preposiciones típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        prepositions = ['in', 'on', 'at', 'by', 'for', 'with', 'from', 'to', 'of', 'about', 'into', 'onto', 'upon', 'within', 'without', 'through', 'during', 'under', 'over', 'above', 'below', 'between', 'among', 'beside', 'besides', 'beyond', 'across', 'against', 'along', 'around', 'behind', 'beneath', 'inside', 'outside', 'throughout', 'toward', 'towards', 'underneath', 'unlike', 'until', 'via', 'within']
        preposition_count = sum(1 for word in words if word.lower() in prepositions)
        
        if preposition_count > len(words) * 0.12:
            score += 0.3
        elif preposition_count > len(words) * 0.10:
            score += 0.2
        elif preposition_count > len(words) * 0.08:
            score += 0.1
        
        complex_prepositions = [
            r'\b(?:in\s+accordance\s+with|in\s+addition\s+to|in\s+comparison\s+to)',
            r'\b(?:in\s+conjunction\s+with|in\s+connection\s+with|in\s+contrast\s+to)',
            r'\b(?:in\s+relation\s+to|in\s+respect\s+of|in\s+terms\s+of)',
            r'\b(?:with\s+regard\s+to|with\s+respect\s+to|with\s+reference\s+to)',
            r'\b(?:on\s+behalf\s+of|on\s+account\s+of|on\s+top\s+of)'
        ]
        
        complex_prep_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in complex_prepositions)
        if complex_prep_count > len(sentences) * 0.15:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_conjunction_patterns(self, text: str) -> float:
        """Detecta patrones de conjunciones típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        coordinating_conjunctions = ['and', 'but', 'or', 'nor', 'for', 'so', 'yet']
        subordinating_conjunctions = ['although', 'though', 'because', 'since', 'if', 'unless', 'while', 'when', 'where', 'whereas', 'whether', 'until', 'before', 'after', 'as', 'so', 'that', 'once', 'provided', 'supposing']
        
        coord_count = sum(1 for word in words if word.lower() in coordinating_conjunctions)
        subord_count = sum(1 for word in words if word.lower() in subordinating_conjunctions)
        
        if (coord_count + subord_count) > len(words) * 0.08:
            score += 0.3
        elif (coord_count + subord_count) > len(words) * 0.06:
            score += 0.2
        elif (coord_count + subord_count) > len(words) * 0.04:
            score += 0.1
        
        conjunction_patterns = [
            r'\b(?:and\s+also|and\s+moreover|and\s+furthermore)',
            r'\b(?:but\s+also|but\s+however|but\s+nevertheless)',
            r'\b(?:not\s+only\s+but\s+also|both\s+and|either\s+or|neither\s+nor)'
        ]
        
        conj_pattern_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in conjunction_patterns)
        if conj_pattern_count > 1:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_determiner_patterns(self, text: str) -> float:
        """Analiza patrones de determinantes típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        
        if len(words) < 30:
            return 0.0
        
        determiners = ['this', 'that', 'these', 'those', 'all', 'each', 'every', 'both', 'either', 'neither', 'some', 'any', 'no', 'many', 'much', 'few', 'little', 'several', 'various', 'other', 'another', 'such', 'same', 'own']
        determiner_count = sum(1 for word in words if word.lower() in determiners)
        
        if determiner_count > len(words) * 0.05:
            score += 0.3
        elif determiner_count > len(words) * 0.04:
            score += 0.2
        elif determiner_count > len(words) * 0.03:
            score += 0.1
        
        determiner_patterns = [
            r'\b(?:this|that|these|those)\s+\w+\s+(?:this|that|these|those)',
            r'\b(?:all|each|every)\s+\w+\s+(?:all|each|every)',
            r'\b(?:some|any|no)\s+\w+\s+(?:some|any|no)'
        ]
        
        determiner_pattern_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in determiner_patterns)
        if determiner_pattern_count > 1:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_pronoun_reference_patterns(self, text: str) -> float:
        """Detecta patrones de referencia pronominal típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        pronoun_reference_patterns = [
            r'\b(?:this|that|these|those|it|they|them|their|its)\s+(?:is|are|was|were|has|have|had)',
            r'\b(?:which|who|whom|whose|where|when)\s+(?:is|are|was|were|has|have|had)',
            r'\b(?:he|she|it|they)\s+(?:is|are|was|were|has|have|had)',
            r'\b(?:this|that|these|those)\s+\w+\s+(?:which|that|who)'
        ]
        
        pronoun_ref_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in pronoun_reference_patterns)
        
        if pronoun_ref_count > len(sentences) * 0.3:
            score += 0.3
        elif pronoun_ref_count > len(sentences) * 0.2:
            score += 0.2
        elif pronoun_ref_count > len(sentences) * 0.1:
            score += 0.1
        
        ambiguous_references = len(re.findall(r'\b(?:this|that|these|those|it|they)\s+\w+', text, re.IGNORECASE))
        if ambiguous_references > len(sentences) * 0.25:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_adverb_patterns(self, text: str) -> float:
        """Analiza patrones de adverbios típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        common_adverbs = ['very', 'quite', 'rather', 'extremely', 'highly', 'particularly', 'especially', 'notably', 'remarkably', 'significantly', 'substantially', 'considerably', 'relatively', 'comparatively', 'fairly', 'somewhat', 'slightly', 'moderately', 'reasonably', 'sufficiently', 'adequately', 'appropriately', 'effectively', 'efficiently', 'successfully', 'carefully', 'clearly', 'obviously', 'evidently', 'apparently', 'presumably', 'probably', 'possibly', 'certainly', 'definitely', 'absolutely', 'completely', 'entirely', 'totally', 'fully', 'perfectly', 'exactly', 'precisely', 'accurately', 'correctly', 'properly', 'adequately']
        adverb_count = sum(1 for word in words if word.lower() in common_adverbs)
        
        if adverb_count > len(words) * 0.06:
            score += 0.3
        elif adverb_count > len(words) * 0.04:
            score += 0.2
        elif adverb_count > len(words) * 0.02:
            score += 0.1
        
        adverb_patterns = [
            r'\b(?:very|extremely|highly|particularly|especially)\s+\w+',
            r'\b(?:quite|rather|somewhat|fairly|relatively)\s+\w+',
            r'\b(?:clearly|obviously|evidently|apparently|presumably)',
            r'\b(?:certainly|definitely|absolutely|completely|entirely)'
        ]
        
        adverb_pattern_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in adverb_patterns)
        if adverb_pattern_count > len(sentences) * 0.3:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_adjective_patterns(self, text: str) -> float:
        """Detecta patrones de adjetivos típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        common_adjectives = ['important', 'crucial', 'essential', 'vital', 'significant', 'relevant', 'appropriate', 'suitable', 'effective', 'efficient', 'successful', 'comprehensive', 'thorough', 'detailed', 'extensive', 'considerable', 'substantial', 'notable', 'remarkable', 'noteworthy', 'prominent', 'distinct', 'unique', 'particular', 'specific', 'general', 'common', 'typical', 'usual', 'normal', 'standard', 'regular', 'ordinary', 'average', 'basic', 'fundamental', 'primary', 'main', 'major', 'minor', 'key', 'central', 'principal', 'chief', 'main', 'leading', 'primary']
        adjective_count = sum(1 for word in words if word.lower() in common_adjectives)
        
        if adjective_count > len(words) * 0.08:
            score += 0.3
        elif adjective_count > len(words) * 0.06:
            score += 0.2
        elif adjective_count > len(words) * 0.04:
            score += 0.1
        
        adjective_patterns = [
            r'\b(?:very|extremely|highly|particularly|especially)\s+\w+',
            r'\b(?:more|most|less|least)\s+\w+',
            r'\b(?:the\s+most|the\s+least|the\s+best|the\s+worst)',
            r'\b(?:important|crucial|essential|vital|significant)\s+\w+'
        ]
        
        adj_pattern_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in adjective_patterns)
        if adj_pattern_count > len(sentences) * 0.25:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_noun_patterns(self, text: str) -> float:
        """Analiza patrones de sustantivos típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        abstract_nouns = ['importance', 'significance', 'relevance', 'effectiveness', 'efficiency', 'success', 'achievement', 'accomplishment', 'progress', 'development', 'improvement', 'enhancement', 'advancement', 'growth', 'expansion', 'increase', 'decrease', 'reduction', 'decline', 'change', 'modification', 'alteration', 'variation', 'difference', 'similarity', 'comparison', 'contrast', 'relationship', 'connection', 'association', 'correlation', 'interaction', 'influence', 'impact', 'effect', 'consequence', 'result', 'outcome', 'conclusion', 'summary', 'overview', 'analysis', 'evaluation', 'assessment', 'examination', 'investigation', 'research', 'study', 'analysis', 'review']
        abstract_noun_count = sum(1 for word in words if word.lower() in abstract_nouns)
        
        if abstract_noun_count > len(words) * 0.05:
            score += 0.3
        elif abstract_noun_count > len(words) * 0.04:
            score += 0.2
        elif abstract_noun_count > len(words) * 0.03:
            score += 0.1
        
        noun_patterns = [
            r'\b(?:the|a|an)\s+\w+\s+of\s+\w+',
            r'\b(?:the|a|an)\s+\w+\s+and\s+\w+',
            r'\b(?:the|a|an)\s+\w+\s+or\s+\w+',
            r'\b(?:the|a|an)\s+\w+\s+,\s+\w+\s+and\s+\w+'
        ]
        
        noun_pattern_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in noun_patterns)
        if noun_pattern_count > len(sentences) * 0.2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_verb_patterns(self, text: str) -> float:
        """Detecta patrones de verbos típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        common_verbs = ['is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'get', 'gets', 'got', 'getting', 'make', 'makes', 'made', 'making', 'take', 'takes', 'took', 'taking', 'give', 'gives', 'gave', 'giving', 'go', 'goes', 'went', 'going', 'come', 'comes', 'came', 'coming', 'see', 'sees', 'saw', 'seeing', 'know', 'knows', 'knew', 'knowing', 'think', 'thinks', 'thought', 'thinking', 'say', 'says', 'said', 'saying', 'tell', 'tells', 'told', 'telling', 'ask', 'asks', 'asked', 'asking', 'try', 'tries', 'tried', 'trying', 'use', 'uses', 'used', 'using', 'find', 'finds', 'found', 'finding', 'want', 'wants', 'wanted', 'wanting', 'need', 'needs', 'needed', 'needing', 'work', 'works', 'worked', 'working', 'call', 'calls', 'called', 'calling', 'show', 'shows', 'showed', 'showing', 'move', 'moves', 'moved', 'moving', 'live', 'lives', 'lived', 'living', 'believe', 'believes', 'believed', 'believing', 'bring', 'brings', 'brought', 'bringing', 'happen', 'happens', 'happened', 'happening', 'write', 'writes', 'wrote', 'writing', 'provide', 'provides', 'provided', 'providing', 'sit', 'sits', 'sat', 'sitting', 'stand', 'stands', 'stood', 'standing', 'lose', 'loses', 'lost', 'losing', 'pay', 'pays', 'paid', 'paying', 'meet', 'meets', 'met', 'meeting', 'include', 'includes', 'included', 'including', 'continue', 'continues', 'continued', 'continuing', 'set', 'sets', 'setting', 'learn', 'learns', 'learned', 'learning', 'change', 'changes', 'changed', 'changing', 'lead', 'leads', 'led', 'leading', 'understand', 'understands', 'understood', 'understanding', 'watch', 'watches', 'watched', 'watching', 'follow', 'follows', 'followed', 'following', 'stop', 'stops', 'stopped', 'stopping', 'create', 'creates', 'created', 'creating', 'speak', 'speaks', 'spoke', 'speaking', 'read', 'reads', 'reading', 'allow', 'allows', 'allowed', 'allowing', 'add', 'adds', 'added', 'adding', 'spend', 'spends', 'spent', 'spending', 'grow', 'grows', 'grew', 'growing', 'open', 'opens', 'opened', 'opening', 'walk', 'walks', 'walked', 'walking', 'win', 'wins', 'won', 'winning', 'offer', 'offers', 'offered', 'offering', 'remember', 'remembers', 'remembered', 'remembering', 'love', 'loves', 'loved', 'loving', 'consider', 'considers', 'considered', 'considering', 'appear', 'appears', 'appeared', 'appearing', 'buy', 'buys', 'bought', 'buying', 'wait', 'waits', 'waited', 'waiting', 'serve', 'serves', 'served', 'serving', 'die', 'dies', 'died', 'dying', 'send', 'sends', 'sent', 'sending', 'build', 'builds', 'built', 'building', 'stay', 'stays', 'stayed', 'staying', 'fall', 'falls', 'fell', 'falling', 'cut', 'cuts', 'cutting', 'reach', 'reaches', 'reached', 'reaching', 'kill', 'kills', 'killed', 'killing', 'raise', 'raises', 'raised', 'raising', 'pass', 'passes', 'passed', 'passing', 'sell', 'sells', 'sold', 'selling', 'decide', 'decides', 'decided', 'deciding', 'return', 'returns', 'returned', 'returning', 'explain', 'explains', 'explained', 'explaining', 'develop', 'develops', 'developed', 'developing', 'carry', 'carries', 'carried', 'carrying', 'break', 'breaks', 'broke', 'breaking', 'receive', 'receives', 'received', 'receiving', 'agree', 'agrees', 'agreed', 'agreeing', 'support', 'supports', 'supported', 'supporting', 'hit', 'hits', 'hitting', 'produce', 'produces', 'produced', 'producing', 'eat', 'eats', 'ate', 'eating', 'cover', 'covers', 'covered', 'covering', 'catch', 'catches', 'caught', 'catching']
        verb_count = sum(1 for word in words if word.lower() in common_verbs)
        
        if verb_count > len(words) * 0.15:
            score += 0.2
        elif verb_count > len(words) * 0.12:
            score += 0.15
        elif verb_count > len(words) * 0.10:
            score += 0.1
        
        verb_patterns = [
            r'\b(?:is|are|was|were)\s+\w+ing',
            r'\b(?:has|have|had)\s+\w+ed',
            r'\b(?:will|would|can|could|should|must)\s+\w+',
            r'\b(?:to\s+be|to\s+have|to\s+do|to\s+get|to\s+make)'
        ]
        
        verb_pattern_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in verb_patterns)
        if verb_pattern_count > len(sentences) * 0.3:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_sentence_length_patterns(self, text: str) -> float:
        """Analiza patrones de longitud de oraciones típicos de IA - NUEVO"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 5:
            return 0.0
        
        sentence_lengths = [len(s.split()) for s in sentences]
        avg_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
        
        if avg_length > 25:
            score += 0.3
        elif avg_length > 20:
            score += 0.2
        elif avg_length > 15:
            score += 0.1
        
        length_variance = sum((l - avg_length) ** 2 for l in sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
        std_dev = length_variance ** 0.5
        
        if std_dev < 5:
            score += 0.3
        elif std_dev < 8:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_paragraph_structure_patterns(self, text: str) -> float:
        """Detecta patrones de estructura de párrafos típicos de IA - NUEVO"""
        score = 0.0
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if len(paragraphs) < 2:
            return 0.0
        
        paragraph_lengths = [len(p.split()) for p in paragraphs]
        avg_para_length = sum(paragraph_lengths) / len(paragraph_lengths) if paragraph_lengths else 0
        
        if avg_para_length > 150:
            score += 0.2
        elif avg_para_length > 100:
            score += 0.15
        
        para_variance = sum((l - avg_para_length) ** 2 for l in paragraph_lengths) / len(paragraph_lengths) if paragraph_lengths else 0
        para_std_dev = para_variance ** 0.5
        
        if para_std_dev < 30:
            score += 0.3
        elif para_std_dev < 50:
            score += 0.2
        
        first_sentence_patterns = [
            r'^(?:In|The|This|That|These|Those|It|There|Here)',
            r'^(?:To|For|With|By|From|At|On|In)',
            r'^(?:When|Where|Why|How|What|Who|Which)'
        ]
        
        first_sentence_matches = sum(1 for para in paragraphs if any(re.match(pattern, para, re.IGNORECASE) for pattern in first_sentence_patterns))
        if first_sentence_matches > len(paragraphs) * 0.7:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_punctuation_patterns(self, text: str) -> float:
        """Analiza patrones de puntuación típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        comma_count = text.count(',')
        semicolon_count = text.count(';')
        colon_count = text.count(':')
        dash_count = text.count('-') + text.count('—') + text.count('–')
        parenthesis_count = text.count('(') + text.count(')')
        
        total_punctuation = comma_count + semicolon_count + colon_count + dash_count + parenthesis_count
        
        if total_punctuation > len(words) * 0.15:
            score += 0.3
        elif total_punctuation > len(words) * 0.12:
            score += 0.2
        elif total_punctuation > len(words) * 0.10:
            score += 0.1
        
        if comma_count > len(sentences) * 2:
            score += 0.2
        
        if semicolon_count > len(sentences) * 0.3:
            score += 0.15
        
        return min(score, 1.0)
    
    def _detect_ai_capitalization_patterns(self, text: str) -> float:
        """Detecta patrones de capitalización típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        capitalized_words = sum(1 for word in words if word and word[0].isupper() and word.isalpha())
        
        if capitalized_words > len(words) * 0.15:
            score += 0.2
        elif capitalized_words > len(words) * 0.12:
            score += 0.15
        elif capitalized_words > len(words) * 0.10:
            score += 0.1
        
        all_caps_words = sum(1 for word in words if word.isupper() and len(word) > 1)
        if all_caps_words > len(words) * 0.02:
            score += 0.15
        
        proper_noun_patterns = [
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+',
            r'\b(?:the|The)\s+[A-Z][a-z]+\s+of\s+[A-Z][a-z]+',
            r'\b(?:of|in|on|at|by|for|with|from|to)\s+[A-Z][a-z]+'
        ]
        
        proper_noun_count = sum(len(re.findall(pattern, text)) for pattern in proper_noun_patterns)
        if proper_noun_count > len(sentences) * 0.3:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_word_frequency_patterns(self, text: str) -> float:
        """Analiza patrones de frecuencia de palabras típicos de IA - NUEVO"""
        score = 0.0
        words = [w.lower().strip('.,!?;:()[]{}"\'') for w in text.split()]
        words = [w for w in words if w]
        
        if len(words) < 50:
            return 0.0
        
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        total_words = len(words)
        unique_words = len(word_freq)
        type_token_ratio = unique_words / total_words if total_words > 0 else 0
        
        if type_token_ratio < 0.4:
            score += 0.3
        elif type_token_ratio < 0.5:
            score += 0.2
        elif type_token_ratio < 0.6:
            score += 0.1
        
        high_freq_words = [word for word, count in word_freq.items() if count > total_words * 0.05]
        if len(high_freq_words) > 5:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_phrase_repetition_patterns(self, text: str) -> float:
        """Detecta patrones de repetición de frases típicos de IA - NUEVO"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 5:
            return 0.0
        
        sentence_lower = [s.lower() for s in sentences]
        sentence_freq = {}
        for sentence in sentence_lower:
            sentence_freq[sentence] = sentence_freq.get(sentence, 0) + 1
        
        repeated_sentences = sum(1 for count in sentence_freq.values() if count > 1)
        if repeated_sentences > len(sentences) * 0.1:
            score += 0.3
        elif repeated_sentences > 0:
            score += 0.15
        
        phrase_patterns = [
            r'\b(?:it is|it\'s|this is|that is|these are|those are)',
            r'\b(?:in order to|so as to|with the aim of)',
            r'\b(?:it should be noted|it is important|it is worth noting)'
        ]
        
        phrase_counts = {}
        for pattern in phrase_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                phrase_counts[match.lower()] = phrase_counts.get(match.lower(), 0) + 1
        
        repeated_phrases = sum(1 for count in phrase_counts.values() if count > 2)
        if repeated_phrases > 2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_semantic_density_patterns(self, text: str) -> float:
        """Analiza patrones de densidad semántica típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 50:
            return 0.0
        
        content_words = []
        function_words = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'by', 'for', 'with', 'from', 'to', 'of', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'can', 'could', 'should', 'may', 'might', 'must', 'this', 'that', 'these', 'those', 'it', 'they', 'them', 'their', 'its', 'he', 'she', 'him', 'her', 'his', 'hers', 'we', 'us', 'our', 'ours', 'you', 'your', 'yours', 'i', 'me', 'my', 'mine']
        
        for word in words:
            word_lower = word.lower().strip('.,!?;:()[]{}"\'')
            if word_lower and word_lower not in function_words:
                content_words.append(word_lower)
        
        content_word_ratio = len(content_words) / len(words) if len(words) > 0 else 0
        
        if content_word_ratio < 0.5:
            score += 0.3
        elif content_word_ratio < 0.6:
            score += 0.2
        elif content_word_ratio < 0.7:
            score += 0.1
        
        unique_content_words = len(set(content_words))
        content_diversity = unique_content_words / len(content_words) if len(content_words) > 0 else 0
        
        if content_diversity < 0.5:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_coherence_markers_patterns(self, text: str) -> float:
        """Detecta patrones de marcadores de coherencia típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        coherence_markers = [
            r'\b(?:first|second|third|fourth|fifth|finally|lastly|last)',
            r'\b(?:in addition|furthermore|moreover|additionally|also|besides)',
            r'\b(?:however|nevertheless|nonetheless|on the other hand|conversely)',
            r'\b(?:therefore|thus|hence|consequently|as a result|accordingly)',
            r'\b(?:for example|for instance|such as|namely|specifically)',
            r'\b(?:in conclusion|to summarize|in summary|overall|ultimately)',
            r'\b(?:in other words|that is|that is to say|to put it differently)',
            r'\b(?:on the one hand|on the other hand)',
            r'\b(?:similarly|likewise|in the same way|equally)',
            r'\b(?:in contrast|on the contrary|conversely|by contrast)'
        ]
        
        coherence_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in coherence_markers)
        
        if coherence_count > len(sentences) * 0.4:
            score += 0.3
        elif coherence_count > len(sentences) * 0.3:
            score += 0.2
        elif coherence_count > len(sentences) * 0.2:
            score += 0.1
        
        transition_density = coherence_count / len(sentences) if len(sentences) > 0 else 0
        if transition_density > 0.5:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_lexical_sophistication_patterns(self, text: str) -> float:
        """Analiza patrones de sofisticación léxica típicos de IA - NUEVO"""
        score = 0.0
        words = [w.lower().strip('.,!?;:()[]{}"\'') for w in text.split()]
        words = [w for w in words if w and len(w) > 2]
        
        if len(words) < 50:
            return 0.0
        
        sophisticated_words = []
        common_words = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'by', 'for', 'with', 'from', 'to', 'of', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'can', 'could', 'should', 'may', 'might', 'must', 'this', 'that', 'these', 'those', 'it', 'they', 'them', 'their', 'its', 'he', 'she', 'him', 'her', 'his', 'hers', 'we', 'us', 'our', 'ours', 'you', 'your', 'yours', 'i', 'me', 'my', 'mine', 'get', 'got', 'make', 'made', 'take', 'took', 'go', 'went', 'come', 'came', 'see', 'saw', 'know', 'knew', 'think', 'thought', 'say', 'said', 'tell', 'told', 'ask', 'asked', 'try', 'tried', 'use', 'used', 'find', 'found', 'want', 'wanted', 'need', 'needed', 'work', 'worked', 'call', 'called', 'show', 'showed', 'move', 'moved', 'live', 'lived', 'believe', 'believed', 'bring', 'brought', 'happen', 'happened', 'write', 'wrote', 'provide', 'provided', 'sit', 'sat', 'stand', 'stood', 'lose', 'lost', 'pay', 'paid', 'meet', 'met', 'include', 'included', 'continue', 'continued', 'set', 'learn', 'learned', 'change', 'changed', 'lead', 'led', 'understand', 'understood', 'watch', 'watched', 'follow', 'followed', 'stop', 'stopped', 'create', 'created', 'speak', 'spoke', 'read', 'allow', 'allowed', 'add', 'added', 'spend', 'spent', 'grow', 'grew', 'open', 'opened', 'walk', 'walked', 'win', 'won', 'offer', 'offered', 'remember', 'remembered', 'love', 'loved', 'consider', 'considered', 'appear', 'appeared', 'buy', 'bought', 'wait', 'waited', 'serve', 'served', 'die', 'died', 'send', 'sent', 'build', 'built', 'stay', 'stayed', 'fall', 'fell', 'cut', 'reach', 'reached', 'kill', 'killed', 'raise', 'raised', 'pass', 'passed', 'sell', 'sold', 'decide', 'decided', 'return', 'returned', 'explain', 'explained', 'develop', 'developed', 'carry', 'carried', 'break', 'broke', 'receive', 'received', 'agree', 'agreed', 'support', 'supported', 'hit', 'produce', 'produced', 'eat', 'ate', 'cover', 'covered', 'catch', 'caught']
        
        for word in words:
            if word not in common_words and len(word) > 5:
                sophisticated_words.append(word)
        
        sophisticated_ratio = len(sophisticated_words) / len(words) if len(words) > 0 else 0
        
        if sophisticated_ratio > 0.3:
            score += 0.2
        elif sophisticated_ratio > 0.2:
            score += 0.15
        elif sophisticated_ratio < 0.1:
            score += 0.1
        
        avg_word_length = sum(len(w) for w in words) / len(words) if words else 0
        if avg_word_length > 6:
            score += 0.15
        elif avg_word_length < 4:
            score += 0.1
        
        return min(score, 1.0)
    
    def _detect_ai_formality_patterns(self, text: str) -> float:
        """Detecta patrones de formalidad típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        formal_markers = [
            r'\b(?:furthermore|moreover|additionally|consequently|therefore|thus|hence)',
            r'\b(?:it is important to note|it should be noted|it is worth noting|it is worth mentioning)',
            r'\b(?:in order to|so as to|with the aim of|with the purpose of)',
            r'\b(?:with regard to|with respect to|with reference to|in relation to)',
            r'\b(?:in accordance with|in compliance with|in conformity with)',
            r'\b(?:it can be observed|it can be seen|it can be noted|it can be concluded)',
            r'\b(?:it is evident that|it is clear that|it is obvious that|it is apparent that)',
            r'\b(?:it is necessary to|it is essential to|it is crucial to|it is vital to)',
            r'\b(?:it is recommended that|it is suggested that|it is advised that)',
            r'\b(?:it should be emphasized|it should be stressed|it should be highlighted)'
        ]
        
        formal_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in formal_markers)
        
        if formal_count > len(sentences) * 0.3:
            score += 0.3
        elif formal_count > len(sentences) * 0.2:
            score += 0.2
        elif formal_count > len(sentences) * 0.1:
            score += 0.1
        
        contractions = len(re.findall(r'\b(?:don\'t|doesn\'t|didn\'t|won\'t|wouldn\'t|couldn\'t|shouldn\'t|can\'t|cannot|isn\'t|aren\'t|wasn\'t|weren\'t|hasn\'t|haven\'t|hadn\'t|it\'s|that\'s|there\'s|here\'s|what\'s|who\'s|where\'s|when\'s|why\'s|how\'s|i\'m|you\'re|we\'re|they\'re|he\'s|she\'s|i\'ve|you\'ve|we\'ve|they\'ve|i\'d|you\'d|we\'d|they\'d|he\'d|she\'d)', text, re.IGNORECASE))
        if contractions == 0 and len(words) > 50:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_register_patterns(self, text: str) -> float:
        """Analiza patrones de registro típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        academic_register = [
            r'\b(?:according to|based on|in terms of|with regard to|with respect to)',
            r'\b(?:it has been|it can be|it should be|it must be|it will be)',
            r'\b(?:in order to|so as to|with the aim of|with the purpose of)',
            r'\b(?:it is important|it is necessary|it is essential|it is crucial)',
            r'\b(?:it should be noted|it is worth noting|it is worth mentioning)',
            r'\b(?:in addition to|furthermore|moreover|additionally)',
            r'\b(?:it can be concluded|it can be observed|it can be seen)',
            r'\b(?:it is evident|it is clear|it is obvious|it is apparent)'
        ]
        
        academic_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in academic_register)
        
        if academic_count > len(sentences) * 0.3:
            score += 0.3
        elif academic_count > len(sentences) * 0.2:
            score += 0.2
        elif academic_count > len(sentences) * 0.1:
            score += 0.1
        
        informal_markers = ['gonna', 'wanna', 'gotta', 'lemme', 'gimme', 'ya', 'yeah', 'yep', 'nope', 'nah', 'haha', 'lol', 'omg', 'wtf', 'btw', 'imo', 'fyi', 'tbh', 'idk', 'ikr', 'smh', 'tbh', 'irl', 'fml', 'rofl', 'lmao']
        informal_count = sum(1 for word in words if word.lower() in informal_markers)
        
        if informal_count == 0 and academic_count > 0:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_discourse_markers_patterns(self, text: str) -> float:
        """Detecta patrones de marcadores discursivos típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        discourse_markers = [
            r'\b(?:well|now|so|then|okay|ok|right|alright|anyway|anyways)',
            r'\b(?:you know|i mean|like|sort of|kind of|you see)',
            r'\b(?:actually|basically|literally|seriously|honestly|frankly)',
            r'\b(?:in fact|as a matter of fact|to be honest|to tell the truth)',
            r'\b(?:by the way|incidentally|speaking of|talking about)',
            r'\b(?:first of all|to begin with|to start with|for starters)',
            r'\b(?:in the end|at the end|finally|lastly|ultimately)',
            r'\b(?:in other words|that is|that is to say|i.e.|e.g.)',
            r'\b(?:for example|for instance|such as|namely|specifically)',
            r'\b(?:on the other hand|on the contrary|conversely|by contrast)'
        ]
        
        discourse_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in discourse_markers)
        
        if discourse_count > len(sentences) * 0.4:
            score += 0.3
        elif discourse_count > len(sentences) * 0.3:
            score += 0.2
        elif discourse_count > len(sentences) * 0.2:
            score += 0.1
        
        if discourse_count == 0 and len(sentences) > 10:
            score += 0.15
        
        return min(score, 1.0)
    
    def _analyze_ai_textual_cohesion_patterns(self, text: str) -> float:
        """Analiza patrones de cohesión textual típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 40:
            return 0.0
        
        cohesion_devices = [
            r'\b(?:this|that|these|those|it|they|them|their|its)\s+\w+',
            r'\b(?:the\s+former|the\s+latter|the\s+above|the\s+following)',
            r'\b(?:as\s+mentioned|as\s+stated|as\s+noted|as\s+discussed)',
            r'\b(?:in\s+this|in\s+that|in\s+these|in\s+those)',
            r'\b(?:of\s+this|of\s+that|of\s+these|of\s+those)'
        ]
        
        cohesion_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in cohesion_devices)
        
        if cohesion_count > len(sentences) * 0.4:
            score += 0.3
        elif cohesion_count > len(sentences) * 0.3:
            score += 0.2
        elif cohesion_count > len(sentences) * 0.2:
            score += 0.1
        
        lexical_chains = []
        for i in range(len(sentences) - 1):
            words1 = set(sentences[i].lower().split())
            words2 = set(sentences[i+1].lower().split())
            overlap = len(words1.intersection(words2))
            if overlap > 2:
                lexical_chains.append(overlap)
        
        if len(lexical_chains) > len(sentences) * 0.5:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_information_density_patterns(self, text: str) -> float:
        """Detecta patrones de densidad de información típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 50:
            return 0.0
        
        information_indicators = [
            r'\b(?:according to|based on|research shows|studies indicate|evidence suggests)',
            r'\b(?:it is estimated|it is calculated|it is determined|it is found)',
            r'\b(?:statistics show|data indicates|figures reveal|numbers suggest)',
            r'\b(?:research has|studies have|evidence has|data has)',
            r'\b(?:percentage|percent|ratio|proportion|rate|frequency)'
        ]
        
        info_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in information_indicators)
        
        if info_count > len(sentences) * 0.3:
            score += 0.3
        elif info_count > len(sentences) * 0.2:
            score += 0.2
        elif info_count > len(sentences) * 0.1:
            score += 0.1
        
        factual_phrases = ['according to', 'based on', 'research', 'study', 'studies', 'data', 'evidence', 'statistics', 'figures', 'numbers', 'percentage', 'percent', 'ratio', 'proportion', 'rate']
        factual_count = sum(1 for word in words if word.lower() in factual_phrases)
        
        if factual_count > len(words) * 0.03:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_hedging_density_patterns(self, text: str) -> float:
        """Analiza patrones de densidad de hedging típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        hedging_expressions = [
            r'\b(?:may|might|could|would|should|possibly|perhaps|maybe|probably|likely|unlikely)',
            r'\b(?:it seems|it appears|it would seem|it might seem|it could be)',
            r'\b(?:to some extent|in some way|in a sense|to a certain degree|in some cases)',
            r'\b(?:generally|usually|typically|often|sometimes|occasionally|rarely|seldom)',
            r'\b(?:suggest|indicate|imply|hint|point to|tend to|appear to|seem to)'
        ]
        
        hedging_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in hedging_expressions)
        
        if hedging_count > len(sentences) * 0.5:
            score += 0.3
        elif hedging_count > len(sentences) * 0.4:
            score += 0.2
        elif hedging_count > len(sentences) * 0.3:
            score += 0.1
        
        hedging_density = hedging_count / len(sentences) if len(sentences) > 0 else 0
        if hedging_density > 0.6:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_authorial_voice_patterns(self, text: str) -> float:
        """Detecta patrones de voz autorial típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        first_person_indicators = ['i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves']
        second_person_indicators = ['you', 'your', 'yours', 'yourself', 'yourselves']
        
        first_person_count = sum(1 for word in words if word.lower() in first_person_indicators)
        second_person_count = sum(1 for word in words if word.lower() in second_person_indicators)
        
        if first_person_count == 0 and second_person_count == 0 and len(words) > 100:
            score += 0.3
        elif first_person_count == 0 and len(words) > 50:
            score += 0.2
        
        personal_opinions = [
            r'\b(?:i think|i believe|i feel|i consider|i find|i see)',
            r'\b(?:in my opinion|from my perspective|in my view|to my mind)',
            r'\b(?:i would say|i would argue|i would suggest|i would recommend)'
        ]
        
        opinion_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in personal_opinions)
        if opinion_count == 0 and len(sentences) > 10:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_textual_variety_patterns(self, text: str) -> float:
        """Analiza patrones de variedad textual típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 50:
            return 0.0
        
        sentence_starters = [s.split()[0].lower() if s.split() else '' for s in sentences]
        unique_starters = len(set(sentence_starters))
        starter_variety = unique_starters / len(sentences) if len(sentences) > 0 else 0
        
        if starter_variety < 0.3:
            score += 0.3
        elif starter_variety < 0.5:
            score += 0.2
        elif starter_variety < 0.7:
            score += 0.1
        
        sentence_structures = []
        for sentence in sentences:
            if re.search(r'^[A-Z][^.!?]*\b(?:is|are|was|were)\b', sentence):
                sentence_structures.append('declarative')
            elif re.search(r'^[A-Z][^.!?]*\b(?:do|does|did|can|could|will|would)\b', sentence):
                sentence_structures.append('interrogative')
            elif re.search(r'^[A-Z][^.!?]*\b(?:let|may|should|must)\b', sentence):
                sentence_structures.append('imperative')
            else:
                sentence_structures.append('other')
        
        structure_variety = len(set(sentence_structures)) / len(sentence_structures) if sentence_structures else 0
        if structure_variety < 0.4:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_lexical_repetition_patterns(self, text: str) -> float:
        """Detecta patrones de repetición léxica típicos de IA - NUEVO"""
        score = 0.0
        words = [w.lower().strip('.,!?;:()[]{}"\'') for w in text.split()]
        words = [w for w in words if w and len(w) > 3]
        
        if len(words) < 40:
            return 0.0
        
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        total_words = len(words)
        unique_words = len(word_freq)
        
        repetition_ratio = (total_words - unique_words) / total_words if total_words > 0 else 0
        
        if repetition_ratio > 0.4:
            score += 0.3
        elif repetition_ratio > 0.3:
            score += 0.2
        elif repetition_ratio > 0.2:
            score += 0.1
        
        high_freq_words = [w for w, f in word_freq.items() if f > total_words * 0.05]
        if len(high_freq_words) > 3:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_syntactic_uniformity_patterns(self, text: str) -> float:
        """Analiza patrones de uniformidad sintáctica típicos de IA - NUEVO"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 5:
            return 0.0
        
        sentence_lengths = [len(s.split()) for s in sentences]
        avg_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
        
        length_variance = sum((l - avg_length) ** 2 for l in sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
        std_dev = length_variance ** 0.5
        
        if std_dev < 3:
            score += 0.3
        elif std_dev < 5:
            score += 0.2
        elif std_dev < 8:
            score += 0.1
        
        sentence_complexity = []
        for sentence in sentences:
            complexity = 0
            complexity += len(re.findall(r'\b(?:and|or|but|so|yet|for|nor)\b', sentence, re.IGNORECASE))
            complexity += len(re.findall(r'\b(?:which|that|who|whom|whose|where|when)\b', sentence, re.IGNORECASE))
            complexity += len(re.findall(r'\b(?:if|when|unless|provided|as long as)\b', sentence, re.IGNORECASE))
            sentence_complexity.append(complexity)
        
        complexity_variance = sum((c - sum(sentence_complexity) / len(sentence_complexity)) ** 2 for c in sentence_complexity) / len(sentence_complexity) if sentence_complexity else 0
        complexity_std = complexity_variance ** 0.5
        
        if complexity_std < 1:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_emotional_expression_patterns(self, text: str) -> float:
        """Detecta patrones de expresión emocional típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        emotional_words = ['happy', 'sad', 'angry', 'excited', 'frustrated', 'disappointed', 'surprised', 'shocked', 'amazed', 'confused', 'worried', 'anxious', 'nervous', 'calm', 'relaxed', 'stressed', 'tired', 'energetic', 'bored', 'interested', 'curious', 'proud', 'ashamed', 'embarrassed', 'guilty', 'jealous', 'envious', 'grateful', 'thankful', 'relieved', 'hopeful', 'hopeless', 'optimistic', 'pessimistic', 'confident', 'insecure', 'lonely', 'loved', 'hated', 'feeling', 'feel', 'felt', 'emotion', 'emotional', 'mood', 'moody']
        emotional_count = sum(1 for word in words if word.lower() in emotional_words)
        
        if emotional_count == 0 and len(words) > 100:
            score += 0.3
        elif emotional_count == 0 and len(words) > 50:
            score += 0.2
        elif emotional_count < len(words) * 0.01:
            score += 0.1
        
        emotional_expressions = [
            r'\b(?:i feel|i\'m feeling|i felt|i\'ve been feeling)',
            r'\b(?:it makes me|it made me|it makes us|it made us)',
            r'\b(?:i\'m so|i was so|i am so|i\'ve been so)',
            r'\b(?:this is|that is|it is)\s+(?:so|very|extremely|really|quite)\s+\w+'
        ]
        
        expression_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in emotional_expressions)
        if expression_count == 0 and len(sentences) > 10:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_contextual_ambiguity_patterns(self, text: str) -> float:
        """Analiza patrones de ambigüedad contextual típicos de IA - NUEVO V28"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        # Patrones de ambigüedad contextual típicos de IA
        ambiguous_phrases = [
            r'\b(?:it depends|it may|it might|it could|possibly|perhaps|maybe)',
            r'\b(?:in some cases|in certain situations|under certain circumstances)',
            r'\b(?:depending on|varies|varies depending|can vary)',
            r'\b(?:generally speaking|broadly speaking|in general|typically)',
            r'\b(?:to some extent|in some way|in a way|somewhat)',
            r'\b(?:not necessarily|not always|not necessarily always)',
            r'\b(?:may or may not|could or could not|might or might not)'
        ]
        
        ambiguous_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in ambiguous_phrases)
        if ambiguous_count > len(sentences) * 0.15:
            score += 0.4
        elif ambiguous_count > len(sentences) * 0.10:
            score += 0.3
        elif ambiguous_count > len(sentences) * 0.05:
            score += 0.2
        
        # Análisis de frases que evitan compromiso
        hedging_phrases = [
            r'\b(?:it seems|it appears|it would seem|it would appear)',
            r'\b(?:one might|one could|one may|one would)',
            r'\b(?:it is possible|it is likely|it is unlikely|it is probable)',
            r'\b(?:there is a chance|there is a possibility|there might be)'
        ]
        
        hedging_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in hedging_phrases)
        if hedging_count > 2:
            score += 0.3
        
        return min(score, 1.0)
    
    def _detect_ai_lexical_richness_patterns(self, text: str) -> float:
        """Detecta patrones de riqueza léxica típicos de IA - NUEVO V28"""
        score = 0.0
        words = [w.lower().strip('.,!?;:()[]{}"\'') for w in text.split()]
        words = [w for w in words if w]
        
        if len(words) < 50:
            return 0.0
        
        # Calcular riqueza léxica (type-token ratio)
        unique_words = len(set(words))
        total_words = len(words)
        lexical_richness = unique_words / total_words if total_words > 0 else 0.0
        
        # IA tiende a tener riqueza léxica muy alta o muy baja
        if lexical_richness > 0.85:
            score += 0.3
        elif lexical_richness > 0.80:
            score += 0.2
        elif lexical_richness < 0.40:
            score += 0.3
        elif lexical_richness < 0.50:
            score += 0.2
        
        # Análisis de palabras poco comunes en contexto común
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        uncommon_words = [w for w in words if w not in common_words and len(w) > 4]
        uncommon_ratio = len(uncommon_words) / total_words if total_words > 0 else 0.0
        
        if uncommon_ratio > 0.60:
            score += 0.3
        elif uncommon_ratio > 0.50:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_syntactic_variation_patterns(self, text: str) -> float:
        """Analiza patrones de variación sintáctica típicos de IA - NUEVO V28"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 5:
            return 0.0
        
        # Análisis de variación en longitud de oraciones
        sentence_lengths = [len(s.split()) for s in sentences]
        if len(sentence_lengths) > 0:
            avg_length = sum(sentence_lengths) / len(sentence_lengths)
            std_dev = (sum((x - avg_length) ** 2 for x in sentence_lengths) / len(sentence_lengths)) ** 0.5
            
            # IA tiende a tener poca variación en longitud
            coefficient_of_variation = std_dev / avg_length if avg_length > 0 else 0.0
            if coefficient_of_variation < 0.3:
                score += 0.3
            elif coefficient_of_variation < 0.4:
                score += 0.2
        
        # Análisis de variación en estructura sintáctica
        structures = []
        for sentence in sentences:
            if re.search(r'^[A-Z][^.!?]*\b(?:is|are|was|were)\b', sentence):
                structures.append('declarative')
            elif re.search(r'^[A-Z][^.!?]*\b(?:do|does|did|can|could|will|would)\b', sentence):
                structures.append('interrogative')
            elif re.search(r'^[A-Z][^.!?]*\b(?:let|may|should|must)\b', sentence):
                structures.append('imperative')
            else:
                structures.append('other')
        
        if len(structures) > 0:
            unique_structures = len(set(structures))
            structure_variety = unique_structures / len(structures)
            
            # IA tiende a tener poca variedad estructural
            if structure_variety < 0.4:
                score += 0.3
            elif structure_variety < 0.5:
                score += 0.2
        
        # Análisis de patrones repetitivos de inicio de oración
        sentence_starts = [s.split()[0].lower() if s.split() else '' for s in sentences[:10]]
        start_counts = {}
        for start in sentence_starts:
            start_counts[start] = start_counts.get(start, 0) + 1
        
        max_repetition = max(start_counts.values()) if start_counts else 0
        if max_repetition > len(sentences) * 0.3:
            score += 0.3
        elif max_repetition > len(sentences) * 0.2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_discourse_coherence_patterns(self, text: str) -> float:
        """Detecta patrones de coherencia discursiva típicos de IA - NUEVO V28"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Marcadores discursivos típicos de IA
        discourse_markers = [
            r'\b(?:first|second|third|fourth|fifth|finally|lastly)',
            r'\b(?:in conclusion|to conclude|to summarize|in summary)',
            r'\b(?:furthermore|moreover|additionally|in addition)',
            r'\b(?:however|nevertheless|nonetheless|on the other hand)',
            r'\b(?:therefore|thus|hence|consequently|as a result)',
            r'\b(?:for example|for instance|such as|namely)',
            r'\b(?:in other words|that is|i\.e\.|e\.g\.)',
            r'\b(?:in fact|indeed|actually|as a matter of fact)'
        ]
        
        marker_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in discourse_markers)
        marker_density = marker_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA tiende a usar muchos marcadores discursivos
        if marker_density > 0.5:
            score += 0.4
        elif marker_density > 0.4:
            score += 0.3
        elif marker_density > 0.3:
            score += 0.2
        
        # Análisis de transiciones entre párrafos
        paragraphs = text.split('\n\n')
        transition_phrases = [
            r'\b(?:moving on|turning to|shifting to|now let\'s|let\'s now)',
            r'\b(?:another|additionally|furthermore|moreover|also)',
            r'\b(?:it is also|it should also|it must also|it can also)'
        ]
        
        transition_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in transition_phrases)
        if transition_count > len(paragraphs) * 0.8:
            score += 0.3
        elif transition_count > len(paragraphs) * 0.5:
            score += 0.2
        
        # Análisis de coherencia temática excesiva
        # IA tiende a mantener coherencia temática muy alta
        if len(sentences) > 5:
            # Contar palabras clave repetidas
            word_freq = {}
            for word in words:
                word_lower = word.lower().strip('.,!?;:()[]{}"\'')
                if len(word_lower) > 4:
                    word_freq[word_lower] = word_freq.get(word_lower, 0) + 1
            
            if word_freq:
                max_freq = max(word_freq.values())
                if max_freq > len(words) * 0.10:
                    score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_textual_rhythm_patterns(self, text: str) -> float:
        """Analiza patrones de ritmo textual típicos de IA - NUEVO V29"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 5 or len(words) < 30:
            return 0.0
        
        # Análisis de ritmo basado en longitud de oraciones
        sentence_lengths = [len(s.split()) for s in sentences]
        if len(sentence_lengths) > 0:
            # Calcular variación en ritmo
            avg_length = sum(sentence_lengths) / len(sentence_lengths)
            variance = sum((x - avg_length) ** 2 for x in sentence_lengths) / len(sentence_lengths)
            std_dev = variance ** 0.5
            
            # IA tiende a tener ritmo muy uniforme
            if std_dev < 3.0:
                score += 0.3
            elif std_dev < 5.0:
                score += 0.2
        
        # Análisis de patrones rítmicos repetitivos
        # Contar sílabas aproximadas por palabra (palabras largas = más sílabas)
        syllable_patterns = []
        for sentence in sentences[:10]:
            sentence_words = sentence.split()
            long_words = sum(1 for w in sentence_words if len(w) > 6)
            syllable_patterns.append(long_words / len(sentence_words) if sentence_words else 0)
        
        if len(syllable_patterns) > 0:
            pattern_variance = sum((x - sum(syllable_patterns) / len(syllable_patterns)) ** 2 for x in syllable_patterns) / len(syllable_patterns)
            if pattern_variance < 0.01:
                score += 0.3
        
        # Análisis de pausas y puntuación
        punctuation_density = len(re.findall(r'[,;:]', text)) / len(sentences) if len(sentences) > 0 else 0
        # IA tiende a usar puntuación de manera muy consistente
        if 0.3 <= punctuation_density <= 0.7:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_semantic_redundancy_patterns(self, text: str) -> float:
        """Detecta patrones de redundancia semántica típicos de IA - NUEVO V29"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Detectar frases redundantes
        redundant_patterns = [
            r'\b(?:free gift|free bonus|free offer)',
            r'\b(?:past history|future plans|end result|final outcome)',
            r'\b(?:completely finished|totally complete|absolutely certain)',
            r'\b(?:each and every|first and foremost|any and all)',
            r'\b(?:basic fundamentals|important essentials|necessary requirements)',
            r'\b(?:surrounding circumstances|general consensus|mutual cooperation)',
            r'\b(?:exact same|very unique|most unique|completely unique)',
            r'\b(?:repeat again|continue on|proceed forward|advance forward)'
        ]
        
        redundant_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in redundant_patterns)
        if redundant_count > 2:
            score += 0.4
        elif redundant_count > 1:
            score += 0.3
        
        # Detectar repetición de conceptos similares
        # Buscar sinónimos cercanos en la misma oración o oraciones adyacentes
        synonym_groups = [
            ['important', 'significant', 'crucial', 'vital', 'essential'],
            ['big', 'large', 'huge', 'enormous', 'massive'],
            ['small', 'tiny', 'little', 'miniature', 'minuscule'],
            ['good', 'great', 'excellent', 'wonderful', 'fantastic'],
            ['bad', 'terrible', 'awful', 'horrible', 'dreadful'],
            ['think', 'believe', 'consider', 'suppose', 'assume'],
            ['say', 'tell', 'speak', 'mention', 'state'],
            ['show', 'demonstrate', 'illustrate', 'reveal', 'display']
        ]
        
        for group in synonym_groups:
            found_in_sentence = []
            for i, sentence in enumerate(sentences):
                sentence_lower = sentence.lower()
                found = [word for word in group if word in sentence_lower]
                if len(found) > 1:
                    found_in_sentence.append(i)
            
            # Si se encuentran múltiples sinónimos en la misma oración
            if len(found_in_sentence) > len(sentences) * 0.2:
                score += 0.3
                break
        
        # Detectar tautologías
        tautology_patterns = [
            r'\b(?:it is what it is|what will be will be|it goes without saying)',
            r'\b(?:the fact of the matter|the truth of the matter)',
            r'\b(?:in my own personal opinion|in my own humble opinion)'
        ]
        
        tautology_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in tautology_patterns)
        if tautology_count > 0:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_lexical_sophistication_advanced(self, text: str) -> float:
        """Análisis avanzado de sofisticación léxica típica de IA - NUEVO V29"""
        score = 0.0
        words = [w.lower().strip('.,!?;:()[]{}"\'') for w in text.split()]
        words = [w for w in words if w and len(w) > 1]
        
        if len(words) < 50:
            return 0.0
        
        # Análisis de palabras sofisticadas vs comunes
        sophisticated_words = [
            'utilize', 'facilitate', 'implement', 'optimize', 'maximize', 'minimize',
            'comprehensive', 'systematic', 'methodical', 'analytical', 'strategic',
            'significant', 'substantial', 'considerable', 'notable', 'remarkable',
            'demonstrate', 'illustrate', 'exemplify', 'characterize', 'signify',
            'consequently', 'subsequently', 'furthermore', 'moreover', 'nevertheless',
            'paradigm', 'framework', 'methodology', 'approach', 'perspective',
            'facilitate', 'enhance', 'improve', 'optimize', 'streamline'
        ]
        
        sophisticated_count = sum(1 for w in words if w in sophisticated_words)
        sophisticated_ratio = sophisticated_count / len(words) if len(words) > 0 else 0.0
        
        # IA tiende a usar palabras sofisticadas de manera excesiva
        if sophisticated_ratio > 0.08:
            score += 0.4
        elif sophisticated_ratio > 0.06:
            score += 0.3
        elif sophisticated_ratio > 0.04:
            score += 0.2
        
        # Análisis de longitud promedio de palabras
        avg_word_length = sum(len(w) for w in words) / len(words) if len(words) > 0 else 0.0
        # IA tiende a usar palabras más largas en promedio
        if avg_word_length > 5.5:
            score += 0.3
        elif avg_word_length > 5.2:
            score += 0.2
        
        # Análisis de palabras técnicas o académicas
        academic_words = [
            'analysis', 'analysis', 'methodology', 'framework', 'paradigm',
            'hypothesis', 'theoretical', 'empirical', 'quantitative', 'qualitative',
            'correlation', 'causation', 'variable', 'parameter', 'criterion',
            'systematic', 'comprehensive', 'thorough', 'rigorous', 'methodical'
        ]
        
        academic_count = sum(1 for w in words if w in academic_words)
        academic_ratio = academic_count / len(words) if len(words) > 0 else 0.0
        
        if academic_ratio > 0.05:
            score += 0.3
        elif academic_ratio > 0.03:
            score += 0.2
        
        # Análisis de uso de latín/griego en palabras
        latin_greek_indicators = ['tion', 'sion', 'ology', 'ism', 'ity', 'ment', 'ance', 'ence']
        latin_greek_count = sum(1 for w in words if any(indicator in w for indicator in latin_greek_indicators) and len(w) > 6)
        latin_greek_ratio = latin_greek_count / len(words) if len(words) > 0 else 0.0
        
        if latin_greek_ratio > 0.15:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_pragmatic_markers_patterns(self, text: str) -> float:
        """Detecta patrones de marcadores pragmáticos típicos de IA - NUEVO V29"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Marcadores pragmáticos típicos de IA
        pragmatic_markers = [
            r'\b(?:it is important to note|it should be noted|it is worth noting)',
            r'\b(?:it is crucial to understand|it is essential to recognize)',
            r'\b(?:it is worth mentioning|it is worth pointing out)',
            r'\b(?:it is interesting to observe|it is noteworthy that)',
            r'\b(?:it is clear that|it is evident that|it is obvious that)',
            r'\b(?:it is important to remember|it is crucial to keep in mind)',
            r'\b(?:it is necessary to|it is essential to|it is vital to)',
            r'\b(?:one must|one should|one needs to|one ought to)',
            r'\b(?:it can be seen that|it can be observed that|it can be noted that)',
            r'\b(?:it should be emphasized|it should be highlighted|it should be stressed)',
            r'\b(?:it is worth considering|it is worth examining|it is worth exploring)',
            r'\b(?:it is important to understand|it is crucial to comprehend)'
        ]
        
        marker_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in pragmatic_markers)
        marker_density = marker_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA tiende a usar muchos marcadores pragmáticos
        if marker_density > 0.3:
            score += 0.4
        elif marker_density > 0.2:
            score += 0.3
        elif marker_density > 0.1:
            score += 0.2
        
        # Análisis de frases metalingüísticas
        metalinguistic_phrases = [
            r'\b(?:in other words|to put it another way|that is to say)',
            r'\b(?:as mentioned|as stated|as previously mentioned)',
            r'\b(?:to clarify|to elaborate|to explain further)',
            r'\b(?:in simple terms|in layman\'s terms|in plain language)',
            r'\b(?:to be more specific|to be more precise|to be exact)'
        ]
        
        metalinguistic_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in metalinguistic_phrases)
        if metalinguistic_count > 2:
            score += 0.3
        elif metalinguistic_count > 1:
            score += 0.2
        
        # Análisis de marcadores de organización del discurso
        organization_markers = [
            r'\b(?:first of all|second of all|third of all)',
            r'\b(?:to begin with|to start with|firstly|secondly|thirdly)',
            r'\b(?:last but not least|finally|in conclusion|to conclude)',
            r'\b(?:on the one hand|on the other hand)',
            r'\b(?:in the first place|in the second place)'
        ]
        
        org_marker_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in organization_markers)
        if org_marker_count > len(sentences) * 0.15:
            score += 0.3
        elif org_marker_count > len(sentences) * 0.10:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_conversational_patterns(self, text: str) -> float:
        """Analiza patrones conversacionales típicos de IA - NUEVO V30"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Patrones conversacionales típicos de IA
        conversational_patterns = [
            r'\b(?:let me|let\'s|i\'ll|i will|i can|i\'m going to)',
            r'\b(?:here\'s|here is|here are|below is|following is)',
            r'\b(?:i hope|i hope this|i hope that|hopefully)',
            r'\b(?:feel free to|don\'t hesitate to|please feel free)',
            r'\b(?:if you have|if you need|if you want|if you\'d like)',
            r'\b(?:i\'d be happy|i\'d be glad|i\'d love to|i\'m happy to)',
            r'\b(?:let me know|please let me know|feel free to let me know)',
            r'\b(?:i can help|i can assist|i can provide|i can offer)',
            r'\b(?:is there anything|do you have any|are there any)',
            r'\b(?:i\'m here to|i\'m available to|i\'m ready to)'
        ]
        
        conversational_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in conversational_patterns)
        conversational_density = conversational_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA tiende a usar muchos patrones conversacionales
        if conversational_density > 0.4:
            score += 0.4
        elif conversational_density > 0.3:
            score += 0.3
        elif conversational_density > 0.2:
            score += 0.2
        
        # Análisis de preguntas retóricas o de engagement
        question_patterns = [
            r'\b(?:have you ever|do you ever|did you ever)',
            r'\b(?:what do you think|what are your thoughts|what\'s your opinion)',
            r'\b(?:would you like|would you prefer|would you want)',
            r'\b(?:can you imagine|can you picture|can you see)'
        ]
        
        question_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in question_patterns)
        if question_count > 2:
            score += 0.3
        elif question_count > 1:
            score += 0.2
        
        # Análisis de frases de cierre conversacional
        closing_patterns = [
            r'\b(?:i hope this helps|hope this helps|hope that helps)',
            r'\b(?:let me know if|please let me know if|feel free to let me know if)',
            r'\b(?:if you have any|if you need any|if you want any)',
            r'\b(?:i\'m here if|i\'m available if|i\'m ready if)',
            r'\b(?:don\'t hesitate to|please don\'t hesitate to)'
        ]
        
        closing_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in closing_patterns)
        if closing_count > 1:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_metadiscourse_patterns(self, text: str) -> float:
        """Detecta patrones de metadiscurso típicos de IA - NUEVO V30"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Marcadores de metadiscurso típicos de IA
        metadiscourse_markers = [
            r'\b(?:in this section|in this part|in this chapter|in this paragraph)',
            r'\b(?:as we have seen|as we saw|as mentioned|as stated)',
            r'\b(?:we will now|we shall now|let us now|now we will)',
            r'\b(?:it should be noted|it must be noted|it is important to note)',
            r'\b(?:to summarize|to sum up|in summary|in conclusion)',
            r'\b(?:we have discussed|we discussed|we have seen|we saw)',
            r'\b(?:we will discuss|we shall discuss|we are going to discuss)',
            r'\b(?:as we will see|as we shall see|as we are about to see)',
            r'\b(?:let us consider|let us examine|let us look at)',
            r'\b(?:we can see that|we see that|we observe that)',
            r'\b(?:this section|this part|this chapter|this paragraph)',
            r'\b(?:the following|the next|the previous|the above)'
        ]
        
        metadiscourse_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in metadiscourse_markers)
        metadiscourse_density = metadiscourse_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA tiende a usar muchos marcadores de metadiscurso
        if metadiscourse_density > 0.3:
            score += 0.4
        elif metadiscourse_density > 0.2:
            score += 0.3
        elif metadiscourse_density > 0.1:
            score += 0.2
        
        # Análisis de referencias a la estructura del texto
        structure_references = [
            r'\b(?:above|below|previously|earlier|later)',
            r'\b(?:in the following|in the next|in the previous)',
            r'\b(?:as shown above|as mentioned above|as stated above)',
            r'\b(?:as we will see below|as shown below|as mentioned below)'
        ]
        
        structure_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in structure_references)
        if structure_count > len(sentences) * 0.15:
            score += 0.3
        elif structure_count > len(sentences) * 0.10:
            score += 0.2
        
        # Análisis de comentarios sobre el proceso de escritura
        process_comments = [
            r'\b(?:we will explore|we will examine|we will analyze)',
            r'\b(?:we have explored|we have examined|we have analyzed)',
            r'\b(?:let us turn to|let us move to|let us proceed to)',
            r'\b(?:we now turn to|we now move to|we now proceed to)'
        ]
        
        process_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in process_comments)
        if process_count > 2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_evidentiality_patterns(self, text: str) -> float:
        """Analiza patrones de evidencialidad típicos de IA - NUEVO V30"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Marcadores de evidencialidad típicos de IA
        evidentiality_markers = [
            r'\b(?:according to|based on|in accordance with|in line with)',
            r'\b(?:research shows|studies show|research indicates|studies indicate)',
            r'\b(?:it has been shown|it has been demonstrated|it has been proven)',
            r'\b(?:evidence suggests|evidence indicates|evidence shows)',
            r'\b(?:it is known that|it is well-known that|it is widely known that)',
            r'\b(?:it is believed that|it is thought that|it is considered that)',
            r'\b(?:it is said that|it is reported that|it is claimed that)',
            r'\b(?:it appears that|it seems that|it would seem that)',
            r'\b(?:it is likely that|it is probable that|it is possible that)',
            r'\b(?:it is clear that|it is evident that|it is obvious that)',
            r'\b(?:it has been found|it has been discovered|it has been revealed)',
            r'\b(?:it is generally accepted|it is widely accepted|it is commonly accepted)'
        ]
        
        evidentiality_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in evidentiality_markers)
        evidentiality_density = evidentiality_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA tiende a usar muchos marcadores de evidencialidad
        if evidentiality_density > 0.3:
            score += 0.4
        elif evidentiality_density > 0.2:
            score += 0.3
        elif evidentiality_density > 0.1:
            score += 0.2
        
        # Análisis de citas genéricas sin referencias específicas
        generic_citations = [
            r'\b(?:studies have shown|research has shown|studies have found)',
            r'\b(?:experts say|experts believe|experts suggest)',
            r'\b(?:many studies|numerous studies|various studies)',
            r'\b(?:some research|some studies|some evidence)',
            r'\b(?:it is generally believed|it is widely believed|it is commonly believed)'
        ]
        
        generic_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in generic_citations)
        if generic_count > 2:
            score += 0.3
        elif generic_count > 1:
            score += 0.2
        
        # Análisis de falta de evidencia específica
        # IA tiende a usar evidencialidad genérica sin referencias concretas
        specific_citations = re.findall(r'\b(?:\([A-Z][a-z]+ et al\.|\([A-Z][a-z]+, \d{4}|\[.*?\])', text)
        if evidentiality_count > 0 and len(specific_citations) == 0:
            score += 0.3
        elif evidentiality_count > 0 and len(specific_citations) < evidentiality_count * 0.3:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_engagement_patterns(self, text: str) -> float:
        """Detecta patrones de engagement típicos de IA - NUEVO V30"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Patrones de engagement típicos de IA
        engagement_patterns = [
            r'\b(?:you might|you may|you could|you can)',
            r'\b(?:you will find|you\'ll find|you will see|you\'ll see)',
            r'\b(?:you should|you ought to|you need to|you must)',
            r'\b(?:you can|you are able to|you have the ability to)',
            r'\b(?:if you|when you|as you|while you)',
            r'\b(?:your|yours|yourself|yourselves)',
            r'\b(?:you are|you\'re|you have|you\'ve)',
            r'\b(?:you will|you\'ll|you would|you\'d)',
            r'\b(?:you can also|you may also|you could also)',
            r'\b(?:you might want to|you may want to|you might consider)'
        ]
        
        engagement_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in engagement_patterns)
        engagement_density = engagement_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA tiende a usar muchos patrones de engagement
        if engagement_density > 0.5:
            score += 0.4
        elif engagement_density > 0.4:
            score += 0.3
        elif engagement_density > 0.3:
            score += 0.2
        
        # Análisis de uso excesivo de "you"
        you_count = len(re.findall(r'\byou\b', text, re.IGNORECASE))
        you_density = you_count / len(words) if len(words) > 0 else 0.0
        
        if you_density > 0.05:
            score += 0.3
        elif you_density > 0.03:
            score += 0.2
        
        # Análisis de imperativos directos
        imperative_patterns = [
            r'\b(?:remember that|keep in mind that|don\'t forget that)',
            r'\b(?:make sure|ensure that|be sure to)',
            r'\b(?:try to|attempt to|strive to)',
            r'\b(?:consider|think about|reflect on)',
            r'\b(?:take note|note that|observe that)'
        ]
        
        imperative_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in imperative_patterns)
        if imperative_count > 2:
            score += 0.2
        
        # Análisis de preguntas directas al lector
        direct_questions = re.findall(r'\b(?:have you|do you|are you|would you|could you|will you|can you)\b', text, re.IGNORECASE)
        if len(direct_questions) > len(sentences) * 0.15:
            score += 0.3
        elif len(direct_questions) > len(sentences) * 0.10:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_politeness_patterns(self, text: str) -> float:
        """Analiza patrones de cortesía típicos de IA - NUEVO V31"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Marcadores de cortesía típicos de IA
        politeness_markers = [
            r'\b(?:please|kindly|if you please|if you would)',
            r'\b(?:thank you|thanks|appreciate|grateful)',
            r'\b(?:i apologize|i\'m sorry|sorry for|apologies)',
            r'\b(?:excuse me|pardon me|forgive me|my apologies)',
            r'\b(?:i hope|i trust|i believe|i assume)',
            r'\b(?:would you|could you|might you|may you)',
            r'\b(?:if you don\'t mind|if it\'s not too much|if possible)',
            r'\b(?:i would appreciate|i\'d appreciate|i\'d be grateful)',
            r'\b(?:at your convenience|when convenient|when possible)',
            r'\b(?:i understand|i see|i realize|i acknowledge)'
        ]
        
        politeness_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in politeness_markers)
        politeness_density = politeness_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA tiende a usar muchos marcadores de cortesía
        if politeness_density > 0.4:
            score += 0.4
        elif politeness_density > 0.3:
            score += 0.3
        elif politeness_density > 0.2:
            score += 0.2
        
        # Análisis de frases de cortesía excesiva
        excessive_politeness = [
            r'\b(?:i would be most grateful|i would be very grateful|i would be extremely grateful)',
            r'\b(?:i sincerely hope|i truly hope|i genuinely hope)',
            r'\b(?:i deeply apologize|i sincerely apologize|i truly apologize)',
            r'\b(?:i cannot thank you enough|i cannot express my gratitude enough)',
            r'\b(?:i would be honored|i would be delighted|i would be thrilled)'
        ]
        
        excessive_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in excessive_politeness)
        if excessive_count > 1:
            score += 0.3
        
        # Análisis de uso de condicionales para cortesía
        conditional_politeness = re.findall(r'\b(?:would|could|might|may)\s+(?:you|i|we|they)\s+', text, re.IGNORECASE)
        if len(conditional_politeness) > len(sentences) * 0.3:
            score += 0.3
        elif len(conditional_politeness) > len(sentences) * 0.2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_formality_markers_patterns(self, text: str) -> float:
        """Detecta patrones de marcadores de formalidad típicos de IA - NUEVO V31"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Marcadores de formalidad típicos de IA
        formality_markers = [
            r'\b(?:it is|it\'s|it has been|it shall be)',
            r'\b(?:one must|one should|one ought to|one needs to)',
            r'\b(?:it should be noted|it must be noted|it ought to be noted)',
            r'\b(?:it is imperative|it is essential|it is crucial|it is vital)',
            r'\b(?:it is necessary|it is required|it is mandatory)',
            r'\b(?:it is important to|it is crucial to|it is essential to)',
            r'\b(?:it is worth|it is worthwhile|it is valuable)',
            r'\b(?:it is recommended|it is suggested|it is advised)',
            r'\b(?:it is considered|it is regarded|it is viewed)',
            r'\b(?:it is believed|it is thought|it is assumed)',
            r'\b(?:it is understood|it is recognized|it is acknowledged)',
            r'\b(?:it is expected|it is anticipated|it is predicted)'
        ]
        
        formality_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in formality_markers)
        formality_density = formality_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA tiende a usar muchos marcadores de formalidad
        if formality_density > 0.4:
            score += 0.4
        elif formality_density > 0.3:
            score += 0.3
        elif formality_density > 0.2:
            score += 0.2
        
        # Análisis de construcciones pasivas formales
        passive_formal = [
            r'\b(?:it has been|it had been|it will be|it would be)\s+\w+ed',
            r'\b(?:it is|it was|it will be|it would be)\s+\w+ed',
            r'\b(?:it can be|it could be|it may be|it might be)\s+\w+ed',
            r'\b(?:it should be|it must be|it ought to be)\s+\w+ed'
        ]
        
        passive_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in passive_formal)
        if passive_count > len(sentences) * 0.2:
            score += 0.3
        elif passive_count > len(sentences) * 0.1:
            score += 0.2
        
        # Análisis de uso de "one" en lugar de "you" o "we"
        one_usage = len(re.findall(r'\bone\s+(?:must|should|ought|needs|can|could|may|might|will|would)', text, re.IGNORECASE))
        if one_usage > len(sentences) * 0.15:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_hedging_advanced_patterns(self, text: str) -> float:
        """Análisis avanzado de patrones de hedging típicos de IA - NUEVO V31"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Marcadores avanzados de hedging típicos de IA
        hedging_advanced = [
            r'\b(?:it seems|it appears|it would seem|it would appear)',
            r'\b(?:it is possible|it is likely|it is probable|it is plausible)',
            r'\b(?:it may be|it might be|it could be|it can be)',
            r'\b(?:it is suggested|it is indicated|it is implied)',
            r'\b(?:it is thought|it is believed|it is considered|it is assumed)',
            r'\b(?:it is generally|it is typically|it is usually|it is commonly)',
            r'\b(?:it is often|it is frequently|it is sometimes|it is occasionally)',
            r'\b(?:it is somewhat|it is rather|it is quite|it is fairly)',
            r'\b(?:it is relatively|it is comparatively|it is relatively speaking)',
            r'\b(?:it is to some extent|it is to a certain extent|it is to some degree)',
            r'\b(?:it is not entirely|it is not completely|it is not fully)',
            r'\b(?:it is arguably|it is debatable|it is questionable)'
        ]
        
        hedging_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in hedging_advanced)
        hedging_density = hedging_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA tiende a usar muchos marcadores de hedging
        if hedging_density > 0.4:
            score += 0.4
        elif hedging_density > 0.3:
            score += 0.3
        elif hedging_density > 0.2:
            score += 0.2
        
        # Análisis de verbos modales de hedging
        modal_hedging = [
            r'\b(?:may|might|could|can)\s+(?:be|have|do|get|make|take|give|see|know|think|say|tell)',
            r'\b(?:would|should|ought to)\s+(?:be|have|do|get|make|take|give|see|know|think|say|tell)'
        ]
        
        modal_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in modal_hedging)
        if modal_count > len(sentences) * 0.3:
            score += 0.3
        elif modal_count > len(sentences) * 0.2:
            score += 0.2
        
        # Análisis de adverbios de hedging
        hedging_adverbs = ['possibly', 'probably', 'perhaps', 'maybe', 'presumably', 'supposedly', 'allegedly', 'reportedly', 'apparently', 'seemingly', 'arguably', 'potentially', 'theoretically', 'hypothetically']
        adverb_count = sum(1 for word in words if word.lower() in hedging_adverbs)
        if adverb_count > len(sentences) * 0.2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_assertiveness_patterns(self, text: str) -> float:
        """Detecta patrones de asertividad típicos de IA - NUEVO V31"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Patrones de asertividad típicos de IA
        assertiveness_patterns = [
            r'\b(?:it is clear that|it is evident that|it is obvious that|it is apparent that)',
            r'\b(?:it is certain that|it is definite that|it is sure that)',
            r'\b(?:it is undeniable that|it is indisputable that|it is unquestionable that)',
            r'\b(?:it is well-established|it is well-known|it is widely recognized)',
            r'\b(?:it is universally accepted|it is generally accepted|it is commonly accepted)',
            r'\b(?:it is a fact that|it is a truth that|it is a reality that)',
            r'\b(?:it is proven that|it is demonstrated that|it is established that)',
            r'\b(?:it is confirmed that|it is verified that|it is validated that)',
            r'\b(?:there is no doubt|there is no question|there is no denying)',
            r'\b(?:it cannot be denied|it cannot be disputed|it cannot be questioned)',
            r'\b(?:it must be|it has to be|it needs to be|it ought to be)',
            r'\b(?:it is imperative|it is essential|it is crucial|it is vital)'
        ]
        
        assertiveness_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in assertiveness_patterns)
        assertiveness_density = assertiveness_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA tiende a usar muchos patrones de asertividad
        if assertiveness_density > 0.3:
            score += 0.4
        elif assertiveness_density > 0.2:
            score += 0.3
        elif assertiveness_density > 0.1:
            score += 0.2
        
        # Análisis de uso de "must" y "should" de manera asertiva
        must_should = len(re.findall(r'\b(?:must|should|ought to|has to|needs to)\s+(?:be|have|do|get|make)', text, re.IGNORECASE))
        if must_should > len(sentences) * 0.2:
            score += 0.3
        elif must_should > len(sentences) * 0.1:
            score += 0.2
        
        # Análisis de afirmaciones categóricas
        categorical_claims = [
            r'\b(?:always|never|all|every|none|no one|nothing|nowhere)',
            r'\b(?:completely|totally|absolutely|entirely|fully|wholly)',
            r'\b(?:definitely|certainly|surely|undoubtedly|unquestionably)',
            r'\b(?:without exception|without doubt|without question)'
        ]
        
        categorical_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in categorical_claims)
        if categorical_count > len(sentences) * 0.3:
            score += 0.3
        elif categorical_count > len(sentences) * 0.2:
            score += 0.2
        
        # Análisis de falta de calificadores en afirmaciones fuertes
        strong_claims = re.findall(r'\b(?:is|are|was|were)\s+(?:always|never|all|every|completely|totally)', text, re.IGNORECASE)
        if len(strong_claims) > 2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_intertextuality_patterns(self, text: str) -> float:
        """Analiza patrones de intertextualidad típicos de IA - NUEVO V32"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Patrones de intertextualidad típicos de IA
        intertextuality_patterns = [
            r'\b(?:as mentioned|as stated|as noted|as discussed)',
            r'\b(?:as we have seen|as we saw|as we observed)',
            r'\b(?:as previously mentioned|as mentioned earlier|as stated above)',
            r'\b(?:as noted in|as stated in|as discussed in|as mentioned in)',
            r'\b(?:in the previous|in the earlier|in the above|in the following)',
            r'\b(?:as we will see|as we shall see|as we are about to see)',
            r'\b(?:as we have discussed|as we discussed|as we have seen)',
            r'\b(?:referring to|referring back to|going back to)',
            r'\b(?:in relation to|in connection with|in reference to)',
            r'\b(?:similar to|similar as|comparable to|comparable with)',
            r'\b(?:in contrast to|in contrast with|contrary to|unlike)',
            r'\b(?:building on|building upon|expanding on|expanding upon)'
        ]
        
        intertextuality_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in intertextuality_patterns)
        intertextuality_density = intertextuality_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA tiende a usar muchos patrones de intertextualidad
        if intertextuality_density > 0.3:
            score += 0.4
        elif intertextuality_density > 0.2:
            score += 0.3
        elif intertextuality_density > 0.1:
            score += 0.2
        
        # Análisis de referencias cruzadas excesivas
        cross_references = [
            r'\b(?:see above|see below|see earlier|see later)',
            r'\b(?:as shown above|as shown below|as demonstrated above)',
            r'\b(?:as indicated above|as indicated below|as noted above)',
            r'\b(?:as we saw|as we have seen|as we will see)'
        ]
        
        cross_ref_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in cross_references)
        if cross_ref_count > len(sentences) * 0.15:
            score += 0.3
        elif cross_ref_count > len(sentences) * 0.10:
            score += 0.2
        
        # Análisis de comparaciones y contrastes excesivos
        comparison_patterns = [
            r'\b(?:compared to|compared with|in comparison to|in comparison with)',
            r'\b(?:unlike|like|similar to|different from)',
            r'\b(?:in contrast to|in contrast with|contrary to|on the contrary)',
            r'\b(?:whereas|while|whilst|although|though)'
        ]
        
        comparison_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in comparison_patterns)
        if comparison_count > len(sentences) * 0.2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_citation_density_patterns(self, text: str) -> float:
        """Detecta patrones de densidad de citas típicos de IA - NUEVO V32"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Detectar citas formales (formato académico)
        formal_citations = re.findall(r'\([A-Z][a-z]+(?:\s+et\s+al\.)?(?:\s+and\s+[A-Z][a-z]+)?,?\s*\d{4}[a-z]?\)', text)
        formal_citation_count = len(formal_citations)
        
        # Detectar citas entre corchetes
        bracket_citations = re.findall(r'\[.*?\]', text)
        bracket_citation_count = len(bracket_citations)
        
        # Detectar referencias numéricas
        numeric_citations = re.findall(r'\[?\d+\]?', text)
        numeric_citation_count = len([c for c in numeric_citations if len(c) <= 4])
        
        total_citations = formal_citation_count + bracket_citation_count + numeric_citation_count
        citation_density = total_citations / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA puede tener densidad de citas muy alta o muy baja
        if citation_density > 1.5:
            score += 0.4
        elif citation_density > 1.0:
            score += 0.3
        elif citation_density == 0 and len(words) > 200:
            # Textos largos sin citas pueden ser IA
            score += 0.3
        elif citation_density == 0 and len(words) > 100:
            score += 0.2
        
        # Análisis de citas genéricas sin referencias específicas
        generic_citation_phrases = [
            r'\b(?:studies show|research shows|studies indicate|research indicates)',
            r'\b(?:experts say|experts believe|experts suggest|experts agree)',
            r'\b(?:it has been shown|it has been demonstrated|it has been proven)',
            r'\b(?:according to research|according to studies|according to experts)',
            r'\b(?:many studies|numerous studies|various studies|several studies)',
            r'\b(?:some research|some studies|some evidence|some data)'
        ]
        
        generic_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in generic_citation_phrases)
        if generic_count > 2 and formal_citation_count == 0:
            score += 0.4
        elif generic_count > 1 and formal_citation_count < 2:
            score += 0.3
        
        # Análisis de distribución de citas
        # IA tiende a tener citas distribuidas de manera muy uniforme
        if formal_citation_count > 5:
            citation_positions = [m.start() for m in re.finditer(r'\([A-Z][a-z]+(?:\s+et\s+al\.)?,?\s*\d{4}[a-z]?\)', text)]
            if len(citation_positions) > 1:
                text_length = len(text)
                intervals = []
                for i in range(len(citation_positions) - 1):
                    intervals.append(citation_positions[i+1] - citation_positions[i])
                
                if intervals:
                    avg_interval = sum(intervals) / len(intervals)
                    variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)
                    coefficient_of_variation = (variance ** 0.5) / avg_interval if avg_interval > 0 else 0
                    
                    # Variación muy baja indica distribución uniforme (típico de IA)
                    if coefficient_of_variation < 0.3:
                        score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_authority_claims_patterns(self, text: str) -> float:
        """Analiza patrones de afirmaciones de autoridad típicos de IA - NUEVO V32"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Patrones de afirmaciones de autoridad típicos de IA
        authority_claims = [
            r'\b(?:experts agree|experts say|experts believe|experts suggest)',
            r'\b(?:it is well-known|it is widely known|it is commonly known)',
            r'\b(?:it is established|it is proven|it is demonstrated|it is confirmed)',
            r'\b(?:it is recognized|it is acknowledged|it is accepted)',
            r'\b(?:it is generally accepted|it is widely accepted|it is commonly accepted)',
            r'\b(?:it is universally accepted|it is globally accepted)',
            r'\b(?:it is a fact|it is a truth|it is a reality)',
            r'\b(?:it is clear|it is evident|it is obvious|it is apparent)',
            r'\b(?:it is certain|it is definite|it is sure)',
            r'\b(?:it is undeniable|it is indisputable|it is unquestionable)',
            r'\b(?:research shows|studies show|research indicates|studies indicate)',
            r'\b(?:science shows|science indicates|science demonstrates)',
            r'\b(?:it has been proven|it has been demonstrated|it has been established)',
            r'\b(?:it has been shown|it has been confirmed|it has been verified)'
        ]
        
        authority_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in authority_claims)
        authority_density = authority_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA tiende a usar muchos patrones de afirmaciones de autoridad
        if authority_density > 0.3:
            score += 0.4
        elif authority_density > 0.2:
            score += 0.3
        elif authority_density > 0.1:
            score += 0.2
        
        # Análisis de afirmaciones de autoridad sin respaldo específico
        unsupported_authority = [
            r'\b(?:experts agree|experts say|experts believe)\s+(?:that|on|about)',
            r'\b(?:it is well-known|it is widely known)\s+(?:that|to)',
            r'\b(?:research shows|studies show)\s+(?:that|how|why)'
        ]
        
        unsupported_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in unsupported_authority)
        # Si hay muchas afirmaciones de autoridad pero pocas citas formales
        formal_citations = len(re.findall(r'\([A-Z][a-z]+(?:\s+et\s+al\.)?,?\s*\d{4}[a-z]?\)', text))
        if unsupported_count > 2 and formal_citations < unsupported_count:
            score += 0.3
        elif unsupported_count > 1 and formal_citations == 0:
            score += 0.2
        
        # Análisis de uso de "experts" sin especificar
        experts_usage = len(re.findall(r'\bexperts\s+(?:say|believe|suggest|agree|think|argue|claim)', text, re.IGNORECASE))
        if experts_usage > len(sentences) * 0.15:
            score += 0.3
        elif experts_usage > len(sentences) * 0.10:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_expertise_markers_patterns(self, text: str) -> float:
        """Detecta patrones de marcadores de expertise típicos de IA - NUEVO V32"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Marcadores de expertise típicos de IA
        expertise_markers = [
            r'\b(?:it is important to understand|it is crucial to understand|it is essential to understand)',
            r'\b(?:it is worth noting|it is worth mentioning|it is worth pointing out)',
            r'\b(?:it should be emphasized|it should be highlighted|it should be stressed)',
            r'\b(?:it is crucial to recognize|it is important to recognize|it is essential to recognize)',
            r'\b(?:it is necessary to|it is essential to|it is crucial to|it is vital to)',
            r'\b(?:one must understand|one should understand|one needs to understand)',
            r'\b(?:it is key to|it is critical to|it is fundamental to)',
            r'\b(?:it is imperative to|it is mandatory to|it is required to)',
            r'\b(?:it is advisable to|it is recommended to|it is suggested to)',
            r'\b(?:it is beneficial to|it is advantageous to|it is useful to)',
            r'\b(?:it is worth considering|it is worth examining|it is worth exploring)',
            r'\b(?:it is important to remember|it is crucial to remember|it is essential to remember)',
            r'\b(?:it is important to keep in mind|it is crucial to keep in mind)',
            r'\b(?:it is important to note|it is crucial to note|it is essential to note)'
        ]
        
        expertise_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in expertise_markers)
        expertise_density = expertise_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA tiende a usar muchos marcadores de expertise
        if expertise_density > 0.3:
            score += 0.4
        elif expertise_density > 0.2:
            score += 0.3
        elif expertise_density > 0.1:
            score += 0.2
        
        # Análisis de frases que indican conocimiento especializado
        specialized_knowledge = [
            r'\b(?:in the field of|in the domain of|in the area of)',
            r'\b(?:according to|based on|in accordance with)',
            r'\b(?:it is known in|it is recognized in|it is accepted in)',
            r'\b(?:within the context of|within the framework of|within the scope of)',
            r'\b(?:from a|from an|from the)\s+(?:perspective|viewpoint|standpoint|angle)',
            r'\b(?:in terms of|with regard to|with respect to|in relation to)'
        ]
        
        specialized_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in specialized_knowledge)
        if specialized_count > len(sentences) * 0.2:
            score += 0.3
        elif specialized_count > len(sentences) * 0.15:
            score += 0.2
        
        # Análisis de uso de terminología técnica sin explicación
        technical_terms = [
            r'\b(?:methodology|framework|paradigm|approach|strategy)',
            r'\b(?:systematic|comprehensive|thorough|rigorous|methodical)',
            r'\b(?:analysis|evaluation|assessment|examination|investigation)',
            r'\b(?:implementation|application|utilization|optimization)',
            r'\b(?:correlation|causation|variable|parameter|criterion)'
        ]
        
        technical_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in technical_terms)
        # Si hay muchos términos técnicos pero pocas explicaciones
        explanation_phrases = len(re.findall(r'\b(?:which means|that is|in other words|to put it simply)', text, re.IGNORECASE))
        if technical_count > 5 and explanation_phrases < technical_count * 0.2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_temporal_coherence_patterns(self, text: str) -> float:
        """Analiza patrones de coherencia temporal típicos de IA - NUEVO V33"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Marcadores temporales típicos de IA
        temporal_markers = [
            r'\b(?:first|second|third|fourth|fifth|finally|lastly|initially|subsequently)',
            r'\b(?:then|next|after|afterward|afterwards|later|meanwhile|simultaneously)',
            r'\b(?:previously|earlier|before|prior to|preceding|former)',
            r'\b(?:now|currently|presently|at present|at the moment)',
            r'\b(?:recently|lately|in recent|in the recent)',
            r'\b(?:eventually|ultimately|finally|in the end|at last)',
            r'\b(?:meanwhile|at the same time|simultaneously|concurrently)',
            r'\b(?:during|while|whilst|throughout|over the course of)',
            r'\b(?:since|until|till|from|to|until now|up to now)',
            r'\b(?:in the past|in the future|in the present|at that time)'
        ]
        
        temporal_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in temporal_markers)
        temporal_density = temporal_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA tiende a usar muchos marcadores temporales
        if temporal_density > 0.4:
            score += 0.4
        elif temporal_density > 0.3:
            score += 0.3
        elif temporal_density > 0.2:
            score += 0.2
        
        # Análisis de secuencia temporal excesivamente estructurada
        sequence_patterns = [
            r'\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)',
            r'\b(?:step one|step two|step three|step four|step five)',
            r'\b(?:firstly|secondly|thirdly|fourthly|fifthly)',
            r'\b(?:in the first place|in the second place|in the third place)'
        ]
        
        sequence_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in sequence_patterns)
        if sequence_count > len(sentences) * 0.2:
            score += 0.3
        elif sequence_count > len(sentences) * 0.15:
            score += 0.2
        
        # Análisis de coherencia temporal perfecta (sin saltos temporales)
        # IA tiende a tener secuencias temporales muy lineales
        time_verbs = ['was', 'were', 'had', 'has', 'have', 'will', 'would', 'is', 'are']
        time_verb_count = sum(1 for word in words if word.lower() in time_verbs)
        time_verb_density = time_verb_count / len(words) if len(words) > 0 else 0.0
        
        # Si hay muchos verbos temporales pero poca variación en tiempos
        if time_verb_density > 0.15:
            # Contar variación en tiempos verbales
            past_forms = len(re.findall(r'\b(?:was|were|had|did|went|came|said|told)', text, re.IGNORECASE))
            present_forms = len(re.findall(r'\b(?:is|are|am|do|does|go|come|say|tell)', text, re.IGNORECASE))
            future_forms = len(re.findall(r'\b(?:will|would|shall|should|going to)', text, re.IGNORECASE))
            
            total_tense_forms = past_forms + present_forms + future_forms
            if total_tense_forms > 0:
                max_tense = max(past_forms, present_forms, future_forms)
                tense_uniformity = max_tense / total_tense_forms
                # Si un tiempo domina mucho (uniformidad alta), puede ser IA
                if tense_uniformity > 0.7:
                    score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_causal_chain_patterns(self, text: str) -> float:
        """Detecta patrones de cadenas causales típicos de IA - NUEVO V33"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Marcadores causales típicos de IA
        causal_markers = [
            r'\b(?:because|since|as|due to|owing to|as a result of)',
            r'\b(?:therefore|thus|hence|consequently|as a result|so)',
            r'\b(?:if|when|whenever|provided that|assuming that)',
            r'\b(?:leads to|results in|causes|brings about|gives rise to)',
            r'\b(?:caused by|resulted from|stemmed from|arose from)',
            r'\b(?:this leads to|this results in|this causes|this brings about)',
            r'\b(?:which leads to|which results in|which causes|which brings about)',
            r'\b(?:in order to|so as to|for the purpose of|with the aim of)',
            r'\b(?:in order that|so that|such that|with the result that)',
            r'\b(?:as a consequence|as a consequence of|consequently|accordingly)'
        ]
        
        causal_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in causal_markers)
        causal_density = causal_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA tiende a usar muchos marcadores causales
        if causal_density > 0.4:
            score += 0.4
        elif causal_density > 0.3:
            score += 0.3
        elif causal_density > 0.2:
            score += 0.2
        
        # Análisis de cadenas causales lineales excesivas
        # IA tiende a crear cadenas causales muy lineales y estructuradas
        causal_chain_patterns = [
            r'\b(?:this|that|which|it)\s+(?:leads to|results in|causes|brings about)',
            r'\b(?:because|since|as)\s+(?:this|that|which|it)\s+(?:leads to|results in|causes)',
            r'\b(?:therefore|thus|hence|consequently)\s+(?:this|that|which|it)',
            r'\b(?:as a result|consequently|therefore)\s+(?:this|that|which|it)\s+(?:leads to|results in)'
        ]
        
        chain_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in causal_chain_patterns)
        if chain_count > 2:
            score += 0.3
        elif chain_count > 1:
            score += 0.2
        
        # Análisis de relaciones causales simplificadas
        # IA tiende a usar relaciones causales muy directas y simples
        simple_causal = len(re.findall(r'\b(?:because|since|as)\s+\w+\s+(?:is|are|was|were|has|have|will|would)', text, re.IGNORECASE))
        if simple_causal > len(sentences) * 0.2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_narrative_structure_patterns(self, text: str) -> float:
        """Analiza patrones de estructura narrativa típicos de IA - NUEVO V33"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        paragraphs = text.split('\n\n')
        
        if len(sentences) < 5 or len(words) < 50:
            return 0.0
        
        # Patrones de estructura narrativa típicos de IA
        narrative_structure = [
            r'\b(?:in the beginning|at the start|initially|first of all)',
            r'\b(?:in the middle|during|meanwhile|at this point)',
            r'\b(?:in the end|finally|ultimately|at last|conclusively)',
            r'\b(?:the story begins|the narrative starts|the tale begins)',
            r'\b(?:the story continues|the narrative continues|the tale continues)',
            r'\b(?:the story ends|the narrative ends|the tale ends)',
            r'\b(?:to begin with|to start with|to commence with)',
            r'\b(?:to conclude|to finish|to end|to wrap up)',
            r'\b(?:in conclusion|in summary|in closing|to sum up)'
        ]
        
        narrative_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in narrative_structure)
        narrative_density = narrative_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA tiende a usar muchos marcadores de estructura narrativa
        if narrative_density > 0.2:
            score += 0.3
        elif narrative_density > 0.15:
            score += 0.2
        
        # Análisis de estructura de tres actos (típico de IA)
        three_act_patterns = [
            r'\b(?:introduction|introducing|introduce|first act)',
            r'\b(?:development|developing|develop|second act|middle act)',
            r'\b(?:conclusion|concluding|conclude|third act|final act)',
            r'\b(?:setup|rising action|climax|falling action|resolution)'
        ]
        
        three_act_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in three_act_patterns)
        if three_act_count > 2:
            score += 0.3
        
        # Análisis de uniformidad en longitud de párrafos
        if len(paragraphs) > 3:
            paragraph_lengths = [len(p.split()) for p in paragraphs if p.strip()]
            if len(paragraph_lengths) > 0:
                avg_length = sum(paragraph_lengths) / len(paragraph_lengths)
                variance = sum((x - avg_length) ** 2 for x in paragraph_lengths) / len(paragraph_lengths)
                std_dev = variance ** 0.5
                coefficient_of_variation = std_dev / avg_length if avg_length > 0 else 0.0
                
                # IA tiende a tener párrafos de longitud muy uniforme
                if coefficient_of_variation < 0.3:
                    score += 0.3
                elif coefficient_of_variation < 0.4:
                    score += 0.2
        
        # Análisis de transiciones narrativas excesivamente estructuradas
        narrative_transitions = [
            r'\b(?:moving forward|moving on|turning to|shifting to)',
            r'\b(?:now let\'s|let\'s now|let us now|we now turn)',
            r'\b(?:next|then|after that|following that|subsequently)',
            r'\b(?:in the next section|in the following section|in the subsequent section)'
        ]
        
        transition_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in narrative_transitions)
        if transition_count > len(paragraphs) * 0.5:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_argumentation_patterns(self, text: str) -> float:
        """Detecta patrones de argumentación típicos de IA - NUEVO V33"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Marcadores de argumentación típicos de IA
        argumentation_markers = [
            r'\b(?:first|second|third|fourth|fifth|furthermore|moreover|additionally)',
            r'\b(?:on the one hand|on the other hand|however|nevertheless|nonetheless)',
            r'\b(?:for example|for instance|such as|namely|specifically)',
            r'\b(?:in other words|that is|i\.e\.|e\.g\.|to put it another way)',
            r'\b(?:therefore|thus|hence|consequently|as a result|so)',
            r'\b(?:it follows that|it can be concluded that|it can be inferred that)',
            r'\b(?:this suggests|this indicates|this implies|this shows)',
            r'\b(?:it is clear that|it is evident that|it is obvious that)',
            r'\b(?:it can be argued that|it can be said that|it can be claimed that)',
            r'\b(?:in support of|in favor of|against|opposed to)',
            r'\b(?:to support|to argue|to claim|to suggest|to propose)',
            r'\b(?:in conclusion|to conclude|to sum up|in summary)'
        ]
        
        argumentation_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in argumentation_markers)
        argumentation_density = argumentation_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA tiende a usar muchos marcadores de argumentación
        if argumentation_density > 0.4:
            score += 0.4
        elif argumentation_density > 0.3:
            score += 0.3
        elif argumentation_density > 0.2:
            score += 0.2
        
        # Análisis de estructura argumentativa excesivamente formal
        formal_argumentation = [
            r'\b(?:premise|conclusion|argument|reasoning|logic)',
            r'\b(?:it follows that|it can be concluded that|it can be inferred that)',
            r'\b(?:therefore|thus|hence|consequently|as a result)',
            r'\b(?:given that|assuming that|provided that|supposing that)',
            r'\b(?:we can conclude|we can infer|we can deduce|we can reason)'
        ]
        
        formal_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in formal_argumentation)
        if formal_count > len(sentences) * 0.15:
            score += 0.3
        elif formal_count > len(sentences) * 0.10:
            score += 0.2
        
        # Análisis de contraargumentos estructurados
        counterargument_patterns = [
            r'\b(?:on the one hand|on the other hand|however|nevertheless)',
            r'\b(?:some may argue|one might argue|it could be argued)',
            r'\b(?:while|whereas|although|though|even though)',
            r'\b(?:despite|in spite of|notwithstanding|regardless of)',
            r'\b(?:contrary to|opposite to|in contrast to|in contrast with)'
        ]
        
        counterargument_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in counterargument_patterns)
        # IA tiende a usar contraargumentos de manera muy estructurada
        if counterargument_count > len(sentences) * 0.2:
            score += 0.2
        
        # Análisis de ejemplos estructurados
        example_patterns = [
            r'\b(?:for example|for instance|such as|namely|specifically)',
            r'\b(?:to illustrate|to demonstrate|to show|to exemplify)',
            r'\b(?:consider|take|look at|examine|observe)',
            r'\b(?:an example|one example|another example|a case in point)'
        ]
        
        example_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in example_patterns)
        if example_count > len(sentences) * 0.25:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_lexical_consistency_patterns(self, text: str) -> float:
        """Analiza patrones de consistencia léxica típicos de IA - NUEVO V34"""
        score = 0.0
        words = [w.lower().strip('.,!?;:()[]{}"\'') for w in text.split()]
        words = [w for w in words if w and len(w) > 1]
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 50 or len(sentences) < 3:
            return 0.0
        
        # Análisis de consistencia en elección de palabras
        # IA tiende a usar las mismas palabras de manera muy consistente
        word_frequency = {}
        for word in words:
            if len(word) > 3:  # Solo palabras significativas
                word_frequency[word] = word_frequency.get(word, 0) + 1
        
        # Calcular índice de repetición de palabras clave
        total_significant_words = sum(1 for w in words if len(w) > 3)
        if total_significant_words > 0:
            repeated_words = sum(1 for count in word_frequency.values() if count > 2)
            repetition_ratio = repeated_words / len(word_frequency) if word_frequency else 0.0
            
            # IA tiende a tener alta consistencia (muchas repeticiones)
            if repetition_ratio > 0.4:
                score += 0.3
            elif repetition_ratio > 0.3:
                score += 0.2
        
        # Análisis de sinónimos - IA tiende a usar el mismo sinónimo consistentemente
        synonym_groups = {
            'important': ['important', 'significant', 'crucial', 'vital', 'essential'],
            'big': ['big', 'large', 'huge', 'enormous', 'massive'],
            'good': ['good', 'great', 'excellent', 'wonderful', 'fantastic'],
            'show': ['show', 'demonstrate', 'illustrate', 'reveal', 'display'],
            'think': ['think', 'believe', 'consider', 'suppose', 'assume']
        }
        
        for group_name, synonyms in synonym_groups.items():
            found_synonyms = [word for word in words if word in synonyms]
            if len(found_synonyms) > 2:
                # Si se usa principalmente un sinónimo (consistencia alta)
                synonym_counts = {}
                for syn in found_synonyms:
                    synonym_counts[syn] = synonym_counts.get(syn, 0) + 1
                
                if synonym_counts:
                    max_count = max(synonym_counts.values())
                    consistency = max_count / len(found_synonyms)
                    if consistency > 0.7:
                        score += 0.2
                        break
        
        # Análisis de variación en vocabulario técnico
        technical_terms = ['methodology', 'framework', 'analysis', 'evaluation', 'assessment', 'implementation', 'optimization']
        technical_found = [w for w in words if w in technical_terms]
        if len(technical_found) > 3:
            # Si se usan términos técnicos de manera muy consistente
            technical_counts = {}
            for term in technical_found:
                technical_counts[term] = technical_counts.get(term, 0) + 1
            
            if technical_counts:
                max_tech = max(technical_counts.values())
                if max_tech > len(technical_found) * 0.5:
                    score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_semantic_field_patterns(self, text: str) -> float:
        """Detecta patrones de campos semánticos típicos de IA - NUEVO V34"""
        score = 0.0
        words = [w.lower().strip('.,!?;:()[]{}"\'') for w in text.split()]
        words = [w for w in words if w and len(w) > 1]
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 50 or len(sentences) < 3:
            return 0.0
        
        # Definir campos semánticos comunes
        semantic_fields = {
            'academic': ['research', 'study', 'analysis', 'methodology', 'framework', 'theory', 'hypothesis', 'empirical', 'quantitative', 'qualitative'],
            'business': ['strategy', 'management', 'organization', 'efficiency', 'productivity', 'optimization', 'implementation', 'stakeholder', 'revenue', 'profit'],
            'technology': ['system', 'platform', 'application', 'software', 'hardware', 'algorithm', 'database', 'network', 'interface', 'protocol'],
            'science': ['experiment', 'observation', 'hypothesis', 'theory', 'data', 'evidence', 'analysis', 'conclusion', 'method', 'result'],
            'general_knowledge': ['important', 'significant', 'crucial', 'essential', 'fundamental', 'key', 'main', 'primary', 'principal', 'major']
        }
        
        # Contar palabras de cada campo semántico
        field_counts = {}
        for field, terms in semantic_fields.items():
            count = sum(1 for word in words if word in terms)
            if count > 0:
                field_counts[field] = count
        
        # IA tiende a usar campos semánticos muy concentrados
        if field_counts:
            total_field_words = sum(field_counts.values())
            max_field = max(field_counts.values())
            concentration = max_field / total_field_words if total_field_words > 0 else 0.0
            
            # Alta concentración en un campo semántico
            if concentration > 0.6:
                score += 0.4
            elif concentration > 0.5:
                score += 0.3
            elif concentration > 0.4:
                score += 0.2
        
        # Análisis de transiciones entre campos semánticos
        # IA tiende a tener transiciones muy abruptas o muy suaves
        if len(sentences) > 5:
            sentence_fields = []
            for sentence in sentences[:10]:
                sentence_words = [w.lower().strip('.,!?;:()[]{}"\'') for w in sentence.split()]
                sentence_field_counts = {}
                for field, terms in semantic_fields.items():
                    count = sum(1 for word in sentence_words if word in terms)
                    if count > 0:
                        sentence_field_counts[field] = count
                
                if sentence_field_counts:
                    dominant_field = max(sentence_field_counts.items(), key=lambda x: x[1])[0]
                    sentence_fields.append(dominant_field)
            
            # Si hay muchas transiciones entre campos (inconsistencia)
            if len(sentence_fields) > 1:
                transitions = sum(1 for i in range(len(sentence_fields) - 1) if sentence_fields[i] != sentence_fields[i+1])
                transition_ratio = transitions / (len(sentence_fields) - 1) if len(sentence_fields) > 1 else 0.0
                
                # Muchas transiciones o muy pocas (ambos pueden ser IA)
                if transition_ratio > 0.7 or transition_ratio < 0.2:
                    score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_register_consistency_patterns(self, text: str) -> float:
        """Analiza patrones de consistencia de registro típicos de IA - NUEVO V34"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Marcadores de registro formal
        formal_markers = [
            r'\b(?:it is|it has been|it shall be|it will be)',
            r'\b(?:one must|one should|one ought to|one needs to)',
            r'\b(?:it should be noted|it must be noted|it ought to be noted)',
            r'\b(?:it is imperative|it is essential|it is crucial)',
            r'\b(?:according to|based on|in accordance with)',
            r'\b(?:furthermore|moreover|additionally|in addition)',
            r'\b(?:therefore|thus|hence|consequently|as a result)'
        ]
        
        # Marcadores de registro informal
        informal_markers = [
            r'\b(?:gonna|wanna|gotta|lemme|dunno)',
            r'\b(?:yeah|yep|nope|nah|uh|um|er)',
            r'\b(?:cool|awesome|great|nice|sweet)',
            r'\b(?:like|you know|I mean|sort of|kind of)',
            r'\b(?:gonna|wanna|gotta|lemme|dunno)',
            r'\b(?:can\'t|won\'t|don\'t|isn\'t|aren\'t)',
            r'\b(?:it\'s|that\'s|what\'s|there\'s|here\'s)'
        ]
        
        formal_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in formal_markers)
        informal_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in informal_markers)
        
        total_markers = formal_count + informal_count
        if total_markers > 0:
            # IA tiende a tener registro muy consistente (todo formal o todo informal)
            if formal_count > 0 and informal_count == 0:
                # Todo formal
                if formal_count > len(sentences) * 0.3:
                    score += 0.3
            elif informal_count > 0 and formal_count == 0:
                # Todo informal
                if informal_count > len(sentences) * 0.3:
                    score += 0.3
            elif formal_count > 0 and informal_count > 0:
                # Mezcla - puede ser humano, pero si es muy poca mezcla, puede ser IA
                mix_ratio = min(formal_count, informal_count) / max(formal_count, informal_count)
                if mix_ratio < 0.2:  # Muy poca mezcla
                    score += 0.2
        
        # Análisis de contracciones
        contractions = len(re.findall(r'\b\w+\'[a-z]+\b', text, re.IGNORECASE))
        no_contractions = len(re.findall(r'\b(?:it is|it has|it will|it would|do not|does not|did not|can not|cannot|will not|would not|should not|could not|must not)', text, re.IGNORECASE))
        
        # IA puede tener consistencia en uso de contracciones
        if contractions > 0 and no_contractions == 0:
            # Solo contracciones
            if contractions > len(sentences) * 0.2:
                score += 0.2
        elif no_contractions > 0 and contractions == 0:
            # Sin contracciones
            if no_contractions > len(sentences) * 0.2:
                score += 0.2
        
        # Análisis de vocabulario - consistencia en nivel de formalidad
        formal_words = ['utilize', 'facilitate', 'implement', 'optimize', 'comprehensive', 'systematic', 'methodical']
        informal_words = ['use', 'help', 'do', 'make', 'big', 'good', 'nice', 'cool']
        
        formal_word_count = sum(1 for word in words if word.lower() in formal_words)
        informal_word_count = sum(1 for word in words if word.lower() in informal_words)
        
        if formal_word_count > 0 and informal_word_count == 0:
            if formal_word_count > len(words) * 0.05:
                score += 0.2
        elif informal_word_count > 0 and formal_word_count == 0:
            if informal_word_count > len(words) * 0.05:
                score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_stylistic_uniformity_patterns(self, text: str) -> float:
        """Detecta patrones de uniformidad estilística típicos de IA - NUEVO V34"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 5 or len(words) < 50:
            return 0.0
        
        # Análisis de uniformidad en longitud de oraciones
        sentence_lengths = [len(s.split()) for s in sentences]
        if len(sentence_lengths) > 0:
            avg_length = sum(sentence_lengths) / len(sentence_lengths)
            variance = sum((x - avg_length) ** 2 for x in sentence_lengths) / len(sentence_lengths)
            std_dev = variance ** 0.5
            coefficient_of_variation = std_dev / avg_length if avg_length > 0 else 0.0
            
            # IA tiende a tener uniformidad alta (baja variación)
            if coefficient_of_variation < 0.3:
                score += 0.3
            elif coefficient_of_variation < 0.4:
                score += 0.2
        
        # Análisis de uniformidad en estructura de oraciones
        sentence_structures = []
        for sentence in sentences:
            # Clasificar estructura básica
            if re.search(r'^[A-Z][^.!?]*\b(?:is|are|was|were)\b', sentence):
                sentence_structures.append('declarative_be')
            elif re.search(r'^[A-Z][^.!?]*\b(?:have|has|had)\b', sentence):
                sentence_structures.append('declarative_have')
            elif re.search(r'^[A-Z][^.!?]*\b(?:do|does|did|can|could|will|would)\b', sentence):
                sentence_structures.append('declarative_modal')
            elif re.search(r'^[A-Z][^.!?]*\?', sentence):
                sentence_structures.append('interrogative')
            else:
                sentence_structures.append('other')
        
        if len(sentence_structures) > 0:
            structure_counts = {}
            for struct in sentence_structures:
                structure_counts[struct] = structure_counts.get(struct, 0) + 1
            
            # Si una estructura domina mucho (uniformidad alta)
            max_structure_count = max(structure_counts.values())
            uniformity_ratio = max_structure_count / len(sentence_structures)
            
            if uniformity_ratio > 0.6:
                score += 0.3
            elif uniformity_ratio > 0.5:
                score += 0.2
        
        # Análisis de uniformidad en puntuación
        punctuation_types = {
            'comma': len(re.findall(r',', text)),
            'semicolon': len(re.findall(r';', text)),
            'colon': len(re.findall(r':', text)),
            'dash': len(re.findall(r'[-—]', text)),
            'parentheses': len(re.findall(r'[()]', text))
        }
        
        total_punctuation = sum(punctuation_types.values())
        if total_punctuation > 0:
            # Si un tipo de puntuación domina mucho
            max_punct_type = max(punctuation_types.values())
            punct_uniformity = max_punct_type / total_punctuation
            
            if punct_uniformity > 0.7:
                score += 0.2
        
        # Análisis de uniformidad en inicio de oraciones
        sentence_starts = []
        for sentence in sentences[:15]:
            words_in_sentence = sentence.split()
            if words_in_sentence:
                first_word = words_in_sentence[0].lower().strip('.,!?;:()[]{}"\'')
                if first_word:
                    sentence_starts.append(first_word)
        
        if len(sentence_starts) > 0:
            start_counts = {}
            for start in sentence_starts:
                start_counts[start] = start_counts.get(start, 0) + 1
            
            # Si hay mucha repetición en inicio de oraciones
            max_start_count = max(start_counts.values())
            start_uniformity = max_start_count / len(sentence_starts)
            
            if start_uniformity > 0.3:
                score += 0.3
            elif start_uniformity > 0.2:
                score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_phraseology_patterns(self, text: str) -> float:
        """Analiza patrones de fraseología típicos de IA - NUEVO V35"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Fraseologías típicas de IA (frases hechas comunes)
        ai_phraseologies = [
            r'\b(?:it is important to note|it should be noted|it must be noted)',
            r'\b(?:it is worth mentioning|it is worth noting|it is worth pointing out)',
            r'\b(?:it is clear that|it is evident that|it is obvious that)',
            r'\b(?:it is important to understand|it is crucial to understand|it is essential to understand)',
            r'\b(?:it is necessary to|it is essential to|it is crucial to|it is vital to)',
            r'\b(?:it is worth considering|it is worth examining|it is worth exploring)',
            r'\b(?:it can be seen that|it can be observed that|it can be noted that)',
            r'\b(?:it should be emphasized|it should be highlighted|it should be stressed)',
            r'\b(?:it is important to remember|it is crucial to remember|it is essential to remember)',
            r'\b(?:it is important to keep in mind|it is crucial to keep in mind)',
            r'\b(?:in other words|that is|i\.e\.|e\.g\.|to put it another way)',
            r'\b(?:for example|for instance|such as|namely|specifically)',
            r'\b(?:in conclusion|to conclude|to summarize|in summary)',
            r'\b(?:furthermore|moreover|additionally|in addition)',
            r'\b(?:however|nevertheless|nonetheless|on the other hand)',
            r'\b(?:therefore|thus|hence|consequently|as a result)'
        ]
        
        phraseology_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in ai_phraseologies)
        phraseology_density = phraseology_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA tiende a usar muchas fraseologías comunes
        if phraseology_density > 0.4:
            score += 0.4
        elif phraseology_density > 0.3:
            score += 0.3
        elif phraseology_density > 0.2:
            score += 0.2
        
        # Análisis de repetición de fraseologías
        # IA tiende a repetir las mismas fraseologías
        phraseology_instances = []
        for pattern in ai_phraseologies:
            matches = re.findall(pattern, text, re.IGNORECASE)
            phraseology_instances.extend(matches)
        
        if len(phraseology_instances) > 0:
            unique_phrases = len(set(phraseology_instances))
            repetition_ratio = 1 - (unique_phrases / len(phraseology_instances)) if len(phraseology_instances) > 0 else 0.0
            
            if repetition_ratio > 0.3:
                score += 0.3
            elif repetition_ratio > 0.2:
                score += 0.2
        
        # Análisis de fraseologías académicas excesivas
        academic_phrases = [
            r'\b(?:according to|based on|in accordance with|in line with)',
            r'\b(?:it has been shown|it has been demonstrated|it has been proven)',
            r'\b(?:research shows|studies show|research indicates|studies indicate)',
            r'\b(?:it is well-known|it is widely known|it is commonly known)',
            r'\b(?:it is established|it is proven|it is demonstrated|it is confirmed)'
        ]
        
        academic_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in academic_phrases)
        if academic_count > len(sentences) * 0.2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_collocation_patterns(self, text: str) -> float:
        """Detecta patrones de colocaciones típicos de IA - NUEVO V35"""
        score = 0.0
        words = [w.lower().strip('.,!?;:()[]{}"\'') for w in text.split()]
        words = [w for w in words if w and len(w) > 1]
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 50 or len(sentences) < 3:
            return 0.0
        
        # Colocaciones comunes en inglés (palabras que frecuentemente aparecen juntas)
        common_collocations = [
            ('make', 'decision'), ('take', 'action'), ('give', 'example'),
            ('play', 'role'), ('have', 'impact'), ('take', 'place'),
            ('make', 'sense'), ('have', 'effect'), ('take', 'care'),
            ('make', 'difference'), ('have', 'influence'), ('take', 'advantage'),
            ('provide', 'information'), ('conduct', 'research'), ('carry', 'out'),
            ('put', 'forward'), ('bring', 'about'), ('come', 'up'),
            ('deal', 'with'), ('look', 'into'), ('focus', 'on'),
            ('depend', 'on'), ('rely', 'on'), ('base', 'on'),
            ('lead', 'to'), ('result', 'in'), ('contribute', 'to')
        ]
        
        collocation_count = 0
        for word1, word2 in common_collocations:
            # Buscar patrones donde word1 y word2 aparecen cerca (dentro de 3 palabras)
            pattern = rf'\b{word1}\b(?:\s+\w+)?(?:\s+\w+)?\s+\b{word2}\b'
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            collocation_count += matches
        
        collocation_density = collocation_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA puede usar colocaciones de manera muy predecible o muy poco natural
        if collocation_density > 0.5:
            score += 0.3
        elif collocation_density > 0.3:
            score += 0.2
        elif collocation_density == 0 and len(words) > 200:
            # Textos largos sin colocaciones comunes pueden ser IA
            score += 0.2
        
        # Análisis de colocaciones académicas excesivas
        academic_collocations = [
            ('conduct', 'analysis'), ('carry', 'out'), ('perform', 'study'),
            ('undertake', 'research'), ('engage', 'in'), ('participate', 'in'),
            ('contribute', 'to'), ('lead', 'to'), ('result', 'from'),
            ('based', 'on'), ('according', 'to'), ('in', 'accordance')
        ]
        
        academic_colloc_count = 0
        for word1, word2 in academic_collocations:
            pattern = rf'\b{word1}\b(?:\s+\w+)?(?:\s+\w+)?\s+\b{word2}\b'
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            academic_colloc_count += matches
        
        if academic_colloc_count > len(sentences) * 0.2:
            score += 0.3
        elif academic_colloc_count > len(sentences) * 0.15:
            score += 0.2
        
        # Análisis de colocaciones poco naturales o forzadas
        # IA a veces crea colocaciones que suenan poco naturales
        unnatural_patterns = [
            r'\b(?:make|do|have|take)\s+(?:a|an|the)\s+\w+ing',
            r'\b(?:provide|give|offer)\s+(?:a|an|the)\s+\w+ing',
            r'\b(?:conduct|perform|carry)\s+(?:a|an|the)\s+\w+ing'
        ]
        
        unnatural_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in unnatural_patterns)
        if unnatural_count > 2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_idiomatic_patterns(self, text: str) -> float:
        """Analiza patrones idiomáticos típicos de IA - NUEVO V35"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Modismos comunes en inglés
        common_idioms = [
            r'\b(?:break the ice|break the mold|break new ground)',
            r'\b(?:hit the nail on the head|hit the mark|hit home)',
            r'\b(?:piece of cake|walk in the park|easy as pie)',
            r'\b(?:once in a blue moon|once in a lifetime|once and for all)',
            r'\b(?:the ball is in your court|the tables have turned)',
            r'\b(?:cost an arm and a leg|pay through the nose)',
            r'\b(?:barking up the wrong tree|beating around the bush)',
            r'\b(?:burn the midnight oil|burn bridges|burn out)',
            r'\b(?:call it a day|call the shots|call off)',
            r'\b(?:cut corners|cut to the chase|cut it out)',
            r'\b(?:get the ball rolling|get a grip|get over it)',
            r'\b(?:go the extra mile|go with the flow|go back on)',
            r'\b(?:keep an eye on|keep in mind|keep up)',
            r'\b(?:let the cat out of the bag|let off steam)',
            r'\b(?:make ends meet|make a long story short|make up)',
            r'\b(?:on cloud nine|on the same page|on thin ice)',
            r'\b(?:pull strings|pull through|pull off)',
            r'\b(?:put your foot down|put up with|put off)',
            r'\b(?:see eye to eye|see the light|see through)',
            r'\b(?:take it with a grain of salt|take the plunge|take off)'
        ]
        
        idiom_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in common_idioms)
        idiom_density = idiom_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA puede usar modismos de manera excesiva o muy poco
        if idiom_density > 0.3:
            score += 0.3
        elif idiom_density == 0 and len(words) > 200:
            # Textos largos sin modismos pueden ser IA
            score += 0.2
        
        # Análisis de uso incorrecto o literal de modismos
        # IA a veces usa modismos de manera incorrecta o demasiado literal
        literal_usage = [
            r'\b(?:break the ice)\s+(?:literally|actually|really)',
            r'\b(?:hit the nail)\s+(?:literally|actually|really)',
            r'\b(?:piece of cake)\s+(?:literally|actually|really)'
        ]
        
        literal_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in literal_usage)
        if literal_count > 0:
            score += 0.3
        
        # Análisis de modismos académicos o formales excesivos
        formal_idioms = [
            r'\b(?:in light of|in view of|in the context of)',
            r'\b(?:with regard to|with respect to|in relation to)',
            r'\b(?:for the purpose of|with the aim of|with a view to)',
            r'\b(?:in order to|so as to|with the intention of)',
            r'\b(?:by means of|by way of|through the use of)',
            r'\b(?:in terms of|in the case of|in the event of)',
            r'\b(?:on the basis of|on the grounds of|on account of)',
            r'\b(?:with reference to|in reference to|with regard to)'
        ]
        
        formal_idiom_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in formal_idioms)
        if formal_idiom_count > len(sentences) * 0.3:
            score += 0.3
        elif formal_idiom_count > len(sentences) * 0.2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_cultural_references_patterns(self, text: str) -> float:
        """Detecta patrones de referencias culturales típicos de IA - NUEVO V35"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Referencias culturales comunes (eventos, lugares, personas, etc.)
        cultural_references = [
            r'\b(?:world war|world war i|world war ii|wwi|wwii)',
            r'\b(?:united states|usa|america|american)',
            r'\b(?:united kingdom|uk|britain|british)',
            r'\b(?:european union|eu|europe)',
            r'\b(?:christmas|thanksgiving|easter|halloween)',
            r'\b(?:hollywood|broadway|silicon valley|wall street)',
            r'\b(?:nobel prize|oscar|grammy|pulitzer)',
            r'\b(?:olympic|olympics|world cup|super bowl)',
            r'\b(?:shakespeare|einstein|darwin|newton)',
            r'\b(?:renaissance|enlightenment|industrial revolution)',
            r'\b(?:great depression|cold war|civil war)',
            r'\b(?:9/11|september 11|twin towers)',
            r'\b(?:brexit|trump|biden|obama)',
            r'\b(?:facebook|google|apple|microsoft|amazon)',
            r'\b(?:iphone|android|windows|mac|linux)'
        ]
        
        cultural_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in cultural_references)
        cultural_density = cultural_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA puede usar referencias culturales de manera excesiva o muy genérica
        if cultural_density > 0.3:
            score += 0.3
        elif cultural_density == 0 and len(words) > 300:
            # Textos largos sin referencias culturales pueden ser IA
            score += 0.2
        
        # Análisis de referencias culturales genéricas o estereotípicas
        generic_cultural = [
            r'\b(?:as everyone knows|as is well-known|as is common knowledge)',
            r'\b(?:in western culture|in eastern culture|in modern society)',
            r'\b(?:in today\'s world|in the modern world|in contemporary society)',
            r'\b(?:throughout history|in human history|since ancient times)',
            r'\b(?:in many cultures|across cultures|in different cultures)',
            r'\b(?:in american culture|in european culture|in asian culture)'
        ]
        
        generic_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in generic_cultural)
        if generic_count > 2:
            score += 0.3
        elif generic_count > 1:
            score += 0.2
        
        # Análisis de referencias culturales desactualizadas o incorrectas
        outdated_references = [
            r'\b(?:y2k|millennium bug|dot-com bubble)',
            r'\b(?:my space|aol|yahoo|nokia)',
            r'\b(?:vhs|betamax|cassette tape|floppy disk)'
        ]
        
        outdated_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in outdated_references)
        if outdated_count > 0:
            score += 0.2
        
        # Análisis de falta de referencias culturales específicas o personales
        # IA tiende a evitar referencias culturales muy específicas o personales
        personal_cultural = [
            r'\b(?:in my country|in my culture|where i\'m from)',
            r'\b(?:in my experience|from my perspective|in my opinion)',
            r'\b(?:i remember|i recall|i\'ve seen|i\'ve heard)',
            r'\b(?:growing up|when i was|back when|in my day)'
        ]
        
        personal_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in personal_cultural)
        if personal_count == 0 and len(words) > 200:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_metaphorical_patterns(self, text: str) -> float:
        """Analiza patrones metafóricos típicos de IA - NUEVO V36"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Patrones metafóricos comunes
        metaphorical_patterns = [
            r'\b(?:like|as|as if|as though)\s+\w+',
            r'\b(?:is|are|was|were)\s+(?:like|as)\s+\w+',
            r'\b(?:metaphor|metaphorical|metaphorically)',
            r'\b(?:symbol|symbolic|symbolize|symbolism)',
            r'\b(?:represent|represents|representation|represents)',
            r'\b(?:embody|embodies|embodiment)',
            r'\b(?:personify|personifies|personification)',
            r'\b(?:compare|compares|comparison|compared to|compared with)',
            r'\b(?:similar to|similar as|akin to|reminiscent of)',
            r'\b(?:resemble|resembles|resemblance)',
            r'\b(?:mirror|mirrors|reflection|reflects)',
            r'\b(?:echo|echoes|echoing)',
            r'\b(?:evoke|evokes|evocation|evocative)',
            r'\b(?:conjure|conjures|conjuring)',
            r'\b(?:suggest|suggests|suggestion|suggestive)'
        ]
        
        metaphorical_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in metaphorical_patterns)
        metaphorical_density = metaphorical_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA puede usar metáforas de manera excesiva o muy poco
        if metaphorical_density > 0.4:
            score += 0.3
        elif metaphorical_density == 0 and len(words) > 200:
            # Textos largos sin metáforas pueden ser IA
            score += 0.2
        
        # Análisis de metáforas comunes o clichés
        cliche_metaphors = [
            r'\b(?:tip of the iceberg|light at the end of the tunnel|needle in a haystack)',
            r'\b(?:elephant in the room|elephant in the corner|800-pound gorilla)',
            r'\b(?:can of worms|pandora\'s box|opening pandora\'s box)',
            r'\b(?:double-edged sword|two sides of the coin|both sides of the coin)',
            r'\b(?:walking on eggshells|walking on thin ice|treading carefully)',
            r'\b(?:beating a dead horse|flogging a dead horse|beating around the bush)',
            r'\b(?:elephant in the room|elephant in the corner)',
            r'\b(?:elephant in the room|elephant in the corner)'
        ]
        
        cliche_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in cliche_metaphors)
        if cliche_count > 2:
            score += 0.3
        elif cliche_count > 1:
            score += 0.2
        
        # Análisis de metáforas forzadas o poco naturales
        forced_metaphors = [
            r'\b(?:is like|are like|was like|were like)\s+(?:a|an|the)\s+\w+\s+(?:that|which)',
            r'\b(?:can be compared to|can be likened to|can be equated with)',
            r'\b(?:serves as a metaphor|acts as a metaphor|functions as a metaphor)',
            r'\b(?:metaphorically speaking|in metaphorical terms|using a metaphor)'
        ]
        
        forced_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in forced_metaphors)
        if forced_count > 1:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_analogical_patterns(self, text: str) -> float:
        """Detecta patrones analógicos típicos de IA - NUEVO V36"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Patrones analógicos comunes
        analogical_patterns = [
            r'\b(?:analogy|analogous|analogously)',
            r'\b(?:similar to|similar as|akin to|reminiscent of)',
            r'\b(?:compare|compares|comparison|compared to|compared with)',
            r'\b(?:just as|just like|much like|very like)',
            r'\b(?:in the same way|in a similar way|in a similar manner)',
            r'\b(?:likewise|similarly|correspondingly|equally)',
            r'\b(?:parallel|parallels|parallelism|parallel to)',
            r'\b(?:equivalent|equivalents|equivalence|equivalent to)',
            r'\b(?:correspond|corresponds|correspondence|corresponds to)',
            r'\b(?:mirror|mirrors|mirroring|mirrors that)',
            r'\b(?:echo|echoes|echoing|echoes that)',
            r'\b(?:resemble|resembles|resemblance|resembles that)',
            r'\b(?:draw a parallel|draw parallels|drawing a parallel)',
            r'\b(?:make an analogy|make analogies|making an analogy)',
            r'\b(?:by analogy|through analogy|using an analogy)'
        ]
        
        analogical_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in analogical_patterns)
        analogical_density = analogical_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA puede usar analogías de manera excesiva o muy estructurada
        if analogical_density > 0.3:
            score += 0.4
        elif analogical_density > 0.2:
            score += 0.3
        elif analogical_density > 0.1:
            score += 0.2
        
        # Análisis de analogías estructuradas excesivamente
        structured_analogies = [
            r'\b(?:just as|just like)\s+\w+\s+(?:so|too|also)\s+\w+',
            r'\b(?:in the same way that|in the same manner that|in the same fashion that)',
            r'\b(?:similar to how|similar to the way|similar to the manner)',
            r'\b(?:by the same token|in the same vein|along the same lines)',
            r'\b(?:to draw an analogy|to make an analogy|to use an analogy)'
        ]
        
        structured_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in structured_analogies)
        if structured_count > 2:
            score += 0.3
        elif structured_count > 1:
            score += 0.2
        
        # Análisis de analogías forzadas o poco naturales
        forced_analogies = [
            r'\b(?:this can be analogized to|this can be compared to|this can be likened to)',
            r'\b(?:an analogy can be drawn|an analogy can be made|an analogy can be used)',
            r'\b(?:using the analogy of|through the analogy of|by means of an analogy)',
            r'\b(?:to use an analogy|to draw an analogy|to make an analogy)\s*[,:]'
        ]
        
        forced_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in forced_analogies)
        if forced_count > 1:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_irony_patterns(self, text: str) -> float:
        """Analiza patrones de ironía típicos de IA - NUEVO V36"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Marcadores de ironía
        irony_markers = [
            r'\b(?:ironically|ironic|irony)',
            r'\b(?:paradoxically|paradoxical|paradox)',
            r'\b(?:surprisingly|surprising|surprise)',
            r'\b(?:unexpectedly|unexpected|unexpected)',
            r'\b(?:ironically enough|ironic as it may seem|ironic though it may be)',
            r'\b(?:it is ironic|it\'s ironic|how ironic)',
            r'\b(?:the irony is|the irony of|the irony that)',
            r'\b(?:what\'s ironic|what is ironic|the ironic thing)',
            r'\b(?:in an ironic twist|in an ironic turn|in an ironic way)',
            r'\b(?:ironically|paradoxically|surprisingly)\s+(?:enough|though|however)'
        ]
        
        irony_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in irony_markers)
        irony_density = irony_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA puede usar marcadores de ironía de manera excesiva o muy explícita
        if irony_density > 0.2:
            score += 0.4
        elif irony_density > 0.1:
            score += 0.3
        
        # Análisis de ironía verbal explícita (típico de IA)
        explicit_irony = [
            r'\b(?:ironically|ironic|irony)\s+(?:speaking|stated|said|noted)',
            r'\b(?:it is ironic that|it\'s ironic that|how ironic that)',
            r'\b(?:the irony lies in|the irony is that|the irony of it is)',
            r'\b(?:this is ironic|that is ironic|which is ironic)',
            r'\b(?:one might find it ironic|some might find it ironic)'
        ]
        
        explicit_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in explicit_irony)
        if explicit_count > 1:
            score += 0.3
        elif explicit_count > 0:
            score += 0.2
        
        # Análisis de falta de ironía sutil o implícita
        # IA tiende a usar ironía muy explícita en lugar de sutil
        subtle_irony_indicators = [
            r'\b(?:of course|naturally|obviously|clearly)\s+(?:not|never|no)',
            r'\b(?:as if|as though)\s+(?:that|this|it)\s+(?:would|could|might)',
            r'\b(?:right|sure|yeah)\s+(?:right|sure|yeah)',
            r'\b(?:oh|ah|well)\s+(?:yes|no|sure|right)'
        ]
        
        subtle_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in subtle_irony_indicators)
        if irony_count > 0 and subtle_count == 0:
            # Si hay ironía pero solo explícita, puede ser IA
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_humor_patterns(self, text: str) -> float:
        """Detecta patrones de humor típicos de IA - NUEVO V36"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Marcadores de humor explícito
        humor_markers = [
            r'\b(?:funny|humor|humorous|humorously)',
            r'\b(?:joke|jokes|joking|jokingly)',
            r'\b(?:amusing|amusement|amusingly)',
            r'\b(?:witty|wittiness|wittily)',
            r'\b(?:hilarious|hilariously|hilarity)',
            r'\b(?:comical|comically|comedy)',
            r'\b(?:laugh|laughs|laughing|laughable)',
            r'\b(?:chuckle|chuckles|chuckling)',
            r'\b(?:giggle|giggles|giggling)',
            r'\b(?:smile|smiles|smiling)',
            r'\b(?:grin|grins|grinning)',
            r'\b(?:ha ha|haha|hehe|lol|lmao|rofl)',
            r'\b(?:that\'s funny|that is funny|how funny)',
            r'\b(?:that\'s amusing|that is amusing|how amusing)',
            r'\b(?:that\'s hilarious|that is hilarious|how hilarious)'
        ]
        
        humor_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in humor_markers)
        humor_density = humor_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA puede usar marcadores de humor de manera excesiva o muy explícita
        if humor_density > 0.2:
            score += 0.4
        elif humor_density > 0.1:
            score += 0.3
        
        # Análisis de chistes o bromas estructuradas
        structured_humor = [
            r'\b(?:here\'s a joke|here is a joke|let me tell you a joke)',
            r'\b(?:that reminds me of a joke|that reminds me of a story)',
            r'\b(?:speaking of|on the subject of|in that vein)',
            r'\b(?:to lighten the mood|to break the ice|to add some humor)',
            r'\b(?:as a joke|as a humorous aside|as a funny note)',
            r'\b(?:jokingly|humorously|amusingly)\s+(?:speaking|stated|said|noted)'
        ]
        
        structured_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in structured_humor)
        if structured_count > 1:
            score += 0.3
        elif structured_count > 0:
            score += 0.2
        
        # Análisis de falta de humor sutil o espontáneo
        # IA tiende a usar humor muy explícito o estructurado
        subtle_humor_indicators = [
            r'\b(?:wink|winks|winking)',
            r'\b(?:nudge|nudges|nudging)',
            r'\b(?:tongue in cheek|tongue-in-cheek)',
            r'\b(?:with a smile|with a grin|with a chuckle)',
            r'\b(?:playfully|teasingly|jokingly)\s+(?:said|stated|noted)'
        ]
        
        subtle_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in subtle_humor_indicators)
        if humor_count > 0 and subtle_count == 0:
            # Si hay humor pero solo explícito, puede ser IA
            score += 0.2
        
        # Análisis de chistes o bromas poco naturales o forzadas
        forced_humor = [
            r'\b(?:ha ha|haha|hehe)\s+(?:ha ha|haha|hehe)',
            r'\b(?:that\'s a good one|that\'s a funny one|that\'s hilarious)',
            r'\b(?:now that\'s funny|now that is funny|now that\'s hilarious)',
            r'\b(?:how funny|how amusing|how hilarious)\s+(?:that|this|it)'
        ]
        
        forced_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in forced_humor)
        if forced_count > 1:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_sarcasm_patterns(self, text: str) -> float:
        """Analiza patrones de sarcasmo típicos de IA - NUEVO V37"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Marcadores de sarcasmo explícito
        sarcasm_markers = [
            r'\b(?:sarcastically|sarcastic|sarcasm)',
            r'\b(?:yeah right|sure thing|oh really|oh sure)',
            r'\b(?:as if|as though)\s+(?:that|this|it)\s+(?:would|could|might)',
            r'\b(?:right|sure|yeah)\s+(?:right|sure|yeah)',
            r'\b(?:oh|ah|well)\s+(?:yes|no|sure|right)',
            r'\b(?:of course|naturally|obviously|clearly)\s+(?:not|never|no)',
            r'\b(?:that\'s|that is)\s+(?:great|wonderful|fantastic|amazing)\s+(?:sarcastically|sarcastic)',
            r'\b(?:how|what)\s+(?:great|wonderful|fantastic|amazing)\s+(?:that|this|it)',
            r'\b(?:isn\'t|aren\'t|wasn\'t|weren\'t)\s+(?:that|this|it)\s+(?:great|wonderful|fantastic)',
            r'\b(?:i\'m|i am)\s+(?:so|very|really|extremely)\s+(?:happy|glad|pleased|thrilled)'
        ]
        
        sarcasm_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in sarcasm_markers)
        sarcasm_density = sarcasm_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA puede usar sarcasmo de manera excesiva o muy explícita
        if sarcasm_density > 0.2:
            score += 0.4
        elif sarcasm_density > 0.1:
            score += 0.3
        
        # Análisis de sarcasmo explícito (típico de IA)
        explicit_sarcasm = [
            r'\b(?:sarcastically|sarcastic|sarcasm)\s+(?:speaking|stated|said|noted)',
            r'\b(?:it is sarcastic|it\'s sarcastic|how sarcastic)',
            r'\b(?:the sarcasm is|the sarcasm of|the sarcasm that)',
            r'\b(?:what\'s sarcastic|what is sarcastic|the sarcastic thing)',
            r'\b(?:in a sarcastic way|in a sarcastic manner|sarcastically speaking)'
        ]
        
        explicit_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in explicit_sarcasm)
        if explicit_count > 1:
            score += 0.3
        elif explicit_count > 0:
            score += 0.2
        
        # Análisis de falta de sarcasmo sutil o implícito
        # IA tiende a usar sarcasmo muy explícito en lugar de sutil
        subtle_sarcasm_indicators = [
            r'\b(?:wink|winks|winking)',
            r'\b(?:nudge|nudges|nudging)',
            r'\b(?:tongue in cheek|tongue-in-cheek)',
            r'\b(?:with a smirk|with a grin|with a raised eyebrow)',
            r'\b(?:playfully|teasingly|sarcastically)\s+(?:said|stated|noted)'
        ]
        
        subtle_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in subtle_sarcasm_indicators)
        if sarcasm_count > 0 and subtle_count == 0:
            # Si hay sarcasmo pero solo explícito, puede ser IA
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_hyperbole_patterns(self, text: str) -> float:
        """Detecta patrones de hipérbole típicos de IA - NUEVO V37"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Marcadores de hipérbole
        hyperbole_markers = [
            r'\b(?:extremely|incredibly|unbelievably|absolutely|completely|totally|utterly)',
            r'\b(?:the most|the best|the worst|the greatest|the smallest|the largest)',
            r'\b(?:never|always|forever|eternally|infinitely|endlessly)',
            r'\b(?:every|all|none|nothing|everything|everyone|nobody)',
            r'\b(?:millions|billions|trillions|countless|innumerable|myriad)',
            r'\b(?:perfect|perfectly|flawless|flawlessly|impeccable|impeccably)',
            r'\b(?:amazing|amazingly|astounding|astoundingly|stunning|stunningly)',
            r'\b(?:incredible|incredibly|unbelievable|unbelievably|remarkable|remarkably)',
            r'\b(?:phenomenal|phenomenally|extraordinary|extraordinarily|exceptional|exceptionally)',
            r'\b(?:outstanding|outstandingly|exceptional|exceptionally|remarkable|remarkably)',
            r'\b(?:beyond|exceed|exceeds|exceeding|surpass|surpasses|surpassing)',
            r'\b(?:unprecedented|unprecedentedly|unparalleled|unparalleled)',
            r'\b(?:revolutionary|revolutionarily|groundbreaking|groundbreakingly)',
            r'\b(?:life-changing|world-changing|game-changing|mind-blowing)',
            r'\b(?:once in a lifetime|once in a million|one of a kind|unique)'
        ]
        
        hyperbole_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in hyperbole_markers)
        hyperbole_density = hyperbole_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA puede usar hipérbole de manera excesiva
        if hyperbole_density > 0.4:
            score += 0.4
        elif hyperbole_density > 0.3:
            score += 0.3
        elif hyperbole_density > 0.2:
            score += 0.2
        
        # Análisis de superlativos excesivos
        superlatives = [
            r'\b(?:the most|the best|the worst|the greatest|the smallest|the largest)',
            r'\b(?:most|best|worst|greatest|smallest|largest)\s+\w+',
            r'\b(?:more|better|worse|greater|smaller|larger)\s+than\s+(?:any|all|every)',
            r'\b(?:better|worse|greater|smaller|larger)\s+than\s+(?:ever|before|anything|anyone)'
        ]
        
        superlative_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in superlatives)
        if superlative_count > len(sentences) * 0.2:
            score += 0.3
        elif superlative_count > len(sentences) * 0.15:
            score += 0.2
        
        # Análisis de absolutos excesivos
        absolutes = [
            r'\b(?:never|always|forever|eternally|infinitely|endlessly)',
            r'\b(?:every|all|none|nothing|everything|everyone|nobody)',
            r'\b(?:completely|totally|utterly|absolutely|entirely|fully|wholly)',
            r'\b(?:without|no|not)\s+(?:exception|doubt|question|fail|error)'
        ]
        
        absolute_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in absolutes)
        if absolute_count > len(sentences) * 0.3:
            score += 0.3
        elif absolute_count > len(sentences) * 0.2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_euphemism_patterns(self, text: str) -> float:
        """Analiza patrones de eufemismo típicos de IA - NUEVO V37"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Eufemismos comunes
        euphemisms = [
            r'\b(?:passed away|passed on|no longer with us|left us)',
            r'\b(?:let go|let go of|released|downsized|rightsized)',
            r'\b(?:economically disadvantaged|financially challenged|low-income)',
            r'\b(?:vertically challenged|height-challenged|petite)',
            r'\b(?:differently abled|physically challenged|special needs)',
            r'\b(?:senior|senior citizen|elderly|golden years)',
            r'\b(?:restroom|bathroom|washroom|powder room)',
            r'\b(?:correctional facility|detention center|penal institution)',
            r'\b(?:pre-owned|pre-loved|gently used|previously owned)',
            r'\b(?:between jobs|in transition|exploring opportunities)',
            r'\b(?:enhanced interrogation|aggressive questioning|intensive questioning)',
            r'\b(?:collateral damage|unintended consequences|side effects)',
            r'\b(?:friendly fire|blue on blue|accidental engagement)',
            r'\b(?:economical|budget-friendly|cost-effective|affordable)',
            r'\b(?:full-figured|plus-sized|curvy|voluptuous)'
        ]
        
        euphemism_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in euphemisms)
        euphemism_density = euphemism_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA puede usar eufemismos de manera excesiva o muy formal
        if euphemism_density > 0.3:
            score += 0.4
        elif euphemism_density > 0.2:
            score += 0.3
        elif euphemism_density > 0.1:
            score += 0.2
        
        # Análisis de eufemismos corporativos o formales excesivos
        corporate_euphemisms = [
            r'\b(?:rightsizing|downsizing|restructuring|reorganizing)',
            r'\b(?:synergy|synergistic|synergize|synergizing)',
            r'\b(?:leverage|leveraging|utilize|utilizing)',
            r'\b(?:paradigm shift|paradigm|paradigmatic)',
            r'\b(?:value-added|value proposition|value creation)',
            r'\b(?:best practices|industry standard|benchmark)',
            r'\b(?:core competency|competitive advantage|market position)',
            r'\b(?:stakeholder|stakeholders|key stakeholder)',
            r'\b(?:deliverable|deliverables|key deliverable)',
            r'\b(?:action item|action items|actionable)'
        ]
        
        corporate_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in corporate_euphemisms)
        if corporate_count > len(sentences) * 0.2:
            score += 0.3
        elif corporate_count > len(sentences) * 0.15:
            score += 0.2
        
        # Análisis de eufemismos políticos o diplomáticos
        political_euphemisms = [
            r'\b(?:enhanced interrogation|aggressive questioning|intensive questioning)',
            r'\b(?:collateral damage|unintended consequences|side effects)',
            r'\b(?:friendly fire|blue on blue|accidental engagement)',
            r'\b(?:preemptive strike|preventive action|defensive measure)',
            r'\b(?:regime change|democratic transition|political transition)',
            r'\b(?:ethnic cleansing|population transfer|forced migration)',
            r'\b(?:extraordinary rendition|enhanced transfer|special transfer)'
        ]
        
        political_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in political_euphemisms)
        if political_count > 1:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_understatement_patterns(self, text: str) -> float:
        """Detecta patrones de lítote típicos de IA - NUEVO V37"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Patrones de lítote (understatement)
        understatement_patterns = [
            r'\b(?:not bad|not too bad|not terrible|not awful)',
            r'\b(?:not bad at all|not too shabby|not half bad)',
            r'\b(?:not the worst|not the best|not great|not terrible)',
            r'\b(?:not exactly|not quite|not really|not entirely)',
            r'\b(?:not un|not in|not without|not lacking)',
            r'\b(?:somewhat|rather|quite|fairly|pretty|a bit|a little)',
            r'\b(?:less than|more than|not more than|not less than)',
            r'\b(?:not insignificant|not unimportant|not trivial)',
            r'\b(?:not inconsiderable|not immaterial|not negligible)',
            r'\b(?:not unimpressive|not unremarkable|not unnoteworthy)',
            r'\b(?:could be worse|could be better|could be worse)',
            r'\b(?:it\'s not nothing|it\'s not bad|it\'s not terrible)',
            r'\b(?:not to be underestimated|not to be overlooked|not to be ignored)',
            r'\b(?:not without merit|not without value|not without importance)',
            r'\b(?:not exactly small|not exactly minor|not exactly insignificant)'
        ]
        
        understatement_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in understatement_patterns)
        understatement_density = understatement_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA puede usar lítote de manera excesiva o muy estructurada
        if understatement_density > 0.3:
            score += 0.4
        elif understatement_density > 0.2:
            score += 0.3
        elif understatement_density > 0.1:
            score += 0.2
        
        # Análisis de dobles negaciones (típico de lítote)
        double_negations = [
            r'\b(?:not un|not in|not without|not lacking)',
            r'\b(?:not insignificant|not unimportant|not trivial)',
            r'\b(?:not inconsiderable|not immaterial|not negligible)',
            r'\b(?:not unimpressive|not unremarkable|not unnoteworthy)',
            r'\b(?:not without merit|not without value|not without importance)',
            r'\b(?:not exactly small|not exactly minor|not exactly insignificant)'
        ]
        
        double_neg_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in double_negations)
        if double_neg_count > 2:
            score += 0.3
        elif double_neg_count > 1:
            score += 0.2
        
        # Análisis de atenuadores excesivos
        attenuators = [
            r'\b(?:somewhat|rather|quite|fairly|pretty|a bit|a little)',
            r'\b(?:more or less|sort of|kind of|rather|quite)',
            r'\b(?:to some extent|to a certain extent|to some degree)',
            r'\b(?:in some way|in a way|in some sense|in a sense)',
            r'\b(?:relatively|comparatively|relatively speaking)'
        ]
        
        attenuator_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in attenuators)
        if attenuator_count > len(sentences) * 0.2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_alliteration_patterns(self, text: str) -> float:
        """Analiza patrones de aliteración típicos de IA - NUEVO V38"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Detectar aliteración (repetición de sonidos consonánticos iniciales)
        alliteration_count = 0
        for sentence in sentences:
            words_in_sentence = [w.lower().strip('.,!?;:()[]{}"\'') for w in sentence.split()]
            words_in_sentence = [w for w in words_in_sentence if w and len(w) > 1]
            
            if len(words_in_sentence) >= 3:
                # Buscar secuencias de palabras que comienzan con la misma letra
                consecutive_alliteration = 0
                for i in range(len(words_in_sentence) - 2):
                    first_letters = [w[0] for w in words_in_sentence[i:i+3] if w[0].isalpha()]
                    if len(first_letters) >= 3 and len(set(first_letters)) == 1:
                        consecutive_alliteration += 1
                        alliteration_count += 1
                
                # También buscar aliteración en palabras cercanas
                for i in range(len(words_in_sentence) - 1):
                    if words_in_sentence[i][0].isalpha() and words_in_sentence[i+1][0].isalpha():
                        if words_in_sentence[i][0] == words_in_sentence[i+1][0]:
                            alliteration_count += 0.5
        
        alliteration_density = alliteration_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA puede usar aliteración de manera excesiva o muy estructurada
        if alliteration_density > 0.5:
            score += 0.4
        elif alliteration_density > 0.3:
            score += 0.3
        elif alliteration_density > 0.2:
            score += 0.2
        
        # Análisis de aliteración forzada o poco natural
        # IA a veces crea aliteración que suena forzada
        forced_alliteration_patterns = [
            r'\b(?:perfect|precise|precise|precise)',
            r'\b(?:comprehensive|complete|comprehensive|complete)',
            r'\b(?:systematic|structured|systematic|structured)',
            r'\b(?:methodical|meticulous|methodical|meticulous)'
        ]
        
        forced_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in forced_alliteration_patterns)
        if forced_count > 1:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_assonance_patterns(self, text: str) -> float:
        """Detecta patrones de asonancia típicos de IA - NUEVO V38"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Detectar asonancia (repetición de sonidos vocálicos)
        # Extraer vocales de palabras
        def extract_vowels(word):
            vowels = 'aeiou'
            return ''.join(c for c in word.lower() if c in vowels)
        
        assonance_count = 0
        for sentence in sentences:
            words_in_sentence = [w.lower().strip('.,!?;:()[]{}"\'') for w in sentence.split()]
            words_in_sentence = [w for w in words_in_sentence if w and len(w) > 1]
            
            if len(words_in_sentence) >= 3:
                # Buscar secuencias de palabras con sonidos vocálicos similares
                vowel_sequences = [extract_vowels(w) for w in words_in_sentence[:5]]
                
                for i in range(len(vowel_sequences) - 1):
                    if vowel_sequences[i] and vowel_sequences[i+1]:
                        # Si las secuencias vocálicas son similares
                        if vowel_sequences[i] == vowel_sequences[i+1] or \
                           (len(vowel_sequences[i]) > 0 and len(vowel_sequences[i+1]) > 0 and 
                            vowel_sequences[i][0] == vowel_sequences[i+1][0]):
                            assonance_count += 0.5
        
        assonance_density = assonance_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA puede usar asonancia de manera excesiva
        if assonance_density > 0.4:
            score += 0.3
        elif assonance_density > 0.3:
            score += 0.2
        
        # Análisis de asonancia en palabras comunes (puede ser accidental)
        common_words = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        common_assonance = sum(1 for w in words if w.lower() in common_words)
        if assonance_count > 0 and common_assonance / len(words) > 0.3:
            # Si hay mucha asonancia pero muchas palabras comunes, puede ser accidental
            score += 0.1
        
        return min(score, 1.0)
    
    def _analyze_ai_rhythm_patterns(self, text: str) -> float:
        """Analiza patrones de ritmo poético típicos de IA - NUEVO V38"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 5 or len(words) < 50:
            return 0.0
        
        # Análisis de ritmo basado en sílabas (aproximación)
        # Contar sílabas aproximadas por palabra (palabras largas = más sílabas)
        def approximate_syllables(word):
            word = word.lower().strip('.,!?;:()[]{}"\'')
            if not word:
                return 0
            # Aproximación simple: contar grupos de vocales
            vowels = 'aeiouy'
            syllable_count = 0
            prev_was_vowel = False
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_was_vowel:
                    syllable_count += 1
                prev_was_vowel = is_vowel
            return max(1, syllable_count)
        
        sentence_syllable_counts = []
        for sentence in sentences[:10]:
            words_in_sentence = sentence.split()
            total_syllables = sum(approximate_syllables(w) for w in words_in_sentence)
            sentence_syllable_counts.append(total_syllables)
        
        if len(sentence_syllable_counts) > 0:
            # Análisis de uniformidad en ritmo
            avg_syllables = sum(sentence_syllable_counts) / len(sentence_syllable_counts)
            variance = sum((x - avg_syllables) ** 2 for x in sentence_syllable_counts) / len(sentence_syllable_counts)
            std_dev = variance ** 0.5
            coefficient_of_variation = std_dev / avg_syllables if avg_syllables > 0 else 0.0
            
            # IA tiende a tener ritmo muy uniforme
            if coefficient_of_variation < 0.2:
                score += 0.4
            elif coefficient_of_variation < 0.3:
                score += 0.3
            elif coefficient_of_variation < 0.4:
                score += 0.2
        
        # Análisis de patrones rítmicos repetitivos
        # Buscar secuencias de sílabas similares
        if len(sentence_syllable_counts) >= 4:
            patterns = []
            for i in range(len(sentence_syllable_counts) - 3):
                pattern = tuple(sentence_syllable_counts[i:i+4])
                patterns.append(pattern)
            
            if patterns:
                pattern_counts = {}
                for pattern in patterns:
                    pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
                
                max_pattern_count = max(pattern_counts.values()) if pattern_counts else 0
                if max_pattern_count > 1:
                    score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_poetic_patterns(self, text: str) -> float:
        """Detecta patrones poéticos típicos de IA - NUEVO V38"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Patrones poéticos comunes
        poetic_patterns = [
            r'\b(?:like|as|as if|as though)\s+\w+',
            r'\b(?:metaphor|metaphorical|metaphorically|simile)',
            r'\b(?:rhyme|rhymes|rhyming|rhymed)',
            r'\b(?:verse|verses|versification|poetic)',
            r'\b(?:stanza|stanzas|strophic|strophic)',
            r'\b(?:alliteration|alliterative|alliteratively)',
            r'\b(?:assonance|assonant|assonantly)',
            r'\b(?:consonance|consonant|consonantly)',
            r'\b(?:meter|metrical|metrically|iambic|trochaic|anapestic|dactylic)',
            r'\b(?:sonnet|sonnets|haiku|haikus|limerick|limericks)',
            r'\b(?:ode|odes|elegy|elegies|ballad|ballads)',
            r'\b(?:imagery|imagistic|imagistically)',
            r'\b(?:symbol|symbolic|symbolize|symbolism)',
            r'\b(?:personification|personify|personifies)',
            r'\b(?:enjambment|enjambed|enjambing)'
        ]
        
        poetic_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in poetic_patterns)
        poetic_density = poetic_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA puede usar términos poéticos de manera excesiva o muy explícita
        if poetic_density > 0.2:
            score += 0.4
        elif poetic_density > 0.1:
            score += 0.3
        
        # Análisis de estructura poética explícita
        explicit_poetic = [
            r'\b(?:in poetic terms|in poetic language|poetically speaking)',
            r'\b(?:using poetic devices|employing poetic techniques|through poetic means)',
            r'\b(?:the poem|the verse|the stanza|the line)',
            r'\b(?:the poet|the writer|the author)\s+(?:uses|employs|utilizes)',
            r'\b(?:this poem|this verse|this stanza|this line)'
        ]
        
        explicit_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in explicit_poetic)
        if explicit_count > 1:
            score += 0.3
        elif explicit_count > 0:
            score += 0.2
        
        # Análisis de falta de elementos poéticos naturales
        # IA tiende a usar términos poéticos explícitos en lugar de crear poesía natural
        natural_poetic_indicators = [
            r'\b(?:whisper|whispers|whispering|whispered)',
            r'\b(?:dance|dances|dancing|danced)',
            r'\b(?:sing|sings|singing|sang)',
            r'\b(?:flow|flows|flowing|flowed)',
            r'\b(?:glow|glows|glowing|glowed)',
            r'\b(?:shine|shines|shining|shone)',
            r'\b(?:sparkle|sparkles|sparkling|sparkled)',
            r'\b(?:twinkle|twinkles|twinkling|twinkled)'
        ]
        
        natural_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in natural_poetic_indicators)
        if poetic_count > 0 and natural_count == 0:
            # Si hay términos poéticos pero no elementos poéticos naturales, puede ser IA
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_lexical_density_patterns(self, text: str) -> float:
        """Analiza patrones de densidad léxica típicos de IA - NUEVO V39"""
        score = 0.0
        words = [w.lower().strip('.,!?;:()[]{}"\'') for w in text.split()]
        words = [w for w in words if w and len(w) > 1]
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 50 or len(sentences) < 3:
            return 0.0
        
        # Calcular densidad léxica (proporción de palabras de contenido vs palabras funcionales)
        content_words = []
        function_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        
        for word in words:
            if word not in function_words and len(word) > 2:
                content_words.append(word)
        
        lexical_density = len(content_words) / len(words) if len(words) > 0 else 0.0
        
        # IA puede tener densidad léxica muy alta o muy baja
        if lexical_density > 0.70:
            score += 0.4
        elif lexical_density > 0.65:
            score += 0.3
        elif lexical_density < 0.40:
            score += 0.3
        elif lexical_density < 0.45:
            score += 0.2
        
        # Análisis de densidad léxica por oración
        sentence_lexical_densities = []
        for sentence in sentences:
            sentence_words = [w.lower().strip('.,!?;:()[]{}"\'') for w in sentence.split()]
            sentence_words = [w for w in sentence_words if w]
            sentence_content = [w for w in sentence_words if w not in function_words and len(w) > 2]
            if len(sentence_words) > 0:
                sent_density = len(sentence_content) / len(sentence_words)
                sentence_lexical_densities.append(sent_density)
        
        if len(sentence_lexical_densities) > 0:
            # Análisis de uniformidad en densidad léxica
            avg_density = sum(sentence_lexical_densities) / len(sentence_lexical_densities)
            variance = sum((x - avg_density) ** 2 for x in sentence_lexical_densities) / len(sentence_lexical_densities)
            std_dev = variance ** 0.5
            coefficient_of_variation = std_dev / avg_density if avg_density > 0 else 0.0
            
            # IA tiende a tener densidad léxica muy uniforme
            if coefficient_of_variation < 0.15:
                score += 0.3
            elif coefficient_of_variation < 0.20:
                score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_semantic_network_patterns(self, text: str) -> float:
        """Detecta patrones de redes semánticas típicos de IA - NUEVO V39"""
        score = 0.0
        words = [w.lower().strip('.,!?;:()[]{}"\'') for w in text.split()]
        words = [w for w in words if w and len(w) > 1]
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 50 or len(sentences) < 3:
            return 0.0
        
        # Análisis de relaciones semánticas entre palabras
        # Contar palabras relacionadas semánticamente que aparecen juntas
        semantic_clusters = {
            'technology': ['computer', 'software', 'hardware', 'system', 'network', 'digital', 'electronic', 'device', 'application', 'platform'],
            'business': ['company', 'organization', 'management', 'strategy', 'market', 'customer', 'product', 'service', 'revenue', 'profit'],
            'academic': ['research', 'study', 'analysis', 'methodology', 'theory', 'hypothesis', 'data', 'evidence', 'conclusion', 'result'],
            'science': ['experiment', 'observation', 'hypothesis', 'theory', 'data', 'evidence', 'analysis', 'conclusion', 'method', 'result'],
            'general': ['important', 'significant', 'crucial', 'essential', 'fundamental', 'key', 'main', 'primary', 'principal', 'major']
        }
        
        cluster_counts = {}
        for cluster_name, cluster_words in semantic_clusters.items():
            count = sum(1 for word in words if word in cluster_words)
            if count > 0:
                cluster_counts[cluster_name] = count
        
        # IA tiende a tener clusters semánticos muy concentrados
        if cluster_counts:
            total_cluster_words = sum(cluster_counts.values())
            max_cluster = max(cluster_counts.values())
            concentration = max_cluster / total_cluster_words if total_cluster_words > 0 else 0.0
            
            if concentration > 0.7:
                score += 0.4
            elif concentration > 0.6:
                score += 0.3
            elif concentration > 0.5:
                score += 0.2
        
        # Análisis de co-ocurrencia de palabras relacionadas
        # IA tiende a usar palabras relacionadas de manera muy estructurada
        related_word_pairs = [
            ('research', 'study'), ('analysis', 'data'), ('method', 'approach'),
            ('strategy', 'plan'), ('system', 'process'), ('theory', 'practice'),
            ('problem', 'solution'), ('cause', 'effect'), ('input', 'output'),
            ('beginning', 'end'), ('start', 'finish'), ('first', 'last')
        ]
        
        co_occurrence_count = 0
        for word1, word2 in related_word_pairs:
            if word1 in words and word2 in words:
                # Verificar si aparecen cerca (dentro de 20 palabras)
                indices1 = [i for i, w in enumerate(words) if w == word1]
                indices2 = [i for i, w in enumerate(words) if w == word2]
                
                for idx1 in indices1:
                    for idx2 in indices2:
                        if abs(idx1 - idx2) <= 20:
                            co_occurrence_count += 1
                            break
                    if co_occurrence_count > 0:
                        break
        
        if co_occurrence_count > len(sentences) * 0.3:
            score += 0.3
        elif co_occurrence_count > len(sentences) * 0.2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_conceptual_coherence_patterns(self, text: str) -> float:
        """Analiza patrones de coherencia conceptual típicos de IA - NUEVO V39"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 5 or len(words) < 50:
            return 0.0
        
        # Análisis de coherencia conceptual excesiva
        # IA tiende a mantener coherencia conceptual muy alta y estructurada
        
        # Contar conceptos clave repetidos
        concept_words = [w.lower().strip('.,!?;:()[]{}"\'') for w in words if len(w) > 4]
        concept_frequency = {}
        for word in concept_words:
            concept_frequency[word] = concept_frequency.get(word, 0) + 1
        
        # Conceptos que aparecen muchas veces
        high_frequency_concepts = {word: count for word, count in concept_frequency.items() if count > 2}
        
        if len(high_frequency_concepts) > 0:
            # Calcular concentración de conceptos
            total_concept_occurrences = sum(high_frequency_concepts.values())
            max_concept_occurrences = max(high_frequency_concepts.values())
            concept_concentration = max_concept_occurrences / total_concept_occurrences if total_concept_occurrences > 0 else 0.0
            
            if concept_concentration > 0.4:
                score += 0.3
            elif concept_concentration > 0.3:
                score += 0.2
        
        # Análisis de transiciones conceptuales
        # IA tiende a tener transiciones conceptuales muy suaves y estructuradas
        transition_markers = [
            r'\b(?:moving to|turning to|shifting to|transitioning to)',
            r'\b(?:in relation to|in connection with|in reference to)',
            r'\b(?:building on|building upon|expanding on|expanding upon)',
            r'\b(?:related to|connected to|linked to|associated with)',
            r'\b(?:similar to|similar as|comparable to|comparable with)',
            r'\b(?:in contrast to|in contrast with|contrary to|unlike)'
        ]
        
        transition_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in transition_markers)
        if transition_count > len(sentences) * 0.2:
            score += 0.3
        elif transition_count > len(sentences) * 0.15:
            score += 0.2
        
        # Análisis de coherencia temática perfecta
        # IA tiende a mantener coherencia temática muy alta sin desviaciones
        if len(sentences) > 5:
            # Analizar distribución de palabras clave
            key_words = [word for word, count in concept_frequency.items() if count > 1]
            if len(key_words) > 0:
                # Contar cuántas oraciones contienen palabras clave
                sentences_with_keywords = 0
                for sentence in sentences:
                    sentence_words = [w.lower().strip('.,!?;:()[]{}"\'') for w in sentence.split()]
                    if any(keyword in sentence_words for keyword in key_words):
                        sentences_with_keywords += 1
                
                keyword_coverage = sentences_with_keywords / len(sentences) if len(sentences) > 0 else 0.0
                # Cobertura muy alta puede indicar IA
                if keyword_coverage > 0.9:
                    score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_knowledge_graph_patterns(self, text: str) -> float:
        """Detecta patrones de grafos de conocimiento típicos de IA - NUEVO V39"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Patrones de relaciones de conocimiento estructuradas (típico de IA)
        knowledge_relations = [
            r'\b(?:is a|are a|was a|were a)\s+\w+',
            r'\b(?:is an|are an|was an|were an)\s+\w+',
            r'\b(?:is part of|are part of|was part of|were part of)',
            r'\b(?:belongs to|belong to|belonged to)',
            r'\b(?:is related to|are related to|was related to|were related to)',
            r'\b(?:is connected to|are connected to|was connected to|were connected to)',
            r'\b(?:is associated with|are associated with|was associated with|were associated with)',
            r'\b(?:is linked to|are linked to|was linked to|were linked to)',
            r'\b(?:is a type of|are a type of|was a type of|were a type of)',
            r'\b(?:is a kind of|are a kind of|was a kind of|were a kind of)',
            r'\b(?:is a form of|are a form of|was a form of|were a form of)',
            r'\b(?:is an example of|are an example of|was an example of|were an example of)',
            r'\b(?:consists of|consist of|consisted of)',
            r'\b(?:comprises|comprise|comprised)',
            r'\b(?:includes|include|included)',
            r'\b(?:contains|contain|contained)',
            r'\b(?:involves|involve|involved)',
            r'\b(?:requires|require|required)',
            r'\b(?:depends on|depend on|depended on)',
            r'\b(?:leads to|lead to|led to)',
            r'\b(?:results in|result in|resulted in)',
            r'\b(?:causes|cause|caused)',
            r'\b(?:affects|affect|affected)',
            r'\b(?:influences|influence|influenced)'
        ]
        
        relation_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in knowledge_relations)
        relation_density = relation_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA tiende a usar muchas relaciones de conocimiento estructuradas
        if relation_density > 0.4:
            score += 0.4
        elif relation_density > 0.3:
            score += 0.3
        elif relation_density > 0.2:
            score += 0.2
        
        # Análisis de jerarquías de conocimiento
        hierarchy_patterns = [
            r'\b(?:category|categories|classify|classification)',
            r'\b(?:hierarchy|hierarchical|hierarchically)',
            r'\b(?:level|levels|tier|tiers)',
            r'\b(?:subcategory|subcategories|subclass|subclasses)',
            r'\b(?:parent|parents|child|children)',
            r'\b(?:superordinate|subordinate|superordinate|subordinate)',
            r'\b(?:general|specific|generalization|specialization)',
            r'\b(?:abstract|concrete|abstraction|concretion)'
        ]
        
        hierarchy_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in hierarchy_patterns)
        if hierarchy_count > 2:
            score += 0.3
        elif hierarchy_count > 1:
            score += 0.2
        
        # Análisis de taxonomías estructuradas
        taxonomy_patterns = [
            r'\b(?:first|second|third|fourth|fifth)\s+(?:category|type|class|level|tier)',
            r'\b(?:type a|type b|type c|type d|type e)',
            r'\b(?:class 1|class 2|class 3|class 4|class 5)',
            r'\b(?:level 1|level 2|level 3|level 4|level 5)',
            r'\b(?:tier 1|tier 2|tier 3|tier 4|tier 5)',
            r'\b(?:category i|category ii|category iii|category iv|category v)'
        ]
        
        taxonomy_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in taxonomy_patterns)
        if taxonomy_count > 1:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_advanced_writing_quality(self, text: str) -> float:
        """Análisis avanzado de calidad de escritura - NUEVO"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(words) < 20:
            return 0.0
        
        # 1. Análisis de variedad sintáctica
        sentence_structures = []
        for sentence in sentences:
            if re.search(r'^[A-Z][^.!?]*\b(?:is|are|was|were)\b', sentence):
                sentence_structures.append('declarative')
            elif re.search(r'^[A-Z][^.!?]*\b(?:do|does|did|can|could|will|would)\b', sentence):
                sentence_structures.append('interrogative')
            elif re.search(r'^[A-Z][^.!?]*\b(?:let|may|should|must)\b', sentence):
                sentence_structures.append('imperative')
            else:
                sentence_structures.append('other')
        
        if sentence_structures:
            structure_diversity = len(set(sentence_structures)) / len(sentence_structures)
            if structure_diversity < 0.4:
                score += 0.25
        
        # 2. Análisis de uso de sinónimos
        word_freq = {}
        for word in words:
            word_clean = word.lower().strip('.,!?;:()[]{}"\'')
            if word_clean and len(word_clean) > 4:
                word_freq[word_clean] = word_freq.get(word_clean, 0) + 1
        
        repeated_words = [w for w, f in word_freq.items() if f > 2]
        if len(repeated_words) > len(word_freq) * 0.3:
            score += 0.2
        
        # 3. Análisis de fluidez y naturalidad
        artificial_phrases = [
            r'\bit is important to note that\b', r'\bit should be noted that\b',
            r'\bit is worth mentioning that\b', r'\bas a result of the fact that\b',
            r'\bin order to be able to\b', r'\bfor the purpose of\b',
            r'\bwith regard to the\b', r'\bin the context of the\b'
        ]
        artificial_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in artificial_phrases)
        if artificial_count > 2:
            score += 0.3
        
        # 4. Análisis de equilibrio en longitud de oraciones
        if len(sentences) > 1:
            sentence_lengths = [len(s.split()) for s in sentences]
            length_variance = np.var(sentence_lengths)
            avg_length = np.mean(sentence_lengths)
            if avg_length > 0:
                cv = np.sqrt(length_variance) / avg_length
                if cv < 0.4:
                    score += 0.15
        
        # 5. Análisis de uso de conectores
        connectors = ['however', 'therefore', 'furthermore', 'moreover', 'consequently', 'additionally', 'nevertheless', 'thus', 'hence', 'accordingly']
        connector_count = sum(1 for c in connectors if c in text.lower())
        connector_ratio = connector_count / len(sentences) if len(sentences) > 0 else 0
        if connector_ratio > 0.3:
            score += 0.2
        
        # 6. Análisis de coherencia en puntuación
        punctuation_patterns = [r'[,;:]', r'[.!?]', r'["\']']
        punct_counts = [len(re.findall(pattern, text)) for pattern in punctuation_patterns]
        if sum(punct_counts) > 0:
            punct_consistency = 1 - (np.std(punct_counts) / np.mean(punct_counts) if np.mean(punct_counts) > 0 else 0)
            if punct_consistency > 0.7:
                score += 0.1
        
        return min(score, 1.0)
    
    def _generate_alerts(self, ai_percentage: float, confidence: float, 
                        detected_models: List[Dict], primary_model: Optional[Dict]) -> List[Dict[str, Any]]:
        """Genera alertas para detecciones de alta confianza - NUEVO"""
        alerts = []
        
        # Alerta 1: Alta confianza de IA
        if ai_percentage > self.alert_threshold * 100:
            alerts.append({
                "type": "high_confidence_ai",
                "severity": "high",
                "message": f"Alta probabilidad de contenido generado por IA ({ai_percentage:.1f}%)",
                "confidence": confidence
            })
        
        # Alerta 2: Modelo específico detectado con alta confianza
        if primary_model and primary_model.get("confidence", 0) > 0.8:
            alerts.append({
                "type": "specific_model_detected",
                "severity": "high",
                "message": f"Modelo {primary_model.get('model_name', 'desconocido')} detectado con alta confianza ({primary_model.get('confidence', 0)*100:.1f}%)",
                "model": primary_model.get("model_name"),
                "confidence": primary_model.get("confidence", 0)
            })
        
        # Alerta 3: Múltiples modelos detectados (posible parafraseo)
        if len(detected_models) > 1:
            alerts.append({
                "type": "multiple_models_detected",
                "severity": "medium",
                "message": f"Múltiples modelos detectados ({len(detected_models)}), posible contenido parafraseado",
                "models": [m.get("model_name") for m in detected_models]
            })
        
        # Alerta 4: Confianza muy alta (>90%)
        if confidence > 0.9:
            alerts.append({
                "type": "very_high_confidence",
                "severity": "critical",
                "message": f"Confianza extremadamente alta ({confidence*100:.1f}%) - Detección muy confiable",
                "confidence": confidence
            })
        
        # Alerta 5: Confianza baja pero porcentaje alto (inconsistencia)
        if ai_percentage > 60 and confidence < 0.5:
            alerts.append({
                "type": "inconsistent_detection",
                "severity": "medium",
                "message": "Inconsistencia detectada: porcentaje alto pero confianza baja - revisar manualmente",
                "ai_percentage": ai_percentage,
                "confidence": confidence
            })
        
        return alerts
    
    def _forensic_analysis(self, text: str, detected_models: List[Dict], text_stats: Optional[Dict] = None) -> Dict[str, Any]:
        """Análisis forense mejorado para estimar el prompt usado"""
        forensic = {
            "estimated_prompt": None,
            "prompt_confidence": 0.0,
            "prompt_patterns": [],
            "generation_parameters": {},
            "forensic_evidence": [],
            "prompt_style": None,
            "estimated_instructions": []
        }
        
        text_lower = text.lower()
        word_count = len(text.split())
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Analizar estructura para inferir tipo de prompt
        prompt_patterns = []
        prompt_confidence_scores = []
        estimated_instructions = []
        
        # Prompt de explicación
        if re.search(r'\b(?:explain|describe|what is|how does|tell me about|can you explain|could you explain)\b', text_lower):
            prompt_patterns.append("explanatory")
            prompt_confidence_scores.append(0.65)
            # Extraer tema del texto
            first_sentence = sentences[0] if sentences else ""
            topic = first_sentence[:100] if len(first_sentence) > 20 else "the topic"
            estimated_instructions.append(f"Explain or describe {topic}")
            forensic["estimated_prompt"] = f"Explain or describe: {topic}..."
        
        # Prompt de lista/enumeración
        if re.search(r'\d+\.\s|\-\s|\*\s|•\s', text):
            prompt_patterns.append("list_generation")
            prompt_confidence_scores.append(0.75)
            list_keywords = re.findall(r'\b(?:list|items|steps|ways|methods|examples|reasons)\b', text_lower)
            if list_keywords:
                estimated_instructions.append(f"Generate a list of {list_keywords[0]}")
            else:
                estimated_instructions.append("Generate a list of items")
            if not forensic["estimated_prompt"]:
                forensic["estimated_prompt"] = "Generate a list of items about..."
        
        # Prompt de análisis
        if re.search(r'\b(?:analyze|analysis|evaluate|assess|review|examine|investigate)\b', text_lower):
            prompt_patterns.append("analytical")
            prompt_confidence_scores.append(0.70)
            estimated_instructions.append("Analyze or evaluate the following content")
            if not forensic["estimated_prompt"]:
                forensic["estimated_prompt"] = "Analyze or evaluate the following..."
        
        # Prompt de resumen
        if word_count < 300 and re.search(r'\b(?:summary|summarize|overview|brief|condense)\b', text_lower):
            prompt_patterns.append("summarization")
            prompt_confidence_scores.append(0.80)
            estimated_instructions.append("Summarize the following content")
            if not forensic["estimated_prompt"]:
                forensic["estimated_prompt"] = "Summarize the following content..."
        
        # Prompt de comparación
        if re.search(r'\b(?:compare|comparison|difference|similarities|contrast)\b', text_lower):
            prompt_patterns.append("comparison")
            prompt_confidence_scores.append(0.75)
            estimated_instructions.append("Compare the following items")
            if not forensic["estimated_prompt"]:
                forensic["estimated_prompt"] = "Compare the following..."
        
        # Prompt de escritura creativa
        if re.search(r'\b(?:write|create|generate|compose|draft)\b', text_lower) and word_count > 200:
            prompt_patterns.append("creative_writing")
            prompt_confidence_scores.append(0.65)
            estimated_instructions.append("Write or create content about")
            if not forensic["estimated_prompt"]:
                forensic["estimated_prompt"] = "Write about..."
        
        # Prompt de pregunta-respuesta
        if text_lower.startswith(('what', 'how', 'why', 'when', 'where', 'who', 'can', 'could', 'would', 'should')):
            prompt_patterns.append("qa")
            prompt_confidence_scores.append(0.70)
            first_q = sentences[0] if sentences else ""
            estimated_instructions.append(f"Answer: {first_q[:80]}")
            if not forensic["estimated_prompt"]:
                forensic["estimated_prompt"] = first_q[:100] + "..."
        
        # Detectar estilo de prompt
        if re.search(r'\b(?:please|kindly|could you|would you|i need|i want)\b', text_lower):
            forensic["prompt_style"] = "polite_request"
        elif re.search(r'\b(?:write|create|generate|make|do)\b', text_lower):
            forensic["prompt_style"] = "direct_command"
        elif re.search(r'\b(?:explain|describe|tell|show)\b', text_lower):
            forensic["prompt_style"] = "informational_request"
        else:
            forensic["prompt_style"] = "general"
        
        forensic["prompt_patterns"] = prompt_patterns
        forensic["estimated_instructions"] = estimated_instructions
        
        # Calcular confianza promedio
        if prompt_confidence_scores:
            forensic["prompt_confidence"] = np.mean(prompt_confidence_scores)
        elif detected_models:
            forensic["prompt_confidence"] = 0.5  # Confianza media si hay modelo detectado
        else:
            forensic["prompt_confidence"] = 0.3  # Baja confianza
        
        # Parámetros de generación estimados mejorados
        avg_sentence_length = text_stats.get("avg_sentence_length", word_count / len(sentences) if sentences else 15) if text_stats else 15
        burstiness = text_stats.get("burstiness", 0.5) if text_stats else 0.5
        
        # Estimar temperature basado en variabilidad
        if burstiness < 0.3:
            estimated_temp = 0.5  # Baja variabilidad = temperatura baja
        elif burstiness < 0.6:
            estimated_temp = 0.7  # Variabilidad media
        else:
            estimated_temp = 0.9  # Alta variabilidad
        
        # Estimar max_tokens basado en longitud
        estimated_max_tokens = int(min(word_count * 1.3, 4000))
        
        forensic["generation_parameters"] = {
            "estimated_max_tokens": estimated_max_tokens,
            "temperature": estimated_temp,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "top_p": 0.9,  # Estimación común
            "estimated_stop_sequences": None
        }
        
        # Evidencia forense mejorada
        forensic["forensic_evidence"] = [
            {
                "type": "text_length",
                "value": word_count,
                "indication": "long_text" if word_count > 500 else "medium_text" if word_count > 200 else "short_text"
            },
            {
                "type": "structure_type",
                "value": prompt_patterns[0] if prompt_patterns else "general",
                "indication": "structured" if prompt_patterns else "unstructured"
            },
            {
                "type": "sentence_structure",
                "value": avg_sentence_length,
                "indication": "uniform" if burstiness < 0.4 else "varied"
            },
            {
                "type": "detected_models_count",
                "value": len(detected_models),
                "indication": "high_confidence" if len(detected_models) > 0 else "low_confidence"
            }
        ]
        
        return forensic
    
    def _detect_image_ai(self, image_data: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Detecta si una imagen fue generada por IA"""
        # Placeholder para detección de imágenes
        # En producción, usar modelos como CLIP, GLIDE detector, etc.
        
        return {
            "is_ai_generated": False,
            "ai_percentage": 0.0,
            "detected_models": [],
            "primary_model": None,
            "forensic_analysis": None,
            "confidence_score": 0.0,
            "detection_methods": ["image_analysis_placeholder"]
        }
    
    def _detect_audio_ai(self, audio_data: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Detecta si un audio fue generado por IA"""
        # Placeholder para detección de audio
        # En producción, usar modelos de detección de deepfake audio
        
        return {
            "is_ai_generated": False,
            "ai_percentage": 0.0,
            "detected_models": [],
            "primary_model": None,
            "forensic_analysis": None,
            "confidence_score": 0.0,
            "detection_methods": ["audio_analysis_placeholder"]
        }
    
    def _detect_video_ai(self, video_data: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Detecta si un video fue generado por IA"""
        # Placeholder para detección de video
        # En producción, usar modelos de detección de deepfake video
        
        return {
            "is_ai_generated": False,
            "ai_percentage": 0.0,
            "detected_models": [],
            "primary_model": None,
            "forensic_analysis": None,
            "confidence_score": 0.0,
            "detection_methods": ["video_analysis_placeholder"]
        }
    
    def _detect_multimodal(self, content: Any, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Detecta contenido multimodal"""
        # Combinar resultados de diferentes modalidades
        return {
            "is_ai_generated": False,
            "ai_percentage": 0.0,
            "detected_models": [],
            "primary_model": None,
            "forensic_analysis": None,
            "confidence_score": 0.0,
            "detection_methods": ["multimodal_analysis_placeholder"]
        }
    
    def get_health(self) -> Dict[str, Any]:
        """Obtiene el estado de salud del detector"""
        return {
            "status": "healthy",
            "version": "1.0.0",
            "models_loaded": self.models_loaded,
            "uptime": time.time() - self.start_time
        }


Implementa la lógica principal de detección de contenido generado por IA
"""

import logging
import time
import hashlib
import re
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class MultimodalAIDetector:
    """Detector multimodal de contenido generado por IA"""
    
    # Patrones comunes de modelos de IA - MEJORADO
    AI_MODEL_PATTERNS = {
        "gpt-3.5": {
            "patterns": [
                r"\b(?:as an ai|i'm an ai|i am an ai|as a language model|as an artificial intelligence)",
                r"\b(?:i cannot|i don't have|i'm not able|i don't have access|i'm unable)",
                r"\b(?:i apologize|i'm sorry|unfortunately|i cannot provide)",
                r"\b(?:please note that|it's important to|keep in mind)",
                r"\b(?:i hope this helps|let me know if|feel free to ask)"
            ],
            "provider": "OpenAI",
            "confidence_base": 0.75,
            "version_patterns": [r"gpt-3\.5", r"gpt-3\.5-turbo", r"text-davinci"]
        },
        "gpt-4": {
            "patterns": [
                r"\b(?:comprehensive|thorough|detailed analysis|in-depth)",
                r"\b(?:let me|i'll|i will provide|allow me to)",
                r"\b(?:here's|here is a|below is|following is)",
                r"\b(?:to summarize|in summary|in conclusion|to conclude)",
                r"\b(?:it's worth noting|it should be noted|important to understand)"
            ],
            "provider": "OpenAI",
            "confidence_base": 0.82,
            "version_patterns": [r"gpt-4", r"gpt-4-turbo", r"gpt-4o"]
        },
        "claude": {
            "patterns": [
                r"\b(?:i'd be happy|i can help|let me assist|i'd be glad)",
                r"\b(?:here's how|here are|i'll explain|let me break this down)",
                r"\b(?:based on|considering|taking into account|given that)",
                r"\b(?:to clarify|to elaborate|in other words|put another way)",
                r"\b(?:i should mention|it's also worth|additionally|furthermore)"
            ],
            "provider": "Anthropic",
            "confidence_base": 0.78,
            "version_patterns": [r"claude-3", r"claude-2", r"claude-instant"]
        },
        "gemini": {
            "patterns": [
                r"\b(?:here's|here are|let me share|i can share)",
                r"\b(?:i can|i'll|i will|i'm able to)",
                r"\b(?:based on|according to|in terms of|with regard to)",
                r"\b(?:to help you|to assist you|for your reference)",
                r"\b(?:it appears|it seems|this suggests|this indicates)"
            ],
            "provider": "Google",
            "confidence_base": 0.72,
            "version_patterns": [r"gemini-pro", r"gemini-ultra", r"gemini-1"]
        },
        "llama": {
            "patterns": [
                r"\b(?:let me|i'll|i will|allow me)",
                r"\b(?:here's|here is|below|following)",
                r"\b(?:i can|i'm able|i have the ability|i'm capable)",
                r"\b(?:to answer|to respond|to help|to assist)",
                r"\b(?:it's important|it should be|one should|we should)"
            ],
            "provider": "Meta",
            "confidence_base": 0.70,
            "version_patterns": [r"llama-2", r"llama-3", r"llama-70b"]
        },
        "mistral": {
            "patterns": [
                r"\b(?:let me|i'll|i can|i will)",
                r"\b(?:here's|here are|below|following)",
                r"\b(?:to help|to assist|to provide|to offer)",
                r"\b(?:it's worth|it should be|one must|we must)"
            ],
            "provider": "Mistral AI",
            "confidence_base": 0.68,
            "version_patterns": [r"mistral-7b", r"mistral-medium", r"mixtral"]
        },
        "cohere": {
            "patterns": [
                r"\b(?:let me|i'll|i can|i will)",
                r"\b(?:here's|here are|below|following)",
                r"\b(?:to help|to assist|to provide|to offer)",
                r"\b(?:based on|according to|in terms of)"
            ],
            "provider": "Cohere",
            "confidence_base": 0.65,
            "version_patterns": [r"command", r"command-light", r"command-nightly"]
        },
        "palm": {
            "patterns": [
                r"\b(?:here's|here are|let me|i can)",
                r"\b(?:based on|according to|in terms of)",
                r"\b(?:to help|to assist|to provide)"
            ],
            "provider": "Google",
            "confidence_base": 0.63,
            "version_patterns": [r"palm-2", r"palm", r"text-bison"]
        },
        "jurassic": {
            "patterns": [
                r"\b(?:let me|i'll|i can|i will)",
                r"\b(?:here's|here are|below|following)",
                r"\b(?:to help|to assist|to provide)"
            ],
            "provider": "AI21 Labs",
            "confidence_base": 0.60,
            "version_patterns": [r"j2", r"jurassic-2", r"jamba"]
        },
        "groq": {
            "patterns": [
                r"\b(?:let me|i'll|i can|i will)",
                r"\b(?:here's|here are|below|following)",
                r"\b(?:to help|to assist|to provide)"
            ],
            "provider": "Groq",
            "confidence_base": 0.62,
            "version_patterns": [r"mixtral", r"llama-3", r"gemma"]
        },
        "openrouter": {
            "patterns": [
                r"\b(?:let me|i'll|i can|i will)",
                r"\b(?:here's|here are|below|following)",
                r"\b(?:to help|to assist|to provide)"
            ],
            "provider": "OpenRouter",
            "confidence_base": 0.60,
            "version_patterns": []
        }
    }
    
    # Características de texto generado por IA
    AI_TEXT_FEATURES = {
        "perplexity_threshold": 50.0,  # Texto de IA suele tener menor perplexity
        "burstiness_threshold": 0.5,   # Menos variación en longitud de oraciones
        "repetition_threshold": 0.3,   # Menos repetición de palabras
        "coherence_score": 0.8,        # Alta coherencia
        "formality_score": 0.7         # Texto más formal
    }
    
    def __init__(self):
        self.start_time = time.time()
        self.models_loaded = 0
        self.detection_cache = {}  # Cache para resultados
        self.cache_max_size = 1000  # Tamaño máximo del cache
        self.detection_history = []  # Historial de detecciones
        self.max_history_size = 100  # Tamaño máximo del historial
        self.adaptive_weights = {}  # Pesos adaptativos basados en historial
        self.model_performance = {}  # Rendimiento de cada método
        self.known_ai_texts = []  # Textos conocidos de IA para comparación
        self.known_human_texts = []  # Textos conocidos humanos para comparación
        self.max_known_texts = 50  # Máximo de textos conocidos
        self.alert_threshold = 0.8  # Umbral para alertas de alta confianza
        self.model_signatures = {}  # Firmas características de cada modelo
        logger.info("MultimodalAIDetector inicializado")
    
    def detect(self, content: str, content_type: str = "text", metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Detecta si el contenido fue generado por IA
        
        Args:
            content: Contenido a analizar
            content_type: Tipo de contenido (text, image, audio, video)
            metadata: Metadatos adicionales
            
        Returns:
            Diccionario con resultados de la detección
        """
        start_time = time.time()
        
        # Verificar cache
        cache_key = self._generate_cache_key(content, content_type)
        if cache_key in self.detection_cache:
            cached_result = self.detection_cache[cache_key].copy()
            cached_result["from_cache"] = True
            cached_result["processing_time"] = time.time() - start_time
            return cached_result
        
        try:
            if content_type == "text":
                result = self._detect_text_ai(content, metadata)
            elif content_type == "image":
                result = self._detect_image_ai(content, metadata)
            elif content_type == "audio":
                result = self._detect_audio_ai(content, metadata)
            elif content_type == "video":
                result = self._detect_video_ai(content, metadata)
            else:
                result = self._detect_multimodal(content, metadata)
            
            result["processing_time"] = time.time() - start_time
            result["timestamp"] = time.time()
            result["from_cache"] = False
            
            # Guardar en cache
            self._add_to_cache(cache_key, result)
            
            # Guardar en historial
            self._add_to_history(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error en detección: {e}", exc_info=True)
            raise
    
    def _generate_cache_key(self, content: str, content_type: str) -> str:
        """Genera una clave de cache para el contenido"""
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        return f"{content_type}:{content_hash}"
    
    def _add_to_cache(self, cache_key: str, result: Dict[str, Any]):
        """Añade resultado al cache"""
        if len(self.detection_cache) >= self.cache_max_size:
            # Eliminar el más antiguo (FIFO)
            oldest_key = next(iter(self.detection_cache))
            del self.detection_cache[oldest_key]
        
        self.detection_cache[cache_key] = result.copy()
    
    def clear_cache(self):
        """Limpia el cache de detecciones"""
        self.detection_cache.clear()
        logger.info("Cache de detecciones limpiado")
    
    def _add_to_history(self, result: Dict[str, Any]):
        """Añade resultado al historial"""
        if len(self.detection_history) >= self.max_history_size:
            # Eliminar el más antiguo (FIFO)
            self.detection_history.pop(0)
        
        # Guardar solo información esencial
        history_entry = {
            "timestamp": result.get("timestamp", time.time()),
            "is_ai_generated": result.get("is_ai_generated", False),
            "ai_percentage": result.get("ai_percentage", 0.0),
            "primary_model": result.get("primary_model", {}).get("model_name") if result.get("primary_model") else None,
            "confidence_score": result.get("confidence_score", 0.0)
        }
        self.detection_history.append(history_entry)
    
    def get_detection_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Obtiene el historial de detecciones"""
        return self.detection_history[-limit:] if limit > 0 else self.detection_history
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas del detector"""
        if not self.detection_history:
            return {
                "total_detections": 0,
                "ai_detections": 0,
                "human_detections": 0,
                "avg_ai_percentage": 0.0,
                "avg_confidence": 0.0,
                "most_common_model": None
            }
        
        total = len(self.detection_history)
        ai_detections = sum(1 for entry in self.detection_history if entry.get("is_ai_generated", False))
        human_detections = total - ai_detections
        
        avg_ai_percentage = np.mean([entry.get("ai_percentage", 0.0) for entry in self.detection_history])
        avg_confidence = np.mean([entry.get("confidence_score", 0.0) for entry in self.detection_history])
        
        # Modelo más común
        models = [entry.get("primary_model") for entry in self.detection_history if entry.get("primary_model")]
        most_common_model = max(set(models), key=models.count) if models else None
        
        return {
            "total_detections": total,
            "ai_detections": ai_detections,
            "human_detections": human_detections,
            "ai_detection_rate": (ai_detections / total * 100) if total > 0 else 0.0,
            "avg_ai_percentage": avg_ai_percentage,
            "avg_confidence": avg_confidence,
            "most_common_model": most_common_model,
            "cache_size": len(self.detection_cache),
            "cache_usage_percent": (len(self.detection_cache) / self.cache_max_size * 100) if self.cache_max_size > 0 else 0
        }
    
    def _detect_text_ai(self, text: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Detecta si un texto fue generado por IA - MEJORADO"""
        detection_methods = []
        scores = []
        detected_models = []
        method_weights = {}  # Pesos para cada método
        
        # Método 1: Análisis de patrones de modelos (peso alto)
        model_detections = self._detect_model_patterns(text)
        if model_detections:
            detected_models.extend(model_detections)
            detection_methods.append("pattern_matching")
            pattern_score = max([m["confidence"] for m in model_detections])
            scores.append(pattern_score)
            method_weights["pattern_matching"] = 0.35  # Peso alto
        
        # Método 2: Análisis estadístico de texto (peso medio-alto)
        text_stats = self._analyze_text_statistics(text)
        if text_stats["ai_likelihood"] > 0.3:
            detection_methods.append("statistical_analysis")
            scores.append(text_stats["ai_likelihood"])
            method_weights["statistical_analysis"] = 0.25
        
        # Método 3: Análisis de estructura y coherencia (peso medio)
        structure_score = self._analyze_text_structure(text)
        if structure_score > 0.4:
            detection_methods.append("structure_analysis")
            scores.append(structure_score)
            method_weights["structure_analysis"] = 0.15
        
        # Método 4: Análisis de estilo (peso medio)
        style_score = self._analyze_text_style(text)
        if style_score > 0.3:
            detection_methods.append("style_analysis")
            scores.append(style_score)
            method_weights["style_analysis"] = 0.15
        
        # Método 5: Análisis de entropía y n-gramas (NUEVO)
        entropy_score = self._analyze_entropy_and_ngrams(text)
        if entropy_score > 0.3:
            detection_methods.append("entropy_analysis")
            scores.append(entropy_score)
            method_weights["entropy_analysis"] = 0.10
        
        # Método 6: Análisis de coherencia semántica (NUEVO)
        semantic_score = self._analyze_semantic_coherence(text)
        if semantic_score > 0.3:
            detection_methods.append("semantic_coherence")
            scores.append(semantic_score)
            method_weights["semantic_coherence"] = 0.08
        
        # Método 7: Análisis de complejidad sintáctica (NUEVO)
        syntactic_score = self._analyze_syntactic_complexity(text)
        if syntactic_score > 0.3:
            detection_methods.append("syntactic_complexity")
            scores.append(syntactic_score)
            method_weights["syntactic_complexity"] = 0.07
        
        # Método 8: Análisis de citas y referencias (NUEVO)
        citation_score = self._analyze_citations_and_references(text)
        if citation_score > 0.3:
            detection_methods.append("citation_analysis")
            scores.append(citation_score)
            method_weights["citation_analysis"] = 0.05
        
        # Método 9: Análisis temporal (NUEVO)
        temporal_score = self._analyze_temporal_consistency(text)
        if temporal_score > 0.3:
            detection_methods.append("temporal_analysis")
            scores.append(temporal_score)
            method_weights["temporal_analysis"] = 0.04
        
        # Método 10: Detección de watermarks (NUEVO)
        watermark_score = self._detect_watermarks(text)
        if watermark_score > 0.2:
            detection_methods.append("watermark_detection")
            scores.append(watermark_score)
            method_weights["watermark_detection"] = 0.03
        
        # Método 11: Análisis de ediciones/parches (NUEVO)
        edit_score = self._detect_edits_and_patches(text)
        if edit_score > 0.2:
            detection_methods.append("edit_detection")
            scores.append(edit_score)
            method_weights["edit_detection"] = 0.02
        
        # Método 12: Análisis de sentimientos (NUEVO)
        sentiment_score = self._analyze_sentiment_patterns(text)
        if sentiment_score > 0.3:
            detection_methods.append("sentiment_analysis")
            scores.append(sentiment_score)
            method_weights["sentiment_analysis"] = 0.02
        
        # Método 13: Análisis de contexto y coherencia temática (NUEVO)
        context_score = self._analyze_contextual_coherence(text)
        if context_score > 0.3:
            detection_methods.append("contextual_analysis")
            scores.append(context_score)
            method_weights["contextual_analysis"] = 0.03
        
        # Método 14: Detección de traducción automática (NUEVO)
        translation_score = self._detect_machine_translation(text)
        if translation_score > 0.2:
            detection_methods.append("translation_detection")
            scores.append(translation_score)
            method_weights["translation_detection"] = 0.02
        
        # Método 15: Análisis de patrones de generación (NUEVO)
        generation_score = self._analyze_generation_patterns(text)
        if generation_score > 0.3:
            detection_methods.append("generation_patterns")
            scores.append(generation_score)
            method_weights["generation_patterns"] = 0.02
        
        # Método 16: Análisis de calidad de escritura (NUEVO)
        quality_score = self._analyze_writing_quality(text)
        if quality_score > 0.3:
            detection_methods.append("writing_quality")
            scores.append(quality_score)
            method_weights["writing_quality"] = 0.02
        
        # Método 17: Detección de parafraseo (NUEVO)
        paraphrase_score = self._detect_paraphrasing(text)
        if paraphrase_score > 0.2:
            detection_methods.append("paraphrase_detection")
            scores.append(paraphrase_score)
            method_weights["paraphrase_detection"] = 0.02
        
        # Método 18: Análisis de riesgo y confiabilidad (NUEVO)
        risk_score = self._analyze_risk_and_reliability(text, detected_models, scores)
        if risk_score > 0.2:
            detection_methods.append("risk_analysis")
            scores.append(risk_score)
            method_weights["risk_analysis"] = 0.01
        
        # Método 19: Análisis de metadatos y contexto (NUEVO)
        if metadata:
            metadata_score = self._analyze_metadata_and_context(text, metadata)
            if metadata_score > 0.2:
                detection_methods.append("metadata_analysis")
                scores.append(metadata_score)
                method_weights["metadata_analysis"] = 0.01
        
        # Método 20: Análisis de idioma y localización (NUEVO)
        language_score = self._analyze_language_patterns(text)
        if language_score > 0.2:
            detection_methods.append("language_analysis")
            scores.append(language_score)
            method_weights["language_analysis"] = 0.01
        
        # Método 21: Análisis de similitud semántica (NUEVO)
        semantic_similarity_score = self._analyze_semantic_similarity(text)
        if semantic_similarity_score > 0.2:
            detection_methods.append("semantic_similarity")
            scores.append(semantic_similarity_score)
            method_weights["semantic_similarity"] = 0.01
        
        # Método 22: Análisis de frecuencia de palabras clave (NUEVO)
        keyword_frequency_score = self._analyze_keyword_frequency(text)
        if keyword_frequency_score > 0.2:
            detection_methods.append("keyword_frequency")
            scores.append(keyword_frequency_score)
            method_weights["keyword_frequency"] = 0.01
        
        # Método 23: Detección de patrones de respuesta típicos (NUEVO)
        response_pattern_score = self._detect_response_patterns(text)
        if response_pattern_score > 0.2:
            detection_methods.append("response_patterns")
            scores.append(response_pattern_score)
            method_weights["response_patterns"] = 0.01
        
        # Método 24: Análisis de coherencia narrativa (NUEVO)
        narrative_coherence_score = self._analyze_narrative_coherence(text)
        if narrative_coherence_score > 0.2:
            detection_methods.append("narrative_coherence")
            scores.append(narrative_coherence_score)
            method_weights["narrative_coherence"] = 0.01
        
        # Sistema de scoring adaptativo (NUEVO)
        method_weights = self._apply_adaptive_weights(method_weights, detection_methods, text)
        
        # Ajustar pesos si hay nuevos métodos
        total_current_weight = sum(method_weights.values())
        if total_current_weight > 1.0:
            # Normalizar pesos si exceden 1.0
            method_weights = {k: v / total_current_weight for k, v in method_weights.items()}
        
        # Análisis de contexto histórico (NUEVO)
        historical_context_score = self._analyze_historical_context(text, detected_models)
        if historical_context_score > 0.2:
            detection_methods.append("historical_context")
            scores.append(historical_context_score)
            method_weights["historical_context"] = 0.01
        
        # Análisis de n-gramas avanzado (NUEVO)
        advanced_ngram_score = self._analyze_advanced_ngrams(text)
        if advanced_ngram_score > 0.2:
            detection_methods.append("advanced_ngrams")
            scores.append(advanced_ngram_score)
            method_weights["advanced_ngrams"] = 0.01
        
        # Análisis comparativo con textos conocidos (NUEVO)
        comparative_score = self._analyze_comparative_similarity(text)
        if comparative_score > 0.2:
            detection_methods.append("comparative_analysis")
            scores.append(comparative_score)
            method_weights["comparative_analysis"] = 0.01
        
        # Análisis de aprendizaje automático básico (NUEVO)
        ml_score = self._analyze_with_ml_patterns(text, detected_models, scores)
        if ml_score > 0.2:
            detection_methods.append("ml_patterns")
            scores.append(ml_score)
            method_weights["ml_patterns"] = 0.01
        
        # Análisis de firmas de modelos específicos (NUEVO)
        signature_score = self._analyze_model_signatures(text, detected_models)
        if signature_score > 0.2:
            detection_methods.append("model_signatures")
            scores.append(signature_score)
            method_weights["model_signatures"] = 0.01
        
        # Análisis de embeddings semánticos básico (NUEVO)
        embedding_score = self._analyze_semantic_embeddings(text)
        if embedding_score > 0.2:
            detection_methods.append("semantic_embeddings")
            scores.append(embedding_score)
            method_weights["semantic_embeddings"] = 0.01
        
        # Análisis de patrones temporales (NUEVO)
        temporal_pattern_score = self._analyze_temporal_patterns(text, metadata)
        if temporal_pattern_score > 0.2:
            detection_methods.append("temporal_patterns")
            scores.append(temporal_pattern_score)
            method_weights["temporal_patterns"] = 0.01
        
        # Detección de modelos híbridos (NUEVO)
        hybrid_score = self._detect_hybrid_models(text, detected_models)
        if hybrid_score > 0.2:
            detection_methods.append("hybrid_detection")
            scores.append(hybrid_score)
            method_weights["hybrid_detection"] = 0.01
        
        # Análisis de frecuencia avanzado (NUEVO)
        frequency_score = self._analyze_advanced_frequency(text)
        if frequency_score > 0.2:
            detection_methods.append("advanced_frequency")
            scores.append(frequency_score)
            method_weights["advanced_frequency"] = 0.01
        
        # Análisis de coherencia contextual avanzado (NUEVO)
        advanced_context_score = self._analyze_advanced_contextual_coherence(text)
        if advanced_context_score > 0.2:
            detection_methods.append("advanced_contextual")
            scores.append(advanced_context_score)
            method_weights["advanced_contextual"] = 0.01
        
        # Detección de deepfake de texto (NUEVO)
        deepfake_score = self._detect_text_deepfake(text, detected_models)
        if deepfake_score > 0.2:
            detection_methods.append("text_deepfake")
            scores.append(deepfake_score)
            method_weights["text_deepfake"] = 0.01
        
        # Análisis de calidad de escritura avanzado (NUEVO)
        advanced_quality_score = self._analyze_advanced_writing_quality(text)
        if advanced_quality_score > 0.2:
            detection_methods.append("advanced_writing_quality")
            scores.append(advanced_quality_score)
            method_weights["advanced_writing_quality"] = 0.01
        
        # Detección de patrones de deepfake (NUEVO)
        deepfake_pattern_score = self._detect_deepfake_patterns(text)
        if deepfake_pattern_score > 0.2:
            detection_methods.append("deepfake_patterns")
            scores.append(deepfake_pattern_score)
            method_weights["deepfake_patterns"] = 0.01
        
        # Sistema de scoring mejorado (NUEVO)
        enhanced_score = self._enhanced_scoring_system(scores, detection_methods, detected_models)
        if enhanced_score > 0.2:
            detection_methods.append("enhanced_scoring")
            scores.append(enhanced_score)
            method_weights["enhanced_scoring"] = 0.02
        
        # Análisis avanzado de patrones de repetición (NUEVO)
        repetition_score = self._analyze_advanced_repetition_patterns(text)
        if repetition_score > 0.2:
            detection_methods.append("advanced_repetition")
            scores.append(repetition_score)
            method_weights["advanced_repetition"] = 0.01
        
        # Detección avanzada de parafraseo con IA (NUEVO)
        ai_paraphrase_score = self._detect_ai_paraphrasing_advanced(text)
        if ai_paraphrase_score > 0.2:
            detection_methods.append("ai_paraphrasing_advanced")
            scores.append(ai_paraphrase_score)
            method_weights["ai_paraphrasing_advanced"] = 0.01
        
        # Detección de mezclas de estilos (NUEVO)
        style_mixture_score = self._analyze_style_mixture(text)
        if style_mixture_score > 0.2:
            detection_methods.append("style_mixture")
            scores.append(style_mixture_score)
            method_weights["style_mixture"] = 0.01
        
        # Análisis sofisticado de patrones de generación (NUEVO)
        generation_sophistication_score = self._analyze_generation_sophistication(text)
        if generation_sophistication_score > 0.2:
            detection_methods.append("generation_sophistication")
            scores.append(generation_sophistication_score)
            method_weights["generation_sophistication"] = 0.01
        
        # Análisis avanzado de diversidad léxica (NUEVO)
        lexical_diversity_score = self._analyze_lexical_diversity_advanced(text)
        if lexical_diversity_score > 0.2:
            detection_methods.append("lexical_diversity_advanced")
            scores.append(lexical_diversity_score)
            method_weights["lexical_diversity_advanced"] = 0.01
        
        # Detección de patrones de hedging (NUEVO)
        hedging_score = self._detect_ai_hedging_patterns(text)
        if hedging_score > 0.2:
            detection_methods.append("ai_hedging_patterns")
            scores.append(hedging_score)
            method_weights["ai_hedging_patterns"] = 0.01
        
        # Análisis de distribución de complejidad de oraciones (NUEVO)
        complexity_distribution_score = self._analyze_sentence_complexity_distribution(text)
        if complexity_distribution_score > 0.2:
            detection_methods.append("complexity_distribution")
            scores.append(complexity_distribution_score)
            method_weights["complexity_distribution"] = 0.01
        
        # Detección de patrones de verbosidad (NUEVO)
        verbosity_score = self._detect_ai_verbosity_patterns(text)
        if verbosity_score > 0.2:
            detection_methods.append("ai_verbosity_patterns")
            scores.append(verbosity_score)
            method_weights["ai_verbosity_patterns"] = 0.01
        
        # Análisis de patrones de uso de pronombres (NUEVO)
        pronoun_score = self._analyze_pronoun_usage_patterns(text)
        if pronoun_score > 0.2:
            detection_methods.append("pronoun_usage_patterns")
            scores.append(pronoun_score)
            method_weights["pronoun_usage_patterns"] = 0.01
        
        # Detección de patrones de preguntas (NUEVO)
        question_score = self._detect_ai_question_patterns(text)
        if question_score > 0.2:
            detection_methods.append("ai_question_patterns")
            scores.append(question_score)
            method_weights["ai_question_patterns"] = 0.01
        
        # Análisis de patrones de cierre (NUEVO)
        closure_score = self._analyze_ai_closure_patterns(text)
        if closure_score > 0.2:
            detection_methods.append("ai_closure_patterns")
            scores.append(closure_score)
            method_weights["ai_closure_patterns"] = 0.01
        
        # Detección de patrones de enumeración (NUEVO)
        enumeration_score = self._detect_ai_enumeration_patterns(text)
        if enumeration_score > 0.2:
            detection_methods.append("ai_enumeration_patterns")
            scores.append(enumeration_score)
            method_weights["ai_enumeration_patterns"] = 0.01
        
        # Análisis de patrones de metáforas (NUEVO)
        metaphor_score = self._analyze_ai_metaphor_patterns(text)
        if metaphor_score > 0.2:
            detection_methods.append("ai_metaphor_patterns")
            scores.append(metaphor_score)
            method_weights["ai_metaphor_patterns"] = 0.01
        
        # Detección de patrones de énfasis (NUEVO)
        emphasis_score = self._detect_ai_emphasis_patterns(text)
        if emphasis_score > 0.2:
            detection_methods.append("ai_emphasis_patterns")
            scores.append(emphasis_score)
            method_weights["ai_emphasis_patterns"] = 0.01
        
        # Análisis de patrones de modificadores (NUEVO)
        modifier_score = self._analyze_ai_modifier_patterns(text)
        if modifier_score > 0.2:
            detection_methods.append("ai_modifier_patterns")
            scores.append(modifier_score)
            method_weights["ai_modifier_patterns"] = 0.01
        
        # Detección de patrones condicionales (NUEVO)
        conditional_score = self._detect_ai_conditional_patterns(text)
        if conditional_score > 0.2:
            detection_methods.append("ai_conditional_patterns")
            scores.append(conditional_score)
            method_weights["ai_conditional_patterns"] = 0.01
        
        # Análisis de patrones de voz pasiva (NUEVO)
        passive_voice_score = self._analyze_ai_passive_voice_patterns(text)
        if passive_voice_score > 0.2:
            detection_methods.append("ai_passive_voice_patterns")
            scores.append(passive_voice_score)
            method_weights["ai_passive_voice_patterns"] = 0.01
        
        # Detección de patrones de conectores (NUEVO)
        connector_score = self._detect_ai_connector_patterns(text)
        if connector_score > 0.2:
            detection_methods.append("ai_connector_patterns")
            scores.append(connector_score)
            method_weights["ai_connector_patterns"] = 0.01
        
        # Análisis de patrones de cuantificadores (NUEVO)
        quantifier_score = self._analyze_ai_quantifier_patterns(text)
        if quantifier_score > 0.2:
            detection_methods.append("ai_quantifier_patterns")
            scores.append(quantifier_score)
            method_weights["ai_quantifier_patterns"] = 0.01
        
        # Detección de patrones de aserciones (NUEVO)
        assertion_score = self._detect_ai_assertion_patterns(text)
        if assertion_score > 0.2:
            detection_methods.append("ai_assertion_patterns")
            scores.append(assertion_score)
            method_weights["ai_assertion_patterns"] = 0.01
        
        # Análisis de patrones de negación (NUEVO)
        negation_score = self._analyze_ai_negation_patterns(text)
        if negation_score > 0.2:
            detection_methods.append("ai_negation_patterns")
            scores.append(negation_score)
            method_weights["ai_negation_patterns"] = 0.01
        
        # Detección de patrones de comparación (NUEVO)
        comparison_score = self._detect_ai_comparison_patterns(text)
        if comparison_score > 0.2:
            detection_methods.append("ai_comparison_patterns")
            scores.append(comparison_score)
            method_weights["ai_comparison_patterns"] = 0.01
        
        # Análisis de patrones de marcadores temporales (NUEVO)
        temporal_marker_score = self._analyze_ai_temporal_marker_patterns(text)
        if temporal_marker_score > 0.2:
            detection_methods.append("ai_temporal_marker_patterns")
            scores.append(temporal_marker_score)
            method_weights["ai_temporal_marker_patterns"] = 0.01
        
        # Detección de patrones de causalidad (NUEVO)
        causality_score = self._detect_ai_causality_patterns(text)
        if causality_score > 0.2:
            detection_methods.append("ai_causality_patterns")
            scores.append(causality_score)
            method_weights["ai_causality_patterns"] = 0.01
        
        # Análisis de patrones de verbos modales (NUEVO)
        modal_verb_score = self._analyze_ai_modal_verb_patterns(text)
        if modal_verb_score > 0.2:
            detection_methods.append("ai_modal_verb_patterns")
            scores.append(modal_verb_score)
            method_weights["ai_modal_verb_patterns"] = 0.01
        
        # Detección de patrones de frases de hedging (NUEVO)
        hedge_phrase_score = self._detect_ai_hedge_phrase_patterns(text)
        if hedge_phrase_score > 0.2:
            detection_methods.append("ai_hedge_phrase_patterns")
            scores.append(hedge_phrase_score)
            method_weights["ai_hedge_phrase_patterns"] = 0.01
        
        # Análisis de patrones de cláusulas relativas (NUEVO)
        relative_clause_score = self._analyze_ai_relative_clause_patterns(text)
        if relative_clause_score > 0.2:
            detection_methods.append("ai_relative_clause_patterns")
            scores.append(relative_clause_score)
            method_weights["ai_relative_clause_patterns"] = 0.01
        
        # Detección de patrones de infinitivos (NUEVO)
        infinitive_score = self._detect_ai_infinitive_patterns(text)
        if infinitive_score > 0.2:
            detection_methods.append("ai_infinitive_patterns")
            scores.append(infinitive_score)
            method_weights["ai_infinitive_patterns"] = 0.01
        
        # Análisis de patrones de gerundios (NUEVO)
        gerund_score = self._analyze_ai_gerund_patterns(text)
        if gerund_score > 0.2:
            detection_methods.append("ai_gerund_patterns")
            scores.append(gerund_score)
            method_weights["ai_gerund_patterns"] = 0.01
        
        # Detección de patrones de participios (NUEVO)
        participle_score = self._detect_ai_participle_patterns(text)
        if participle_score > 0.2:
            detection_methods.append("ai_participle_patterns")
            scores.append(participle_score)
            method_weights["ai_participle_patterns"] = 0.01
        
        # Análisis de patrones de subjuntivo (NUEVO)
        subjunctive_score = self._analyze_ai_subjunctive_patterns(text)
        if subjunctive_score > 0.2:
            detection_methods.append("ai_subjunctive_patterns")
            scores.append(subjunctive_score)
            method_weights["ai_subjunctive_patterns"] = 0.01
        
        # Detección de patrones de artículos (NUEVO)
        article_score = self._detect_ai_article_patterns(text)
        if article_score > 0.2:
            detection_methods.append("ai_article_patterns")
            scores.append(article_score)
            method_weights["ai_article_patterns"] = 0.01
        
        # Análisis de patrones de preposiciones (NUEVO)
        preposition_score = self._analyze_ai_preposition_patterns(text)
        if preposition_score > 0.2:
            detection_methods.append("ai_preposition_patterns")
            scores.append(preposition_score)
            method_weights["ai_preposition_patterns"] = 0.01
        
        # Detección de patrones de conjunciones (NUEVO)
        conjunction_score = self._detect_ai_conjunction_patterns(text)
        if conjunction_score > 0.2:
            detection_methods.append("ai_conjunction_patterns")
            scores.append(conjunction_score)
            method_weights["ai_conjunction_patterns"] = 0.01
        
        # Análisis de patrones de determinantes (NUEVO)
        determiner_score = self._analyze_ai_determiner_patterns(text)
        if determiner_score > 0.2:
            detection_methods.append("ai_determiner_patterns")
            scores.append(determiner_score)
            method_weights["ai_determiner_patterns"] = 0.01
        
        # Detección de patrones de referencia pronominal (NUEVO)
        pronoun_ref_score = self._detect_ai_pronoun_reference_patterns(text)
        if pronoun_ref_score > 0.2:
            detection_methods.append("ai_pronoun_reference_patterns")
            scores.append(pronoun_ref_score)
            method_weights["ai_pronoun_reference_patterns"] = 0.01
        
        # Análisis de patrones de adverbios (NUEVO)
        adverb_score = self._analyze_ai_adverb_patterns(text)
        if adverb_score > 0.2:
            detection_methods.append("ai_adverb_patterns")
            scores.append(adverb_score)
            method_weights["ai_adverb_patterns"] = 0.01
        
        # Detección de patrones de adjetivos (NUEVO)
        adjective_score = self._detect_ai_adjective_patterns(text)
        if adjective_score > 0.2:
            detection_methods.append("ai_adjective_patterns")
            scores.append(adjective_score)
            method_weights["ai_adjective_patterns"] = 0.01
        
        # Análisis de patrones de sustantivos (NUEVO)
        noun_score = self._analyze_ai_noun_patterns(text)
        if noun_score > 0.2:
            detection_methods.append("ai_noun_patterns")
            scores.append(noun_score)
            method_weights["ai_noun_patterns"] = 0.01
        
        # Detección de patrones de verbos (NUEVO)
        verb_score = self._detect_ai_verb_patterns(text)
        if verb_score > 0.2:
            detection_methods.append("ai_verb_patterns")
            scores.append(verb_score)
            method_weights["ai_verb_patterns"] = 0.01
        
        # Análisis de patrones de longitud de oraciones (NUEVO)
        sentence_length_score = self._analyze_ai_sentence_length_patterns(text)
        if sentence_length_score > 0.2:
            detection_methods.append("ai_sentence_length_patterns")
            scores.append(sentence_length_score)
            method_weights["ai_sentence_length_patterns"] = 0.01
        
        # Detección de patrones de estructura de párrafos (NUEVO)
        paragraph_structure_score = self._detect_ai_paragraph_structure_patterns(text)
        if paragraph_structure_score > 0.2:
            detection_methods.append("ai_paragraph_structure_patterns")
            scores.append(paragraph_structure_score)
            method_weights["ai_paragraph_structure_patterns"] = 0.01
        
        # Análisis de patrones de puntuación (NUEVO)
        punctuation_score = self._analyze_ai_punctuation_patterns(text)
        if punctuation_score > 0.2:
            detection_methods.append("ai_punctuation_patterns")
            scores.append(punctuation_score)
            method_weights["ai_punctuation_patterns"] = 0.01
        
        # Detección de patrones de capitalización (NUEVO)
        capitalization_score = self._detect_ai_capitalization_patterns(text)
        if capitalization_score > 0.2:
            detection_methods.append("ai_capitalization_patterns")
            scores.append(capitalization_score)
            method_weights["ai_capitalization_patterns"] = 0.01
        
        # Análisis de patrones de frecuencia de palabras (NUEVO)
        word_frequency_score = self._analyze_ai_word_frequency_patterns(text)
        if word_frequency_score > 0.2:
            detection_methods.append("ai_word_frequency_patterns")
            scores.append(word_frequency_score)
            method_weights["ai_word_frequency_patterns"] = 0.01
        
        # Detección de patrones de repetición de frases (NUEVO)
        phrase_repetition_score = self._detect_ai_phrase_repetition_patterns(text)
        if phrase_repetition_score > 0.2:
            detection_methods.append("ai_phrase_repetition_patterns")
            scores.append(phrase_repetition_score)
            method_weights["ai_phrase_repetition_patterns"] = 0.01
        
        # Análisis de patrones de densidad semántica (NUEVO)
        semantic_density_score = self._analyze_ai_semantic_density_patterns(text)
        if semantic_density_score > 0.2:
            detection_methods.append("ai_semantic_density_patterns")
            scores.append(semantic_density_score)
            method_weights["ai_semantic_density_patterns"] = 0.01
        
        # Detección de patrones de marcadores de coherencia (NUEVO)
        coherence_markers_score = self._detect_ai_coherence_markers_patterns(text)
        if coherence_markers_score > 0.2:
            detection_methods.append("ai_coherence_markers_patterns")
            scores.append(coherence_markers_score)
            method_weights["ai_coherence_markers_patterns"] = 0.01
        
        # Análisis de patrones de sofisticación léxica (NUEVO)
        lexical_sophistication_score = self._analyze_ai_lexical_sophistication_patterns(text)
        if lexical_sophistication_score > 0.2:
            detection_methods.append("ai_lexical_sophistication_patterns")
            scores.append(lexical_sophistication_score)
            method_weights["ai_lexical_sophistication_patterns"] = 0.01
        
        # Detección de patrones de formalidad (NUEVO)
        formality_score = self._detect_ai_formality_patterns(text)
        if formality_score > 0.2:
            detection_methods.append("ai_formality_patterns")
            scores.append(formality_score)
            method_weights["ai_formality_patterns"] = 0.01
        
        # Análisis de patrones de registro (NUEVO)
        register_score = self._analyze_ai_register_patterns(text)
        if register_score > 0.2:
            detection_methods.append("ai_register_patterns")
            scores.append(register_score)
            method_weights["ai_register_patterns"] = 0.01
        
        # Detección de patrones de marcadores discursivos (NUEVO)
        discourse_markers_score = self._detect_ai_discourse_markers_patterns(text)
        if discourse_markers_score > 0.2:
            detection_methods.append("ai_discourse_markers_patterns")
            scores.append(discourse_markers_score)
            method_weights["ai_discourse_markers_patterns"] = 0.01
        
        # Análisis de patrones de cohesión textual (NUEVO)
        textual_cohesion_score = self._analyze_ai_textual_cohesion_patterns(text)
        if textual_cohesion_score > 0.2:
            detection_methods.append("ai_textual_cohesion_patterns")
            scores.append(textual_cohesion_score)
            method_weights["ai_textual_cohesion_patterns"] = 0.01
        
        # Detección de patrones de densidad de información (NUEVO)
        information_density_score = self._detect_ai_information_density_patterns(text)
        if information_density_score > 0.2:
            detection_methods.append("ai_information_density_patterns")
            scores.append(information_density_score)
            method_weights["ai_information_density_patterns"] = 0.01
        
        # Análisis de patrones de densidad de hedging (NUEVO)
        hedging_density_score = self._analyze_ai_hedging_density_patterns(text)
        if hedging_density_score > 0.2:
            detection_methods.append("ai_hedging_density_patterns")
            scores.append(hedging_density_score)
            method_weights["ai_hedging_density_patterns"] = 0.01
        
        # Detección de patrones de voz autorial (NUEVO)
        authorial_voice_score = self._detect_ai_authorial_voice_patterns(text)
        if authorial_voice_score > 0.2:
            detection_methods.append("ai_authorial_voice_patterns")
            scores.append(authorial_voice_score)
            method_weights["ai_authorial_voice_patterns"] = 0.01
        
        # Análisis de patrones de variedad textual (NUEVO)
        textual_variety_score = self._analyze_ai_textual_variety_patterns(text)
        if textual_variety_score > 0.2:
            detection_methods.append("ai_textual_variety_patterns")
            scores.append(textual_variety_score)
            method_weights["ai_textual_variety_patterns"] = 0.01
        
        # Detección de patrones de repetición léxica (NUEVO)
        lexical_repetition_score = self._detect_ai_lexical_repetition_patterns(text)
        if lexical_repetition_score > 0.2:
            detection_methods.append("ai_lexical_repetition_patterns")
            scores.append(lexical_repetition_score)
            method_weights["ai_lexical_repetition_patterns"] = 0.01
        
        # Análisis de patrones de uniformidad sintáctica (NUEVO)
        syntactic_uniformity_score = self._analyze_ai_syntactic_uniformity_patterns(text)
        if syntactic_uniformity_score > 0.2:
            detection_methods.append("ai_syntactic_uniformity_patterns")
            scores.append(syntactic_uniformity_score)
            method_weights["ai_syntactic_uniformity_patterns"] = 0.01
        
        # Detección de patrones de expresión emocional (NUEVO)
        emotional_expression_score = self._detect_ai_emotional_expression_patterns(text)
        if emotional_expression_score > 0.2:
            detection_methods.append("ai_emotional_expression_patterns")
            scores.append(emotional_expression_score)
            method_weights["ai_emotional_expression_patterns"] = 0.01
        
        # Análisis de patrones de ambigüedad contextual (NUEVO V28)
        contextual_ambiguity_score = self._analyze_ai_contextual_ambiguity_patterns(text)
        if contextual_ambiguity_score > 0.2:
            detection_methods.append("ai_contextual_ambiguity_patterns")
            scores.append(contextual_ambiguity_score)
            method_weights["ai_contextual_ambiguity_patterns"] = 0.01
        
        # Detección de patrones de riqueza léxica (NUEVO V28)
        lexical_richness_score = self._detect_ai_lexical_richness_patterns(text)
        if lexical_richness_score > 0.2:
            detection_methods.append("ai_lexical_richness_patterns")
            scores.append(lexical_richness_score)
            method_weights["ai_lexical_richness_patterns"] = 0.01
        
        # Análisis de patrones de variación sintáctica (NUEVO V28)
        syntactic_variation_score = self._analyze_ai_syntactic_variation_patterns(text)
        if syntactic_variation_score > 0.2:
            detection_methods.append("ai_syntactic_variation_patterns")
            scores.append(syntactic_variation_score)
            method_weights["ai_syntactic_variation_patterns"] = 0.01
        
        # Detección de patrones de coherencia discursiva (NUEVO V28)
        discourse_coherence_score = self._detect_ai_discourse_coherence_patterns(text)
        if discourse_coherence_score > 0.2:
            detection_methods.append("ai_discourse_coherence_patterns")
            scores.append(discourse_coherence_score)
            method_weights["ai_discourse_coherence_patterns"] = 0.01
        
        # Análisis de patrones de ritmo textual (NUEVO V29)
        textual_rhythm_score = self._analyze_ai_textual_rhythm_patterns(text)
        if textual_rhythm_score > 0.2:
            detection_methods.append("ai_textual_rhythm_patterns")
            scores.append(textual_rhythm_score)
            method_weights["ai_textual_rhythm_patterns"] = 0.01
        
        # Detección de patrones de redundancia semántica (NUEVO V29)
        semantic_redundancy_score = self._detect_ai_semantic_redundancy_patterns(text)
        if semantic_redundancy_score > 0.2:
            detection_methods.append("ai_semantic_redundancy_patterns")
            scores.append(semantic_redundancy_score)
            method_weights["ai_semantic_redundancy_patterns"] = 0.01
        
        # Análisis avanzado de sofisticación léxica (NUEVO V29)
        lexical_sophistication_advanced_score = self._analyze_ai_lexical_sophistication_advanced(text)
        if lexical_sophistication_advanced_score > 0.2:
            detection_methods.append("ai_lexical_sophistication_advanced")
            scores.append(lexical_sophistication_advanced_score)
            method_weights["ai_lexical_sophistication_advanced"] = 0.01
        
        # Detección de patrones de marcadores pragmáticos (NUEVO V29)
        pragmatic_markers_score = self._detect_ai_pragmatic_markers_patterns(text)
        if pragmatic_markers_score > 0.2:
            detection_methods.append("ai_pragmatic_markers_patterns")
            scores.append(pragmatic_markers_score)
            method_weights["ai_pragmatic_markers_patterns"] = 0.01
        
        # Análisis de patrones conversacionales (NUEVO V30)
        conversational_score = self._analyze_ai_conversational_patterns(text)
        if conversational_score > 0.2:
            detection_methods.append("ai_conversational_patterns")
            scores.append(conversational_score)
            method_weights["ai_conversational_patterns"] = 0.01
        
        # Detección de patrones de metadiscurso (NUEVO V30)
        metadiscourse_score = self._detect_ai_metadiscourse_patterns(text)
        if metadiscourse_score > 0.2:
            detection_methods.append("ai_metadiscourse_patterns")
            scores.append(metadiscourse_score)
            method_weights["ai_metadiscourse_patterns"] = 0.01
        
        # Análisis de patrones de evidencialidad (NUEVO V30)
        evidentiality_score = self._analyze_ai_evidentiality_patterns(text)
        if evidentiality_score > 0.2:
            detection_methods.append("ai_evidentiality_patterns")
            scores.append(evidentiality_score)
            method_weights["ai_evidentiality_patterns"] = 0.01
        
        # Detección de patrones de engagement (NUEVO V30)
        engagement_score = self._detect_ai_engagement_patterns(text)
        if engagement_score > 0.2:
            detection_methods.append("ai_engagement_patterns")
            scores.append(engagement_score)
            method_weights["ai_engagement_patterns"] = 0.01
        
        # Análisis de patrones de cortesía (NUEVO V31)
        politeness_score = self._analyze_ai_politeness_patterns(text)
        if politeness_score > 0.2:
            detection_methods.append("ai_politeness_patterns")
            scores.append(politeness_score)
            method_weights["ai_politeness_patterns"] = 0.01
        
        # Detección de patrones de marcadores de formalidad (NUEVO V31)
        formality_markers_score = self._detect_ai_formality_markers_patterns(text)
        if formality_markers_score > 0.2:
            detection_methods.append("ai_formality_markers_patterns")
            scores.append(formality_markers_score)
            method_weights["ai_formality_markers_patterns"] = 0.01
        
        # Análisis avanzado de patrones de hedging (NUEVO V31)
        hedging_advanced_score = self._analyze_ai_hedging_advanced_patterns(text)
        if hedging_advanced_score > 0.2:
            detection_methods.append("ai_hedging_advanced_patterns")
            scores.append(hedging_advanced_score)
            method_weights["ai_hedging_advanced_patterns"] = 0.01
        
        # Detección de patrones de asertividad (NUEVO V31)
        assertiveness_score = self._detect_ai_assertiveness_patterns(text)
        if assertiveness_score > 0.2:
            detection_methods.append("ai_assertiveness_patterns")
            scores.append(assertiveness_score)
            method_weights["ai_assertiveness_patterns"] = 0.01
        
        # Análisis de patrones de intertextualidad (NUEVO V32)
        intertextuality_score = self._analyze_ai_intertextuality_patterns(text)
        if intertextuality_score > 0.2:
            detection_methods.append("ai_intertextuality_patterns")
            scores.append(intertextuality_score)
            method_weights["ai_intertextuality_patterns"] = 0.01
        
        # Detección de patrones de densidad de citas (NUEVO V32)
        citation_density_score = self._detect_ai_citation_density_patterns(text)
        if citation_density_score > 0.2:
            detection_methods.append("ai_citation_density_patterns")
            scores.append(citation_density_score)
            method_weights["ai_citation_density_patterns"] = 0.01
        
        # Análisis de patrones de afirmaciones de autoridad (NUEVO V32)
        authority_claims_score = self._analyze_ai_authority_claims_patterns(text)
        if authority_claims_score > 0.2:
            detection_methods.append("ai_authority_claims_patterns")
            scores.append(authority_claims_score)
            method_weights["ai_authority_claims_patterns"] = 0.01
        
        # Detección de patrones de marcadores de expertise (NUEVO V32)
        expertise_markers_score = self._detect_ai_expertise_markers_patterns(text)
        if expertise_markers_score > 0.2:
            detection_methods.append("ai_expertise_markers_patterns")
            scores.append(expertise_markers_score)
            method_weights["ai_expertise_markers_patterns"] = 0.01
        
        # Análisis de patrones de coherencia temporal (NUEVO V33)
        temporal_coherence_score = self._analyze_ai_temporal_coherence_patterns(text)
        if temporal_coherence_score > 0.2:
            detection_methods.append("ai_temporal_coherence_patterns")
            scores.append(temporal_coherence_score)
            method_weights["ai_temporal_coherence_patterns"] = 0.01
        
        # Detección de patrones de cadenas causales (NUEVO V33)
        causal_chain_score = self._detect_ai_causal_chain_patterns(text)
        if causal_chain_score > 0.2:
            detection_methods.append("ai_causal_chain_patterns")
            scores.append(causal_chain_score)
            method_weights["ai_causal_chain_patterns"] = 0.01
        
        # Análisis de patrones de estructura narrativa (NUEVO V33)
        narrative_structure_score = self._analyze_ai_narrative_structure_patterns(text)
        if narrative_structure_score > 0.2:
            detection_methods.append("ai_narrative_structure_patterns")
            scores.append(narrative_structure_score)
            method_weights["ai_narrative_structure_patterns"] = 0.01
        
        # Detección de patrones de argumentación (NUEVO V33)
        argumentation_score = self._detect_ai_argumentation_patterns(text)
        if argumentation_score > 0.2:
            detection_methods.append("ai_argumentation_patterns")
            scores.append(argumentation_score)
            method_weights["ai_argumentation_patterns"] = 0.01
        
        # Análisis de patrones de consistencia léxica (NUEVO V34)
        lexical_consistency_score = self._analyze_ai_lexical_consistency_patterns(text)
        if lexical_consistency_score > 0.2:
            detection_methods.append("ai_lexical_consistency_patterns")
            scores.append(lexical_consistency_score)
            method_weights["ai_lexical_consistency_patterns"] = 0.01
        
        # Detección de patrones de campos semánticos (NUEVO V34)
        semantic_field_score = self._detect_ai_semantic_field_patterns(text)
        if semantic_field_score > 0.2:
            detection_methods.append("ai_semantic_field_patterns")
            scores.append(semantic_field_score)
            method_weights["ai_semantic_field_patterns"] = 0.01
        
        # Análisis de patrones de consistencia de registro (NUEVO V34)
        register_consistency_score = self._analyze_ai_register_consistency_patterns(text)
        if register_consistency_score > 0.2:
            detection_methods.append("ai_register_consistency_patterns")
            scores.append(register_consistency_score)
            method_weights["ai_register_consistency_patterns"] = 0.01
        
        # Detección de patrones de uniformidad estilística (NUEVO V34)
        stylistic_uniformity_score = self._detect_ai_stylistic_uniformity_patterns(text)
        if stylistic_uniformity_score > 0.2:
            detection_methods.append("ai_stylistic_uniformity_patterns")
            scores.append(stylistic_uniformity_score)
            method_weights["ai_stylistic_uniformity_patterns"] = 0.01
        
        # Análisis de patrones de fraseología (NUEVO V35)
        phraseology_score = self._analyze_ai_phraseology_patterns(text)
        if phraseology_score > 0.2:
            detection_methods.append("ai_phraseology_patterns")
            scores.append(phraseology_score)
            method_weights["ai_phraseology_patterns"] = 0.01
        
        # Detección de patrones de colocaciones (NUEVO V35)
        collocation_score = self._detect_ai_collocation_patterns(text)
        if collocation_score > 0.2:
            detection_methods.append("ai_collocation_patterns")
            scores.append(collocation_score)
            method_weights["ai_collocation_patterns"] = 0.01
        
        # Análisis de patrones idiomáticos (NUEVO V35)
        idiomatic_score = self._analyze_ai_idiomatic_patterns(text)
        if idiomatic_score > 0.2:
            detection_methods.append("ai_idiomatic_patterns")
            scores.append(idiomatic_score)
            method_weights["ai_idiomatic_patterns"] = 0.01
        
        # Detección de patrones de referencias culturales (NUEVO V35)
        cultural_references_score = self._detect_ai_cultural_references_patterns(text)
        if cultural_references_score > 0.2:
            detection_methods.append("ai_cultural_references_patterns")
            scores.append(cultural_references_score)
            method_weights["ai_cultural_references_patterns"] = 0.01
        
        # Análisis de patrones metafóricos (NUEVO V36)
        metaphorical_score = self._analyze_ai_metaphorical_patterns(text)
        if metaphorical_score > 0.2:
            detection_methods.append("ai_metaphorical_patterns")
            scores.append(metaphorical_score)
            method_weights["ai_metaphorical_patterns"] = 0.01
        
        # Detección de patrones analógicos (NUEVO V36)
        analogical_score = self._detect_ai_analogical_patterns(text)
        if analogical_score > 0.2:
            detection_methods.append("ai_analogical_patterns")
            scores.append(analogical_score)
            method_weights["ai_analogical_patterns"] = 0.01
        
        # Análisis de patrones de ironía (NUEVO V36)
        irony_score = self._analyze_ai_irony_patterns(text)
        if irony_score > 0.2:
            detection_methods.append("ai_irony_patterns")
            scores.append(irony_score)
            method_weights["ai_irony_patterns"] = 0.01
        
        # Detección de patrones de humor (NUEVO V36)
        humor_score = self._detect_ai_humor_patterns(text)
        if humor_score > 0.2:
            detection_methods.append("ai_humor_patterns")
            scores.append(humor_score)
            method_weights["ai_humor_patterns"] = 0.01
        
        # Análisis de patrones de sarcasmo (NUEVO V37)
        sarcasm_score = self._analyze_ai_sarcasm_patterns(text)
        if sarcasm_score > 0.2:
            detection_methods.append("ai_sarcasm_patterns")
            scores.append(sarcasm_score)
            method_weights["ai_sarcasm_patterns"] = 0.01
        
        # Detección de patrones de hipérbole (NUEVO V37)
        hyperbole_score = self._detect_ai_hyperbole_patterns(text)
        if hyperbole_score > 0.2:
            detection_methods.append("ai_hyperbole_patterns")
            scores.append(hyperbole_score)
            method_weights["ai_hyperbole_patterns"] = 0.01
        
        # Análisis de patrones de eufemismo (NUEVO V37)
        euphemism_score = self._analyze_ai_euphemism_patterns(text)
        if euphemism_score > 0.2:
            detection_methods.append("ai_euphemism_patterns")
            scores.append(euphemism_score)
            method_weights["ai_euphemism_patterns"] = 0.01
        
        # Detección de patrones de lítote (NUEVO V37)
        understatement_score = self._detect_ai_understatement_patterns(text)
        if understatement_score > 0.2:
            detection_methods.append("ai_understatement_patterns")
            scores.append(understatement_score)
            method_weights["ai_understatement_patterns"] = 0.01
        
        # Análisis de patrones de aliteración (NUEVO V38)
        alliteration_score = self._analyze_ai_alliteration_patterns(text)
        if alliteration_score > 0.2:
            detection_methods.append("ai_alliteration_patterns")
            scores.append(alliteration_score)
            method_weights["ai_alliteration_patterns"] = 0.01
        
        # Detección de patrones de asonancia (NUEVO V38)
        assonance_score = self._detect_ai_assonance_patterns(text)
        if assonance_score > 0.2:
            detection_methods.append("ai_assonance_patterns")
            scores.append(assonance_score)
            method_weights["ai_assonance_patterns"] = 0.01
        
        # Análisis de patrones de ritmo poético (NUEVO V38)
        rhythm_score = self._analyze_ai_rhythm_patterns(text)
        if rhythm_score > 0.2:
            detection_methods.append("ai_rhythm_patterns")
            scores.append(rhythm_score)
            method_weights["ai_rhythm_patterns"] = 0.01
        
        # Detección de patrones poéticos (NUEVO V38)
        poetic_score = self._detect_ai_poetic_patterns(text)
        if poetic_score > 0.2:
            detection_methods.append("ai_poetic_patterns")
            scores.append(poetic_score)
            method_weights["ai_poetic_patterns"] = 0.01
        
        # Análisis de patrones de densidad léxica (NUEVO V39)
        lexical_density_score = self._analyze_ai_lexical_density_patterns(text)
        if lexical_density_score > 0.2:
            detection_methods.append("ai_lexical_density_patterns")
            scores.append(lexical_density_score)
            method_weights["ai_lexical_density_patterns"] = 0.01
        
        # Detección de patrones de redes semánticas (NUEVO V39)
        semantic_network_score = self._detect_ai_semantic_network_patterns(text)
        if semantic_network_score > 0.2:
            detection_methods.append("ai_semantic_network_patterns")
            scores.append(semantic_network_score)
            method_weights["ai_semantic_network_patterns"] = 0.01
        
        # Análisis de patrones de coherencia conceptual (NUEVO V39)
        conceptual_coherence_score = self._analyze_ai_conceptual_coherence_patterns(text)
        if conceptual_coherence_score > 0.2:
            detection_methods.append("ai_conceptual_coherence_patterns")
            scores.append(conceptual_coherence_score)
            method_weights["ai_conceptual_coherence_patterns"] = 0.01
        
        # Detección de patrones de grafos de conocimiento (NUEVO V39)
        knowledge_graph_score = self._detect_ai_knowledge_graph_patterns(text)
        if knowledge_graph_score > 0.2:
            detection_methods.append("ai_knowledge_graph_patterns")
            scores.append(knowledge_graph_score)
            method_weights["ai_knowledge_graph_patterns"] = 0.01
        
        # Calcular porcentaje de IA con pesos ponderados
        if scores:
            # Si hay pesos, usar promedio ponderado
            if method_weights and len(scores) == len(method_weights):
                total_weight = sum(method_weights.values())
                weighted_sum = sum(score * method_weights.get(method, 0.1) 
                                 for score, method in zip(scores, detection_methods))
                avg_score = weighted_sum / total_weight if total_weight > 0 else np.mean(scores)
            else:
                avg_score = np.mean(scores)
            
            ai_percentage = avg_score * 100
            is_ai = ai_percentage > 50.0
            confidence = max(scores) if scores else 0.0
        else:
            ai_percentage = 0.0
            is_ai = False
            confidence = 0.0
        
        # Análisis forense mejorado
        forensic = self._forensic_analysis(text, detected_models, text_stats)
        
        # Modelo principal
        primary_model = None
        if detected_models:
            primary_model = max(detected_models, key=lambda x: x["confidence"])
        
        # Calcular score de riesgo y confiabilidad
        risk_score = self._analyze_risk_and_reliability(text, detected_models, scores) if scores else 0.0
        
        # Información adicional de calidad
        quality_info = {
            "writing_quality": self._analyze_writing_quality(text) if len(text.split()) > 10 else 0.0,
            "paraphrase_likelihood": self._detect_paraphrasing(text) if len(text.split()) > 20 else 0.0,
            "risk_score": risk_score,
            "reliability": "high" if risk_score > 0.7 else "medium" if risk_score > 0.4 else "low"
        }
        
        # Sistema de alertas (NUEVO)
        alerts = self._generate_alerts(ai_percentage, confidence, detected_models, primary_model)
        
        return {
            "is_ai_generated": is_ai,
            "ai_percentage": ai_percentage,
            "detected_models": detected_models,
            "primary_model": primary_model,
            "forensic_analysis": forensic,
            "confidence_score": confidence,
            "detection_methods": detection_methods,
            "quality_info": quality_info,
            "alerts": alerts
        }
    
    def _detect_model_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Detecta patrones específicos de modelos de IA - MEJORADO"""
        detections = []
        text_lower = text.lower()
        
        for model_name, model_info in self.AI_MODEL_PATTERNS.items():
            matches = 0
            total_patterns = len(model_info["patterns"])
            pattern_matches = []
            
            # Contar matches de patrones
            for pattern in model_info["patterns"]:
                pattern_matches_found = len(re.findall(pattern, text_lower, re.IGNORECASE))
                if pattern_matches_found > 0:
                    matches += 1
                    pattern_matches.append(pattern_matches_found)
            
            # Detectar versión si hay version_patterns
            detected_version = None
            if "version_patterns" in model_info:
                for version_pattern in model_info["version_patterns"]:
                    if re.search(version_pattern, text_lower, re.IGNORECASE):
                        detected_version = version_pattern.replace(r"\.", ".").replace(r"\-", "-")
                        break
            
            if matches > 0:
                # Calcular confianza mejorada
                base_confidence = model_info["confidence_base"]
                match_ratio = matches / total_patterns
                
                # Bonus por múltiples matches del mismo patrón
                total_occurrences = sum(pattern_matches) if pattern_matches else matches
                occurrence_bonus = min(0.1 * (total_occurrences - matches), 0.15)
                
                confidence = base_confidence * match_ratio + occurrence_bonus
                confidence = min(confidence, 0.95)
                
                detections.append({
                    "model_name": model_name,
                    "confidence": confidence,
                    "provider": model_info["provider"],
                    "version": detected_version,
                    "matches": matches,
                    "total_occurrences": total_occurrences
                })
        
        return sorted(detections, key=lambda x: x["confidence"], reverse=True)
    
    def _analyze_text_statistics(self, text: str) -> Dict[str, Any]:
        """Analiza estadísticas del texto para detectar IA - MEJORADO"""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) == 0 or len(sentences) == 0:
            return {"ai_likelihood": 0.0}
        
        # Longitud promedio de oraciones
        sentence_lengths = [len(s.split()) for s in sentences]
        avg_sentence_length = np.mean(sentence_lengths)
        
        # Variación en longitud de oraciones (burstiness)
        if len(sentence_lengths) > 1:
            burstiness = np.std(sentence_lengths) / np.mean(sentence_lengths) if np.mean(sentence_lengths) > 0 else 0
        else:
            burstiness = 0
        
        # Repetición de palabras
        word_freq = {}
        for word in words:
            word_lower = word.lower().strip('.,!?;:()[]{}"\'')
            if word_lower:
                word_freq[word_lower] = word_freq.get(word_lower, 0) + 1
        
        max_freq = max(word_freq.values()) if word_freq else 0
        repetition_ratio = max_freq / len(words) if len(words) > 0 else 0
        
        # Diversidad de vocabulario (type-token ratio)
        unique_words = len(word_freq)
        vocab_diversity = unique_words / len(words) if len(words) > 0 else 0
        
        # Análisis de palabras funcionales vs contenido
        functional_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                          'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 
                          'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 
                          'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
        functional_count = sum(1 for w in words if w.lower() in functional_words)
        functional_ratio = functional_count / len(words) if len(words) > 0 else 0
        
        # Calcular likelihood de IA mejorado
        ai_score = 0.0
        
        # 1. Longitud de oraciones (IA suele tener oraciones de longitud similar)
        if 10 <= avg_sentence_length <= 25:
            ai_score += 0.15
        elif 8 <= avg_sentence_length <= 30:
            ai_score += 0.10
        
        # 2. Burstiness (IA tiene menos variación)
        if burstiness < 0.3:
            ai_score += 0.25
        elif burstiness < 0.5:
            ai_score += 0.15
        
        # 3. Repetición (IA evita repetición excesiva)
        if repetition_ratio < 0.2:
            ai_score += 0.15
        elif repetition_ratio < 0.3:
            ai_score += 0.10
        
        # 4. Diversidad de vocabulario (IA tiene buena diversidad)
        if 0.4 <= vocab_diversity <= 0.7:
            ai_score += 0.15
        elif vocab_diversity > 0.7:
            ai_score += 0.05  # Demasiada diversidad puede ser humano
        
        # 5. Ratio de palabras funcionales (IA usa más palabras funcionales)
        if 0.3 <= functional_ratio <= 0.5:
            ai_score += 0.15
        
        # 6. Longitud del texto (textos largos de IA son más coherentes)
        if len(words) > 100:
            ai_score += 0.10
        elif len(words) > 50:
            ai_score += 0.05
        
        # 7. Consistencia en puntuación
        punctuation_consistency = self._check_punctuation_consistency(text)
        if punctuation_consistency > 0.7:
            ai_score += 0.05
        
        # Normalizar a 0-1
        ai_likelihood = min(ai_score, 1.0)
        
        return {
            "ai_likelihood": ai_likelihood,
            "avg_sentence_length": avg_sentence_length,
            "burstiness": burstiness,
            "repetition_ratio": repetition_ratio,
            "vocab_diversity": vocab_diversity,
            "functional_ratio": functional_ratio
        }
    
    def _check_punctuation_consistency(self, text: str) -> float:
        """Verifica consistencia en el uso de puntuación"""
        # Contar diferentes tipos de puntuación
        periods = text.count('.')
        commas = text.count(',')
        exclamations = text.count('!')
        questions = text.count('?')
        semicolons = text.count(';')
        colons = text.count(':')
        
        total_punct = periods + commas + exclamations + questions + semicolons + colons
        if total_punct == 0:
            return 0.5  # Sin puntuación, neutral
        
        # IA suele usar más puntos y comas de forma consistente
        if periods > 0 and commas > 0:
            consistency = 0.8
        elif periods > 0:
            consistency = 0.6
        else:
            consistency = 0.4
        
        return consistency
    
    def _analyze_text_structure(self, text: str) -> float:
        """Analiza la estructura del texto"""
        score = 0.0
        
        # Presencia de estructura organizada
        if re.search(r'\b(first|second|third|finally|in conclusion|to summarize)', text, re.IGNORECASE):
            score += 0.3
        
        # Uso de listas o enumeraciones
        if re.search(r'\d+\.\s|\-\s|\*\s', text):
            score += 0.2
        
        # Párrafos bien formados
        paragraphs = text.split('\n\n')
        if len(paragraphs) > 1:
            avg_para_length = np.mean([len(p.split()) for p in paragraphs if p.strip()])
            if 50 <= avg_para_length <= 200:
                score += 0.3
        
        # Coherencia temática
        if len(text.split()) > 100:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_text_style(self, text: str) -> float:
        """Analiza el estilo del texto"""
        score = 0.0
        
        # Formalidad
        formal_words = ['therefore', 'however', 'furthermore', 'moreover', 'consequently', 'additionally']
        formal_count = sum(1 for word in formal_words if word.lower() in text.lower())
        if formal_count > 0:
            score += 0.3
        
        # Evita contracciones (más formal)
        contractions = ["don't", "won't", "can't", "it's", "that's", "there's"]
        contraction_count = sum(1 for c in contractions if c.lower() in text.lower())
        if contraction_count == 0 and len(text.split()) > 50:
            score += 0.2
        
        # Uso de vocabulario sofisticado
        sophisticated_patterns = [
            r'\b(?:utilize|facilitate|implement|optimize|enhance)\b',
            r'\b(?:comprehensive|systematic|methodical|strategic)\b'
        ]
        sophisticated_count = sum(1 for pattern in sophisticated_patterns if re.search(pattern, text, re.IGNORECASE))
        if sophisticated_count > 0:
            score += 0.3
        
        # Evita errores comunes
        common_errors = ['teh', 'adn', 'taht', 'recieve']
        error_count = sum(1 for error in common_errors if error in text.lower())
        if error_count == 0:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_entropy_and_ngrams(self, text: str) -> float:
        """Analiza entropía y n-gramas para detectar IA - NUEVO"""
        words = text.lower().split()
        if len(words) < 10:
            return 0.0
        
        # Calcular entropía de caracteres
        char_freq = {}
        for char in text.lower():
            if char.isalnum() or char.isspace():
                char_freq[char] = char_freq.get(char, 0) + 1
        
        total_chars = sum(char_freq.values())
        if total_chars == 0:
            return 0.0
        
        entropy = 0.0
        for count in char_freq.values():
            prob = count / total_chars
            if prob > 0:
                entropy -= prob * np.log2(prob)
        
        # Texto de IA suele tener entropía más baja (más predecible)
        # Entropía normal típica: 4-5 bits, IA: 3.5-4.5 bits
        if 3.5 <= entropy <= 4.5:
            entropy_score = 0.4
        elif entropy < 3.5:
            entropy_score = 0.6
        else:
            entropy_score = 0.2
        
        # Análisis de bigramas comunes de IA
        bigrams = {}
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            bigrams[bigram] = bigrams.get(bigram, 0) + 1
        
        # Bigramas muy comunes sugieren texto de IA
        if bigrams:
            max_bigram_freq = max(bigrams.values())
            bigram_diversity = len(bigrams) / len(words) if len(words) > 0 else 0
            
            if bigram_diversity < 0.3:  # Poca diversidad
                bigram_score = 0.3
            else:
                bigram_score = 0.1
        
        return min((entropy_score + bigram_score) / 2, 1.0)
    
    def _analyze_semantic_coherence(self, text: str) -> float:
        """Analiza coherencia semántica del texto - NUEVO"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 3:
            return 0.0
        
        score = 0.0
        
        # 1. Palabras de transición (indican coherencia)
        transition_words = [
            'however', 'therefore', 'furthermore', 'moreover', 'consequently',
            'additionally', 'meanwhile', 'subsequently', 'nevertheless', 'thus',
            'hence', 'accordingly', 'indeed', 'specifically', 'particularly'
        ]
        transition_count = sum(1 for word in transition_words if word.lower() in text.lower())
        if transition_count > 0:
            score += min(transition_count * 0.1, 0.3)
        
        # 2. Referencias pronominales consistentes
        pronouns = ['it', 'this', 'that', 'these', 'those', 'they', 'he', 'she']
        pronoun_count = sum(1 for p in pronouns if re.search(rf'\b{p}\b', text.lower()))
        if pronoun_count > 2:
            score += 0.2
        
        # 3. Repetición de conceptos clave (coherencia temática)
        words = text.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 4:  # Solo palabras significativas
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Conceptos que aparecen múltiples veces
        key_concepts = [w for w, count in word_freq.items() if count > 2]
        if len(key_concepts) > 0:
            score += min(len(key_concepts) * 0.05, 0.3)
        
        # 4. Estructura lógica (si...entonces, porque...por lo tanto)
        logical_patterns = [
            r'\b(?:if|when|because|since|as)\b.*\b(?:then|therefore|thus|so|hence)\b',
            r'\b(?:first|second|third|finally|in conclusion)\b',
            r'\b(?:on one hand|on the other hand)\b'
        ]
        logical_count = sum(1 for pattern in logical_patterns if re.search(pattern, text, re.IGNORECASE))
        if logical_count > 0:
            score += min(logical_count * 0.15, 0.2)
        
        return min(score, 1.0)
    
    def _analyze_syntactic_complexity(self, text: str) -> float:
        """Analiza complejidad sintáctica del texto - NUEVO"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) == 0:
            return 0.0
        
        score = 0.0
        
        # 1. Oraciones compuestas y complejas
        complex_markers = [
            r'\b(?:although|though|while|whereas|despite|in spite of)\b',
            r'\b(?:which|that|who|whom|whose)\b',  # Cláusulas relativas
            r'\b(?:because|since|as|due to|owing to)\b',  # Cláusulas causales
            r'\b(?:if|unless|provided that|as long as)\b'  # Cláusulas condicionales
        ]
        
        complex_sentences = 0
        for sentence in sentences:
            complex_count = sum(1 for marker in complex_markers if re.search(marker, sentence, re.IGNORECASE))
            if complex_count > 0:
                complex_sentences += 1
        
        if len(sentences) > 0:
            complex_ratio = complex_sentences / len(sentences)
            if 0.3 <= complex_ratio <= 0.7:  # Balance ideal
                score += 0.3
            elif complex_ratio > 0.7:
                score += 0.2
        
        # 2. Uso de gerundios y participios
        gerund_patterns = [
            r'\b\w+ing\b',  # Gerundios
            r'\b\w+ed\b',   # Participios pasados
            r'\b\w+en\b'    # Participios irregulares
        ]
        gerund_count = sum(len(re.findall(pattern, text)) for pattern in gerund_patterns)
        if gerund_count > len(sentences) * 0.5:
            score += 0.2
        
        # 3. Uso de preposiciones complejas
        complex_prepositions = [
            'according to', 'in addition to', 'in spite of', 'on behalf of',
            'with regard to', 'in terms of', 'in case of', 'by means of'
        ]
        prep_count = sum(1 for prep in complex_prepositions if prep in text.lower())
        if prep_count > 0:
            score += min(prep_count * 0.1, 0.2)
        
        # 4. Uso de voz pasiva (más compleja)
        passive_patterns = [
            r'\b(?:is|are|was|were|been|being)\s+\w+ed\b',
            r'\b(?:is|are|was|were)\s+\w+en\b'
        ]
        passive_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in passive_patterns)
        if passive_count > 0:
            score += min(passive_count * 0.05, 0.15)
        
        # 5. Longitud promedio de oraciones (oraciones más largas = más complejas)
        avg_length = np.mean([len(s.split()) for s in sentences])
        if 15 <= avg_length <= 25:
            score += 0.15
        
        return min(score, 1.0)
    
    def _analyze_citations_and_references(self, text: str) -> float:
        """Analiza citas y referencias en el texto - NUEVO"""
        score = 0.0
        
        # 1. Detectar citas directas
        direct_quotes = len(re.findall(r'["""].*?["""]', text))
        if direct_quotes > 0:
            score += 0.2
        
        # 2. Detectar referencias académicas
        academic_patterns = [
            r'\([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?,\s*\d{4}\)',  # (Author, 2024)
            r'\[.*?\]',  # [1], [Author 2024]
            r'\d+\.\s*[A-Z][a-z]+.*?\d{4}',  # 1. Author (2024)
            r'\(see\s+.*?\)',  # (see Author, 2024)
            r'according to\s+[A-Z][a-z]+',  # according to Author
            r'as\s+[A-Z][a-z]+\s+notes?',  # as Author notes
            r'[A-Z][a-z]+\s+\(\d{4}\)'  # Author (2024)
        ]
        
        citation_count = sum(len(re.findall(pattern, text)) for pattern in academic_patterns)
        if citation_count > 0:
            score += min(citation_count * 0.15, 0.4)
        
        # 3. Detectar números de página o secciones
        page_refs = len(re.findall(r'\b(?:p\.|pp\.|page|pages)\s+\d+', text, re.IGNORECASE))
        if page_refs > 0:
            score += 0.2
        
        # 4. Detectar URLs o enlaces
        url_pattern = r'https?://\S+|www\.\S+'
        urls = len(re.findall(url_pattern, text))
        if urls > 0:
            score += min(urls * 0.1, 0.2)
        
        # 5. Detectar notas al pie
        footnote_patterns = [
            r'\d+\s*\(footnote|note\)',
            r'\[note\s+\d+\]',
            r'\(see note \d+\)'
        ]
        footnotes = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in footnote_patterns)
        if footnotes > 0:
            score += 0.1
        
        # Texto con muchas citas suele ser más humano (académico)
        # Pero texto de IA puede imitar citas, así que score moderado
        return min(score, 1.0)
    
    def _analyze_temporal_consistency(self, text: str) -> float:
        """Analiza consistencia temporal del texto - NUEVO"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 3:
            return 0.0
        
        score = 0.0
        
        # 1. Análisis de cambios de estilo a lo largo del texto
        # Dividir texto en tercios
        third = len(sentences) // 3
        first_third = sentences[:third] if third > 0 else sentences[:1]
        middle_third = sentences[third:2*third] if third > 0 else sentences[1:2]
        last_third = sentences[2*third:] if third > 0 else sentences[2:]
        
        # Calcular longitud promedio de oraciones en cada tercio
        if first_third and middle_third and last_third:
            avg_first = np.mean([len(s.split()) for s in first_third])
            avg_middle = np.mean([len(s.split()) for s in middle_third])
            avg_last = np.mean([len(s.split()) for s in last_third])
            
            # Texto de IA suele tener consistencia (poca variación)
            variation = np.std([avg_first, avg_middle, avg_last])
            if variation < 3.0:  # Poca variación = más consistente = posible IA
                score += 0.3
        
        # 2. Análisis de vocabulario a lo largo del texto
        first_words = set(word.lower() for s in first_third for word in s.split() if len(word) > 4)
        last_words = set(word.lower() for s in last_third for word in s.split() if len(word) > 4)
        
        if first_words and last_words:
            overlap = len(first_words & last_words) / len(first_words | last_words) if (first_words | last_words) else 0
            if overlap > 0.3:  # Alto solapamiento = consistencia temática
                score += 0.2
        
        # 3. Detección de cambios abruptos de tono
        formal_words = ['therefore', 'however', 'furthermore', 'moreover', 'consequently']
        informal_words = ["don't", "won't", "can't", "it's", "that's", "gonna", "wanna"]
        
        formal_count_first = sum(1 for word in formal_words if any(word in s.lower() for s in first_third))
        informal_count_first = sum(1 for word in informal_words if any(word in s.lower() for s in first_third))
        formal_count_last = sum(1 for word in formal_words if any(word in s.lower() for s in last_third))
        informal_count_last = sum(1 for word in informal_words if any(word in s.lower() for s in last_third))
        
        # Cambios abruptos sugieren edición humana
        if abs(formal_count_first - formal_count_last) > 2 or abs(informal_count_first - informal_count_last) > 2:
            score -= 0.2  # Penalizar cambios abruptos (más humano)
        
        return max(min(score, 1.0), 0.0)
    
    def _detect_watermarks(self, text: str) -> float:
        """Detecta posibles watermarks en el texto - NUEVO"""
        score = 0.0
        
        # 1. Patrones de watermark conocidos
        watermark_patterns = [
            r'\b(?:generated by|created by|produced by)\s+(?:ai|artificial intelligence|chatgpt|gpt|claude)',
            r'\b(?:this (?:text|content|document) (?:was|is) (?:generated|created|produced))',
            r'\[ai generated\]|\[generated by ai\]|\[ai content\]',
            r'<!--.*?ai.*?-->',  # Comentarios HTML
            r'\/\*.*?ai.*?\*\/',  # Comentarios de código
        ]
        
        watermark_matches = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in watermark_patterns)
        if watermark_matches > 0:
            score += 0.5
        
        # 2. Patrones de caracteres especiales sospechosos
        # Algunos watermarks usan caracteres Unicode especiales
        special_chars = ['\u200b', '\u200c', '\u200d', '\u2060', '\u2061', '\u2062', '\u2063']
        special_count = sum(text.count(char) for char in special_chars)
        if special_count > 0:
            score += min(special_count * 0.1, 0.3)
        
        # 3. Patrones de espaciado inusual
        # Espacios múltiples o patrones repetitivos
        double_spaces = len(re.findall(r'  +', text))
        if double_spaces > len(text.split()) * 0.1:  # Más del 10% de espacios dobles
            score += 0.1
        
        # 4. Patrones de hash o identificadores ocultos
        hash_patterns = [
            r'\b[a-f0-9]{8,}\b',  # Hashes hexadecimales
            r'[A-Z0-9]{10,}',  # Identificadores alfanuméricos largos
        ]
        hash_matches = sum(len(re.findall(pattern, text)) for pattern in hash_patterns)
        if hash_matches > 0:
            score += min(hash_matches * 0.05, 0.1)
        
        return min(score, 1.0)
    
    def _detect_edits_and_patches(self, text: str) -> float:
        """Detecta ediciones y parches en el texto - NUEVO"""
        score = 0.0
        
        # 1. Detectar correcciones o ediciones explícitas
        edit_markers = [
            r'\[edit\]|\[edited\]|\[correction\]|\[updated\]',
            r'\(edit:.*?\)|\(edited:.*?\)|\(correction:.*?\)',
            r'note:.*?edit|note:.*?correction',
        ]
        
        edit_matches = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in edit_markers)
        if edit_matches > 0:
            score += 0.4
        
        # 2. Detectar cambios de formato abruptos
        # Mezcla de mayúsculas/minúsculas inconsistentes
        words = text.split()
        if len(words) > 10:
            all_caps = sum(1 for w in words if w.isupper() and len(w) > 1)
            all_lower = sum(1 for w in words if w.islower() and len(w) > 1)
            mixed_case = len(words) - all_caps - all_lower
            
            # Mezcla excesiva sugiere edición
            if mixed_case > len(words) * 0.3:
                score += 0.2
        
        # 3. Detectar paréntesis o corchetes de edición
        # Muchos paréntesis pueden indicar aclaraciones añadidas
        paren_ratio = (text.count('(') + text.count('[')) / len(text.split()) if len(text.split()) > 0 else 0
        if paren_ratio > 0.1:  # Más del 10% de palabras tienen paréntesis
            score += 0.2
        
        # 4. Detectar cambios de estilo dentro del texto
        # Texto que empieza formal y termina informal (o viceversa)
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) > 5:
            first_half = ' '.join(sentences[:len(sentences)//2])
            second_half = ' '.join(sentences[len(sentences)//2:])
            
            # Contar palabras formales vs informales
            formal_words = ['therefore', 'however', 'furthermore', 'moreover']
            informal_words = ["don't", "won't", "can't", "it's", "that's"]
            
            formal_first = sum(1 for w in formal_words if w in first_half.lower())
            informal_first = sum(1 for w in informal_words if w in first_half.lower())
            formal_second = sum(1 for w in formal_words if w in second_half.lower())
            informal_second = sum(1 for w in informal_words if w in second_half.lower())
            
            # Cambio significativo de estilo
            if abs(formal_first - formal_second) > 2 or abs(informal_first - informal_second) > 2:
                score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_sentiment_patterns(self, text: str) -> float:
        """Analiza patrones de sentimientos en el texto - NUEVO"""
        score = 0.0
        
        # 1. Detectar emociones típicas de IA
        # Texto de IA suele ser más neutral y menos emocional
        emotional_words = [
            'love', 'hate', 'angry', 'sad', 'happy', 'excited', 'frustrated',
            'amazing', 'terrible', 'awesome', 'awful', 'fantastic', 'horrible'
        ]
        emotional_count = sum(1 for word in emotional_words if word.lower() in text.lower())
        
        # Texto con muy pocas emociones puede ser IA
        word_count = len(text.split())
        if word_count > 50:
            emotional_ratio = emotional_count / word_count
            if emotional_ratio < 0.01:  # Menos del 1% de palabras emocionales
                score += 0.3
            elif emotional_ratio < 0.02:
                score += 0.2
        
        # 2. Detectar uso de emojis o expresiones
        # Texto de IA suele evitar emojis
        emoji_pattern = r'[😀-🙏🌀-🗿]'
        emoji_count = len(re.findall(emoji_pattern, text))
        if emoji_count == 0 and word_count > 50:
            score += 0.1
        
        # 3. Análisis de polaridad
        # Texto de IA suele ser más balanceado
        positive_words = ['good', 'great', 'excellent', 'wonderful', 'fantastic', 'amazing', 'best']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'poor', 'worst']
        
        positive_count = sum(1 for word in positive_words if word.lower() in text.lower())
        negative_count = sum(1 for word in negative_words if word.lower() in text.lower())
        
        # Balance entre positivo y negativo
        if positive_count > 0 and negative_count > 0:
            balance = min(positive_count, negative_count) / max(positive_count, negative_count)
            if balance > 0.5:  # Balanceado
                score += 0.2
        elif positive_count == 0 and negative_count == 0:
            score += 0.2  # Muy neutral
        
        # 4. Detectar expresiones de incertidumbre
        # IA a veces usa expresiones de incertidumbre
        uncertainty_words = [
            'perhaps', 'maybe', 'possibly', 'might', 'could', 'may',
            'uncertain', 'unclear', 'possibly', 'potentially'
        ]
        uncertainty_count = sum(1 for word in uncertainty_words if word.lower() in text.lower())
        if uncertainty_count > word_count * 0.02:  # Más del 2%
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_contextual_coherence(self, text: str) -> float:
        """Analiza coherencia contextual y temática avanzada - NUEVO"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 3:
            return 0.0
        
        score = 0.0
        
        # 1. Análisis de coherencia temática
        # Extraer palabras clave de cada oración
        all_words = []
        for sentence in sentences:
            words = [w.lower() for w in sentence.split() if len(w) > 4 and w.isalpha()]
            all_words.extend(words)
        
        # Contar frecuencia de palabras clave
        word_freq = {}
        for word in all_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Palabras que aparecen múltiples veces (temas centrales)
        key_themes = [w for w, count in word_freq.items() if count > 2]
        if len(key_themes) > 0:
            # Alta coherencia temática
            theme_coherence = len(key_themes) / len(set(all_words)) if len(set(all_words)) > 0 else 0
            if theme_coherence > 0.2:
                score += 0.3
        
        # 2. Análisis de progresión lógica
        # Detectar si el texto sigue una progresión lógica
        progression_markers = [
            r'\b(?:first|initially|to begin|starting with)\b',
            r'\b(?:second|next|then|following|subsequently)\b',
            r'\b(?:third|furthermore|additionally|moreover)\b',
            r'\b(?:finally|lastly|in conclusion|to conclude|ultimately)\b'
        ]
        
        progression_count = 0
        for i, sentence in enumerate(sentences):
            for pattern in progression_markers:
                if re.search(pattern, sentence, re.IGNORECASE):
                    progression_count += 1
                    break
        
        if progression_count >= 2:
            score += 0.25
        
        # 3. Análisis de referencias cruzadas
        # Detectar referencias a conceptos mencionados anteriormente
        reference_words = ['this', 'that', 'these', 'those', 'it', 'they', 'he', 'she']
        reference_count = sum(1 for word in reference_words if word.lower() in text.lower())
        
        if reference_count > len(sentences) * 0.3:  # Más del 30% de oraciones tienen referencias
            score += 0.2
        
        # 4. Análisis de coherencia semántica entre párrafos
        paragraphs = text.split('\n\n')
        if len(paragraphs) > 1:
            para_words = []
            for para in paragraphs:
                words = [w.lower() for w in para.split() if len(w) > 4 and w.isalpha()]
                para_words.append(set(words))
            
            # Calcular solapamiento entre párrafos
            overlaps = []
            for i in range(len(para_words) - 1):
                if para_words[i] and para_words[i+1]:
                    overlap = len(para_words[i] & para_words[i+1]) / len(para_words[i] | para_words[i+1])
                    overlaps.append(overlap)
            
            if overlaps:
                avg_overlap = np.mean(overlaps)
                if avg_overlap > 0.15:  # Solapamiento significativo
                    score += 0.25
        
        return min(score, 1.0)
    
    def _detect_machine_translation(self, text: str) -> float:
        """Detecta si el texto fue traducido automáticamente - NUEVO"""
        score = 0.0
        
        # 1. Patrones típicos de traducción automática
        translation_patterns = [
            r'\b(?:the the|a a|an an)\b',  # Repetición de artículos
            r'\b(?:is is|are are|was was|were were)\b',  # Repetición de verbos
            r'\b(?:of of|in in|on on|at at)\b',  # Repetición de preposiciones
            r'\b(?:and and|or or|but but)\b',  # Repetición de conjunciones
        ]
        
        translation_matches = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in translation_patterns)
        if translation_matches > 0:
            score += 0.4
        
        # 2. Orden de palabras inusual (típico de traducción)
        # Frases que suenan "traducidas"
        unusual_patterns = [
            r'\b(?:very much|so much|too much)\s+\w+',  # Construcciones traducidas
            r'\b(?:the same|the different|the new|the old)\s+\w+',  # Artículos innecesarios
        ]
        
        unusual_matches = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in unusual_patterns)
        if unusual_matches > 2:
            score += 0.3
        
        # 3. Uso excesivo de palabras de traducción literal
        literal_words = ['very', 'much', 'quite', 'rather', 'quite', 'indeed']
        literal_count = sum(1 for word in literal_words if word.lower() in text.lower())
        word_count = len(text.split())
        
        if word_count > 0:
            literal_ratio = literal_count / word_count
            if literal_ratio > 0.05:  # Más del 5%
                score += 0.2
        
        # 4. Falta de modismos o expresiones naturales
        # Texto traducido suele carecer de modismos
        idioms = [
            'break the ice', 'piece of cake', 'once in a blue moon',
            'the ball is in your court', 'barking up the wrong tree'
        ]
        idiom_count = sum(1 for idiom in idioms if idiom.lower() in text.lower())
        
        if idiom_count == 0 and word_count > 100:
            score += 0.1  # Texto largo sin modismos puede ser traducido
        
        return min(score, 1.0)
    
    def _analyze_generation_patterns(self, text: str) -> float:
        """Analiza patrones específicos de generación de IA - NUEVO"""
        score = 0.0
        
        # 1. Patrones de inicio típicos de IA
        start_patterns = [
            r'^(?:let me|i\'ll|i will|i can|i\'d like to|i would like to)',
            r'^(?:here\'s|here are|below is|following is|as follows)',
            r'^(?:to answer|to respond|to help|to assist)',
            r'^(?:based on|according to|in terms of|with regard to)',
        ]
        
        first_sentence = text.split('.')[0] if '.' in text else text[:100]
        start_matches = sum(1 for pattern in start_patterns if re.search(pattern, first_sentence, re.IGNORECASE))
        if start_matches > 0:
            score += 0.3
        
        # 2. Patrones de estructura repetitiva
        # IA suele usar estructuras similares
        structure_patterns = [
            r'\b(?:first|second|third|fourth|fifth)\b.*?\b(?:then|next|after|following)\b',
            r'\b(?:one|two|three)\s+(?:way|method|approach|solution)\b',
            r'\b(?:on one hand|on the other hand)\b',
            r'\b(?:in addition|furthermore|moreover|additionally)\b',
        ]
        
        structure_matches = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in structure_patterns)
        if structure_matches > 1:
            score += 0.25
        
        # 3. Uso de frases de transición excesivo
        transition_phrases = [
            'in conclusion', 'to summarize', 'in summary', 'to conclude',
            'in other words', 'that is to say', 'put another way',
            'for example', 'for instance', 'such as', 'namely'
        ]
        
        transition_count = sum(1 for phrase in transition_phrases if phrase.lower() in text.lower())
        word_count = len(text.split())
        
        if word_count > 0:
            transition_ratio = transition_count / (word_count / 100)  # Por cada 100 palabras
            if transition_ratio > 2:  # Más de 2 transiciones por 100 palabras
                score += 0.2
        
        # 4. Patrones de cierre típicos de IA
        end_patterns = [
            r'\b(?:i hope this helps|i hope this|hope this helps)\b',
            r'\b(?:let me know if|feel free to|if you have any)\b',
            r'\b(?:please let me know|don\'t hesitate to|if you need)\b',
            r'\b(?:in conclusion|to summarize|in summary|to conclude)\b',
        ]
        
        last_sentences = ' '.join(text.split('.')[-3:]) if '.' in text else text[-200:]
        end_matches = sum(1 for pattern in end_patterns if re.search(pattern, last_sentences, re.IGNORECASE))
        if end_matches > 0:
            score += 0.25
        
        return min(score, 1.0)
    
    def _analyze_writing_quality(self, text: str) -> float:
        """Analiza la calidad de escritura del texto - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) == 0 or len(sentences) == 0:
            return 0.0
        
        # 1. Análisis de errores gramaticales comunes
        # Texto de IA suele tener menos errores gramaticales
        common_errors = [
            'teh', 'adn', 'taht', 'recieve', 'seperate', 'occured',
            'definately', 'accomodate', 'begining', 'existance'
        ]
        error_count = sum(1 for error in common_errors if error in text.lower())
        
        if error_count == 0:
            score += 0.3  # Sin errores comunes
        elif error_count <= 2:
            score += 0.1  # Pocos errores
        
        # 2. Análisis de puntuación correcta
        # Texto de IA suele tener puntuación más consistente
        punctuation_errors = [
            r'\s+[.!?]',  # Espacios antes de puntuación
            r'[.!?]\s*[a-z]',  # Falta de mayúscula después de punto
            r',\s*,',  # Comas dobles
            r'\.\s*\.',  # Puntos dobles
        ]
        
        punct_errors = sum(len(re.findall(pattern, text)) for pattern in punctuation_errors)
        if punct_errors == 0:
            score += 0.2
        
        # 3. Análisis de vocabulario variado
        unique_words = len(set(word.lower() for word in words))
        vocab_ratio = unique_words / len(words) if len(words) > 0 else 0
        
        if 0.5 <= vocab_ratio <= 0.8:  # Balance ideal
            score += 0.2
        elif vocab_ratio > 0.8:
            score += 0.1  # Demasiada variedad puede ser sospechoso
        
        # 4. Análisis de longitud de palabras
        # Texto de IA puede tener palabras más largas en promedio
        avg_word_length = np.mean([len(word) for word in words if word.isalpha()])
        if 4.5 <= avg_word_length <= 6.5:
            score += 0.15
        
        # 5. Análisis de estructura de párrafos
        paragraphs = text.split('\n\n')
        if len(paragraphs) > 1:
            para_lengths = [len(p.split()) for p in paragraphs if p.strip()]
            if para_lengths:
                avg_para_length = np.mean(para_lengths)
                if 50 <= avg_para_length <= 200:  # Párrafos bien formados
                    score += 0.15
        
        return min(score, 1.0)
    
    def _detect_paraphrasing(self, text: str) -> float:
        """Detecta si el texto fue parafraseado - NUEVO"""
        score = 0.0
        
        # 1. Detectar sinónimos excesivos
        # Texto parafraseado suele usar muchos sinónimos
        synonym_pairs = [
            ('big', 'large'), ('small', 'little'), ('good', 'great'),
            ('bad', 'terrible'), ('important', 'significant'), ('help', 'assist'),
            ('use', 'utilize'), ('show', 'demonstrate'), ('tell', 'inform'),
            ('get', 'obtain'), ('make', 'create'), ('do', 'perform')
        ]
        
        synonym_count = 0
        for word1, word2 in synonym_pairs:
            if word1 in text.lower() and word2 in text.lower():
                synonym_count += 1
        
        if synonym_count > 3:
            score += 0.3
        
        # 2. Detectar cambios de estructura sin cambios de significado
        # Frases que dicen lo mismo de forma diferente
        similar_phrases = [
            (r'\b(?:in order to|to)\b', r'\b(?:so that|so as to)\b'),
            (r'\b(?:because|since)\b', r'\b(?:due to|owing to)\b'),
            (r'\b(?:although|though)\b', r'\b(?:even though|despite)\b'),
        ]
        
        phrase_variations = 0
        for pattern1, pattern2 in similar_phrases:
            if re.search(pattern1, text, re.IGNORECASE) and re.search(pattern2, text, re.IGNORECASE):
                phrase_variations += 1
        
        if phrase_variations > 1:
            score += 0.25
        
        # 3. Detectar uso de palabras más formales de lo necesario
        # Parafraseo suele usar vocabulario más formal
        formal_replacements = [
            ('use', 'utilize'), ('help', 'assist'), ('show', 'demonstrate'),
            ('tell', 'inform'), ('get', 'obtain'), ('make', 'create'),
            ('start', 'commence'), ('end', 'conclude'), ('try', 'attempt')
        ]
        
        formal_count = sum(1 for informal, formal in formal_replacements 
                          if formal in text.lower() and informal not in text.lower())
        
        if formal_count > 2:
            score += 0.25
        
        # 4. Detectar cambios de voz (activa a pasiva o viceversa)
        # Parafraseo a veces cambia la voz
        active_patterns = [
            r'\b(?:we|they|he|she|it)\s+\w+ed\b',
            r'\b(?:we|they|he|she|it)\s+\w+s\b'
        ]
        passive_patterns = [
            r'\b(?:is|are|was|were)\s+\w+ed\s+by\b',
            r'\b(?:is|are|was|were)\s+\w+en\s+by\b'
        ]
        
        active_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in active_patterns)
        passive_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in passive_patterns)
        
        if active_count > 0 and passive_count > 0:
            score += 0.2  # Mezcla de voces puede indicar parafraseo
        
        return min(score, 1.0)
    
    def _analyze_risk_and_reliability(self, text: str, detected_models: List[Dict], scores: List[float]) -> float:
        """Analiza el riesgo y confiabilidad de la detección - NUEVO"""
        score = 0.0
        
        # 1. Análisis de confianza basado en métodos de detección
        if len(scores) > 5:  # Múltiples métodos coinciden
            score += 0.2
        elif len(scores) > 3:
            score += 0.1
        
        # 2. Análisis de consistencia entre métodos
        if scores:
            score_std = np.std(scores)
            if score_std < 0.2:  # Baja desviación = alta consistencia
                score += 0.3
            elif score_std < 0.3:
                score += 0.15
        
        # 3. Análisis de modelos detectados
        if len(detected_models) > 1:  # Múltiples modelos detectados
            score += 0.2
        elif len(detected_models) == 1:
            model_confidence = detected_models[0].get("confidence", 0)
            if model_confidence > 0.8:
                score += 0.3
            elif model_confidence > 0.6:
                score += 0.15
        
        # 4. Análisis de longitud del texto
        # Textos más largos suelen dar resultados más confiables
        word_count = len(text.split())
        if word_count > 500:
            score += 0.15
        elif word_count > 200:
            score += 0.1
        elif word_count < 50:
            score -= 0.1  # Textos muy cortos son menos confiables
        
        # 5. Análisis de calidad del texto
        # Texto de baja calidad puede dar falsos positivos
        quality_indicators = [
            len(re.findall(r'\b(?:the|a|an|and|or|but|in|on|at|to|for|of|with|by)\b', text.lower())),
            len(re.findall(r'[.!?]', text)),
            len(re.findall(r'[A-Z][a-z]+', text))
        ]
        
        if all(ind > 0 for ind in quality_indicators):
            score += 0.1  # Texto tiene estructura básica
        
        return max(min(score, 1.0), 0.0)
    
    def _analyze_metadata_and_context(self, text: str, metadata: Dict[str, Any]) -> float:
        """Analiza metadatos y contexto para detectar IA - NUEVO"""
        score = 0.0
        
        # 1. Análisis de fuente
        source = metadata.get("source", "").lower()
        ai_sources = ["chatgpt", "gpt", "claude", "gemini", "ai", "openai", "anthropic"]
        if any(ai_source in source for ai_source in ai_sources):
            score += 0.4
        
        # 2. Análisis de timestamp
        # Contenido generado recientemente puede ser más probable que sea IA
        timestamp = metadata.get("timestamp")
        if timestamp:
            import datetime
            try:
                if isinstance(timestamp, (int, float)):
                    content_time = datetime.datetime.fromtimestamp(timestamp)
                else:
                    content_time = datetime.datetime.fromisoformat(str(timestamp))
                
                now = datetime.datetime.now()
                age_hours = (now - content_time).total_seconds() / 3600
                
                # Contenido muy reciente puede ser más sospechoso
                if age_hours < 24:  # Menos de 24 horas
                    score += 0.1
            except:
                pass
        
        # 3. Análisis de user agent o aplicación
        user_agent = metadata.get("user_agent", "").lower()
        app_name = metadata.get("app_name", "").lower()
        
        ai_apps = ["chatgpt", "claude", "bard", "copilot", "ai", "assistant"]
        if any(ai_app in user_agent or ai_app in app_name for ai_app in ai_apps):
            score += 0.3
        
        # 4. Análisis de idioma en metadatos vs texto
        metadata_lang = metadata.get("language", "").lower()
        # Si hay discrepancia puede indicar traducción automática
        if metadata_lang and metadata_lang not in ["en", "english"]:
            # Texto en inglés pero metadata en otro idioma puede indicar traducción
            score += 0.1
        
        # 5. Análisis de referrer o origen
        referrer = metadata.get("referrer", "").lower()
        origin = metadata.get("origin", "").lower()
        
        ai_origins = ["openai", "anthropic", "google ai", "chatgpt", "claude"]
        if any(ai_origin in referrer or ai_origin in origin for ai_origin in ai_origins):
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_language_patterns(self, text: str) -> float:
        """Analiza patrones de idioma y localización - NUEVO"""
        score = 0.0
        
        # 1. Detectar mezcla de idiomas (típico de traducción automática)
        # Patrones de palabras en diferentes idiomas
        english_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with']
        spanish_words = ['el', 'la', 'los', 'las', 'de', 'en', 'y', 'o', 'pero', 'con', 'por']
        french_words = ['le', 'la', 'les', 'de', 'en', 'et', 'ou', 'mais', 'avec', 'pour']
        
        english_count = sum(1 for word in english_words if word.lower() in text.lower())
        spanish_count = sum(1 for word in spanish_words if word.lower() in text.lower())
        french_count = sum(1 for word in french_words if word.lower() in text.lower())
        
        # Mezcla de idiomas puede indicar traducción automática
        languages_detected = sum([
            english_count > 5,
            spanish_count > 5,
            french_count > 5
        ])
        
        if languages_detected > 1:
            score += 0.3
        
        # 2. Análisis de caracteres especiales por idioma
        # Texto de IA traducido puede tener caracteres especiales mal usados
        special_chars = ['á', 'é', 'í', 'ó', 'ú', 'ñ', 'ü', 'ç', 'à', 'è', 'ì', 'ò', 'ù']
        special_count = sum(1 for char in special_chars if char in text.lower())
        
        # Muchos caracteres especiales pueden indicar traducción
        if special_count > len(text) * 0.05:  # Más del 5%
            score += 0.2
        
        # 3. Análisis de orden de palabras (SVO vs otros órdenes)
        # Texto traducido puede tener orden de palabras inusual
        # Detectar patrones de orden inusual
        unusual_patterns = [
            r'\b(?:the|a|an)\s+\w+\s+(?:the|a|an)\b',  # Artículos duplicados
            r'\b\w+\s+is\s+is\b',  # "is is"
            r'\b\w+\s+the\s+the\b',  # "the the"
        ]
        
        unusual_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in unusual_patterns)
        if unusual_count > 0:
            score += 0.3
        
        # 4. Análisis de expresiones idiomáticas
        # Texto traducido suele carecer de expresiones idiomáticas naturales
        idioms = [
            'break the ice', 'piece of cake', 'once in a blue moon',
            'the ball is in your court', 'barking up the wrong tree',
            'cost an arm and a leg', 'hit the nail on the head'
        ]
        
        idiom_count = sum(1 for idiom in idioms if idiom.lower() in text.lower())
        word_count = len(text.split())
        
        # Texto largo sin modismos puede ser traducido
        if idiom_count == 0 and word_count > 200:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_semantic_similarity(self, text: str) -> float:
        """Analiza similitud semántica usando técnicas básicas - NUEVO"""
        score = 0.0
        
        # Dividir texto en oraciones
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.split()) > 3]
        
        if len(sentences) < 2:
            return 0.0
        
        # 1. Análisis de similitud entre oraciones (Jaccard)
        similarities = []
        for i in range(len(sentences) - 1):
            words1 = set(re.findall(r'\b\w+\b', sentences[i].lower()))
            words2 = set(re.findall(r'\b\w+\b', sentences[i+1].lower()))
            
            if len(words1) > 0 and len(words2) > 0:
                intersection = len(words1.intersection(words2))
                union = len(words1.union(words2))
                similarity = intersection / union if union > 0 else 0
                similarities.append(similarity)
        
        if similarities:
            avg_similarity = np.mean(similarities)
            # Alta similitud entre oraciones puede indicar IA (texto muy coherente)
            if avg_similarity > 0.3:
                score += 0.3
        
        # 2. Análisis de repetición de conceptos clave
        words = re.findall(r'\b\w+\b', text.lower())
        word_freq = {}
        for word in words:
            if len(word) > 4:  # Solo palabras significativas
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Conceptos muy repetidos pueden indicar IA
        if word_freq:
            max_freq = max(word_freq.values())
            total_words = len(words)
            if max_freq / total_words > 0.05:  # Más del 5% de repetición
                score += 0.2
        
        # 3. Análisis de coherencia temática (palabras relacionadas)
        # Detectar si hay un tema dominante
        significant_words = [w for w, f in word_freq.items() if f > 2 and len(w) > 4]
        if len(significant_words) < len(word_freq) * 0.3:  # Pocas palabras dominantes
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_keyword_frequency(self, text: str) -> float:
        """Analiza frecuencia de palabras clave típicas de IA - NUEVO"""
        score = 0.0
        text_lower = text.lower()
        
        # Palabras clave típicas de respuestas de IA
        ai_keywords = [
            'however', 'furthermore', 'moreover', 'additionally', 'consequently',
            'therefore', 'thus', 'hence', 'accordingly', 'nevertheless',
            'in conclusion', 'to summarize', 'in summary', 'overall',
            'it is important to note', 'it should be noted', 'it is worth mentioning',
            'as a result', 'in other words', 'for instance', 'for example',
            'specifically', 'particularly', 'especially', 'notably'
        ]
        
        # Contar ocurrencias de palabras clave
        keyword_count = sum(1 for keyword in ai_keywords if keyword in text_lower)
        word_count = len(text.split())
        
        # Ratio de palabras clave
        if word_count > 0:
            keyword_ratio = keyword_count / (word_count / 100)  # Por cada 100 palabras
            
            # Alto ratio indica posible IA
            if keyword_ratio > 2:  # Más de 2 palabras clave por cada 100 palabras
                score += 0.4
            elif keyword_ratio > 1:
                score += 0.2
        
        # Frases típicas de IA
        ai_phrases = [
            r'\bit is (?:important|worth noting|essential|crucial)',
            r'\b(?:in|to) (?:conclusion|summary|conclude)',
            r'\b(?:as|it) (?:can|may) be (?:seen|observed|noted)',
            r'\b(?:this|these) (?:suggests?|indicates?|demonstrates?)',
            r'\b(?:it|this) (?:is|should be) (?:noted|mentioned|understood)',
            r'\b(?:in|for) (?:order|addition) to',
            r'\b(?:on|with) (?:regard|respect) to'
        ]
        
        phrase_count = sum(len(re.findall(phrase, text_lower)) for phrase in ai_phrases)
        if phrase_count > 2:
            score += 0.3
        elif phrase_count > 0:
            score += 0.15
        
        return min(score, 1.0)
    
    def _detect_response_patterns(self, text: str) -> float:
        """Detecta patrones típicos de respuestas de IA - NUEVO"""
        score = 0.0
        text_lower = text.lower()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # 1. Patrones de inicio típicos de IA
        ai_start_patterns = [
            r'^(?:i|we|this|these|it|they) (?:would|will|can|may|might)',
            r'^(?:to|in order to)',
            r'^(?:based on|according to|in accordance with)',
            r'^(?:it is|this is|these are)',
            r'^(?:let me|allow me)'
        ]
        
        start_matches = sum(1 for pattern in ai_start_patterns 
                          if re.search(pattern, sentences[0].lower() if sentences else "", re.IGNORECASE))
        if start_matches > 0:
            score += 0.2
        
        # 2. Estructura de respuesta típica (introducción, desarrollo, conclusión)
        # Detectar si tiene estructura muy organizada
        has_intro = any(word in text_lower[:200] for word in ['introduction', 'overview', 'summary', 'begin', 'start'])
        has_conclusion = any(word in text_lower[-200:] for word in ['conclusion', 'summary', 'conclude', 'finally', 'overall'])
        
        if has_intro and has_conclusion:
            score += 0.3
        
        # 3. Uso excesivo de conectores lógicos
        logical_connectors = ['however', 'therefore', 'thus', 'hence', 'consequently', 
                            'furthermore', 'moreover', 'additionally', 'nevertheless']
        connector_count = sum(1 for connector in logical_connectors if connector in text_lower)
        
        if connector_count > len(sentences) * 0.2:  # Más del 20% de las oraciones
            score += 0.3
        
        # 4. Formato de lista numerada o con viñetas (típico de IA)
        list_patterns = [
            r'\d+\.\s+[A-Z]',  # Numeración
            r'[•\-\*]\s+[A-Z]',  # Viñetas
            r'(?:first|second|third|fourth|fifth|finally|lastly)',
        ]
        
        list_count = sum(len(re.findall(pattern, text)) for pattern in list_patterns)
        if list_count > 2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_narrative_coherence(self, text: str) -> float:
        """Analiza coherencia narrativa del texto - NUEVO"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.split()) > 3]
        
        if len(sentences) < 3:
            return 0.0
        
        # 1. Análisis de referencias pronominales
        # Texto de IA suele tener referencias pronominales muy claras
        pronouns = ['it', 'this', 'that', 'these', 'those', 'he', 'she', 'they', 'we', 'you']
        pronoun_count = sum(len(re.findall(rf'\b{pronoun}\b', text.lower())) for pronoun in pronouns)
        word_count = len(text.split())
        
        if word_count > 0:
            pronoun_ratio = pronoun_count / word_count
            # Ratio muy alto o muy bajo puede indicar IA
            if pronoun_ratio > 0.08 or pronoun_ratio < 0.02:
                score += 0.2
        
        # 2. Análisis de progresión temática
        # Detectar si el tema se mantiene constante
        first_half_words = set(re.findall(r'\b\w{4,}\b', ' '.join(sentences[:len(sentences)//2]).lower()))
        second_half_words = set(re.findall(r'\b\w{4,}\b', ' '.join(sentences[len(sentences)//2:]).lower()))
        
        if len(first_half_words) > 0 and len(second_half_words) > 0:
            overlap = len(first_half_words.intersection(second_half_words))
            total_unique = len(first_half_words.union(second_half_words))
            
            if total_unique > 0:
                overlap_ratio = overlap / total_unique
                # Alta superposición temática puede indicar IA
                if overlap_ratio > 0.4:
                    score += 0.3
        
        # 3. Análisis de variación en longitud de oraciones
        sentence_lengths = [len(s.split()) for s in sentences]
        if len(sentence_lengths) > 1:
            length_variance = np.var(sentence_lengths)
            avg_length = np.mean(sentence_lengths)
            
            # Baja variación puede indicar IA (oraciones muy uniformes)
            if avg_length > 0:
                cv = np.sqrt(length_variance) / avg_length  # Coeficiente de variación
                if cv < 0.3:  # Baja variación
                    score += 0.2
        
        # 4. Análisis de transiciones entre párrafos
        # Texto de IA suele tener transiciones muy explícitas
        transition_words = ['however', 'therefore', 'furthermore', 'moreover', 
                           'additionally', 'consequently', 'meanwhile', 'subsequently']
        transition_count = sum(1 for word in transition_words if word in text.lower())
        
        if len(sentences) > 0:
            transition_ratio = transition_count / len(sentences)
            # Ratio alto de transiciones puede indicar IA
            if transition_ratio > 0.15:
                score += 0.3
        
        return min(score, 1.0)
    
    def _apply_adaptive_weights(self, method_weights: Dict[str, float], 
                                detection_methods: List[str], text: str) -> Dict[str, float]:
        """Aplica pesos adaptativos basados en el historial y contexto - NUEVO"""
        # Si no hay historial suficiente, usar pesos por defecto
        if len(self.detection_history) < 10:
            return method_weights
        
        # Calcular rendimiento de cada método basado en historial
        if not self.model_performance:
            # Inicializar rendimiento
            for method in detection_methods:
                self.model_performance[method] = {
                    "success_count": 0,
                    "total_count": 0,
                    "avg_score": 0.5
                }
        
        # Ajustar pesos basado en longitud del texto
        text_length = len(text.split())
        if text_length < 50:
            # Textos cortos: más peso a pattern matching
            if "pattern_matching" in method_weights:
                method_weights["pattern_matching"] *= 1.2
        elif text_length > 500:
            # Textos largos: más peso a análisis estructural
            if "structure_analysis" in method_weights:
                method_weights["structure_analysis"] *= 1.15
            if "semantic_coherence" in method_weights:
                method_weights["semantic_coherence"] *= 1.1
        
        # Ajustar pesos basado en modelos detectados
        if detection_methods and "pattern_matching" in detection_methods:
            # Si hay pattern matching, aumentar su peso ligeramente
            if "pattern_matching" in method_weights:
                method_weights["pattern_matching"] *= 1.1
        
        # Normalizar para mantener suma <= 1.0
        total = sum(method_weights.values())
        if total > 1.0:
            method_weights = {k: v / total for k, v in method_weights.items()}
        
        return method_weights
    
    def _analyze_historical_context(self, text: str, detected_models: List[Dict]) -> float:
        """Analiza contexto histórico comparando con detecciones anteriores - NUEVO"""
        score = 0.0
        
        if len(self.detection_history) < 5:
            return 0.0
        
        # 1. Comparar con detecciones recientes similares
        recent_detections = self.detection_history[-10:]
        
        # Si hay modelos detectados, verificar si son consistentes con el historial
        if detected_models:
            model_names = [m["model_name"] for m in detected_models]
            recent_models = [entry.get("primary_model") for entry in recent_detections 
                           if entry.get("primary_model")]
            
            # Si el modelo detectado aparece frecuentemente en el historial
            if recent_models:
                model_frequency = sum(1 for m in recent_models if m in model_names)
                if model_frequency > len(recent_models) * 0.3:  # Más del 30%
                    score += 0.3
        
        # 2. Análisis de tendencias
        # Si hay muchas detecciones de IA recientes, puede indicar patrón
        recent_ai_count = sum(1 for entry in recent_detections 
                             if entry.get("is_ai_generated", False))
        recent_ai_ratio = recent_ai_count / len(recent_detections) if recent_detections else 0
        
        # Si hay un patrón consistente de detecciones de IA
        if recent_ai_ratio > 0.7:  # Más del 70% son IA
            score += 0.2
        elif recent_ai_ratio < 0.3:  # Menos del 30% son IA
            score -= 0.1  # Penalizar si el patrón es inconsistente
        
        # 3. Análisis de confianza histórica
        recent_confidences = [entry.get("confidence_score", 0.0) 
                            for entry in recent_detections]
        if recent_confidences:
            avg_confidence = np.mean(recent_confidences)
            # Si la confianza promedio es alta, puede indicar patrón
            if avg_confidence > 0.7:
                score += 0.2
        
        return max(min(score, 1.0), 0.0)
    
    def _analyze_advanced_ngrams(self, text: str) -> float:
        """Análisis avanzado de n-gramas (trigramas, 4-gramas) - NUEVO"""
        score = 0.0
        words = text.lower().split()
        
        if len(words) < 10:
            return 0.0
        
        # 1. Análisis de trigramas
        trigrams = {}
        for i in range(len(words) - 2):
            trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
            trigrams[trigram] = trigrams.get(trigram, 0) + 1
        
        if trigrams:
            # Calcular diversidad de trigramas
            trigram_diversity = len(trigrams) / len(words) if len(words) > 0 else 0
            
            # Baja diversidad de trigramas puede indicar IA
            if trigram_diversity < 0.2:
                score += 0.3
            elif trigram_diversity < 0.3:
                score += 0.15
            
            # Trigramas muy repetidos
            max_trigram_freq = max(trigrams.values())
            if max_trigram_freq > 3:  # Aparece más de 3 veces
                score += 0.2
        
        # 2. Análisis de 4-gramas (frases comunes de IA)
        fourgrams = {}
        for i in range(len(words) - 3):
            fourgram = f"{words[i]} {words[i+1]} {words[i+2]} {words[i+3]}"
            fourgrams[fourgram] = fourgrams.get(fourgram, 0) + 1
        
        # Frases típicas de IA (4-gramas comunes)
        ai_fourgrams = [
            'it is important to', 'it should be noted', 'as a result of',
            'in order to be', 'it is worth noting', 'in the case of',
            'on the other hand', 'in addition to the', 'as well as the'
        ]
        
        ai_fourgram_count = sum(1 for fg in ai_fourgrams if fg in ' '.join(words))
        if ai_fourgram_count > 0:
            score += min(ai_fourgram_count * 0.15, 0.3)
        
        # 3. Análisis de secuencias repetitivas
        # Detectar patrones como "A, B, and C" repetidos
        repetitive_patterns = [
            r'\w+,\s+\w+,\s+and\s+\w+',  # A, B, and C
            r'\w+\s+and\s+\w+\s+and\s+\w+',  # A and B and C
        ]
        
        repetitive_count = sum(len(re.findall(pattern, text.lower())) 
                              for pattern in repetitive_patterns)
        if repetitive_count > 2:
            score += 0.2
        
        # 4. Análisis de distribución de n-gramas
        # Texto de IA suele tener distribución más uniforme
        if trigrams:
            trigram_freqs = list(trigrams.values())
            if len(trigram_freqs) > 1:
                freq_variance = np.var(trigram_freqs)
                avg_freq = np.mean(trigram_freqs)
                
                # Baja varianza indica distribución uniforme (típico de IA)
                if avg_freq > 0:
                    cv = np.sqrt(freq_variance) / avg_freq
                    if cv < 0.5:  # Baja variación
                        score += 0.15
        
        return min(score, 1.0)
    
    def _analyze_comparative_similarity(self, text: str) -> float:
        """Analiza similitud comparativa con textos conocidos - NUEVO"""
        score = 0.0
        
        if not self.known_ai_texts and not self.known_human_texts:
            return 0.0
        
        text_words = set(re.findall(r'\b\w+\b', text.lower()))
        
        # Comparar con textos conocidos de IA
        if self.known_ai_texts:
            ai_similarities = []
            for known_text in self.known_ai_texts[:10]:  # Limitar a 10 para rendimiento
                known_words = set(re.findall(r'\b\w+\b', known_text.lower()))
                
                if len(text_words) > 0 and len(known_words) > 0:
                    intersection = len(text_words.intersection(known_words))
                    union = len(text_words.union(known_words))
                    similarity = intersection / union if union > 0 else 0
                    ai_similarities.append(similarity)
            
            if ai_similarities:
                max_ai_similarity = max(ai_similarities)
                avg_ai_similarity = np.mean(ai_similarities)
                
                # Alta similitud con textos de IA conocidos
                if max_ai_similarity > 0.4:
                    score += 0.4
                elif avg_ai_similarity > 0.3:
                    score += 0.2
        
        # Comparar con textos conocidos humanos
        if self.known_human_texts:
            human_similarities = []
            for known_text in self.known_human_texts[:10]:  # Limitar a 10
                known_words = set(re.findall(r'\b\w+\b', known_text.lower()))
                
                if len(text_words) > 0 and len(known_words) > 0:
                    intersection = len(text_words.intersection(known_words))
                    union = len(text_words.union(known_words))
                    similarity = intersection / union if union > 0 else 0
                    human_similarities.append(similarity)
            
            if human_similarities:
                max_human_similarity = max(human_similarities)
                avg_human_similarity = np.mean(human_similarities)
                
                # Alta similitud con textos humanos conocidos
                if max_human_similarity > 0.4:
                    score -= 0.2  # Penalizar si es similar a texto humano
                elif avg_human_similarity > 0.3:
                    score -= 0.1
        
        # Análisis de patrones comunes
        # Si el texto tiene patrones similares a textos de IA conocidos
        if self.known_ai_texts:
            # Analizar longitud promedio
            known_lengths = [len(t.split()) for t in self.known_ai_texts[:10]]
            text_length = len(text.split())
            
            if known_lengths:
                avg_known_length = np.mean(known_lengths)
                if abs(text_length - avg_known_length) < avg_known_length * 0.3:
                    score += 0.1
        
        return max(min(score, 1.0), 0.0)
    
    def _analyze_with_ml_patterns(self, text: str, detected_models: List[Dict], 
                                  scores: List[float]) -> float:
        """Análisis con patrones de machine learning básico - NUEVO"""
        score = 0.0
        
        # 1. Análisis de combinación de características
        # Si múltiples métodos detectan IA, aumenta la confianza
        if len(scores) > 5:
            high_scores = [s for s in scores if s > 0.6]
            if len(high_scores) > 3:  # Más de 3 métodos con score alto
                score += 0.3
        
        # 2. Análisis de consistencia entre métodos
        if len(scores) > 1:
            score_variance = np.var(scores)
            avg_score = np.mean(scores)
            
            # Baja varianza indica consistencia (típico de IA)
            if avg_score > 0:
                cv = np.sqrt(score_variance) / avg_score
                if cv < 0.3:  # Baja variación
                    score += 0.2
        
        # 3. Análisis de modelos detectados
        if detected_models:
            # Si hay modelos detectados con alta confianza
            high_confidence_models = [m for m in detected_models if m.get("confidence", 0) > 0.7]
            if len(high_confidence_models) > 0:
                score += 0.2
            
            # Si múltiples modelos detectados (puede indicar texto mixto o parafraseo)
            if len(detected_models) > 1:
                score += 0.1
        
        # 4. Análisis de longitud y complejidad
        word_count = len(text.split())
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Textos de IA suelen tener características específicas según longitud
        if word_count > 500:
            # Textos largos de IA: alta coherencia, estructura clara
            if len(sentences) > 0:
                avg_sentence_length = np.mean([len(s.split()) for s in sentences])
                if 12 <= avg_sentence_length <= 22:
                    score += 0.1
        elif word_count < 100:
            # Textos cortos de IA: muy estructurados, sin errores
            if len(sentences) > 0:
                avg_sentence_length = np.mean([len(s.split()) for s in sentences])
                if 8 <= avg_sentence_length <= 18:
                    score += 0.1
        
        # 5. Análisis de características combinadas
        # Combinar múltiples señales para decisión final
        feature_count = 0
        
        # Señal 1: Múltiples métodos activos
        if len(scores) > 8:
            feature_count += 1
        
        # Señal 2: Modelos detectados
        if detected_models:
            feature_count += 1
        
        # Señal 3: Scores consistentes
        if len(scores) > 1 and np.std(scores) < 0.2:
            feature_count += 1
        
        # Señal 4: Texto estructurado
        if re.search(r'\b(first|second|third|finally|in conclusion)', text, re.IGNORECASE):
            feature_count += 1
        
        # Si hay múltiples señales, aumenta score
        if feature_count >= 3:
            score += 0.2
        elif feature_count >= 2:
            score += 0.1
        
        return min(score, 1.0)
    
    def add_known_text(self, text: str, is_ai: bool):
        """Añade un texto conocido al sistema de aprendizaje - NUEVO"""
        if is_ai:
            if len(self.known_ai_texts) >= self.max_known_texts:
                self.known_ai_texts.pop(0)  # FIFO
            self.known_ai_texts.append(text)
        else:
            if len(self.known_human_texts) >= self.max_known_texts:
                self.known_human_texts.pop(0)  # FIFO
            self.known_human_texts.append(text)
        
        logger.info(f"Texto conocido añadido: {'IA' if is_ai else 'Humano'}")
    
    def _analyze_model_signatures(self, text: str, detected_models: List[Dict]) -> float:
        """Analiza firmas características específicas de cada modelo - NUEVO"""
        score = 0.0
        
        if not detected_models:
            return 0.0
        
        # Firmas específicas por modelo
        model_signatures = {
            "gpt-4": {
                "phrases": ["it's important to note", "it's worth mentioning", "it should be emphasized"],
                "structure": "highly_structured",
                "formality": "very_formal"
            },
            "gpt-3.5": {
                "phrases": ["let me", "i'd like to", "i think", "in my opinion"],
                "structure": "moderately_structured",
                "formality": "moderate"
            },
            "claude": {
                "phrases": ["i understand", "to clarify", "it's worth considering", "let's explore"],
                "structure": "very_structured",
                "formality": "very_formal"
            },
            "gemini": {
                "phrases": ["here's", "let's", "i'll", "you can"],
                "structure": "moderately_structured",
                "formality": "moderate"
            }
        }
        
        text_lower = text.lower()
        primary_model = max(detected_models, key=lambda x: x["confidence"]) if detected_models else None
        
        if primary_model:
            model_name = primary_model.get("model_name", "").lower()
            
            # Verificar firmas del modelo detectado
            if model_name in model_signatures:
                signatures = model_signatures[model_name]
                
                # Verificar frases características
                phrase_matches = sum(1 for phrase in signatures["phrases"] if phrase in text_lower)
                if phrase_matches > 0:
                    score += min(phrase_matches * 0.15, 0.4)
                
                # Verificar estructura
                if signatures["structure"] == "highly_structured":
                    if re.search(r'\b(first|second|third|finally|in conclusion)', text_lower):
                        score += 0.2
                elif signatures["structure"] == "very_structured":
                    if re.search(r'\b(?:introduction|overview|summary|conclusion)', text_lower):
                        score += 0.2
                
                # Verificar formalidad
                if signatures["formality"] == "very_formal":
                    formal_words = ['therefore', 'however', 'furthermore', 'moreover', 'consequently']
                    formal_count = sum(1 for word in formal_words if word in text_lower)
                    if formal_count > 2:
                        score += 0.2
        
        # Análisis de combinación de características del modelo
        # Si múltiples características coinciden, aumenta confianza
        if score > 0.3:
            score += 0.1  # Bonus por múltiples coincidencias
        
        return min(score, 1.0)
    
    def _analyze_semantic_embeddings(self, text: str) -> float:
        """Análisis básico de embeddings semánticos sin modelos externos - NUEVO"""
        score = 0.0
        words = text.lower().split()
        
        if len(words) < 10:
            return 0.0
        
        # 1. Análisis de clusters semánticos básico
        # Agrupar palabras por similitud de longitud y frecuencia
        word_lengths = [len(w) for w in words if len(w) > 3]
        if len(word_lengths) > 0:
            avg_length = np.mean(word_lengths)
            length_std = np.std(word_lengths)
            
            # Texto de IA suele tener distribución más uniforme de longitudes
            if length_std < 2.0:  # Baja desviación
                score += 0.2
        
        # 2. Análisis de co-ocurrencia de palabras
        # Palabras que aparecen juntas frecuentemente
        word_pairs = {}
        for i in range(len(words) - 1):
            if len(words[i]) > 3 and len(words[i+1]) > 3:
                pair = f"{words[i]}_{words[i+1]}"
                word_pairs[pair] = word_pairs.get(pair, 0) + 1
        
        # Muchas co-ocurrencias repetidas pueden indicar IA
        if word_pairs:
            max_cooccurrence = max(word_pairs.values())
            if max_cooccurrence > 2:  # Aparece más de 2 veces
                score += 0.2
        
        # 3. Análisis de densidad semántica
        # Palabras significativas vs palabras funcionales
        significant_words = [w for w in words if len(w) > 4]
        functional_words = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with']
        functional_count = sum(1 for w in words if w in functional_words)
        
        if len(words) > 0:
            significant_ratio = len(significant_words) / len(words)
            functional_ratio = functional_count / len(words)
            
            # Texto de IA suele tener balance específico
            if 0.4 <= significant_ratio <= 0.6 and 0.3 <= functional_ratio <= 0.5:
                score += 0.2
        
        # 4. Análisis de distribución de palabras
        # Texto de IA tiene distribución más predecible
        word_freq = {}
        for word in words:
            if len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        if word_freq:
            freqs = list(word_freq.values())
            if len(freqs) > 1:
                freq_variance = np.var(freqs)
                avg_freq = np.mean(freqs)
                
                # Baja varianza relativa indica distribución uniforme
                if avg_freq > 0:
                    cv = np.sqrt(freq_variance) / avg_freq
                    if cv < 0.5:  # Baja variación
                        score += 0.2
        
        # 5. Análisis de contexto semántico
        # Palabras relacionadas semánticamente (sinónimos, antónimos básicos)
        semantic_groups = [
            ['good', 'great', 'excellent', 'wonderful', 'fantastic'],
            ['important', 'significant', 'crucial', 'essential', 'vital'],
            ['problem', 'issue', 'challenge', 'difficulty', 'obstacle'],
            ['solution', 'answer', 'resolution', 'fix', 'remedy']
        ]
        
        semantic_matches = 0
        for group in semantic_groups:
            group_matches = sum(1 for word in group if word in words)
            if group_matches > 1:  # Múltiples palabras del mismo grupo
                semantic_matches += 1
        
        if semantic_matches > 1:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_temporal_patterns(self, text: str, metadata: Optional[Dict]) -> float:
        """Analiza patrones temporales de generación - NUEVO"""
        score = 0.0
        
        # 1. Análisis de timestamp en metadatos
        if metadata and metadata.get("timestamp"):
            import datetime
            try:
                if isinstance(metadata["timestamp"], (int, float)):
                    content_time = datetime.datetime.fromtimestamp(metadata["timestamp"])
                else:
                    content_time = datetime.datetime.fromisoformat(str(metadata["timestamp"]))
                
                now = datetime.datetime.now()
                age_hours = (now - content_time).total_seconds() / 3600
                
                # Contenido muy reciente puede ser más sospechoso
                if age_hours < 1:  # Menos de 1 hora
                    score += 0.3
                elif age_hours < 24:  # Menos de 24 horas
                    score += 0.1
            except:
                pass
        
        # 2. Análisis de patrones de tiempo en el texto
        # Referencias temporales específicas pueden indicar generación reciente
        time_patterns = [
            r'\b(?:today|yesterday|recently|lately|now|currently)\b',
            r'\b(?:in \d{4}|this year|this month|this week)\b',
            r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b'
        ]
        
        time_matches = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in time_patterns)
        if time_matches > 0:
            score += min(time_matches * 0.1, 0.2)
        
        # 3. Análisis de referencias a eventos recientes
        # Texto de IA puede tener referencias a eventos muy recientes
        recent_event_patterns = [
            r'\b(?:latest|newest|most recent|current)\b',
            r'\b(?:as of|up to date|updated)\b'
        ]
        
        recent_matches = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in recent_event_patterns)
        if recent_matches > 1:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_hybrid_models(self, text: str, detected_models: List[Dict]) -> float:
        """Detecta uso de modelos híbridos o combinados - NUEVO"""
        score = 0.0
        
        if len(detected_models) < 2:
            return 0.0
        
        # 1. Múltiples modelos con confianza similar
        confidences = [m.get("confidence", 0) for m in detected_models]
        if len(confidences) > 1:
            conf_std = np.std(confidences)
            conf_mean = np.mean(confidences)
            
            # Si las confianzas son similares, puede indicar uso híbrido
            if conf_mean > 0 and conf_std / conf_mean < 0.3:  # Baja variación relativa
                score += 0.3
        
        # 2. Modelos de diferentes proveedores
        providers = [m.get("provider") for m in detected_models if m.get("provider")]
        unique_providers = len(set(providers))
        
        if unique_providers > 1:
            score += 0.3  # Múltiples proveedores = posible uso híbrido
        
        # 3. Patrones de diferentes modelos en diferentes partes del texto
        # Dividir texto en partes y analizar cada una
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) > 5:
            # Analizar primera mitad vs segunda mitad
            first_half = ' '.join(sentences[:len(sentences)//2])
            second_half = ' '.join(sentences[len(sentences)//2:])
            
            # Detectar modelos en cada mitad
            first_models = self._detect_model_patterns(first_half)
            second_models = self._detect_model_patterns(second_half)
            
            if first_models and second_models:
                first_model_names = {m.get("model_name") for m in first_models}
                second_model_names = {m.get("model_name") for m in second_models}
                
                # Si hay modelos diferentes en cada mitad
                if not first_model_names.intersection(second_model_names):
                    score += 0.4  # Modelos completamente diferentes = híbrido
        
        # 4. Análisis de estilo mixto
        # Texto que combina características de diferentes modelos
        text_lower = text.lower()
        
        # Características de GPT
        gpt_features = sum(1 for phrase in ["it's important to note", "it's worth mentioning"] if phrase in text_lower)
        # Características de Claude
        claude_features = sum(1 for phrase in ["i understand", "to clarify", "let's explore"] if phrase in text_lower)
        # Características de Gemini
        gemini_features = sum(1 for phrase in ["here's", "let's", "you can"] if phrase in text_lower)
        
        # Si hay características de múltiples modelos
        model_features = sum([gpt_features > 0, claude_features > 0, gemini_features > 0])
        if model_features > 1:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_advanced_frequency(self, text: str) -> float:
        """Análisis avanzado de frecuencia de palabras y patrones - NUEVO"""
        score = 0.0
        words = text.lower().split()
        
        if len(words) < 20:
            return 0.0
        
        # 1. Análisis de distribución de Zipf
        # Texto natural sigue distribución de Zipf aproximadamente
        word_freq = {}
        for word in words:
            word_clean = word.strip('.,!?;:()[]{}"\'')
            if word_clean and len(word_clean) > 2:
                word_freq[word_clean] = word_freq.get(word_clean, 0) + 1
        
        if word_freq:
            freqs = sorted(word_freq.values(), reverse=True)
            
            # Calcular si sigue distribución de Zipf
            # En Zipf, la frecuencia del rango n es aproximadamente 1/n
            if len(freqs) > 5:
                # Comparar con distribución esperada
                expected_ratio = freqs[0] / freqs[1] if freqs[1] > 0 else 0
                actual_ratio = freqs[1] / freqs[2] if len(freqs) > 2 and freqs[2] > 0 else 0
                
                # Texto de IA puede desviarse de Zipf
                if abs(expected_ratio - actual_ratio) > 0.5:
                    score += 0.2
        
        # 2. Análisis de palabras raras vs comunes
        # Texto de IA suele tener menos palabras muy raras
        common_words = {'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 
                       'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at'}
        rare_words = [w for w, f in word_freq.items() if f == 1 and w not in common_words]
        
        if len(words) > 0:
            rare_ratio = len(rare_words) / len(words)
            # Texto de IA tiene menos palabras raras (hapax legomena)
            if rare_ratio < 0.3:  # Menos del 30% son palabras raras
                score += 0.2
        
        # 3. Análisis de frecuencia de palabras funcionales específicas
        # Texto de IA usa ciertas palabras funcionales más frecuentemente
        ai_functional_words = ['the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                             'have', 'has', 'had', 'will', 'would', 'could', 'should']
        
        ai_functional_count = sum(1 for w in words if w in ai_functional_words)
        if len(words) > 0:
            ai_functional_ratio = ai_functional_count / len(words)
            # Ratio muy alto puede indicar IA
            if ai_functional_ratio > 0.15:  # Más del 15%
                score += 0.2
        
        # 4. Análisis de frecuencia de palabras de contenido
        # Texto de IA tiene distribución más uniforme de palabras de contenido
        content_words = [w for w in words if len(w) > 4 and w not in common_words]
        content_freq = {}
        for word in content_words:
            content_freq[word] = content_freq.get(word, 0) + 1
        
        if content_freq:
            content_freqs = list(content_freq.values())
            if len(content_freqs) > 1:
                freq_variance = np.var(content_freqs)
                avg_freq = np.mean(content_freqs)
                
                # Baja varianza indica distribución uniforme (típico de IA)
                if avg_freq > 0:
                    cv = np.sqrt(freq_variance) / avg_freq
                    if cv < 0.4:  # Baja variación
                        score += 0.2
        
        # 5. Análisis de palabras de alta frecuencia
        # Texto de IA puede tener palabras que aparecen con frecuencia inusual
        if word_freq:
            max_freq = max(word_freq.values())
            total_words = len(words)
            
            # Si una palabra aparece más del 5% del tiempo
            if max_freq / total_words > 0.05:
                # Pero no es una palabra funcional común
                most_common = [w for w, f in word_freq.items() if f == max_freq][0]
                if most_common not in common_words:
                    score += 0.2  # Palabra de contenido muy repetida
        
        return min(score, 1.0)
    
    def _analyze_advanced_contextual_coherence(self, text: str) -> float:
        """Análisis avanzado de coherencia contextual - NUEVO"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 5:
            return 0.0
        
        # 1. Análisis de progresión temática
        words_per_sentence = [set(s.lower().split()) for s in sentences]
        overlaps = []
        for i in range(len(words_per_sentence) - 1):
            if words_per_sentence[i] and words_per_sentence[i+1]:
                overlap = len(words_per_sentence[i] & words_per_sentence[i+1])
                union = len(words_per_sentence[i] | words_per_sentence[i+1])
                jaccard = overlap / union if union > 0 else 0
                overlaps.append(jaccard)
        
        if overlaps:
            avg_overlap = np.mean(overlaps)
            if avg_overlap > 0.15:
                score += 0.3
            elif avg_overlap > 0.10:
                score += 0.2
        
        # 2. Análisis de referencias cruzadas
        pronoun_patterns = [
            r'\b(this|that|these|those|it|they|he|she)\s+\w+',
            r'\b(such|same|similar|different)\s+\w+',
            r'\b(above|below|previously|earlier|later)\s+\w+'
        ]
        cross_ref_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in pronoun_patterns)
        if cross_ref_count > len(sentences) * 0.3:
            score += 0.25
        
        # 3. Análisis de coherencia lógica
        logical_connectors = [
            r'\b(?:because|since|as)\s+\w+.*\b(?:therefore|thus|hence|so)\s+\w+',
            r'\b(?:if|when)\s+\w+.*\b(?:then|consequently|as a result)\s+\w+',
            r'\b(?:although|though|while)\s+\w+.*\b(?:however|nevertheless|yet)\s+\w+'
        ]
        logical_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in logical_connectors)
        if logical_count > 0:
            score += min(logical_count * 0.1, 0.25)
        
        # 4. Análisis de consistencia de perspectiva
        first_person = len(re.findall(r'\b(?:i|me|my|mine|we|us|our|ours)\b', text, re.IGNORECASE))
        second_person = len(re.findall(r'\b(?:you|your|yours)\b', text, re.IGNORECASE))
        third_person = len(re.findall(r'\b(?:he|she|him|her|his|hers|they|them|their|theirs|it|its)\b', text, re.IGNORECASE))
        total_perspective = first_person + second_person + third_person
        if total_perspective > 0:
            max_perspective = max(first_person, second_person, third_person)
            consistency = max_perspective / total_perspective
            if consistency > 0.6:
                score += 0.2
        
        return min(score, 1.0)
    
    def _detect_text_deepfake(self, text: str, detected_models: List[Dict]) -> float:
        """Detección de deepfake de texto - manipulación o combinación artificial - NUEVO"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 6:
            return 0.0
        
        # 1. Detección de cambios abruptos de estilo
        segment_size = len(sentences) // 3
        segments = [' '.join(sentences[i:i+segment_size]) for i in range(0, len(sentences), segment_size)]
        segment_features = []
        for segment in segments:
            words = segment.lower().split()
            if len(words) > 0:
                avg_length = np.mean([len(w) for w in words])
                formal_words = sum(1 for w in ['therefore', 'however', 'furthermore', 'moreover', 'consequently'] if w in segment.lower())
                segment_features.append({'avg_word_length': avg_length, 'formal_count': formal_words, 'length': len(words)})
        
        if len(segment_features) > 1:
            word_length_vars = [f['avg_word_length'] for f in segment_features]
            if len(word_length_vars) > 1:
                word_length_cv = np.std(word_length_vars) / np.mean(word_length_vars) if np.mean(word_length_vars) > 0 else 0
                if word_length_cv > 0.3:
                    score += 0.3
        
        # 2. Detección de modelos múltiples
        if len(detected_models) > 1:
            model_names = [m['model_name'] for m in detected_models]
            unique_models = len(set(model_names))
            if unique_models > 1:
                score += 0.25
        
        # 3. Detección de parches o ediciones
        edit_markers = [r'\[.*?\]', r'\(.*?edited.*?\)', r'\[edit\]', r'\.\.\.']
        edit_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in edit_markers)
        if edit_count > 2:
            score += 0.2
        
        # 4. Detección de inconsistencias temporales
        past_tense = len(re.findall(r'\b(?:was|were|had|did|went|came|said|told)\b', text, re.IGNORECASE))
        present_tense = len(re.findall(r'\b(?:is|are|am|do|does|go|come|say|tell)\b', text, re.IGNORECASE))
        future_tense = len(re.findall(r'\b(?:will|shall|going to|would|could|should)\b', text, re.IGNORECASE))
        total_tense = past_tense + present_tense + future_tense
        if total_tense > 0:
            max_tense = max(past_tense, present_tense, future_tense)
            tense_consistency = max_tense / total_tense
            if tense_consistency < 0.5:
                score += 0.15
        
        # 5. Detección de vocabulario mixto
        formal_words = ['therefore', 'however', 'furthermore', 'moreover', 'consequently', 'additionally', 'nevertheless', 'accordingly', 'subsequently']
        informal_words = ["don't", "won't", "can't", "it's", "that's", "there's", "gonna", "wanna", "yeah", "ok", "okay"]
        formal_count = sum(1 for w in formal_words if w in text.lower())
        informal_count = sum(1 for w in informal_words if w in text.lower())
        if formal_count > 2 and informal_count > 2:
            score += 0.15
        
        return min(score, 1.0)
    
    def _detect_deepfake_patterns(self, text: str) -> float:
        """Detecta patrones de deepfake o contenido sintético - NUEVO"""
        score = 0.0
        
        synthetic_patterns = [
            r'\b(?:synthetic|artificial|generated|created by ai|ai-generated)\b',
            r'\b(?:deepfake|fake|fabricated|manufactured)\b',
            r'\b(?:this (?:content|text|document) (?:was|is) (?:generated|created|produced) (?:by|using))\b'
        ]
        
        synthetic_matches = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in synthetic_patterns)
        if synthetic_matches > 0:
            score += 0.4
        
        manipulation_patterns = [
            r'\b(?:edited|modified|altered|manipulated|processed)\b',
            r'\b(?:original (?:text|content|version))\b',
            r'\b(?:before|after|original|modified version)\b'
        ]
        
        manipulation_matches = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in manipulation_patterns)
        if manipulation_matches > 1:
            score += 0.2
        
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) > 5:
            sentence_lengths = [len(s.split()) for s in sentences]
            if len(sentence_lengths) > 1:
                length_cv = np.std(sentence_lengths) / np.mean(sentence_lengths) if np.mean(sentence_lengths) > 0 else 0
                if length_cv > 0.8:
                    score += 0.2
        
        watermark_indicators = [
            r'\[.*?watermark.*?\]',
            r'<!--.*?generated.*?-->',
            r'\/\*.*?ai.*?\*\/',
            r'generated.*?by.*?ai',
            r'created.*?using.*?artificial.*?intelligence'
        ]
        
        watermark_matches = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in watermark_indicators)
        if watermark_matches > 0:
            score += 0.3
        
        hidden_patterns = [r'[^\x00-\x7F]', r'\s{3,}', r'\t+']
        hidden_count = sum(len(re.findall(pattern, text)) for pattern in hidden_patterns)
        if hidden_count > len(text) * 0.01:
            score += 0.1
        
        return min(score, 1.0)
    
    def _enhanced_scoring_system(self, scores: List[float], detection_methods: List[str], 
                                detected_models: List[Dict]) -> float:
        """Sistema de scoring mejorado con machine learning básico - NUEVO"""
        score = 0.0
        
        if not scores or len(scores) == 0:
            return 0.0
        
        high_scores = [s for s in scores if s > 0.6]
        medium_scores = [s for s in scores if 0.4 <= s <= 0.6]
        total_methods = len(scores)
        
        if total_methods > 0:
            high_ratio = len(high_scores) / total_methods
            medium_ratio = len(medium_scores) / total_methods
            
            if high_ratio > 0.4:
                score += 0.3
            elif high_ratio > 0.25:
                score += 0.2
            
            if 0.3 <= high_ratio <= 0.6 and 0.2 <= medium_ratio <= 0.5:
                score += 0.2
        
        critical_methods = ['pattern_matching', 'statistical_analysis', 'model_signatures']
        critical_scores = [s for m, s in zip(detection_methods, scores) if m in critical_methods]
        
        if critical_scores:
            avg_critical = np.mean(critical_scores)
            if avg_critical > 0.7:
                score += 0.3
            elif avg_critical > 0.5:
                score += 0.2
        
        if detected_models:
            high_confidence_models = [m for m in detected_models if m.get("confidence", 0) > 0.75]
            if len(high_confidence_models) > 0:
                score += 0.2
            
            if len(detected_models) > 1:
                confidences = [m.get("confidence", 0) for m in detected_models]
                if len(confidences) > 1:
                    conf_std = np.std(confidences)
                    if conf_std < 0.15:
                        score += 0.1
        
        if len(scores) > 3:
            score_mean = np.mean(scores)
            score_std = np.std(scores)
            if score_mean > 0 and score_std / score_mean < 0.3:
                score += 0.2
        
        complementary_pairs = [
            ('pattern_matching', 'model_signatures'),
            ('statistical_analysis', 'advanced_frequency'),
            ('semantic_coherence', 'narrative_coherence'),
            ('structure_analysis', 'syntactic_complexity')
        ]
        
        for method1, method2 in complementary_pairs:
            if method1 in detection_methods and method2 in detection_methods:
                idx1 = detection_methods.index(method1)
                idx2 = detection_methods.index(method2)
                score1 = scores[idx1]
                score2 = scores[idx2]
                if score1 > 0.5 and score2 > 0.5:
                    score += 0.1
        
        return min(score, 1.0)
    
    def _analyze_advanced_repetition_patterns(self, text: str) -> float:
        """Análisis avanzado de patrones de repetición - NUEVO"""
        score = 0.0
        words = text.lower().split()
        
        if len(words) < 30:
            return 0.0
        
        word_freq = {}
        for word in words:
            word_clean = word.strip('.,!?;:()[]{}"\'')
            if word_clean and len(word_clean) > 3:
                word_freq[word_clean] = word_freq.get(word_clean, 0) + 1
        
        if not word_freq:
            return 0.0
        
        total_words = len(words)
        max_freq = max(word_freq.values())
        max_freq_ratio = max_freq / total_words if total_words > 0 else 0
        
        if max_freq_ratio > 0.08:
            score += 0.3
        elif max_freq_ratio > 0.05:
            score += 0.2
        
        repeated_words = [w for w, f in word_freq.items() if f > 3]
        if len(repeated_words) > len(word_freq) * 0.15:
            score += 0.25
        
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) > 3:
            sentence_starts = [s.split()[0].lower() if s.split() else '' for s in sentences]
            start_repetition = len(sentence_starts) - len(set(sentence_starts))
            if start_repetition > len(sentences) * 0.3:
                score += 0.2
        
        phrase_patterns = {}
        for i in range(len(words) - 2):
            phrase = ' '.join(words[i:i+3])
            phrase_patterns[phrase] = phrase_patterns.get(phrase, 0) + 1
        
        repeated_phrases = [p for p, f in phrase_patterns.items() if f > 1]
        if len(repeated_phrases) > 0 and len(repeated_phrases) / len(phrase_patterns) > 0.1:
            score += 0.25
        
        return min(score, 1.0)
    
    def _detect_ai_paraphrasing_advanced(self, text: str) -> float:
        """Detección avanzada de parafraseo con IA - NUEVO"""
        score = 0.0
        words = text.split()
        
        if len(words) < 50:
            return 0.0
        
        synonym_patterns = [
            (r'\b(?:important|significant|crucial|vital|essential)\b', 0.1),
            (r'\b(?:help|assist|aid|support|facilitate)\b', 0.1),
            (r'\b(?:good|excellent|outstanding|remarkable|exceptional)\b', 0.1),
            (r'\b(?:bad|poor|terrible|awful|dreadful)\b', 0.1),
            (r'\b(?:big|large|huge|enormous|massive)\b', 0.1),
            (r'\b(?:small|tiny|little|miniature|minuscule)\b', 0.1)
        ]
        
        synonym_count = 0
        for pattern, weight in synonym_patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            if matches > 2:
                synonym_count += matches
                score += min(matches * weight, 0.15)
        
        structural_variations = [
            r'\b(?:in order to|so as to|for the purpose of)\b',
            r'\b(?:due to the fact that|because|since|as)\b',
            r'\b(?:in the event that|if|should|when)\b',
            r'\b(?:with regard to|regarding|concerning|about)\b'
        ]
        
        variation_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in structural_variations)
        if variation_count > 3:
            score += 0.2
        
        formal_alternatives = [
            (r'\b(?:utilize|use|employ)\b', 0.1),
            (r'\b(?:facilitate|help|make easier)\b', 0.1),
            (r'\b(?:implement|put into effect|carry out)\b', 0.1),
            (r'\b(?:optimize|improve|make better)\b', 0.1)
        ]
        
        for pattern, weight in formal_alternatives:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            if matches > 1:
                score += min(matches * weight, 0.1)
        
        if synonym_count > 5:
            score += 0.15
        
        return min(score, 1.0)
    
    def _analyze_style_mixture(self, text: str) -> float:
        """Detecta mezclas de estilos (formal/informal) - NUEVO"""
        score = 0.0
        words = text.split()
        
        if len(words) < 40:
            return 0.0
        
        formal_indicators = [
            'therefore', 'however', 'furthermore', 'moreover', 'consequently',
            'additionally', 'nevertheless', 'accordingly', 'subsequently',
            'utilize', 'facilitate', 'implement', 'optimize', 'enhance'
        ]
        
        informal_indicators = [
            "don't", "won't", "can't", "it's", "that's", "there's",
            "gonna", "wanna", "yeah", "ok", "okay", "lol", "btw",
            "imo", "tbh", "nvm", "idk"
        ]
        
        formal_count = sum(1 for word in formal_indicators if word.lower() in text.lower())
        informal_count = sum(1 for word in informal_indicators if word.lower() in text.lower())
        
        if formal_count > 3 and informal_count > 2:
            score += 0.4
        elif formal_count > 2 and informal_count > 1:
            score += 0.3
        
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) > 5:
            formal_sentences = sum(1 for s in sentences if any(fi in s.lower() for fi in formal_indicators))
            informal_sentences = sum(1 for s in sentences if any(ii in s.lower() for ii in informal_indicators))
            
            if formal_sentences > 0 and informal_sentences > 0:
                mixture_ratio = min(formal_sentences, informal_sentences) / max(formal_sentences, informal_sentences)
                if mixture_ratio > 0.3:
                    score += 0.3
        
        academic_words = ['furthermore', 'moreover', 'consequently', 'therefore', 'nevertheless']
        casual_words = ["don't", "won't", "can't", "gonna", "wanna"]
        
        academic_count = sum(1 for w in academic_words if w in text.lower())
        casual_count = sum(1 for w in casual_words if w in text.lower())
        
        if academic_count > 2 and casual_count > 1:
            score += 0.3
        
        return min(score, 1.0)
    
    def _analyze_generation_sophistication(self, text: str) -> float:
        """Análisis sofisticado de patrones de generación - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 50 or len(sentences) < 5:
            return 0.0
        
        sophisticated_starters = [
            r'^(?:it is|it\'s) (?:important|worth noting|crucial|essential) (?:to note|to mention|to point out)',
            r'^(?:in|when) (?:considering|analyzing|examining|evaluating)',
            r'^(?:to|in order to) (?:better|more effectively|more accurately)',
            r'^(?:this|these) (?:findings|results|observations|insights)',
            r'^(?:it|this) (?:should|must|needs to) (?:be noted|be emphasized|be highlighted)'
        ]
        
        sophisticated_count = 0
        for pattern in sophisticated_starters:
            matches = sum(1 for s in sentences if re.search(pattern, s, re.IGNORECASE))
            sophisticated_count += matches
        
        if sophisticated_count > 2:
            score += 0.3
        elif sophisticated_count > 0:
            score += 0.15
        
        transition_overuse = [
            'however', 'therefore', 'furthermore', 'moreover', 'consequently',
            'additionally', 'meanwhile', 'subsequently', 'nevertheless', 'thus'
        ]
        
        transition_count = sum(1 for word in transition_overuse if word.lower() in text.lower())
        if transition_count > len(sentences) * 0.2:
            score += 0.25
        
        hedging_phrases = [
            r'\b(?:it (?:is|may be|might be|could be) (?:possible|likely|probable|plausible))',
            r'\b(?:there (?:is|may be|might be|could be) (?:a possibility|a chance|a likelihood))',
            r'\b(?:it (?:seems|appears|looks) (?:that|as if|as though))',
            r'\b(?:one (?:might|could|may) (?:argue|suggest|propose|consider))'
        ]
        
        hedging_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in hedging_phrases)
        if hedging_count > 2:
            score += 0.2
        
        qualifier_overuse = [
            'somewhat', 'rather', 'quite', 'fairly', 'relatively', 'comparatively',
            'generally', 'typically', 'usually', 'often', 'frequently'
        ]
        
        qualifier_count = sum(1 for q in qualifier_overuse if q.lower() in text.lower())
        if qualifier_count > len(words) * 0.02:
            score += 0.15
        
        return min(score, 1.0)
    
    def _analyze_lexical_diversity_advanced(self, text: str) -> float:
        """Análisis avanzado de diversidad léxica - NUEVO"""
        score = 0.0
        words = text.lower().split()
        
        if len(words) < 50:
            return 0.0
        
        word_freq = {}
        for word in words:
            word_clean = word.strip('.,!?;:()[]{}"\'')
            if word_clean and len(word_clean) > 2:
                word_freq[word_clean] = word_freq.get(word_clean, 0) + 1
        
        if not word_freq:
            return 0.0
        
        unique_words = len(word_freq)
        total_words = len(words)
        type_token_ratio = unique_words / total_words if total_words > 0 else 0
        
        if 0.3 <= type_token_ratio <= 0.5:
            score += 0.3
        elif type_token_ratio < 0.3:
            score += 0.2
        
        hapax_legomena = [w for w, f in word_freq.items() if f == 1]
        hapax_ratio = len(hapax_legomena) / unique_words if unique_words > 0 else 0
        
        if hapax_ratio < 0.4:
            score += 0.25
        
        word_lengths = [len(w) for w in word_freq.keys()]
        if word_lengths:
            avg_word_length = np.mean(word_lengths)
            if 4.5 <= avg_word_length <= 6.5:
                score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_hedging_patterns(self, text: str) -> float:
        """Detecta patrones de hedging (cautela) típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        
        if len(words) < 30:
            return 0.0
        
        hedging_phrases = [
            r'\b(?:it (?:is|may be|might be|could be|seems|appears) (?:possible|likely|probable|plausible|important|worth noting))',
            r'\b(?:there (?:is|may be|might be|could be) (?:a possibility|a chance|a likelihood|evidence))',
            r'\b(?:it (?:seems|appears|looks) (?:that|as if|as though))',
            r'\b(?:one (?:might|could|may|should) (?:argue|suggest|propose|consider|note))',
            r'\b(?:it (?:would|could|might) (?:be (?:argued|suggested|noted|considered))|seem)',
            r'\b(?:to (?:some|a certain) (?:extent|degree))',
            r'\b(?:in (?:some|many|most) (?:cases|instances|situations))',
            r'\b(?:generally|typically|usually|often|frequently|commonly)'
        ]
        
        hedging_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in hedging_phrases)
        
        if hedging_count > len(words) * 0.03:
            score += 0.4
        elif hedging_count > len(words) * 0.02:
            score += 0.3
        elif hedging_count > 0:
            score += 0.15
        
        uncertainty_markers = [
            'perhaps', 'maybe', 'possibly', 'potentially', 'presumably',
            'supposedly', 'allegedly', 'reportedly', 'apparently', 'seemingly'
        ]
        
        uncertainty_count = sum(1 for marker in uncertainty_markers if marker.lower() in text.lower())
        if uncertainty_count > 2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_sentence_complexity_distribution(self, text: str) -> float:
        """Analiza la distribución de complejidad de oraciones - NUEVO"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 10:
            return 0.0
        
        sentence_lengths = [len(s.split()) for s in sentences]
        if len(sentence_lengths) < 2:
            return 0.0
        
        mean_length = np.mean(sentence_lengths)
        std_length = np.std(sentence_lengths)
        cv = std_length / mean_length if mean_length > 0 else 0
        
        if cv < 0.3:
            score += 0.3
        elif cv < 0.4:
            score += 0.2
        
        complex_sentences = 0
        for sentence in sentences:
            clause_count = len(re.findall(r'\b(?:and|or|but|because|since|although|while|if|when)\b', sentence, re.IGNORECASE))
            if clause_count > 1:
                complex_sentences += 1
        
        complex_ratio = complex_sentences / len(sentences) if len(sentences) > 0 else 0
        
        if 0.4 <= complex_ratio <= 0.7:
            score += 0.25
        elif complex_ratio > 0.7:
            score += 0.15
        
        simple_sentences = sum(1 for s in sentences if len(s.split()) < 10)
        medium_sentences = sum(1 for s in sentences if 10 <= len(s.split()) <= 20)
        long_sentences = sum(1 for s in sentences if len(s.split()) > 20)
        
        if len(sentences) > 0:
            simple_ratio = simple_sentences / len(sentences)
            medium_ratio = medium_sentences / len(sentences)
            long_ratio = long_sentences / len(sentences)
            
            if 0.3 <= medium_ratio <= 0.6 and simple_ratio < 0.3 and long_ratio < 0.3:
                score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_verbosity_patterns(self, text: str) -> float:
        """Detecta patrones de verbosidad típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 40:
            return 0.0
        
        verbose_phrases = [
            r'\b(?:it is important to note that|it should be noted that|it is worth mentioning that)',
            r'\b(?:in order to|so as to|for the purpose of)',
            r'\b(?:due to the fact that|because of the fact that)',
            r'\b(?:with regard to|in regard to|concerning the matter of)',
            r'\b(?:in the context of|within the framework of|in the realm of)',
            r'\b(?:it can be observed that|it may be seen that|it is evident that)',
            r'\b(?:as a result of the fact that|owing to the fact that)',
            r'\b(?:in the event that|in case that|should it be the case that)'
        ]
        
        verbose_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in verbose_phrases)
        
        if verbose_count > 3:
            score += 0.4
        elif verbose_count > 1:
            score += 0.25
        
        redundant_phrases = [
            r'\b(?:each and every|first and foremost|any and all)',
            r'\b(?:various different|many different|several different)',
            r'\b(?:completely finished|totally complete|absolutely certain)',
            r'\b(?:free gift|new innovation|past history)'
        ]
        
        redundant_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in redundant_phrases)
        if redundant_count > 1:
            score += 0.2
        
        avg_words_per_sentence = len(words) / len(sentences) if len(sentences) > 0 else 0
        
        if 15 <= avg_words_per_sentence <= 25:
            score += 0.15
        
        return min(score, 1.0)
    
    def _analyze_pronoun_usage_patterns(self, text: str) -> float:
        """Analiza patrones de uso de pronombres - NUEVO"""
        score = 0.0
        words = text.split()
        
        if len(words) < 30:
            return 0.0
        
        first_person = len(re.findall(r'\b(?:i|me|my|mine|we|us|our|ours)\b', text, re.IGNORECASE))
        second_person = len(re.findall(r'\b(?:you|your|yours)\b', text, re.IGNORECASE))
        third_person = len(re.findall(r'\b(?:he|she|him|her|his|hers|they|them|their|theirs|it|its)\b', text, re.IGNORECASE))
        
        total_pronouns = first_person + second_person + third_person
        
        if total_pronouns > 0:
            first_ratio = first_person / total_pronouns
            second_ratio = second_person / total_pronouns
            third_ratio = third_person / total_pronouns
            
            if third_ratio > 0.6:
                score += 0.3
            elif third_ratio > 0.5:
                score += 0.2
            
            if first_ratio < 0.1 and second_ratio < 0.1:
                score += 0.2
        
        pronoun_repetition = len(re.findall(r'\b(it|this|that|these|those)\s+\w+\s+\1\b', text, re.IGNORECASE))
        if pronoun_repetition > 2:
            score += 0.15
        
        return min(score, 1.0)
    
    def _detect_ai_question_patterns(self, text: str) -> float:
        """Detecta patrones de preguntas típicos de IA - NUEVO"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 5:
            return 0.0
        
        question_count = text.count('?')
        question_ratio = question_count / len(sentences) if len(sentences) > 0 else 0
        
        if question_ratio > 0.3:
            score += 0.3
        elif question_ratio > 0.2:
            score += 0.2
        
        rhetorical_questions = [
            r'\b(?:have you ever|did you know|are you aware|do you realize)',
            r'\b(?:what if|what would happen if|imagine if)',
            r'\b(?:isn\'t it|don\'t you think|wouldn\'t you agree)'
        ]
        
        rhetorical_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in rhetorical_questions)
        if rhetorical_count > 1:
            score += 0.25
        
        question_starters = [
            r'^(?:what|how|why|when|where|who|which|can|could|would|should|may|might)',
            r'^(?:is|are|was|were|do|does|did|have|has|had)'
        ]
        
        question_starter_count = sum(1 for s in sentences 
                                    if any(re.search(pattern, s, re.IGNORECASE) for pattern in question_starters))
        
        if question_starter_count > len(sentences) * 0.2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_closure_patterns(self, text: str) -> float:
        """Analiza patrones de cierre típicos de IA - NUEVO"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 5:
            return 0.0
        
        last_sentences = ' '.join(sentences[-3:]) if len(sentences) >= 3 else ' '.join(sentences)
        
        closure_patterns = [
            r'\b(?:i hope this helps|hope this helps|i hope this|hope this)',
            r'\b(?:let me know if|feel free to|don\'t hesitate to|if you have any)',
            r'\b(?:please let me know|please feel free|if you need|if you require)',
            r'\b(?:in conclusion|to summarize|to conclude|in summary|to sum up)',
            r'\b(?:thank you|thanks|appreciate|grateful)',
            r'\b(?:if you have any questions|if you need clarification|if you\'d like to know more)'
        ]
        
        closure_count = sum(len(re.findall(pattern, last_sentences, re.IGNORECASE)) for pattern in closure_patterns)
        
        if closure_count > 1:
            score += 0.4
        elif closure_count > 0:
            score += 0.25
        
        ending_phrases = [
            'best regards', 'sincerely', 'yours truly', 'take care',
            'have a great day', 'good luck', 'all the best'
        ]
        
        ending_count = sum(1 for phrase in ending_phrases if phrase.lower() in last_sentences.lower())
        if ending_count > 0:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_enumeration_patterns(self, text: str) -> float:
        """Detecta patrones de enumeración típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        
        if len(words) < 40:
            return 0.0
        
        enumeration_markers = [
            r'\b(?:first|second|third|fourth|fifth|sixth|finally|lastly|last)',
            r'\b(?:one|two|three|four|five|six|seven|eight|nine|ten)',
            r'\b(?:firstly|secondly|thirdly|fourthly|fifthly)',
            r'\b(?:initially|subsequently|then|next|afterward|finally)'
        ]
        
        enum_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in enumeration_markers)
        
        if enum_count > 5:
            score += 0.4
        elif enum_count > 3:
            score += 0.3
        elif enum_count > 1:
            score += 0.15
        
        numbered_lists = len(re.findall(r'\d+\.\s+[A-Z]', text))
        bullet_lists = len(re.findall(r'[•\-\*]\s+[A-Z]', text))
        
        if numbered_lists > 3 or bullet_lists > 3:
            score += 0.3
        elif numbered_lists > 1 or bullet_lists > 1:
            score += 0.2
        
        sequential_patterns = [
            r'\b(?:step \d+|stage \d+|phase \d+|part \d+)',
            r'\b(?:point \d+|item \d+|element \d+)'
        ]
        
        sequential_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in sequential_patterns)
        if sequential_count > 2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_metaphor_patterns(self, text: str) -> float:
        """Analiza patrones de metáforas y lenguaje figurado - NUEVO"""
        score = 0.0
        words = text.split()
        
        if len(words) < 50:
            return 0.0
        
        metaphor_indicators = [
            r'\b(?:like|as|similar to|comparable to|akin to)\s+\w+',
            r'\b(?:metaphor|analogy|simile|comparison)',
            r'\b(?:is like|is as|resembles|mirrors|reflects)',
            r'\b(?:think of|imagine|picture|visualize)'
        ]
        
        metaphor_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in metaphor_indicators)
        
        if metaphor_count == 0 and len(words) > 100:
            score += 0.3
        
        figurative_language = [
            r'\b(?:it\'s like|it\'s as if|it\'s similar to)',
            r'\b(?:think of it as|imagine it as|picture it as)',
            r'\b(?:just like|much like|very much like)'
        ]
        
        figurative_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in figurative_language)
        if figurative_count > 2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_emphasis_patterns(self, text: str) -> float:
        """Detecta patrones de énfasis típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        
        if len(words) < 30:
            return 0.0
        
        emphasis_markers = [
            r'\b(?:important|crucial|essential|vital|critical|significant)',
            r'\b(?:very|extremely|highly|particularly|especially|notably)',
            r'\b(?:indeed|certainly|definitely|absolutely|undoubtedly)',
            r'\b(?:it is important|it is crucial|it is essential|it is vital)',
            r'\b(?:must|should|need to|have to|ought to)'
        ]
        
        emphasis_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in emphasis_markers)
        
        if emphasis_count > len(words) * 0.03:
            score += 0.4
        elif emphasis_count > len(words) * 0.02:
            score += 0.3
        elif emphasis_count > 3:
            score += 0.15
        
        intensifiers = ['very', 'extremely', 'highly', 'particularly', 'especially', 'notably', 'remarkably']
        intensifier_count = sum(1 for word in intensifiers if word.lower() in text.lower())
        
        if intensifier_count > len(words) * 0.02:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_modifier_patterns(self, text: str) -> float:
        """Analiza patrones de modificadores típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        
        if len(words) < 40:
            return 0.0
        
        modifier_patterns = [
            r'\b(?:quite|rather|somewhat|fairly|relatively|comparatively)\s+\w+',
            r'\b(?:very|extremely|highly|particularly|especially)\s+\w+',
            r'\b(?:more|less|most|least)\s+\w+',
            r'\b(?:incredibly|remarkably|exceptionally|notably)\s+\w+'
        ]
        
        modifier_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in modifier_patterns)
        
        if modifier_count > len(words) * 0.04:
            score += 0.3
        elif modifier_count > len(words) * 0.02:
            score += 0.2
        
        redundant_modifiers = [
            r'\b(?:very|extremely|incredibly)\s+(?:important|crucial|essential|significant)',
            r'\b(?:quite|rather|somewhat)\s+(?:interesting|notable|remarkable)',
            r'\b(?:highly|extremely|very)\s+(?:effective|efficient|successful)'
        ]
        
        redundant_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in redundant_modifiers)
        if redundant_count > 1:
            score += 0.25
        
        return min(score, 1.0)
    
    def _detect_ai_conditional_patterns(self, text: str) -> float:
        """Detecta patrones condicionales típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        conditional_markers = [
            r'\b(?:if|when|unless|provided that|as long as|in case)',
            r'\b(?:would|could|should|might|may)\s+\w+',
            r'\b(?:if\s+\w+\s+then|if\s+\w+\s+would|if\s+\w+\s+could)',
            r'\b(?:assuming|supposing|given that|considering that)'
        ]
        
        conditional_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in conditional_markers)
        
        if conditional_count > len(sentences) * 0.3:
            score += 0.3
        elif conditional_count > len(sentences) * 0.2:
            score += 0.2
        
        hypothetical_phrases = [
            r'\b(?:if\s+you|if\s+one|if\s+someone|if\s+we)',
            r'\b(?:suppose|imagine|assume|presume)',
            r'\b(?:what if|what would happen if|what might happen if)'
        ]
        
        hypothetical_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in hypothetical_phrases)
        if hypothetical_count > 2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_passive_voice_patterns(self, text: str) -> float:
        """Analiza patrones de voz pasiva típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        passive_voice_patterns = [
            r'\b(?:is|are|was|were|been|being)\s+\w+ed\b',
            r'\b(?:is|are|was|were|been|being)\s+\w+en\b',
            r'\b(?:is|are|was|were|been|being)\s+\w+ed\s+by\b',
            r'\b(?:has|have|had)\s+been\s+\w+ed\b',
            r'\b(?:will|would|can|could|should|must)\s+be\s+\w+ed\b'
        ]
        
        passive_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in passive_voice_patterns)
        
        if passive_count > len(sentences) * 0.4:
            score += 0.4
        elif passive_count > len(sentences) * 0.25:
            score += 0.3
        elif passive_count > len(sentences) * 0.15:
            score += 0.15
        
        passive_indicators = ['by', 'was', 'were', 'been', 'being']
        passive_indicator_count = sum(1 for word in words if word.lower() in passive_indicators)
        
        if passive_indicator_count > len(words) * 0.05:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_connector_patterns(self, text: str) -> float:
        """Detecta patrones de conectores típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 40:
            return 0.0
        
        connector_patterns = [
            r'\b(?:furthermore|moreover|additionally|in addition|also|besides)',
            r'\b(?:however|nevertheless|nonetheless|on the other hand|conversely)',
            r'\b(?:therefore|thus|hence|consequently|as a result|accordingly)',
            r'\b(?:for instance|for example|such as|namely|specifically)',
            r'\b(?:in conclusion|to summarize|in summary|overall|ultimately)'
        ]
        
        connector_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in connector_patterns)
        
        if connector_count > len(sentences) * 0.3:
            score += 0.4
        elif connector_count > len(sentences) * 0.2:
            score += 0.3
        elif connector_count > 3:
            score += 0.15
        
        transition_phrases = [
            r'\b(?:first of all|secondly|thirdly|finally|lastly)',
            r'\b(?:on the one hand|on the other hand)',
            r'\b(?:in other words|that is to say|to put it differently)'
        ]
        
        transition_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in transition_phrases)
        if transition_count > 2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_quantifier_patterns(self, text: str) -> float:
        """Analiza patrones de cuantificadores típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        
        if len(words) < 30:
            return 0.0
        
        quantifier_patterns = [
            r'\b(?:many|most|several|various|numerous|multiple|various)',
            r'\b(?:some|few|many|most|all|every|each)',
            r'\b(?:a number of|a variety of|a range of|a series of)',
            r'\b(?:the majority of|the majority|most of|many of)'
        ]
        
        quantifier_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in quantifier_patterns)
        
        if quantifier_count > len(words) * 0.03:
            score += 0.3
        elif quantifier_count > len(words) * 0.02:
            score += 0.2
        elif quantifier_count > 2:
            score += 0.1
        
        vague_quantifiers = ['many', 'some', 'several', 'various', 'numerous', 'multiple']
        vague_count = sum(1 for word in words if word.lower() in vague_quantifiers)
        
        if vague_count > len(words) * 0.02:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_assertion_patterns(self, text: str) -> float:
        """Detecta patrones de aserciones típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        assertion_patterns = [
            r'\b(?:it is clear that|it is evident that|it is obvious that)',
            r'\b(?:there is no doubt that|undoubtedly|certainly|definitely)',
            r'\b(?:it can be seen that|it can be observed that|it can be noted that)',
            r'\b(?:it should be noted that|it is important to note that)',
            r'\b(?:it is worth noting that|it is worth mentioning that)'
        ]
        
        assertion_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in assertion_patterns)
        
        if assertion_count > len(sentences) * 0.2:
            score += 0.4
        elif assertion_count > len(sentences) * 0.1:
            score += 0.3
        elif assertion_count > 1:
            score += 0.15
        
        definitive_statements = [
            r'\b(?:always|never|all|every|none|no one|nothing)',
            r'\b(?:must|should|ought to|have to|need to)',
            r'\b(?:cannot|can never|will never|must not)'
        ]
        
        definitive_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in definitive_statements)
        if definitive_count > len(sentences) * 0.3:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_negation_patterns(self, text: str) -> float:
        """Analiza patrones de negación típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        negation_patterns = [
            r'\b(?:not|no|never|neither|nor|none|nothing|nobody|nowhere)',
            r'\b(?:cannot|cannot|could not|would not|should not|must not)',
            r'\b(?:is not|are not|was not|were not|has not|have not)',
            r'\b(?:does not|do not|did not|will not|would not)'
        ]
        
        negation_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in negation_patterns)
        
        if negation_count > len(sentences) * 0.3:
            score += 0.3
        elif negation_count > len(sentences) * 0.2:
            score += 0.2
        elif negation_count > 2:
            score += 0.1
        
        double_negation = len(re.findall(r'\b(?:not|no|never)\s+\w+\s+(?:not|no|never)', text, re.IGNORECASE))
        if double_negation > 0:
            score += 0.15
        
        return min(score, 1.0)
    
    def _detect_ai_comparison_patterns(self, text: str) -> float:
        """Detecta patrones de comparación típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        comparison_patterns = [
            r'\b(?:compared to|compared with|in comparison to|in comparison with)',
            r'\b(?:similar to|similar as|as similar as|like|as)',
            r'\b(?:different from|different than|unlike|contrary to)',
            r'\b(?:more than|less than|greater than|smaller than)',
            r'\b(?:as\s+\w+\s+as|so\s+\w+\s+as|such\s+as)'
        ]
        
        comparison_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in comparison_patterns)
        
        if comparison_count > len(sentences) * 0.25:
            score += 0.3
        elif comparison_count > len(sentences) * 0.15:
            score += 0.2
        elif comparison_count > 2:
            score += 0.1
        
        superlative_patterns = [
            r'\b(?:the most|the least|the best|the worst|the greatest|the smallest)',
            r'\b(?:more\s+\w+|\w+er\s+than)',
            r'\b(?:most\s+\w+|least\s+\w+)'
        ]
        
        superlative_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in superlative_patterns)
        if superlative_count > len(sentences) * 0.2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_temporal_marker_patterns(self, text: str) -> float:
        """Analiza patrones de marcadores temporales típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        temporal_markers = [
            r'\b(?:first|initially|originally|at first|in the beginning)',
            r'\b(?:then|next|subsequently|afterward|after that|following that)',
            r'\b(?:finally|ultimately|eventually|in the end|at last)',
            r'\b(?:meanwhile|simultaneously|at the same time|concurrently)',
            r'\b(?:previously|before|earlier|prior to|in the past)',
            r'\b(?:later|afterwards|subsequently|then|following)'
        ]
        
        temporal_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in temporal_markers)
        
        if temporal_count > len(sentences) * 0.3:
            score += 0.3
        elif temporal_count > len(sentences) * 0.2:
            score += 0.2
        elif temporal_count > 2:
            score += 0.1
        
        sequential_markers = ['first', 'second', 'third', 'fourth', 'fifth', 'then', 'next', 'finally']
        sequential_count = sum(1 for word in words if word.lower() in sequential_markers)
        
        if sequential_count > len(words) * 0.02:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_causality_patterns(self, text: str) -> float:
        """Detecta patrones de causalidad típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        causality_patterns = [
            r'\b(?:because|since|as|due to|owing to|as a result of)',
            r'\b(?:therefore|thus|hence|consequently|as a result|accordingly)',
            r'\b(?:causes|caused by|leads to|results in|brings about)',
            r'\b(?:if\s+\w+\s+then|when\s+\w+\s+then|whenever\s+\w+\s+then)',
            r'\b(?:the reason why|the reason that|why\s+\w+\s+is)'
        ]
        
        causality_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in causality_patterns)
        
        if causality_count > len(sentences) * 0.3:
            score += 0.3
        elif causality_count > len(sentences) * 0.2:
            score += 0.2
        elif causality_count > 2:
            score += 0.1
        
        effect_markers = ['therefore', 'thus', 'hence', 'consequently', 'as a result', 'accordingly']
        effect_count = sum(1 for word in words if word.lower() in effect_markers)
        
        if effect_count > len(sentences) * 0.15:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_modal_verb_patterns(self, text: str) -> float:
        """Analiza patrones de verbos modales típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        modal_verbs = ['can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would']
        modal_count = sum(1 for word in words if word.lower() in modal_verbs)
        
        if modal_count > len(sentences) * 0.4:
            score += 0.3
        elif modal_count > len(sentences) * 0.3:
            score += 0.2
        elif modal_count > len(sentences) * 0.2:
            score += 0.1
        
        modal_patterns = [
            r'\b(?:can|could|may|might|must|should|would)\s+be',
            r'\b(?:can|could|may|might|must|should|would)\s+have',
            r'\b(?:can|could|may|might|must|should|would)\s+not',
            r'\b(?:it\s+(?:can|could|may|might|must|should|would))'
        ]
        
        modal_pattern_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in modal_patterns)
        if modal_pattern_count > len(sentences) * 0.25:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_hedge_phrase_patterns(self, text: str) -> float:
        """Detecta patrones de frases de hedging típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        hedge_phrases = [
            r'\b(?:it seems|it appears|it would seem|it might seem)',
            r'\b(?:perhaps|maybe|possibly|probably|likely|unlikely)',
            r'\b(?:to some extent|in some way|in a sense|to a certain degree)',
            r'\b(?:might be|could be|may be|seems to be|appears to be)',
            r'\b(?:generally|usually|typically|often|sometimes|occasionally)'
        ]
        
        hedge_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in hedge_phrases)
        
        if hedge_count > len(sentences) * 0.3:
            score += 0.3
        elif hedge_count > len(sentences) * 0.2:
            score += 0.2
        elif hedge_count > 2:
            score += 0.1
        
        uncertainty_markers = ['perhaps', 'maybe', 'possibly', 'probably', 'likely', 'unlikely', 'might', 'could']
        uncertainty_count = sum(1 for word in words if word.lower() in uncertainty_markers)
        
        if uncertainty_count > len(words) * 0.02:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_relative_clause_patterns(self, text: str) -> float:
        """Analiza patrones de cláusulas relativas típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        relative_clause_patterns = [
            r'\b(?:which|that|who|whom|whose|where|when)\s+\w+',
            r'\b(?:,\s+which|,\s+that|,\s+who|,\s+whom|,\s+whose)',
            r'\b(?:of which|of whom|of whose|in which|at which|on which)',
            r'\b(?:the\s+\w+\s+(?:which|that|who|whom|whose))'
        ]
        
        relative_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in relative_clause_patterns)
        
        if relative_count > len(sentences) * 0.4:
            score += 0.3
        elif relative_count > len(sentences) * 0.3:
            score += 0.2
        elif relative_count > len(sentences) * 0.2:
            score += 0.1
        
        relative_pronouns = ['which', 'that', 'who', 'whom', 'whose', 'where', 'when']
        relative_pronoun_count = sum(1 for word in words if word.lower() in relative_pronouns)
        
        if relative_pronoun_count > len(words) * 0.03:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_infinitive_patterns(self, text: str) -> float:
        """Detecta patrones de infinitivos típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        infinitive_patterns = [
            r'\b(?:to\s+\w+\s+the|to\s+\w+\s+a|to\s+\w+\s+an)',
            r'\b(?:in order to|so as to|in an attempt to|with the aim to)',
            r'\b(?:it is\s+\w+\s+to|it was\s+\w+\s+to|it would be\s+\w+\s+to)',
            r'\b(?:the\s+\w+\s+to\s+\w+)',
            r'\b(?:to\s+\w+\s+and\s+to\s+\w+)'
        ]
        
        infinitive_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in infinitive_patterns)
        
        if infinitive_count > len(sentences) * 0.3:
            score += 0.3
        elif infinitive_count > len(sentences) * 0.2:
            score += 0.2
        elif infinitive_count > 2:
            score += 0.1
        
        to_infinitive = len(re.findall(r'\bto\s+\w+', text, re.IGNORECASE))
        if to_infinitive > len(words) * 0.05:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_gerund_patterns(self, text: str) -> float:
        """Analiza patrones de gerundios típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        gerund_patterns = [
            r'\b\w+ing\s+(?:the|a|an|this|that|these|those)',
            r'\b(?:by|while|when|after|before|upon)\s+\w+ing',
            r'\b(?:is|are|was|were|being)\s+\w+ing',
            r'\b(?:start|begin|continue|keep|stop|finish)\s+\w+ing',
            r'\b(?:without|instead of|in addition to)\s+\w+ing'
        ]
        
        gerund_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in gerund_patterns)
        
        if gerund_count > len(sentences) * 0.3:
            score += 0.3
        elif gerund_count > len(sentences) * 0.2:
            score += 0.2
        elif gerund_count > 2:
            score += 0.1
        
        ing_words = len(re.findall(r'\b\w+ing\b', text, re.IGNORECASE))
        if ing_words > len(words) * 0.04:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_participle_patterns(self, text: str) -> float:
        """Detecta patrones de participios típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        participle_patterns = [
            r'\b\w+ed\s+(?:by|with|in|on|at|from|to)',
            r'\b\w+en\s+(?:by|with|in|on|at|from|to)',
            r'\b(?:having|being|having been)\s+\w+ed',
            r'\b(?:having|being|having been)\s+\w+en',
            r'\b(?:the|a|an)\s+\w+ed\s+\w+',
            r'\b(?:the|a|an)\s+\w+en\s+\w+'
        ]
        
        participle_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in participle_patterns)
        
        if participle_count > len(sentences) * 0.3:
            score += 0.3
        elif participle_count > len(sentences) * 0.2:
            score += 0.2
        elif participle_count > 2:
            score += 0.1
        
        past_participle = len(re.findall(r'\b\w+ed\b', text, re.IGNORECASE))
        past_participle_en = len(re.findall(r'\b\w+en\b', text, re.IGNORECASE))
        
        if (past_participle + past_participle_en) > len(words) * 0.05:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_subjunctive_patterns(self, text: str) -> float:
        """Analiza patrones de subjuntivo típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        subjunctive_patterns = [
            r'\b(?:if\s+\w+\s+were|if\s+\w+\s+had|if\s+\w+\s+could)',
            r'\b(?:it\s+is\s+important\s+that|it\s+is\s+crucial\s+that|it\s+is\s+essential\s+that)',
            r'\b(?:suggest|recommend|insist|demand|require)\s+that',
            r'\b(?:wish|hope|prefer|desire)\s+(?:that|to)',
            r'\b(?:as\s+if|as\s+though|even\s+if|even\s+though)'
        ]
        
        subjunctive_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in subjunctive_patterns)
        
        if subjunctive_count > len(sentences) * 0.2:
            score += 0.3
        elif subjunctive_count > len(sentences) * 0.1:
            score += 0.2
        elif subjunctive_count > 1:
            score += 0.1
        
        conditional_subjunctive = len(re.findall(r'\bif\s+\w+\s+(?:were|had|could|would|should)', text, re.IGNORECASE))
        if conditional_subjunctive > len(sentences) * 0.15:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_article_patterns(self, text: str) -> float:
        """Detecta patrones de uso de artículos típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        
        if len(words) < 30:
            return 0.0
        
        articles = ['a', 'an', 'the']
        article_count = sum(1 for word in words if word.lower() in articles)
        
        if article_count > len(words) * 0.12:
            score += 0.2
        elif article_count > len(words) * 0.10:
            score += 0.15
        elif article_count > len(words) * 0.08:
            score += 0.1
        
        article_patterns = [
            r'\b(?:the|a|an)\s+\w+\s+(?:the|a|an)\s+\w+',
            r'\b(?:the|a|an)\s+\w+\s+(?:the|a|an)\s+\w+\s+(?:the|a|an)',
            r'\b(?:the|a|an)\s+\w+\s+and\s+(?:the|a|an)\s+\w+'
        ]
        
        article_pattern_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in article_patterns)
        if article_pattern_count > len(words) * 0.02:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_preposition_patterns(self, text: str) -> float:
        """Analiza patrones de preposiciones típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        prepositions = ['in', 'on', 'at', 'by', 'for', 'with', 'from', 'to', 'of', 'about', 'into', 'onto', 'upon', 'within', 'without', 'through', 'during', 'under', 'over', 'above', 'below', 'between', 'among', 'beside', 'besides', 'beyond', 'across', 'against', 'along', 'around', 'behind', 'beneath', 'inside', 'outside', 'throughout', 'toward', 'towards', 'underneath', 'unlike', 'until', 'via', 'within']
        preposition_count = sum(1 for word in words if word.lower() in prepositions)
        
        if preposition_count > len(words) * 0.12:
            score += 0.3
        elif preposition_count > len(words) * 0.10:
            score += 0.2
        elif preposition_count > len(words) * 0.08:
            score += 0.1
        
        complex_prepositions = [
            r'\b(?:in\s+accordance\s+with|in\s+addition\s+to|in\s+comparison\s+to)',
            r'\b(?:in\s+conjunction\s+with|in\s+connection\s+with|in\s+contrast\s+to)',
            r'\b(?:in\s+relation\s+to|in\s+respect\s+of|in\s+terms\s+of)',
            r'\b(?:with\s+regard\s+to|with\s+respect\s+to|with\s+reference\s+to)',
            r'\b(?:on\s+behalf\s+of|on\s+account\s+of|on\s+top\s+of)'
        ]
        
        complex_prep_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in complex_prepositions)
        if complex_prep_count > len(sentences) * 0.15:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_conjunction_patterns(self, text: str) -> float:
        """Detecta patrones de conjunciones típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        coordinating_conjunctions = ['and', 'but', 'or', 'nor', 'for', 'so', 'yet']
        subordinating_conjunctions = ['although', 'though', 'because', 'since', 'if', 'unless', 'while', 'when', 'where', 'whereas', 'whether', 'until', 'before', 'after', 'as', 'so', 'that', 'once', 'provided', 'supposing']
        
        coord_count = sum(1 for word in words if word.lower() in coordinating_conjunctions)
        subord_count = sum(1 for word in words if word.lower() in subordinating_conjunctions)
        
        if (coord_count + subord_count) > len(words) * 0.08:
            score += 0.3
        elif (coord_count + subord_count) > len(words) * 0.06:
            score += 0.2
        elif (coord_count + subord_count) > len(words) * 0.04:
            score += 0.1
        
        conjunction_patterns = [
            r'\b(?:and\s+also|and\s+moreover|and\s+furthermore)',
            r'\b(?:but\s+also|but\s+however|but\s+nevertheless)',
            r'\b(?:not\s+only\s+but\s+also|both\s+and|either\s+or|neither\s+nor)'
        ]
        
        conj_pattern_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in conjunction_patterns)
        if conj_pattern_count > 1:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_determiner_patterns(self, text: str) -> float:
        """Analiza patrones de determinantes típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        
        if len(words) < 30:
            return 0.0
        
        determiners = ['this', 'that', 'these', 'those', 'all', 'each', 'every', 'both', 'either', 'neither', 'some', 'any', 'no', 'many', 'much', 'few', 'little', 'several', 'various', 'other', 'another', 'such', 'same', 'own']
        determiner_count = sum(1 for word in words if word.lower() in determiners)
        
        if determiner_count > len(words) * 0.05:
            score += 0.3
        elif determiner_count > len(words) * 0.04:
            score += 0.2
        elif determiner_count > len(words) * 0.03:
            score += 0.1
        
        determiner_patterns = [
            r'\b(?:this|that|these|those)\s+\w+\s+(?:this|that|these|those)',
            r'\b(?:all|each|every)\s+\w+\s+(?:all|each|every)',
            r'\b(?:some|any|no)\s+\w+\s+(?:some|any|no)'
        ]
        
        determiner_pattern_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in determiner_patterns)
        if determiner_pattern_count > 1:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_pronoun_reference_patterns(self, text: str) -> float:
        """Detecta patrones de referencia pronominal típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        pronoun_reference_patterns = [
            r'\b(?:this|that|these|those|it|they|them|their|its)\s+(?:is|are|was|were|has|have|had)',
            r'\b(?:which|who|whom|whose|where|when)\s+(?:is|are|was|were|has|have|had)',
            r'\b(?:he|she|it|they)\s+(?:is|are|was|were|has|have|had)',
            r'\b(?:this|that|these|those)\s+\w+\s+(?:which|that|who)'
        ]
        
        pronoun_ref_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in pronoun_reference_patterns)
        
        if pronoun_ref_count > len(sentences) * 0.3:
            score += 0.3
        elif pronoun_ref_count > len(sentences) * 0.2:
            score += 0.2
        elif pronoun_ref_count > len(sentences) * 0.1:
            score += 0.1
        
        ambiguous_references = len(re.findall(r'\b(?:this|that|these|those|it|they)\s+\w+', text, re.IGNORECASE))
        if ambiguous_references > len(sentences) * 0.25:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_adverb_patterns(self, text: str) -> float:
        """Analiza patrones de adverbios típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        common_adverbs = ['very', 'quite', 'rather', 'extremely', 'highly', 'particularly', 'especially', 'notably', 'remarkably', 'significantly', 'substantially', 'considerably', 'relatively', 'comparatively', 'fairly', 'somewhat', 'slightly', 'moderately', 'reasonably', 'sufficiently', 'adequately', 'appropriately', 'effectively', 'efficiently', 'successfully', 'carefully', 'clearly', 'obviously', 'evidently', 'apparently', 'presumably', 'probably', 'possibly', 'certainly', 'definitely', 'absolutely', 'completely', 'entirely', 'totally', 'fully', 'perfectly', 'exactly', 'precisely', 'accurately', 'correctly', 'properly', 'adequately']
        adverb_count = sum(1 for word in words if word.lower() in common_adverbs)
        
        if adverb_count > len(words) * 0.06:
            score += 0.3
        elif adverb_count > len(words) * 0.04:
            score += 0.2
        elif adverb_count > len(words) * 0.02:
            score += 0.1
        
        adverb_patterns = [
            r'\b(?:very|extremely|highly|particularly|especially)\s+\w+',
            r'\b(?:quite|rather|somewhat|fairly|relatively)\s+\w+',
            r'\b(?:clearly|obviously|evidently|apparently|presumably)',
            r'\b(?:certainly|definitely|absolutely|completely|entirely)'
        ]
        
        adverb_pattern_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in adverb_patterns)
        if adverb_pattern_count > len(sentences) * 0.3:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_adjective_patterns(self, text: str) -> float:
        """Detecta patrones de adjetivos típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        common_adjectives = ['important', 'crucial', 'essential', 'vital', 'significant', 'relevant', 'appropriate', 'suitable', 'effective', 'efficient', 'successful', 'comprehensive', 'thorough', 'detailed', 'extensive', 'considerable', 'substantial', 'notable', 'remarkable', 'noteworthy', 'prominent', 'distinct', 'unique', 'particular', 'specific', 'general', 'common', 'typical', 'usual', 'normal', 'standard', 'regular', 'ordinary', 'average', 'basic', 'fundamental', 'primary', 'main', 'major', 'minor', 'key', 'central', 'principal', 'chief', 'main', 'leading', 'primary']
        adjective_count = sum(1 for word in words if word.lower() in common_adjectives)
        
        if adjective_count > len(words) * 0.08:
            score += 0.3
        elif adjective_count > len(words) * 0.06:
            score += 0.2
        elif adjective_count > len(words) * 0.04:
            score += 0.1
        
        adjective_patterns = [
            r'\b(?:very|extremely|highly|particularly|especially)\s+\w+',
            r'\b(?:more|most|less|least)\s+\w+',
            r'\b(?:the\s+most|the\s+least|the\s+best|the\s+worst)',
            r'\b(?:important|crucial|essential|vital|significant)\s+\w+'
        ]
        
        adj_pattern_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in adjective_patterns)
        if adj_pattern_count > len(sentences) * 0.25:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_noun_patterns(self, text: str) -> float:
        """Analiza patrones de sustantivos típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        abstract_nouns = ['importance', 'significance', 'relevance', 'effectiveness', 'efficiency', 'success', 'achievement', 'accomplishment', 'progress', 'development', 'improvement', 'enhancement', 'advancement', 'growth', 'expansion', 'increase', 'decrease', 'reduction', 'decline', 'change', 'modification', 'alteration', 'variation', 'difference', 'similarity', 'comparison', 'contrast', 'relationship', 'connection', 'association', 'correlation', 'interaction', 'influence', 'impact', 'effect', 'consequence', 'result', 'outcome', 'conclusion', 'summary', 'overview', 'analysis', 'evaluation', 'assessment', 'examination', 'investigation', 'research', 'study', 'analysis', 'review']
        abstract_noun_count = sum(1 for word in words if word.lower() in abstract_nouns)
        
        if abstract_noun_count > len(words) * 0.05:
            score += 0.3
        elif abstract_noun_count > len(words) * 0.04:
            score += 0.2
        elif abstract_noun_count > len(words) * 0.03:
            score += 0.1
        
        noun_patterns = [
            r'\b(?:the|a|an)\s+\w+\s+of\s+\w+',
            r'\b(?:the|a|an)\s+\w+\s+and\s+\w+',
            r'\b(?:the|a|an)\s+\w+\s+or\s+\w+',
            r'\b(?:the|a|an)\s+\w+\s+,\s+\w+\s+and\s+\w+'
        ]
        
        noun_pattern_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in noun_patterns)
        if noun_pattern_count > len(sentences) * 0.2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_verb_patterns(self, text: str) -> float:
        """Detecta patrones de verbos típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        common_verbs = ['is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'get', 'gets', 'got', 'getting', 'make', 'makes', 'made', 'making', 'take', 'takes', 'took', 'taking', 'give', 'gives', 'gave', 'giving', 'go', 'goes', 'went', 'going', 'come', 'comes', 'came', 'coming', 'see', 'sees', 'saw', 'seeing', 'know', 'knows', 'knew', 'knowing', 'think', 'thinks', 'thought', 'thinking', 'say', 'says', 'said', 'saying', 'tell', 'tells', 'told', 'telling', 'ask', 'asks', 'asked', 'asking', 'try', 'tries', 'tried', 'trying', 'use', 'uses', 'used', 'using', 'find', 'finds', 'found', 'finding', 'want', 'wants', 'wanted', 'wanting', 'need', 'needs', 'needed', 'needing', 'work', 'works', 'worked', 'working', 'call', 'calls', 'called', 'calling', 'show', 'shows', 'showed', 'showing', 'move', 'moves', 'moved', 'moving', 'live', 'lives', 'lived', 'living', 'believe', 'believes', 'believed', 'believing', 'bring', 'brings', 'brought', 'bringing', 'happen', 'happens', 'happened', 'happening', 'write', 'writes', 'wrote', 'writing', 'provide', 'provides', 'provided', 'providing', 'sit', 'sits', 'sat', 'sitting', 'stand', 'stands', 'stood', 'standing', 'lose', 'loses', 'lost', 'losing', 'pay', 'pays', 'paid', 'paying', 'meet', 'meets', 'met', 'meeting', 'include', 'includes', 'included', 'including', 'continue', 'continues', 'continued', 'continuing', 'set', 'sets', 'setting', 'learn', 'learns', 'learned', 'learning', 'change', 'changes', 'changed', 'changing', 'lead', 'leads', 'led', 'leading', 'understand', 'understands', 'understood', 'understanding', 'watch', 'watches', 'watched', 'watching', 'follow', 'follows', 'followed', 'following', 'stop', 'stops', 'stopped', 'stopping', 'create', 'creates', 'created', 'creating', 'speak', 'speaks', 'spoke', 'speaking', 'read', 'reads', 'reading', 'allow', 'allows', 'allowed', 'allowing', 'add', 'adds', 'added', 'adding', 'spend', 'spends', 'spent', 'spending', 'grow', 'grows', 'grew', 'growing', 'open', 'opens', 'opened', 'opening', 'walk', 'walks', 'walked', 'walking', 'win', 'wins', 'won', 'winning', 'offer', 'offers', 'offered', 'offering', 'remember', 'remembers', 'remembered', 'remembering', 'love', 'loves', 'loved', 'loving', 'consider', 'considers', 'considered', 'considering', 'appear', 'appears', 'appeared', 'appearing', 'buy', 'buys', 'bought', 'buying', 'wait', 'waits', 'waited', 'waiting', 'serve', 'serves', 'served', 'serving', 'die', 'dies', 'died', 'dying', 'send', 'sends', 'sent', 'sending', 'build', 'builds', 'built', 'building', 'stay', 'stays', 'stayed', 'staying', 'fall', 'falls', 'fell', 'falling', 'cut', 'cuts', 'cutting', 'reach', 'reaches', 'reached', 'reaching', 'kill', 'kills', 'killed', 'killing', 'raise', 'raises', 'raised', 'raising', 'pass', 'passes', 'passed', 'passing', 'sell', 'sells', 'sold', 'selling', 'decide', 'decides', 'decided', 'deciding', 'return', 'returns', 'returned', 'returning', 'explain', 'explains', 'explained', 'explaining', 'develop', 'develops', 'developed', 'developing', 'carry', 'carries', 'carried', 'carrying', 'break', 'breaks', 'broke', 'breaking', 'receive', 'receives', 'received', 'receiving', 'agree', 'agrees', 'agreed', 'agreeing', 'support', 'supports', 'supported', 'supporting', 'hit', 'hits', 'hitting', 'produce', 'produces', 'produced', 'producing', 'eat', 'eats', 'ate', 'eating', 'cover', 'covers', 'covered', 'covering', 'catch', 'catches', 'caught', 'catching']
        verb_count = sum(1 for word in words if word.lower() in common_verbs)
        
        if verb_count > len(words) * 0.15:
            score += 0.2
        elif verb_count > len(words) * 0.12:
            score += 0.15
        elif verb_count > len(words) * 0.10:
            score += 0.1
        
        verb_patterns = [
            r'\b(?:is|are|was|were)\s+\w+ing',
            r'\b(?:has|have|had)\s+\w+ed',
            r'\b(?:will|would|can|could|should|must)\s+\w+',
            r'\b(?:to\s+be|to\s+have|to\s+do|to\s+get|to\s+make)'
        ]
        
        verb_pattern_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in verb_patterns)
        if verb_pattern_count > len(sentences) * 0.3:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_sentence_length_patterns(self, text: str) -> float:
        """Analiza patrones de longitud de oraciones típicos de IA - NUEVO"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 5:
            return 0.0
        
        sentence_lengths = [len(s.split()) for s in sentences]
        avg_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
        
        if avg_length > 25:
            score += 0.3
        elif avg_length > 20:
            score += 0.2
        elif avg_length > 15:
            score += 0.1
        
        length_variance = sum((l - avg_length) ** 2 for l in sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
        std_dev = length_variance ** 0.5
        
        if std_dev < 5:
            score += 0.3
        elif std_dev < 8:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_paragraph_structure_patterns(self, text: str) -> float:
        """Detecta patrones de estructura de párrafos típicos de IA - NUEVO"""
        score = 0.0
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if len(paragraphs) < 2:
            return 0.0
        
        paragraph_lengths = [len(p.split()) for p in paragraphs]
        avg_para_length = sum(paragraph_lengths) / len(paragraph_lengths) if paragraph_lengths else 0
        
        if avg_para_length > 150:
            score += 0.2
        elif avg_para_length > 100:
            score += 0.15
        
        para_variance = sum((l - avg_para_length) ** 2 for l in paragraph_lengths) / len(paragraph_lengths) if paragraph_lengths else 0
        para_std_dev = para_variance ** 0.5
        
        if para_std_dev < 30:
            score += 0.3
        elif para_std_dev < 50:
            score += 0.2
        
        first_sentence_patterns = [
            r'^(?:In|The|This|That|These|Those|It|There|Here)',
            r'^(?:To|For|With|By|From|At|On|In)',
            r'^(?:When|Where|Why|How|What|Who|Which)'
        ]
        
        first_sentence_matches = sum(1 for para in paragraphs if any(re.match(pattern, para, re.IGNORECASE) for pattern in first_sentence_patterns))
        if first_sentence_matches > len(paragraphs) * 0.7:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_punctuation_patterns(self, text: str) -> float:
        """Analiza patrones de puntuación típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        comma_count = text.count(',')
        semicolon_count = text.count(';')
        colon_count = text.count(':')
        dash_count = text.count('-') + text.count('—') + text.count('–')
        parenthesis_count = text.count('(') + text.count(')')
        
        total_punctuation = comma_count + semicolon_count + colon_count + dash_count + parenthesis_count
        
        if total_punctuation > len(words) * 0.15:
            score += 0.3
        elif total_punctuation > len(words) * 0.12:
            score += 0.2
        elif total_punctuation > len(words) * 0.10:
            score += 0.1
        
        if comma_count > len(sentences) * 2:
            score += 0.2
        
        if semicolon_count > len(sentences) * 0.3:
            score += 0.15
        
        return min(score, 1.0)
    
    def _detect_ai_capitalization_patterns(self, text: str) -> float:
        """Detecta patrones de capitalización típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        capitalized_words = sum(1 for word in words if word and word[0].isupper() and word.isalpha())
        
        if capitalized_words > len(words) * 0.15:
            score += 0.2
        elif capitalized_words > len(words) * 0.12:
            score += 0.15
        elif capitalized_words > len(words) * 0.10:
            score += 0.1
        
        all_caps_words = sum(1 for word in words if word.isupper() and len(word) > 1)
        if all_caps_words > len(words) * 0.02:
            score += 0.15
        
        proper_noun_patterns = [
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+',
            r'\b(?:the|The)\s+[A-Z][a-z]+\s+of\s+[A-Z][a-z]+',
            r'\b(?:of|in|on|at|by|for|with|from|to)\s+[A-Z][a-z]+'
        ]
        
        proper_noun_count = sum(len(re.findall(pattern, text)) for pattern in proper_noun_patterns)
        if proper_noun_count > len(sentences) * 0.3:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_word_frequency_patterns(self, text: str) -> float:
        """Analiza patrones de frecuencia de palabras típicos de IA - NUEVO"""
        score = 0.0
        words = [w.lower().strip('.,!?;:()[]{}"\'') for w in text.split()]
        words = [w for w in words if w]
        
        if len(words) < 50:
            return 0.0
        
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        total_words = len(words)
        unique_words = len(word_freq)
        type_token_ratio = unique_words / total_words if total_words > 0 else 0
        
        if type_token_ratio < 0.4:
            score += 0.3
        elif type_token_ratio < 0.5:
            score += 0.2
        elif type_token_ratio < 0.6:
            score += 0.1
        
        high_freq_words = [word for word, count in word_freq.items() if count > total_words * 0.05]
        if len(high_freq_words) > 5:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_phrase_repetition_patterns(self, text: str) -> float:
        """Detecta patrones de repetición de frases típicos de IA - NUEVO"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 5:
            return 0.0
        
        sentence_lower = [s.lower() for s in sentences]
        sentence_freq = {}
        for sentence in sentence_lower:
            sentence_freq[sentence] = sentence_freq.get(sentence, 0) + 1
        
        repeated_sentences = sum(1 for count in sentence_freq.values() if count > 1)
        if repeated_sentences > len(sentences) * 0.1:
            score += 0.3
        elif repeated_sentences > 0:
            score += 0.15
        
        phrase_patterns = [
            r'\b(?:it is|it\'s|this is|that is|these are|those are)',
            r'\b(?:in order to|so as to|with the aim of)',
            r'\b(?:it should be noted|it is important|it is worth noting)'
        ]
        
        phrase_counts = {}
        for pattern in phrase_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                phrase_counts[match.lower()] = phrase_counts.get(match.lower(), 0) + 1
        
        repeated_phrases = sum(1 for count in phrase_counts.values() if count > 2)
        if repeated_phrases > 2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_semantic_density_patterns(self, text: str) -> float:
        """Analiza patrones de densidad semántica típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 50:
            return 0.0
        
        content_words = []
        function_words = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'by', 'for', 'with', 'from', 'to', 'of', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'can', 'could', 'should', 'may', 'might', 'must', 'this', 'that', 'these', 'those', 'it', 'they', 'them', 'their', 'its', 'he', 'she', 'him', 'her', 'his', 'hers', 'we', 'us', 'our', 'ours', 'you', 'your', 'yours', 'i', 'me', 'my', 'mine']
        
        for word in words:
            word_lower = word.lower().strip('.,!?;:()[]{}"\'')
            if word_lower and word_lower not in function_words:
                content_words.append(word_lower)
        
        content_word_ratio = len(content_words) / len(words) if len(words) > 0 else 0
        
        if content_word_ratio < 0.5:
            score += 0.3
        elif content_word_ratio < 0.6:
            score += 0.2
        elif content_word_ratio < 0.7:
            score += 0.1
        
        unique_content_words = len(set(content_words))
        content_diversity = unique_content_words / len(content_words) if len(content_words) > 0 else 0
        
        if content_diversity < 0.5:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_coherence_markers_patterns(self, text: str) -> float:
        """Detecta patrones de marcadores de coherencia típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        coherence_markers = [
            r'\b(?:first|second|third|fourth|fifth|finally|lastly|last)',
            r'\b(?:in addition|furthermore|moreover|additionally|also|besides)',
            r'\b(?:however|nevertheless|nonetheless|on the other hand|conversely)',
            r'\b(?:therefore|thus|hence|consequently|as a result|accordingly)',
            r'\b(?:for example|for instance|such as|namely|specifically)',
            r'\b(?:in conclusion|to summarize|in summary|overall|ultimately)',
            r'\b(?:in other words|that is|that is to say|to put it differently)',
            r'\b(?:on the one hand|on the other hand)',
            r'\b(?:similarly|likewise|in the same way|equally)',
            r'\b(?:in contrast|on the contrary|conversely|by contrast)'
        ]
        
        coherence_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in coherence_markers)
        
        if coherence_count > len(sentences) * 0.4:
            score += 0.3
        elif coherence_count > len(sentences) * 0.3:
            score += 0.2
        elif coherence_count > len(sentences) * 0.2:
            score += 0.1
        
        transition_density = coherence_count / len(sentences) if len(sentences) > 0 else 0
        if transition_density > 0.5:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_lexical_sophistication_patterns(self, text: str) -> float:
        """Analiza patrones de sofisticación léxica típicos de IA - NUEVO"""
        score = 0.0
        words = [w.lower().strip('.,!?;:()[]{}"\'') for w in text.split()]
        words = [w for w in words if w and len(w) > 2]
        
        if len(words) < 50:
            return 0.0
        
        sophisticated_words = []
        common_words = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'by', 'for', 'with', 'from', 'to', 'of', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'can', 'could', 'should', 'may', 'might', 'must', 'this', 'that', 'these', 'those', 'it', 'they', 'them', 'their', 'its', 'he', 'she', 'him', 'her', 'his', 'hers', 'we', 'us', 'our', 'ours', 'you', 'your', 'yours', 'i', 'me', 'my', 'mine', 'get', 'got', 'make', 'made', 'take', 'took', 'go', 'went', 'come', 'came', 'see', 'saw', 'know', 'knew', 'think', 'thought', 'say', 'said', 'tell', 'told', 'ask', 'asked', 'try', 'tried', 'use', 'used', 'find', 'found', 'want', 'wanted', 'need', 'needed', 'work', 'worked', 'call', 'called', 'show', 'showed', 'move', 'moved', 'live', 'lived', 'believe', 'believed', 'bring', 'brought', 'happen', 'happened', 'write', 'wrote', 'provide', 'provided', 'sit', 'sat', 'stand', 'stood', 'lose', 'lost', 'pay', 'paid', 'meet', 'met', 'include', 'included', 'continue', 'continued', 'set', 'learn', 'learned', 'change', 'changed', 'lead', 'led', 'understand', 'understood', 'watch', 'watched', 'follow', 'followed', 'stop', 'stopped', 'create', 'created', 'speak', 'spoke', 'read', 'allow', 'allowed', 'add', 'added', 'spend', 'spent', 'grow', 'grew', 'open', 'opened', 'walk', 'walked', 'win', 'won', 'offer', 'offered', 'remember', 'remembered', 'love', 'loved', 'consider', 'considered', 'appear', 'appeared', 'buy', 'bought', 'wait', 'waited', 'serve', 'served', 'die', 'died', 'send', 'sent', 'build', 'built', 'stay', 'stayed', 'fall', 'fell', 'cut', 'reach', 'reached', 'kill', 'killed', 'raise', 'raised', 'pass', 'passed', 'sell', 'sold', 'decide', 'decided', 'return', 'returned', 'explain', 'explained', 'develop', 'developed', 'carry', 'carried', 'break', 'broke', 'receive', 'received', 'agree', 'agreed', 'support', 'supported', 'hit', 'produce', 'produced', 'eat', 'ate', 'cover', 'covered', 'catch', 'caught']
        
        for word in words:
            if word not in common_words and len(word) > 5:
                sophisticated_words.append(word)
        
        sophisticated_ratio = len(sophisticated_words) / len(words) if len(words) > 0 else 0
        
        if sophisticated_ratio > 0.3:
            score += 0.2
        elif sophisticated_ratio > 0.2:
            score += 0.15
        elif sophisticated_ratio < 0.1:
            score += 0.1
        
        avg_word_length = sum(len(w) for w in words) / len(words) if words else 0
        if avg_word_length > 6:
            score += 0.15
        elif avg_word_length < 4:
            score += 0.1
        
        return min(score, 1.0)
    
    def _detect_ai_formality_patterns(self, text: str) -> float:
        """Detecta patrones de formalidad típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        formal_markers = [
            r'\b(?:furthermore|moreover|additionally|consequently|therefore|thus|hence)',
            r'\b(?:it is important to note|it should be noted|it is worth noting|it is worth mentioning)',
            r'\b(?:in order to|so as to|with the aim of|with the purpose of)',
            r'\b(?:with regard to|with respect to|with reference to|in relation to)',
            r'\b(?:in accordance with|in compliance with|in conformity with)',
            r'\b(?:it can be observed|it can be seen|it can be noted|it can be concluded)',
            r'\b(?:it is evident that|it is clear that|it is obvious that|it is apparent that)',
            r'\b(?:it is necessary to|it is essential to|it is crucial to|it is vital to)',
            r'\b(?:it is recommended that|it is suggested that|it is advised that)',
            r'\b(?:it should be emphasized|it should be stressed|it should be highlighted)'
        ]
        
        formal_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in formal_markers)
        
        if formal_count > len(sentences) * 0.3:
            score += 0.3
        elif formal_count > len(sentences) * 0.2:
            score += 0.2
        elif formal_count > len(sentences) * 0.1:
            score += 0.1
        
        contractions = len(re.findall(r'\b(?:don\'t|doesn\'t|didn\'t|won\'t|wouldn\'t|couldn\'t|shouldn\'t|can\'t|cannot|isn\'t|aren\'t|wasn\'t|weren\'t|hasn\'t|haven\'t|hadn\'t|it\'s|that\'s|there\'s|here\'s|what\'s|who\'s|where\'s|when\'s|why\'s|how\'s|i\'m|you\'re|we\'re|they\'re|he\'s|she\'s|i\'ve|you\'ve|we\'ve|they\'ve|i\'d|you\'d|we\'d|they\'d|he\'d|she\'d)', text, re.IGNORECASE))
        if contractions == 0 and len(words) > 50:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_register_patterns(self, text: str) -> float:
        """Analiza patrones de registro típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        academic_register = [
            r'\b(?:according to|based on|in terms of|with regard to|with respect to)',
            r'\b(?:it has been|it can be|it should be|it must be|it will be)',
            r'\b(?:in order to|so as to|with the aim of|with the purpose of)',
            r'\b(?:it is important|it is necessary|it is essential|it is crucial)',
            r'\b(?:it should be noted|it is worth noting|it is worth mentioning)',
            r'\b(?:in addition to|furthermore|moreover|additionally)',
            r'\b(?:it can be concluded|it can be observed|it can be seen)',
            r'\b(?:it is evident|it is clear|it is obvious|it is apparent)'
        ]
        
        academic_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in academic_register)
        
        if academic_count > len(sentences) * 0.3:
            score += 0.3
        elif academic_count > len(sentences) * 0.2:
            score += 0.2
        elif academic_count > len(sentences) * 0.1:
            score += 0.1
        
        informal_markers = ['gonna', 'wanna', 'gotta', 'lemme', 'gimme', 'ya', 'yeah', 'yep', 'nope', 'nah', 'haha', 'lol', 'omg', 'wtf', 'btw', 'imo', 'fyi', 'tbh', 'idk', 'ikr', 'smh', 'tbh', 'irl', 'fml', 'rofl', 'lmao']
        informal_count = sum(1 for word in words if word.lower() in informal_markers)
        
        if informal_count == 0 and academic_count > 0:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_discourse_markers_patterns(self, text: str) -> float:
        """Detecta patrones de marcadores discursivos típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        discourse_markers = [
            r'\b(?:well|now|so|then|okay|ok|right|alright|anyway|anyways)',
            r'\b(?:you know|i mean|like|sort of|kind of|you see)',
            r'\b(?:actually|basically|literally|seriously|honestly|frankly)',
            r'\b(?:in fact|as a matter of fact|to be honest|to tell the truth)',
            r'\b(?:by the way|incidentally|speaking of|talking about)',
            r'\b(?:first of all|to begin with|to start with|for starters)',
            r'\b(?:in the end|at the end|finally|lastly|ultimately)',
            r'\b(?:in other words|that is|that is to say|i.e.|e.g.)',
            r'\b(?:for example|for instance|such as|namely|specifically)',
            r'\b(?:on the other hand|on the contrary|conversely|by contrast)'
        ]
        
        discourse_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in discourse_markers)
        
        if discourse_count > len(sentences) * 0.4:
            score += 0.3
        elif discourse_count > len(sentences) * 0.3:
            score += 0.2
        elif discourse_count > len(sentences) * 0.2:
            score += 0.1
        
        if discourse_count == 0 and len(sentences) > 10:
            score += 0.15
        
        return min(score, 1.0)
    
    def _analyze_ai_textual_cohesion_patterns(self, text: str) -> float:
        """Analiza patrones de cohesión textual típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 40:
            return 0.0
        
        cohesion_devices = [
            r'\b(?:this|that|these|those|it|they|them|their|its)\s+\w+',
            r'\b(?:the\s+former|the\s+latter|the\s+above|the\s+following)',
            r'\b(?:as\s+mentioned|as\s+stated|as\s+noted|as\s+discussed)',
            r'\b(?:in\s+this|in\s+that|in\s+these|in\s+those)',
            r'\b(?:of\s+this|of\s+that|of\s+these|of\s+those)'
        ]
        
        cohesion_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in cohesion_devices)
        
        if cohesion_count > len(sentences) * 0.4:
            score += 0.3
        elif cohesion_count > len(sentences) * 0.3:
            score += 0.2
        elif cohesion_count > len(sentences) * 0.2:
            score += 0.1
        
        lexical_chains = []
        for i in range(len(sentences) - 1):
            words1 = set(sentences[i].lower().split())
            words2 = set(sentences[i+1].lower().split())
            overlap = len(words1.intersection(words2))
            if overlap > 2:
                lexical_chains.append(overlap)
        
        if len(lexical_chains) > len(sentences) * 0.5:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_information_density_patterns(self, text: str) -> float:
        """Detecta patrones de densidad de información típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 50:
            return 0.0
        
        information_indicators = [
            r'\b(?:according to|based on|research shows|studies indicate|evidence suggests)',
            r'\b(?:it is estimated|it is calculated|it is determined|it is found)',
            r'\b(?:statistics show|data indicates|figures reveal|numbers suggest)',
            r'\b(?:research has|studies have|evidence has|data has)',
            r'\b(?:percentage|percent|ratio|proportion|rate|frequency)'
        ]
        
        info_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in information_indicators)
        
        if info_count > len(sentences) * 0.3:
            score += 0.3
        elif info_count > len(sentences) * 0.2:
            score += 0.2
        elif info_count > len(sentences) * 0.1:
            score += 0.1
        
        factual_phrases = ['according to', 'based on', 'research', 'study', 'studies', 'data', 'evidence', 'statistics', 'figures', 'numbers', 'percentage', 'percent', 'ratio', 'proportion', 'rate']
        factual_count = sum(1 for word in words if word.lower() in factual_phrases)
        
        if factual_count > len(words) * 0.03:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_hedging_density_patterns(self, text: str) -> float:
        """Analiza patrones de densidad de hedging típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        hedging_expressions = [
            r'\b(?:may|might|could|would|should|possibly|perhaps|maybe|probably|likely|unlikely)',
            r'\b(?:it seems|it appears|it would seem|it might seem|it could be)',
            r'\b(?:to some extent|in some way|in a sense|to a certain degree|in some cases)',
            r'\b(?:generally|usually|typically|often|sometimes|occasionally|rarely|seldom)',
            r'\b(?:suggest|indicate|imply|hint|point to|tend to|appear to|seem to)'
        ]
        
        hedging_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in hedging_expressions)
        
        if hedging_count > len(sentences) * 0.5:
            score += 0.3
        elif hedging_count > len(sentences) * 0.4:
            score += 0.2
        elif hedging_count > len(sentences) * 0.3:
            score += 0.1
        
        hedging_density = hedging_count / len(sentences) if len(sentences) > 0 else 0
        if hedging_density > 0.6:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_authorial_voice_patterns(self, text: str) -> float:
        """Detecta patrones de voz autorial típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        first_person_indicators = ['i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves']
        second_person_indicators = ['you', 'your', 'yours', 'yourself', 'yourselves']
        
        first_person_count = sum(1 for word in words if word.lower() in first_person_indicators)
        second_person_count = sum(1 for word in words if word.lower() in second_person_indicators)
        
        if first_person_count == 0 and second_person_count == 0 and len(words) > 100:
            score += 0.3
        elif first_person_count == 0 and len(words) > 50:
            score += 0.2
        
        personal_opinions = [
            r'\b(?:i think|i believe|i feel|i consider|i find|i see)',
            r'\b(?:in my opinion|from my perspective|in my view|to my mind)',
            r'\b(?:i would say|i would argue|i would suggest|i would recommend)'
        ]
        
        opinion_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in personal_opinions)
        if opinion_count == 0 and len(sentences) > 10:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_textual_variety_patterns(self, text: str) -> float:
        """Analiza patrones de variedad textual típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 50:
            return 0.0
        
        sentence_starters = [s.split()[0].lower() if s.split() else '' for s in sentences]
        unique_starters = len(set(sentence_starters))
        starter_variety = unique_starters / len(sentences) if len(sentences) > 0 else 0
        
        if starter_variety < 0.3:
            score += 0.3
        elif starter_variety < 0.5:
            score += 0.2
        elif starter_variety < 0.7:
            score += 0.1
        
        sentence_structures = []
        for sentence in sentences:
            if re.search(r'^[A-Z][^.!?]*\b(?:is|are|was|were)\b', sentence):
                sentence_structures.append('declarative')
            elif re.search(r'^[A-Z][^.!?]*\b(?:do|does|did|can|could|will|would)\b', sentence):
                sentence_structures.append('interrogative')
            elif re.search(r'^[A-Z][^.!?]*\b(?:let|may|should|must)\b', sentence):
                sentence_structures.append('imperative')
            else:
                sentence_structures.append('other')
        
        structure_variety = len(set(sentence_structures)) / len(sentence_structures) if sentence_structures else 0
        if structure_variety < 0.4:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_lexical_repetition_patterns(self, text: str) -> float:
        """Detecta patrones de repetición léxica típicos de IA - NUEVO"""
        score = 0.0
        words = [w.lower().strip('.,!?;:()[]{}"\'') for w in text.split()]
        words = [w for w in words if w and len(w) > 3]
        
        if len(words) < 40:
            return 0.0
        
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        total_words = len(words)
        unique_words = len(word_freq)
        
        repetition_ratio = (total_words - unique_words) / total_words if total_words > 0 else 0
        
        if repetition_ratio > 0.4:
            score += 0.3
        elif repetition_ratio > 0.3:
            score += 0.2
        elif repetition_ratio > 0.2:
            score += 0.1
        
        high_freq_words = [w for w, f in word_freq.items() if f > total_words * 0.05]
        if len(high_freq_words) > 3:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_syntactic_uniformity_patterns(self, text: str) -> float:
        """Analiza patrones de uniformidad sintáctica típicos de IA - NUEVO"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 5:
            return 0.0
        
        sentence_lengths = [len(s.split()) for s in sentences]
        avg_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
        
        length_variance = sum((l - avg_length) ** 2 for l in sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
        std_dev = length_variance ** 0.5
        
        if std_dev < 3:
            score += 0.3
        elif std_dev < 5:
            score += 0.2
        elif std_dev < 8:
            score += 0.1
        
        sentence_complexity = []
        for sentence in sentences:
            complexity = 0
            complexity += len(re.findall(r'\b(?:and|or|but|so|yet|for|nor)\b', sentence, re.IGNORECASE))
            complexity += len(re.findall(r'\b(?:which|that|who|whom|whose|where|when)\b', sentence, re.IGNORECASE))
            complexity += len(re.findall(r'\b(?:if|when|unless|provided|as long as)\b', sentence, re.IGNORECASE))
            sentence_complexity.append(complexity)
        
        complexity_variance = sum((c - sum(sentence_complexity) / len(sentence_complexity)) ** 2 for c in sentence_complexity) / len(sentence_complexity) if sentence_complexity else 0
        complexity_std = complexity_variance ** 0.5
        
        if complexity_std < 1:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_emotional_expression_patterns(self, text: str) -> float:
        """Detecta patrones de expresión emocional típicos de IA - NUEVO"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        emotional_words = ['happy', 'sad', 'angry', 'excited', 'frustrated', 'disappointed', 'surprised', 'shocked', 'amazed', 'confused', 'worried', 'anxious', 'nervous', 'calm', 'relaxed', 'stressed', 'tired', 'energetic', 'bored', 'interested', 'curious', 'proud', 'ashamed', 'embarrassed', 'guilty', 'jealous', 'envious', 'grateful', 'thankful', 'relieved', 'hopeful', 'hopeless', 'optimistic', 'pessimistic', 'confident', 'insecure', 'lonely', 'loved', 'hated', 'feeling', 'feel', 'felt', 'emotion', 'emotional', 'mood', 'moody']
        emotional_count = sum(1 for word in words if word.lower() in emotional_words)
        
        if emotional_count == 0 and len(words) > 100:
            score += 0.3
        elif emotional_count == 0 and len(words) > 50:
            score += 0.2
        elif emotional_count < len(words) * 0.01:
            score += 0.1
        
        emotional_expressions = [
            r'\b(?:i feel|i\'m feeling|i felt|i\'ve been feeling)',
            r'\b(?:it makes me|it made me|it makes us|it made us)',
            r'\b(?:i\'m so|i was so|i am so|i\'ve been so)',
            r'\b(?:this is|that is|it is)\s+(?:so|very|extremely|really|quite)\s+\w+'
        ]
        
        expression_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in emotional_expressions)
        if expression_count == 0 and len(sentences) > 10:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_contextual_ambiguity_patterns(self, text: str) -> float:
        """Analiza patrones de ambigüedad contextual típicos de IA - NUEVO V28"""
        score = 0.0
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 30:
            return 0.0
        
        # Patrones de ambigüedad contextual típicos de IA
        ambiguous_phrases = [
            r'\b(?:it depends|it may|it might|it could|possibly|perhaps|maybe)',
            r'\b(?:in some cases|in certain situations|under certain circumstances)',
            r'\b(?:depending on|varies|varies depending|can vary)',
            r'\b(?:generally speaking|broadly speaking|in general|typically)',
            r'\b(?:to some extent|in some way|in a way|somewhat)',
            r'\b(?:not necessarily|not always|not necessarily always)',
            r'\b(?:may or may not|could or could not|might or might not)'
        ]
        
        ambiguous_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in ambiguous_phrases)
        if ambiguous_count > len(sentences) * 0.15:
            score += 0.4
        elif ambiguous_count > len(sentences) * 0.10:
            score += 0.3
        elif ambiguous_count > len(sentences) * 0.05:
            score += 0.2
        
        # Análisis de frases que evitan compromiso
        hedging_phrases = [
            r'\b(?:it seems|it appears|it would seem|it would appear)',
            r'\b(?:one might|one could|one may|one would)',
            r'\b(?:it is possible|it is likely|it is unlikely|it is probable)',
            r'\b(?:there is a chance|there is a possibility|there might be)'
        ]
        
        hedging_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in hedging_phrases)
        if hedging_count > 2:
            score += 0.3
        
        return min(score, 1.0)
    
    def _detect_ai_lexical_richness_patterns(self, text: str) -> float:
        """Detecta patrones de riqueza léxica típicos de IA - NUEVO V28"""
        score = 0.0
        words = [w.lower().strip('.,!?;:()[]{}"\'') for w in text.split()]
        words = [w for w in words if w]
        
        if len(words) < 50:
            return 0.0
        
        # Calcular riqueza léxica (type-token ratio)
        unique_words = len(set(words))
        total_words = len(words)
        lexical_richness = unique_words / total_words if total_words > 0 else 0.0
        
        # IA tiende a tener riqueza léxica muy alta o muy baja
        if lexical_richness > 0.85:
            score += 0.3
        elif lexical_richness > 0.80:
            score += 0.2
        elif lexical_richness < 0.40:
            score += 0.3
        elif lexical_richness < 0.50:
            score += 0.2
        
        # Análisis de palabras poco comunes en contexto común
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        uncommon_words = [w for w in words if w not in common_words and len(w) > 4]
        uncommon_ratio = len(uncommon_words) / total_words if total_words > 0 else 0.0
        
        if uncommon_ratio > 0.60:
            score += 0.3
        elif uncommon_ratio > 0.50:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_syntactic_variation_patterns(self, text: str) -> float:
        """Analiza patrones de variación sintáctica típicos de IA - NUEVO V28"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 5:
            return 0.0
        
        # Análisis de variación en longitud de oraciones
        sentence_lengths = [len(s.split()) for s in sentences]
        if len(sentence_lengths) > 0:
            avg_length = sum(sentence_lengths) / len(sentence_lengths)
            std_dev = (sum((x - avg_length) ** 2 for x in sentence_lengths) / len(sentence_lengths)) ** 0.5
            
            # IA tiende a tener poca variación en longitud
            coefficient_of_variation = std_dev / avg_length if avg_length > 0 else 0.0
            if coefficient_of_variation < 0.3:
                score += 0.3
            elif coefficient_of_variation < 0.4:
                score += 0.2
        
        # Análisis de variación en estructura sintáctica
        structures = []
        for sentence in sentences:
            if re.search(r'^[A-Z][^.!?]*\b(?:is|are|was|were)\b', sentence):
                structures.append('declarative')
            elif re.search(r'^[A-Z][^.!?]*\b(?:do|does|did|can|could|will|would)\b', sentence):
                structures.append('interrogative')
            elif re.search(r'^[A-Z][^.!?]*\b(?:let|may|should|must)\b', sentence):
                structures.append('imperative')
            else:
                structures.append('other')
        
        if len(structures) > 0:
            unique_structures = len(set(structures))
            structure_variety = unique_structures / len(structures)
            
            # IA tiende a tener poca variedad estructural
            if structure_variety < 0.4:
                score += 0.3
            elif structure_variety < 0.5:
                score += 0.2
        
        # Análisis de patrones repetitivos de inicio de oración
        sentence_starts = [s.split()[0].lower() if s.split() else '' for s in sentences[:10]]
        start_counts = {}
        for start in sentence_starts:
            start_counts[start] = start_counts.get(start, 0) + 1
        
        max_repetition = max(start_counts.values()) if start_counts else 0
        if max_repetition > len(sentences) * 0.3:
            score += 0.3
        elif max_repetition > len(sentences) * 0.2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_discourse_coherence_patterns(self, text: str) -> float:
        """Detecta patrones de coherencia discursiva típicos de IA - NUEVO V28"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Marcadores discursivos típicos de IA
        discourse_markers = [
            r'\b(?:first|second|third|fourth|fifth|finally|lastly)',
            r'\b(?:in conclusion|to conclude|to summarize|in summary)',
            r'\b(?:furthermore|moreover|additionally|in addition)',
            r'\b(?:however|nevertheless|nonetheless|on the other hand)',
            r'\b(?:therefore|thus|hence|consequently|as a result)',
            r'\b(?:for example|for instance|such as|namely)',
            r'\b(?:in other words|that is|i\.e\.|e\.g\.)',
            r'\b(?:in fact|indeed|actually|as a matter of fact)'
        ]
        
        marker_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in discourse_markers)
        marker_density = marker_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA tiende a usar muchos marcadores discursivos
        if marker_density > 0.5:
            score += 0.4
        elif marker_density > 0.4:
            score += 0.3
        elif marker_density > 0.3:
            score += 0.2
        
        # Análisis de transiciones entre párrafos
        paragraphs = text.split('\n\n')
        transition_phrases = [
            r'\b(?:moving on|turning to|shifting to|now let\'s|let\'s now)',
            r'\b(?:another|additionally|furthermore|moreover|also)',
            r'\b(?:it is also|it should also|it must also|it can also)'
        ]
        
        transition_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in transition_phrases)
        if transition_count > len(paragraphs) * 0.8:
            score += 0.3
        elif transition_count > len(paragraphs) * 0.5:
            score += 0.2
        
        # Análisis de coherencia temática excesiva
        # IA tiende a mantener coherencia temática muy alta
        if len(sentences) > 5:
            # Contar palabras clave repetidas
            word_freq = {}
            for word in words:
                word_lower = word.lower().strip('.,!?;:()[]{}"\'')
                if len(word_lower) > 4:
                    word_freq[word_lower] = word_freq.get(word_lower, 0) + 1
            
            if word_freq:
                max_freq = max(word_freq.values())
                if max_freq > len(words) * 0.10:
                    score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_textual_rhythm_patterns(self, text: str) -> float:
        """Analiza patrones de ritmo textual típicos de IA - NUEVO V29"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 5 or len(words) < 30:
            return 0.0
        
        # Análisis de ritmo basado en longitud de oraciones
        sentence_lengths = [len(s.split()) for s in sentences]
        if len(sentence_lengths) > 0:
            # Calcular variación en ritmo
            avg_length = sum(sentence_lengths) / len(sentence_lengths)
            variance = sum((x - avg_length) ** 2 for x in sentence_lengths) / len(sentence_lengths)
            std_dev = variance ** 0.5
            
            # IA tiende a tener ritmo muy uniforme
            if std_dev < 3.0:
                score += 0.3
            elif std_dev < 5.0:
                score += 0.2
        
        # Análisis de patrones rítmicos repetitivos
        # Contar sílabas aproximadas por palabra (palabras largas = más sílabas)
        syllable_patterns = []
        for sentence in sentences[:10]:
            sentence_words = sentence.split()
            long_words = sum(1 for w in sentence_words if len(w) > 6)
            syllable_patterns.append(long_words / len(sentence_words) if sentence_words else 0)
        
        if len(syllable_patterns) > 0:
            pattern_variance = sum((x - sum(syllable_patterns) / len(syllable_patterns)) ** 2 for x in syllable_patterns) / len(syllable_patterns)
            if pattern_variance < 0.01:
                score += 0.3
        
        # Análisis de pausas y puntuación
        punctuation_density = len(re.findall(r'[,;:]', text)) / len(sentences) if len(sentences) > 0 else 0
        # IA tiende a usar puntuación de manera muy consistente
        if 0.3 <= punctuation_density <= 0.7:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_semantic_redundancy_patterns(self, text: str) -> float:
        """Detecta patrones de redundancia semántica típicos de IA - NUEVO V29"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Detectar frases redundantes
        redundant_patterns = [
            r'\b(?:free gift|free bonus|free offer)',
            r'\b(?:past history|future plans|end result|final outcome)',
            r'\b(?:completely finished|totally complete|absolutely certain)',
            r'\b(?:each and every|first and foremost|any and all)',
            r'\b(?:basic fundamentals|important essentials|necessary requirements)',
            r'\b(?:surrounding circumstances|general consensus|mutual cooperation)',
            r'\b(?:exact same|very unique|most unique|completely unique)',
            r'\b(?:repeat again|continue on|proceed forward|advance forward)'
        ]
        
        redundant_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in redundant_patterns)
        if redundant_count > 2:
            score += 0.4
        elif redundant_count > 1:
            score += 0.3
        
        # Detectar repetición de conceptos similares
        # Buscar sinónimos cercanos en la misma oración o oraciones adyacentes
        synonym_groups = [
            ['important', 'significant', 'crucial', 'vital', 'essential'],
            ['big', 'large', 'huge', 'enormous', 'massive'],
            ['small', 'tiny', 'little', 'miniature', 'minuscule'],
            ['good', 'great', 'excellent', 'wonderful', 'fantastic'],
            ['bad', 'terrible', 'awful', 'horrible', 'dreadful'],
            ['think', 'believe', 'consider', 'suppose', 'assume'],
            ['say', 'tell', 'speak', 'mention', 'state'],
            ['show', 'demonstrate', 'illustrate', 'reveal', 'display']
        ]
        
        for group in synonym_groups:
            found_in_sentence = []
            for i, sentence in enumerate(sentences):
                sentence_lower = sentence.lower()
                found = [word for word in group if word in sentence_lower]
                if len(found) > 1:
                    found_in_sentence.append(i)
            
            # Si se encuentran múltiples sinónimos en la misma oración
            if len(found_in_sentence) > len(sentences) * 0.2:
                score += 0.3
                break
        
        # Detectar tautologías
        tautology_patterns = [
            r'\b(?:it is what it is|what will be will be|it goes without saying)',
            r'\b(?:the fact of the matter|the truth of the matter)',
            r'\b(?:in my own personal opinion|in my own humble opinion)'
        ]
        
        tautology_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in tautology_patterns)
        if tautology_count > 0:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_lexical_sophistication_advanced(self, text: str) -> float:
        """Análisis avanzado de sofisticación léxica típica de IA - NUEVO V29"""
        score = 0.0
        words = [w.lower().strip('.,!?;:()[]{}"\'') for w in text.split()]
        words = [w for w in words if w and len(w) > 1]
        
        if len(words) < 50:
            return 0.0
        
        # Análisis de palabras sofisticadas vs comunes
        sophisticated_words = [
            'utilize', 'facilitate', 'implement', 'optimize', 'maximize', 'minimize',
            'comprehensive', 'systematic', 'methodical', 'analytical', 'strategic',
            'significant', 'substantial', 'considerable', 'notable', 'remarkable',
            'demonstrate', 'illustrate', 'exemplify', 'characterize', 'signify',
            'consequently', 'subsequently', 'furthermore', 'moreover', 'nevertheless',
            'paradigm', 'framework', 'methodology', 'approach', 'perspective',
            'facilitate', 'enhance', 'improve', 'optimize', 'streamline'
        ]
        
        sophisticated_count = sum(1 for w in words if w in sophisticated_words)
        sophisticated_ratio = sophisticated_count / len(words) if len(words) > 0 else 0.0
        
        # IA tiende a usar palabras sofisticadas de manera excesiva
        if sophisticated_ratio > 0.08:
            score += 0.4
        elif sophisticated_ratio > 0.06:
            score += 0.3
        elif sophisticated_ratio > 0.04:
            score += 0.2
        
        # Análisis de longitud promedio de palabras
        avg_word_length = sum(len(w) for w in words) / len(words) if len(words) > 0 else 0.0
        # IA tiende a usar palabras más largas en promedio
        if avg_word_length > 5.5:
            score += 0.3
        elif avg_word_length > 5.2:
            score += 0.2
        
        # Análisis de palabras técnicas o académicas
        academic_words = [
            'analysis', 'analysis', 'methodology', 'framework', 'paradigm',
            'hypothesis', 'theoretical', 'empirical', 'quantitative', 'qualitative',
            'correlation', 'causation', 'variable', 'parameter', 'criterion',
            'systematic', 'comprehensive', 'thorough', 'rigorous', 'methodical'
        ]
        
        academic_count = sum(1 for w in words if w in academic_words)
        academic_ratio = academic_count / len(words) if len(words) > 0 else 0.0
        
        if academic_ratio > 0.05:
            score += 0.3
        elif academic_ratio > 0.03:
            score += 0.2
        
        # Análisis de uso de latín/griego en palabras
        latin_greek_indicators = ['tion', 'sion', 'ology', 'ism', 'ity', 'ment', 'ance', 'ence']
        latin_greek_count = sum(1 for w in words if any(indicator in w for indicator in latin_greek_indicators) and len(w) > 6)
        latin_greek_ratio = latin_greek_count / len(words) if len(words) > 0 else 0.0
        
        if latin_greek_ratio > 0.15:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_pragmatic_markers_patterns(self, text: str) -> float:
        """Detecta patrones de marcadores pragmáticos típicos de IA - NUEVO V29"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Marcadores pragmáticos típicos de IA
        pragmatic_markers = [
            r'\b(?:it is important to note|it should be noted|it is worth noting)',
            r'\b(?:it is crucial to understand|it is essential to recognize)',
            r'\b(?:it is worth mentioning|it is worth pointing out)',
            r'\b(?:it is interesting to observe|it is noteworthy that)',
            r'\b(?:it is clear that|it is evident that|it is obvious that)',
            r'\b(?:it is important to remember|it is crucial to keep in mind)',
            r'\b(?:it is necessary to|it is essential to|it is vital to)',
            r'\b(?:one must|one should|one needs to|one ought to)',
            r'\b(?:it can be seen that|it can be observed that|it can be noted that)',
            r'\b(?:it should be emphasized|it should be highlighted|it should be stressed)',
            r'\b(?:it is worth considering|it is worth examining|it is worth exploring)',
            r'\b(?:it is important to understand|it is crucial to comprehend)'
        ]
        
        marker_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in pragmatic_markers)
        marker_density = marker_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA tiende a usar muchos marcadores pragmáticos
        if marker_density > 0.3:
            score += 0.4
        elif marker_density > 0.2:
            score += 0.3
        elif marker_density > 0.1:
            score += 0.2
        
        # Análisis de frases metalingüísticas
        metalinguistic_phrases = [
            r'\b(?:in other words|to put it another way|that is to say)',
            r'\b(?:as mentioned|as stated|as previously mentioned)',
            r'\b(?:to clarify|to elaborate|to explain further)',
            r'\b(?:in simple terms|in layman\'s terms|in plain language)',
            r'\b(?:to be more specific|to be more precise|to be exact)'
        ]
        
        metalinguistic_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in metalinguistic_phrases)
        if metalinguistic_count > 2:
            score += 0.3
        elif metalinguistic_count > 1:
            score += 0.2
        
        # Análisis de marcadores de organización del discurso
        organization_markers = [
            r'\b(?:first of all|second of all|third of all)',
            r'\b(?:to begin with|to start with|firstly|secondly|thirdly)',
            r'\b(?:last but not least|finally|in conclusion|to conclude)',
            r'\b(?:on the one hand|on the other hand)',
            r'\b(?:in the first place|in the second place)'
        ]
        
        org_marker_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in organization_markers)
        if org_marker_count > len(sentences) * 0.15:
            score += 0.3
        elif org_marker_count > len(sentences) * 0.10:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_conversational_patterns(self, text: str) -> float:
        """Analiza patrones conversacionales típicos de IA - NUEVO V30"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Patrones conversacionales típicos de IA
        conversational_patterns = [
            r'\b(?:let me|let\'s|i\'ll|i will|i can|i\'m going to)',
            r'\b(?:here\'s|here is|here are|below is|following is)',
            r'\b(?:i hope|i hope this|i hope that|hopefully)',
            r'\b(?:feel free to|don\'t hesitate to|please feel free)',
            r'\b(?:if you have|if you need|if you want|if you\'d like)',
            r'\b(?:i\'d be happy|i\'d be glad|i\'d love to|i\'m happy to)',
            r'\b(?:let me know|please let me know|feel free to let me know)',
            r'\b(?:i can help|i can assist|i can provide|i can offer)',
            r'\b(?:is there anything|do you have any|are there any)',
            r'\b(?:i\'m here to|i\'m available to|i\'m ready to)'
        ]
        
        conversational_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in conversational_patterns)
        conversational_density = conversational_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA tiende a usar muchos patrones conversacionales
        if conversational_density > 0.4:
            score += 0.4
        elif conversational_density > 0.3:
            score += 0.3
        elif conversational_density > 0.2:
            score += 0.2
        
        # Análisis de preguntas retóricas o de engagement
        question_patterns = [
            r'\b(?:have you ever|do you ever|did you ever)',
            r'\b(?:what do you think|what are your thoughts|what\'s your opinion)',
            r'\b(?:would you like|would you prefer|would you want)',
            r'\b(?:can you imagine|can you picture|can you see)'
        ]
        
        question_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in question_patterns)
        if question_count > 2:
            score += 0.3
        elif question_count > 1:
            score += 0.2
        
        # Análisis de frases de cierre conversacional
        closing_patterns = [
            r'\b(?:i hope this helps|hope this helps|hope that helps)',
            r'\b(?:let me know if|please let me know if|feel free to let me know if)',
            r'\b(?:if you have any|if you need any|if you want any)',
            r'\b(?:i\'m here if|i\'m available if|i\'m ready if)',
            r'\b(?:don\'t hesitate to|please don\'t hesitate to)'
        ]
        
        closing_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in closing_patterns)
        if closing_count > 1:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_metadiscourse_patterns(self, text: str) -> float:
        """Detecta patrones de metadiscurso típicos de IA - NUEVO V30"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Marcadores de metadiscurso típicos de IA
        metadiscourse_markers = [
            r'\b(?:in this section|in this part|in this chapter|in this paragraph)',
            r'\b(?:as we have seen|as we saw|as mentioned|as stated)',
            r'\b(?:we will now|we shall now|let us now|now we will)',
            r'\b(?:it should be noted|it must be noted|it is important to note)',
            r'\b(?:to summarize|to sum up|in summary|in conclusion)',
            r'\b(?:we have discussed|we discussed|we have seen|we saw)',
            r'\b(?:we will discuss|we shall discuss|we are going to discuss)',
            r'\b(?:as we will see|as we shall see|as we are about to see)',
            r'\b(?:let us consider|let us examine|let us look at)',
            r'\b(?:we can see that|we see that|we observe that)',
            r'\b(?:this section|this part|this chapter|this paragraph)',
            r'\b(?:the following|the next|the previous|the above)'
        ]
        
        metadiscourse_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in metadiscourse_markers)
        metadiscourse_density = metadiscourse_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA tiende a usar muchos marcadores de metadiscurso
        if metadiscourse_density > 0.3:
            score += 0.4
        elif metadiscourse_density > 0.2:
            score += 0.3
        elif metadiscourse_density > 0.1:
            score += 0.2
        
        # Análisis de referencias a la estructura del texto
        structure_references = [
            r'\b(?:above|below|previously|earlier|later)',
            r'\b(?:in the following|in the next|in the previous)',
            r'\b(?:as shown above|as mentioned above|as stated above)',
            r'\b(?:as we will see below|as shown below|as mentioned below)'
        ]
        
        structure_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in structure_references)
        if structure_count > len(sentences) * 0.15:
            score += 0.3
        elif structure_count > len(sentences) * 0.10:
            score += 0.2
        
        # Análisis de comentarios sobre el proceso de escritura
        process_comments = [
            r'\b(?:we will explore|we will examine|we will analyze)',
            r'\b(?:we have explored|we have examined|we have analyzed)',
            r'\b(?:let us turn to|let us move to|let us proceed to)',
            r'\b(?:we now turn to|we now move to|we now proceed to)'
        ]
        
        process_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in process_comments)
        if process_count > 2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_evidentiality_patterns(self, text: str) -> float:
        """Analiza patrones de evidencialidad típicos de IA - NUEVO V30"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Marcadores de evidencialidad típicos de IA
        evidentiality_markers = [
            r'\b(?:according to|based on|in accordance with|in line with)',
            r'\b(?:research shows|studies show|research indicates|studies indicate)',
            r'\b(?:it has been shown|it has been demonstrated|it has been proven)',
            r'\b(?:evidence suggests|evidence indicates|evidence shows)',
            r'\b(?:it is known that|it is well-known that|it is widely known that)',
            r'\b(?:it is believed that|it is thought that|it is considered that)',
            r'\b(?:it is said that|it is reported that|it is claimed that)',
            r'\b(?:it appears that|it seems that|it would seem that)',
            r'\b(?:it is likely that|it is probable that|it is possible that)',
            r'\b(?:it is clear that|it is evident that|it is obvious that)',
            r'\b(?:it has been found|it has been discovered|it has been revealed)',
            r'\b(?:it is generally accepted|it is widely accepted|it is commonly accepted)'
        ]
        
        evidentiality_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in evidentiality_markers)
        evidentiality_density = evidentiality_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA tiende a usar muchos marcadores de evidencialidad
        if evidentiality_density > 0.3:
            score += 0.4
        elif evidentiality_density > 0.2:
            score += 0.3
        elif evidentiality_density > 0.1:
            score += 0.2
        
        # Análisis de citas genéricas sin referencias específicas
        generic_citations = [
            r'\b(?:studies have shown|research has shown|studies have found)',
            r'\b(?:experts say|experts believe|experts suggest)',
            r'\b(?:many studies|numerous studies|various studies)',
            r'\b(?:some research|some studies|some evidence)',
            r'\b(?:it is generally believed|it is widely believed|it is commonly believed)'
        ]
        
        generic_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in generic_citations)
        if generic_count > 2:
            score += 0.3
        elif generic_count > 1:
            score += 0.2
        
        # Análisis de falta de evidencia específica
        # IA tiende a usar evidencialidad genérica sin referencias concretas
        specific_citations = re.findall(r'\b(?:\([A-Z][a-z]+ et al\.|\([A-Z][a-z]+, \d{4}|\[.*?\])', text)
        if evidentiality_count > 0 and len(specific_citations) == 0:
            score += 0.3
        elif evidentiality_count > 0 and len(specific_citations) < evidentiality_count * 0.3:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_engagement_patterns(self, text: str) -> float:
        """Detecta patrones de engagement típicos de IA - NUEVO V30"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Patrones de engagement típicos de IA
        engagement_patterns = [
            r'\b(?:you might|you may|you could|you can)',
            r'\b(?:you will find|you\'ll find|you will see|you\'ll see)',
            r'\b(?:you should|you ought to|you need to|you must)',
            r'\b(?:you can|you are able to|you have the ability to)',
            r'\b(?:if you|when you|as you|while you)',
            r'\b(?:your|yours|yourself|yourselves)',
            r'\b(?:you are|you\'re|you have|you\'ve)',
            r'\b(?:you will|you\'ll|you would|you\'d)',
            r'\b(?:you can also|you may also|you could also)',
            r'\b(?:you might want to|you may want to|you might consider)'
        ]
        
        engagement_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in engagement_patterns)
        engagement_density = engagement_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA tiende a usar muchos patrones de engagement
        if engagement_density > 0.5:
            score += 0.4
        elif engagement_density > 0.4:
            score += 0.3
        elif engagement_density > 0.3:
            score += 0.2
        
        # Análisis de uso excesivo de "you"
        you_count = len(re.findall(r'\byou\b', text, re.IGNORECASE))
        you_density = you_count / len(words) if len(words) > 0 else 0.0
        
        if you_density > 0.05:
            score += 0.3
        elif you_density > 0.03:
            score += 0.2
        
        # Análisis de imperativos directos
        imperative_patterns = [
            r'\b(?:remember that|keep in mind that|don\'t forget that)',
            r'\b(?:make sure|ensure that|be sure to)',
            r'\b(?:try to|attempt to|strive to)',
            r'\b(?:consider|think about|reflect on)',
            r'\b(?:take note|note that|observe that)'
        ]
        
        imperative_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in imperative_patterns)
        if imperative_count > 2:
            score += 0.2
        
        # Análisis de preguntas directas al lector
        direct_questions = re.findall(r'\b(?:have you|do you|are you|would you|could you|will you|can you)\b', text, re.IGNORECASE)
        if len(direct_questions) > len(sentences) * 0.15:
            score += 0.3
        elif len(direct_questions) > len(sentences) * 0.10:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_politeness_patterns(self, text: str) -> float:
        """Analiza patrones de cortesía típicos de IA - NUEVO V31"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Marcadores de cortesía típicos de IA
        politeness_markers = [
            r'\b(?:please|kindly|if you please|if you would)',
            r'\b(?:thank you|thanks|appreciate|grateful)',
            r'\b(?:i apologize|i\'m sorry|sorry for|apologies)',
            r'\b(?:excuse me|pardon me|forgive me|my apologies)',
            r'\b(?:i hope|i trust|i believe|i assume)',
            r'\b(?:would you|could you|might you|may you)',
            r'\b(?:if you don\'t mind|if it\'s not too much|if possible)',
            r'\b(?:i would appreciate|i\'d appreciate|i\'d be grateful)',
            r'\b(?:at your convenience|when convenient|when possible)',
            r'\b(?:i understand|i see|i realize|i acknowledge)'
        ]
        
        politeness_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in politeness_markers)
        politeness_density = politeness_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA tiende a usar muchos marcadores de cortesía
        if politeness_density > 0.4:
            score += 0.4
        elif politeness_density > 0.3:
            score += 0.3
        elif politeness_density > 0.2:
            score += 0.2
        
        # Análisis de frases de cortesía excesiva
        excessive_politeness = [
            r'\b(?:i would be most grateful|i would be very grateful|i would be extremely grateful)',
            r'\b(?:i sincerely hope|i truly hope|i genuinely hope)',
            r'\b(?:i deeply apologize|i sincerely apologize|i truly apologize)',
            r'\b(?:i cannot thank you enough|i cannot express my gratitude enough)',
            r'\b(?:i would be honored|i would be delighted|i would be thrilled)'
        ]
        
        excessive_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in excessive_politeness)
        if excessive_count > 1:
            score += 0.3
        
        # Análisis de uso de condicionales para cortesía
        conditional_politeness = re.findall(r'\b(?:would|could|might|may)\s+(?:you|i|we|they)\s+', text, re.IGNORECASE)
        if len(conditional_politeness) > len(sentences) * 0.3:
            score += 0.3
        elif len(conditional_politeness) > len(sentences) * 0.2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_formality_markers_patterns(self, text: str) -> float:
        """Detecta patrones de marcadores de formalidad típicos de IA - NUEVO V31"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Marcadores de formalidad típicos de IA
        formality_markers = [
            r'\b(?:it is|it\'s|it has been|it shall be)',
            r'\b(?:one must|one should|one ought to|one needs to)',
            r'\b(?:it should be noted|it must be noted|it ought to be noted)',
            r'\b(?:it is imperative|it is essential|it is crucial|it is vital)',
            r'\b(?:it is necessary|it is required|it is mandatory)',
            r'\b(?:it is important to|it is crucial to|it is essential to)',
            r'\b(?:it is worth|it is worthwhile|it is valuable)',
            r'\b(?:it is recommended|it is suggested|it is advised)',
            r'\b(?:it is considered|it is regarded|it is viewed)',
            r'\b(?:it is believed|it is thought|it is assumed)',
            r'\b(?:it is understood|it is recognized|it is acknowledged)',
            r'\b(?:it is expected|it is anticipated|it is predicted)'
        ]
        
        formality_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in formality_markers)
        formality_density = formality_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA tiende a usar muchos marcadores de formalidad
        if formality_density > 0.4:
            score += 0.4
        elif formality_density > 0.3:
            score += 0.3
        elif formality_density > 0.2:
            score += 0.2
        
        # Análisis de construcciones pasivas formales
        passive_formal = [
            r'\b(?:it has been|it had been|it will be|it would be)\s+\w+ed',
            r'\b(?:it is|it was|it will be|it would be)\s+\w+ed',
            r'\b(?:it can be|it could be|it may be|it might be)\s+\w+ed',
            r'\b(?:it should be|it must be|it ought to be)\s+\w+ed'
        ]
        
        passive_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in passive_formal)
        if passive_count > len(sentences) * 0.2:
            score += 0.3
        elif passive_count > len(sentences) * 0.1:
            score += 0.2
        
        # Análisis de uso de "one" en lugar de "you" o "we"
        one_usage = len(re.findall(r'\bone\s+(?:must|should|ought|needs|can|could|may|might|will|would)', text, re.IGNORECASE))
        if one_usage > len(sentences) * 0.15:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_hedging_advanced_patterns(self, text: str) -> float:
        """Análisis avanzado de patrones de hedging típicos de IA - NUEVO V31"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Marcadores avanzados de hedging típicos de IA
        hedging_advanced = [
            r'\b(?:it seems|it appears|it would seem|it would appear)',
            r'\b(?:it is possible|it is likely|it is probable|it is plausible)',
            r'\b(?:it may be|it might be|it could be|it can be)',
            r'\b(?:it is suggested|it is indicated|it is implied)',
            r'\b(?:it is thought|it is believed|it is considered|it is assumed)',
            r'\b(?:it is generally|it is typically|it is usually|it is commonly)',
            r'\b(?:it is often|it is frequently|it is sometimes|it is occasionally)',
            r'\b(?:it is somewhat|it is rather|it is quite|it is fairly)',
            r'\b(?:it is relatively|it is comparatively|it is relatively speaking)',
            r'\b(?:it is to some extent|it is to a certain extent|it is to some degree)',
            r'\b(?:it is not entirely|it is not completely|it is not fully)',
            r'\b(?:it is arguably|it is debatable|it is questionable)'
        ]
        
        hedging_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in hedging_advanced)
        hedging_density = hedging_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA tiende a usar muchos marcadores de hedging
        if hedging_density > 0.4:
            score += 0.4
        elif hedging_density > 0.3:
            score += 0.3
        elif hedging_density > 0.2:
            score += 0.2
        
        # Análisis de verbos modales de hedging
        modal_hedging = [
            r'\b(?:may|might|could|can)\s+(?:be|have|do|get|make|take|give|see|know|think|say|tell)',
            r'\b(?:would|should|ought to)\s+(?:be|have|do|get|make|take|give|see|know|think|say|tell)'
        ]
        
        modal_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in modal_hedging)
        if modal_count > len(sentences) * 0.3:
            score += 0.3
        elif modal_count > len(sentences) * 0.2:
            score += 0.2
        
        # Análisis de adverbios de hedging
        hedging_adverbs = ['possibly', 'probably', 'perhaps', 'maybe', 'presumably', 'supposedly', 'allegedly', 'reportedly', 'apparently', 'seemingly', 'arguably', 'potentially', 'theoretically', 'hypothetically']
        adverb_count = sum(1 for word in words if word.lower() in hedging_adverbs)
        if adverb_count > len(sentences) * 0.2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_assertiveness_patterns(self, text: str) -> float:
        """Detecta patrones de asertividad típicos de IA - NUEVO V31"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Patrones de asertividad típicos de IA
        assertiveness_patterns = [
            r'\b(?:it is clear that|it is evident that|it is obvious that|it is apparent that)',
            r'\b(?:it is certain that|it is definite that|it is sure that)',
            r'\b(?:it is undeniable that|it is indisputable that|it is unquestionable that)',
            r'\b(?:it is well-established|it is well-known|it is widely recognized)',
            r'\b(?:it is universally accepted|it is generally accepted|it is commonly accepted)',
            r'\b(?:it is a fact that|it is a truth that|it is a reality that)',
            r'\b(?:it is proven that|it is demonstrated that|it is established that)',
            r'\b(?:it is confirmed that|it is verified that|it is validated that)',
            r'\b(?:there is no doubt|there is no question|there is no denying)',
            r'\b(?:it cannot be denied|it cannot be disputed|it cannot be questioned)',
            r'\b(?:it must be|it has to be|it needs to be|it ought to be)',
            r'\b(?:it is imperative|it is essential|it is crucial|it is vital)'
        ]
        
        assertiveness_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in assertiveness_patterns)
        assertiveness_density = assertiveness_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA tiende a usar muchos patrones de asertividad
        if assertiveness_density > 0.3:
            score += 0.4
        elif assertiveness_density > 0.2:
            score += 0.3
        elif assertiveness_density > 0.1:
            score += 0.2
        
        # Análisis de uso de "must" y "should" de manera asertiva
        must_should = len(re.findall(r'\b(?:must|should|ought to|has to|needs to)\s+(?:be|have|do|get|make)', text, re.IGNORECASE))
        if must_should > len(sentences) * 0.2:
            score += 0.3
        elif must_should > len(sentences) * 0.1:
            score += 0.2
        
        # Análisis de afirmaciones categóricas
        categorical_claims = [
            r'\b(?:always|never|all|every|none|no one|nothing|nowhere)',
            r'\b(?:completely|totally|absolutely|entirely|fully|wholly)',
            r'\b(?:definitely|certainly|surely|undoubtedly|unquestionably)',
            r'\b(?:without exception|without doubt|without question)'
        ]
        
        categorical_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in categorical_claims)
        if categorical_count > len(sentences) * 0.3:
            score += 0.3
        elif categorical_count > len(sentences) * 0.2:
            score += 0.2
        
        # Análisis de falta de calificadores en afirmaciones fuertes
        strong_claims = re.findall(r'\b(?:is|are|was|were)\s+(?:always|never|all|every|completely|totally)', text, re.IGNORECASE)
        if len(strong_claims) > 2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_intertextuality_patterns(self, text: str) -> float:
        """Analiza patrones de intertextualidad típicos de IA - NUEVO V32"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Patrones de intertextualidad típicos de IA
        intertextuality_patterns = [
            r'\b(?:as mentioned|as stated|as noted|as discussed)',
            r'\b(?:as we have seen|as we saw|as we observed)',
            r'\b(?:as previously mentioned|as mentioned earlier|as stated above)',
            r'\b(?:as noted in|as stated in|as discussed in|as mentioned in)',
            r'\b(?:in the previous|in the earlier|in the above|in the following)',
            r'\b(?:as we will see|as we shall see|as we are about to see)',
            r'\b(?:as we have discussed|as we discussed|as we have seen)',
            r'\b(?:referring to|referring back to|going back to)',
            r'\b(?:in relation to|in connection with|in reference to)',
            r'\b(?:similar to|similar as|comparable to|comparable with)',
            r'\b(?:in contrast to|in contrast with|contrary to|unlike)',
            r'\b(?:building on|building upon|expanding on|expanding upon)'
        ]
        
        intertextuality_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in intertextuality_patterns)
        intertextuality_density = intertextuality_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA tiende a usar muchos patrones de intertextualidad
        if intertextuality_density > 0.3:
            score += 0.4
        elif intertextuality_density > 0.2:
            score += 0.3
        elif intertextuality_density > 0.1:
            score += 0.2
        
        # Análisis de referencias cruzadas excesivas
        cross_references = [
            r'\b(?:see above|see below|see earlier|see later)',
            r'\b(?:as shown above|as shown below|as demonstrated above)',
            r'\b(?:as indicated above|as indicated below|as noted above)',
            r'\b(?:as we saw|as we have seen|as we will see)'
        ]
        
        cross_ref_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in cross_references)
        if cross_ref_count > len(sentences) * 0.15:
            score += 0.3
        elif cross_ref_count > len(sentences) * 0.10:
            score += 0.2
        
        # Análisis de comparaciones y contrastes excesivos
        comparison_patterns = [
            r'\b(?:compared to|compared with|in comparison to|in comparison with)',
            r'\b(?:unlike|like|similar to|different from)',
            r'\b(?:in contrast to|in contrast with|contrary to|on the contrary)',
            r'\b(?:whereas|while|whilst|although|though)'
        ]
        
        comparison_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in comparison_patterns)
        if comparison_count > len(sentences) * 0.2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_citation_density_patterns(self, text: str) -> float:
        """Detecta patrones de densidad de citas típicos de IA - NUEVO V32"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Detectar citas formales (formato académico)
        formal_citations = re.findall(r'\([A-Z][a-z]+(?:\s+et\s+al\.)?(?:\s+and\s+[A-Z][a-z]+)?,?\s*\d{4}[a-z]?\)', text)
        formal_citation_count = len(formal_citations)
        
        # Detectar citas entre corchetes
        bracket_citations = re.findall(r'\[.*?\]', text)
        bracket_citation_count = len(bracket_citations)
        
        # Detectar referencias numéricas
        numeric_citations = re.findall(r'\[?\d+\]?', text)
        numeric_citation_count = len([c for c in numeric_citations if len(c) <= 4])
        
        total_citations = formal_citation_count + bracket_citation_count + numeric_citation_count
        citation_density = total_citations / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA puede tener densidad de citas muy alta o muy baja
        if citation_density > 1.5:
            score += 0.4
        elif citation_density > 1.0:
            score += 0.3
        elif citation_density == 0 and len(words) > 200:
            # Textos largos sin citas pueden ser IA
            score += 0.3
        elif citation_density == 0 and len(words) > 100:
            score += 0.2
        
        # Análisis de citas genéricas sin referencias específicas
        generic_citation_phrases = [
            r'\b(?:studies show|research shows|studies indicate|research indicates)',
            r'\b(?:experts say|experts believe|experts suggest|experts agree)',
            r'\b(?:it has been shown|it has been demonstrated|it has been proven)',
            r'\b(?:according to research|according to studies|according to experts)',
            r'\b(?:many studies|numerous studies|various studies|several studies)',
            r'\b(?:some research|some studies|some evidence|some data)'
        ]
        
        generic_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in generic_citation_phrases)
        if generic_count > 2 and formal_citation_count == 0:
            score += 0.4
        elif generic_count > 1 and formal_citation_count < 2:
            score += 0.3
        
        # Análisis de distribución de citas
        # IA tiende a tener citas distribuidas de manera muy uniforme
        if formal_citation_count > 5:
            citation_positions = [m.start() for m in re.finditer(r'\([A-Z][a-z]+(?:\s+et\s+al\.)?,?\s*\d{4}[a-z]?\)', text)]
            if len(citation_positions) > 1:
                text_length = len(text)
                intervals = []
                for i in range(len(citation_positions) - 1):
                    intervals.append(citation_positions[i+1] - citation_positions[i])
                
                if intervals:
                    avg_interval = sum(intervals) / len(intervals)
                    variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)
                    coefficient_of_variation = (variance ** 0.5) / avg_interval if avg_interval > 0 else 0
                    
                    # Variación muy baja indica distribución uniforme (típico de IA)
                    if coefficient_of_variation < 0.3:
                        score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_authority_claims_patterns(self, text: str) -> float:
        """Analiza patrones de afirmaciones de autoridad típicos de IA - NUEVO V32"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Patrones de afirmaciones de autoridad típicos de IA
        authority_claims = [
            r'\b(?:experts agree|experts say|experts believe|experts suggest)',
            r'\b(?:it is well-known|it is widely known|it is commonly known)',
            r'\b(?:it is established|it is proven|it is demonstrated|it is confirmed)',
            r'\b(?:it is recognized|it is acknowledged|it is accepted)',
            r'\b(?:it is generally accepted|it is widely accepted|it is commonly accepted)',
            r'\b(?:it is universally accepted|it is globally accepted)',
            r'\b(?:it is a fact|it is a truth|it is a reality)',
            r'\b(?:it is clear|it is evident|it is obvious|it is apparent)',
            r'\b(?:it is certain|it is definite|it is sure)',
            r'\b(?:it is undeniable|it is indisputable|it is unquestionable)',
            r'\b(?:research shows|studies show|research indicates|studies indicate)',
            r'\b(?:science shows|science indicates|science demonstrates)',
            r'\b(?:it has been proven|it has been demonstrated|it has been established)',
            r'\b(?:it has been shown|it has been confirmed|it has been verified)'
        ]
        
        authority_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in authority_claims)
        authority_density = authority_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA tiende a usar muchos patrones de afirmaciones de autoridad
        if authority_density > 0.3:
            score += 0.4
        elif authority_density > 0.2:
            score += 0.3
        elif authority_density > 0.1:
            score += 0.2
        
        # Análisis de afirmaciones de autoridad sin respaldo específico
        unsupported_authority = [
            r'\b(?:experts agree|experts say|experts believe)\s+(?:that|on|about)',
            r'\b(?:it is well-known|it is widely known)\s+(?:that|to)',
            r'\b(?:research shows|studies show)\s+(?:that|how|why)'
        ]
        
        unsupported_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in unsupported_authority)
        # Si hay muchas afirmaciones de autoridad pero pocas citas formales
        formal_citations = len(re.findall(r'\([A-Z][a-z]+(?:\s+et\s+al\.)?,?\s*\d{4}[a-z]?\)', text))
        if unsupported_count > 2 and formal_citations < unsupported_count:
            score += 0.3
        elif unsupported_count > 1 and formal_citations == 0:
            score += 0.2
        
        # Análisis de uso de "experts" sin especificar
        experts_usage = len(re.findall(r'\bexperts\s+(?:say|believe|suggest|agree|think|argue|claim)', text, re.IGNORECASE))
        if experts_usage > len(sentences) * 0.15:
            score += 0.3
        elif experts_usage > len(sentences) * 0.10:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_expertise_markers_patterns(self, text: str) -> float:
        """Detecta patrones de marcadores de expertise típicos de IA - NUEVO V32"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Marcadores de expertise típicos de IA
        expertise_markers = [
            r'\b(?:it is important to understand|it is crucial to understand|it is essential to understand)',
            r'\b(?:it is worth noting|it is worth mentioning|it is worth pointing out)',
            r'\b(?:it should be emphasized|it should be highlighted|it should be stressed)',
            r'\b(?:it is crucial to recognize|it is important to recognize|it is essential to recognize)',
            r'\b(?:it is necessary to|it is essential to|it is crucial to|it is vital to)',
            r'\b(?:one must understand|one should understand|one needs to understand)',
            r'\b(?:it is key to|it is critical to|it is fundamental to)',
            r'\b(?:it is imperative to|it is mandatory to|it is required to)',
            r'\b(?:it is advisable to|it is recommended to|it is suggested to)',
            r'\b(?:it is beneficial to|it is advantageous to|it is useful to)',
            r'\b(?:it is worth considering|it is worth examining|it is worth exploring)',
            r'\b(?:it is important to remember|it is crucial to remember|it is essential to remember)',
            r'\b(?:it is important to keep in mind|it is crucial to keep in mind)',
            r'\b(?:it is important to note|it is crucial to note|it is essential to note)'
        ]
        
        expertise_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in expertise_markers)
        expertise_density = expertise_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA tiende a usar muchos marcadores de expertise
        if expertise_density > 0.3:
            score += 0.4
        elif expertise_density > 0.2:
            score += 0.3
        elif expertise_density > 0.1:
            score += 0.2
        
        # Análisis de frases que indican conocimiento especializado
        specialized_knowledge = [
            r'\b(?:in the field of|in the domain of|in the area of)',
            r'\b(?:according to|based on|in accordance with)',
            r'\b(?:it is known in|it is recognized in|it is accepted in)',
            r'\b(?:within the context of|within the framework of|within the scope of)',
            r'\b(?:from a|from an|from the)\s+(?:perspective|viewpoint|standpoint|angle)',
            r'\b(?:in terms of|with regard to|with respect to|in relation to)'
        ]
        
        specialized_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in specialized_knowledge)
        if specialized_count > len(sentences) * 0.2:
            score += 0.3
        elif specialized_count > len(sentences) * 0.15:
            score += 0.2
        
        # Análisis de uso de terminología técnica sin explicación
        technical_terms = [
            r'\b(?:methodology|framework|paradigm|approach|strategy)',
            r'\b(?:systematic|comprehensive|thorough|rigorous|methodical)',
            r'\b(?:analysis|evaluation|assessment|examination|investigation)',
            r'\b(?:implementation|application|utilization|optimization)',
            r'\b(?:correlation|causation|variable|parameter|criterion)'
        ]
        
        technical_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in technical_terms)
        # Si hay muchos términos técnicos pero pocas explicaciones
        explanation_phrases = len(re.findall(r'\b(?:which means|that is|in other words|to put it simply)', text, re.IGNORECASE))
        if technical_count > 5 and explanation_phrases < technical_count * 0.2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_temporal_coherence_patterns(self, text: str) -> float:
        """Analiza patrones de coherencia temporal típicos de IA - NUEVO V33"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Marcadores temporales típicos de IA
        temporal_markers = [
            r'\b(?:first|second|third|fourth|fifth|finally|lastly|initially|subsequently)',
            r'\b(?:then|next|after|afterward|afterwards|later|meanwhile|simultaneously)',
            r'\b(?:previously|earlier|before|prior to|preceding|former)',
            r'\b(?:now|currently|presently|at present|at the moment)',
            r'\b(?:recently|lately|in recent|in the recent)',
            r'\b(?:eventually|ultimately|finally|in the end|at last)',
            r'\b(?:meanwhile|at the same time|simultaneously|concurrently)',
            r'\b(?:during|while|whilst|throughout|over the course of)',
            r'\b(?:since|until|till|from|to|until now|up to now)',
            r'\b(?:in the past|in the future|in the present|at that time)'
        ]
        
        temporal_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in temporal_markers)
        temporal_density = temporal_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA tiende a usar muchos marcadores temporales
        if temporal_density > 0.4:
            score += 0.4
        elif temporal_density > 0.3:
            score += 0.3
        elif temporal_density > 0.2:
            score += 0.2
        
        # Análisis de secuencia temporal excesivamente estructurada
        sequence_patterns = [
            r'\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)',
            r'\b(?:step one|step two|step three|step four|step five)',
            r'\b(?:firstly|secondly|thirdly|fourthly|fifthly)',
            r'\b(?:in the first place|in the second place|in the third place)'
        ]
        
        sequence_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in sequence_patterns)
        if sequence_count > len(sentences) * 0.2:
            score += 0.3
        elif sequence_count > len(sentences) * 0.15:
            score += 0.2
        
        # Análisis de coherencia temporal perfecta (sin saltos temporales)
        # IA tiende a tener secuencias temporales muy lineales
        time_verbs = ['was', 'were', 'had', 'has', 'have', 'will', 'would', 'is', 'are']
        time_verb_count = sum(1 for word in words if word.lower() in time_verbs)
        time_verb_density = time_verb_count / len(words) if len(words) > 0 else 0.0
        
        # Si hay muchos verbos temporales pero poca variación en tiempos
        if time_verb_density > 0.15:
            # Contar variación en tiempos verbales
            past_forms = len(re.findall(r'\b(?:was|were|had|did|went|came|said|told)', text, re.IGNORECASE))
            present_forms = len(re.findall(r'\b(?:is|are|am|do|does|go|come|say|tell)', text, re.IGNORECASE))
            future_forms = len(re.findall(r'\b(?:will|would|shall|should|going to)', text, re.IGNORECASE))
            
            total_tense_forms = past_forms + present_forms + future_forms
            if total_tense_forms > 0:
                max_tense = max(past_forms, present_forms, future_forms)
                tense_uniformity = max_tense / total_tense_forms
                # Si un tiempo domina mucho (uniformidad alta), puede ser IA
                if tense_uniformity > 0.7:
                    score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_causal_chain_patterns(self, text: str) -> float:
        """Detecta patrones de cadenas causales típicos de IA - NUEVO V33"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Marcadores causales típicos de IA
        causal_markers = [
            r'\b(?:because|since|as|due to|owing to|as a result of)',
            r'\b(?:therefore|thus|hence|consequently|as a result|so)',
            r'\b(?:if|when|whenever|provided that|assuming that)',
            r'\b(?:leads to|results in|causes|brings about|gives rise to)',
            r'\b(?:caused by|resulted from|stemmed from|arose from)',
            r'\b(?:this leads to|this results in|this causes|this brings about)',
            r'\b(?:which leads to|which results in|which causes|which brings about)',
            r'\b(?:in order to|so as to|for the purpose of|with the aim of)',
            r'\b(?:in order that|so that|such that|with the result that)',
            r'\b(?:as a consequence|as a consequence of|consequently|accordingly)'
        ]
        
        causal_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in causal_markers)
        causal_density = causal_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA tiende a usar muchos marcadores causales
        if causal_density > 0.4:
            score += 0.4
        elif causal_density > 0.3:
            score += 0.3
        elif causal_density > 0.2:
            score += 0.2
        
        # Análisis de cadenas causales lineales excesivas
        # IA tiende a crear cadenas causales muy lineales y estructuradas
        causal_chain_patterns = [
            r'\b(?:this|that|which|it)\s+(?:leads to|results in|causes|brings about)',
            r'\b(?:because|since|as)\s+(?:this|that|which|it)\s+(?:leads to|results in|causes)',
            r'\b(?:therefore|thus|hence|consequently)\s+(?:this|that|which|it)',
            r'\b(?:as a result|consequently|therefore)\s+(?:this|that|which|it)\s+(?:leads to|results in)'
        ]
        
        chain_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in causal_chain_patterns)
        if chain_count > 2:
            score += 0.3
        elif chain_count > 1:
            score += 0.2
        
        # Análisis de relaciones causales simplificadas
        # IA tiende a usar relaciones causales muy directas y simples
        simple_causal = len(re.findall(r'\b(?:because|since|as)\s+\w+\s+(?:is|are|was|were|has|have|will|would)', text, re.IGNORECASE))
        if simple_causal > len(sentences) * 0.2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_narrative_structure_patterns(self, text: str) -> float:
        """Analiza patrones de estructura narrativa típicos de IA - NUEVO V33"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        paragraphs = text.split('\n\n')
        
        if len(sentences) < 5 or len(words) < 50:
            return 0.0
        
        # Patrones de estructura narrativa típicos de IA
        narrative_structure = [
            r'\b(?:in the beginning|at the start|initially|first of all)',
            r'\b(?:in the middle|during|meanwhile|at this point)',
            r'\b(?:in the end|finally|ultimately|at last|conclusively)',
            r'\b(?:the story begins|the narrative starts|the tale begins)',
            r'\b(?:the story continues|the narrative continues|the tale continues)',
            r'\b(?:the story ends|the narrative ends|the tale ends)',
            r'\b(?:to begin with|to start with|to commence with)',
            r'\b(?:to conclude|to finish|to end|to wrap up)',
            r'\b(?:in conclusion|in summary|in closing|to sum up)'
        ]
        
        narrative_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in narrative_structure)
        narrative_density = narrative_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA tiende a usar muchos marcadores de estructura narrativa
        if narrative_density > 0.2:
            score += 0.3
        elif narrative_density > 0.15:
            score += 0.2
        
        # Análisis de estructura de tres actos (típico de IA)
        three_act_patterns = [
            r'\b(?:introduction|introducing|introduce|first act)',
            r'\b(?:development|developing|develop|second act|middle act)',
            r'\b(?:conclusion|concluding|conclude|third act|final act)',
            r'\b(?:setup|rising action|climax|falling action|resolution)'
        ]
        
        three_act_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in three_act_patterns)
        if three_act_count > 2:
            score += 0.3
        
        # Análisis de uniformidad en longitud de párrafos
        if len(paragraphs) > 3:
            paragraph_lengths = [len(p.split()) for p in paragraphs if p.strip()]
            if len(paragraph_lengths) > 0:
                avg_length = sum(paragraph_lengths) / len(paragraph_lengths)
                variance = sum((x - avg_length) ** 2 for x in paragraph_lengths) / len(paragraph_lengths)
                std_dev = variance ** 0.5
                coefficient_of_variation = std_dev / avg_length if avg_length > 0 else 0.0
                
                # IA tiende a tener párrafos de longitud muy uniforme
                if coefficient_of_variation < 0.3:
                    score += 0.3
                elif coefficient_of_variation < 0.4:
                    score += 0.2
        
        # Análisis de transiciones narrativas excesivamente estructuradas
        narrative_transitions = [
            r'\b(?:moving forward|moving on|turning to|shifting to)',
            r'\b(?:now let\'s|let\'s now|let us now|we now turn)',
            r'\b(?:next|then|after that|following that|subsequently)',
            r'\b(?:in the next section|in the following section|in the subsequent section)'
        ]
        
        transition_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in narrative_transitions)
        if transition_count > len(paragraphs) * 0.5:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_argumentation_patterns(self, text: str) -> float:
        """Detecta patrones de argumentación típicos de IA - NUEVO V33"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Marcadores de argumentación típicos de IA
        argumentation_markers = [
            r'\b(?:first|second|third|fourth|fifth|furthermore|moreover|additionally)',
            r'\b(?:on the one hand|on the other hand|however|nevertheless|nonetheless)',
            r'\b(?:for example|for instance|such as|namely|specifically)',
            r'\b(?:in other words|that is|i\.e\.|e\.g\.|to put it another way)',
            r'\b(?:therefore|thus|hence|consequently|as a result|so)',
            r'\b(?:it follows that|it can be concluded that|it can be inferred that)',
            r'\b(?:this suggests|this indicates|this implies|this shows)',
            r'\b(?:it is clear that|it is evident that|it is obvious that)',
            r'\b(?:it can be argued that|it can be said that|it can be claimed that)',
            r'\b(?:in support of|in favor of|against|opposed to)',
            r'\b(?:to support|to argue|to claim|to suggest|to propose)',
            r'\b(?:in conclusion|to conclude|to sum up|in summary)'
        ]
        
        argumentation_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in argumentation_markers)
        argumentation_density = argumentation_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA tiende a usar muchos marcadores de argumentación
        if argumentation_density > 0.4:
            score += 0.4
        elif argumentation_density > 0.3:
            score += 0.3
        elif argumentation_density > 0.2:
            score += 0.2
        
        # Análisis de estructura argumentativa excesivamente formal
        formal_argumentation = [
            r'\b(?:premise|conclusion|argument|reasoning|logic)',
            r'\b(?:it follows that|it can be concluded that|it can be inferred that)',
            r'\b(?:therefore|thus|hence|consequently|as a result)',
            r'\b(?:given that|assuming that|provided that|supposing that)',
            r'\b(?:we can conclude|we can infer|we can deduce|we can reason)'
        ]
        
        formal_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in formal_argumentation)
        if formal_count > len(sentences) * 0.15:
            score += 0.3
        elif formal_count > len(sentences) * 0.10:
            score += 0.2
        
        # Análisis de contraargumentos estructurados
        counterargument_patterns = [
            r'\b(?:on the one hand|on the other hand|however|nevertheless)',
            r'\b(?:some may argue|one might argue|it could be argued)',
            r'\b(?:while|whereas|although|though|even though)',
            r'\b(?:despite|in spite of|notwithstanding|regardless of)',
            r'\b(?:contrary to|opposite to|in contrast to|in contrast with)'
        ]
        
        counterargument_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in counterargument_patterns)
        # IA tiende a usar contraargumentos de manera muy estructurada
        if counterargument_count > len(sentences) * 0.2:
            score += 0.2
        
        # Análisis de ejemplos estructurados
        example_patterns = [
            r'\b(?:for example|for instance|such as|namely|specifically)',
            r'\b(?:to illustrate|to demonstrate|to show|to exemplify)',
            r'\b(?:consider|take|look at|examine|observe)',
            r'\b(?:an example|one example|another example|a case in point)'
        ]
        
        example_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in example_patterns)
        if example_count > len(sentences) * 0.25:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_lexical_consistency_patterns(self, text: str) -> float:
        """Analiza patrones de consistencia léxica típicos de IA - NUEVO V34"""
        score = 0.0
        words = [w.lower().strip('.,!?;:()[]{}"\'') for w in text.split()]
        words = [w for w in words if w and len(w) > 1]
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 50 or len(sentences) < 3:
            return 0.0
        
        # Análisis de consistencia en elección de palabras
        # IA tiende a usar las mismas palabras de manera muy consistente
        word_frequency = {}
        for word in words:
            if len(word) > 3:  # Solo palabras significativas
                word_frequency[word] = word_frequency.get(word, 0) + 1
        
        # Calcular índice de repetición de palabras clave
        total_significant_words = sum(1 for w in words if len(w) > 3)
        if total_significant_words > 0:
            repeated_words = sum(1 for count in word_frequency.values() if count > 2)
            repetition_ratio = repeated_words / len(word_frequency) if word_frequency else 0.0
            
            # IA tiende a tener alta consistencia (muchas repeticiones)
            if repetition_ratio > 0.4:
                score += 0.3
            elif repetition_ratio > 0.3:
                score += 0.2
        
        # Análisis de sinónimos - IA tiende a usar el mismo sinónimo consistentemente
        synonym_groups = {
            'important': ['important', 'significant', 'crucial', 'vital', 'essential'],
            'big': ['big', 'large', 'huge', 'enormous', 'massive'],
            'good': ['good', 'great', 'excellent', 'wonderful', 'fantastic'],
            'show': ['show', 'demonstrate', 'illustrate', 'reveal', 'display'],
            'think': ['think', 'believe', 'consider', 'suppose', 'assume']
        }
        
        for group_name, synonyms in synonym_groups.items():
            found_synonyms = [word for word in words if word in synonyms]
            if len(found_synonyms) > 2:
                # Si se usa principalmente un sinónimo (consistencia alta)
                synonym_counts = {}
                for syn in found_synonyms:
                    synonym_counts[syn] = synonym_counts.get(syn, 0) + 1
                
                if synonym_counts:
                    max_count = max(synonym_counts.values())
                    consistency = max_count / len(found_synonyms)
                    if consistency > 0.7:
                        score += 0.2
                        break
        
        # Análisis de variación en vocabulario técnico
        technical_terms = ['methodology', 'framework', 'analysis', 'evaluation', 'assessment', 'implementation', 'optimization']
        technical_found = [w for w in words if w in technical_terms]
        if len(technical_found) > 3:
            # Si se usan términos técnicos de manera muy consistente
            technical_counts = {}
            for term in technical_found:
                technical_counts[term] = technical_counts.get(term, 0) + 1
            
            if technical_counts:
                max_tech = max(technical_counts.values())
                if max_tech > len(technical_found) * 0.5:
                    score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_semantic_field_patterns(self, text: str) -> float:
        """Detecta patrones de campos semánticos típicos de IA - NUEVO V34"""
        score = 0.0
        words = [w.lower().strip('.,!?;:()[]{}"\'') for w in text.split()]
        words = [w for w in words if w and len(w) > 1]
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 50 or len(sentences) < 3:
            return 0.0
        
        # Definir campos semánticos comunes
        semantic_fields = {
            'academic': ['research', 'study', 'analysis', 'methodology', 'framework', 'theory', 'hypothesis', 'empirical', 'quantitative', 'qualitative'],
            'business': ['strategy', 'management', 'organization', 'efficiency', 'productivity', 'optimization', 'implementation', 'stakeholder', 'revenue', 'profit'],
            'technology': ['system', 'platform', 'application', 'software', 'hardware', 'algorithm', 'database', 'network', 'interface', 'protocol'],
            'science': ['experiment', 'observation', 'hypothesis', 'theory', 'data', 'evidence', 'analysis', 'conclusion', 'method', 'result'],
            'general_knowledge': ['important', 'significant', 'crucial', 'essential', 'fundamental', 'key', 'main', 'primary', 'principal', 'major']
        }
        
        # Contar palabras de cada campo semántico
        field_counts = {}
        for field, terms in semantic_fields.items():
            count = sum(1 for word in words if word in terms)
            if count > 0:
                field_counts[field] = count
        
        # IA tiende a usar campos semánticos muy concentrados
        if field_counts:
            total_field_words = sum(field_counts.values())
            max_field = max(field_counts.values())
            concentration = max_field / total_field_words if total_field_words > 0 else 0.0
            
            # Alta concentración en un campo semántico
            if concentration > 0.6:
                score += 0.4
            elif concentration > 0.5:
                score += 0.3
            elif concentration > 0.4:
                score += 0.2
        
        # Análisis de transiciones entre campos semánticos
        # IA tiende a tener transiciones muy abruptas o muy suaves
        if len(sentences) > 5:
            sentence_fields = []
            for sentence in sentences[:10]:
                sentence_words = [w.lower().strip('.,!?;:()[]{}"\'') for w in sentence.split()]
                sentence_field_counts = {}
                for field, terms in semantic_fields.items():
                    count = sum(1 for word in sentence_words if word in terms)
                    if count > 0:
                        sentence_field_counts[field] = count
                
                if sentence_field_counts:
                    dominant_field = max(sentence_field_counts.items(), key=lambda x: x[1])[0]
                    sentence_fields.append(dominant_field)
            
            # Si hay muchas transiciones entre campos (inconsistencia)
            if len(sentence_fields) > 1:
                transitions = sum(1 for i in range(len(sentence_fields) - 1) if sentence_fields[i] != sentence_fields[i+1])
                transition_ratio = transitions / (len(sentence_fields) - 1) if len(sentence_fields) > 1 else 0.0
                
                # Muchas transiciones o muy pocas (ambos pueden ser IA)
                if transition_ratio > 0.7 or transition_ratio < 0.2:
                    score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_register_consistency_patterns(self, text: str) -> float:
        """Analiza patrones de consistencia de registro típicos de IA - NUEVO V34"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Marcadores de registro formal
        formal_markers = [
            r'\b(?:it is|it has been|it shall be|it will be)',
            r'\b(?:one must|one should|one ought to|one needs to)',
            r'\b(?:it should be noted|it must be noted|it ought to be noted)',
            r'\b(?:it is imperative|it is essential|it is crucial)',
            r'\b(?:according to|based on|in accordance with)',
            r'\b(?:furthermore|moreover|additionally|in addition)',
            r'\b(?:therefore|thus|hence|consequently|as a result)'
        ]
        
        # Marcadores de registro informal
        informal_markers = [
            r'\b(?:gonna|wanna|gotta|lemme|dunno)',
            r'\b(?:yeah|yep|nope|nah|uh|um|er)',
            r'\b(?:cool|awesome|great|nice|sweet)',
            r'\b(?:like|you know|I mean|sort of|kind of)',
            r'\b(?:gonna|wanna|gotta|lemme|dunno)',
            r'\b(?:can\'t|won\'t|don\'t|isn\'t|aren\'t)',
            r'\b(?:it\'s|that\'s|what\'s|there\'s|here\'s)'
        ]
        
        formal_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in formal_markers)
        informal_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in informal_markers)
        
        total_markers = formal_count + informal_count
        if total_markers > 0:
            # IA tiende a tener registro muy consistente (todo formal o todo informal)
            if formal_count > 0 and informal_count == 0:
                # Todo formal
                if formal_count > len(sentences) * 0.3:
                    score += 0.3
            elif informal_count > 0 and formal_count == 0:
                # Todo informal
                if informal_count > len(sentences) * 0.3:
                    score += 0.3
            elif formal_count > 0 and informal_count > 0:
                # Mezcla - puede ser humano, pero si es muy poca mezcla, puede ser IA
                mix_ratio = min(formal_count, informal_count) / max(formal_count, informal_count)
                if mix_ratio < 0.2:  # Muy poca mezcla
                    score += 0.2
        
        # Análisis de contracciones
        contractions = len(re.findall(r'\b\w+\'[a-z]+\b', text, re.IGNORECASE))
        no_contractions = len(re.findall(r'\b(?:it is|it has|it will|it would|do not|does not|did not|can not|cannot|will not|would not|should not|could not|must not)', text, re.IGNORECASE))
        
        # IA puede tener consistencia en uso de contracciones
        if contractions > 0 and no_contractions == 0:
            # Solo contracciones
            if contractions > len(sentences) * 0.2:
                score += 0.2
        elif no_contractions > 0 and contractions == 0:
            # Sin contracciones
            if no_contractions > len(sentences) * 0.2:
                score += 0.2
        
        # Análisis de vocabulario - consistencia en nivel de formalidad
        formal_words = ['utilize', 'facilitate', 'implement', 'optimize', 'comprehensive', 'systematic', 'methodical']
        informal_words = ['use', 'help', 'do', 'make', 'big', 'good', 'nice', 'cool']
        
        formal_word_count = sum(1 for word in words if word.lower() in formal_words)
        informal_word_count = sum(1 for word in words if word.lower() in informal_words)
        
        if formal_word_count > 0 and informal_word_count == 0:
            if formal_word_count > len(words) * 0.05:
                score += 0.2
        elif informal_word_count > 0 and formal_word_count == 0:
            if informal_word_count > len(words) * 0.05:
                score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_stylistic_uniformity_patterns(self, text: str) -> float:
        """Detecta patrones de uniformidad estilística típicos de IA - NUEVO V34"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 5 or len(words) < 50:
            return 0.0
        
        # Análisis de uniformidad en longitud de oraciones
        sentence_lengths = [len(s.split()) for s in sentences]
        if len(sentence_lengths) > 0:
            avg_length = sum(sentence_lengths) / len(sentence_lengths)
            variance = sum((x - avg_length) ** 2 for x in sentence_lengths) / len(sentence_lengths)
            std_dev = variance ** 0.5
            coefficient_of_variation = std_dev / avg_length if avg_length > 0 else 0.0
            
            # IA tiende a tener uniformidad alta (baja variación)
            if coefficient_of_variation < 0.3:
                score += 0.3
            elif coefficient_of_variation < 0.4:
                score += 0.2
        
        # Análisis de uniformidad en estructura de oraciones
        sentence_structures = []
        for sentence in sentences:
            # Clasificar estructura básica
            if re.search(r'^[A-Z][^.!?]*\b(?:is|are|was|were)\b', sentence):
                sentence_structures.append('declarative_be')
            elif re.search(r'^[A-Z][^.!?]*\b(?:have|has|had)\b', sentence):
                sentence_structures.append('declarative_have')
            elif re.search(r'^[A-Z][^.!?]*\b(?:do|does|did|can|could|will|would)\b', sentence):
                sentence_structures.append('declarative_modal')
            elif re.search(r'^[A-Z][^.!?]*\?', sentence):
                sentence_structures.append('interrogative')
            else:
                sentence_structures.append('other')
        
        if len(sentence_structures) > 0:
            structure_counts = {}
            for struct in sentence_structures:
                structure_counts[struct] = structure_counts.get(struct, 0) + 1
            
            # Si una estructura domina mucho (uniformidad alta)
            max_structure_count = max(structure_counts.values())
            uniformity_ratio = max_structure_count / len(sentence_structures)
            
            if uniformity_ratio > 0.6:
                score += 0.3
            elif uniformity_ratio > 0.5:
                score += 0.2
        
        # Análisis de uniformidad en puntuación
        punctuation_types = {
            'comma': len(re.findall(r',', text)),
            'semicolon': len(re.findall(r';', text)),
            'colon': len(re.findall(r':', text)),
            'dash': len(re.findall(r'[-—]', text)),
            'parentheses': len(re.findall(r'[()]', text))
        }
        
        total_punctuation = sum(punctuation_types.values())
        if total_punctuation > 0:
            # Si un tipo de puntuación domina mucho
            max_punct_type = max(punctuation_types.values())
            punct_uniformity = max_punct_type / total_punctuation
            
            if punct_uniformity > 0.7:
                score += 0.2
        
        # Análisis de uniformidad en inicio de oraciones
        sentence_starts = []
        for sentence in sentences[:15]:
            words_in_sentence = sentence.split()
            if words_in_sentence:
                first_word = words_in_sentence[0].lower().strip('.,!?;:()[]{}"\'')
                if first_word:
                    sentence_starts.append(first_word)
        
        if len(sentence_starts) > 0:
            start_counts = {}
            for start in sentence_starts:
                start_counts[start] = start_counts.get(start, 0) + 1
            
            # Si hay mucha repetición en inicio de oraciones
            max_start_count = max(start_counts.values())
            start_uniformity = max_start_count / len(sentence_starts)
            
            if start_uniformity > 0.3:
                score += 0.3
            elif start_uniformity > 0.2:
                score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_phraseology_patterns(self, text: str) -> float:
        """Analiza patrones de fraseología típicos de IA - NUEVO V35"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Fraseologías típicas de IA (frases hechas comunes)
        ai_phraseologies = [
            r'\b(?:it is important to note|it should be noted|it must be noted)',
            r'\b(?:it is worth mentioning|it is worth noting|it is worth pointing out)',
            r'\b(?:it is clear that|it is evident that|it is obvious that)',
            r'\b(?:it is important to understand|it is crucial to understand|it is essential to understand)',
            r'\b(?:it is necessary to|it is essential to|it is crucial to|it is vital to)',
            r'\b(?:it is worth considering|it is worth examining|it is worth exploring)',
            r'\b(?:it can be seen that|it can be observed that|it can be noted that)',
            r'\b(?:it should be emphasized|it should be highlighted|it should be stressed)',
            r'\b(?:it is important to remember|it is crucial to remember|it is essential to remember)',
            r'\b(?:it is important to keep in mind|it is crucial to keep in mind)',
            r'\b(?:in other words|that is|i\.e\.|e\.g\.|to put it another way)',
            r'\b(?:for example|for instance|such as|namely|specifically)',
            r'\b(?:in conclusion|to conclude|to summarize|in summary)',
            r'\b(?:furthermore|moreover|additionally|in addition)',
            r'\b(?:however|nevertheless|nonetheless|on the other hand)',
            r'\b(?:therefore|thus|hence|consequently|as a result)'
        ]
        
        phraseology_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in ai_phraseologies)
        phraseology_density = phraseology_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA tiende a usar muchas fraseologías comunes
        if phraseology_density > 0.4:
            score += 0.4
        elif phraseology_density > 0.3:
            score += 0.3
        elif phraseology_density > 0.2:
            score += 0.2
        
        # Análisis de repetición de fraseologías
        # IA tiende a repetir las mismas fraseologías
        phraseology_instances = []
        for pattern in ai_phraseologies:
            matches = re.findall(pattern, text, re.IGNORECASE)
            phraseology_instances.extend(matches)
        
        if len(phraseology_instances) > 0:
            unique_phrases = len(set(phraseology_instances))
            repetition_ratio = 1 - (unique_phrases / len(phraseology_instances)) if len(phraseology_instances) > 0 else 0.0
            
            if repetition_ratio > 0.3:
                score += 0.3
            elif repetition_ratio > 0.2:
                score += 0.2
        
        # Análisis de fraseologías académicas excesivas
        academic_phrases = [
            r'\b(?:according to|based on|in accordance with|in line with)',
            r'\b(?:it has been shown|it has been demonstrated|it has been proven)',
            r'\b(?:research shows|studies show|research indicates|studies indicate)',
            r'\b(?:it is well-known|it is widely known|it is commonly known)',
            r'\b(?:it is established|it is proven|it is demonstrated|it is confirmed)'
        ]
        
        academic_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in academic_phrases)
        if academic_count > len(sentences) * 0.2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_collocation_patterns(self, text: str) -> float:
        """Detecta patrones de colocaciones típicos de IA - NUEVO V35"""
        score = 0.0
        words = [w.lower().strip('.,!?;:()[]{}"\'') for w in text.split()]
        words = [w for w in words if w and len(w) > 1]
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 50 or len(sentences) < 3:
            return 0.0
        
        # Colocaciones comunes en inglés (palabras que frecuentemente aparecen juntas)
        common_collocations = [
            ('make', 'decision'), ('take', 'action'), ('give', 'example'),
            ('play', 'role'), ('have', 'impact'), ('take', 'place'),
            ('make', 'sense'), ('have', 'effect'), ('take', 'care'),
            ('make', 'difference'), ('have', 'influence'), ('take', 'advantage'),
            ('provide', 'information'), ('conduct', 'research'), ('carry', 'out'),
            ('put', 'forward'), ('bring', 'about'), ('come', 'up'),
            ('deal', 'with'), ('look', 'into'), ('focus', 'on'),
            ('depend', 'on'), ('rely', 'on'), ('base', 'on'),
            ('lead', 'to'), ('result', 'in'), ('contribute', 'to')
        ]
        
        collocation_count = 0
        for word1, word2 in common_collocations:
            # Buscar patrones donde word1 y word2 aparecen cerca (dentro de 3 palabras)
            pattern = rf'\b{word1}\b(?:\s+\w+)?(?:\s+\w+)?\s+\b{word2}\b'
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            collocation_count += matches
        
        collocation_density = collocation_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA puede usar colocaciones de manera muy predecible o muy poco natural
        if collocation_density > 0.5:
            score += 0.3
        elif collocation_density > 0.3:
            score += 0.2
        elif collocation_density == 0 and len(words) > 200:
            # Textos largos sin colocaciones comunes pueden ser IA
            score += 0.2
        
        # Análisis de colocaciones académicas excesivas
        academic_collocations = [
            ('conduct', 'analysis'), ('carry', 'out'), ('perform', 'study'),
            ('undertake', 'research'), ('engage', 'in'), ('participate', 'in'),
            ('contribute', 'to'), ('lead', 'to'), ('result', 'from'),
            ('based', 'on'), ('according', 'to'), ('in', 'accordance')
        ]
        
        academic_colloc_count = 0
        for word1, word2 in academic_collocations:
            pattern = rf'\b{word1}\b(?:\s+\w+)?(?:\s+\w+)?\s+\b{word2}\b'
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            academic_colloc_count += matches
        
        if academic_colloc_count > len(sentences) * 0.2:
            score += 0.3
        elif academic_colloc_count > len(sentences) * 0.15:
            score += 0.2
        
        # Análisis de colocaciones poco naturales o forzadas
        # IA a veces crea colocaciones que suenan poco naturales
        unnatural_patterns = [
            r'\b(?:make|do|have|take)\s+(?:a|an|the)\s+\w+ing',
            r'\b(?:provide|give|offer)\s+(?:a|an|the)\s+\w+ing',
            r'\b(?:conduct|perform|carry)\s+(?:a|an|the)\s+\w+ing'
        ]
        
        unnatural_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in unnatural_patterns)
        if unnatural_count > 2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_idiomatic_patterns(self, text: str) -> float:
        """Analiza patrones idiomáticos típicos de IA - NUEVO V35"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Modismos comunes en inglés
        common_idioms = [
            r'\b(?:break the ice|break the mold|break new ground)',
            r'\b(?:hit the nail on the head|hit the mark|hit home)',
            r'\b(?:piece of cake|walk in the park|easy as pie)',
            r'\b(?:once in a blue moon|once in a lifetime|once and for all)',
            r'\b(?:the ball is in your court|the tables have turned)',
            r'\b(?:cost an arm and a leg|pay through the nose)',
            r'\b(?:barking up the wrong tree|beating around the bush)',
            r'\b(?:burn the midnight oil|burn bridges|burn out)',
            r'\b(?:call it a day|call the shots|call off)',
            r'\b(?:cut corners|cut to the chase|cut it out)',
            r'\b(?:get the ball rolling|get a grip|get over it)',
            r'\b(?:go the extra mile|go with the flow|go back on)',
            r'\b(?:keep an eye on|keep in mind|keep up)',
            r'\b(?:let the cat out of the bag|let off steam)',
            r'\b(?:make ends meet|make a long story short|make up)',
            r'\b(?:on cloud nine|on the same page|on thin ice)',
            r'\b(?:pull strings|pull through|pull off)',
            r'\b(?:put your foot down|put up with|put off)',
            r'\b(?:see eye to eye|see the light|see through)',
            r'\b(?:take it with a grain of salt|take the plunge|take off)'
        ]
        
        idiom_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in common_idioms)
        idiom_density = idiom_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA puede usar modismos de manera excesiva o muy poco
        if idiom_density > 0.3:
            score += 0.3
        elif idiom_density == 0 and len(words) > 200:
            # Textos largos sin modismos pueden ser IA
            score += 0.2
        
        # Análisis de uso incorrecto o literal de modismos
        # IA a veces usa modismos de manera incorrecta o demasiado literal
        literal_usage = [
            r'\b(?:break the ice)\s+(?:literally|actually|really)',
            r'\b(?:hit the nail)\s+(?:literally|actually|really)',
            r'\b(?:piece of cake)\s+(?:literally|actually|really)'
        ]
        
        literal_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in literal_usage)
        if literal_count > 0:
            score += 0.3
        
        # Análisis de modismos académicos o formales excesivos
        formal_idioms = [
            r'\b(?:in light of|in view of|in the context of)',
            r'\b(?:with regard to|with respect to|in relation to)',
            r'\b(?:for the purpose of|with the aim of|with a view to)',
            r'\b(?:in order to|so as to|with the intention of)',
            r'\b(?:by means of|by way of|through the use of)',
            r'\b(?:in terms of|in the case of|in the event of)',
            r'\b(?:on the basis of|on the grounds of|on account of)',
            r'\b(?:with reference to|in reference to|with regard to)'
        ]
        
        formal_idiom_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in formal_idioms)
        if formal_idiom_count > len(sentences) * 0.3:
            score += 0.3
        elif formal_idiom_count > len(sentences) * 0.2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_cultural_references_patterns(self, text: str) -> float:
        """Detecta patrones de referencias culturales típicos de IA - NUEVO V35"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Referencias culturales comunes (eventos, lugares, personas, etc.)
        cultural_references = [
            r'\b(?:world war|world war i|world war ii|wwi|wwii)',
            r'\b(?:united states|usa|america|american)',
            r'\b(?:united kingdom|uk|britain|british)',
            r'\b(?:european union|eu|europe)',
            r'\b(?:christmas|thanksgiving|easter|halloween)',
            r'\b(?:hollywood|broadway|silicon valley|wall street)',
            r'\b(?:nobel prize|oscar|grammy|pulitzer)',
            r'\b(?:olympic|olympics|world cup|super bowl)',
            r'\b(?:shakespeare|einstein|darwin|newton)',
            r'\b(?:renaissance|enlightenment|industrial revolution)',
            r'\b(?:great depression|cold war|civil war)',
            r'\b(?:9/11|september 11|twin towers)',
            r'\b(?:brexit|trump|biden|obama)',
            r'\b(?:facebook|google|apple|microsoft|amazon)',
            r'\b(?:iphone|android|windows|mac|linux)'
        ]
        
        cultural_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in cultural_references)
        cultural_density = cultural_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA puede usar referencias culturales de manera excesiva o muy genérica
        if cultural_density > 0.3:
            score += 0.3
        elif cultural_density == 0 and len(words) > 300:
            # Textos largos sin referencias culturales pueden ser IA
            score += 0.2
        
        # Análisis de referencias culturales genéricas o estereotípicas
        generic_cultural = [
            r'\b(?:as everyone knows|as is well-known|as is common knowledge)',
            r'\b(?:in western culture|in eastern culture|in modern society)',
            r'\b(?:in today\'s world|in the modern world|in contemporary society)',
            r'\b(?:throughout history|in human history|since ancient times)',
            r'\b(?:in many cultures|across cultures|in different cultures)',
            r'\b(?:in american culture|in european culture|in asian culture)'
        ]
        
        generic_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in generic_cultural)
        if generic_count > 2:
            score += 0.3
        elif generic_count > 1:
            score += 0.2
        
        # Análisis de referencias culturales desactualizadas o incorrectas
        outdated_references = [
            r'\b(?:y2k|millennium bug|dot-com bubble)',
            r'\b(?:my space|aol|yahoo|nokia)',
            r'\b(?:vhs|betamax|cassette tape|floppy disk)'
        ]
        
        outdated_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in outdated_references)
        if outdated_count > 0:
            score += 0.2
        
        # Análisis de falta de referencias culturales específicas o personales
        # IA tiende a evitar referencias culturales muy específicas o personales
        personal_cultural = [
            r'\b(?:in my country|in my culture|where i\'m from)',
            r'\b(?:in my experience|from my perspective|in my opinion)',
            r'\b(?:i remember|i recall|i\'ve seen|i\'ve heard)',
            r'\b(?:growing up|when i was|back when|in my day)'
        ]
        
        personal_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in personal_cultural)
        if personal_count == 0 and len(words) > 200:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_metaphorical_patterns(self, text: str) -> float:
        """Analiza patrones metafóricos típicos de IA - NUEVO V36"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Patrones metafóricos comunes
        metaphorical_patterns = [
            r'\b(?:like|as|as if|as though)\s+\w+',
            r'\b(?:is|are|was|were)\s+(?:like|as)\s+\w+',
            r'\b(?:metaphor|metaphorical|metaphorically)',
            r'\b(?:symbol|symbolic|symbolize|symbolism)',
            r'\b(?:represent|represents|representation|represents)',
            r'\b(?:embody|embodies|embodiment)',
            r'\b(?:personify|personifies|personification)',
            r'\b(?:compare|compares|comparison|compared to|compared with)',
            r'\b(?:similar to|similar as|akin to|reminiscent of)',
            r'\b(?:resemble|resembles|resemblance)',
            r'\b(?:mirror|mirrors|reflection|reflects)',
            r'\b(?:echo|echoes|echoing)',
            r'\b(?:evoke|evokes|evocation|evocative)',
            r'\b(?:conjure|conjures|conjuring)',
            r'\b(?:suggest|suggests|suggestion|suggestive)'
        ]
        
        metaphorical_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in metaphorical_patterns)
        metaphorical_density = metaphorical_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA puede usar metáforas de manera excesiva o muy poco
        if metaphorical_density > 0.4:
            score += 0.3
        elif metaphorical_density == 0 and len(words) > 200:
            # Textos largos sin metáforas pueden ser IA
            score += 0.2
        
        # Análisis de metáforas comunes o clichés
        cliche_metaphors = [
            r'\b(?:tip of the iceberg|light at the end of the tunnel|needle in a haystack)',
            r'\b(?:elephant in the room|elephant in the corner|800-pound gorilla)',
            r'\b(?:can of worms|pandora\'s box|opening pandora\'s box)',
            r'\b(?:double-edged sword|two sides of the coin|both sides of the coin)',
            r'\b(?:walking on eggshells|walking on thin ice|treading carefully)',
            r'\b(?:beating a dead horse|flogging a dead horse|beating around the bush)',
            r'\b(?:elephant in the room|elephant in the corner)',
            r'\b(?:elephant in the room|elephant in the corner)'
        ]
        
        cliche_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in cliche_metaphors)
        if cliche_count > 2:
            score += 0.3
        elif cliche_count > 1:
            score += 0.2
        
        # Análisis de metáforas forzadas o poco naturales
        forced_metaphors = [
            r'\b(?:is like|are like|was like|were like)\s+(?:a|an|the)\s+\w+\s+(?:that|which)',
            r'\b(?:can be compared to|can be likened to|can be equated with)',
            r'\b(?:serves as a metaphor|acts as a metaphor|functions as a metaphor)',
            r'\b(?:metaphorically speaking|in metaphorical terms|using a metaphor)'
        ]
        
        forced_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in forced_metaphors)
        if forced_count > 1:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_analogical_patterns(self, text: str) -> float:
        """Detecta patrones analógicos típicos de IA - NUEVO V36"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Patrones analógicos comunes
        analogical_patterns = [
            r'\b(?:analogy|analogous|analogously)',
            r'\b(?:similar to|similar as|akin to|reminiscent of)',
            r'\b(?:compare|compares|comparison|compared to|compared with)',
            r'\b(?:just as|just like|much like|very like)',
            r'\b(?:in the same way|in a similar way|in a similar manner)',
            r'\b(?:likewise|similarly|correspondingly|equally)',
            r'\b(?:parallel|parallels|parallelism|parallel to)',
            r'\b(?:equivalent|equivalents|equivalence|equivalent to)',
            r'\b(?:correspond|corresponds|correspondence|corresponds to)',
            r'\b(?:mirror|mirrors|mirroring|mirrors that)',
            r'\b(?:echo|echoes|echoing|echoes that)',
            r'\b(?:resemble|resembles|resemblance|resembles that)',
            r'\b(?:draw a parallel|draw parallels|drawing a parallel)',
            r'\b(?:make an analogy|make analogies|making an analogy)',
            r'\b(?:by analogy|through analogy|using an analogy)'
        ]
        
        analogical_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in analogical_patterns)
        analogical_density = analogical_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA puede usar analogías de manera excesiva o muy estructurada
        if analogical_density > 0.3:
            score += 0.4
        elif analogical_density > 0.2:
            score += 0.3
        elif analogical_density > 0.1:
            score += 0.2
        
        # Análisis de analogías estructuradas excesivamente
        structured_analogies = [
            r'\b(?:just as|just like)\s+\w+\s+(?:so|too|also)\s+\w+',
            r'\b(?:in the same way that|in the same manner that|in the same fashion that)',
            r'\b(?:similar to how|similar to the way|similar to the manner)',
            r'\b(?:by the same token|in the same vein|along the same lines)',
            r'\b(?:to draw an analogy|to make an analogy|to use an analogy)'
        ]
        
        structured_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in structured_analogies)
        if structured_count > 2:
            score += 0.3
        elif structured_count > 1:
            score += 0.2
        
        # Análisis de analogías forzadas o poco naturales
        forced_analogies = [
            r'\b(?:this can be analogized to|this can be compared to|this can be likened to)',
            r'\b(?:an analogy can be drawn|an analogy can be made|an analogy can be used)',
            r'\b(?:using the analogy of|through the analogy of|by means of an analogy)',
            r'\b(?:to use an analogy|to draw an analogy|to make an analogy)\s*[,:]'
        ]
        
        forced_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in forced_analogies)
        if forced_count > 1:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_irony_patterns(self, text: str) -> float:
        """Analiza patrones de ironía típicos de IA - NUEVO V36"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Marcadores de ironía
        irony_markers = [
            r'\b(?:ironically|ironic|irony)',
            r'\b(?:paradoxically|paradoxical|paradox)',
            r'\b(?:surprisingly|surprising|surprise)',
            r'\b(?:unexpectedly|unexpected|unexpected)',
            r'\b(?:ironically enough|ironic as it may seem|ironic though it may be)',
            r'\b(?:it is ironic|it\'s ironic|how ironic)',
            r'\b(?:the irony is|the irony of|the irony that)',
            r'\b(?:what\'s ironic|what is ironic|the ironic thing)',
            r'\b(?:in an ironic twist|in an ironic turn|in an ironic way)',
            r'\b(?:ironically|paradoxically|surprisingly)\s+(?:enough|though|however)'
        ]
        
        irony_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in irony_markers)
        irony_density = irony_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA puede usar marcadores de ironía de manera excesiva o muy explícita
        if irony_density > 0.2:
            score += 0.4
        elif irony_density > 0.1:
            score += 0.3
        
        # Análisis de ironía verbal explícita (típico de IA)
        explicit_irony = [
            r'\b(?:ironically|ironic|irony)\s+(?:speaking|stated|said|noted)',
            r'\b(?:it is ironic that|it\'s ironic that|how ironic that)',
            r'\b(?:the irony lies in|the irony is that|the irony of it is)',
            r'\b(?:this is ironic|that is ironic|which is ironic)',
            r'\b(?:one might find it ironic|some might find it ironic)'
        ]
        
        explicit_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in explicit_irony)
        if explicit_count > 1:
            score += 0.3
        elif explicit_count > 0:
            score += 0.2
        
        # Análisis de falta de ironía sutil o implícita
        # IA tiende a usar ironía muy explícita en lugar de sutil
        subtle_irony_indicators = [
            r'\b(?:of course|naturally|obviously|clearly)\s+(?:not|never|no)',
            r'\b(?:as if|as though)\s+(?:that|this|it)\s+(?:would|could|might)',
            r'\b(?:right|sure|yeah)\s+(?:right|sure|yeah)',
            r'\b(?:oh|ah|well)\s+(?:yes|no|sure|right)'
        ]
        
        subtle_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in subtle_irony_indicators)
        if irony_count > 0 and subtle_count == 0:
            # Si hay ironía pero solo explícita, puede ser IA
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_humor_patterns(self, text: str) -> float:
        """Detecta patrones de humor típicos de IA - NUEVO V36"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Marcadores de humor explícito
        humor_markers = [
            r'\b(?:funny|humor|humorous|humorously)',
            r'\b(?:joke|jokes|joking|jokingly)',
            r'\b(?:amusing|amusement|amusingly)',
            r'\b(?:witty|wittiness|wittily)',
            r'\b(?:hilarious|hilariously|hilarity)',
            r'\b(?:comical|comically|comedy)',
            r'\b(?:laugh|laughs|laughing|laughable)',
            r'\b(?:chuckle|chuckles|chuckling)',
            r'\b(?:giggle|giggles|giggling)',
            r'\b(?:smile|smiles|smiling)',
            r'\b(?:grin|grins|grinning)',
            r'\b(?:ha ha|haha|hehe|lol|lmao|rofl)',
            r'\b(?:that\'s funny|that is funny|how funny)',
            r'\b(?:that\'s amusing|that is amusing|how amusing)',
            r'\b(?:that\'s hilarious|that is hilarious|how hilarious)'
        ]
        
        humor_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in humor_markers)
        humor_density = humor_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA puede usar marcadores de humor de manera excesiva o muy explícita
        if humor_density > 0.2:
            score += 0.4
        elif humor_density > 0.1:
            score += 0.3
        
        # Análisis de chistes o bromas estructuradas
        structured_humor = [
            r'\b(?:here\'s a joke|here is a joke|let me tell you a joke)',
            r'\b(?:that reminds me of a joke|that reminds me of a story)',
            r'\b(?:speaking of|on the subject of|in that vein)',
            r'\b(?:to lighten the mood|to break the ice|to add some humor)',
            r'\b(?:as a joke|as a humorous aside|as a funny note)',
            r'\b(?:jokingly|humorously|amusingly)\s+(?:speaking|stated|said|noted)'
        ]
        
        structured_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in structured_humor)
        if structured_count > 1:
            score += 0.3
        elif structured_count > 0:
            score += 0.2
        
        # Análisis de falta de humor sutil o espontáneo
        # IA tiende a usar humor muy explícito o estructurado
        subtle_humor_indicators = [
            r'\b(?:wink|winks|winking)',
            r'\b(?:nudge|nudges|nudging)',
            r'\b(?:tongue in cheek|tongue-in-cheek)',
            r'\b(?:with a smile|with a grin|with a chuckle)',
            r'\b(?:playfully|teasingly|jokingly)\s+(?:said|stated|noted)'
        ]
        
        subtle_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in subtle_humor_indicators)
        if humor_count > 0 and subtle_count == 0:
            # Si hay humor pero solo explícito, puede ser IA
            score += 0.2
        
        # Análisis de chistes o bromas poco naturales o forzadas
        forced_humor = [
            r'\b(?:ha ha|haha|hehe)\s+(?:ha ha|haha|hehe)',
            r'\b(?:that\'s a good one|that\'s a funny one|that\'s hilarious)',
            r'\b(?:now that\'s funny|now that is funny|now that\'s hilarious)',
            r'\b(?:how funny|how amusing|how hilarious)\s+(?:that|this|it)'
        ]
        
        forced_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in forced_humor)
        if forced_count > 1:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_sarcasm_patterns(self, text: str) -> float:
        """Analiza patrones de sarcasmo típicos de IA - NUEVO V37"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Marcadores de sarcasmo explícito
        sarcasm_markers = [
            r'\b(?:sarcastically|sarcastic|sarcasm)',
            r'\b(?:yeah right|sure thing|oh really|oh sure)',
            r'\b(?:as if|as though)\s+(?:that|this|it)\s+(?:would|could|might)',
            r'\b(?:right|sure|yeah)\s+(?:right|sure|yeah)',
            r'\b(?:oh|ah|well)\s+(?:yes|no|sure|right)',
            r'\b(?:of course|naturally|obviously|clearly)\s+(?:not|never|no)',
            r'\b(?:that\'s|that is)\s+(?:great|wonderful|fantastic|amazing)\s+(?:sarcastically|sarcastic)',
            r'\b(?:how|what)\s+(?:great|wonderful|fantastic|amazing)\s+(?:that|this|it)',
            r'\b(?:isn\'t|aren\'t|wasn\'t|weren\'t)\s+(?:that|this|it)\s+(?:great|wonderful|fantastic)',
            r'\b(?:i\'m|i am)\s+(?:so|very|really|extremely)\s+(?:happy|glad|pleased|thrilled)'
        ]
        
        sarcasm_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in sarcasm_markers)
        sarcasm_density = sarcasm_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA puede usar sarcasmo de manera excesiva o muy explícita
        if sarcasm_density > 0.2:
            score += 0.4
        elif sarcasm_density > 0.1:
            score += 0.3
        
        # Análisis de sarcasmo explícito (típico de IA)
        explicit_sarcasm = [
            r'\b(?:sarcastically|sarcastic|sarcasm)\s+(?:speaking|stated|said|noted)',
            r'\b(?:it is sarcastic|it\'s sarcastic|how sarcastic)',
            r'\b(?:the sarcasm is|the sarcasm of|the sarcasm that)',
            r'\b(?:what\'s sarcastic|what is sarcastic|the sarcastic thing)',
            r'\b(?:in a sarcastic way|in a sarcastic manner|sarcastically speaking)'
        ]
        
        explicit_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in explicit_sarcasm)
        if explicit_count > 1:
            score += 0.3
        elif explicit_count > 0:
            score += 0.2
        
        # Análisis de falta de sarcasmo sutil o implícito
        # IA tiende a usar sarcasmo muy explícito en lugar de sutil
        subtle_sarcasm_indicators = [
            r'\b(?:wink|winks|winking)',
            r'\b(?:nudge|nudges|nudging)',
            r'\b(?:tongue in cheek|tongue-in-cheek)',
            r'\b(?:with a smirk|with a grin|with a raised eyebrow)',
            r'\b(?:playfully|teasingly|sarcastically)\s+(?:said|stated|noted)'
        ]
        
        subtle_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in subtle_sarcasm_indicators)
        if sarcasm_count > 0 and subtle_count == 0:
            # Si hay sarcasmo pero solo explícito, puede ser IA
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_hyperbole_patterns(self, text: str) -> float:
        """Detecta patrones de hipérbole típicos de IA - NUEVO V37"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Marcadores de hipérbole
        hyperbole_markers = [
            r'\b(?:extremely|incredibly|unbelievably|absolutely|completely|totally|utterly)',
            r'\b(?:the most|the best|the worst|the greatest|the smallest|the largest)',
            r'\b(?:never|always|forever|eternally|infinitely|endlessly)',
            r'\b(?:every|all|none|nothing|everything|everyone|nobody)',
            r'\b(?:millions|billions|trillions|countless|innumerable|myriad)',
            r'\b(?:perfect|perfectly|flawless|flawlessly|impeccable|impeccably)',
            r'\b(?:amazing|amazingly|astounding|astoundingly|stunning|stunningly)',
            r'\b(?:incredible|incredibly|unbelievable|unbelievably|remarkable|remarkably)',
            r'\b(?:phenomenal|phenomenally|extraordinary|extraordinarily|exceptional|exceptionally)',
            r'\b(?:outstanding|outstandingly|exceptional|exceptionally|remarkable|remarkably)',
            r'\b(?:beyond|exceed|exceeds|exceeding|surpass|surpasses|surpassing)',
            r'\b(?:unprecedented|unprecedentedly|unparalleled|unparalleled)',
            r'\b(?:revolutionary|revolutionarily|groundbreaking|groundbreakingly)',
            r'\b(?:life-changing|world-changing|game-changing|mind-blowing)',
            r'\b(?:once in a lifetime|once in a million|one of a kind|unique)'
        ]
        
        hyperbole_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in hyperbole_markers)
        hyperbole_density = hyperbole_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA puede usar hipérbole de manera excesiva
        if hyperbole_density > 0.4:
            score += 0.4
        elif hyperbole_density > 0.3:
            score += 0.3
        elif hyperbole_density > 0.2:
            score += 0.2
        
        # Análisis de superlativos excesivos
        superlatives = [
            r'\b(?:the most|the best|the worst|the greatest|the smallest|the largest)',
            r'\b(?:most|best|worst|greatest|smallest|largest)\s+\w+',
            r'\b(?:more|better|worse|greater|smaller|larger)\s+than\s+(?:any|all|every)',
            r'\b(?:better|worse|greater|smaller|larger)\s+than\s+(?:ever|before|anything|anyone)'
        ]
        
        superlative_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in superlatives)
        if superlative_count > len(sentences) * 0.2:
            score += 0.3
        elif superlative_count > len(sentences) * 0.15:
            score += 0.2
        
        # Análisis de absolutos excesivos
        absolutes = [
            r'\b(?:never|always|forever|eternally|infinitely|endlessly)',
            r'\b(?:every|all|none|nothing|everything|everyone|nobody)',
            r'\b(?:completely|totally|utterly|absolutely|entirely|fully|wholly)',
            r'\b(?:without|no|not)\s+(?:exception|doubt|question|fail|error)'
        ]
        
        absolute_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in absolutes)
        if absolute_count > len(sentences) * 0.3:
            score += 0.3
        elif absolute_count > len(sentences) * 0.2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_euphemism_patterns(self, text: str) -> float:
        """Analiza patrones de eufemismo típicos de IA - NUEVO V37"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Eufemismos comunes
        euphemisms = [
            r'\b(?:passed away|passed on|no longer with us|left us)',
            r'\b(?:let go|let go of|released|downsized|rightsized)',
            r'\b(?:economically disadvantaged|financially challenged|low-income)',
            r'\b(?:vertically challenged|height-challenged|petite)',
            r'\b(?:differently abled|physically challenged|special needs)',
            r'\b(?:senior|senior citizen|elderly|golden years)',
            r'\b(?:restroom|bathroom|washroom|powder room)',
            r'\b(?:correctional facility|detention center|penal institution)',
            r'\b(?:pre-owned|pre-loved|gently used|previously owned)',
            r'\b(?:between jobs|in transition|exploring opportunities)',
            r'\b(?:enhanced interrogation|aggressive questioning|intensive questioning)',
            r'\b(?:collateral damage|unintended consequences|side effects)',
            r'\b(?:friendly fire|blue on blue|accidental engagement)',
            r'\b(?:economical|budget-friendly|cost-effective|affordable)',
            r'\b(?:full-figured|plus-sized|curvy|voluptuous)'
        ]
        
        euphemism_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in euphemisms)
        euphemism_density = euphemism_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA puede usar eufemismos de manera excesiva o muy formal
        if euphemism_density > 0.3:
            score += 0.4
        elif euphemism_density > 0.2:
            score += 0.3
        elif euphemism_density > 0.1:
            score += 0.2
        
        # Análisis de eufemismos corporativos o formales excesivos
        corporate_euphemisms = [
            r'\b(?:rightsizing|downsizing|restructuring|reorganizing)',
            r'\b(?:synergy|synergistic|synergize|synergizing)',
            r'\b(?:leverage|leveraging|utilize|utilizing)',
            r'\b(?:paradigm shift|paradigm|paradigmatic)',
            r'\b(?:value-added|value proposition|value creation)',
            r'\b(?:best practices|industry standard|benchmark)',
            r'\b(?:core competency|competitive advantage|market position)',
            r'\b(?:stakeholder|stakeholders|key stakeholder)',
            r'\b(?:deliverable|deliverables|key deliverable)',
            r'\b(?:action item|action items|actionable)'
        ]
        
        corporate_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in corporate_euphemisms)
        if corporate_count > len(sentences) * 0.2:
            score += 0.3
        elif corporate_count > len(sentences) * 0.15:
            score += 0.2
        
        # Análisis de eufemismos políticos o diplomáticos
        political_euphemisms = [
            r'\b(?:enhanced interrogation|aggressive questioning|intensive questioning)',
            r'\b(?:collateral damage|unintended consequences|side effects)',
            r'\b(?:friendly fire|blue on blue|accidental engagement)',
            r'\b(?:preemptive strike|preventive action|defensive measure)',
            r'\b(?:regime change|democratic transition|political transition)',
            r'\b(?:ethnic cleansing|population transfer|forced migration)',
            r'\b(?:extraordinary rendition|enhanced transfer|special transfer)'
        ]
        
        political_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in political_euphemisms)
        if political_count > 1:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_understatement_patterns(self, text: str) -> float:
        """Detecta patrones de lítote típicos de IA - NUEVO V37"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Patrones de lítote (understatement)
        understatement_patterns = [
            r'\b(?:not bad|not too bad|not terrible|not awful)',
            r'\b(?:not bad at all|not too shabby|not half bad)',
            r'\b(?:not the worst|not the best|not great|not terrible)',
            r'\b(?:not exactly|not quite|not really|not entirely)',
            r'\b(?:not un|not in|not without|not lacking)',
            r'\b(?:somewhat|rather|quite|fairly|pretty|a bit|a little)',
            r'\b(?:less than|more than|not more than|not less than)',
            r'\b(?:not insignificant|not unimportant|not trivial)',
            r'\b(?:not inconsiderable|not immaterial|not negligible)',
            r'\b(?:not unimpressive|not unremarkable|not unnoteworthy)',
            r'\b(?:could be worse|could be better|could be worse)',
            r'\b(?:it\'s not nothing|it\'s not bad|it\'s not terrible)',
            r'\b(?:not to be underestimated|not to be overlooked|not to be ignored)',
            r'\b(?:not without merit|not without value|not without importance)',
            r'\b(?:not exactly small|not exactly minor|not exactly insignificant)'
        ]
        
        understatement_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in understatement_patterns)
        understatement_density = understatement_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA puede usar lítote de manera excesiva o muy estructurada
        if understatement_density > 0.3:
            score += 0.4
        elif understatement_density > 0.2:
            score += 0.3
        elif understatement_density > 0.1:
            score += 0.2
        
        # Análisis de dobles negaciones (típico de lítote)
        double_negations = [
            r'\b(?:not un|not in|not without|not lacking)',
            r'\b(?:not insignificant|not unimportant|not trivial)',
            r'\b(?:not inconsiderable|not immaterial|not negligible)',
            r'\b(?:not unimpressive|not unremarkable|not unnoteworthy)',
            r'\b(?:not without merit|not without value|not without importance)',
            r'\b(?:not exactly small|not exactly minor|not exactly insignificant)'
        ]
        
        double_neg_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in double_negations)
        if double_neg_count > 2:
            score += 0.3
        elif double_neg_count > 1:
            score += 0.2
        
        # Análisis de atenuadores excesivos
        attenuators = [
            r'\b(?:somewhat|rather|quite|fairly|pretty|a bit|a little)',
            r'\b(?:more or less|sort of|kind of|rather|quite)',
            r'\b(?:to some extent|to a certain extent|to some degree)',
            r'\b(?:in some way|in a way|in some sense|in a sense)',
            r'\b(?:relatively|comparatively|relatively speaking)'
        ]
        
        attenuator_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in attenuators)
        if attenuator_count > len(sentences) * 0.2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_alliteration_patterns(self, text: str) -> float:
        """Analiza patrones de aliteración típicos de IA - NUEVO V38"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Detectar aliteración (repetición de sonidos consonánticos iniciales)
        alliteration_count = 0
        for sentence in sentences:
            words_in_sentence = [w.lower().strip('.,!?;:()[]{}"\'') for w in sentence.split()]
            words_in_sentence = [w for w in words_in_sentence if w and len(w) > 1]
            
            if len(words_in_sentence) >= 3:
                # Buscar secuencias de palabras que comienzan con la misma letra
                consecutive_alliteration = 0
                for i in range(len(words_in_sentence) - 2):
                    first_letters = [w[0] for w in words_in_sentence[i:i+3] if w[0].isalpha()]
                    if len(first_letters) >= 3 and len(set(first_letters)) == 1:
                        consecutive_alliteration += 1
                        alliteration_count += 1
                
                # También buscar aliteración en palabras cercanas
                for i in range(len(words_in_sentence) - 1):
                    if words_in_sentence[i][0].isalpha() and words_in_sentence[i+1][0].isalpha():
                        if words_in_sentence[i][0] == words_in_sentence[i+1][0]:
                            alliteration_count += 0.5
        
        alliteration_density = alliteration_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA puede usar aliteración de manera excesiva o muy estructurada
        if alliteration_density > 0.5:
            score += 0.4
        elif alliteration_density > 0.3:
            score += 0.3
        elif alliteration_density > 0.2:
            score += 0.2
        
        # Análisis de aliteración forzada o poco natural
        # IA a veces crea aliteración que suena forzada
        forced_alliteration_patterns = [
            r'\b(?:perfect|precise|precise|precise)',
            r'\b(?:comprehensive|complete|comprehensive|complete)',
            r'\b(?:systematic|structured|systematic|structured)',
            r'\b(?:methodical|meticulous|methodical|meticulous)'
        ]
        
        forced_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in forced_alliteration_patterns)
        if forced_count > 1:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_assonance_patterns(self, text: str) -> float:
        """Detecta patrones de asonancia típicos de IA - NUEVO V38"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Detectar asonancia (repetición de sonidos vocálicos)
        # Extraer vocales de palabras
        def extract_vowels(word):
            vowels = 'aeiou'
            return ''.join(c for c in word.lower() if c in vowels)
        
        assonance_count = 0
        for sentence in sentences:
            words_in_sentence = [w.lower().strip('.,!?;:()[]{}"\'') for w in sentence.split()]
            words_in_sentence = [w for w in words_in_sentence if w and len(w) > 1]
            
            if len(words_in_sentence) >= 3:
                # Buscar secuencias de palabras con sonidos vocálicos similares
                vowel_sequences = [extract_vowels(w) for w in words_in_sentence[:5]]
                
                for i in range(len(vowel_sequences) - 1):
                    if vowel_sequences[i] and vowel_sequences[i+1]:
                        # Si las secuencias vocálicas son similares
                        if vowel_sequences[i] == vowel_sequences[i+1] or \
                           (len(vowel_sequences[i]) > 0 and len(vowel_sequences[i+1]) > 0 and 
                            vowel_sequences[i][0] == vowel_sequences[i+1][0]):
                            assonance_count += 0.5
        
        assonance_density = assonance_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA puede usar asonancia de manera excesiva
        if assonance_density > 0.4:
            score += 0.3
        elif assonance_density > 0.3:
            score += 0.2
        
        # Análisis de asonancia en palabras comunes (puede ser accidental)
        common_words = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        common_assonance = sum(1 for w in words if w.lower() in common_words)
        if assonance_count > 0 and common_assonance / len(words) > 0.3:
            # Si hay mucha asonancia pero muchas palabras comunes, puede ser accidental
            score += 0.1
        
        return min(score, 1.0)
    
    def _analyze_ai_rhythm_patterns(self, text: str) -> float:
        """Analiza patrones de ritmo poético típicos de IA - NUEVO V38"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 5 or len(words) < 50:
            return 0.0
        
        # Análisis de ritmo basado en sílabas (aproximación)
        # Contar sílabas aproximadas por palabra (palabras largas = más sílabas)
        def approximate_syllables(word):
            word = word.lower().strip('.,!?;:()[]{}"\'')
            if not word:
                return 0
            # Aproximación simple: contar grupos de vocales
            vowels = 'aeiouy'
            syllable_count = 0
            prev_was_vowel = False
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_was_vowel:
                    syllable_count += 1
                prev_was_vowel = is_vowel
            return max(1, syllable_count)
        
        sentence_syllable_counts = []
        for sentence in sentences[:10]:
            words_in_sentence = sentence.split()
            total_syllables = sum(approximate_syllables(w) for w in words_in_sentence)
            sentence_syllable_counts.append(total_syllables)
        
        if len(sentence_syllable_counts) > 0:
            # Análisis de uniformidad en ritmo
            avg_syllables = sum(sentence_syllable_counts) / len(sentence_syllable_counts)
            variance = sum((x - avg_syllables) ** 2 for x in sentence_syllable_counts) / len(sentence_syllable_counts)
            std_dev = variance ** 0.5
            coefficient_of_variation = std_dev / avg_syllables if avg_syllables > 0 else 0.0
            
            # IA tiende a tener ritmo muy uniforme
            if coefficient_of_variation < 0.2:
                score += 0.4
            elif coefficient_of_variation < 0.3:
                score += 0.3
            elif coefficient_of_variation < 0.4:
                score += 0.2
        
        # Análisis de patrones rítmicos repetitivos
        # Buscar secuencias de sílabas similares
        if len(sentence_syllable_counts) >= 4:
            patterns = []
            for i in range(len(sentence_syllable_counts) - 3):
                pattern = tuple(sentence_syllable_counts[i:i+4])
                patterns.append(pattern)
            
            if patterns:
                pattern_counts = {}
                for pattern in patterns:
                    pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
                
                max_pattern_count = max(pattern_counts.values()) if pattern_counts else 0
                if max_pattern_count > 1:
                    score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_poetic_patterns(self, text: str) -> float:
        """Detecta patrones poéticos típicos de IA - NUEVO V38"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Patrones poéticos comunes
        poetic_patterns = [
            r'\b(?:like|as|as if|as though)\s+\w+',
            r'\b(?:metaphor|metaphorical|metaphorically|simile)',
            r'\b(?:rhyme|rhymes|rhyming|rhymed)',
            r'\b(?:verse|verses|versification|poetic)',
            r'\b(?:stanza|stanzas|strophic|strophic)',
            r'\b(?:alliteration|alliterative|alliteratively)',
            r'\b(?:assonance|assonant|assonantly)',
            r'\b(?:consonance|consonant|consonantly)',
            r'\b(?:meter|metrical|metrically|iambic|trochaic|anapestic|dactylic)',
            r'\b(?:sonnet|sonnets|haiku|haikus|limerick|limericks)',
            r'\b(?:ode|odes|elegy|elegies|ballad|ballads)',
            r'\b(?:imagery|imagistic|imagistically)',
            r'\b(?:symbol|symbolic|symbolize|symbolism)',
            r'\b(?:personification|personify|personifies)',
            r'\b(?:enjambment|enjambed|enjambing)'
        ]
        
        poetic_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in poetic_patterns)
        poetic_density = poetic_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA puede usar términos poéticos de manera excesiva o muy explícita
        if poetic_density > 0.2:
            score += 0.4
        elif poetic_density > 0.1:
            score += 0.3
        
        # Análisis de estructura poética explícita
        explicit_poetic = [
            r'\b(?:in poetic terms|in poetic language|poetically speaking)',
            r'\b(?:using poetic devices|employing poetic techniques|through poetic means)',
            r'\b(?:the poem|the verse|the stanza|the line)',
            r'\b(?:the poet|the writer|the author)\s+(?:uses|employs|utilizes)',
            r'\b(?:this poem|this verse|this stanza|this line)'
        ]
        
        explicit_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in explicit_poetic)
        if explicit_count > 1:
            score += 0.3
        elif explicit_count > 0:
            score += 0.2
        
        # Análisis de falta de elementos poéticos naturales
        # IA tiende a usar términos poéticos explícitos en lugar de crear poesía natural
        natural_poetic_indicators = [
            r'\b(?:whisper|whispers|whispering|whispered)',
            r'\b(?:dance|dances|dancing|danced)',
            r'\b(?:sing|sings|singing|sang)',
            r'\b(?:flow|flows|flowing|flowed)',
            r'\b(?:glow|glows|glowing|glowed)',
            r'\b(?:shine|shines|shining|shone)',
            r'\b(?:sparkle|sparkles|sparkling|sparkled)',
            r'\b(?:twinkle|twinkles|twinkling|twinkled)'
        ]
        
        natural_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in natural_poetic_indicators)
        if poetic_count > 0 and natural_count == 0:
            # Si hay términos poéticos pero no elementos poéticos naturales, puede ser IA
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_lexical_density_patterns(self, text: str) -> float:
        """Analiza patrones de densidad léxica típicos de IA - NUEVO V39"""
        score = 0.0
        words = [w.lower().strip('.,!?;:()[]{}"\'') for w in text.split()]
        words = [w for w in words if w and len(w) > 1]
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 50 or len(sentences) < 3:
            return 0.0
        
        # Calcular densidad léxica (proporción de palabras de contenido vs palabras funcionales)
        content_words = []
        function_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        
        for word in words:
            if word not in function_words and len(word) > 2:
                content_words.append(word)
        
        lexical_density = len(content_words) / len(words) if len(words) > 0 else 0.0
        
        # IA puede tener densidad léxica muy alta o muy baja
        if lexical_density > 0.70:
            score += 0.4
        elif lexical_density > 0.65:
            score += 0.3
        elif lexical_density < 0.40:
            score += 0.3
        elif lexical_density < 0.45:
            score += 0.2
        
        # Análisis de densidad léxica por oración
        sentence_lexical_densities = []
        for sentence in sentences:
            sentence_words = [w.lower().strip('.,!?;:()[]{}"\'') for w in sentence.split()]
            sentence_words = [w for w in sentence_words if w]
            sentence_content = [w for w in sentence_words if w not in function_words and len(w) > 2]
            if len(sentence_words) > 0:
                sent_density = len(sentence_content) / len(sentence_words)
                sentence_lexical_densities.append(sent_density)
        
        if len(sentence_lexical_densities) > 0:
            # Análisis de uniformidad en densidad léxica
            avg_density = sum(sentence_lexical_densities) / len(sentence_lexical_densities)
            variance = sum((x - avg_density) ** 2 for x in sentence_lexical_densities) / len(sentence_lexical_densities)
            std_dev = variance ** 0.5
            coefficient_of_variation = std_dev / avg_density if avg_density > 0 else 0.0
            
            # IA tiende a tener densidad léxica muy uniforme
            if coefficient_of_variation < 0.15:
                score += 0.3
            elif coefficient_of_variation < 0.20:
                score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_semantic_network_patterns(self, text: str) -> float:
        """Detecta patrones de redes semánticas típicos de IA - NUEVO V39"""
        score = 0.0
        words = [w.lower().strip('.,!?;:()[]{}"\'') for w in text.split()]
        words = [w for w in words if w and len(w) > 1]
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 50 or len(sentences) < 3:
            return 0.0
        
        # Análisis de relaciones semánticas entre palabras
        # Contar palabras relacionadas semánticamente que aparecen juntas
        semantic_clusters = {
            'technology': ['computer', 'software', 'hardware', 'system', 'network', 'digital', 'electronic', 'device', 'application', 'platform'],
            'business': ['company', 'organization', 'management', 'strategy', 'market', 'customer', 'product', 'service', 'revenue', 'profit'],
            'academic': ['research', 'study', 'analysis', 'methodology', 'theory', 'hypothesis', 'data', 'evidence', 'conclusion', 'result'],
            'science': ['experiment', 'observation', 'hypothesis', 'theory', 'data', 'evidence', 'analysis', 'conclusion', 'method', 'result'],
            'general': ['important', 'significant', 'crucial', 'essential', 'fundamental', 'key', 'main', 'primary', 'principal', 'major']
        }
        
        cluster_counts = {}
        for cluster_name, cluster_words in semantic_clusters.items():
            count = sum(1 for word in words if word in cluster_words)
            if count > 0:
                cluster_counts[cluster_name] = count
        
        # IA tiende a tener clusters semánticos muy concentrados
        if cluster_counts:
            total_cluster_words = sum(cluster_counts.values())
            max_cluster = max(cluster_counts.values())
            concentration = max_cluster / total_cluster_words if total_cluster_words > 0 else 0.0
            
            if concentration > 0.7:
                score += 0.4
            elif concentration > 0.6:
                score += 0.3
            elif concentration > 0.5:
                score += 0.2
        
        # Análisis de co-ocurrencia de palabras relacionadas
        # IA tiende a usar palabras relacionadas de manera muy estructurada
        related_word_pairs = [
            ('research', 'study'), ('analysis', 'data'), ('method', 'approach'),
            ('strategy', 'plan'), ('system', 'process'), ('theory', 'practice'),
            ('problem', 'solution'), ('cause', 'effect'), ('input', 'output'),
            ('beginning', 'end'), ('start', 'finish'), ('first', 'last')
        ]
        
        co_occurrence_count = 0
        for word1, word2 in related_word_pairs:
            if word1 in words and word2 in words:
                # Verificar si aparecen cerca (dentro de 20 palabras)
                indices1 = [i for i, w in enumerate(words) if w == word1]
                indices2 = [i for i, w in enumerate(words) if w == word2]
                
                for idx1 in indices1:
                    for idx2 in indices2:
                        if abs(idx1 - idx2) <= 20:
                            co_occurrence_count += 1
                            break
                    if co_occurrence_count > 0:
                        break
        
        if co_occurrence_count > len(sentences) * 0.3:
            score += 0.3
        elif co_occurrence_count > len(sentences) * 0.2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_ai_conceptual_coherence_patterns(self, text: str) -> float:
        """Analiza patrones de coherencia conceptual típicos de IA - NUEVO V39"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 5 or len(words) < 50:
            return 0.0
        
        # Análisis de coherencia conceptual excesiva
        # IA tiende a mantener coherencia conceptual muy alta y estructurada
        
        # Contar conceptos clave repetidos
        concept_words = [w.lower().strip('.,!?;:()[]{}"\'') for w in words if len(w) > 4]
        concept_frequency = {}
        for word in concept_words:
            concept_frequency[word] = concept_frequency.get(word, 0) + 1
        
        # Conceptos que aparecen muchas veces
        high_frequency_concepts = {word: count for word, count in concept_frequency.items() if count > 2}
        
        if len(high_frequency_concepts) > 0:
            # Calcular concentración de conceptos
            total_concept_occurrences = sum(high_frequency_concepts.values())
            max_concept_occurrences = max(high_frequency_concepts.values())
            concept_concentration = max_concept_occurrences / total_concept_occurrences if total_concept_occurrences > 0 else 0.0
            
            if concept_concentration > 0.4:
                score += 0.3
            elif concept_concentration > 0.3:
                score += 0.2
        
        # Análisis de transiciones conceptuales
        # IA tiende a tener transiciones conceptuales muy suaves y estructuradas
        transition_markers = [
            r'\b(?:moving to|turning to|shifting to|transitioning to)',
            r'\b(?:in relation to|in connection with|in reference to)',
            r'\b(?:building on|building upon|expanding on|expanding upon)',
            r'\b(?:related to|connected to|linked to|associated with)',
            r'\b(?:similar to|similar as|comparable to|comparable with)',
            r'\b(?:in contrast to|in contrast with|contrary to|unlike)'
        ]
        
        transition_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in transition_markers)
        if transition_count > len(sentences) * 0.2:
            score += 0.3
        elif transition_count > len(sentences) * 0.15:
            score += 0.2
        
        # Análisis de coherencia temática perfecta
        # IA tiende a mantener coherencia temática muy alta sin desviaciones
        if len(sentences) > 5:
            # Analizar distribución de palabras clave
            key_words = [word for word, count in concept_frequency.items() if count > 1]
            if len(key_words) > 0:
                # Contar cuántas oraciones contienen palabras clave
                sentences_with_keywords = 0
                for sentence in sentences:
                    sentence_words = [w.lower().strip('.,!?;:()[]{}"\'') for w in sentence.split()]
                    if any(keyword in sentence_words for keyword in key_words):
                        sentences_with_keywords += 1
                
                keyword_coverage = sentences_with_keywords / len(sentences) if len(sentences) > 0 else 0.0
                # Cobertura muy alta puede indicar IA
                if keyword_coverage > 0.9:
                    score += 0.2
        
        return min(score, 1.0)
    
    def _detect_ai_knowledge_graph_patterns(self, text: str) -> float:
        """Detecta patrones de grafos de conocimiento típicos de IA - NUEVO V39"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(sentences) < 3 or len(words) < 30:
            return 0.0
        
        # Patrones de relaciones de conocimiento estructuradas (típico de IA)
        knowledge_relations = [
            r'\b(?:is a|are a|was a|were a)\s+\w+',
            r'\b(?:is an|are an|was an|were an)\s+\w+',
            r'\b(?:is part of|are part of|was part of|were part of)',
            r'\b(?:belongs to|belong to|belonged to)',
            r'\b(?:is related to|are related to|was related to|were related to)',
            r'\b(?:is connected to|are connected to|was connected to|were connected to)',
            r'\b(?:is associated with|are associated with|was associated with|were associated with)',
            r'\b(?:is linked to|are linked to|was linked to|were linked to)',
            r'\b(?:is a type of|are a type of|was a type of|were a type of)',
            r'\b(?:is a kind of|are a kind of|was a kind of|were a kind of)',
            r'\b(?:is a form of|are a form of|was a form of|were a form of)',
            r'\b(?:is an example of|are an example of|was an example of|were an example of)',
            r'\b(?:consists of|consist of|consisted of)',
            r'\b(?:comprises|comprise|comprised)',
            r'\b(?:includes|include|included)',
            r'\b(?:contains|contain|contained)',
            r'\b(?:involves|involve|involved)',
            r'\b(?:requires|require|required)',
            r'\b(?:depends on|depend on|depended on)',
            r'\b(?:leads to|lead to|led to)',
            r'\b(?:results in|result in|resulted in)',
            r'\b(?:causes|cause|caused)',
            r'\b(?:affects|affect|affected)',
            r'\b(?:influences|influence|influenced)'
        ]
        
        relation_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in knowledge_relations)
        relation_density = relation_count / len(sentences) if len(sentences) > 0 else 0.0
        
        # IA tiende a usar muchas relaciones de conocimiento estructuradas
        if relation_density > 0.4:
            score += 0.4
        elif relation_density > 0.3:
            score += 0.3
        elif relation_density > 0.2:
            score += 0.2
        
        # Análisis de jerarquías de conocimiento
        hierarchy_patterns = [
            r'\b(?:category|categories|classify|classification)',
            r'\b(?:hierarchy|hierarchical|hierarchically)',
            r'\b(?:level|levels|tier|tiers)',
            r'\b(?:subcategory|subcategories|subclass|subclasses)',
            r'\b(?:parent|parents|child|children)',
            r'\b(?:superordinate|subordinate|superordinate|subordinate)',
            r'\b(?:general|specific|generalization|specialization)',
            r'\b(?:abstract|concrete|abstraction|concretion)'
        ]
        
        hierarchy_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in hierarchy_patterns)
        if hierarchy_count > 2:
            score += 0.3
        elif hierarchy_count > 1:
            score += 0.2
        
        # Análisis de taxonomías estructuradas
        taxonomy_patterns = [
            r'\b(?:first|second|third|fourth|fifth)\s+(?:category|type|class|level|tier)',
            r'\b(?:type a|type b|type c|type d|type e)',
            r'\b(?:class 1|class 2|class 3|class 4|class 5)',
            r'\b(?:level 1|level 2|level 3|level 4|level 5)',
            r'\b(?:tier 1|tier 2|tier 3|tier 4|tier 5)',
            r'\b(?:category i|category ii|category iii|category iv|category v)'
        ]
        
        taxonomy_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in taxonomy_patterns)
        if taxonomy_count > 1:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_advanced_writing_quality(self, text: str) -> float:
        """Análisis avanzado de calidad de escritura - NUEVO"""
        score = 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        if len(words) < 20:
            return 0.0
        
        # 1. Análisis de variedad sintáctica
        sentence_structures = []
        for sentence in sentences:
            if re.search(r'^[A-Z][^.!?]*\b(?:is|are|was|were)\b', sentence):
                sentence_structures.append('declarative')
            elif re.search(r'^[A-Z][^.!?]*\b(?:do|does|did|can|could|will|would)\b', sentence):
                sentence_structures.append('interrogative')
            elif re.search(r'^[A-Z][^.!?]*\b(?:let|may|should|must)\b', sentence):
                sentence_structures.append('imperative')
            else:
                sentence_structures.append('other')
        
        if sentence_structures:
            structure_diversity = len(set(sentence_structures)) / len(sentence_structures)
            if structure_diversity < 0.4:
                score += 0.25
        
        # 2. Análisis de uso de sinónimos
        word_freq = {}
        for word in words:
            word_clean = word.lower().strip('.,!?;:()[]{}"\'')
            if word_clean and len(word_clean) > 4:
                word_freq[word_clean] = word_freq.get(word_clean, 0) + 1
        
        repeated_words = [w for w, f in word_freq.items() if f > 2]
        if len(repeated_words) > len(word_freq) * 0.3:
            score += 0.2
        
        # 3. Análisis de fluidez y naturalidad
        artificial_phrases = [
            r'\bit is important to note that\b', r'\bit should be noted that\b',
            r'\bit is worth mentioning that\b', r'\bas a result of the fact that\b',
            r'\bin order to be able to\b', r'\bfor the purpose of\b',
            r'\bwith regard to the\b', r'\bin the context of the\b'
        ]
        artificial_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in artificial_phrases)
        if artificial_count > 2:
            score += 0.3
        
        # 4. Análisis de equilibrio en longitud de oraciones
        if len(sentences) > 1:
            sentence_lengths = [len(s.split()) for s in sentences]
            length_variance = np.var(sentence_lengths)
            avg_length = np.mean(sentence_lengths)
            if avg_length > 0:
                cv = np.sqrt(length_variance) / avg_length
                if cv < 0.4:
                    score += 0.15
        
        # 5. Análisis de uso de conectores
        connectors = ['however', 'therefore', 'furthermore', 'moreover', 'consequently', 'additionally', 'nevertheless', 'thus', 'hence', 'accordingly']
        connector_count = sum(1 for c in connectors if c in text.lower())
        connector_ratio = connector_count / len(sentences) if len(sentences) > 0 else 0
        if connector_ratio > 0.3:
            score += 0.2
        
        # 6. Análisis de coherencia en puntuación
        punctuation_patterns = [r'[,;:]', r'[.!?]', r'["\']']
        punct_counts = [len(re.findall(pattern, text)) for pattern in punctuation_patterns]
        if sum(punct_counts) > 0:
            punct_consistency = 1 - (np.std(punct_counts) / np.mean(punct_counts) if np.mean(punct_counts) > 0 else 0)
            if punct_consistency > 0.7:
                score += 0.1
        
        return min(score, 1.0)
    
    def _generate_alerts(self, ai_percentage: float, confidence: float, 
                        detected_models: List[Dict], primary_model: Optional[Dict]) -> List[Dict[str, Any]]:
        """Genera alertas para detecciones de alta confianza - NUEVO"""
        alerts = []
        
        # Alerta 1: Alta confianza de IA
        if ai_percentage > self.alert_threshold * 100:
            alerts.append({
                "type": "high_confidence_ai",
                "severity": "high",
                "message": f"Alta probabilidad de contenido generado por IA ({ai_percentage:.1f}%)",
                "confidence": confidence
            })
        
        # Alerta 2: Modelo específico detectado con alta confianza
        if primary_model and primary_model.get("confidence", 0) > 0.8:
            alerts.append({
                "type": "specific_model_detected",
                "severity": "high",
                "message": f"Modelo {primary_model.get('model_name', 'desconocido')} detectado con alta confianza ({primary_model.get('confidence', 0)*100:.1f}%)",
                "model": primary_model.get("model_name"),
                "confidence": primary_model.get("confidence", 0)
            })
        
        # Alerta 3: Múltiples modelos detectados (posible parafraseo)
        if len(detected_models) > 1:
            alerts.append({
                "type": "multiple_models_detected",
                "severity": "medium",
                "message": f"Múltiples modelos detectados ({len(detected_models)}), posible contenido parafraseado",
                "models": [m.get("model_name") for m in detected_models]
            })
        
        # Alerta 4: Confianza muy alta (>90%)
        if confidence > 0.9:
            alerts.append({
                "type": "very_high_confidence",
                "severity": "critical",
                "message": f"Confianza extremadamente alta ({confidence*100:.1f}%) - Detección muy confiable",
                "confidence": confidence
            })
        
        # Alerta 5: Confianza baja pero porcentaje alto (inconsistencia)
        if ai_percentage > 60 and confidence < 0.5:
            alerts.append({
                "type": "inconsistent_detection",
                "severity": "medium",
                "message": "Inconsistencia detectada: porcentaje alto pero confianza baja - revisar manualmente",
                "ai_percentage": ai_percentage,
                "confidence": confidence
            })
        
        return alerts
    
    def _forensic_analysis(self, text: str, detected_models: List[Dict], text_stats: Optional[Dict] = None) -> Dict[str, Any]:
        """Análisis forense mejorado para estimar el prompt usado"""
        forensic = {
            "estimated_prompt": None,
            "prompt_confidence": 0.0,
            "prompt_patterns": [],
            "generation_parameters": {},
            "forensic_evidence": [],
            "prompt_style": None,
            "estimated_instructions": []
        }
        
        text_lower = text.lower()
        word_count = len(text.split())
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Analizar estructura para inferir tipo de prompt
        prompt_patterns = []
        prompt_confidence_scores = []
        estimated_instructions = []
        
        # Prompt de explicación
        if re.search(r'\b(?:explain|describe|what is|how does|tell me about|can you explain|could you explain)\b', text_lower):
            prompt_patterns.append("explanatory")
            prompt_confidence_scores.append(0.65)
            # Extraer tema del texto
            first_sentence = sentences[0] if sentences else ""
            topic = first_sentence[:100] if len(first_sentence) > 20 else "the topic"
            estimated_instructions.append(f"Explain or describe {topic}")
            forensic["estimated_prompt"] = f"Explain or describe: {topic}..."
        
        # Prompt de lista/enumeración
        if re.search(r'\d+\.\s|\-\s|\*\s|•\s', text):
            prompt_patterns.append("list_generation")
            prompt_confidence_scores.append(0.75)
            list_keywords = re.findall(r'\b(?:list|items|steps|ways|methods|examples|reasons)\b', text_lower)
            if list_keywords:
                estimated_instructions.append(f"Generate a list of {list_keywords[0]}")
            else:
                estimated_instructions.append("Generate a list of items")
            if not forensic["estimated_prompt"]:
                forensic["estimated_prompt"] = "Generate a list of items about..."
        
        # Prompt de análisis
        if re.search(r'\b(?:analyze|analysis|evaluate|assess|review|examine|investigate)\b', text_lower):
            prompt_patterns.append("analytical")
            prompt_confidence_scores.append(0.70)
            estimated_instructions.append("Analyze or evaluate the following content")
            if not forensic["estimated_prompt"]:
                forensic["estimated_prompt"] = "Analyze or evaluate the following..."
        
        # Prompt de resumen
        if word_count < 300 and re.search(r'\b(?:summary|summarize|overview|brief|condense)\b', text_lower):
            prompt_patterns.append("summarization")
            prompt_confidence_scores.append(0.80)
            estimated_instructions.append("Summarize the following content")
            if not forensic["estimated_prompt"]:
                forensic["estimated_prompt"] = "Summarize the following content..."
        
        # Prompt de comparación
        if re.search(r'\b(?:compare|comparison|difference|similarities|contrast)\b', text_lower):
            prompt_patterns.append("comparison")
            prompt_confidence_scores.append(0.75)
            estimated_instructions.append("Compare the following items")
            if not forensic["estimated_prompt"]:
                forensic["estimated_prompt"] = "Compare the following..."
        
        # Prompt de escritura creativa
        if re.search(r'\b(?:write|create|generate|compose|draft)\b', text_lower) and word_count > 200:
            prompt_patterns.append("creative_writing")
            prompt_confidence_scores.append(0.65)
            estimated_instructions.append("Write or create content about")
            if not forensic["estimated_prompt"]:
                forensic["estimated_prompt"] = "Write about..."
        
        # Prompt de pregunta-respuesta
        if text_lower.startswith(('what', 'how', 'why', 'when', 'where', 'who', 'can', 'could', 'would', 'should')):
            prompt_patterns.append("qa")
            prompt_confidence_scores.append(0.70)
            first_q = sentences[0] if sentences else ""
            estimated_instructions.append(f"Answer: {first_q[:80]}")
            if not forensic["estimated_prompt"]:
                forensic["estimated_prompt"] = first_q[:100] + "..."
        
        # Detectar estilo de prompt
        if re.search(r'\b(?:please|kindly|could you|would you|i need|i want)\b', text_lower):
            forensic["prompt_style"] = "polite_request"
        elif re.search(r'\b(?:write|create|generate|make|do)\b', text_lower):
            forensic["prompt_style"] = "direct_command"
        elif re.search(r'\b(?:explain|describe|tell|show)\b', text_lower):
            forensic["prompt_style"] = "informational_request"
        else:
            forensic["prompt_style"] = "general"
        
        forensic["prompt_patterns"] = prompt_patterns
        forensic["estimated_instructions"] = estimated_instructions
        
        # Calcular confianza promedio
        if prompt_confidence_scores:
            forensic["prompt_confidence"] = np.mean(prompt_confidence_scores)
        elif detected_models:
            forensic["prompt_confidence"] = 0.5  # Confianza media si hay modelo detectado
        else:
            forensic["prompt_confidence"] = 0.3  # Baja confianza
        
        # Parámetros de generación estimados mejorados
        avg_sentence_length = text_stats.get("avg_sentence_length", word_count / len(sentences) if sentences else 15) if text_stats else 15
        burstiness = text_stats.get("burstiness", 0.5) if text_stats else 0.5
        
        # Estimar temperature basado en variabilidad
        if burstiness < 0.3:
            estimated_temp = 0.5  # Baja variabilidad = temperatura baja
        elif burstiness < 0.6:
            estimated_temp = 0.7  # Variabilidad media
        else:
            estimated_temp = 0.9  # Alta variabilidad
        
        # Estimar max_tokens basado en longitud
        estimated_max_tokens = int(min(word_count * 1.3, 4000))
        
        forensic["generation_parameters"] = {
            "estimated_max_tokens": estimated_max_tokens,
            "temperature": estimated_temp,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "top_p": 0.9,  # Estimación común
            "estimated_stop_sequences": None
        }
        
        # Evidencia forense mejorada
        forensic["forensic_evidence"] = [
            {
                "type": "text_length",
                "value": word_count,
                "indication": "long_text" if word_count > 500 else "medium_text" if word_count > 200 else "short_text"
            },
            {
                "type": "structure_type",
                "value": prompt_patterns[0] if prompt_patterns else "general",
                "indication": "structured" if prompt_patterns else "unstructured"
            },
            {
                "type": "sentence_structure",
                "value": avg_sentence_length,
                "indication": "uniform" if burstiness < 0.4 else "varied"
            },
            {
                "type": "detected_models_count",
                "value": len(detected_models),
                "indication": "high_confidence" if len(detected_models) > 0 else "low_confidence"
            }
        ]
        
        return forensic
    
    def _detect_image_ai(self, image_data: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Detecta si una imagen fue generada por IA"""
        # Placeholder para detección de imágenes
        # En producción, usar modelos como CLIP, GLIDE detector, etc.
        
        return {
            "is_ai_generated": False,
            "ai_percentage": 0.0,
            "detected_models": [],
            "primary_model": None,
            "forensic_analysis": None,
            "confidence_score": 0.0,
            "detection_methods": ["image_analysis_placeholder"]
        }
    
    def _detect_audio_ai(self, audio_data: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Detecta si un audio fue generado por IA"""
        # Placeholder para detección de audio
        # En producción, usar modelos de detección de deepfake audio
        
        return {
            "is_ai_generated": False,
            "ai_percentage": 0.0,
            "detected_models": [],
            "primary_model": None,
            "forensic_analysis": None,
            "confidence_score": 0.0,
            "detection_methods": ["audio_analysis_placeholder"]
        }
    
    def _detect_video_ai(self, video_data: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Detecta si un video fue generado por IA"""
        # Placeholder para detección de video
        # En producción, usar modelos de detección de deepfake video
        
        return {
            "is_ai_generated": False,
            "ai_percentage": 0.0,
            "detected_models": [],
            "primary_model": None,
            "forensic_analysis": None,
            "confidence_score": 0.0,
            "detection_methods": ["video_analysis_placeholder"]
        }
    
    def _detect_multimodal(self, content: Any, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Detecta contenido multimodal"""
        # Combinar resultados de diferentes modalidades
        return {
            "is_ai_generated": False,
            "ai_percentage": 0.0,
            "detected_models": [],
            "primary_model": None,
            "forensic_analysis": None,
            "confidence_score": 0.0,
            "detection_methods": ["multimodal_analysis_placeholder"]
        }
    
    def get_health(self) -> Dict[str, Any]:
        """Obtiene el estado de salud del detector"""
        return {
            "status": "healthy",
            "version": "1.0.0",
            "models_loaded": self.models_loaded,
            "uptime": time.time() - self.start_time
        }

