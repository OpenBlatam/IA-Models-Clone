"""
NLP Core Engine - Motor principal del sistema NLP
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
from .text_processor import TextProcessor
from .sentiment_analyzer import SentimentAnalyzer
from .language_detector import LanguageDetector
from .text_generator import TextGenerator
from .summarizer import TextSummarizer
from .translator import TextTranslator

logger = logging.getLogger(__name__)


class NLPEngine:
    """
    Motor principal del sistema de procesamiento de lenguaje natural.
    """
    
    def __init__(self):
        """Inicializar el motor NLP."""
        self.text_processor = TextProcessor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.language_detector = LanguageDetector()
        self.text_generator = TextGenerator()
        self.summarizer = TextSummarizer()
        self.translator = TextTranslator()
        
        self._initialized = False
        self._start_time = time.time()
        
        # Métricas de rendimiento
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_processing_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Cache de resultados
        self.cache = {}
        self.cache_ttl = 3600  # 1 hora
        
        logger.info("NLP Engine inicializado")
    
    async def initialize(self):
        """Inicializar el motor NLP y cargar modelos."""
        if not self._initialized:
            try:
                # Inicializar componentes
                await self.text_processor.initialize()
                await self.sentiment_analyzer.initialize()
                await self.language_detector.initialize()
                await self.text_generator.initialize()
                await self.summarizer.initialize()
                await self.translator.initialize()
                
                self._initialized = True
                logger.info("NLP Engine completamente inicializado")
                
            except Exception as e:
                logger.error(f"Error al inicializar NLP Engine: {e}")
                raise
    
    async def shutdown(self):
        """Cerrar el motor NLP y limpiar recursos."""
        if self._initialized:
            try:
                # Limpiar cache
                self.cache.clear()
                
                # Cerrar componentes
                await self.text_processor.shutdown()
                await self.sentiment_analyzer.shutdown()
                await self.language_detector.shutdown()
                await self.text_generator.shutdown()
                await self.summarizer.shutdown()
                await self.translator.shutdown()
                
                self._initialized = False
                logger.info("NLP Engine cerrado")
                
            except Exception as e:
                logger.error(f"Error al cerrar NLP Engine: {e}")
    
    async def analyze_text(self, request: NLPAnalysisRequest) -> NLPAnalysisResponse:
        """
        Analizar texto con múltiples técnicas NLP.
        
        Args:
            request: Solicitud de análisis
            
        Returns:
            Respuesta con resultados del análisis
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
            
            # Procesar análisis
            results = {}
            
            # Análisis básico de texto
            if "text_analysis" in request.analysis_types:
                results["text_analysis"] = await self._analyze_text_basic(request.text)
            
            # Análisis de sentimiento
            if "sentiment" in request.analysis_types:
                results["sentiment"] = await self.sentiment_analyzer.analyze(request.text)
            
            # Detección de idioma
            if "language" in request.analysis_types:
                results["language"] = await self.language_detector.detect(request.text)
            
            # Reconocimiento de entidades
            if "entities" in request.analysis_types:
                results["entities"] = await self._extract_entities(request.text)
            
            # Extracción de palabras clave
            if "keywords" in request.analysis_types:
                results["keywords"] = await self._extract_keywords(request.text)
            
            # Modelado de temas
            if "topics" in request.analysis_types:
                results["topics"] = await self._extract_topics(request.text)
            
            # Clasificación de texto
            if "classification" in request.analysis_types:
                results["classification"] = await self._classify_text(request.text)
            
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
            
            logger.info(f"Análisis NLP completado: {request_id} (tiempo: {processing_time:.2f}s)")
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_metrics(processing_time, False)
            
            logger.error(f"Error en análisis NLP: {e}")
            
            return NLPAnalysisResponse(
                request_id=request_id,
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
    
    async def _analyze_text_basic(self, text: str) -> TextAnalysisResult:
        """Análisis básico de texto."""
        # Procesar texto
        processed_text = await self.text_processor.process(text)
        
        # Detectar idioma
        language_result = await self.language_detector.detect(text)
        
        # Analizar sentimiento
        sentiment_result = await self.sentiment_analyzer.analyze(text)
        
        # Calcular métricas básicas
        word_count = len(text.split())
        sentence_count = len(text.split('.'))
        character_count = len(text)
        
        # Calcular puntuación de legibilidad (simplificada)
        readability_score = self._calculate_readibility_score(text)
        
        return TextAnalysisResult(
            text=text,
            language=language_result.detected_language,
            sentiment=sentiment_result.sentiment,
            confidence=sentiment_result.confidence,
            word_count=word_count,
            sentence_count=sentence_count,
            character_count=character_count,
            readability_score=readability_score
        )
    
    async def _extract_entities(self, text: str) -> EntityRecognitionResult:
        """Extraer entidades del texto."""
        # Implementación simplificada - en producción usar spaCy o similar
        entities = []
        
        # Detectar entidades básicas (nombres, lugares, fechas)
        words = text.split()
        for i, word in enumerate(words):
            if word.istitle() and len(word) > 2:
                entities.append({
                    "text": word,
                    "label": "PERSON" if word.isupper() else "ORG",
                    "start": text.find(word),
                    "end": text.find(word) + len(word),
                    "confidence": 0.8
                })
        
        return EntityRecognitionResult(
            text=text,
            entities=entities,
            confidence=0.8
        )
    
    async def _extract_keywords(self, text: str) -> KeywordExtractionResult:
        """Extraer palabras clave del texto."""
        # Implementación simplificada - en producción usar RAKE o similar
        words = text.lower().split()
        
        # Filtrar palabras comunes
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Contar frecuencia
        word_freq = {}
        for word in filtered_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Obtener palabras más frecuentes
        keywords = []
        for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]:
            keywords.append({
                "word": word,
                "frequency": freq,
                "score": freq / len(filtered_words)
            })
        
        return KeywordExtractionResult(
            text=text,
            keywords=keywords,
            confidence=0.7
        )
    
    async def _extract_topics(self, text: str) -> TopicModelingResult:
        """Extraer temas del texto."""
        # Implementación simplificada - en producción usar LDA o similar
        topics = []
        
        # Análisis básico de temas por palabras clave
        keywords_result = await self._extract_keywords(text)
        
        if keywords_result.keywords:
            # Agrupar palabras relacionadas
            topic_words = [kw["word"] for kw in keywords_result.keywords[:5]]
            topics.append({
                "topic": " ".join(topic_words),
                "words": topic_words,
                "score": 0.8
            })
        
        return TopicModelingResult(
            text=text,
            topics=topics,
            dominant_topic=topics[0]["topic"] if topics else None,
            confidence=0.7
        )
    
    async def _classify_text(self, text: str) -> TextClassificationResult:
        """Clasificar texto por tipo."""
        # Implementación simplificada - en producción usar clasificador entrenado
        text_lower = text.lower()
        
        # Clasificación básica por palabras clave
        if any(word in text_lower for word in ["news", "report", "article"]):
            predicted_class = "news"
            confidence = 0.8
        elif any(word in text_lower for word in ["tweet", "post", "social"]):
            predicted_class = "social_media"
            confidence = 0.7
        elif any(word in text_lower for word in ["research", "study", "analysis"]):
            predicted_class = "academic"
            confidence = 0.8
        elif any(word in text_lower for word in ["technical", "code", "system"]):
            predicted_class = "technical"
            confidence = 0.7
        else:
            predicted_class = "general"
            confidence = 0.5
        
        return TextClassificationResult(
            text=text,
            predicted_class=predicted_class,
            confidence=confidence,
            all_classes=[
                {"class": "news", "score": 0.3},
                {"class": "social_media", "score": 0.2},
                {"class": "academic", "score": 0.2},
                {"class": "technical", "score": 0.2},
                {"class": "general", "score": 0.1}
            ]
        )
    
    def _calculate_readibility_score(self, text: str) -> float:
        """Calcular puntuación de legibilidad (Flesch Reading Ease simplificada)."""
        sentences = text.split('.')
        words = text.split()
        
        if not sentences or not words:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = sum(self._count_syllables(word) for word in words) / len(words)
        
        # Fórmula simplificada
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        return max(0, min(100, score))
    
    def _count_syllables(self, word: str) -> int:
        """Contar sílabas en una palabra (aproximación)."""
        word = word.lower()
        vowels = "aeiouy"
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            if char in vowels:
                if not prev_was_vowel:
                    syllable_count += 1
                prev_was_vowel = True
            else:
                prev_was_vowel = False
        
        # Ajustar para palabras que terminan en 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def _generate_cache_key(self, request: NLPAnalysisRequest) -> str:
        """Generar clave de cache."""
        import hashlib
        text_hash = hashlib.md5(request.text.encode()).hexdigest()
        analysis_types = "_".join(sorted(request.analysis_types))
        return f"{text_hash}_{analysis_types}"
    
    def _update_metrics(self, processing_time: float, success: bool):
        """Actualizar métricas de rendimiento."""
        self.metrics["total_requests"] += 1
        
        if success:
            self.metrics["successful_requests"] += 1
        else:
            self.metrics["failed_requests"] += 1
        
        # Actualizar tiempo promedio
        total_time = self.metrics["average_processing_time"] * (self.metrics["total_requests"] - 1)
        self.metrics["average_processing_time"] = (total_time + processing_time) / self.metrics["total_requests"]
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Obtener métricas del motor NLP."""
        uptime = time.time() - self._start_time
        
        return {
            "uptime_seconds": uptime,
            "uptime_formatted": str(timedelta(seconds=int(uptime))),
            "total_requests": self.metrics["total_requests"],
            "successful_requests": self.metrics["successful_requests"],
            "failed_requests": self.metrics["failed_requests"],
            "success_rate": (self.metrics["successful_requests"] / self.metrics["total_requests"] * 100) if self.metrics["total_requests"] > 0 else 0,
            "average_processing_time": self.metrics["average_processing_time"],
            "cache_hits": self.metrics["cache_hits"],
            "cache_misses": self.metrics["cache_misses"],
            "cache_hit_rate": (self.metrics["cache_hits"] / (self.metrics["cache_hits"] + self.metrics["cache_misses"]) * 100) if (self.metrics["cache_hits"] + self.metrics["cache_misses"]) > 0 else 0,
            "cache_size": len(self.cache),
            "last_updated": datetime.now().isoformat()
        }
    
    async def clear_cache(self):
        """Limpiar cache."""
        self.cache.clear()
        logger.info("Cache NLP limpiado")
    
    async def health_check(self) -> Dict[str, Any]:
        """Verificar salud del motor NLP."""
        try:
            # Verificar componentes
            components_status = {
                "text_processor": await self.text_processor.health_check(),
                "sentiment_analyzer": await self.sentiment_analyzer.health_check(),
                "language_detector": await self.language_detector.health_check(),
                "text_generator": await self.text_generator.health_check(),
                "summarizer": await self.summarizer.health_check(),
                "translator": await self.translator.health_check()
            }
            
            all_healthy = all(components_status.values())
            
            return {
                "status": "healthy" if all_healthy else "unhealthy",
                "components": components_status,
                "initialized": self._initialized,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error en health check: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


# Instancia global del motor NLP
_nlp_engine: Optional[NLPEngine] = None


def get_nlp_engine() -> NLPEngine:
    """Obtener la instancia global del motor NLP."""
    global _nlp_engine
    if _nlp_engine is None:
        _nlp_engine = NLPEngine()
    return _nlp_engine




