"""
Advanced NLP Analytics - Sistema avanzado de análisis y métricas NLP
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
import statistics
from collections import Counter, defaultdict

from ..models import TextAnalysisResult, SentimentResult, LanguageDetectionResult
from .embeddings import EmbeddingManager

logger = logging.getLogger(__name__)


@dataclass
class TextMetrics:
    """Métricas detalladas de texto."""
    readability_score: float = 0.0
    complexity_score: float = 0.0
    diversity_score: float = 0.0
    coherence_score: float = 0.0
    word_frequency: Dict[str, int] = field(default_factory=dict)
    sentence_lengths: List[int] = field(default_factory=list)
    paragraph_count: int = 0
    unique_words: int = 0
    total_words: int = 0
    lexical_diversity: float = 0.0


@dataclass
class ContentInsights:
    """Insights de contenido."""
    main_topics: List[str] = field(default_factory=list)
    key_entities: List[Dict[str, Any]] = field(default_factory=list)
    emotional_tone: str = ""
    writing_style: str = ""
    target_audience: str = ""
    content_quality: float = 0.0
    improvement_suggestions: List[str] = field(default_factory=list)
    seo_potential: float = 0.0


@dataclass
class TrendAnalysis:
    """Análisis de tendencias."""
    trending_topics: List[Dict[str, Any]] = field(default_factory=list)
    sentiment_trends: Dict[str, float] = field(default_factory=dict)
    language_evolution: Dict[str, Any] = field(default_factory=dict)
    content_patterns: List[str] = field(default_factory=list)
    seasonal_patterns: Dict[str, Any] = field(default_factory=dict)


class AdvancedNLPAnalytics:
    """
    Sistema avanzado de análisis y métricas NLP.
    """
    
    def __init__(self, embedding_manager: EmbeddingManager):
        """Inicializar analíticas avanzadas."""
        self.embedding_manager = embedding_manager
        self.analytics_cache = {}
        self.trend_data = defaultdict(list)
        self.performance_metrics = {
            "total_analyses": 0,
            "cache_hits": 0,
            "average_processing_time": 0.0
        }
        
        # Configuración
        self.cache_ttl = 3600  # 1 hora
        self.max_trend_history = 1000
        
        logger.info("AdvancedNLPAnalytics inicializado")
    
    async def analyze_text_comprehensive(self, text: str) -> Dict[str, Any]:
        """Análisis comprensivo de texto."""
        try:
            start_time = datetime.now()
            
            # Verificar cache
            cache_key = self._generate_cache_key(text)
            if cache_key in self.analytics_cache:
                cached_result = self.analytics_cache[cache_key]
                if datetime.now() - cached_result['timestamp'] < timedelta(seconds=self.cache_ttl):
                    self.performance_metrics["cache_hits"] += 1
                    return cached_result['result']
            
            # Realizar análisis completo
            analysis_result = {
                "text_metrics": await self._calculate_text_metrics(text),
                "content_insights": await self._generate_content_insights(text),
                "sentiment_analysis": await self._advanced_sentiment_analysis(text),
                "topic_modeling": await self._extract_topics_advanced(text),
                "entity_analysis": await self._analyze_entities_advanced(text),
                "readability_analysis": await self._analyze_readability(text),
                "writing_style": await self._analyze_writing_style(text),
                "seo_analysis": await self._analyze_seo_potential(text),
                "quality_score": await self._calculate_quality_score(text),
                "recommendations": await self._generate_recommendations(text),
                "analyzed_at": datetime.now().isoformat()
            }
            
            # Guardar en cache
            self.analytics_cache[cache_key] = {
                'result': analysis_result,
                'timestamp': datetime.now()
            }
            
            # Actualizar métricas
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_performance_metrics(processing_time)
            
            logger.info(f"Análisis comprensivo completado en {processing_time:.2f}s")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error en análisis comprensivo: {e}")
            raise
    
    async def _calculate_text_metrics(self, text: str) -> TextMetrics:
        """Calcular métricas detalladas del texto."""
        try:
            # Métricas básicas
            words = text.split()
            sentences = text.split('.')
            paragraphs = text.split('\n\n')
            
            # Longitudes de oraciones
            sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
            
            # Frecuencia de palabras
            word_frequency = Counter(words)
            
            # Diversidad léxica
            unique_words = len(set(words))
            total_words = len(words)
            lexical_diversity = unique_words / total_words if total_words > 0 else 0
            
            # Puntuación de legibilidad (Flesch Reading Ease simplificada)
            readability_score = self._calculate_flesch_score(text)
            
            # Puntuación de complejidad
            complexity_score = self._calculate_complexity_score(text)
            
            # Puntuación de diversidad
            diversity_score = self._calculate_diversity_score(text)
            
            # Puntuación de coherencia
            coherence_score = await self._calculate_coherence_score(text)
            
            return TextMetrics(
                readability_score=readability_score,
                complexity_score=complexity_score,
                diversity_score=diversity_score,
                coherence_score=coherence_score,
                word_frequency=dict(word_frequency.most_common(20)),
                sentence_lengths=sentence_lengths,
                paragraph_count=len(paragraphs),
                unique_words=unique_words,
                total_words=total_words,
                lexical_diversity=lexical_diversity
            )
            
        except Exception as e:
            logger.error(f"Error al calcular métricas de texto: {e}")
            return TextMetrics()
    
    async def _generate_content_insights(self, text: str) -> ContentInsights:
        """Generar insights de contenido."""
        try:
            # Extraer temas principales
            main_topics = await self._extract_main_topics(text)
            
            # Analizar entidades clave
            key_entities = await self._extract_key_entities(text)
            
            # Determinar tono emocional
            emotional_tone = await self._determine_emotional_tone(text)
            
            # Analizar estilo de escritura
            writing_style = await self._analyze_writing_style(text)
            
            # Determinar audiencia objetivo
            target_audience = await self._determine_target_audience(text)
            
            # Calcular calidad del contenido
            content_quality = await self._calculate_content_quality(text)
            
            # Generar sugerencias de mejora
            improvement_suggestions = await self._generate_improvement_suggestions(text)
            
            # Calcular potencial SEO
            seo_potential = await self._calculate_seo_potential(text)
            
            return ContentInsights(
                main_topics=main_topics,
                key_entities=key_entities,
                emotional_tone=emotional_tone,
                writing_style=writing_style,
                target_audience=target_audience,
                content_quality=content_quality,
                improvement_suggestions=improvement_suggestions,
                seo_potential=seo_potential
            )
            
        except Exception as e:
            logger.error(f"Error al generar insights de contenido: {e}")
            return ContentInsights()
    
    async def _advanced_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """Análisis de sentimiento avanzado."""
        try:
            # Análisis básico de sentimiento
            sentences = text.split('.')
            sentence_sentiments = []
            
            for sentence in sentences:
                if sentence.strip():
                    sentiment = await self._analyze_sentence_sentiment(sentence.strip())
                    sentence_sentiments.append(sentiment)
            
            # Calcular sentimiento general
            if sentence_sentiments:
                avg_positive = statistics.mean([s.get('positive', 0) for s in sentence_sentiments])
                avg_negative = statistics.mean([s.get('negative', 0) for s in sentence_sentiments])
                avg_neutral = statistics.mean([s.get('neutral', 0) for s in sentence_sentiments])
                
                overall_sentiment = "positive" if avg_positive > avg_negative else "negative" if avg_negative > avg_positive else "neutral"
                confidence = max(avg_positive, avg_negative, avg_neutral)
            else:
                overall_sentiment = "neutral"
                confidence = 0.5
                avg_positive = avg_negative = avg_neutral = 0.33
            
            # Análisis emocional
            emotions = await self._detect_emotions(text)
            
            # Análisis de polaridad por secciones
            polarity_by_section = await self._analyze_polarity_by_section(text)
            
            return {
                "overall_sentiment": overall_sentiment,
                "confidence": confidence,
                "positive_score": avg_positive,
                "negative_score": avg_negative,
                "neutral_score": avg_neutral,
                "emotions": emotions,
                "polarity_by_section": polarity_by_section,
                "sentiment_consistency": self._calculate_sentiment_consistency(sentence_sentiments),
                "emotional_intensity": self._calculate_emotional_intensity(sentence_sentiments)
            }
            
        except Exception as e:
            logger.error(f"Error en análisis de sentimiento avanzado: {e}")
            return {"overall_sentiment": "neutral", "confidence": 0.0}
    
    async def _extract_topics_advanced(self, text: str) -> Dict[str, Any]:
        """Extracción avanzada de temas."""
        try:
            # Análisis de palabras clave
            keywords = await self._extract_keywords_advanced(text)
            
            # Agrupación de temas
            topic_clusters = await self._cluster_topics(text, keywords)
            
            # Análisis de temas principales
            main_topics = await self._identify_main_topics(topic_clusters)
            
            # Análisis de subtemas
            subtopics = await self._identify_subtopics(text, main_topics)
            
            # Análisis de coherencia temática
            topic_coherence = await self._analyze_topic_coherence(text, main_topics)
            
            return {
                "main_topics": main_topics,
                "subtopics": subtopics,
                "topic_clusters": topic_clusters,
                "keywords": keywords,
                "topic_coherence": topic_coherence,
                "topic_diversity": len(main_topics),
                "topic_depth": self._calculate_topic_depth(subtopics)
            }
            
        except Exception as e:
            logger.error(f"Error en extracción avanzada de temas: {e}")
            return {"main_topics": [], "subtopics": [], "keywords": []}
    
    async def _analyze_entities_advanced(self, text: str) -> Dict[str, Any]:
        """Análisis avanzado de entidades."""
        try:
            # Extraer entidades básicas
            basic_entities = await self._extract_basic_entities(text)
            
            # Análisis de relaciones entre entidades
            entity_relationships = await self._analyze_entity_relationships(text, basic_entities)
            
            # Clasificación de entidades por importancia
            entity_importance = await self._rank_entity_importance(basic_entities, text)
            
            # Análisis de entidades nombradas
            named_entities = await self._extract_named_entities(text)
            
            # Análisis de entidades temporales
            temporal_entities = await self._extract_temporal_entities(text)
            
            # Análisis de entidades geográficas
            geographic_entities = await self._extract_geographic_entities(text)
            
            return {
                "basic_entities": basic_entities,
                "named_entities": named_entities,
                "temporal_entities": temporal_entities,
                "geographic_entities": geographic_entities,
                "entity_relationships": entity_relationships,
                "entity_importance": entity_importance,
                "entity_density": len(basic_entities) / len(text.split()) if text.split() else 0,
                "entity_diversity": len(set([e.get('type', '') for e in basic_entities]))
            }
            
        except Exception as e:
            logger.error(f"Error en análisis avanzado de entidades: {e}")
            return {"basic_entities": [], "named_entities": [], "entity_relationships": []}
    
    async def _analyze_readability(self, text: str) -> Dict[str, Any]:
        """Análisis de legibilidad."""
        try:
            # Puntuación Flesch Reading Ease
            flesch_score = self._calculate_flesch_score(text)
            
            # Puntuación Flesch-Kincaid Grade Level
            fk_grade = self._calculate_fk_grade_level(text)
            
            # Puntuación Gunning Fog
            fog_score = self._calculate_gunning_fog(text)
            
            # Puntuación SMOG
            smog_score = self._calculate_smog_index(text)
            
            # Análisis de complejidad de oraciones
            sentence_complexity = await self._analyze_sentence_complexity(text)
            
            # Análisis de vocabulario
            vocabulary_analysis = await self._analyze_vocabulary_complexity(text)
            
            # Recomendaciones de legibilidad
            readability_recommendations = self._generate_readability_recommendations(
                flesch_score, fk_grade, fog_score, smog_score
            )
            
            return {
                "flesch_reading_ease": flesch_score,
                "flesch_kincaid_grade": fk_grade,
                "gunning_fog": fog_score,
                "smog_index": smog_score,
                "sentence_complexity": sentence_complexity,
                "vocabulary_analysis": vocabulary_analysis,
                "readability_level": self._determine_readability_level(flesch_score),
                "recommendations": readability_recommendations
            }
            
        except Exception as e:
            logger.error(f"Error en análisis de legibilidad: {e}")
            return {"flesch_reading_ease": 0.0, "readability_level": "unknown"}
    
    async def _analyze_writing_style(self, text: str) -> Dict[str, Any]:
        """Análisis de estilo de escritura."""
        try:
            # Análisis de tono
            tone_analysis = await self._analyze_tone(text)
            
            # Análisis de voz
            voice_analysis = await self._analyze_voice(text)
            
            # Análisis de estructura
            structure_analysis = await self._analyze_structure(text)
            
            # Análisis de puntuación
            punctuation_analysis = await self._analyze_punctuation(text)
            
            # Análisis de conectores
            connector_analysis = await self._analyze_connectors(text)
            
            # Análisis de repetición
            repetition_analysis = await self._analyze_repetition(text)
            
            return {
                "tone": tone_analysis,
                "voice": voice_analysis,
                "structure": structure_analysis,
                "punctuation": punctuation_analysis,
                "connectors": connector_analysis,
                "repetition": repetition_analysis,
                "style_consistency": await self._analyze_style_consistency(text),
                "writing_quality": await self._assess_writing_quality(text)
            }
            
        except Exception as e:
            logger.error(f"Error en análisis de estilo de escritura: {e}")
            return {"tone": "neutral", "voice": "formal", "style_consistency": 0.0}
    
    async def _analyze_seo_potential(self, text: str) -> Dict[str, Any]:
        """Análisis de potencial SEO."""
        try:
            # Análisis de palabras clave
            keyword_analysis = await self._analyze_keywords_seo(text)
            
            # Análisis de densidad de palabras clave
            keyword_density = await self._analyze_keyword_density(text)
            
            # Análisis de estructura SEO
            seo_structure = await self._analyze_seo_structure(text)
            
            # Análisis de contenido único
            uniqueness_analysis = await self._analyze_content_uniqueness(text)
            
            # Análisis de longitud de contenido
            length_analysis = await self._analyze_content_length(text)
            
            # Recomendaciones SEO
            seo_recommendations = await self._generate_seo_recommendations(text)
            
            return {
                "keyword_analysis": keyword_analysis,
                "keyword_density": keyword_density,
                "seo_structure": seo_structure,
                "uniqueness_score": uniqueness_analysis,
                "length_analysis": length_analysis,
                "seo_score": await self._calculate_seo_score(text),
                "recommendations": seo_recommendations
            }
            
        except Exception as e:
            logger.error(f"Error en análisis de potencial SEO: {e}")
            return {"seo_score": 0.0, "recommendations": []}
    
    async def _calculate_quality_score(self, text: str) -> float:
        """Calcular puntuación de calidad del contenido."""
        try:
            # Factores de calidad
            readability_score = self._calculate_flesch_score(text)
            coherence_score = await self._calculate_coherence_score(text)
            diversity_score = self._calculate_diversity_score(text)
            structure_score = await self._analyze_structure_score(text)
            
            # Ponderación de factores
            quality_score = (
                readability_score * 0.3 +
                coherence_score * 0.3 +
                diversity_score * 0.2 +
                structure_score * 0.2
            )
            
            return min(100.0, max(0.0, quality_score))
            
        except Exception as e:
            logger.error(f"Error al calcular puntuación de calidad: {e}")
            return 0.0
    
    async def _generate_recommendations(self, text: str) -> List[str]:
        """Generar recomendaciones de mejora."""
        try:
            recommendations = []
            
            # Análisis de legibilidad
            flesch_score = self._calculate_flesch_score(text)
            if flesch_score < 30:
                recommendations.append("Considera simplificar el lenguaje para mejorar la legibilidad")
            elif flesch_score > 80:
                recommendations.append("El contenido es muy simple, considera agregar más complejidad")
            
            # Análisis de longitud
            word_count = len(text.split())
            if word_count < 300:
                recommendations.append("El contenido es muy corto, considera expandirlo")
            elif word_count > 2000:
                recommendations.append("El contenido es muy largo, considera dividirlo en secciones")
            
            # Análisis de estructura
            paragraphs = text.split('\n\n')
            if len(paragraphs) < 3:
                recommendations.append("Considera dividir el contenido en más párrafos")
            
            # Análisis de diversidad
            diversity_score = self._calculate_diversity_score(text)
            if diversity_score < 0.5:
                recommendations.append("Considera usar un vocabulario más diverso")
            
            # Análisis de coherencia
            coherence_score = await self._calculate_coherence_score(text)
            if coherence_score < 0.6:
                recommendations.append("Mejora la coherencia entre párrafos y oraciones")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error al generar recomendaciones: {e}")
            return []
    
    # Métodos auxiliares
    def _calculate_flesch_score(self, text: str) -> float:
        """Calcular puntuación Flesch Reading Ease."""
        try:
            sentences = [s for s in text.split('.') if s.strip()]
            words = text.split()
            
            if not sentences or not words:
                return 0.0
            
            avg_sentence_length = len(words) / len(sentences)
            avg_syllables_per_word = sum(self._count_syllables(word) for word in words) / len(words)
            
            score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
            return max(0, min(100, score))
            
        except Exception:
            return 0.0
    
    def _count_syllables(self, word: str) -> int:
        """Contar sílabas en una palabra."""
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
        
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def _calculate_complexity_score(self, text: str) -> float:
        """Calcular puntuación de complejidad."""
        try:
            words = text.split()
            if not words:
                return 0.0
            
            # Factores de complejidad
            avg_word_length = sum(len(word) for word in words) / len(words)
            long_words = sum(1 for word in words if len(word) > 6)
            long_word_ratio = long_words / len(words)
            
            # Puntuación de complejidad (0-100)
            complexity = (avg_word_length * 5) + (long_word_ratio * 50)
            return min(100.0, complexity)
            
        except Exception:
            return 0.0
    
    def _calculate_diversity_score(self, text: str) -> float:
        """Calcular puntuación de diversidad."""
        try:
            words = text.lower().split()
            if not words:
                return 0.0
            
            unique_words = len(set(words))
            total_words = len(words)
            
            return unique_words / total_words
            
        except Exception:
            return 0.0
    
    async def _calculate_coherence_score(self, text: str) -> float:
        """Calcular puntuación de coherencia."""
        try:
            # Análisis básico de coherencia
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if len(sentences) < 2:
                return 1.0
            
            # Análisis de conectores
            connectors = ['sin embargo', 'por lo tanto', 'además', 'por otro lado', 'en consecuencia']
            connector_count = sum(1 for sentence in sentences for connector in connectors if connector in sentence.lower())
            
            # Análisis de repetición de palabras clave
            words = text.lower().split()
            word_freq = Counter(words)
            common_words = [word for word, freq in word_freq.most_common(5) if freq > 1]
            
            # Puntuación de coherencia
            coherence = min(1.0, (connector_count / len(sentences)) + (len(common_words) / 10))
            return coherence
            
        except Exception:
            return 0.0
    
    def _generate_cache_key(self, text: str) -> str:
        """Generar clave de cache."""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()
    
    def _update_performance_metrics(self, processing_time: float):
        """Actualizar métricas de rendimiento."""
        self.performance_metrics["total_analyses"] += 1
        
        # Actualizar tiempo promedio
        total_time = self.performance_metrics["average_processing_time"] * (self.performance_metrics["total_analyses"] - 1)
        self.performance_metrics["average_processing_time"] = (total_time + processing_time) / self.performance_metrics["total_analyses"]
    
    async def get_analytics_metrics(self) -> Dict[str, Any]:
        """Obtener métricas de analíticas."""
        return {
            **self.performance_metrics,
            "cache_size": len(self.analytics_cache),
            "trend_data_size": len(self.trend_data),
            "timestamp": datetime.now().isoformat()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Verificar salud del sistema de analíticas."""
        try:
            return {
                "status": "healthy",
                "cache_size": len(self.analytics_cache),
                "performance_metrics": self.performance_metrics,
                "embedding_manager_status": await self.embedding_manager.health_check(),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error en health check de analíticas: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }




