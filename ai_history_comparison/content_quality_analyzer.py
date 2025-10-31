"""
Advanced Content Quality Analysis System for AI History Comparison
Sistema avanzado de análisis de calidad de contenido para análisis de historial de IA
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import re
import math
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# NLP imports
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from textstat import flesch_reading_ease, flesch_kincaid_grade, gunning_fog, smog_index
import spacy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QualityDimension(Enum):
    """Dimensiones de calidad"""
    READABILITY = "readability"
    COHERENCE = "coherence"
    CLARITY = "clarity"
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    RELEVANCE = "relevance"
    ENGAGEMENT = "engagement"
    STRUCTURE = "structure"
    STYLE = "style"
    ORIGINALITY = "originality"

class QualityLevel(Enum):
    """Niveles de calidad"""
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"
    VERY_POOR = "very_poor"

class ContentType(Enum):
    """Tipos de contenido"""
    TECHNICAL = "technical"
    ACADEMIC = "academic"
    BUSINESS = "business"
    CREATIVE = "creative"
    INFORMATIONAL = "informational"
    PERSUASIVE = "persuasive"
    NARRATIVE = "narrative"

@dataclass
class QualityMetric:
    """Métrica de calidad"""
    dimension: QualityDimension
    score: float
    weight: float
    details: Dict[str, Any]
    recommendations: List[str]

@dataclass
class ContentQualityAnalysis:
    """Análisis de calidad de contenido"""
    id: str
    document_id: str
    content_type: ContentType
    overall_score: float
    quality_level: QualityLevel
    dimension_scores: Dict[QualityDimension, float]
    quality_metrics: List[QualityMetric]
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    benchmark_comparison: Dict[str, Any]
    readability_metrics: Dict[str, float]
    linguistic_features: Dict[str, Any]
    structural_features: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class QualityInsight:
    """Insight de calidad"""
    id: str
    insight_type: str
    description: str
    significance: float
    confidence: float
    related_dimensions: List[QualityDimension]
    actionable_recommendations: List[str]
    benchmark_context: str
    created_at: datetime = field(default_factory=datetime.now)

class AdvancedContentQualityAnalyzer:
    """
    Analizador avanzado de calidad de contenido
    """
    
    def __init__(
        self,
        enable_benchmarking: bool = True,
        enable_linguistic_analysis: bool = True,
        enable_structural_analysis: bool = True,
        enable_style_analysis: bool = True
    ):
        self.enable_benchmarking = enable_benchmarking
        self.enable_linguistic_analysis = enable_linguistic_analysis
        self.enable_structural_analysis = enable_structural_analysis
        self.enable_style_analysis = enable_style_analysis
        
        # Almacenamiento de análisis
        self.quality_analyses: Dict[str, ContentQualityAnalysis] = {}
        self.quality_insights: Dict[str, QualityInsight] = {}
        
        # Configuración de pesos por dimensión
        self.dimension_weights = {
            QualityDimension.READABILITY: 0.15,
            QualityDimension.COHERENCE: 0.15,
            QualityDimension.CLARITY: 0.15,
            QualityDimension.COMPLETENESS: 0.10,
            QualityDimension.ACCURACY: 0.10,
            QualityDimension.RELEVANCE: 0.10,
            QualityDimension.ENGAGEMENT: 0.10,
            QualityDimension.STRUCTURE: 0.10,
            QualityDimension.STYLE: 0.05
        }
        
        # Benchmarks de calidad
        self.quality_benchmarks = self._initialize_quality_benchmarks()
        
        # Configuración
        self.config = {
            "min_text_length": 50,
            "max_text_length": 100000,
            "readability_thresholds": {
                "excellent": 80,
                "good": 60,
                "average": 40,
                "poor": 20
            },
            "coherence_thresholds": {
                "excellent": 0.9,
                "good": 0.7,
                "average": 0.5,
                "poor": 0.3
            }
        }
    
    def _initialize_quality_benchmarks(self) -> Dict[str, Dict[str, float]]:
        """Inicializar benchmarks de calidad"""
        return {
            "technical": {
                "readability": 60.0,
                "coherence": 0.8,
                "clarity": 0.8,
                "completeness": 0.9,
                "accuracy": 0.95,
                "relevance": 0.9,
                "engagement": 0.6,
                "structure": 0.8,
                "style": 0.7
            },
            "academic": {
                "readability": 50.0,
                "coherence": 0.9,
                "clarity": 0.8,
                "completeness": 0.95,
                "accuracy": 0.98,
                "relevance": 0.9,
                "engagement": 0.5,
                "structure": 0.9,
                "style": 0.8
            },
            "business": {
                "readability": 70.0,
                "coherence": 0.8,
                "clarity": 0.9,
                "completeness": 0.8,
                "accuracy": 0.9,
                "relevance": 0.95,
                "engagement": 0.7,
                "structure": 0.8,
                "style": 0.7
            },
            "creative": {
                "readability": 65.0,
                "coherence": 0.7,
                "clarity": 0.7,
                "completeness": 0.7,
                "accuracy": 0.8,
                "relevance": 0.8,
                "engagement": 0.9,
                "structure": 0.6,
                "style": 0.9
            },
            "informational": {
                "readability": 75.0,
                "coherence": 0.8,
                "clarity": 0.9,
                "completeness": 0.9,
                "accuracy": 0.9,
                "relevance": 0.9,
                "engagement": 0.6,
                "structure": 0.8,
                "style": 0.6
            }
        }
    
    async def analyze_content_quality(
        self,
        text: str,
        document_id: str,
        content_type: ContentType = ContentType.INFORMATIONAL,
        context: Optional[Dict[str, Any]] = None
    ) -> ContentQualityAnalysis:
        """
        Analizar calidad de contenido
        
        Args:
            text: Texto a analizar
            document_id: ID del documento
            content_type: Tipo de contenido
            context: Contexto adicional
            
        Returns:
            Análisis completo de calidad
        """
        try:
            logger.info(f"Analyzing content quality for document {document_id}")
            
            # Validar texto
            if len(text) < self.config["min_text_length"]:
                raise ValueError(f"Text too short: {len(text)} characters")
            
            if len(text) > self.config["max_text_length"]:
                logger.warning(f"Text very long: {len(text)} characters, truncating analysis")
                text = text[:self.config["max_text_length"]]
            
            # Análisis por dimensiones
            dimension_scores = {}
            quality_metrics = []
            
            # 1. Legibilidad
            readability_score, readability_metrics = await self._analyze_readability(text)
            dimension_scores[QualityDimension.READABILITY] = readability_score
            quality_metrics.append(QualityMetric(
                dimension=QualityDimension.READABILITY,
                score=readability_score,
                weight=self.dimension_weights[QualityDimension.READABILITY],
                details=readability_metrics,
                recommendations=self._get_readability_recommendations(readability_score, readability_metrics)
            ))
            
            # 2. Coherencia
            coherence_score, coherence_details = await self._analyze_coherence(text)
            dimension_scores[QualityDimension.COHERENCE] = coherence_score
            quality_metrics.append(QualityMetric(
                dimension=QualityDimension.COHERENCE,
                score=coherence_score,
                weight=self.dimension_weights[QualityDimension.COHERENCE],
                details=coherence_details,
                recommendations=self._get_coherence_recommendations(coherence_score, coherence_details)
            ))
            
            # 3. Claridad
            clarity_score, clarity_details = await self._analyze_clarity(text)
            dimension_scores[QualityDimension.CLARITY] = clarity_score
            quality_metrics.append(QualityMetric(
                dimension=QualityDimension.CLARITY,
                score=clarity_score,
                weight=self.dimension_weights[QualityDimension.CLARITY],
                details=clarity_details,
                recommendations=self._get_clarity_recommendations(clarity_score, clarity_details)
            ))
            
            # 4. Completitud
            completeness_score, completeness_details = await self._analyze_completeness(text, context)
            dimension_scores[QualityDimension.COMPLETENESS] = completeness_score
            quality_metrics.append(QualityMetric(
                dimension=QualityDimension.COMPLETENESS,
                score=completeness_score,
                weight=self.dimension_weights[QualityDimension.COMPLETENESS],
                details=completeness_details,
                recommendations=self._get_completeness_recommendations(completeness_score, completeness_details)
            ))
            
            # 5. Precisión
            accuracy_score, accuracy_details = await self._analyze_accuracy(text, context)
            dimension_scores[QualityDimension.ACCURACY] = accuracy_score
            quality_metrics.append(QualityMetric(
                dimension=QualityDimension.ACCURACY,
                score=accuracy_score,
                weight=self.dimension_weights[QualityDimension.ACCURACY],
                details=accuracy_details,
                recommendations=self._get_accuracy_recommendations(accuracy_score, accuracy_details)
            ))
            
            # 6. Relevancia
            relevance_score, relevance_details = await self._analyze_relevance(text, context)
            dimension_scores[QualityDimension.RELEVANCE] = relevance_score
            quality_metrics.append(QualityMetric(
                dimension=QualityDimension.RELEVANCE,
                score=relevance_score,
                weight=self.dimension_weights[QualityDimension.RELEVANCE],
                details=relevance_details,
                recommendations=self._get_relevance_recommendations(relevance_score, relevance_details)
            ))
            
            # 7. Engagement
            engagement_score, engagement_details = await self._analyze_engagement(text)
            dimension_scores[QualityDimension.ENGAGEMENT] = engagement_score
            quality_metrics.append(QualityMetric(
                dimension=QualityDimension.ENGAGEMENT,
                score=engagement_score,
                weight=self.dimension_weights[QualityDimension.ENGAGEMENT],
                details=engagement_details,
                recommendations=self._get_engagement_recommendations(engagement_score, engagement_details)
            ))
            
            # 8. Estructura
            structure_score, structure_details = await self._analyze_structure(text)
            dimension_scores[QualityDimension.STRUCTURE] = structure_score
            quality_metrics.append(QualityMetric(
                dimension=QualityDimension.STRUCTURE,
                score=structure_score,
                weight=self.dimension_weights[QualityDimension.STRUCTURE],
                details=structure_details,
                recommendations=self._get_structure_recommendations(structure_score, structure_details)
            ))
            
            # 9. Estilo
            style_score, style_details = await self._analyze_style(text, content_type)
            dimension_scores[QualityDimension.STYLE] = style_score
            quality_metrics.append(QualityMetric(
                dimension=QualityDimension.STYLE,
                score=style_score,
                weight=self.dimension_weights[QualityDimension.STYLE],
                details=style_details,
                recommendations=self._get_style_recommendations(style_score, style_details)
            ))
            
            # Calcular score general
            overall_score = sum(
                metric.score * metric.weight for metric in quality_metrics
            )
            
            # Determinar nivel de calidad
            quality_level = self._determine_quality_level(overall_score)
            
            # Identificar fortalezas y debilidades
            strengths, weaknesses = self._identify_strengths_weaknesses(quality_metrics)
            
            # Generar recomendaciones generales
            recommendations = self._generate_general_recommendations(quality_metrics, overall_score)
            
            # Comparación con benchmarks
            benchmark_comparison = await self._compare_with_benchmarks(
                dimension_scores, content_type
            )
            
            # Análisis de características lingüísticas
            linguistic_features = await self._analyze_linguistic_features(text)
            
            # Análisis de características estructurales
            structural_features = await self._analyze_structural_features(text)
            
            # Crear análisis
            analysis = ContentQualityAnalysis(
                id=f"quality_{document_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                document_id=document_id,
                content_type=content_type,
                overall_score=overall_score,
                quality_level=quality_level,
                dimension_scores=dimension_scores,
                quality_metrics=quality_metrics,
                strengths=strengths,
                weaknesses=weaknesses,
                recommendations=recommendations,
                benchmark_comparison=benchmark_comparison,
                readability_metrics=readability_metrics,
                linguistic_features=linguistic_features,
                structural_features=structural_features
            )
            
            # Almacenar análisis
            self.quality_analyses[analysis.id] = analysis
            
            # Generar insights
            await self._generate_quality_insights(analysis)
            
            logger.info(f"Content quality analysis completed for document {document_id}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing content quality: {e}")
            raise
    
    async def _analyze_readability(self, text: str) -> Tuple[float, Dict[str, float]]:
        """Analizar legibilidad"""
        try:
            # Métricas de legibilidad
            flesch_ease = flesch_reading_ease(text)
            flesch_grade = flesch_kincaid_grade(text)
            gunning_fog_score = gunning_fog(text)
            smog_score = smog_index(text)
            
            # Normalizar score (0-1)
            # Flesch Reading Ease: 0-100, donde 100 es más fácil
            readability_score = min(1.0, max(0.0, flesch_ease / 100))
            
            metrics = {
                "flesch_reading_ease": flesch_ease,
                "flesch_kincaid_grade": flesch_grade,
                "gunning_fog": gunning_fog_score,
                "smog_index": smog_score,
                "average_grade_level": (flesch_grade + gunning_fog_score + smog_score) / 3
            }
            
            return readability_score, metrics
            
        except Exception as e:
            logger.error(f"Error analyzing readability: {e}")
            return 0.5, {}
    
    async def _analyze_coherence(self, text: str) -> Tuple[float, Dict[str, Any]]:
        """Analizar coherencia"""
        try:
            sentences = sent_tokenize(text)
            if len(sentences) < 2:
                return 0.5, {"sentence_count": len(sentences)}
            
            # Análisis de coherencia basado en conectores
            coherence_indicators = [
                r'\b(therefore|thus|hence|consequently|as a result)\b',
                r'\b(however|nevertheless|nonetheless|on the other hand)\b',
                r'\b(furthermore|moreover|additionally|in addition)\b',
                r'\b(first|second|third|finally|lastly)\b',
                r'\b(for example|for instance|specifically)\b',
                r'\b(in conclusion|to summarize|in summary)\b'
            ]
            
            total_indicators = 0
            for pattern in coherence_indicators:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                total_indicators += matches
            
            # Normalizar por número de oraciones
            coherence_score = min(1.0, total_indicators / len(sentences))
            
            # Análisis de transiciones entre oraciones
            transition_score = await self._analyze_sentence_transitions(sentences)
            
            details = {
                "sentence_count": len(sentences),
                "coherence_indicators": total_indicators,
                "transition_score": transition_score,
                "average_sentence_length": sum(len(s.split()) for s in sentences) / len(sentences)
            }
            
            # Combinar scores
            final_score = (coherence_score * 0.6 + transition_score * 0.4)
            
            return final_score, details
            
        except Exception as e:
            logger.error(f"Error analyzing coherence: {e}")
            return 0.5, {}
    
    async def _analyze_sentence_transitions(self, sentences: List[str]) -> float:
        """Analizar transiciones entre oraciones"""
        try:
            if len(sentences) < 2:
                return 0.5
            
            # Palabras de transición comunes
            transition_words = [
                'however', 'therefore', 'furthermore', 'moreover', 'additionally',
                'consequently', 'nevertheless', 'meanwhile', 'subsequently',
                'first', 'second', 'third', 'finally', 'lastly', 'next',
                'for example', 'for instance', 'specifically', 'namely'
            ]
            
            smooth_transitions = 0
            total_transitions = len(sentences) - 1
            
            for i in range(len(sentences) - 1):
                current_sentence = sentences[i].lower()
                next_sentence = sentences[i + 1].lower()
                
                # Verificar si hay palabras de transición
                has_transition = any(word in next_sentence for word in transition_words)
                
                # Verificar coherencia temática (palabras clave compartidas)
                current_words = set(current_sentence.split())
                next_words = set(next_sentence.split())
                shared_words = current_words & next_words
                
                # Excluir palabras comunes
                common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
                shared_words = shared_words - common_words
                
                if has_transition or len(shared_words) > 0:
                    smooth_transitions += 1
            
            return smooth_transitions / total_transitions if total_transitions > 0 else 0.5
            
        except Exception as e:
            logger.error(f"Error analyzing sentence transitions: {e}")
            return 0.5
    
    async def _analyze_clarity(self, text: str) -> Tuple[float, Dict[str, Any]]:
        """Analizar claridad"""
        try:
            # Análisis de claridad basado en varios factores
            
            # 1. Longitud promedio de oraciones
            sentences = sent_tokenize(text)
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
            
            # 2. Longitud promedio de palabras
            words = word_tokenize(text)
            avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
            
            # 3. Proporción de palabras complejas (más de 3 sílabas)
            complex_words = 0
            for word in words:
                if len(word) > 6:  # Aproximación simple
                    complex_words += 1
            
            complex_word_ratio = complex_words / len(words) if words else 0
            
            # 4. Uso de voz pasiva
            passive_voice_pattern = r'\b(was|were|been|being)\s+\w+ed\b'
            passive_matches = len(re.findall(passive_voice_pattern, text, re.IGNORECASE))
            passive_ratio = passive_matches / len(sentences) if sentences else 0
            
            # 5. Uso de jerga técnica
            technical_terms = len(re.findall(r'\b[A-Z]{2,}\b', text))  # Acrónimos
            technical_ratio = technical_terms / len(words) if words else 0
            
            # Calcular score de claridad
            clarity_factors = {
                "sentence_length": max(0, 1 - (avg_sentence_length - 15) / 20),  # Óptimo: 15 palabras
                "word_length": max(0, 1 - (avg_word_length - 5) / 3),  # Óptimo: 5 caracteres
                "complexity": max(0, 1 - complex_word_ratio * 2),  # Menos palabras complejas = mejor
                "passive_voice": max(0, 1 - passive_ratio * 3),  # Menos voz pasiva = mejor
                "technical_terms": max(0, 1 - technical_ratio * 5)  # Menos jerga = mejor
            }
            
            clarity_score = sum(clarity_factors.values()) / len(clarity_factors)
            
            details = {
                "average_sentence_length": avg_sentence_length,
                "average_word_length": avg_word_length,
                "complex_word_ratio": complex_word_ratio,
                "passive_voice_ratio": passive_ratio,
                "technical_terms_ratio": technical_ratio,
                "clarity_factors": clarity_factors
            }
            
            return clarity_score, details
            
        except Exception as e:
            logger.error(f"Error analyzing clarity: {e}")
            return 0.5, {}
    
    async def _analyze_completeness(self, text: str, context: Optional[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
        """Analizar completitud"""
        try:
            # Análisis de completitud basado en estructura y contenido
            
            # 1. Presencia de elementos estructurales
            has_introduction = bool(re.search(r'\b(introduction|overview|summary|abstract)\b', text, re.IGNORECASE))
            has_conclusion = bool(re.search(r'\b(conclusion|summary|in conclusion|to summarize)\b', text, re.IGNORECASE))
            has_headings = bool(re.search(r'^#+\s', text, re.MULTILINE)) or bool(re.search(r'^\d+\.\s', text, re.MULTILINE))
            
            # 2. Longitud del contenido
            word_count = len(word_tokenize(text))
            length_score = min(1.0, word_count / 500)  # Óptimo: 500+ palabras
            
            # 3. Densidad de información
            sentences = sent_tokenize(text)
            avg_info_per_sentence = word_count / len(sentences) if sentences else 0
            info_density_score = min(1.0, avg_info_per_sentence / 20)  # Óptimo: 20 palabras por oración
            
            # 4. Cobertura de temas (basado en palabras clave únicas)
            unique_words = len(set(word.lower() for word in word_tokenize(text) if len(word) > 3))
            topic_coverage_score = min(1.0, unique_words / 100)  # Óptimo: 100+ palabras únicas
            
            # 5. Presencia de ejemplos y evidencia
            has_examples = bool(re.search(r'\b(for example|for instance|such as|including)\b', text, re.IGNORECASE))
            has_evidence = bool(re.search(r'\b(according to|research shows|studies indicate|data suggests)\b', text, re.IGNORECASE))
            
            # Calcular score de completitud
            structural_score = (has_introduction + has_conclusion + has_headings) / 3
            content_score = (length_score + info_density_score + topic_coverage_score) / 3
            support_score = (has_examples + has_evidence) / 2
            
            completeness_score = (structural_score * 0.3 + content_score * 0.5 + support_score * 0.2)
            
            details = {
                "word_count": word_count,
                "sentence_count": len(sentences),
                "unique_words": unique_words,
                "has_introduction": has_introduction,
                "has_conclusion": has_conclusion,
                "has_headings": has_headings,
                "has_examples": has_examples,
                "has_evidence": has_evidence,
                "structural_score": structural_score,
                "content_score": content_score,
                "support_score": support_score
            }
            
            return completeness_score, details
            
        except Exception as e:
            logger.error(f"Error analyzing completeness: {e}")
            return 0.5, {}
    
    async def _analyze_accuracy(self, text: str, context: Optional[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
        """Analizar precisión"""
        try:
            # Análisis de precisión basado en indicadores de calidad
            
            # 1. Uso de fuentes y referencias
            has_citations = bool(re.search(r'\[.*?\]|\(.*?\d{4}.*?\)|\b(source|reference|citation)\b', text, re.IGNORECASE))
            
            # 2. Uso de datos específicos
            has_numbers = bool(re.search(r'\b\d+(\.\d+)?%?\b', text))
            has_dates = bool(re.search(r'\b\d{4}|\b(january|february|march|april|may|june|july|august|september|october|november|december)\b', text, re.IGNORECASE))
            
            # 3. Uso de lenguaje preciso
            vague_words = ['very', 'quite', 'rather', 'somewhat', 'fairly', 'pretty', 'really', 'totally', 'completely']
            vague_count = sum(1 for word in vague_words if word in text.lower())
            precision_score = max(0, 1 - vague_count / 10)  # Penalizar uso excesivo de palabras vagas
            
            # 4. Consistencia en terminología
            words = word_tokenize(text.lower())
            word_freq = Counter(words)
            repeated_terms = sum(1 for word, count in word_freq.items() if count > 3 and len(word) > 4)
            consistency_score = min(1.0, repeated_terms / 10)  # Algunos términos repetidos indican consistencia
            
            # 5. Uso de conectores lógicos
            logical_connectors = ['therefore', 'thus', 'hence', 'consequently', 'because', 'since', 'as a result']
            connector_count = sum(1 for connector in logical_connectors if connector in text.lower())
            logic_score = min(1.0, connector_count / 5)
            
            # Calcular score de precisión
            accuracy_factors = {
                "citations": 1.0 if has_citations else 0.5,
                "data_specificity": (1.0 if has_numbers else 0.5) + (1.0 if has_dates else 0.5),
                "precision": precision_score,
                "consistency": consistency_score,
                "logic": logic_score
            }
            
            accuracy_score = sum(accuracy_factors.values()) / len(accuracy_factors)
            
            details = {
                "has_citations": has_citations,
                "has_numbers": has_numbers,
                "has_dates": has_dates,
                "vague_words_count": vague_count,
                "repeated_terms": repeated_terms,
                "logical_connectors": connector_count,
                "accuracy_factors": accuracy_factors
            }
            
            return accuracy_score, details
            
        except Exception as e:
            logger.error(f"Error analyzing accuracy: {e}")
            return 0.5, {}
    
    async def _analyze_relevance(self, text: str, context: Optional[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
        """Analizar relevancia"""
        try:
            # Análisis de relevancia basado en contexto y contenido
            
            # 1. Densidad de palabras clave (si se proporciona contexto)
            keyword_density = 0.5  # Valor por defecto
            if context and 'keywords' in context:
                keywords = context['keywords']
                text_lower = text.lower()
                keyword_matches = sum(1 for keyword in keywords if keyword.lower() in text_lower)
                keyword_density = min(1.0, keyword_matches / len(keywords)) if keywords else 0.5
            
            # 2. Cobertura de temas principales
            topic_indicators = [
                r'\b(important|significant|key|main|primary|essential)\b',
                r'\b(problem|issue|challenge|solution|approach)\b',
                r'\b(impact|effect|influence|result|outcome)\b'
            ]
            
            topic_coverage = 0
            for pattern in topic_indicators:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                topic_coverage += matches
            
            topic_score = min(1.0, topic_coverage / 10)
            
            # 3. Actualidad del contenido
            current_year = datetime.now().year
            year_mentions = re.findall(r'\b(20\d{2})\b', text)
            recent_years = sum(1 for year in year_mentions if int(year) >= current_year - 5)
            recency_score = min(1.0, recent_years / 3) if year_mentions else 0.5
            
            # 4. Pertinencia del lenguaje
            relevant_terms = len(re.findall(r'\b(current|recent|latest|new|modern|contemporary)\b', text, re.IGNORECASE))
            relevance_language_score = min(1.0, relevant_terms / 5)
            
            # Calcular score de relevancia
            relevance_score = (keyword_density * 0.4 + topic_score * 0.3 + recency_score * 0.2 + relevance_language_score * 0.1)
            
            details = {
                "keyword_density": keyword_density,
                "topic_coverage": topic_coverage,
                "recent_years_mentioned": recent_years,
                "relevance_terms": relevant_terms,
                "year_mentions": year_mentions
            }
            
            return relevance_score, details
            
        except Exception as e:
            logger.error(f"Error analyzing relevance: {e}")
            return 0.5, {}
    
    async def _analyze_engagement(self, text: str) -> Tuple[float, Dict[str, Any]]:
        """Analizar engagement"""
        try:
            # Análisis de engagement basado en elementos que captan atención
            
            # 1. Uso de preguntas
            questions = len(re.findall(r'\?', text))
            question_score = min(1.0, questions / 5)  # Óptimo: 5 preguntas
            
            # 2. Uso de elementos visuales (markdown)
            has_bold = bool(re.search(r'\*\*.*?\*\*|__.*?__', text))
            has_italic = bool(re.search(r'\*.*?\*|_.*?_', text))
            has_lists = bool(re.search(r'^\s*[-*+]\s|^\s*\d+\.\s', text, re.MULTILINE))
            has_links = bool(re.search(r'\[.*?\]\(.*?\)', text))
            
            visual_elements = sum([has_bold, has_italic, has_lists, has_links])
            visual_score = visual_elements / 4
            
            # 3. Uso de lenguaje emocional
            emotional_words = [
                'amazing', 'incredible', 'fantastic', 'wonderful', 'excellent',
                'surprising', 'shocking', 'remarkable', 'outstanding', 'impressive'
            ]
            emotional_count = sum(1 for word in emotional_words if word in text.lower())
            emotional_score = min(1.0, emotional_count / 5)
            
            # 4. Uso de storytelling
            story_indicators = [
                r'\b(imagine|picture|consider|suppose)\b',
                r'\b(story|example|case|scenario)\b',
                r'\b(beginning|start|end|conclusion)\b'
            ]
            story_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in story_indicators)
            story_score = min(1.0, story_count / 5)
            
            # 5. Interactividad
            interactive_elements = [
                r'\b(think about|consider|reflect on|imagine)\b',
                r'\b(what if|suppose|let\'s)\b',
                r'\b(try|attempt|experiment)\b'
            ]
            interactive_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in interactive_elements)
            interactive_score = min(1.0, interactive_count / 5)
            
            # Calcular score de engagement
            engagement_score = (question_score * 0.2 + visual_score * 0.2 + emotional_score * 0.2 + 
                              story_score * 0.2 + interactive_score * 0.2)
            
            details = {
                "questions_count": questions,
                "visual_elements": visual_elements,
                "emotional_words": emotional_count,
                "story_indicators": story_count,
                "interactive_elements": interactive_count,
                "has_bold": has_bold,
                "has_italic": has_italic,
                "has_lists": has_lists,
                "has_links": has_links
            }
            
            return engagement_score, details
            
        except Exception as e:
            logger.error(f"Error analyzing engagement: {e}")
            return 0.5, {}
    
    async def _analyze_structure(self, text: str) -> Tuple[float, Dict[str, Any]]:
        """Analizar estructura"""
        try:
            # Análisis de estructura del documento
            
            # 1. Presencia de encabezados
            markdown_headers = len(re.findall(r'^#+\s', text, re.MULTILINE))
            numbered_headers = len(re.findall(r'^\d+\.\s', text, re.MULTILINE))
            total_headers = markdown_headers + numbered_headers
            header_score = min(1.0, total_headers / 5)  # Óptimo: 5 encabezados
            
            # 2. Organización en párrafos
            paragraphs = text.split('\n\n')
            paragraph_count = len([p for p in paragraphs if p.strip()])
            avg_paragraph_length = sum(len(p.split()) for p in paragraphs if p.strip()) / paragraph_count if paragraph_count > 0 else 0
            
            # Párrafos bien estructurados (50-200 palabras)
            well_structured_paragraphs = sum(1 for p in paragraphs if 50 <= len(p.split()) <= 200)
            paragraph_structure_score = well_structured_paragraphs / paragraph_count if paragraph_count > 0 else 0
            
            # 3. Uso de listas
            bullet_lists = len(re.findall(r'^\s*[-*+]\s', text, re.MULTILINE))
            numbered_lists = len(re.findall(r'^\s*\d+\.\s', text, re.MULTILINE))
            total_lists = bullet_lists + numbered_lists
            list_score = min(1.0, total_lists / 3)  # Óptimo: 3 listas
            
            # 4. Transiciones entre secciones
            transition_phrases = [
                r'\b(first|second|third|next|then|finally)\b',
                r'\b(moreover|furthermore|additionally)\b',
                r'\b(however|nevertheless|on the other hand)\b'
            ]
            transition_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in transition_phrases)
            transition_score = min(1.0, transition_count / 5)
            
            # 5. Longitud del documento
            word_count = len(word_tokenize(text))
            length_score = min(1.0, word_count / 1000)  # Óptimo: 1000+ palabras
            
            # Calcular score de estructura
            structure_score = (header_score * 0.3 + paragraph_structure_score * 0.3 + 
                             list_score * 0.2 + transition_score * 0.1 + length_score * 0.1)
            
            details = {
                "total_headers": total_headers,
                "markdown_headers": markdown_headers,
                "numbered_headers": numbered_headers,
                "paragraph_count": paragraph_count,
                "avg_paragraph_length": avg_paragraph_length,
                "well_structured_paragraphs": well_structured_paragraphs,
                "total_lists": total_lists,
                "bullet_lists": bullet_lists,
                "numbered_lists": numbered_lists,
                "transition_count": transition_count,
                "word_count": word_count
            }
            
            return structure_score, details
            
        except Exception as e:
            logger.error(f"Error analyzing structure: {e}")
            return 0.5, {}
    
    async def _analyze_style(self, text: str, content_type: ContentType) -> Tuple[float, Dict[str, Any]]:
        """Analizar estilo"""
        try:
            # Análisis de estilo basado en el tipo de contenido
            
            # 1. Consistencia en el tono
            formal_indicators = ['therefore', 'thus', 'consequently', 'furthermore', 'moreover']
            informal_indicators = ['awesome', 'cool', 'great', 'amazing', 'fantastic']
            
            formal_count = sum(1 for word in formal_indicators if word in text.lower())
            informal_count = sum(1 for word in informal_indicators if word in text.lower())
            
            # Determinar tono apropiado según el tipo de contenido
            if content_type in [ContentType.ACADEMIC, ContentType.TECHNICAL, ContentType.BUSINESS]:
                tone_score = formal_count / (formal_count + informal_count + 1)
            else:
                tone_score = informal_count / (formal_count + informal_count + 1)
            
            # 2. Variedad en el vocabulario
            words = word_tokenize(text.lower())
            unique_words = len(set(words))
            total_words = len(words)
            vocabulary_diversity = unique_words / total_words if total_words > 0 else 0
            diversity_score = min(1.0, vocabulary_diversity * 2)  # Normalizar
            
            # 3. Uso de conectores
            connectors = ['and', 'but', 'or', 'so', 'yet', 'for', 'nor']
            connector_count = sum(1 for word in words if word in connectors)
            connector_score = min(1.0, connector_count / 20)  # Óptimo: 20 conectores
            
            # 4. Longitud de oraciones
            sentences = sent_tokenize(text)
            sentence_lengths = [len(s.split()) for s in sentences]
            avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
            
            # Score basado en longitud apropiada según tipo de contenido
            if content_type in [ContentType.ACADEMIC, ContentType.TECHNICAL]:
                optimal_length = 20
            else:
                optimal_length = 15
            
            length_score = max(0, 1 - abs(avg_sentence_length - optimal_length) / optimal_length)
            
            # 5. Uso de voz activa vs pasiva
            passive_pattern = r'\b(was|were|been|being)\s+\w+ed\b'
            passive_count = len(re.findall(passive_pattern, text, re.IGNORECASE))
            active_score = max(0, 1 - passive_count / len(sentences)) if sentences else 0.5
            
            # Calcular score de estilo
            style_score = (tone_score * 0.3 + diversity_score * 0.2 + connector_score * 0.2 + 
                          length_score * 0.2 + active_score * 0.1)
            
            details = {
                "formal_indicators": formal_count,
                "informal_indicators": informal_count,
                "vocabulary_diversity": vocabulary_diversity,
                "unique_words": unique_words,
                "total_words": total_words,
                "connector_count": connector_count,
                "avg_sentence_length": avg_sentence_length,
                "passive_voice_count": passive_count,
                "tone_score": tone_score,
                "diversity_score": diversity_score,
                "connector_score": connector_score,
                "length_score": length_score,
                "active_score": active_score
            }
            
            return style_score, details
            
        except Exception as e:
            logger.error(f"Error analyzing style: {e}")
            return 0.5, {}
    
    async def _analyze_linguistic_features(self, text: str) -> Dict[str, Any]:
        """Analizar características lingüísticas"""
        try:
            words = word_tokenize(text)
            sentences = sent_tokenize(text)
            
            # Estadísticas básicas
            word_count = len(words)
            sentence_count = len(sentences)
            avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
            
            # Análisis de POS tags
            pos_tags = pos_tag(words)
            pos_counts = Counter(tag for word, tag in pos_tags)
            
            # Análisis de complejidad
            complex_words = sum(1 for word in words if len(word) > 6)
            complex_word_ratio = complex_words / word_count if word_count > 0 else 0
            
            # Análisis de repetición
            word_freq = Counter(word.lower() for word in words if len(word) > 3)
            most_common_words = word_freq.most_common(10)
            
            return {
                "word_count": word_count,
                "sentence_count": sentence_count,
                "avg_sentence_length": avg_sentence_length,
                "pos_distribution": dict(pos_counts),
                "complex_word_ratio": complex_word_ratio,
                "most_common_words": most_common_words,
                "unique_word_ratio": len(set(word.lower() for word in words)) / word_count if word_count > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing linguistic features: {e}")
            return {}
    
    async def _analyze_structural_features(self, text: str) -> Dict[str, Any]:
        """Analizar características estructurales"""
        try:
            # Análisis de estructura del documento
            lines = text.split('\n')
            
            # Encabezados
            markdown_headers = len([line for line in lines if line.strip().startswith('#')])
            numbered_sections = len([line for line in lines if re.match(r'^\d+\.\s', line.strip())])
            
            # Listas
            bullet_points = len([line for line in lines if re.match(r'^\s*[-*+]\s', line)])
            numbered_lists = len([line for line in lines if re.match(r'^\s*\d+\.\s', line)])
            
            # Párrafos
            paragraphs = [p for p in text.split('\n\n') if p.strip()]
            paragraph_lengths = [len(p.split()) for p in paragraphs]
            
            # Enlaces y referencias
            links = len(re.findall(r'\[.*?\]\(.*?\)', text))
            citations = len(re.findall(r'\[.*?\]|\(.*?\d{4}.*?\)', text))
            
            return {
                "markdown_headers": markdown_headers,
                "numbered_sections": numbered_sections,
                "bullet_points": bullet_points,
                "numbered_lists": numbered_lists,
                "paragraph_count": len(paragraphs),
                "avg_paragraph_length": sum(paragraph_lengths) / len(paragraph_lengths) if paragraph_lengths else 0,
                "links": links,
                "citations": citations,
                "total_lines": len(lines)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing structural features: {e}")
            return {}
    
    def _determine_quality_level(self, overall_score: float) -> QualityLevel:
        """Determinar nivel de calidad"""
        if overall_score >= 0.9:
            return QualityLevel.EXCELLENT
        elif overall_score >= 0.7:
            return QualityLevel.GOOD
        elif overall_score >= 0.5:
            return QualityLevel.AVERAGE
        elif overall_score >= 0.3:
            return QualityLevel.POOR
        else:
            return QualityLevel.VERY_POOR
    
    def _identify_strengths_weaknesses(self, quality_metrics: List[QualityMetric]) -> Tuple[List[str], List[str]]:
        """Identificar fortalezas y debilidades"""
        strengths = []
        weaknesses = []
        
        for metric in quality_metrics:
            if metric.score >= 0.8:
                strengths.append(f"{metric.dimension.value}: {metric.score:.2f}")
            elif metric.score <= 0.4:
                weaknesses.append(f"{metric.dimension.value}: {metric.score:.2f}")
        
        return strengths, weaknesses
    
    def _generate_general_recommendations(self, quality_metrics: List[QualityMetric], overall_score: float) -> List[str]:
        """Generar recomendaciones generales"""
        recommendations = []
        
        # Recomendaciones basadas en score general
        if overall_score < 0.5:
            recommendations.append("El contenido necesita mejoras significativas en múltiples dimensiones")
        elif overall_score < 0.7:
            recommendations.append("El contenido es aceptable pero puede mejorarse")
        else:
            recommendations.append("El contenido tiene buena calidad general")
        
        # Recomendaciones específicas por dimensión
        for metric in quality_metrics:
            if metric.score < 0.5:
                recommendations.extend(metric.recommendations[:2])  # Top 2 recomendaciones
        
        return recommendations[:10]  # Máximo 10 recomendaciones
    
    async def _compare_with_benchmarks(
        self,
        dimension_scores: Dict[QualityDimension, float],
        content_type: ContentType
    ) -> Dict[str, Any]:
        """Comparar con benchmarks"""
        try:
            if not self.enable_benchmarking:
                return {}
            
            benchmark_key = content_type.value
            if benchmark_key not in self.quality_benchmarks:
                benchmark_key = "informational"  # Fallback
            
            benchmarks = self.quality_benchmarks[benchmark_key]
            
            comparison = {}
            for dimension, score in dimension_scores.items():
                benchmark_score = benchmarks.get(dimension.value, 0.5)
                difference = score - benchmark_score
                percentage_diff = (difference / benchmark_score) * 100 if benchmark_score > 0 else 0
                
                comparison[dimension.value] = {
                    "actual_score": score,
                    "benchmark_score": benchmark_score,
                    "difference": difference,
                    "percentage_difference": percentage_diff,
                    "status": "above" if difference > 0 else "below" if difference < -0.1 else "meets"
                }
            
            return {
                "benchmark_type": benchmark_key,
                "dimensions": comparison,
                "overall_benchmark_met": all(
                    comp["status"] in ["above", "meets"] 
                    for comp in comparison.values()
                )
            }
            
        except Exception as e:
            logger.error(f"Error comparing with benchmarks: {e}")
            return {}
    
    async def _generate_quality_insights(self, analysis: ContentQualityAnalysis):
        """Generar insights de calidad"""
        insights = []
        
        # Insight 1: Score general
        insight = QualityInsight(
            id=f"overall_quality_{analysis.document_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            insight_type="overall_quality",
            description=f"Calidad general: {analysis.quality_level.value} ({analysis.overall_score:.2f})",
            significance=analysis.overall_score,
            confidence=0.9,
            related_dimensions=list(analysis.dimension_scores.keys()),
            actionable_recommendations=analysis.recommendations[:5],
            benchmark_context=f"Comparado con estándares de {analysis.content_type.value}"
        )
        insights.append(insight)
        
        # Insight 2: Dimensión más fuerte
        strongest_dimension = max(analysis.dimension_scores.items(), key=lambda x: x[1])
        insight = QualityInsight(
            id=f"strongest_dimension_{analysis.document_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            insight_type="strength",
            description=f"Fortaleza principal: {strongest_dimension[0].value} ({strongest_dimension[1]:.2f})",
            significance=strongest_dimension[1],
            confidence=0.8,
            related_dimensions=[strongest_dimension[0]],
            actionable_recommendations=[f"Mantener el alto nivel en {strongest_dimension[0].value}"],
            benchmark_context="Fortaleza identificada"
        )
        insights.append(insight)
        
        # Insight 3: Dimensión más débil
        weakest_dimension = min(analysis.dimension_scores.items(), key=lambda x: x[1])
        if weakest_dimension[1] < 0.6:
            insight = QualityInsight(
                id=f"weakest_dimension_{analysis.document_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                insight_type="improvement",
                description=f"Área de mejora: {weakest_dimension[0].value} ({weakest_dimension[1]:.2f})",
                significance=1 - weakest_dimension[1],
                confidence=0.8,
                related_dimensions=[weakest_dimension[0]],
                actionable_recommendations=self._get_dimension_recommendations(weakest_dimension[0]),
                benchmark_context="Área que requiere atención"
            )
            insights.append(insight)
        
        # Almacenar insights
        for insight in insights:
            self.quality_insights[insight.id] = insight
    
    def _get_dimension_recommendations(self, dimension: QualityDimension) -> List[str]:
        """Obtener recomendaciones específicas por dimensión"""
        recommendations = {
            QualityDimension.READABILITY: [
                "Usar oraciones más cortas y simples",
                "Reducir el uso de palabras complejas",
                "Mejorar la estructura de párrafos"
            ],
            QualityDimension.COHERENCE: [
                "Agregar más conectores lógicos",
                "Mejorar las transiciones entre ideas",
                "Estructurar mejor el flujo de información"
            ],
            QualityDimension.CLARITY: [
                "Simplificar el lenguaje técnico",
                "Usar ejemplos para explicar conceptos complejos",
                "Evitar la voz pasiva excesiva"
            ],
            QualityDimension.COMPLETENESS: [
                "Agregar una introducción clara",
                "Incluir una conclusión",
                "Proporcionar más ejemplos y evidencia"
            ],
            QualityDimension.ACCURACY: [
                "Agregar citas y referencias",
                "Incluir datos específicos",
                "Verificar la consistencia en terminología"
            ],
            QualityDimension.RELEVANCE: [
                "Enfocarse más en el tema principal",
                "Agregar información más actual",
                "Mejorar la cobertura de temas clave"
            ],
            QualityDimension.ENGAGEMENT: [
                "Agregar preguntas para el lector",
                "Usar elementos visuales (negrita, listas)",
                "Incluir ejemplos y casos de estudio"
            ],
            QualityDimension.STRUCTURE: [
                "Agregar encabezados y subencabezados",
                "Organizar mejor los párrafos",
                "Usar listas para información estructurada"
            ],
            QualityDimension.STYLE: [
                "Mantener consistencia en el tono",
                "Variar la longitud de las oraciones",
                "Usar más voz activa"
            ]
        }
        
        return recommendations.get(dimension, ["Revisar y mejorar esta dimensión"])
    
    # Métodos de recomendaciones específicas por dimensión
    def _get_readability_recommendations(self, score: float, metrics: Dict[str, float]) -> List[str]:
        """Recomendaciones de legibilidad"""
        recommendations = []
        
        if score < 0.4:
            recommendations.extend([
                "Usar oraciones más cortas (objetivo: 15-20 palabras)",
                "Simplificar vocabulario complejo",
                "Dividir párrafos largos"
            ])
        elif score < 0.7:
            recommendations.extend([
                "Mejorar la estructura de oraciones",
                "Reducir el uso de jerga técnica"
            ])
        
        return recommendations
    
    def _get_coherence_recommendations(self, score: float, details: Dict[str, Any]) -> List[str]:
        """Recomendaciones de coherencia"""
        recommendations = []
        
        if score < 0.5:
            recommendations.extend([
                "Agregar más conectores lógicos",
                "Mejorar las transiciones entre párrafos",
                "Estructurar mejor el flujo de ideas"
            ])
        
        return recommendations
    
    def _get_clarity_recommendations(self, score: float, details: Dict[str, Any]) -> List[str]:
        """Recomendaciones de claridad"""
        recommendations = []
        
        if score < 0.5:
            recommendations.extend([
                "Simplificar el lenguaje",
                "Usar más ejemplos",
                "Evitar la voz pasiva excesiva"
            ])
        
        return recommendations
    
    def _get_completeness_recommendations(self, score: float, details: Dict[str, Any]) -> List[str]:
        """Recomendaciones de completitud"""
        recommendations = []
        
        if not details.get("has_introduction", False):
            recommendations.append("Agregar una introducción clara")
        
        if not details.get("has_conclusion", False):
            recommendations.append("Incluir una conclusión")
        
        if not details.get("has_examples", False):
            recommendations.append("Proporcionar más ejemplos")
        
        return recommendations
    
    def _get_accuracy_recommendations(self, score: float, details: Dict[str, Any]) -> List[str]:
        """Recomendaciones de precisión"""
        recommendations = []
        
        if not details.get("has_citations", False):
            recommendations.append("Agregar citas y referencias")
        
        if not details.get("has_numbers", False):
            recommendations.append("Incluir datos específicos")
        
        return recommendations
    
    def _get_relevance_recommendations(self, score: float, details: Dict[str, Any]) -> List[str]:
        """Recomendaciones de relevancia"""
        recommendations = []
        
        if score < 0.6:
            recommendations.extend([
                "Enfocarse más en el tema principal",
                "Agregar información más actual"
            ])
        
        return recommendations
    
    def _get_engagement_recommendations(self, score: float, details: Dict[str, Any]) -> List[str]:
        """Recomendaciones de engagement"""
        recommendations = []
        
        if details.get("questions_count", 0) < 2:
            recommendations.append("Agregar preguntas para el lector")
        
        if details.get("visual_elements", 0) < 2:
            recommendations.append("Usar más elementos visuales (negrita, listas)")
        
        return recommendations
    
    def _get_structure_recommendations(self, score: float, details: Dict[str, Any]) -> List[str]:
        """Recomendaciones de estructura"""
        recommendations = []
        
        if details.get("total_headers", 0) < 3:
            recommendations.append("Agregar más encabezados y subencabezados")
        
        if details.get("total_lists", 0) < 2:
            recommendations.append("Usar listas para organizar información")
        
        return recommendations
    
    def _get_style_recommendations(self, score: float, details: Dict[str, Any]) -> List[str]:
        """Recomendaciones de estilo"""
        recommendations = []
        
        if details.get("passive_voice_count", 0) > 3:
            recommendations.append("Usar más voz activa")
        
        if details.get("vocabulary_diversity", 0) < 0.3:
            recommendations.append("Variar más el vocabulario")
        
        return recommendations
    
    async def compare_content_quality(
        self,
        document_ids: List[str]
    ) -> Dict[str, Any]:
        """Comparar calidad de contenido entre documentos"""
        try:
            if len(document_ids) < 2:
                raise ValueError("Se necesitan al menos 2 documentos para comparar")
            
            # Obtener análisis
            analyses = []
            for doc_id in document_ids:
                analysis = next((a for a in self.quality_analyses.values() if a.document_id == doc_id), None)
                if analysis:
                    analyses.append(analysis)
                else:
                    logger.warning(f"Quality analysis not found for document {doc_id}")
            
            if len(analyses) < 2:
                raise ValueError("No hay suficientes análisis para comparar")
            
            # Calcular comparaciones
            quality_comparison = self._calculate_quality_comparison(analyses)
            dimension_comparison = self._calculate_dimension_comparison(analyses)
            benchmark_comparison = self._calculate_benchmark_comparison(analyses)
            
            return {
                "document_ids": document_ids,
                "analyses_count": len(analyses),
                "quality_comparison": quality_comparison,
                "dimension_comparison": dimension_comparison,
                "benchmark_comparison": benchmark_comparison,
                "comparison_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error comparing content quality: {e}")
            raise
    
    def _calculate_quality_comparison(self, analyses: List[ContentQualityAnalysis]) -> Dict[str, Any]:
        """Calcular comparación de calidad general"""
        overall_scores = [analysis.overall_score for analysis in analyses]
        quality_levels = [analysis.quality_level for analysis in analyses]
        
        return {
            "overall_scores": {
                analysis.document_id: analysis.overall_score for analysis in analyses
            },
            "quality_levels": {
                analysis.document_id: analysis.quality_level.value for analysis in analyses
            },
            "average_score": np.mean(overall_scores),
            "score_range": f"{min(overall_scores):.2f} - {max(overall_scores):.2f}",
            "best_document": max(analyses, key=lambda x: x.overall_score).document_id,
            "worst_document": min(analyses, key=lambda x: x.overall_score).document_id
        }
    
    def _calculate_dimension_comparison(self, analyses: List[ContentQualityAnalysis]) -> Dict[str, Any]:
        """Calcular comparación por dimensiones"""
        dimension_comparison = {}
        
        for dimension in QualityDimension:
            scores = [analysis.dimension_scores.get(dimension, 0) for analysis in analyses]
            if scores:
                dimension_comparison[dimension.value] = {
                    "scores": {
                        analysis.document_id: analysis.dimension_scores.get(dimension, 0)
                        for analysis in analyses
                    },
                    "average": np.mean(scores),
                    "range": f"{min(scores):.2f} - {max(scores):.2f}",
                    "best_document": analyses[np.argmax(scores)].document_id
                }
        
        return dimension_comparison
    
    def _calculate_benchmark_comparison(self, analyses: List[ContentQualityAnalysis]) -> Dict[str, Any]:
        """Calcular comparación con benchmarks"""
        benchmark_comparison = {}
        
        for analysis in analyses:
            if analysis.benchmark_comparison:
                benchmark_comparison[analysis.document_id] = {
                    "benchmark_type": analysis.benchmark_comparison.get("benchmark_type", "unknown"),
                    "benchmark_met": analysis.benchmark_comparison.get("overall_benchmark_met", False),
                    "dimensions_above_benchmark": sum(
                        1 for dim in analysis.benchmark_comparison.get("dimensions", {}).values()
                        if dim.get("status") == "above"
                    ),
                    "dimensions_below_benchmark": sum(
                        1 for dim in analysis.benchmark_comparison.get("dimensions", {}).values()
                        if dim.get("status") == "below"
                    )
                }
        
        return benchmark_comparison
    
    async def get_quality_summary(self) -> Dict[str, Any]:
        """Obtener resumen de análisis de calidad"""
        if not self.quality_analyses:
            return {"message": "No quality analyses available"}
        
        # Estadísticas generales
        total_analyses = len(self.quality_analyses)
        overall_scores = [analysis.overall_score for analysis in self.quality_analyses.values()]
        quality_levels = [analysis.quality_level for analysis in self.quality_analyses.values()]
        
        # Distribución de niveles de calidad
        quality_distribution = Counter(quality_levels)
        
        # Distribución de tipos de contenido
        content_types = [analysis.content_type for analysis in self.quality_analyses.values()]
        content_type_distribution = Counter(content_types)
        
        # Estadísticas por dimensión
        dimension_stats = {}
        for dimension in QualityDimension:
            scores = [analysis.dimension_scores.get(dimension, 0) for analysis in self.quality_analyses.values()]
            if scores:
                dimension_stats[dimension.value] = {
                    "average": np.mean(scores),
                    "std": np.std(scores),
                    "min": min(scores),
                    "max": max(scores)
                }
        
        return {
            "total_analyses": total_analyses,
            "average_overall_score": np.mean(overall_scores),
            "overall_score_std": np.std(overall_scores),
            "quality_distribution": {level.value: count for level, count in quality_distribution.items()},
            "content_type_distribution": {content_type.value: count for content_type, count in content_type_distribution.items()},
            "dimension_statistics": dimension_stats,
            "total_insights": len(self.quality_insights),
            "last_analysis": max([analysis.created_at for analysis in self.quality_analyses.values()]).isoformat()
        }
    
    async def export_quality_analysis(self, filepath: str = None) -> str:
        """Exportar análisis de calidad"""
        try:
            if filepath is None:
                filepath = f"exports/content_quality_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Crear directorio si no existe
            import os
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Preparar datos para exportación
            export_data = {
                "quality_analyses": {
                    analysis_id: {
                        "document_id": analysis.document_id,
                        "content_type": analysis.content_type.value,
                        "overall_score": analysis.overall_score,
                        "quality_level": analysis.quality_level.value,
                        "dimension_scores": {dim.value: score for dim, score in analysis.dimension_scores.items()},
                        "quality_metrics": [
                            {
                                "dimension": metric.dimension.value,
                                "score": metric.score,
                                "weight": metric.weight,
                                "details": metric.details,
                                "recommendations": metric.recommendations
                            }
                            for metric in analysis.quality_metrics
                        ],
                        "strengths": analysis.strengths,
                        "weaknesses": analysis.weaknesses,
                        "recommendations": analysis.recommendations,
                        "benchmark_comparison": analysis.benchmark_comparison,
                        "readability_metrics": analysis.readability_metrics,
                        "linguistic_features": analysis.linguistic_features,
                        "structural_features": analysis.structural_features,
                        "created_at": analysis.created_at.isoformat()
                    }
                    for analysis_id, analysis in self.quality_analyses.items()
                },
                "quality_insights": {
                    insight_id: {
                        "insight_type": insight.insight_type,
                        "description": insight.description,
                        "significance": insight.significance,
                        "confidence": insight.confidence,
                        "related_dimensions": [dim.value for dim in insight.related_dimensions],
                        "actionable_recommendations": insight.actionable_recommendations,
                        "benchmark_context": insight.benchmark_context,
                        "created_at": insight.created_at.isoformat()
                    }
                    for insight_id, insight in self.quality_insights.items()
                },
                "summary": await self.get_quality_summary(),
                "exported_at": datetime.now().isoformat()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Content quality analysis exported to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting quality analysis: {e}")
            raise

























