"""
Advanced Text Quality Evaluation System for AI History Comparison
Sistema avanzado de evaluación de calidad de texto para análisis de historial de IA
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import re
from collections import Counter, defaultdict
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QualityDimension(Enum):
    """Dimensiones de calidad del texto"""
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

@dataclass
class QualityMetric:
    """Métrica de calidad individual"""
    dimension: QualityDimension
    score: float
    weight: float
    explanation: str
    suggestions: List[str]
    confidence: float

@dataclass
class QualityReport:
    """Reporte completo de calidad"""
    document_id: str
    overall_score: float
    quality_level: QualityLevel
    dimensions: Dict[QualityDimension, QualityMetric]
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    benchmark_comparison: Dict[str, float]
    generated_at: datetime = field(default_factory=datetime.now)

@dataclass
class QualityBenchmark:
    """Benchmark de calidad"""
    dimension: QualityDimension
    excellent_threshold: float
    good_threshold: float
    average_threshold: float
    poor_threshold: float
    industry_standard: float
    best_practices: List[str]

class AdvancedTextQualityEvaluator:
    """
    Evaluador avanzado de calidad de texto
    """
    
    def __init__(
        self,
        enable_ai_evaluation: bool = True,
        enable_benchmarking: bool = True,
        custom_weights: Optional[Dict[QualityDimension, float]] = None
    ):
        self.enable_ai_evaluation = enable_ai_evaluation
        self.enable_benchmarking = enable_benchmarking
        
        # Pesos por defecto para dimensiones de calidad
        self.default_weights = {
            QualityDimension.READABILITY: 0.15,
            QualityDimension.COHERENCE: 0.20,
            QualityDimension.CLARITY: 0.15,
            QualityDimension.COMPLETENESS: 0.10,
            QualityDimension.ACCURACY: 0.15,
            QualityDimension.RELEVANCE: 0.10,
            QualityDimension.ENGAGEMENT: 0.05,
            QualityDimension.STRUCTURE: 0.05,
            QualityDimension.STYLE: 0.03,
            QualityDimension.ORIGINALITY: 0.02
        }
        
        self.weights = custom_weights or self.default_weights
        
        # Inicializar benchmarks
        self.benchmarks = self._initialize_benchmarks()
        
        # Patrones de calidad
        self._initialize_quality_patterns()
        
        # Almacenamiento de evaluaciones
        self.evaluations: Dict[str, QualityReport] = {}
        
        # Configuración
        self.config = {
            "min_text_length": 50,
            "max_text_length": 10000,
            "sentence_complexity_threshold": 20,
            "paragraph_length_threshold": 150,
            "repetition_threshold": 0.1,
            "passive_voice_threshold": 0.3
        }
    
    def _initialize_benchmarks(self) -> Dict[QualityDimension, QualityBenchmark]:
        """Inicializar benchmarks de calidad"""
        benchmarks = {}
        
        # Benchmark de legibilidad
        benchmarks[QualityDimension.READABILITY] = QualityBenchmark(
            dimension=QualityDimension.READABILITY,
            excellent_threshold=0.8,
            good_threshold=0.6,
            average_threshold=0.4,
            poor_threshold=0.2,
            industry_standard=0.5,
            best_practices=[
                "Usar oraciones cortas y claras",
                "Evitar jerga técnica innecesaria",
                "Usar párrafos bien estructurados",
                "Incluir transiciones entre ideas"
            ]
        )
        
        # Benchmark de coherencia
        benchmarks[QualityDimension.COHERENCE] = QualityBenchmark(
            dimension=QualityDimension.COHERENCE,
            excellent_threshold=0.9,
            good_threshold=0.7,
            average_threshold=0.5,
            poor_threshold=0.3,
            industry_standard=0.6,
            best_practices=[
                "Mantener un hilo conductor claro",
                "Usar conectores lógicos",
                "Evitar saltos abruptos de tema",
                "Estructurar ideas de forma lógica"
            ]
        )
        
        # Benchmark de claridad
        benchmarks[QualityDimension.CLARITY] = QualityBenchmark(
            dimension=QualityDimension.CLARITY,
            excellent_threshold=0.85,
            good_threshold=0.65,
            average_threshold=0.45,
            poor_threshold=0.25,
            industry_standard=0.55,
            best_practices=[
                "Usar lenguaje simple y directo",
                "Definir términos técnicos",
                "Evitar ambigüedades",
                "Usar ejemplos cuando sea necesario"
            ]
        )
        
        # Benchmark de completitud
        benchmarks[QualityDimension.COMPLETENESS] = QualityBenchmark(
            dimension=QualityDimension.COMPLETENESS,
            excellent_threshold=0.9,
            good_threshold=0.7,
            average_threshold=0.5,
            poor_threshold=0.3,
            industry_standard=0.6,
            best_practices=[
                "Cubrir todos los aspectos del tema",
                "Incluir información relevante",
                "Proporcionar contexto suficiente",
                "Responder a preguntas implícitas"
            ]
        )
        
        # Benchmark de precisión
        benchmarks[QualityDimension.ACCURACY] = QualityBenchmark(
            dimension=QualityDimension.ACCURACY,
            excellent_threshold=0.95,
            good_threshold=0.8,
            average_threshold=0.6,
            poor_threshold=0.4,
            industry_standard=0.7,
            best_practices=[
                "Verificar hechos y datos",
                "Usar fuentes confiables",
                "Evitar información obsoleta",
                "Corregir errores tipográficos"
            ]
        )
        
        # Benchmark de relevancia
        benchmarks[QualityDimension.RELEVANCE] = QualityBenchmark(
            dimension=QualityDimension.RELEVANCE,
            excellent_threshold=0.9,
            good_threshold=0.7,
            average_threshold=0.5,
            poor_threshold=0.3,
            industry_standard=0.6,
            best_practices=[
                "Mantener el foco en el tema principal",
                "Evitar información irrelevante",
                "Conectar ideas con el objetivo",
                "Priorizar información importante"
            ]
        )
        
        # Benchmark de engagement
        benchmarks[QualityDimension.ENGAGEMENT] = QualityBenchmark(
            dimension=QualityDimension.ENGAGEMENT,
            excellent_threshold=0.8,
            good_threshold=0.6,
            average_threshold=0.4,
            poor_threshold=0.2,
            industry_standard=0.5,
            best_practices=[
                "Usar un tono apropiado",
                "Incluir elementos interactivos",
                "Hacer el contenido atractivo",
                "Usar storytelling cuando sea apropiado"
            ]
        )
        
        # Benchmark de estructura
        benchmarks[QualityDimension.STRUCTURE] = QualityBenchmark(
            dimension=QualityDimension.STRUCTURE,
            excellent_threshold=0.85,
            good_threshold=0.65,
            average_threshold=0.45,
            poor_threshold=0.25,
            industry_standard=0.55,
            best_practices=[
                "Usar títulos y subtítulos",
                "Organizar información lógicamente",
                "Usar listas cuando sea apropiado",
                "Mantener consistencia en formato"
            ]
        )
        
        # Benchmark de estilo
        benchmarks[QualityDimension.STYLE] = QualityBenchmark(
            dimension=QualityDimension.STYLE,
            excellent_threshold=0.8,
            good_threshold=0.6,
            average_threshold=0.4,
            poor_threshold=0.2,
            industry_standard=0.5,
            best_practices=[
                "Mantener consistencia en tono",
                "Usar variedad en estructura de oraciones",
                "Evitar repeticiones excesivas",
                "Adaptar el estilo al público objetivo"
            ]
        )
        
        # Benchmark de originalidad
        benchmarks[QualityDimension.ORIGINALITY] = QualityBenchmark(
            dimension=QualityDimension.ORIGINALITY,
            excellent_threshold=0.9,
            good_threshold=0.7,
            average_threshold=0.5,
            poor_threshold=0.3,
            industry_standard=0.6,
            best_practices=[
                "Proporcionar perspectivas únicas",
                "Evitar contenido duplicado",
                "Agregar valor original",
                "Usar ejemplos propios"
            ]
        )
        
        return benchmarks
    
    def _initialize_quality_patterns(self):
        """Inicializar patrones de calidad"""
        # Patrones positivos
        self.positive_patterns = {
            "clear_transitions": [
                r"\b(además|por otro lado|sin embargo|por lo tanto|en consecuencia)\b",
                r"\b(furthermore|however|therefore|consequently|moreover)\b"
            ],
            "specific_examples": [
                r"\b(por ejemplo|como ejemplo|específicamente|en particular)\b",
                r"\b(for example|such as|specifically|in particular)\b"
            ],
            "logical_structure": [
                r"\b(primero|segundo|tercero|finalmente|en conclusión)\b",
                r"\b(first|second|third|finally|in conclusion)\b"
            ],
            "active_voice": [
                r"\b(realizamos|desarrollamos|implementamos|creamos)\b",
                r"\b(we perform|we develop|we implement|we create)\b"
            ]
        }
        
        # Patrones negativos
        self.negative_patterns = {
            "passive_voice": [
                r"\b(es realizado|fue desarrollado|será implementado)\b",
                r"\b(is performed|was developed|will be implemented)\b"
            ],
            "vague_language": [
                r"\b(algo|algunos|varios|muchos|pocos)\b",
                r"\b(something|some|various|many|few)\b"
            ],
            "redundancy": [
                r"\b(completamente lleno|totalmente vacío|absolutamente necesario)\b",
                r"\b(completely full|totally empty|absolutely necessary)\b"
            ],
            "weak_modifiers": [
                r"\b(muy|bastante|algo|un poco)\b",
                r"\b(very|quite|somewhat|a bit)\b"
            ]
        }
    
    async def evaluate_text_quality(
        self,
        text: str,
        document_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> QualityReport:
        """
        Evaluar la calidad del texto
        
        Args:
            text: Texto a evaluar
            document_id: ID del documento
            context: Contexto adicional para la evaluación
            
        Returns:
            Reporte de calidad completo
        """
        try:
            logger.info(f"Evaluating text quality for document {document_id}")
            
            # Validar texto
            if not self._validate_text(text):
                raise ValueError("Text does not meet minimum requirements")
            
            # Evaluar cada dimensión
            dimensions = {}
            for dimension in QualityDimension:
                metric = await self._evaluate_dimension(text, dimension, context)
                dimensions[dimension] = metric
            
            # Calcular score general
            overall_score = self._calculate_overall_score(dimensions)
            
            # Determinar nivel de calidad
            quality_level = self._determine_quality_level(overall_score)
            
            # Generar fortalezas y debilidades
            strengths, weaknesses = self._identify_strengths_weaknesses(dimensions)
            
            # Generar recomendaciones
            recommendations = self._generate_recommendations(dimensions, context)
            
            # Comparación con benchmarks
            benchmark_comparison = self._compare_with_benchmarks(dimensions)
            
            # Crear reporte
            report = QualityReport(
                document_id=document_id,
                overall_score=overall_score,
                quality_level=quality_level,
                dimensions=dimensions,
                strengths=strengths,
                weaknesses=weaknesses,
                recommendations=recommendations,
                benchmark_comparison=benchmark_comparison
            )
            
            # Almacenar evaluación
            self.evaluations[document_id] = report
            
            logger.info(f"Quality evaluation completed for document {document_id}")
            return report
            
        except Exception as e:
            logger.error(f"Error evaluating text quality: {e}")
            raise
    
    def _validate_text(self, text: str) -> bool:
        """Validar que el texto cumple los requisitos mínimos"""
        if not text or len(text.strip()) < self.config["min_text_length"]:
            return False
        
        if len(text) > self.config["max_text_length"]:
            return False
        
        return True
    
    async def _evaluate_dimension(
        self,
        text: str,
        dimension: QualityDimension,
        context: Optional[Dict[str, Any]] = None
    ) -> QualityMetric:
        """Evaluar una dimensión específica de calidad"""
        
        if dimension == QualityDimension.READABILITY:
            return await self._evaluate_readability(text)
        elif dimension == QualityDimension.COHERENCE:
            return await self._evaluate_coherence(text)
        elif dimension == QualityDimension.CLARITY:
            return await self._evaluate_clarity(text)
        elif dimension == QualityDimension.COMPLETENESS:
            return await self._evaluate_completeness(text, context)
        elif dimension == QualityDimension.ACCURACY:
            return await self._evaluate_accuracy(text, context)
        elif dimension == QualityDimension.RELEVANCE:
            return await self._evaluate_relevance(text, context)
        elif dimension == QualityDimension.ENGAGEMENT:
            return await self._evaluate_engagement(text)
        elif dimension == QualityDimension.STRUCTURE:
            return await self._evaluate_structure(text)
        elif dimension == QualityDimension.STYLE:
            return await self._evaluate_style(text)
        elif dimension == QualityDimension.ORIGINALITY:
            return await self._evaluate_originality(text, context)
        else:
            raise ValueError(f"Unknown quality dimension: {dimension}")
    
    async def _evaluate_readability(self, text: str) -> QualityMetric:
        """Evaluar legibilidad del texto"""
        # Calcular métricas de legibilidad
        sentences = self._split_sentences(text)
        words = self._split_words(text)
        
        if not sentences or not words:
            return QualityMetric(
                dimension=QualityDimension.READABILITY,
                score=0.0,
                weight=self.weights[QualityDimension.READABILITY],
                explanation="No se pudo analizar la legibilidad",
                suggestions=["Asegúrate de que el texto tenga oraciones y palabras"],
                confidence=0.0
            )
        
        # Longitud promedio de oraciones
        avg_sentence_length = len(words) / len(sentences)
        
        # Longitud promedio de palabras
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Complejidad de sílabas (aproximada)
        complex_words = sum(1 for word in words if len(word) > 6)
        complexity_ratio = complex_words / len(words)
        
        # Calcular score de legibilidad (0-1)
        readability_score = 1.0
        
        # Penalizar oraciones muy largas
        if avg_sentence_length > 20:
            readability_score -= 0.3
        elif avg_sentence_length > 15:
            readability_score -= 0.1
        
        # Penalizar palabras muy largas
        if avg_word_length > 6:
            readability_score -= 0.2
        elif avg_word_length > 5:
            readability_score -= 0.1
        
        # Penalizar alta complejidad
        if complexity_ratio > 0.3:
            readability_score -= 0.2
        elif complexity_ratio > 0.2:
            readability_score -= 0.1
        
        # Asegurar que el score esté entre 0 y 1
        readability_score = max(0.0, min(1.0, readability_score))
        
        # Generar explicación y sugerencias
        explanation = f"Legibilidad basada en longitud promedio de oraciones ({avg_sentence_length:.1f} palabras) y complejidad de vocabulario"
        
        suggestions = []
        if avg_sentence_length > 20:
            suggestions.append("Considera dividir oraciones largas en oraciones más cortas")
        if avg_word_length > 6:
            suggestions.append("Usa palabras más simples cuando sea posible")
        if complexity_ratio > 0.3:
            suggestions.append("Reduce el uso de palabras técnicas complejas")
        
        return QualityMetric(
            dimension=QualityDimension.READABILITY,
            score=readability_score,
            weight=self.weights[QualityDimension.READABILITY],
            explanation=explanation,
            suggestions=suggestions,
            confidence=0.8
        )
    
    async def _evaluate_coherence(self, text: str) -> QualityMetric:
        """Evaluar coherencia del texto"""
        # Buscar conectores lógicos
        transition_words = 0
        for pattern_list in self.positive_patterns["clear_transitions"]:
            for pattern in pattern_list:
                matches = re.findall(pattern, text, re.IGNORECASE)
                transition_words += len(matches)
        
        # Calcular densidad de conectores
        words = self._split_words(text)
        transition_density = transition_words / len(words) if words else 0
        
        # Evaluar estructura lógica
        logical_structure_score = 0
        for pattern_list in self.positive_patterns["logical_structure"]:
            for pattern in pattern_list:
                if re.search(pattern, text, re.IGNORECASE):
                    logical_structure_score += 0.2
        
        # Calcular score de coherencia
        coherence_score = min(1.0, transition_density * 10 + logical_structure_score)
        
        explanation = f"Coherencia evaluada por conectores lógicos ({transition_words} encontrados) y estructura organizacional"
        
        suggestions = []
        if transition_density < 0.01:
            suggestions.append("Agrega más conectores lógicos entre ideas")
        if logical_structure_score < 0.4:
            suggestions.append("Usa marcadores de estructura como 'primero', 'segundo', 'finalmente'")
        
        return QualityMetric(
            dimension=QualityDimension.COHERENCE,
            score=coherence_score,
            weight=self.weights[QualityDimension.COHERENCE],
            explanation=explanation,
            suggestions=suggestions,
            confidence=0.7
        )
    
    async def _evaluate_clarity(self, text: str) -> QualityMetric:
        """Evaluar claridad del texto"""
        # Detectar lenguaje vago
        vague_words = 0
        for pattern_list in self.negative_patterns["vague_language"]:
            for pattern in pattern_list:
                matches = re.findall(pattern, text, re.IGNORECASE)
                vague_words += len(matches)
        
        # Detectar modificadores débiles
        weak_modifiers = 0
        for pattern_list in self.negative_patterns["weak_modifiers"]:
            for pattern in pattern_list:
                matches = re.findall(pattern, text, re.IGNORECASE)
                weak_modifiers += len(matches)
        
        # Detectar ejemplos específicos
        specific_examples = 0
        for pattern_list in self.positive_patterns["specific_examples"]:
            for pattern in pattern_list:
                matches = re.findall(pattern, text, re.IGNORECASE)
                specific_examples += len(matches)
        
        # Calcular score de claridad
        words = self._split_words(text)
        total_words = len(words)
        
        clarity_score = 1.0
        clarity_score -= (vague_words / total_words) * 0.5 if total_words > 0 else 0
        clarity_score -= (weak_modifiers / total_words) * 0.3 if total_words > 0 else 0
        clarity_score += (specific_examples / total_words) * 0.2 if total_words > 0 else 0
        
        clarity_score = max(0.0, min(1.0, clarity_score))
        
        explanation = f"Claridad evaluada por uso de lenguaje específico vs. vago ({vague_words} palabras vagas, {specific_examples} ejemplos específicos)"
        
        suggestions = []
        if vague_words > total_words * 0.05:
            suggestions.append("Reduce el uso de lenguaje vago como 'algo', 'algunos', 'varios'")
        if weak_modifiers > total_words * 0.03:
            suggestions.append("Evita modificadores débiles como 'muy', 'bastante', 'algo'")
        if specific_examples < 2:
            suggestions.append("Agrega más ejemplos específicos para clarificar conceptos")
        
        return QualityMetric(
            dimension=QualityDimension.CLARITY,
            score=clarity_score,
            weight=self.weights[QualityDimension.CLARITY],
            explanation=explanation,
            suggestions=suggestions,
            confidence=0.8
        )
    
    async def _evaluate_completeness(self, text: str, context: Optional[Dict[str, Any]] = None) -> QualityMetric:
        """Evaluar completitud del texto"""
        # Análisis básico de completitud
        sentences = self._split_sentences(text)
        words = self._split_words(text)
        
        # Verificar si tiene introducción, desarrollo y conclusión
        has_introduction = any(word in text.lower() for word in ["introducción", "introduction", "primero", "first"])
        has_conclusion = any(word in text.lower() for word in ["conclusión", "conclusion", "finalmente", "finally"])
        
        # Verificar longitud adecuada
        length_score = min(1.0, len(words) / 200)  # Esperamos al menos 200 palabras
        
        # Verificar diversidad de vocabulario
        unique_words = len(set(word.lower() for word in words))
        vocabulary_diversity = unique_words / len(words) if words else 0
        
        # Calcular score de completitud
        completeness_score = 0.0
        completeness_score += 0.3 if has_introduction else 0
        completeness_score += 0.3 if has_conclusion else 0
        completeness_score += length_score * 0.2
        completeness_score += vocabulary_diversity * 0.2
        
        explanation = f"Completitud evaluada por estructura ({'con introducción y conclusión' if has_introduction and has_conclusion else 'estructura incompleta'}), longitud ({len(words)} palabras) y diversidad de vocabulario"
        
        suggestions = []
        if not has_introduction:
            suggestions.append("Agrega una introducción clara al tema")
        if not has_conclusion:
            suggestions.append("Incluye una conclusión que resuma los puntos principales")
        if len(words) < 200:
            suggestions.append("Desarrolla más el contenido para mayor completitud")
        if vocabulary_diversity < 0.5:
            suggestions.append("Usa más variedad en el vocabulario")
        
        return QualityMetric(
            dimension=QualityDimension.COMPLETENESS,
            score=completeness_score,
            weight=self.weights[QualityDimension.COMPLETENESS],
            explanation=explanation,
            suggestions=suggestions,
            confidence=0.7
        )
    
    async def _evaluate_accuracy(self, text: str, context: Optional[Dict[str, Any]] = None) -> QualityMetric:
        """Evaluar precisión del texto"""
        # Detectar posibles errores tipográficos (básico)
        words = self._split_words(text)
        
        # Verificar palabras muy largas que podrían ser errores
        suspicious_words = [word for word in words if len(word) > 20]
        
        # Verificar repeticiones excesivas
        word_counts = Counter(word.lower() for word in words)
        max_repetition = max(word_counts.values()) if word_counts else 0
        repetition_ratio = max_repetition / len(words) if words else 0
        
        # Verificar uso de voz pasiva (puede indicar imprecisión)
        passive_voice = 0
        for pattern_list in self.negative_patterns["passive_voice"]:
            for pattern in pattern_list:
                matches = re.findall(pattern, text, re.IGNORECASE)
                passive_voice += len(matches)
        
        # Calcular score de precisión
        accuracy_score = 1.0
        accuracy_score -= len(suspicious_words) * 0.1
        accuracy_score -= repetition_ratio * 0.3
        accuracy_score -= (passive_voice / len(words)) * 0.2 if words else 0
        
        accuracy_score = max(0.0, min(1.0, accuracy_score))
        
        explanation = f"Precisión evaluada por errores potenciales ({len(suspicious_words)} palabras sospechosas), repeticiones ({repetition_ratio:.2f} ratio) y uso de voz pasiva"
        
        suggestions = []
        if suspicious_words:
            suggestions.append("Revisa palabras muy largas que podrían ser errores tipográficos")
        if repetition_ratio > 0.1:
            suggestions.append("Reduce la repetición excesiva de palabras")
        if passive_voice > len(words) * 0.1:
            suggestions.append("Usa más voz activa para mayor precisión")
        
        return QualityMetric(
            dimension=QualityDimension.ACCURACY,
            score=accuracy_score,
            weight=self.weights[QualityDimension.ACCURACY],
            explanation=explanation,
            suggestions=suggestions,
            confidence=0.6
        )
    
    async def _evaluate_relevance(self, text: str, context: Optional[Dict[str, Any]] = None) -> QualityMetric:
        """Evaluar relevancia del texto"""
        # Análisis básico de relevancia
        words = self._split_words(text)
        
        # Verificar si el texto mantiene el foco
        # Esto es una implementación simplificada
        focus_score = 0.8  # Placeholder
        
        # Verificar longitud apropiada para el tema
        length_appropriateness = min(1.0, len(words) / 300)  # Esperamos al menos 300 palabras para un tema completo
        
        # Calcular score de relevancia
        relevance_score = (focus_score + length_appropriateness) / 2
        
        explanation = f"Relevancia evaluada por mantenimiento del foco y longitud apropiada ({len(words)} palabras)"
        
        suggestions = []
        if len(words) < 300:
            suggestions.append("Desarrolla más el tema para mayor relevancia")
        
        return QualityMetric(
            dimension=QualityDimension.RELEVANCE,
            score=relevance_score,
            weight=self.weights[QualityDimension.RELEVANCE],
            explanation=explanation,
            suggestions=suggestions,
            confidence=0.5
        )
    
    async def _evaluate_engagement(self, text: str) -> QualityMetric:
        """Evaluar engagement del texto"""
        # Detectar elementos que aumentan el engagement
        questions = len(re.findall(r'\?', text))
        exclamations = len(re.findall(r'!', text))
        
        # Detectar uso de segunda persona
        second_person = len(re.findall(r'\b(tú|usted|you|your)\b', text, re.IGNORECASE))
        
        # Detectar storytelling elements
        storytelling_words = len(re.findall(r'\b(historia|ejemplo|caso|experiencia|story|example|case)\b', text, re.IGNORECASE))
        
        words = self._split_words(text)
        total_words = len(words)
        
        # Calcular score de engagement
        engagement_score = 0.0
        engagement_score += min(0.3, questions / total_words * 10) if total_words > 0 else 0
        engagement_score += min(0.2, exclamations / total_words * 10) if total_words > 0 else 0
        engagement_score += min(0.3, second_person / total_words * 10) if total_words > 0 else 0
        engagement_score += min(0.2, storytelling_words / total_words * 10) if total_words > 0 else 0
        
        explanation = f"Engagement evaluado por preguntas ({questions}), exclamaciones ({exclamations}), uso de segunda persona ({second_person}) y elementos narrativos ({storytelling_words})"
        
        suggestions = []
        if questions == 0:
            suggestions.append("Considera agregar preguntas retóricas para involucrar al lector")
        if second_person < 3:
            suggestions.append("Usa más segunda persona para conectar con el lector")
        if storytelling_words < 2:
            suggestions.append("Incluye ejemplos o historias para hacer el contenido más atractivo")
        
        return QualityMetric(
            dimension=QualityDimension.ENGAGEMENT,
            score=engagement_score,
            weight=self.weights[QualityDimension.ENGAGEMENT],
            explanation=explanation,
            suggestions=suggestions,
            confidence=0.7
        )
    
    async def _evaluate_structure(self, text: str) -> QualityMetric:
        """Evaluar estructura del texto"""
        # Detectar títulos y subtítulos
        headings = len(re.findall(r'^#{1,6}\s+', text, re.MULTILINE))
        
        # Detectar listas
        lists = len(re.findall(r'^\s*[-*•]\s+', text, re.MULTILINE))
        
        # Detectar párrafos
        paragraphs = len(text.split('\n\n'))
        
        # Calcular score de estructura
        structure_score = 0.0
        structure_score += min(0.4, headings * 0.1)
        structure_score += min(0.3, lists * 0.05)
        structure_score += min(0.3, min(1.0, paragraphs / 5))
        
        explanation = f"Estructura evaluada por títulos ({headings}), listas ({lists}) y párrafos ({paragraphs})"
        
        suggestions = []
        if headings == 0:
            suggestions.append("Agrega títulos y subtítulos para mejorar la estructura")
        if lists == 0:
            suggestions.append("Considera usar listas para organizar información")
        if paragraphs < 3:
            suggestions.append("Divide el texto en más párrafos para mejor legibilidad")
        
        return QualityMetric(
            dimension=QualityDimension.STRUCTURE,
            score=structure_score,
            weight=self.weights[QualityDimension.STRUCTURE],
            explanation=explanation,
            suggestions=suggestions,
            confidence=0.8
        )
    
    async def _evaluate_style(self, text: str) -> QualityMetric:
        """Evaluar estilo del texto"""
        # Detectar consistencia en tono
        formal_words = len(re.findall(r'\b(por lo tanto|sin embargo|además|furthermore|however|therefore)\b', text, re.IGNORECASE))
        informal_words = len(re.findall(r'\b(genial|increíble|awesome|amazing)\b', text, re.IGNORECASE))
        
        # Detectar variedad en estructura de oraciones
        sentences = self._split_sentences(text)
        sentence_lengths = [len(sentence.split()) for sentence in sentences]
        length_variety = np.std(sentence_lengths) / np.mean(sentence_lengths) if sentence_lengths else 0
        
        # Detectar repeticiones de estilo
        words = self._split_words(text)
        word_counts = Counter(word.lower() for word in words)
        style_repetition = sum(1 for count in word_counts.values() if count > 3) / len(words) if words else 0
        
        # Calcular score de estilo
        style_score = 1.0
        style_score -= style_repetition * 0.5
        style_score += min(0.3, length_variety * 0.1)
        
        # Bonus por consistencia de tono
        if formal_words > 0 and informal_words == 0:
            style_score += 0.1
        elif informal_words > 0 and formal_words == 0:
            style_score += 0.1
        
        style_score = max(0.0, min(1.0, style_score))
        
        explanation = f"Estilo evaluado por consistencia de tono, variedad en estructura de oraciones y repeticiones"
        
        suggestions = []
        if style_repetition > 0.1:
            suggestions.append("Reduce las repeticiones excesivas de palabras")
        if length_variety < 0.3:
            suggestions.append("Varía la longitud de las oraciones para mejor estilo")
        
        return QualityMetric(
            dimension=QualityDimension.STYLE,
            score=style_score,
            weight=self.weights[QualityDimension.STYLE],
            explanation=explanation,
            suggestions=suggestions,
            confidence=0.6
        )
    
    async def _evaluate_originality(self, text: str, context: Optional[Dict[str, Any]] = None) -> QualityMetric:
        """Evaluar originalidad del texto"""
        # Análisis básico de originalidad
        words = self._split_words(text)
        
        # Calcular diversidad de vocabulario
        unique_words = len(set(word.lower() for word in words))
        vocabulary_diversity = unique_words / len(words) if words else 0
        
        # Detectar frases comunes
        common_phrases = [
            "en conclusión", "in conclusion", "por ejemplo", "for example",
            "es importante", "it is important", "debe tenerse en cuenta", "it should be noted"
        ]
        
        common_phrase_count = sum(1 for phrase in common_phrases if phrase in text.lower())
        
        # Calcular score de originalidad
        originality_score = vocabulary_diversity
        originality_score -= common_phrase_count * 0.1
        
        originality_score = max(0.0, min(1.0, originality_score))
        
        explanation = f"Originalidad evaluada por diversidad de vocabulario ({vocabulary_diversity:.2f}) y uso de frases comunes ({common_phrase_count})"
        
        suggestions = []
        if vocabulary_diversity < 0.6:
            suggestions.append("Usa más variedad en el vocabulario para mayor originalidad")
        if common_phrase_count > 3:
            suggestions.append("Reduce el uso de frases muy comunes")
        
        return QualityMetric(
            dimension=QualityDimension.ORIGINALITY,
            score=originality_score,
            weight=self.weights[QualityDimension.ORIGINALITY],
            explanation=explanation,
            suggestions=suggestions,
            confidence=0.5
        )
    
    def _calculate_overall_score(self, dimensions: Dict[QualityDimension, QualityMetric]) -> float:
        """Calcular score general ponderado"""
        total_score = 0.0
        total_weight = 0.0
        
        for dimension, metric in dimensions.items():
            total_score += metric.score * metric.weight
            total_weight += metric.weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _determine_quality_level(self, overall_score: float) -> QualityLevel:
        """Determinar nivel de calidad basado en el score"""
        if overall_score >= 0.8:
            return QualityLevel.EXCELLENT
        elif overall_score >= 0.6:
            return QualityLevel.GOOD
        elif overall_score >= 0.4:
            return QualityLevel.AVERAGE
        elif overall_score >= 0.2:
            return QualityLevel.POOR
        else:
            return QualityLevel.VERY_POOR
    
    def _identify_strengths_weaknesses(
        self,
        dimensions: Dict[QualityDimension, QualityMetric]
    ) -> Tuple[List[str], List[str]]:
        """Identificar fortalezas y debilidades"""
        strengths = []
        weaknesses = []
        
        for dimension, metric in dimensions.items():
            if metric.score >= 0.7:
                strengths.append(f"{dimension.value}: {metric.explanation}")
            elif metric.score <= 0.4:
                weaknesses.append(f"{dimension.value}: {metric.explanation}")
        
        return strengths, weaknesses
    
    def _generate_recommendations(
        self,
        dimensions: Dict[QualityDimension, QualityMetric],
        context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Generar recomendaciones de mejora"""
        recommendations = []
        
        # Agregar sugerencias de dimensiones con score bajo
        for dimension, metric in dimensions.items():
            if metric.score < 0.6:
                recommendations.extend(metric.suggestions)
        
        # Agregar recomendaciones generales
        if len(recommendations) == 0:
            recommendations.append("El texto tiene buena calidad general, mantén el nivel actual")
        
        return list(set(recommendations))  # Remover duplicados
    
    def _compare_with_benchmarks(
        self,
        dimensions: Dict[QualityDimension, QualityMetric]
    ) -> Dict[str, float]:
        """Comparar con benchmarks de la industria"""
        comparison = {}
        
        for dimension, metric in dimensions.items():
            if dimension in self.benchmarks:
                benchmark = self.benchmarks[dimension]
                comparison[dimension.value] = {
                    "score": metric.score,
                    "industry_standard": benchmark.industry_standard,
                    "excellent_threshold": benchmark.excellent_threshold,
                    "above_industry": metric.score > benchmark.industry_standard,
                    "above_excellent": metric.score > benchmark.excellent_threshold
                }
        
        return comparison
    
    def _split_sentences(self, text: str) -> List[str]:
        """Dividir texto en oraciones"""
        # Implementación básica de división de oraciones
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_words(self, text: str) -> List[str]:
        """Dividir texto en palabras"""
        # Remover puntuación y dividir
        words = re.findall(r'\b\w+\b', text)
        return words
    
    async def get_quality_summary(self) -> Dict[str, Any]:
        """Obtener resumen de evaluaciones de calidad"""
        if not self.evaluations:
            return {"message": "No quality evaluations available"}
        
        # Estadísticas generales
        total_evaluations = len(self.evaluations)
        quality_levels = Counter([report.quality_level.value for report in self.evaluations.values()])
        
        # Scores promedio por dimensión
        avg_scores = {}
        for dimension in QualityDimension:
            scores = [report.dimensions[dimension].score for report in self.evaluations.values()]
            avg_scores[dimension.value] = np.mean(scores) if scores else 0.0
        
        # Score promedio general
        overall_scores = [report.overall_score for report in self.evaluations.values()]
        avg_overall_score = np.mean(overall_scores) if overall_scores else 0.0
        
        return {
            "total_evaluations": total_evaluations,
            "quality_level_distribution": dict(quality_levels),
            "average_scores_by_dimension": avg_scores,
            "average_overall_score": avg_overall_score,
            "last_evaluation": max([report.generated_at for report in self.evaluations.values()]).isoformat()
        }
    
    async def export_quality_report(self, document_id: str, filepath: str = None) -> str:
        """Exportar reporte de calidad a archivo"""
        if document_id not in self.evaluations:
            raise ValueError(f"Quality evaluation for document {document_id} not found")
        
        if filepath is None:
            filepath = f"exports/quality_report_{document_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Crear directorio si no existe
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        report = self.evaluations[document_id]
        
        # Convertir a diccionario
        report_data = {
            "document_id": report.document_id,
            "overall_score": report.overall_score,
            "quality_level": report.quality_level.value,
            "dimensions": {
                dim.value: {
                    "score": metric.score,
                    "weight": metric.weight,
                    "explanation": metric.explanation,
                    "suggestions": metric.suggestions,
                    "confidence": metric.confidence
                }
                for dim, metric in report.dimensions.items()
            },
            "strengths": report.strengths,
            "weaknesses": report.weaknesses,
            "recommendations": report.recommendations,
            "benchmark_comparison": report.benchmark_comparison,
            "generated_at": report.generated_at.isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Quality report exported to {filepath}")
        return filepath


























