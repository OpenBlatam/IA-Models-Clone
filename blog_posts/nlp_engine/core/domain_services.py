from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import re
from typing import List, Dict, Any, Optional
from .entities import AnalysisScore, TextFingerprint
from .enums import AnalysisType, ProcessingTier
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
🎯 DOMAIN SERVICES - Servicios del Dominio NLP
=============================================

Servicios que encapsulan lógica de dominio que no pertenece
a ninguna entidad específica.
"""



class ScoreValidator:
    """Servicio de dominio para validación de scores."""
    
    @staticmethod
    def validate_sentiment_score(score: float, method: str) -> bool:
        """Validar score de sentimiento según método."""
        if method == "lexicon_based":
            # Métodos léxicos tienden a ser más extremos
            return 0 <= score <= 100
        elif method == "ml_based":
            # Métodos ML tienden a ser más conservadores
            return 10 <= score <= 90
        else:
            return 0 <= score <= 100
    
    @staticmethod
    def validate_quality_score(score: float, text_length: int) -> bool:
        """Validar score de calidad según longitud del texto."""
        if text_length < 10:
            # Textos muy cortos no pueden tener calidad muy alta
            return score <= 60
        elif text_length > 1000:
            # Textos largos pueden tener calidad alta si están bien estructurados
            return 0 <= score <= 100
        else:
            return 0 <= score <= 100
    
    @staticmethod
    def calculate_confidence_adjustment(
        score: AnalysisScore, 
        text_length: int,
        analysis_type: AnalysisType
    ) -> float:
        """Calcular ajuste de confianza basado en contexto."""
        base_confidence = score.confidence
        
        # Ajuste por longitud de texto
        if text_length < 5:
            base_confidence *= 0.5  # Muy poco texto
        elif text_length < 20:
            base_confidence *= 0.7  # Poco texto
        elif text_length > 1000:
            base_confidence *= 0.9  # Mucho texto puede ser ruidoso
        
        # Ajuste por tipo de análisis
        if analysis_type == AnalysisType.SENTIMENT and text_length < 10:
            base_confidence *= 0.6  # Sentimiento necesita más contexto
        elif analysis_type == AnalysisType.QUALITY_ASSESSMENT and text_length < 50:
            base_confidence *= 0.7  # Calidad necesita contenido suficiente
        
        return min(1.0, max(0.0, base_confidence))


class TextProcessor:
    """Servicio de dominio para procesamiento de texto."""
    
    @staticmethod
    def extract_basic_features(text: str) -> Dict[str, Any]:
        """Extraer características básicas del texto."""
        if not text:
            return {
                'word_count': 0,
                'sentence_count': 0,
                'avg_word_length': 0,
                'has_punctuation': False,
                'has_uppercase': False,
                'has_numbers': False
            }
        
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        return {
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
            'has_punctuation': bool(re.search(r'[.!?,:;]', text)),
            'has_uppercase': any(c.isupper() for c in text),
            'has_numbers': bool(re.search(r'\d', text)),
            'character_count': len(text),
            'unique_words': len(set(word.lower() for word in words))
        }
    
    @staticmethod
    def detect_language_hints(text: str) -> Optional[str]:
        """Detectar posibles idiomas basado en patrones simples."""
        if not text:
            return None
        
        # Patrones básicos para detección rápida
        text_lower = text.lower()
        
        # Palabras comunes en español
        spanish_words = {'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'para', 'al', 'está', 'todo', 'como', 'uno', 'bien', 'puede', 'este', 'del', 'ha', 'sido', 'más', 'muy', 'pero', 'ya', 'que', 'está', 'aquí', 'ser', 'hacer', 'día', 'vida', 'casa', 'agua', 'tiempo', 'trabajo', 'ciudad', 'historia', 'libro', 'música'}
        
        # Palabras comunes en inglés
        english_words = {'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us'}
        
        words = text_lower.split()
        if not words:
            return None
        
        spanish_count = sum(1 for word in words if word in spanish_words)
        english_count = sum(1 for word in words if word in english_words)
        
        spanish_ratio = spanish_count / len(words)
        english_ratio = english_count / len(words)
        
        if spanish_ratio > 0.3 and spanish_ratio > english_ratio:
            return 'es'
        elif english_ratio > 0.3 and english_ratio > spanish_ratio:
            return 'en'
        
        return None
    
    @staticmethod
    def sanitize_text(text: str) -> str:
        """Sanitizar texto para procesamiento seguro."""
        if not text:
            return ""
        
        # Remover caracteres de control excepto espacios y saltos de línea
        sanitized = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
        
        # Normalizar espacios en blanco
        sanitized = re.sub(r'\s+', ' ', sanitized)
        
        # Trim
        sanitized = sanitized.strip()
        
        return sanitized
    
    @staticmethod
    def calculate_readability_score(text: str) -> float:
        """Calcular score básico de legibilidad."""
        features = TextProcessor.extract_basic_features(text)
        
        if features['word_count'] == 0:
            return 0.0
        
        # Factores de legibilidad básicos
        avg_sentence_length = features['word_count'] / max(features['sentence_count'], 1)
        avg_word_length = features['avg_word_length']
        
        # Score basado en longitud promedio de oraciones y palabras
        # Oraciones más cortas y palabras más cortas = mejor legibilidad
        sentence_score = max(0, 100 - (avg_sentence_length - 15) * 3)
        word_score = max(0, 100 - (avg_word_length - 5) * 10)
        
        # Bonus por puntuación
        punctuation_bonus = 10 if features['has_punctuation'] else 0
        
        # Score final ponderado
        readability = (sentence_score * 0.5 + word_score * 0.4) + punctuation_bonus
        
        return min(100.0, max(0.0, readability))


class AnalysisOrchestrator:
    """Servicio de dominio para orquestar múltiples análisis."""
    
    @staticmethod
    def determine_optimal_tier(
        text_length: int, 
        analysis_types: List[AnalysisType],
        performance_requirement: Optional[str] = None
    ) -> ProcessingTier:
        """Determinar tier óptimo basado en requerimientos."""
        
        # Requerimiento explícito
        if performance_requirement:
            tier_map = {
                'ultra_fast': ProcessingTier.ULTRA_FAST,
                'balanced': ProcessingTier.BALANCED,
                'high_quality': ProcessingTier.HIGH_QUALITY,
                'research': ProcessingTier.RESEARCH_GRADE
            }
            return tier_map.get(performance_requirement, ProcessingTier.BALANCED)
        
        # Texto muy corto -> ultra fast
        if text_length < 20:
            return ProcessingTier.ULTRA_FAST
        
        # Muchos análisis complejos -> tier alto
        complex_analyses = {
            AnalysisType.SEMANTIC_SIMILARITY,
            AnalysisType.ENTITY_EXTRACTION,
            AnalysisType.SUMMARIZATION
        }
        
        if any(at in complex_analyses for at in analysis_types):
            return ProcessingTier.HIGH_QUALITY
        
        # Texto largo -> balanced
        if text_length > 500:
            return ProcessingTier.BALANCED
        
        # Default
        return ProcessingTier.BALANCED
    
    @staticmethod
    def prioritize_analysis_types(
        analysis_types: List[AnalysisType], 
        tier: ProcessingTier
    ) -> List[AnalysisType]:
        """Priorizar tipos de análisis según tier."""
        
        # Definir prioridades por tier
        priorities = {
            ProcessingTier.ULTRA_FAST: [
                AnalysisType.SENTIMENT,
                AnalysisType.QUALITY_ASSESSMENT,
                AnalysisType.READABILITY
            ],
            ProcessingTier.BALANCED: [
                AnalysisType.SENTIMENT,
                AnalysisType.QUALITY_ASSESSMENT,
                AnalysisType.READABILITY,
                AnalysisType.LANGUAGE_DETECTION,
                AnalysisType.KEYWORDS
            ],
            ProcessingTier.HIGH_QUALITY: [
                AnalysisType.SENTIMENT,
                AnalysisType.QUALITY_ASSESSMENT,
                AnalysisType.ENTITY_EXTRACTION,
                AnalysisType.SEMANTIC_SIMILARITY,
                AnalysisType.KEYWORDS,
                AnalysisType.READABILITY,
                AnalysisType.LANGUAGE_DETECTION
            ],
            ProcessingTier.RESEARCH_GRADE: list(AnalysisType)  # Todos
        }
        
        priority_order = priorities.get(tier, list(AnalysisType))
        
        # Ordenar analysis_types según prioridad
        prioritized = []
        for priority_type in priority_order:
            if priority_type in analysis_types:
                prioritized.append(priority_type)
        
        # Añadir tipos no priorizados al final
        for analysis_type in analysis_types:
            if analysis_type not in prioritized:
                prioritized.append(analysis_type)
        
        return prioritized 