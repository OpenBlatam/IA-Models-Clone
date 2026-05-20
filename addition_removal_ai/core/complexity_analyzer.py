"""
Complexity Analyzer - Sistema de análisis de complejidad de texto
"""

import logging
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ComplexityLevel(Enum):
    """Niveles de complejidad"""
    VERY_SIMPLE = "very_simple"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


@dataclass
class ComplexityScore:
    """Puntuación de complejidad"""
    overall: float
    lexical: float
    syntactic: float
    semantic: float
    level: ComplexityLevel
    metrics: Dict[str, Any]


class ComplexityAnalyzer:
    """Analizador de complejidad de texto"""

    def __init__(self):
        """Inicializar analizador"""
        # Palabras complejas (más de 3 sílabas o palabras técnicas)
        self.complex_words = set()
        
        # Palabras técnicas comunes
        self.technical_words = {
            'tecnología', 'tecnológico', 'tecnológico', 'implementación',
            'optimización', 'configuración', 'especificación', 'documentación',
            'technology', 'implementation', 'optimization', 'configuration',
            'specification', 'documentation', 'algorithm', 'architecture'
        }

    def analyze(
        self,
        content: str,
        detailed: bool = False
    ) -> Dict[str, Any]:
        """
        Analizar complejidad del contenido.

        Args:
            content: Contenido a analizar
            detailed: Si incluir detalles

        Returns:
            Análisis de complejidad
        """
        if not content:
            return {
                "overall": 0.0,
                "level": ComplexityLevel.VERY_SIMPLE.value,
                "error": "Contenido vacío"
            }
        
        # Métricas básicas
        word_count = len(content.split())
        sentence_count = len(re.findall(r'[.!?]+', content))
        paragraph_count = len([p for p in content.split('\n\n') if p.strip()])
        char_count = len(content)
        
        # Análisis léxico
        lexical_complexity = self._analyze_lexical_complexity(content)
        
        # Análisis sintáctico
        syntactic_complexity = self._analyze_syntactic_complexity(content)
        
        # Análisis semántico
        semantic_complexity = self._analyze_semantic_complexity(content)
        
        # Calcular complejidad general
        overall = (
            lexical_complexity * 0.4 +
            syntactic_complexity * 0.35 +
            semantic_complexity * 0.25
        )
        
        # Determinar nivel
        if overall < 0.2:
            level = ComplexityLevel.VERY_SIMPLE
        elif overall < 0.4:
            level = ComplexityLevel.SIMPLE
        elif overall < 0.6:
            level = ComplexityLevel.MODERATE
        elif overall < 0.8:
            level = ComplexityLevel.COMPLEX
        else:
            level = ComplexityLevel.VERY_COMPLEX
        
        result = {
            "overall": overall,
            "level": level.value,
            "lexical": lexical_complexity,
            "syntactic": syntactic_complexity,
            "semantic": semantic_complexity,
            "metrics": {
                "word_count": word_count,
                "sentence_count": sentence_count,
                "paragraph_count": paragraph_count,
                "char_count": char_count,
                "avg_words_per_sentence": word_count / sentence_count if sentence_count > 0 else 0,
                "avg_chars_per_word": char_count / word_count if word_count > 0 else 0
            }
        }
        
        if detailed:
            result["detailed_metrics"] = self._get_detailed_metrics(content)
        
        return result

    def _analyze_lexical_complexity(self, content: str) -> float:
        """Analizar complejidad léxica"""
        words = content.lower().split()
        if not words:
            return 0.0
        
        # Contar palabras complejas (más de 3 sílabas estimadas)
        complex_count = 0
        technical_count = 0
        
        for word in words:
            word_clean = re.sub(r'[^\w]', '', word)
            
            # Estimar sílabas (aproximación)
            syllables = self._estimate_syllables(word_clean)
            if syllables > 3:
                complex_count += 1
            
            # Verificar palabras técnicas
            if word_clean in self.technical_words:
                technical_count += 1
        
        # Calcular ratio
        complex_ratio = complex_count / len(words)
        technical_ratio = technical_count / len(words)
        
        # Combinar ratios
        lexical_score = min(1.0, (complex_ratio * 0.7 + technical_ratio * 0.3) * 2)
        
        return lexical_score

    def _analyze_syntactic_complexity(self, content: str) -> float:
        """Analizar complejidad sintáctica"""
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        # Longitud promedio de oraciones
        avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
        
        # Número de comas (subordinadas)
        comma_count = content.count(',')
        comma_ratio = comma_count / len(sentences) if sentences else 0
        
        # Palabras de conexión complejas
        complex_connectors = {
            'sin embargo', 'no obstante', 'por lo tanto', 'en consecuencia',
            'además', 'asimismo', 'por otro lado', 'mientras tanto',
            'however', 'nevertheless', 'therefore', 'consequently',
            'furthermore', 'moreover', 'meanwhile', 'whereas'
        }
        
        connector_count = sum(
            1 for connector in complex_connectors
            if connector in content.lower()
        )
        
        # Calcular score
        length_score = min(1.0, avg_length / 30)  # Normalizar a 30 palabras
        comma_score = min(1.0, comma_ratio / 3)  # Normalizar a 3 comas por oración
        connector_score = min(1.0, connector_count / 5)  # Normalizar a 5 conectores
        
        syntactic_score = (length_score * 0.5 + comma_score * 0.3 + connector_score * 0.2)
        
        return syntactic_score

    def _analyze_semantic_complexity(self, content: str) -> float:
        """Analizar complejidad semántica"""
        words = content.lower().split()
        if not words:
            return 0.0
        
        # Diversidad léxica (ratio de palabras únicas)
        unique_words = len(set(words))
        diversity_ratio = unique_words / len(words) if words else 0
        
        # Palabras abstractas comunes
        abstract_words = {
            'concepto', 'idea', 'teoría', 'principio', 'método', 'proceso',
            'concept', 'idea', 'theory', 'principle', 'method', 'process',
            'abstracción', 'abstraction', 'filosofía', 'philosophy'
        }
        
        abstract_count = sum(1 for word in words if word in abstract_words)
        abstract_ratio = abstract_count / len(words) if words else 0
        
        # Calcular score
        semantic_score = (diversity_ratio * 0.6 + abstract_ratio * 0.4)
        
        return semantic_score

    def _estimate_syllables(self, word: str) -> int:
        """Estimar número de sílabas"""
        if not word:
            return 0
        
        # Aproximación simple: contar grupos de vocales
        vowels = re.findall(r'[aeiouáéíóúüAEIOUÁÉÍÓÚÜ]+', word)
        return len(vowels) if vowels else 1

    def _get_detailed_metrics(self, content: str) -> Dict[str, Any]:
        """Obtener métricas detalladas"""
        words = content.split()
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return {
            "unique_words": len(set(words)),
            "total_words": len(words),
            "vocabulary_richness": len(set(words)) / len(words) if words else 0,
            "longest_sentence": max((len(s.split()) for s in sentences), default=0),
            "shortest_sentence": min((len(s.split()) for s in sentences), default=0),
            "avg_sentence_length": sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        }






