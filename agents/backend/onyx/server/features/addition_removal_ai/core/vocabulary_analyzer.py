"""
Vocabulary Analyzer - Sistema de análisis de vocabulario
"""

import logging
import re
from typing import Dict, Any, Optional, List
from collections import Counter

logger = logging.getLogger(__name__)


class VocabularyAnalyzer:
    """Analizador de vocabulario"""

    def __init__(self):
        """Inicializar analizador"""
        # Stop words
        self.stop_words = {
            'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
            'de', 'del', 'en', 'a', 'y', 'o', 'pero', 'si', 'no',
            'que', 'es', 'son', 'fue', 'ser', 'estar', 'tener',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is',
            'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do'
        }
        
        # Palabras técnicas comunes
        self.technical_words = {
            'tecnología', 'tecnológico', 'implementación', 'optimización',
            'configuración', 'especificación', 'documentación', 'algoritmo',
            'technology', 'implementation', 'optimization', 'configuration',
            'specification', 'documentation', 'algorithm', 'architecture'
        }

    def analyze(self, content: str) -> Dict[str, Any]:
        """
        Analizar vocabulario del contenido.

        Args:
            content: Contenido

        Returns:
            Análisis de vocabulario
        """
        words = self._extract_words(content)
        
        if not words:
            return {"error": "No se encontraron palabras"}
        
        # Análisis de diversidad
        diversity = self._analyze_diversity(words)
        
        # Análisis de complejidad
        complexity = self._analyze_complexity(words)
        
        # Análisis de frecuencia
        frequency = self._analyze_frequency(words)
        
        # Análisis de palabras técnicas
        technical = self._analyze_technical_words(words)
        
        # Análisis de longitud de palabras
        word_length = self._analyze_word_length(words)
        
        # Calcular score de vocabulario
        vocabulary_score = (
            diversity["score"] * 0.3 +
            complexity["score"] * 0.25 +
            frequency["score"] * 0.20 +
            technical["score"] * 0.15 +
            word_length["score"] * 0.10
        )
        
        return {
            "vocabulary_score": vocabulary_score,
            "diversity": diversity,
            "complexity": complexity,
            "frequency": frequency,
            "technical": technical,
            "word_length": word_length,
            "total_words": len(words),
            "unique_words": len(set(words)),
            "suggestions": self._generate_vocabulary_suggestions(
                vocabulary_score,
                diversity,
                complexity
            )
        }

    def _extract_words(self, content: str) -> List[str]:
        """Extraer palabras"""
        words = re.findall(r'\b\w+\b', content.lower())
        # Filtrar stop words
        words = [w for w in words if w not in self.stop_words]
        return words

    def _analyze_diversity(self, words: List[str]) -> Dict[str, Any]:
        """Analizar diversidad de vocabulario"""
        unique_words = len(set(words))
        total_words = len(words)
        
        if total_words == 0:
            return {"score": 0.0, "diversity_ratio": 0}
        
        diversity_ratio = unique_words / total_words
        
        # Score: más diversidad es mejor (hasta cierto punto)
        # Ideal: ratio entre 0.5-0.8
        if 0.5 <= diversity_ratio <= 0.8:
            score = 1.0
        elif diversity_ratio < 0.5:
            score = diversity_ratio / 0.5
        else:
            score = max(0.7, 1.0 - (diversity_ratio - 0.8) / 0.2)
        
        return {
            "score": score,
            "diversity_ratio": diversity_ratio,
            "unique_words": unique_words,
            "total_words": total_words
        }

    def _analyze_complexity(self, words: List[str]) -> Dict[str, Any]:
        """Analizar complejidad de vocabulario"""
        # Palabras largas (más de 6 caracteres)
        long_words = [w for w in words if len(w) > 6]
        long_word_ratio = len(long_words) / len(words) if words else 0
        
        # Palabras técnicas
        technical_count = sum(1 for w in words if w in self.technical_words)
        technical_ratio = technical_count / len(words) if words else 0
        
        # Score: algo de complejidad es bueno
        complexity_score = (long_word_ratio * 0.6 + technical_ratio * 0.4)
        score = min(1.0, complexity_score * 3)  # Normalizar
        
        return {
            "score": score,
            "long_word_ratio": long_word_ratio,
            "technical_ratio": technical_ratio,
            "long_words": len(long_words),
            "technical_words": technical_count
        }

    def _analyze_frequency(self, words: List[str]) -> Dict[str, Any]:
        """Analizar frecuencia de palabras"""
        word_freq = Counter(words)
        
        # Palabras más frecuentes
        most_common = word_freq.most_common(10)
        
        # Verificar si hay palabras muy repetidas
        if most_common:
            max_freq = most_common[0][1]
            max_ratio = max_freq / len(words) if words else 0
            
            # Score: menos repetición es mejor
            if max_ratio > 0.1:  # Más del 10% es una palabra
                score = 0.5
            elif max_ratio > 0.05:  # Más del 5%
                score = 0.7
            else:
                score = 1.0
        else:
            score = 0.5
        
        return {
            "score": score,
            "most_common": [{"word": word, "count": count} for word, count in most_common],
            "max_frequency_ratio": max_ratio if most_common else 0
        }

    def _analyze_technical_words(self, words: List[str]) -> Dict[str, Any]:
        """Analizar palabras técnicas"""
        technical_found = [w for w in words if w in self.technical_words]
        technical_ratio = len(technical_found) / len(words) if words else 0
        
        # Score: algo de palabras técnicas es bueno para contenido técnico
        score = min(1.0, technical_ratio * 10)  # Normalizar
        
        return {
            "score": score,
            "technical_count": len(technical_found),
            "technical_ratio": technical_ratio,
            "technical_words": list(set(technical_found))[:10]
        }

    def _analyze_word_length(self, words: List[str]) -> Dict[str, Any]:
        """Analizar longitud de palabras"""
        if not words:
            return {"score": 0.0, "avg_length": 0}
        
        lengths = [len(w) for w in words]
        avg_length = sum(lengths) / len(lengths)
        
        # Score: longitud promedio entre 4-6 caracteres es ideal
        if 4 <= avg_length <= 6:
            score = 1.0
        elif avg_length < 4:
            score = avg_length / 4
        else:
            score = max(0.5, 1.0 - (avg_length - 6) / 4)
        
        return {
            "score": score,
            "avg_length": avg_length,
            "min_length": min(lengths),
            "max_length": max(lengths)
        }

    def _generate_vocabulary_suggestions(
        self,
        vocabulary_score: float,
        diversity: Dict[str, Any],
        complexity: Dict[str, Any]
    ) -> List[str]:
        """Generar sugerencias de vocabulario"""
        suggestions = []
        
        if vocabulary_score < 0.6:
            suggestions.append("Considera mejorar la diversidad y complejidad del vocabulario.")
        
        if diversity.get("diversity_ratio", 0) < 0.4:
            suggestions.append("El vocabulario es muy repetitivo. Usa sinónimos y variaciones.")
        
        if complexity.get("score", 0) < 0.3:
            suggestions.append("El vocabulario es muy simple. Considera usar palabras más específicas.")
        
        return suggestions






