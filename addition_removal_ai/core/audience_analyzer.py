"""
Audience Analyzer - Sistema de análisis de audiencia
"""

import logging
import re
from typing import Dict, Any, Optional, List
from collections import Counter

logger = logging.getLogger(__name__)


class AudienceAnalyzer:
    """Analizador de audiencia"""

    def __init__(self):
        """Inicializar analizador"""
        # Niveles de complejidad por audiencia
        self.audience_complexity = {
            "general": {"min": 0.0, "max": 1.0, "optimal": 0.5},
            "beginner": {"min": 0.0, "max": 0.4, "optimal": 0.2},
            "intermediate": {"min": 0.3, "max": 0.7, "optimal": 0.5},
            "advanced": {"min": 0.6, "max": 1.0, "optimal": 0.8},
            "expert": {"min": 0.8, "max": 1.0, "optimal": 0.9}
        }
        
        # Vocabulario por audiencia
        self.audience_vocabulary = {
            "beginner": {
                "simple_words": True,
                "technical_words": False,
                "jargon": False
            },
            "intermediate": {
                "simple_words": True,
                "technical_words": True,
                "jargon": False
            },
            "advanced": {
                "simple_words": True,
                "technical_words": True,
                "jargon": True
            },
            "expert": {
                "simple_words": False,
                "technical_words": True,
                "jargon": True
            }
        }

    def analyze_audience_fit(
        self,
        content: str,
        target_audience: str = "general"
    ) -> Dict[str, Any]:
        """
        Analizar si el contenido se ajusta a la audiencia objetivo.

        Args:
            content: Contenido
            target_audience: Audiencia objetivo

        Returns:
            Análisis de ajuste a audiencia
        """
        # Calcular complejidad del contenido
        complexity = self._calculate_complexity(content)
        
        # Obtener complejidad objetivo
        target_complexity = self.audience_complexity.get(
            target_audience,
            self.audience_complexity["general"]
        )
        
        # Verificar ajuste
        complexity_fit = self._check_complexity_fit(complexity, target_complexity)
        
        # Analizar vocabulario
        vocabulary_fit = self._check_vocabulary_fit(content, target_audience)
        
        # Analizar longitud
        length_fit = self._check_length_fit(content, target_audience)
        
        # Calcular score de ajuste
        fit_score = (
            complexity_fit["score"] * 0.4 +
            vocabulary_fit["score"] * 0.35 +
            length_fit["score"] * 0.25
        )
        
        return {
            "target_audience": target_audience,
            "fit_score": fit_score,
            "complexity": {
                "current": complexity,
                "target": target_complexity,
                "fit": complexity_fit
            },
            "vocabulary": vocabulary_fit,
            "length": length_fit,
            "suggestions": self._generate_audience_suggestions(
                fit_score,
                complexity_fit,
                vocabulary_fit,
                target_audience
            )
        }

    def _calculate_complexity(self, content: str) -> float:
        """Calcular complejidad del contenido"""
        words = content.split()
        sentences = [s for s in content.split('.') if s.strip()]
        
        if not words or not sentences:
            return 0.5
        
        # Longitud promedio de oraciones
        avg_sentence_length = len(words) / len(sentences)
        
        # Longitud promedio de palabras
        avg_word_length = sum(len(w) for w in words) / len(words)
        
        # Palabras técnicas
        technical_words = {
            'tecnología', 'implementación', 'optimización', 'algoritmo',
            'technology', 'implementation', 'optimization', 'algorithm'
        }
        technical_count = sum(1 for w in words if w.lower() in technical_words)
        technical_ratio = technical_count / len(words)
        
        # Calcular score de complejidad
        complexity = (
            min(1.0, avg_sentence_length / 30) * 0.4 +
            min(1.0, avg_word_length / 8) * 0.3 +
            min(1.0, technical_ratio * 10) * 0.3
        )
        
        return complexity

    def _check_complexity_fit(
        self,
        current_complexity: float,
        target_complexity: Dict[str, float]
    ) -> Dict[str, Any]:
        """Verificar ajuste de complejidad"""
        min_complexity = target_complexity["min"]
        max_complexity = target_complexity["max"]
        optimal_complexity = target_complexity["optimal"]
        
        if min_complexity <= current_complexity <= max_complexity:
            # Calcular score basado en proximidad al óptimo
            distance_from_optimal = abs(current_complexity - optimal_complexity)
            max_distance = max(
                optimal_complexity - min_complexity,
                max_complexity - optimal_complexity
            )
            score = 1.0 - (distance_from_optimal / max_distance) if max_distance > 0 else 1.0
            status = "fit"
        elif current_complexity < min_complexity:
            score = current_complexity / min_complexity if min_complexity > 0 else 0.0
            status = "too_simple"
        else:
            score = (max_complexity / current_complexity) if current_complexity > 0 else 0.0
            status = "too_complex"
        
        return {
            "score": score,
            "status": status,
            "current": current_complexity,
            "target_range": [min_complexity, max_complexity],
            "optimal": optimal_complexity
        }

    def _check_vocabulary_fit(
        self,
        content: str,
        target_audience: str
    ) -> Dict[str, Any]:
        """Verificar ajuste de vocabulario"""
        words = content.lower().split()
        
        # Palabras técnicas
        technical_words = {
            'tecnología', 'implementación', 'optimización', 'algoritmo',
            'technology', 'implementation', 'optimization', 'algorithm'
        }
        technical_count = sum(1 for w in words if w in technical_words)
        technical_ratio = technical_count / len(words) if words else 0
        
        # Jargon (palabras muy específicas)
        jargon_words = {
            'paradigma', 'heurística', 'metodología', 'framework',
            'paradigm', 'heuristic', 'methodology', 'framework'
        }
        jargon_count = sum(1 for w in words if w in jargon_words)
        jargon_ratio = jargon_count / len(words) if words else 0
        
        # Obtener requisitos de audiencia
        requirements = self.audience_vocabulary.get(
            target_audience,
            self.audience_vocabulary["intermediate"]
        )
        
        score = 1.0
        
        # Verificar palabras técnicas
        if not requirements.get("technical_words", True) and technical_ratio > 0.05:
            score -= 0.3
        
        # Verificar jargon
        if not requirements.get("jargon", False) and jargon_ratio > 0.02:
            score -= 0.3
        
        score = max(0.0, min(1.0, score))
        
        return {
            "score": score,
            "technical_ratio": technical_ratio,
            "jargon_ratio": jargon_ratio,
            "requirements": requirements
        }

    def _check_length_fit(
        self,
        content: str,
        target_audience: str
    ) -> Dict[str, Any]:
        """Verificar ajuste de longitud"""
        word_count = len(content.split())
        
        # Longitudes objetivo por audiencia
        target_lengths = {
            "beginner": {"min": 100, "max": 500, "optimal": 300},
            "intermediate": {"min": 300, "max": 1000, "optimal": 600},
            "advanced": {"min": 500, "max": 2000, "optimal": 1200},
            "expert": {"min": 800, "max": 3000, "optimal": 2000},
            "general": {"min": 200, "max": 1500, "optimal": 800}
        }
        
        target = target_lengths.get(target_audience, target_lengths["general"])
        
        if target["min"] <= word_count <= target["max"]:
            distance = abs(word_count - target["optimal"])
            max_distance = max(
                target["optimal"] - target["min"],
                target["max"] - target["optimal"]
            )
            score = 1.0 - (distance / max_distance) if max_distance > 0 else 1.0
            status = "fit"
        elif word_count < target["min"]:
            score = word_count / target["min"] if target["min"] > 0 else 0.0
            status = "too_short"
        else:
            score = target["max"] / word_count if word_count > 0 else 0.0
            status = "too_long"
        
        return {
            "score": score,
            "status": status,
            "word_count": word_count,
            "target_range": [target["min"], target["max"]],
            "optimal": target["optimal"]
        }

    def _generate_audience_suggestions(
        self,
        fit_score: float,
        complexity_fit: Dict[str, Any],
        vocabulary_fit: Dict[str, Any],
        target_audience: str
    ) -> List[str]:
        """Generar sugerencias de ajuste a audiencia"""
        suggestions = []
        
        if fit_score < 0.6:
            suggestions.append(f"El contenido no se ajusta bien a la audiencia '{target_audience}'.")
        
        complexity_status = complexity_fit.get("status")
        if complexity_status == "too_complex":
            suggestions.append("El contenido es demasiado complejo. Simplifica el lenguaje y las oraciones.")
        elif complexity_status == "too_simple":
            suggestions.append("El contenido es demasiado simple. Considera agregar más profundidad.")
        
        if vocabulary_fit.get("score", 1.0) < 0.7:
            if target_audience == "beginner":
                suggestions.append("Reduce el uso de palabras técnicas y jerga.")
            elif target_audience in ["advanced", "expert"]:
                suggestions.append("Considera usar más terminología técnica y especializada.")
        
        return suggestions






