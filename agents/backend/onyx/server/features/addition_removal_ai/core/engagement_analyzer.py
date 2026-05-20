"""
Engagement Analyzer - Sistema de análisis de engagement
"""

import logging
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EngagementMetric:
    """Métrica de engagement"""
    name: str
    value: float
    score: float
    description: str


class EngagementAnalyzer:
    """Analizador de engagement"""

    def __init__(self):
        """Inicializar analizador"""
        # Palabras que generan engagement
        self.engagement_words = {
            'gratis', 'descarga', 'ahora', 'únete', 'descubre', 'aprende',
            'gratis', 'download', 'now', 'join', 'discover', 'learn',
            'exclusivo', 'limitado', 'oferta', 'especial',
            'exclusive', 'limited', 'offer', 'special'
        }
        
        # Palabras de acción
        self.action_words = {
            'haz', 'compra', 'regístrate', 'suscríbete', 'comparte',
            'do', 'buy', 'register', 'subscribe', 'share',
            'descarga', 'consigue', 'obtén', 'aprovecha',
            'download', 'get', 'obtain', 'take advantage'
        }
        
        # Palabras emocionales
        self.emotional_words = {
            'increíble', 'asombroso', 'fantástico', 'genial', 'maravilloso',
            'incredible', 'amazing', 'fantastic', 'great', 'wonderful',
            'sorprendente', 'impresionante', 'extraordinario',
            'surprising', 'impressive', 'extraordinary'
        }

    def analyze(self, content: str) -> Dict[str, Any]:
        """
        Analizar engagement del contenido.

        Args:
            content: Contenido

        Returns:
            Análisis de engagement
        """
        content_lower = content.lower()
        
        # Métricas de engagement
        engagement_metrics = []
        
        # Análisis de palabras de engagement
        engagement_score = self._analyze_engagement_words(content_lower)
        engagement_metrics.append(EngagementMetric(
            name="engagement_words",
            value=engagement_score["count"],
            score=engagement_score["score"],
            description="Palabras que generan engagement"
        ))
        
        # Análisis de palabras de acción
        action_score = self._analyze_action_words(content_lower)
        engagement_metrics.append(EngagementMetric(
            name="action_words",
            value=action_score["count"],
            score=action_score["score"],
            description="Palabras de acción (CTAs)"
        ))
        
        # Análisis de palabras emocionales
        emotional_score = self._analyze_emotional_words(content_lower)
        engagement_metrics.append(EngagementMetric(
            name="emotional_words",
            value=emotional_score["count"],
            score=emotional_score["score"],
            description="Palabras emocionales"
        ))
        
        # Análisis de preguntas
        question_score = self._analyze_questions(content)
        engagement_metrics.append(EngagementMetric(
            name="questions",
            value=question_score["count"],
            score=question_score["score"],
            description="Preguntas que generan interacción"
        ))
        
        # Análisis de llamadas a la acción
        cta_score = self._analyze_ctas(content_lower)
        engagement_metrics.append(EngagementMetric(
            name="ctas",
            value=cta_score["count"],
            score=cta_score["score"],
            description="Llamadas a la acción"
        ))
        
        # Calcular score general
        overall_score = sum(m.score for m in engagement_metrics) / len(engagement_metrics) if engagement_metrics else 0.0
        
        return {
            "engagement_score": overall_score,
            "metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "score": m.score,
                    "description": m.description
                }
                for m in engagement_metrics
            ],
            "suggestions": self._generate_engagement_suggestions(overall_score, engagement_metrics)
        }

    def _analyze_engagement_words(self, content: str) -> Dict[str, Any]:
        """Analizar palabras de engagement"""
        words = content.split()
        engagement_count = sum(1 for word in words if word in self.engagement_words)
        
        total_words = len(words)
        if total_words == 0:
            return {"count": 0, "score": 0.0}
        
        ratio = engagement_count / total_words
        score = min(1.0, ratio * 20)  # Normalizar
        
        return {
            "count": engagement_count,
            "score": score,
            "words_found": [w for w in words if w in self.engagement_words][:10]
        }

    def _analyze_action_words(self, content: str) -> Dict[str, Any]:
        """Analizar palabras de acción"""
        words = content.split()
        action_count = sum(1 for word in words if word in self.action_words)
        
        total_words = len(words)
        if total_words == 0:
            return {"count": 0, "score": 0.0}
        
        ratio = action_count / total_words
        score = min(1.0, ratio * 15)  # Normalizar
        
        return {
            "count": action_count,
            "score": score,
            "words_found": [w for w in words if w in self.action_words][:10]
        }

    def _analyze_emotional_words(self, content: str) -> Dict[str, Any]:
        """Analizar palabras emocionales"""
        words = content.split()
        emotional_count = sum(1 for word in words if word in self.emotional_words)
        
        total_words = len(words)
        if total_words == 0:
            return {"count": 0, "score": 0.0}
        
        ratio = emotional_count / total_words
        score = min(1.0, ratio * 25)  # Normalizar
        
        return {
            "count": emotional_count,
            "score": score,
            "words_found": [w for w in words if w in self.emotional_words][:10]
        }

    def _analyze_questions(self, content: str) -> Dict[str, Any]:
        """Analizar preguntas"""
        questions = re.findall(r'[¿?]', content)
        question_count = len(questions)
        
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return {"count": 0, "score": 0.0}
        
        ratio = question_count / len(sentences)
        score = min(1.0, ratio * 5)  # Normalizar
        
        return {
            "count": question_count,
            "score": score
        }

    def _analyze_ctas(self, content: str) -> Dict[str, Any]:
        """Analizar llamadas a la acción"""
        cta_patterns = [
            r'regístrate\s+ahora',
            r'suscríbete\s+ahora',
            r'descarga\s+ahora',
            r'compra\s+ahora',
            r'register\s+now',
            r'subscribe\s+now',
            r'download\s+now',
            r'buy\s+now',
            r'¡[A-Z][^!]*!',
            r'[A-Z][^.]*!'
        ]
        
        cta_count = 0
        for pattern in cta_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            cta_count += len(matches)
        
        words = content.split()
        if not words:
            return {"count": 0, "score": 0.0}
        
        ratio = cta_count / len(words) if words else 0
        score = min(1.0, ratio * 100)  # Normalizar
        
        return {
            "count": cta_count,
            "score": score
        }

    def _generate_engagement_suggestions(
        self,
        overall_score: float,
        metrics: List[EngagementMetric]
    ) -> List[str]:
        """Generar sugerencias de engagement"""
        suggestions = []
        
        if overall_score < 0.5:
            suggestions.append("El contenido tiene bajo engagement. Considera agregar más elementos interactivos.")
        
        # Verificar métricas específicas
        action_metric = next((m for m in metrics if m.name == "action_words"), None)
        if action_metric and action_metric.score < 0.3:
            suggestions.append("Agrega más palabras de acción y llamadas a la acción (CTAs).")
        
        question_metric = next((m for m in metrics if m.name == "questions"), None)
        if question_metric and question_metric.score < 0.2:
            suggestions.append("Agrega preguntas para generar más interacción con el lector.")
        
        emotional_metric = next((m for m in metrics if m.name == "emotional_words"), None)
        if emotional_metric and emotional_metric.score < 0.3:
            suggestions.append("Usa más palabras emocionales para conectar mejor con la audiencia.")
        
        return suggestions






