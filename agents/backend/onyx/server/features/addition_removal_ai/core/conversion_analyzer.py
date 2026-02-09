"""
Conversion Analyzer - Sistema de análisis de conversión
"""

import logging
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ConversionElement:
    """Elemento de conversión"""
    type: str
    text: str
    position: int
    strength: float


class ConversionAnalyzer:
    """Analizador de conversión"""

    def __init__(self):
        """Inicializar analizador"""
        # Palabras de acción fuertes
        self.strong_action_words = {
            'compra', 'descarga', 'regístrate', 'suscríbete', 'únete',
            'buy', 'download', 'register', 'subscribe', 'join',
            'consigue', 'obtén', 'aprovecha', 'actúa',
            'get', 'obtain', 'take advantage', 'act'
        }
        
        # Palabras de urgencia
        self.urgency_words = {
            'ahora', 'inmediatamente', 'urgente', 'limitado', 'exclusivo',
            'now', 'immediately', 'urgent', 'limited', 'exclusive',
            'hoy', 'rápido', 'ya', 'aprovecha',
            'today', 'fast', 'already', 'take advantage'
        }
        
        # Palabras de beneficio
        self.benefit_words = {
            'gratis', 'gratuito', 'descuento', 'oferta', 'bonificación',
            'free', 'discount', 'offer', 'bonus',
            'ahorro', 'ventaja', 'beneficio', 'mejora',
            'save', 'advantage', 'benefit', 'improve'
        }

    def analyze_conversion_potential(self, content: str) -> Dict[str, Any]:
        """
        Analizar potencial de conversión del contenido.

        Args:
            content: Contenido

        Returns:
            Análisis de potencial de conversión
        """
        content_lower = content.lower()
        
        # Detectar elementos de conversión
        conversion_elements = self._detect_conversion_elements(content)
        
        # Analizar palabras de acción
        action_analysis = self._analyze_action_words(content_lower)
        
        # Analizar urgencia
        urgency_analysis = self._analyze_urgency(content_lower)
        
        # Analizar beneficios
        benefit_analysis = self._analyze_benefits(content_lower)
        
        # Analizar CTAs
        cta_analysis = self._analyze_ctas(content)
        
        # Calcular score de conversión
        conversion_score = (
            action_analysis["score"] * 0.3 +
            urgency_analysis["score"] * 0.25 +
            benefit_analysis["score"] * 0.25 +
            cta_analysis["score"] * 0.2
        )
        
        return {
            "conversion_score": conversion_score,
            "conversion_elements": [
                {
                    "type": elem.type,
                    "text": elem.text,
                    "position": elem.position,
                    "strength": elem.strength
                }
                for elem in conversion_elements
            ],
            "action_words": action_analysis,
            "urgency": urgency_analysis,
            "benefits": benefit_analysis,
            "ctas": cta_analysis,
            "suggestions": self._generate_conversion_suggestions(
                conversion_score,
                action_analysis,
                urgency_analysis,
                benefit_analysis,
                cta_analysis
            )
        }

    def _detect_conversion_elements(self, content: str) -> List[ConversionElement]:
        """Detectar elementos de conversión"""
        elements = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            
            # Detectar CTAs
            for action_word in self.strong_action_words:
                if action_word in line_lower:
                    strength = 0.8 if any(urg in line_lower for urg in self.urgency_words) else 0.6
                    elements.append(ConversionElement(
                        type="cta",
                        text=line[:100],  # Primeros 100 caracteres
                        position=i,
                        strength=strength
                    ))
                    break
            
            # Detectar beneficios
            for benefit_word in self.benefit_words:
                if benefit_word in line_lower:
                    elements.append(ConversionElement(
                        type="benefit",
                        text=line[:100],
                        position=i,
                        strength=0.7
                    ))
                    break
        
        return elements

    def _analyze_action_words(self, content: str) -> Dict[str, Any]:
        """Analizar palabras de acción"""
        words = content.split()
        action_count = sum(1 for word in words if word in self.strong_action_words)
        
        total_words = len(words)
        if total_words == 0:
            return {"count": 0, "score": 0.0}
        
        ratio = action_count / total_words
        score = min(1.0, ratio * 50)  # Normalizar
        
        return {
            "count": action_count,
            "ratio": ratio,
            "score": score
        }

    def _analyze_urgency(self, content: str) -> Dict[str, Any]:
        """Analizar urgencia"""
        words = content.split()
        urgency_count = sum(1 for word in words if word in self.urgency_words)
        
        total_words = len(words)
        if total_words == 0:
            return {"count": 0, "score": 0.0}
        
        ratio = urgency_count / total_words
        score = min(1.0, ratio * 30)  # Normalizar
        
        return {
            "count": urgency_count,
            "ratio": ratio,
            "score": score
        }

    def _analyze_benefits(self, content: str) -> Dict[str, Any]:
        """Analizar beneficios"""
        words = content.split()
        benefit_count = sum(1 for word in words if word in self.benefit_words)
        
        total_words = len(words)
        if total_words == 0:
            return {"count": 0, "score": 0.0}
        
        ratio = benefit_count / total_words
        score = min(1.0, ratio * 40)  # Normalizar
        
        return {
            "count": benefit_count,
            "ratio": ratio,
            "score": score
        }

    def _analyze_ctas(self, content: str) -> Dict[str, Any]:
        """Analizar llamadas a la acción"""
        # Buscar patrones de CTA
        cta_patterns = [
            r'[¡!]\s*[A-Z][^!]*!',  # Exclamaciones
            r'(?:compra|descarga|regístrate|suscríbete|únete)\s+ahora',
            r'(?:buy|download|register|subscribe|join)\s+now'
        ]
        
        cta_count = 0
        for pattern in cta_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            cta_count += len(matches)
        
        sentences = [s for s in content.split('.') if s.strip()]
        if not sentences:
            return {"count": 0, "score": 0.0}
        
        ratio = cta_count / len(sentences)
        score = min(1.0, ratio * 5)  # Normalizar
        
        return {
            "count": cta_count,
            "ratio": ratio,
            "score": score
        }

    def _generate_conversion_suggestions(
        self,
        conversion_score: float,
        action_analysis: Dict[str, Any],
        urgency_analysis: Dict[str, Any],
        benefit_analysis: Dict[str, Any],
        cta_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generar sugerencias de conversión"""
        suggestions = []
        
        if conversion_score < 0.5:
            suggestions.append("El contenido tiene bajo potencial de conversión.")
        
        if action_analysis.get("score", 0) < 0.3:
            suggestions.append("Agrega más palabras de acción (compra, descarga, regístrate).")
        
        if urgency_analysis.get("score", 0) < 0.2:
            suggestions.append("Crea urgencia con palabras como 'ahora', 'limitado', 'exclusivo'.")
        
        if benefit_analysis.get("score", 0) < 0.3:
            suggestions.append("Destaca los beneficios (gratis, descuento, oferta).")
        
        if cta_analysis.get("score", 0) < 0.2:
            suggestions.append("Agrega llamadas a la acción claras y directas.")
        
        return suggestions






