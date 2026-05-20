"""
Tone Analyzer - Sistema de análisis de tono/voz
"""

import logging
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ToneType(Enum):
    """Tipos de tono"""
    FORMAL = "formal"
    INFORMAL = "informal"
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    FRIENDLY = "friendly"
    AUTHORITATIVE = "authoritative"
    NEUTRAL = "neutral"


@dataclass
class ToneAnalysis:
    """Análisis de tono"""
    primary_tone: ToneType
    confidence: float
    tone_scores: Dict[str, float]
    indicators: List[str]


class ToneAnalyzer:
    """Analizador de tono/voz"""

    def __init__(self):
        """Inicializar analizador"""
        # Indicadores de tono formal
        self.formal_indicators = {
            'usted', 'ustedes', 'debe', 'debería', 'es necesario',
            'se requiere', 'se recomienda', 'por favor', 'le agradecemos',
            'you', 'should', 'must', 'required', 'recommended', 'please',
            'thank you', 'appreciate'
        }
        
        # Indicadores de tono informal
        self.informal_indicators = {
            'tú', 'vos', 'vosotros', 'puedes', 'pueden', 'haz', 'hagan',
            'you', 'can', 'do', 'make', 'hey', 'hi', 'hello', 'yeah',
            'ok', 'okay', 'cool', 'awesome'
        }
        
        # Indicadores de tono profesional
        self.professional_indicators = {
            'implementar', 'optimizar', 'desarrollar', 'gestionar',
            'implement', 'optimize', 'develop', 'manage', 'strategy',
            'process', 'methodology', 'framework', 'solution'
        }
        
        # Indicadores de tono casual
        self.casual_indicators = {
            'genial', 'chévere', 'bacán', 'cool', 'awesome', 'great',
            'nice', 'good', 'sure', 'yeah', 'yep', 'nope'
        }
        
        # Indicadores de tono amigable
        self.friendly_indicators = {
            'gracias', 'por favor', 'disculpa', 'perdón', 'gracias',
            'thank you', 'please', 'sorry', 'excuse me', 'welcome',
            'happy', 'glad', 'pleased', 'delighted'
        }
        
        # Indicadores de tono autoritario
        self.authoritative_indicators = {
            'debe', 'debería', 'es obligatorio', 'se requiere',
            'must', 'should', 'required', 'mandatory', 'essential',
            'critical', 'important', 'necessary'
        }

    def analyze(self, content: str) -> Dict[str, Any]:
        """
        Analizar tono del contenido.

        Args:
            content: Contenido

        Returns:
            Análisis de tono
        """
        content_lower = content.lower()
        
        # Calcular scores para cada tono
        tone_scores = {
            "formal": self._calculate_tone_score(content_lower, self.formal_indicators),
            "informal": self._calculate_tone_score(content_lower, self.informal_indicators),
            "professional": self._calculate_tone_score(content_lower, self.professional_indicators),
            "casual": self._calculate_tone_score(content_lower, self.casual_indicators),
            "friendly": self._calculate_tone_score(content_lower, self.friendly_indicators),
            "authoritative": self._calculate_tone_score(content_lower, self.authoritative_indicators),
            "neutral": 0.5  # Neutral por defecto
        }
        
        # Determinar tono primario
        primary_tone = max(tone_scores.items(), key=lambda x: x[1])[0]
        max_score = tone_scores[primary_tone]
        
        # Si el score es muy bajo, es neutral
        if max_score < 0.3:
            primary_tone = "neutral"
            max_score = 0.5
        
        # Obtener indicadores
        indicators = self._get_indicators(content_lower, primary_tone)
        
        return {
            "primary_tone": primary_tone,
            "confidence": max_score,
            "tone_scores": tone_scores,
            "indicators": indicators,
            "suggestions": self._generate_tone_suggestions(primary_tone, tone_scores)
        }

    def _calculate_tone_score(self, content: str, indicators: set) -> float:
        """Calcular score de tono"""
        if not indicators:
            return 0.0
        
        matches = sum(1 for indicator in indicators if indicator in content)
        total_words = len(content.split())
        
        if total_words == 0:
            return 0.0
        
        # Normalizar score
        score = min(1.0, (matches / len(indicators)) * 2)
        
        return score

    def _get_indicators(self, content: str, tone: str) -> List[str]:
        """Obtener indicadores encontrados"""
        indicator_sets = {
            "formal": self.formal_indicators,
            "informal": self.informal_indicators,
            "professional": self.professional_indicators,
            "casual": self.casual_indicators,
            "friendly": self.friendly_indicators,
            "authoritative": self.authoritative_indicators
        }
        
        indicators = indicator_sets.get(tone, set())
        found = [ind for ind in indicators if ind in content]
        
        return found[:10]  # Limitar a 10

    def _generate_tone_suggestions(
        self,
        primary_tone: str,
        tone_scores: Dict[str, float]
    ) -> List[str]:
        """Generar sugerencias de tono"""
        suggestions = []
        
        if primary_tone == "neutral":
            suggestions.append("El tono es neutral. Considera usar un tono más específico según el contexto.")
        
        # Verificar mezcla de tonos
        high_scores = [tone for tone, score in tone_scores.items() if score > 0.4 and tone != primary_tone]
        if high_scores:
            suggestions.append(f"El contenido mezcla tonos ({primary_tone} y {', '.join(high_scores)}). Considera mantener consistencia.")
        
        return suggestions

    def analyze_by_sections(self, content: str) -> Dict[str, Any]:
        """
        Analizar tono por secciones.

        Args:
            content: Contenido

        Returns:
            Análisis por secciones
        """
        sections = self._split_into_sections(content)
        
        section_tones = []
        for section_name, section_content in sections.items():
            tone = self.analyze(section_content)
            section_tones.append({
                "section": section_name,
                "tone": tone
            })
        
        return {
            "sections": section_tones,
            "total_sections": len(sections)
        }

    def _split_into_sections(self, content: str) -> Dict[str, str]:
        """Dividir en secciones"""
        sections = {}
        current_section = "Introduction"
        current_content = []
        
        lines = content.split('\n')
        for line in lines:
            if line.startswith('#'):
                if current_content:
                    sections[current_section] = ' '.join(current_content)
                current_section = line.lstrip('#').strip()
                current_content = []
            else:
                current_content.append(line)
        
        if current_content:
            sections[current_section] = ' '.join(current_content)
        
        return sections






