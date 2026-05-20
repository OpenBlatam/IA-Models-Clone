"""
Social Media Analyzer - Sistema de análisis de contenido de redes sociales
"""

import logging
import re
from typing import Dict, Any, Optional, List
from collections import Counter

logger = logging.getLogger(__name__)


class SocialMediaAnalyzer:
    """Analizador de contenido de redes sociales"""

    def __init__(self):
        """Inicializar analizador"""
        # Elementos de redes sociales
        self.social_elements = {
            "hashtags": [
                r'#\w+',  # Hashtags
            ],
            "mentions": [
                r'@\w+',  # Menciones
            ],
            "emojis": [
                r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]+',  # Emojis
            ],
            "links": [
                r'https?://[^\s]+',  # URLs
            ],
            "questions": [
                r'\?',  # Signos de interrogación
                r'(?:qué|what|cuál|which|cómo|how|por qué|why)',
            ],
            "exclamations": [
                r'!',  # Signos de exclamación
                r'(?:¡|wow|increíble|amazing|genial|great)',
            ],
            "ctas": [
                r'(?:comparte|share|dale like|like|sigue|follow)',
                r'(?:comenta|comment|opina|opinion)'
            ],
            "trending": [
                r'(?:tendencia|trending|viral|popular|hot)',
            ]
        }

    def analyze_social_content(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar contenido de redes sociales.

        Args:
            content: Contenido

        Returns:
            Análisis de contenido de redes sociales
        """
        element_counts = {}
        
        # Contar elementos de redes sociales
        for element_type, patterns in self.social_elements.items():
            count = 0
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE | re.UNICODE)
                count += len(matches)
            element_counts[element_type] = count
        
        # Calcular score de redes sociales
        total_elements = sum(element_counts.values())
        social_score = min(1.0, total_elements / 20)  # Normalizar
        
        # Verificar si es contenido de redes sociales
        is_social = (
            element_counts.get("hashtags", 0) > 0 or
            element_counts.get("mentions", 0) > 0 or
            element_counts.get("emojis", 0) > 0 or
            element_counts.get("ctas", 0) > 0
        )
        
        return {
            "element_counts": element_counts,
            "social_score": social_score,
            "total_elements": total_elements,
            "is_social": is_social
        }

    def analyze_social_virality(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar potencial de viralidad del contenido.

        Args:
            content: Contenido

        Returns:
            Análisis de viralidad
        """
        social_analysis = self.analyze_social_content(content)
        element_counts = social_analysis["element_counts"]
        
        # Verificar elementos de viralidad
        has_hashtags = element_counts.get("hashtags", 0) > 0
        has_mentions = element_counts.get("mentions", 0) > 0
        has_emojis = element_counts.get("emojis", 0) > 0
        has_questions = element_counts.get("questions", 0) > 0
        has_exclamations = element_counts.get("exclamations", 0) > 0
        has_ctas = element_counts.get("ctas", 0) > 0
        has_trending = element_counts.get("trending", 0) > 0
        
        # Calcular score de viralidad
        virality_score = (
            (1.0 if has_hashtags else 0.0) * 0.2 +
            (1.0 if has_mentions else 0.0) * 0.15 +
            (1.0 if has_emojis else 0.0) * 0.15 +
            (1.0 if has_questions else 0.0) * 0.15 +
            (1.0 if has_exclamations else 0.0) * 0.15 +
            (1.0 if has_ctas else 0.0) * 0.15 +
            (1.0 if has_trending else 0.0) * 0.05
        )
        
        return {
            "virality_score": virality_score,
            "has_hashtags": has_hashtags,
            "has_mentions": has_mentions,
            "has_emojis": has_emojis,
            "has_questions": has_questions,
            "has_exclamations": has_exclamations,
            "has_ctas": has_ctas,
            "has_trending": has_trending,
            "virality_level": (
                "high" if virality_score > 0.7 else
                "medium" if virality_score > 0.4 else
                "low"
            )
        }


