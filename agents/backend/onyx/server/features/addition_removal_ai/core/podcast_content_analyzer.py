"""
Podcast Content Analyzer - Sistema de análisis de contenido de podcast/audio
"""

import logging
import re
from typing import Dict, Any, Optional, List
from collections import Counter

logger = logging.getLogger(__name__)


class PodcastContentAnalyzer:
    """Analizador de contenido de podcast/audio"""

    def __init__(self):
        """Inicializar analizador"""
        # Elementos de podcast
        self.podcast_elements = {
            "episodes": [
                r'(?:episodio|episode|programa|program|show)',
                r'(?:episodio \d+|episode \d+|#\d+)'
            ],
            "guests": [
                r'(?:invitado|guest|entrevistado|interviewee)',
                r'(?:con|with|entrevista|interview)'
            ],
            "topics": [
                r'(?:tema|topic|tema del día|topic of the day)',
                r'(?:hablamos de|we talk about|discutimos|we discuss)'
            ],
            "timestamps": [
                r'\d{1,2}:\d{2}',  # Timestamps (MM:SS o HH:MM:SS)
                r'(?:minuto|minute|segundo|second|hora|hour)'
            ],
            "segments": [
                r'(?:segmento|segment|parte|part|sección|section)',
                r'(?:intro|introducción|outro|conclusión|conclusion)'
            ],
            "calls_to_action": [
                r'(?:suscríbete|subscribe|síguenos|follow us|comparte|share)',
                r'(?:déjanos una reseña|leave a review|califica|rate)'
            ],
            "sponsors": [
                r'(?:patrocinador|sponsor|anuncio|ad|publicidad|advertisement)',
                r'(?:gracias a|thanks to|presentado por|presented by)'
            ]
        }

    def analyze_podcast_content(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar contenido de podcast.

        Args:
            content: Contenido

        Returns:
            Análisis de contenido de podcast
        """
        element_counts = {}
        
        # Contar elementos de podcast
        for element_type, patterns in self.podcast_elements.items():
            count = 0
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                count += len(matches)
            element_counts[element_type] = count
        
        # Calcular score de podcast
        total_elements = sum(element_counts.values())
        podcast_score = min(1.0, total_elements / 20)  # Normalizar
        
        # Verificar si es contenido de podcast
        is_podcast = (
            element_counts.get("episodes", 0) > 0 or
            element_counts.get("guests", 0) > 0 or
            element_counts.get("topics", 0) > 0 or
            element_counts.get("timestamps", 0) > 0
        )
        
        return {
            "element_counts": element_counts,
            "podcast_score": podcast_score,
            "total_elements": total_elements,
            "is_podcast": is_podcast
        }

    def analyze_podcast_structure(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar estructura del podcast.

        Args:
            content: Contenido

        Returns:
            Análisis de estructura del podcast
        """
        podcast_analysis = self.analyze_podcast_content(content)
        element_counts = podcast_analysis["element_counts"]
        
        # Verificar elementos de estructura
        has_episodes = element_counts.get("episodes", 0) > 0
        has_guests = element_counts.get("guests", 0) > 0
        has_topics = element_counts.get("topics", 0) > 0
        has_timestamps = element_counts.get("timestamps", 0) > 0
        has_segments = element_counts.get("segments", 0) > 0
        has_ctas = element_counts.get("calls_to_action", 0) > 0
        
        # Calcular score de estructura
        structure_score = (
            (1.0 if has_episodes else 0.0) * 0.25 +
            (1.0 if has_guests else 0.0) * 0.15 +
            (1.0 if has_topics else 0.0) * 0.2 +
            (1.0 if has_timestamps else 0.0) * 0.15 +
            (1.0 if has_segments else 0.0) * 0.15 +
            (1.0 if has_ctas else 0.0) * 0.1
        )
        
        return {
            "structure_score": structure_score,
            "has_episodes": has_episodes,
            "has_guests": has_guests,
            "has_topics": has_topics,
            "has_timestamps": has_timestamps,
            "has_segments": has_segments,
            "has_ctas": has_ctas,
            "structure_level": (
                "high" if structure_score > 0.7 else
                "medium" if structure_score > 0.4 else
                "low"
            )
        }


