"""
Video Content Analyzer - Sistema de análisis de contenido de video/YouTube
"""

import logging
import re
from typing import Dict, Any, Optional, List
from collections import Counter

logger = logging.getLogger(__name__)


class VideoContentAnalyzer:
    """Analizador de contenido de video/YouTube"""

    def __init__(self):
        """Inicializar analizador"""
        # Elementos de video
        self.video_elements = {
            "video_links": [
                r'(?:youtube\.com|youtu\.be|vimeo\.com|dailymotion\.com)',
                r'https?://[^\s]*(?:youtube|youtu|vimeo|dailymotion)',
            ],
            "timestamps": [
                r'\d{1,2}:\d{2}',  # Timestamps (MM:SS o HH:MM:SS)
                r'(?:minuto|minute|segundo|second|hora|hour)',
                r'(?:en el minuto|at minute|en|at)'
            ],
            "chapters": [
                r'(?:capítulo|chapter|parte|part|sección|section)',
                r'(?:capítulo \d+|chapter \d+|parte \d+|part \d+)'
            ],
            "thumbnails": [
                r'(?:miniatura|thumbnail|portada|cover)',
                r'<img[^>]*thumbnail[^>]*>',  # HTML thumbnail
            ],
            "descriptions": [
                r'(?:descripción|description|resumen|summary)',
                r'(?:en este video|in this video|en este|in this)'
            ],
            "ctas": [
                r'(?:suscríbete|subscribe|dale like|like|comparte|share)',
                r'(?:activa la campana|ring the bell|notificaciones|notifications)'
            ],
            "tags": [
                r'(?:etiqueta|tag|hashtag|#\w+)',
                r'(?:palabras clave|keywords)'
            ],
            "transcripts": [
                r'(?:transcripción|transcript|subtítulos|subtitles|cc)',
                r'(?:texto completo|full text)'
            ]
        }

    def analyze_video_content(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar contenido de video.

        Args:
            content: Contenido

        Returns:
            Análisis de contenido de video
        """
        element_counts = {}
        
        # Contar elementos de video
        for element_type, patterns in self.video_elements.items():
            count = 0
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                count += len(matches)
            element_counts[element_type] = count
        
        # Calcular score de video
        total_elements = sum(element_counts.values())
        video_score = min(1.0, total_elements / 25)  # Normalizar
        
        # Verificar si es contenido de video
        is_video = (
            element_counts.get("video_links", 0) > 0 or
            element_counts.get("timestamps", 0) > 0 or
            element_counts.get("chapters", 0) > 0 or
            element_counts.get("descriptions", 0) > 0
        )
        
        return {
            "element_counts": element_counts,
            "video_score": video_score,
            "total_elements": total_elements,
            "is_video": is_video
        }

    def analyze_video_optimization(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar optimización del contenido de video.

        Args:
            content: Contenido

        Returns:
            Análisis de optimización del video
        """
        video_analysis = self.analyze_video_content(content)
        element_counts = video_analysis["element_counts"]
        
        # Verificar elementos de optimización
        has_links = element_counts.get("video_links", 0) > 0
        has_timestamps = element_counts.get("timestamps", 0) > 0
        has_chapters = element_counts.get("chapters", 0) > 0
        has_thumbnails = element_counts.get("thumbnails", 0) > 0
        has_descriptions = element_counts.get("descriptions", 0) > 0
        has_ctas = element_counts.get("ctas", 0) > 0
        has_tags = element_counts.get("tags", 0) > 0
        has_transcripts = element_counts.get("transcripts", 0) > 0
        
        # Calcular score de optimización
        optimization_score = (
            (1.0 if has_links else 0.0) * 0.15 +
            (1.0 if has_timestamps else 0.0) * 0.15 +
            (1.0 if has_chapters else 0.0) * 0.15 +
            (1.0 if has_thumbnails else 0.0) * 0.1 +
            (1.0 if has_descriptions else 0.0) * 0.15 +
            (1.0 if has_ctas else 0.0) * 0.15 +
            (1.0 if has_tags else 0.0) * 0.1 +
            (1.0 if has_transcripts else 0.0) * 0.05
        )
        
        return {
            "optimization_score": optimization_score,
            "has_links": has_links,
            "has_timestamps": has_timestamps,
            "has_chapters": has_chapters,
            "has_thumbnails": has_thumbnails,
            "has_descriptions": has_descriptions,
            "has_ctas": has_ctas,
            "has_tags": has_tags,
            "has_transcripts": has_transcripts,
            "optimization_level": (
                "high" if optimization_score > 0.7 else
                "medium" if optimization_score > 0.4 else
                "low"
            )
        }


