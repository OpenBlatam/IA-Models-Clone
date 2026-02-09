"""
News Content Analyzer - Sistema de análisis de contenido de noticias
"""

import logging
import re
from typing import Dict, Any, Optional, List
from collections import Counter

logger = logging.getLogger(__name__)


class NewsContentAnalyzer:
    """Analizador de contenido de noticias"""

    def __init__(self):
        """Inicializar analizador"""
        # Elementos de noticias
        self.news_elements = {
            "headlines": [
                r'^#+\s+[A-Z]',  # Markdown headers
                r'<h[1-6][^>]*>',  # HTML headers
            ],
            "byline": [
                r'(?:por|by|escrito por|written by|autor|author)',
                r'(?:publicado|published|fecha|date)'
            ],
            "lead": [
                r'(?:resumen|summary|introducción|introduction|lead)',
                r'(?:en resumen|in summary|en pocas palabras|in short)'
            ],
            "quotes": [
                r'"[^"]*"',
                r''[^']*'',
                r'(?:dijo|said|declaró|declared|afirmó|affirmed)'
            ],
            "sources": [
                r'(?:según|according to|fuente|source|informó|reported)',
                r'(?:agencia|agency|medio|media|periódico|newspaper)'
            ],
            "dates": [
                r'\d{1,2}\/\d{1,2}\/\d{4}',
                r'(?:hoy|today|ayer|yesterday|mañana|tomorrow)',
                r'(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\s+\d+'
            ],
            "locations": [
                r'(?:en|in|desde|from)\s+[A-Z][a-z]+',
                r'(?:ciudad|city|país|country|región|region)'
            ],
            "breaking": [
                r'(?:última hora|breaking news|noticia de último momento|urgent)',
                r'(?:urgente|urgent|inmediato|immediate)'
            ]
        }

    def analyze_news_content(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar contenido de noticias.

        Args:
            content: Contenido

        Returns:
            Análisis de contenido de noticias
        """
        element_counts = {}
        
        # Contar elementos de noticias
        for element_type, patterns in self.news_elements.items():
            count = 0
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                count += len(matches)
            element_counts[element_type] = count
        
        # Calcular score de noticias
        total_elements = sum(element_counts.values())
        news_score = min(1.0, total_elements / 20)  # Normalizar
        
        # Verificar si es contenido de noticias
        is_news = (
            element_counts.get("headlines", 0) > 0 or
            element_counts.get("byline", 0) > 0 or
            element_counts.get("quotes", 0) > 0 or
            element_counts.get("sources", 0) > 0
        )
        
        return {
            "element_counts": element_counts,
            "news_score": news_score,
            "total_elements": total_elements,
            "is_news": is_news
        }

    def analyze_news_credibility(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar credibilidad del contenido de noticias.

        Args:
            content: Contenido

        Returns:
            Análisis de credibilidad
        """
        news_analysis = self.analyze_news_content(content)
        element_counts = news_analysis["element_counts"]
        
        # Verificar elementos de credibilidad
        has_byline = element_counts.get("byline", 0) > 0
        has_sources = element_counts.get("sources", 0) > 0
        has_quotes = element_counts.get("quotes", 0) > 0
        has_dates = element_counts.get("dates", 0) > 0
        has_locations = element_counts.get("locations", 0) > 0
        
        # Calcular score de credibilidad
        credibility_score = (
            (1.0 if has_byline else 0.0) * 0.2 +
            (1.0 if has_sources else 0.0) * 0.3 +
            (1.0 if has_quotes else 0.0) * 0.2 +
            (1.0 if has_dates else 0.0) * 0.15 +
            (1.0 if has_locations else 0.0) * 0.15
        )
        
        return {
            "credibility_score": credibility_score,
            "has_byline": has_byline,
            "has_sources": has_sources,
            "has_quotes": has_quotes,
            "has_dates": has_dates,
            "has_locations": has_locations,
            "credibility_level": (
                "high" if credibility_score > 0.7 else
                "medium" if credibility_score > 0.4 else
                "low"
            )
        }


