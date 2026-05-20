"""
Journalistic Content Analyzer - Sistema de análisis de contenido periodístico
"""

import logging
import re
from typing import Dict, Any, Optional, List
from collections import Counter

logger = logging.getLogger(__name__)


class JournalisticContentAnalyzer:
    """Analizador de contenido periodístico"""

    def __init__(self):
        """Inicializar analizador"""
        # Elementos periodísticos
        self.journalistic_elements = {
            "5w1h": {
                "what": [r'(?:qué|what|qué es|what is)'],
                "who": [r'(?:quién|who|quiénes|who are)'],
                "where": [r'(?:dónde|where|en dónde|in where)'],
                "when": [r'(?:cuándo|when|en qué momento|at what time)'],
                "why": [r'(?:por qué|why|razón|reason)'],
                "how": [r'(?:cómo|how|de qué manera|in what way)']
            },
            "quotes": [
                r'"[^"]*"',
                r''[^']*'',
                r'(?:dijo|said|declaró|declared|afirmó|affirmed)'
            ],
            "sources": [
                r'(?:según|according to|fuente|source|informó|reported)',
                r'(?:agencia|agency|medio|media|periódico|newspaper)'
            ],
            "headlines": [
                r'^#+\s+[A-Z]',  # Markdown headers
                r'<h[1-6][^>]*>',  # HTML headers
            ],
            "dates": [
                r'\d{1,2}\/\d{1,2}\/\d{4}',
                r'(?:hoy|today|ayer|yesterday|mañana|tomorrow)',
                r'(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\s+\d+'
            ],
            "locations": [
                r'(?:en|in|desde|from)\s+[A-Z][a-z]+',  # Nombres de lugares
                r'(?:ciudad|city|país|country|región|region)'
            ]
        }

    def analyze_journalistic_content(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar contenido periodístico.

        Args:
            content: Contenido

        Returns:
            Análisis de contenido periodístico
        """
        element_counts = {}
        
        # Analizar 5W1H
        content_lower = content.lower()
        w5h1_count = {}
        for question_type, patterns in self.journalistic_elements["5w1h"].items():
            count = sum(
                len(re.findall(pattern, content_lower, re.IGNORECASE))
                for pattern in patterns
            )
            w5h1_count[question_type] = count
        element_counts["5w1h"] = w5h1_count
        
        # Contar otros elementos periodísticos
        for element_type, patterns in self.journalistic_elements.items():
            if element_type != "5w1h":
                count = 0
                for pattern in patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                    count += len(matches)
                element_counts[element_type] = count
        
        # Calcular score periodístico
        total_5w1h = sum(w5h1_count.values())
        total_elements = total_5w1h + sum(
            v for k, v in element_counts.items() if k != "5w1h"
        )
        journalistic_score = min(1.0, total_elements / 20)  # Normalizar
        
        # Verificar si es contenido periodístico
        is_journalistic = (
            total_5w1h >= 3 or
            element_counts.get("quotes", 0) > 0 or
            element_counts.get("sources", 0) > 0
        )
        
        return {
            "element_counts": element_counts,
            "journalistic_score": journalistic_score,
            "total_elements": total_elements,
            "is_journalistic": is_journalistic,
            "5w1h_coverage": total_5w1h / 6 if total_5w1h > 0 else 0  # Cobertura de 5W1H
        }

    def analyze_journalistic_quality(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar calidad periodística del contenido.

        Args:
            content: Contenido

        Returns:
            Análisis de calidad periodística
        """
        journalistic_analysis = self.analyze_journalistic_content(content)
        element_counts = journalistic_analysis["element_counts"]
        
        # Verificar elementos de calidad
        has_5w1h = journalistic_analysis["5w1h_coverage"] > 0.5
        has_quotes = element_counts.get("quotes", 0) > 0
        has_sources = element_counts.get("sources", 0) > 0
        has_dates = element_counts.get("dates", 0) > 0
        has_locations = element_counts.get("locations", 0) > 0
        
        # Calcular score de calidad
        quality_score = (
            (1.0 if has_5w1h else 0.0) * 0.3 +
            (1.0 if has_quotes else 0.0) * 0.2 +
            (1.0 if has_sources else 0.0) * 0.2 +
            (1.0 if has_dates else 0.0) * 0.15 +
            (1.0 if has_locations else 0.0) * 0.15
        )
        
        return {
            "quality_score": quality_score,
            "has_5w1h": has_5w1h,
            "has_quotes": has_quotes,
            "has_sources": has_sources,
            "has_dates": has_dates,
            "has_locations": has_locations,
            "quality_level": (
                "high" if quality_score > 0.7 else
                "medium" if quality_score > 0.4 else
                "low"
            )
        }






