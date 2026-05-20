"""
Whitepaper Content Analyzer - Sistema de análisis de contenido de whitepapers
"""

import logging
import re
from typing import Dict, Any, Optional, List
from collections import Counter

logger = logging.getLogger(__name__)


class WhitepaperContentAnalyzer:
    """Analizador de contenido de whitepapers"""

    def __init__(self):
        """Inicializar analizador"""
        # Elementos de whitepapers
        self.whitepaper_elements = {
            "executive_summary": [
                r'(?:resumen ejecutivo|executive summary|resumen|summary)',
                r'(?:introducción|introduction|overview|vista general)'
            ],
            "sections": [
                r'^#+\s+[A-Z]',  # Markdown headers
                r'<h[1-6][^>]*>',  # HTML headers
            ],
            "data": [
                r'(?:datos|data|estadísticas|statistics|números|numbers)',
                r'\d+%',  # Porcentajes
                r'\d+\.\d+',  # Decimales
            ],
            "charts": [
                r'(?:gráfico|chart|gráfica|graph|tabla|table)',
                r'(?:figura|figure|diagrama|diagram)'
            ],
            "citations": [
                r'(?:cita|citation|referencia|reference|fuente|source)',
                r'\[.*?\]\(.*?\)',  # Referencias en markdown
                r'\([A-Z][a-z]+\s+et\s+al\.\s+\d{4}\)',  # Formato académico
            ],
            "methodology": [
                r'(?:metodología|methodology|método|method|enfoque|approach)',
                r'(?:procedimiento|procedure|proceso|process)'
            ],
            "conclusions": [
                r'(?:conclusión|conclusion|conclusiones|conclusions)',
                r'(?:resumen final|final summary|en resumen|in summary)'
            ],
            "recommendations": [
                r'(?:recomendación|recommendation|recomendaciones|recommendations)',
                r'(?:sugerencia|suggestion|sugerencias|suggestions)'
            ]
        }

    def analyze_whitepaper_content(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar contenido de whitepaper.

        Args:
            content: Contenido

        Returns:
            Análisis de contenido de whitepaper
        """
        element_counts = {}
        
        # Contar elementos de whitepaper
        for element_type, patterns in self.whitepaper_elements.items():
            count = 0
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                count += len(matches)
            element_counts[element_type] = count
        
        # Calcular score de whitepaper
        total_elements = sum(element_counts.values())
        whitepaper_score = min(1.0, total_elements / 30)  # Normalizar
        
        # Verificar si es contenido de whitepaper
        is_whitepaper = (
            element_counts.get("executive_summary", 0) > 0 or
            element_counts.get("sections", 0) > 2 or
            element_counts.get("data", 0) > 0 or
            element_counts.get("citations", 0) > 0
        )
        
        return {
            "element_counts": element_counts,
            "whitepaper_score": whitepaper_score,
            "total_elements": total_elements,
            "is_whitepaper": is_whitepaper
        }

    def analyze_whitepaper_quality(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar calidad del whitepaper.

        Args:
            content: Contenido

        Returns:
            Análisis de calidad del whitepaper
        """
        whitepaper_analysis = self.analyze_whitepaper_content(content)
        element_counts = whitepaper_analysis["element_counts"]
        
        # Verificar elementos de calidad
        has_summary = element_counts.get("executive_summary", 0) > 0
        has_sections = element_counts.get("sections", 0) > 2
        has_data = element_counts.get("data", 0) > 0
        has_charts = element_counts.get("charts", 0) > 0
        has_citations = element_counts.get("citations", 0) > 0
        has_methodology = element_counts.get("methodology", 0) > 0
        has_conclusions = element_counts.get("conclusions", 0) > 0
        has_recommendations = element_counts.get("recommendations", 0) > 0
        
        # Calcular score de calidad
        quality_score = (
            (1.0 if has_summary else 0.0) * 0.15 +
            (1.0 if has_sections else 0.0) * 0.15 +
            (1.0 if has_data else 0.0) * 0.15 +
            (1.0 if has_charts else 0.0) * 0.1 +
            (1.0 if has_citations else 0.0) * 0.15 +
            (1.0 if has_methodology else 0.0) * 0.1 +
            (1.0 if has_conclusions else 0.0) * 0.1 +
            (1.0 if has_recommendations else 0.0) * 0.1
        )
        
        return {
            "quality_score": quality_score,
            "has_summary": has_summary,
            "has_sections": has_sections,
            "has_data": has_data,
            "has_charts": has_charts,
            "has_citations": has_citations,
            "has_methodology": has_methodology,
            "has_conclusions": has_conclusions,
            "has_recommendations": has_recommendations,
            "quality_level": (
                "high" if quality_score > 0.7 else
                "medium" if quality_score > 0.4 else
                "low"
            )
        }


