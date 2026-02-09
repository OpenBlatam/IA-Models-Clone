"""
Report Content Analyzer - Sistema de análisis de contenido de informes
"""

import logging
import re
from typing import Dict, Any, Optional, List
from collections import Counter

logger = logging.getLogger(__name__)


class ReportContentAnalyzer:
    """Analizador de contenido de informes"""

    def __init__(self):
        """Inicializar analizador"""
        # Elementos de informes
        self.report_elements = {
            "title": [
                r'^#+\s+[A-Z]',  # Markdown headers
                r'<h1[^>]*>',  # HTML H1
            ],
            "summary": [
                r'(?:resumen|summary|sinopsis|synopsis|abstract)',
                r'(?:resumen ejecutivo|executive summary)'
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
            "tables": [
                r'\|[^|]+\|',  # Tablas Markdown
                r'<table[^>]*>',  # Tablas HTML
            ],
            "charts": [
                r'(?:gráfico|chart|gráfica|graph|diagrama|diagram)',
                r'(?:figura|figure|visualización|visualization)'
            ],
            "conclusions": [
                r'(?:conclusión|conclusion|conclusiones|conclusions)',
                r'(?:resumen final|final summary|en resumen|in summary)'
            ],
            "recommendations": [
                r'(?:recomendación|recommendation|recomendaciones|recommendations)',
                r'(?:sugerencia|suggestion|sugerencias|suggestions)'
            ],
            "appendix": [
                r'(?:apéndice|appendix|anexo|annex)',
                r'(?:referencias|references|bibliografía|bibliography)'
            ]
        }

    def analyze_report_content(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar contenido de informe.

        Args:
            content: Contenido

        Returns:
            Análisis de contenido de informe
        """
        element_counts = {}
        
        # Contar elementos de informe
        for element_type, patterns in self.report_elements.items():
            count = 0
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                count += len(matches)
            element_counts[element_type] = count
        
        # Calcular score de informe
        total_elements = sum(element_counts.values())
        report_score = min(1.0, total_elements / 30)  # Normalizar
        
        # Verificar si es contenido de informe
        is_report = (
            element_counts.get("title", 0) > 0 or
            element_counts.get("summary", 0) > 0 or
            element_counts.get("sections", 0) > 2 or
            element_counts.get("data", 0) > 0
        )
        
        return {
            "element_counts": element_counts,
            "report_score": report_score,
            "total_elements": total_elements,
            "is_report": is_report
        }

    def analyze_report_quality(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar calidad del informe.

        Args:
            content: Contenido

        Returns:
            Análisis de calidad del informe
        """
        report_analysis = self.analyze_report_content(content)
        element_counts = report_analysis["element_counts"]
        
        # Verificar elementos de calidad
        has_title = element_counts.get("title", 0) > 0
        has_summary = element_counts.get("summary", 0) > 0
        has_sections = element_counts.get("sections", 0) > 2
        has_data = element_counts.get("data", 0) > 0
        has_tables = element_counts.get("tables", 0) > 0
        has_charts = element_counts.get("charts", 0) > 0
        has_conclusions = element_counts.get("conclusions", 0) > 0
        has_recommendations = element_counts.get("recommendations", 0) > 0
        
        # Calcular score de calidad
        quality_score = (
            (1.0 if has_title else 0.0) * 0.1 +
            (1.0 if has_summary else 0.0) * 0.15 +
            (1.0 if has_sections else 0.0) * 0.15 +
            (1.0 if has_data else 0.0) * 0.15 +
            (1.0 if has_tables else 0.0) * 0.1 +
            (1.0 if has_charts else 0.0) * 0.1 +
            (1.0 if has_conclusions else 0.0) * 0.15 +
            (1.0 if has_recommendations else 0.0) * 0.1
        )
        
        return {
            "quality_score": quality_score,
            "has_title": has_title,
            "has_summary": has_summary,
            "has_sections": has_sections,
            "has_data": has_data,
            "has_tables": has_tables,
            "has_charts": has_charts,
            "has_conclusions": has_conclusions,
            "has_recommendations": has_recommendations,
            "quality_level": (
                "high" if quality_score > 0.7 else
                "medium" if quality_score > 0.4 else
                "low"
            )
        }


