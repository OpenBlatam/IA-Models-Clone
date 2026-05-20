"""
Support Content Analyzer - Sistema de análisis de contenido de soporte técnico
"""

import logging
import re
from typing import Dict, Any, Optional, List
from collections import Counter

logger = logging.getLogger(__name__)


class SupportContentAnalyzer:
    """Analizador de contenido de soporte técnico"""

    def __init__(self):
        """Inicializar analizador"""
        # Elementos de soporte
        self.support_elements = {
            "troubleshooting": [
                r'(?:solución|solution|resolver|resolve|solucionar|fix)',
                r'(?:problema|problem|error|issue|fallo|failure)'
            ],
            "steps": [
                r'(?:paso|step|paso \d+|step \d+)',
                r'(?:primero|first|segundo|second|tercero|third)',
                r'^\d+\.',  # Lista numerada
            ],
            "warnings": [
                r'(?:advertencia|warning|precaución|caution|atención|attention)',
                r'(?:importante|important|nota|note)'
            ],
            "code_examples": [
                r'```[\s\S]*?```',  # Bloques de código
                r'`[^`]+`',  # Código inline
            ],
            "links": [
                r'https?://[^\s]+',  # URLs
                r'\[([^\]]+)\]\(([^)]+)\)',  # Markdown links
            ],
            "faq": [
                r'(?:pregunta|question|pregunta frecuente|frequently asked)',
                r'(?:respuesta|answer|r:|a:)'
            ]
        }

    def analyze_support_content(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar contenido de soporte técnico.

        Args:
            content: Contenido

        Returns:
            Análisis de contenido de soporte
        """
        element_counts = {}
        
        # Contar elementos de soporte
        for element_type, patterns in self.support_elements.items():
            count = 0
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                count += len(matches)
            element_counts[element_type] = count
        
        # Calcular score de soporte
        total_elements = sum(element_counts.values())
        support_score = min(1.0, total_elements / 20)  # Normalizar
        
        # Verificar si es contenido de soporte
        is_support = (
            element_counts.get("troubleshooting", 0) > 0 or
            element_counts.get("steps", 0) > 0 or
            element_counts.get("code_examples", 0) > 0
        )
        
        return {
            "element_counts": element_counts,
            "support_score": support_score,
            "total_elements": total_elements,
            "is_support": is_support
        }

    def analyze_support_quality(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar calidad del contenido de soporte.

        Args:
            content: Contenido

        Returns:
            Análisis de calidad de soporte
        """
        support_analysis = self.analyze_support_content(content)
        element_counts = support_analysis["element_counts"]
        
        # Verificar elementos de calidad
        has_troubleshooting = element_counts.get("troubleshooting", 0) > 0
        has_steps = element_counts.get("steps", 0) > 0
        has_warnings = element_counts.get("warnings", 0) > 0
        has_code_examples = element_counts.get("code_examples", 0) > 0
        has_links = element_counts.get("links", 0) > 0
        
        # Calcular score de calidad
        quality_score = (
            (1.0 if has_troubleshooting else 0.0) * 0.3 +
            (1.0 if has_steps else 0.0) * 0.25 +
            (1.0 if has_warnings else 0.0) * 0.15 +
            (1.0 if has_code_examples else 0.0) * 0.2 +
            (1.0 if has_links else 0.0) * 0.1
        )
        
        return {
            "quality_score": quality_score,
            "has_troubleshooting": has_troubleshooting,
            "has_steps": has_steps,
            "has_warnings": has_warnings,
            "has_code_examples": has_code_examples,
            "has_links": has_links,
            "quality_level": (
                "high" if quality_score > 0.7 else
                "medium" if quality_score > 0.4 else
                "low"
            )
        }






