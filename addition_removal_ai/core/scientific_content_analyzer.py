"""
Scientific Content Analyzer - Sistema de análisis de contenido científico
"""

import logging
import re
from typing import Dict, Any, Optional, List
from collections import Counter

logger = logging.getLogger(__name__)


class ScientificContentAnalyzer:
    """Analizador de contenido científico"""

    def __init__(self):
        """Inicializar analizador"""
        # Elementos científicos
        self.scientific_elements = {
            "hypotheses": [
                r'(?:hipótesis|hypothesis|suposición|assumption)',
                r'(?:si\s+[a-z]+\s+entonces|if\s+[a-z]+\s+then)'
            ],
            "methodology": [
                r'(?:metodología|methodology|método|method)',
                r'(?:procedimiento|procedure|protocolo|protocol)'
            ],
            "data": [
                r'(?:datos|data|resultados|results|mediciones|measurements)',
                r'\d+\.\d+',  # Números decimales
                r'\d+\s*%',  # Porcentajes
            ],
            "citations": [
                r'\[.*?\]\(.*?\)',  # Referencias
                r'\(.*?\d{4}.*?\)',  # Citas con año
                r'et\s+al\.',  # Et al.
                r'doi:',  # DOI
            ],
            "formulas": [
                r'\$[^\$]+\$',  # LaTeX
                r'[A-Za-z]+\s*=\s*[A-Za-z0-9\+\-\*\/\(\)]+',  # Fórmulas
                r'[A-Za-z]+\s*\([^)]+\)\s*=\s*[A-Za-z0-9\+\-\*\/\(\)]+'  # Funciones
            ],
            "statistics": [
                r'(?:estadística|statistics|p\s*=\s*\d+\.\d+|correlación|correlation)',
                r'(?:significativo|significant|p\s*<\s*0\.\d+)'
            ]
        }

    def analyze_scientific_content(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar contenido científico.

        Args:
            content: Contenido

        Returns:
            Análisis de contenido científico
        """
        element_counts = {}
        
        # Contar elementos científicos
        for element_type, patterns in self.scientific_elements.items():
            count = 0
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                count += len(matches)
            element_counts[element_type] = count
        
        # Calcular score científico
        total_elements = sum(element_counts.values())
        scientific_score = min(1.0, total_elements / 25)  # Normalizar
        
        # Verificar si es contenido científico
        is_scientific = (
            element_counts.get("methodology", 0) > 0 or
            element_counts.get("data", 0) > 3 or
            element_counts.get("citations", 0) > 2 or
            element_counts.get("statistics", 0) > 0
        )
        
        return {
            "element_counts": element_counts,
            "scientific_score": scientific_score,
            "total_elements": total_elements,
            "is_scientific": is_scientific
        }

    def analyze_scientific_rigor(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar rigor científico del contenido.

        Args:
            content: Contenido

        Returns:
            Análisis de rigor científico
        """
        scientific_analysis = self.analyze_scientific_content(content)
        element_counts = scientific_analysis["element_counts"]
        
        # Verificar elementos de rigor
        has_hypothesis = element_counts.get("hypotheses", 0) > 0
        has_methodology = element_counts.get("methodology", 0) > 0
        has_data = element_counts.get("data", 0) > 3
        has_citations = element_counts.get("citations", 0) > 2
        has_statistics = element_counts.get("statistics", 0) > 0
        
        # Calcular score de rigor
        rigor_score = (
            (1.0 if has_hypothesis else 0.0) * 0.2 +
            (1.0 if has_methodology else 0.0) * 0.2 +
            (1.0 if has_data else 0.0) * 0.2 +
            (1.0 if has_citations else 0.0) * 0.2 +
            (1.0 if has_statistics else 0.0) * 0.2
        )
        
        return {
            "rigor_score": rigor_score,
            "has_hypothesis": has_hypothesis,
            "has_methodology": has_methodology,
            "has_data": has_data,
            "has_citations": has_citations,
            "has_statistics": has_statistics,
            "rigor_level": (
                "high" if rigor_score > 0.8 else
                "medium" if rigor_score > 0.5 else
                "low"
            )
        }






