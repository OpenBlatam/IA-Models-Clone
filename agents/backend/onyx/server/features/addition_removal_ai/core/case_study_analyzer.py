"""
Case Study Analyzer - Sistema de análisis de contenido de casos de estudio
"""

import logging
import re
from typing import Dict, Any, Optional, List
from collections import Counter

logger = logging.getLogger(__name__)


class CaseStudyAnalyzer:
    """Analizador de contenido de casos de estudio"""

    def __init__(self):
        """Inicializar analizador"""
        # Elementos de casos de estudio
        self.case_study_elements = {
            "challenge": [
                r'(?:desafío|challenge|problema|problem|reto|challenge)',
                r'(?:situación|situation|contexto|context)'
            ],
            "solution": [
                r'(?:solución|solution|solución|approach|enfoque)',
                r'(?:método|method|estrategia|strategy)'
            ],
            "results": [
                r'(?:resultado|result|resultados|results|impacto|impact)',
                r'(?:mejora|improvement|incremento|increase|reducción|reduction)'
            ],
            "metrics": [
                r'\d+%',  # Porcentajes
                r'\d+\.\d+',  # Decimales
                r'(?:aumentó|increased|mejoró|improved|redujo|reduced)'
            ],
            "testimonials": [
                r'"[^"]*"',  # Citas
                r'(?:testimonio|testimonial|dijo|said|declaró|declared)'
            ],
            "company_info": [
                r'(?:empresa|company|organización|organization|cliente|customer)',
                r'(?:industria|industry|sector|sector)'
            ],
            "timeline": [
                r'(?:tiempo|time|duración|duration|período|period)',
                r'(?:meses|months|años|years|semanas|weeks)'
            ]
        }

    def analyze_case_study_content(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar contenido de caso de estudio.

        Args:
            content: Contenido

        Returns:
            Análisis de contenido de caso de estudio
        """
        element_counts = {}
        
        # Contar elementos de caso de estudio
        for element_type, patterns in self.case_study_elements.items():
            count = 0
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                count += len(matches)
            element_counts[element_type] = count
        
        # Calcular score de caso de estudio
        total_elements = sum(element_counts.values())
        case_study_score = min(1.0, total_elements / 25)  # Normalizar
        
        # Verificar si es contenido de caso de estudio
        is_case_study = (
            element_counts.get("challenge", 0) > 0 or
            element_counts.get("solution", 0) > 0 or
            element_counts.get("results", 0) > 0
        )
        
        return {
            "element_counts": element_counts,
            "case_study_score": case_study_score,
            "total_elements": total_elements,
            "is_case_study": is_case_study
        }

    def analyze_case_study_structure(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar estructura del caso de estudio.

        Args:
            content: Contenido

        Returns:
            Análisis de estructura del caso de estudio
        """
        case_study_analysis = self.analyze_case_study_content(content)
        element_counts = case_study_analysis["element_counts"]
        
        # Verificar elementos de estructura
        has_challenge = element_counts.get("challenge", 0) > 0
        has_solution = element_counts.get("solution", 0) > 0
        has_results = element_counts.get("results", 0) > 0
        has_metrics = element_counts.get("metrics", 0) > 0
        has_testimonials = element_counts.get("testimonials", 0) > 0
        has_company_info = element_counts.get("company_info", 0) > 0
        has_timeline = element_counts.get("timeline", 0) > 0
        
        # Calcular score de estructura
        structure_score = (
            (1.0 if has_challenge else 0.0) * 0.2 +
            (1.0 if has_solution else 0.0) * 0.2 +
            (1.0 if has_results else 0.0) * 0.2 +
            (1.0 if has_metrics else 0.0) * 0.15 +
            (1.0 if has_testimonials else 0.0) * 0.1 +
            (1.0 if has_company_info else 0.0) * 0.1 +
            (1.0 if has_timeline else 0.0) * 0.05
        )
        
        return {
            "structure_score": structure_score,
            "has_challenge": has_challenge,
            "has_solution": has_solution,
            "has_results": has_results,
            "has_metrics": has_metrics,
            "has_testimonials": has_testimonials,
            "has_company_info": has_company_info,
            "has_timeline": has_timeline,
            "structure_level": (
                "high" if structure_score > 0.7 else
                "medium" if structure_score > 0.4 else
                "low"
            )
        }


