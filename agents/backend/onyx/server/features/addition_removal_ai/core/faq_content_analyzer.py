"""
FAQ Content Analyzer - Sistema de análisis de contenido de FAQ
"""

import logging
import re
from typing import Dict, Any, Optional, List
from collections import Counter

logger = logging.getLogger(__name__)


class FAQContentAnalyzer:
    """Analizador de contenido de FAQ"""

    def __init__(self):
        """Inicializar analizador"""
        # Elementos de FAQ
        self.faq_elements = {
            "questions": [
                r'^\s*[Qq]:\s*',  # Q: formato
                r'^\s*\?\s*',  # Pregunta con ?
                r'(?:pregunta|question|p\.|q\.)',
                r'^[A-Z][^?]*\?',  # Pregunta que termina con ?
            ],
            "answers": [
                r'^\s*[Aa]:\s*',  # A: formato
                r'(?:respuesta|answer|r\.|a\.)',
            ],
            "categories": [
                r'(?:categoría|category|tema|topic|sección|section)',
                r'(?:sobre|about|acerca de|regarding)'
            ],
            "links": [
                r'\[([^\]]+)\]\(([^)]+)\)',  # Markdown links
                r'<a[^>]*href[^>]*>',  # HTML links
            ],
            "examples": [
                r'(?:ejemplo|example|ejemplos|examples)',
                r'(?:por ejemplo|for example|e\.g\.|i\.e\.)'
            ],
            "steps": [
                r'(?:paso|step|paso \d+|step \d+)',
                r'^\d+\.',  # Lista numerada
            ]
        }

    def analyze_faq_content(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar contenido de FAQ.

        Args:
            content: Contenido

        Returns:
            Análisis de contenido de FAQ
        """
        element_counts = {}
        
        # Contar elementos de FAQ
        for element_type, patterns in self.faq_elements.items():
            count = 0
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                count += len(matches)
            element_counts[element_type] = count
        
        # Calcular score de FAQ
        total_elements = sum(element_counts.values())
        faq_score = min(1.0, total_elements / 20)  # Normalizar
        
        # Verificar si es contenido de FAQ
        is_faq = (
            element_counts.get("questions", 0) > 0 and
            element_counts.get("answers", 0) > 0
        )
        
        return {
            "element_counts": element_counts,
            "faq_score": faq_score,
            "total_elements": total_elements,
            "is_faq": is_faq,
            "qa_ratio": (
                element_counts.get("answers", 0) / element_counts.get("questions", 1)
                if element_counts.get("questions", 0) > 0 else 0
            )
        }

    def analyze_faq_completeness(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar completitud del FAQ.

        Args:
            content: Contenido

        Returns:
            Análisis de completitud del FAQ
        """
        faq_analysis = self.analyze_faq_content(content)
        element_counts = faq_analysis["element_counts"]
        
        # Verificar elementos de completitud
        has_questions = element_counts.get("questions", 0) > 0
        has_answers = element_counts.get("answers", 0) > 0
        has_categories = element_counts.get("categories", 0) > 0
        has_links = element_counts.get("links", 0) > 0
        has_examples = element_counts.get("examples", 0) > 0
        has_steps = element_counts.get("steps", 0) > 0
        
        # Verificar balance Q&A
        qa_balanced = faq_analysis["qa_ratio"] >= 0.8 and faq_analysis["qa_ratio"] <= 1.2
        
        # Calcular score de completitud
        completeness_score = (
            (1.0 if has_questions else 0.0) * 0.25 +
            (1.0 if has_answers else 0.0) * 0.25 +
            (1.0 if has_categories else 0.0) * 0.1 +
            (1.0 if has_links else 0.0) * 0.15 +
            (1.0 if has_examples else 0.0) * 0.15 +
            (1.0 if qa_balanced else 0.0) * 0.1
        )
        
        return {
            "completeness_score": completeness_score,
            "has_questions": has_questions,
            "has_answers": has_answers,
            "has_categories": has_categories,
            "has_links": has_links,
            "has_examples": has_examples,
            "has_steps": has_steps,
            "qa_balanced": qa_balanced,
            "completeness_level": (
                "high" if completeness_score > 0.7 else
                "medium" if completeness_score > 0.4 else
                "low"
            )
        }


