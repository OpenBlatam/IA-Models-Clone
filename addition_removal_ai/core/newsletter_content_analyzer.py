"""
Newsletter Content Analyzer - Sistema de análisis de contenido de newsletters
"""

import logging
import re
from typing import Dict, Any, Optional, List
from collections import Counter

logger = logging.getLogger(__name__)


class NewsletterContentAnalyzer:
    """Analizador de contenido de newsletters"""

    def __init__(self):
        """Inicializar analizador"""
        # Elementos de newsletters
        self.newsletter_elements = {
            "subject": [
                r'(?:asunto|subject|título|title)',
            ],
            "greeting": [
                r'(?:hola|hello|hi|estimado|dear|querido|querida)',
                r'(?:saludos|greetings|buenos días|good morning)'
            ],
            "sections": [
                r'(?:sección|section|parte|part|artículo|article)',
                r'^#+\s+[A-Z]',  # Markdown headers
            ],
            "ctas": [
                r'(?:haz clic|click|lee más|read more|descubre|discover)',
                r'(?:suscríbete|subscribe|comparte|share)',
                r'<button[^>]*>',  # HTML buttons
            ],
            "links": [
                r'\[([^\]]+)\]\(([^)]+)\)',  # Markdown links
                r'<a[^>]*href[^>]*>',  # HTML links
            ],
            "images": [
                r'!\[([^\]]*)\]\(([^)]+)\)',  # Markdown images
                r'<img[^>]*>',  # HTML images
            ],
            "unsubscribe": [
                r'(?:cancelar suscripción|unsubscribe|darse de baja|opt out)',
            ],
            "social_links": [
                r'(?:síguenos|follow us|conecta|connect|redes sociales|social media)',
            ]
        }

    def analyze_newsletter_content(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar contenido de newsletter.

        Args:
            content: Contenido

        Returns:
            Análisis de contenido de newsletter
        """
        element_counts = {}
        
        # Contar elementos de newsletter
        for element_type, patterns in self.newsletter_elements.items():
            count = 0
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                count += len(matches)
            element_counts[element_type] = count
        
        # Calcular score de newsletter
        total_elements = sum(element_counts.values())
        newsletter_score = min(1.0, total_elements / 25)  # Normalizar
        
        # Verificar si es contenido de newsletter
        is_newsletter = (
            element_counts.get("greeting", 0) > 0 or
            element_counts.get("sections", 0) > 0 or
            element_counts.get("ctas", 0) > 0
        )
        
        return {
            "element_counts": element_counts,
            "newsletter_score": newsletter_score,
            "total_elements": total_elements,
            "is_newsletter": is_newsletter
        }

    def analyze_newsletter_effectiveness(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar efectividad del newsletter.

        Args:
            content: Contenido

        Returns:
            Análisis de efectividad del newsletter
        """
        newsletter_analysis = self.analyze_newsletter_content(content)
        element_counts = newsletter_analysis["element_counts"]
        
        # Verificar elementos de efectividad
        has_subject = element_counts.get("subject", 0) > 0
        has_greeting = element_counts.get("greeting", 0) > 0
        has_sections = element_counts.get("sections", 0) > 0
        has_ctas = element_counts.get("ctas", 0) > 0
        has_links = element_counts.get("links", 0) > 0
        has_images = element_counts.get("images", 0) > 0
        has_unsubscribe = element_counts.get("unsubscribe", 0) > 0
        has_social = element_counts.get("social_links", 0) > 0
        
        # Calcular score de efectividad
        effectiveness_score = (
            (1.0 if has_subject else 0.0) * 0.1 +
            (1.0 if has_greeting else 0.0) * 0.1 +
            (1.0 if has_sections else 0.0) * 0.2 +
            (1.0 if has_ctas else 0.0) * 0.2 +
            (1.0 if has_links else 0.0) * 0.15 +
            (1.0 if has_images else 0.0) * 0.15 +
            (1.0 if has_unsubscribe else 0.0) * 0.05 +  # Requerido legalmente
            (1.0 if has_social else 0.0) * 0.05
        )
        
        return {
            "effectiveness_score": effectiveness_score,
            "has_subject": has_subject,
            "has_greeting": has_greeting,
            "has_sections": has_sections,
            "has_ctas": has_ctas,
            "has_links": has_links,
            "has_images": has_images,
            "has_unsubscribe": has_unsubscribe,
            "has_social": has_social,
            "effectiveness_level": (
                "high" if effectiveness_score > 0.7 else
                "medium" if effectiveness_score > 0.4 else
                "low"
            )
        }


