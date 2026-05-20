"""
Blog Content Analyzer - Sistema de análisis de contenido de blog
"""

import logging
import re
from typing import Dict, Any, Optional, List
from collections import Counter

logger = logging.getLogger(__name__)


class BlogContentAnalyzer:
    """Analizador de contenido de blog"""

    def __init__(self):
        """Inicializar analizador"""
        # Elementos de blog
        self.blog_elements = {
            "headings": [
                r'^#+\s+[A-Z]',  # Markdown headers
                r'<h[1-6][^>]*>',  # HTML headers
            ],
            "images": [
                r'!\[([^\]]*)\]\(([^)]+)\)',  # Markdown images
                r'<img[^>]*>',  # HTML images
            ],
            "links": [
                r'\[([^\]]+)\]\(([^)]+)\)',  # Markdown links
                r'<a[^>]*href[^>]*>',  # HTML links
            ],
            "lists": [
                r'^\s*[-*+]\s+',  # Listas con viñetas
                r'^\s*\d+\.\s+',  # Listas numeradas
            ],
            "quotes": [
                r'^>\s+',  # Markdown blockquotes
                r'<blockquote[^>]*>',  # HTML blockquotes
            ],
            "ctas": [
                r'(?:lee más|read more|descubre|discover|aprende|learn)',
                r'(?:suscríbete|subscribe|comparte|share)'
            ],
            "tags": [
                r'#\w+',  # Hashtags
                r'(?:etiqueta|tag|categoría|category)'
            ],
            "engagement": [
                r'(?:comenta|comment|opina|opinion|déjanos|let us)',
                r'(?:¿qué piensas?|what do you think|cuéntanos|tell us)'
            ]
        }

    def analyze_blog_content(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar contenido de blog.

        Args:
            content: Contenido

        Returns:
            Análisis de contenido de blog
        """
        element_counts = {}
        
        # Contar elementos de blog
        for element_type, patterns in self.blog_elements.items():
            count = 0
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                count += len(matches)
            element_counts[element_type] = count
        
        # Calcular score de blog
        total_elements = sum(element_counts.values())
        blog_score = min(1.0, total_elements / 25)  # Normalizar
        
        # Verificar si es contenido de blog
        is_blog = (
            element_counts.get("headings", 0) > 0 or
            element_counts.get("images", 0) > 0 or
            element_counts.get("links", 0) > 0
        )
        
        return {
            "element_counts": element_counts,
            "blog_score": blog_score,
            "total_elements": total_elements,
            "is_blog": is_blog
        }

    def analyze_blog_engagement(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar potencial de engagement del blog.

        Args:
            content: Contenido

        Returns:
            Análisis de engagement del blog
        """
        blog_analysis = self.analyze_blog_content(content)
        element_counts = blog_analysis["element_counts"]
        
        # Verificar elementos de engagement
        has_headings = element_counts.get("headings", 0) > 0
        has_images = element_counts.get("images", 0) > 0
        has_links = element_counts.get("links", 0) > 0
        has_ctas = element_counts.get("ctas", 0) > 0
        has_engagement = element_counts.get("engagement", 0) > 0
        
        # Calcular score de engagement
        engagement_score = (
            (1.0 if has_headings else 0.0) * 0.2 +
            (1.0 if has_images else 0.0) * 0.25 +
            (1.0 if has_links else 0.0) * 0.2 +
            (1.0 if has_ctas else 0.0) * 0.2 +
            (1.0 if has_engagement else 0.0) * 0.15
        )
        
        return {
            "engagement_score": engagement_score,
            "has_headings": has_headings,
            "has_images": has_images,
            "has_links": has_links,
            "has_ctas": has_ctas,
            "has_engagement": has_engagement,
            "engagement_level": (
                "high" if engagement_score > 0.7 else
                "medium" if engagement_score > 0.4 else
                "low"
            )
        }


