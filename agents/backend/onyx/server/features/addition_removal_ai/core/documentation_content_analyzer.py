"""
Documentation Content Analyzer - Sistema de análisis de contenido de documentación técnica
"""

import logging
import re
from typing import Dict, Any, Optional, List
from collections import Counter

logger = logging.getLogger(__name__)


class DocumentationContentAnalyzer:
    """Analizador de contenido de documentación técnica"""

    def __init__(self):
        """Inicializar analizador"""
        # Elementos de documentación
        self.documentation_elements = {
            "headers": [
                r'^#+\s+[A-Z]',  # Markdown headers
                r'<h[1-6][^>]*>',  # HTML headers
            ],
            "code_blocks": [
                r'```[\s\S]*?```',  # Bloques de código
                r'<pre[^>]*>[\s\S]*?</pre>',  # HTML pre
            ],
            "lists": [
                r'^\s*[-*+]\s+',  # Listas con viñetas
                r'^\s*\d+\.\s+',  # Listas numeradas
            ],
            "tables": [
                r'\|[^|]+\|',  # Tablas Markdown
                r'<table[^>]*>',  # Tablas HTML
            ],
            "links": [
                r'\[([^\]]+)\]\(([^)]+)\)',  # Markdown links
                r'<a[^>]*href[^>]*>',  # HTML links
            ],
            "images": [
                r'!\[([^\]]*)\]\(([^)]+)\)',  # Markdown images
                r'<img[^>]*>',  # HTML images
            ],
            "api_references": [
                r'(?:endpoint|api|method|función|function|método)',
                r'(?:GET|POST|PUT|DELETE|PATCH)\s+/[^\s]+',
            ],
            "examples": [
                r'(?:ejemplo|example|ejemplos|examples)',
                r'(?:ejemplo:|example:)',
            ]
        }

    def analyze_documentation_content(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar contenido de documentación técnica.

        Args:
            content: Contenido

        Returns:
            Análisis de contenido de documentación
        """
        element_counts = {}
        
        # Contar elementos de documentación
        for element_type, patterns in self.documentation_elements.items():
            count = 0
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                count += len(matches)
            element_counts[element_type] = count
        
        # Calcular score de documentación
        total_elements = sum(element_counts.values())
        documentation_score = min(1.0, total_elements / 30)  # Normalizar
        
        # Verificar si es contenido de documentación
        is_documentation = (
            element_counts.get("headers", 0) > 0 or
            element_counts.get("code_blocks", 0) > 0 or
            element_counts.get("api_references", 0) > 0
        )
        
        return {
            "element_counts": element_counts,
            "documentation_score": documentation_score,
            "total_elements": total_elements,
            "is_documentation": is_documentation
        }

    def analyze_documentation_structure(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar estructura de la documentación.

        Args:
            content: Contenido

        Returns:
            Análisis de estructura de documentación
        """
        doc_analysis = self.analyze_documentation_content(content)
        element_counts = doc_analysis["element_counts"]
        
        # Verificar elementos de estructura
        has_headers = element_counts.get("headers", 0) > 0
        has_code_blocks = element_counts.get("code_blocks", 0) > 0
        has_lists = element_counts.get("lists", 0) > 0
        has_tables = element_counts.get("tables", 0) > 0
        has_links = element_counts.get("links", 0) > 0
        has_examples = element_counts.get("examples", 0) > 0
        
        # Calcular score de estructura
        structure_score = (
            (1.0 if has_headers else 0.0) * 0.25 +
            (1.0 if has_code_blocks else 0.0) * 0.2 +
            (1.0 if has_lists else 0.0) * 0.15 +
            (1.0 if has_tables else 0.0) * 0.1 +
            (1.0 if has_links else 0.0) * 0.15 +
            (1.0 if has_examples else 0.0) * 0.15
        )
        
        return {
            "structure_score": structure_score,
            "has_headers": has_headers,
            "has_code_blocks": has_code_blocks,
            "has_lists": has_lists,
            "has_tables": has_tables,
            "has_links": has_links,
            "has_examples": has_examples,
            "structure_level": (
                "high" if structure_score > 0.7 else
                "medium" if structure_score > 0.4 else
                "low"
            )
        }






