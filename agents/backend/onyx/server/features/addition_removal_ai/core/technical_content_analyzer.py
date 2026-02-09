"""
Technical Content Analyzer - Sistema de análisis de contenido técnico
"""

import logging
import re
from typing import Dict, Any, Optional, List
from collections import Counter

logger = logging.getLogger(__name__)


class TechnicalContentAnalyzer:
    """Analizador de contenido técnico"""

    def __init__(self):
        """Inicializar analizador"""
        # Elementos técnicos
        self.technical_elements = {
            "code_snippets": [
                r'```[\s\S]*?```',
                r'<code[^>]*>[\s\S]*?</code>',
                r'<pre[^>]*>[\s\S]*?</pre>'
            ],
            "technical_terms": {
                "es": ["algoritmo", "función", "variable", "clase", "objeto", "método", "API", "framework"],
                "en": ["algorithm", "function", "variable", "class", "object", "method", "API", "framework"]
            },
            "formulas": [
                r'\$[^\$]+\$',  # LaTeX inline
                r'\$\$[^\$]+\$\$',  # LaTeX block
                r'[A-Za-z]+\s*=\s*[A-Za-z0-9\+\-\*\/\(\)]+'  # Fórmulas simples
            ],
            "diagrams": [
                r'```mermaid',
                r'```graph',
                r'<svg[^>]*>',
                r'graph\s+[A-Z]'
            ],
            "references": [
                r'\[.*?\]\(.*?\)',  # Markdown links
                r'\[.*?\]\[.*?\]',  # Referencias
                r'@\w+',  # Menciones
                r'RFC\s+\d+',  # RFC references
                r'ISO\s+\d+'  # ISO references
            ]
        }

    def analyze_technical_content(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar contenido técnico.

        Args:
            content: Contenido

        Returns:
            Análisis de contenido técnico
        """
        element_counts = {}
        
        # Contar elementos técnicos
        for element_type, patterns in self.technical_elements.items():
            count = 0
            if element_type == "technical_terms":
                content_lower = content.lower()
                for lang_words in patterns.values():
                    count += sum(1 for word in lang_words if word in content_lower)
            else:
                for pattern in patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                    count += len(matches)
            
            element_counts[element_type] = count
        
        # Calcular score técnico
        total_elements = sum(element_counts.values())
        technical_score = min(1.0, total_elements / 20)  # Normalizar
        
        # Verificar si es contenido técnico
        is_technical = (
            element_counts.get("code_snippets", 0) > 0 or
            element_counts.get("technical_terms", 0) > 5 or
            element_counts.get("formulas", 0) > 0
        )
        
        return {
            "element_counts": element_counts,
            "technical_score": technical_score,
            "total_elements": total_elements,
            "is_technical": is_technical
        }

    def analyze_technical_complexity(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar complejidad técnica del contenido.

        Args:
            content: Contenido

        Returns:
            Análisis de complejidad técnica
        """
        technical_analysis = self.analyze_technical_content(content)
        
        # Contar bloques de código
        code_blocks = len(re.findall(r'```[\s\S]*?```', content))
        
        # Análisis de términos técnicos avanzados
        advanced_terms = {
            "es": ["arquitectura", "optimización", "escalabilidad", "concurrencia", "distribuido"],
            "en": ["architecture", "optimization", "scalability", "concurrency", "distributed"]
        }
        
        content_lower = content.lower()
        advanced_term_count = sum(
            sum(1 for word in words if word in content_lower)
            for words in advanced_terms.values()
        )
        
        # Calcular complejidad
        complexity = (
            min(1.0, code_blocks / 5) * 0.4 +
            min(1.0, advanced_term_count / 10) * 0.3 +
            technical_analysis["technical_score"] * 0.3
        )
        
        return {
            "complexity": complexity,
            "code_blocks": code_blocks,
            "advanced_terms": advanced_term_count,
            "technical_score": technical_analysis["technical_score"],
            "complexity_level": (
                "high" if complexity > 0.7 else
                "medium" if complexity > 0.4 else
                "low"
            )
        }






