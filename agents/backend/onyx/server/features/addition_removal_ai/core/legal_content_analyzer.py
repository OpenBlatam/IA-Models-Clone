"""
Legal Content Analyzer - Sistema de análisis de contenido legal
"""

import logging
import re
from typing import Dict, Any, Optional, List
from collections import Counter

logger = logging.getLogger(__name__)


class LegalContentAnalyzer:
    """Analizador de contenido legal"""

    def __init__(self):
        """Inicializar analizador"""
        # Elementos legales
        self.legal_elements = {
            "legal_terms": {
                "es": ["contrato", "acuerdo", "cláusula", "artículo", "ley", "norma", "reglamento", "jurisdicción"],
                "en": ["contract", "agreement", "clause", "article", "law", "regulation", "jurisdiction"]
            },
            "obligations": [
                r'(?:debe|must|shall|will|obligado|obligated)',
                r'(?:responsable|responsible|liable)'
            ],
            "rights": [
                r'(?:derecho|right|tiene derecho|has the right)',
                r'(?:puede|may|can|autorizado|authorized)'
            ],
            "penalties": [
                r'(?:penalización|penalty|multa|fine|sanción|sanction)',
                r'(?:incumplimiento|breach|violación|violation)'
            ],
            "references": [
                r'(?:artículo|article)\s+\d+',
                r'(?:sección|section)\s+\d+',
                r'(?:ley|law)\s+no\.?\s*\d+',
                r'Art\.\s*\d+',
                r'§\s*\d+'
            ],
            "definitions": [
                r'(?:se entiende por|means|definición|definition)',
                r'(?:a los efectos de|for the purposes of)'
            ]
        }

    def analyze_legal_content(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar contenido legal.

        Args:
            content: Contenido

        Returns:
            Análisis de contenido legal
        """
        content_lower = content.lower()
        element_counts = {}
        
        # Contar términos legales
        legal_terms_count = 0
        for lang_words in self.legal_elements["legal_terms"].values():
            legal_terms_count += sum(1 for word in lang_words if word in content_lower)
        element_counts["legal_terms"] = legal_terms_count
        
        # Contar otros elementos legales
        for element_type, patterns in self.legal_elements.items():
            if element_type != "legal_terms":
                count = 0
                for pattern in patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    count += len(matches)
                element_counts[element_type] = count
        
        # Calcular score legal
        total_elements = sum(element_counts.values())
        legal_score = min(1.0, total_elements / 20)  # Normalizar
        
        # Verificar si es contenido legal
        is_legal = (
            element_counts.get("legal_terms", 0) > 3 or
            element_counts.get("obligations", 0) > 0 or
            element_counts.get("rights", 0) > 0 or
            element_counts.get("references", 0) > 0
        )
        
        return {
            "element_counts": element_counts,
            "legal_score": legal_score,
            "total_elements": total_elements,
            "is_legal": is_legal
        }

    def analyze_legal_structure(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar estructura legal del contenido.

        Args:
            content: Contenido

        Returns:
            Análisis de estructura legal
        """
        legal_analysis = self.analyze_legal_content(content)
        
        # Detectar artículos/secciones
        articles = len(re.findall(r'(?:artículo|article|art\.)\s+\d+', content, re.IGNORECASE))
        sections = len(re.findall(r'(?:sección|section|sec\.)\s+\d+', content, re.IGNORECASE))
        
        # Detectar definiciones
        definitions = legal_analysis["element_counts"].get("definitions", 0)
        
        # Detectar obligaciones y derechos
        obligations = legal_analysis["element_counts"].get("obligations", 0)
        rights = legal_analysis["element_counts"].get("rights", 0)
        
        # Calcular score de estructura
        structure_score = (
            min(1.0, articles / 5) * 0.3 +
            min(1.0, sections / 5) * 0.2 +
            min(1.0, definitions / 3) * 0.2 +
            min(1.0, (obligations + rights) / 5) * 0.3
        )
        
        return {
            "structure_score": structure_score,
            "articles": articles,
            "sections": sections,
            "definitions": definitions,
            "obligations": obligations,
            "rights": rights,
            "has_proper_structure": structure_score > 0.5
        }






