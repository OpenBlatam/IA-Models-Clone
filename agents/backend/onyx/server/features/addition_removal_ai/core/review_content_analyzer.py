"""
Review Content Analyzer - Sistema de análisis de contenido de reseñas
"""

import logging
import re
from typing import Dict, Any, Optional, List
from collections import Counter

logger = logging.getLogger(__name__)


class ReviewContentAnalyzer:
    """Analizador de contenido de reseñas"""

    def __init__(self):
        """Inicializar analizador"""
        # Elementos de reseñas
        self.review_elements = {
            "ratings": [
                r'\d+\s*(?:estrellas|stars|puntos|points)',
                r'(?:calificación|rating|puntuación|score)',
                r'[1-5]\s*/\s*5',
            ],
            "pros": [
                r'(?:pros|ventajas|aspectos positivos|positive aspects)',
                r'(?:lo que me gustó|what I liked|me encantó|I loved)'
            ],
            "cons": [
                r'(?:contras|desventajas|aspectos negativos|negative aspects)',
                r'(?:lo que no me gustó|what I didn\'t like|no me gustó|I didn\'t like)'
            ],
            "recommendations": [
                r'(?:recomiendo|recommend|recomendación|recommendation)',
                r'(?:lo recomiendo|I recommend it|altamente recomendado|highly recommended)'
            ],
            "experience": [
                r'(?:experiencia|experience|uso|use|utilicé|I used)',
                r'(?:después de|after|durante|during)'
            ],
            "comparisons": [
                r'(?:comparado con|compared to|vs|versus)',
                r'(?:mejor que|better than|peor que|worse than)'
            ],
            "verification": [
                r'(?:verificado|verified|comprobado|confirmed)',
                r'(?:compra verificada|verified purchase)'
            ]
        }

    def analyze_review_content(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar contenido de reseñas.

        Args:
            content: Contenido

        Returns:
            Análisis de contenido de reseñas
        """
        element_counts = {}
        
        # Contar elementos de reseñas
        for element_type, patterns in self.review_elements.items():
            count = 0
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                count += len(matches)
            element_counts[element_type] = count
        
        # Calcular score de reseñas
        total_elements = sum(element_counts.values())
        review_score = min(1.0, total_elements / 20)  # Normalizar
        
        # Verificar si es contenido de reseñas
        is_review = (
            element_counts.get("ratings", 0) > 0 or
            element_counts.get("pros", 0) > 0 or
            element_counts.get("cons", 0) > 0 or
            element_counts.get("recommendations", 0) > 0
        )
        
        return {
            "element_counts": element_counts,
            "review_score": review_score,
            "total_elements": total_elements,
            "is_review": is_review
        }

    def analyze_review_helpfulness(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar utilidad de la reseña.

        Args:
            content: Contenido

        Returns:
            Análisis de utilidad de la reseña
        """
        review_analysis = self.analyze_review_content(content)
        element_counts = review_analysis["element_counts"]
        
        # Verificar elementos de utilidad
        has_ratings = element_counts.get("ratings", 0) > 0
        has_pros = element_counts.get("pros", 0) > 0
        has_cons = element_counts.get("cons", 0) > 0
        has_experience = element_counts.get("experience", 0) > 0
        has_recommendations = element_counts.get("recommendations", 0) > 0
        has_verification = element_counts.get("verification", 0) > 0
        
        # Calcular score de utilidad
        helpfulness_score = (
            (1.0 if has_ratings else 0.0) * 0.2 +
            (1.0 if has_pros else 0.0) * 0.2 +
            (1.0 if has_cons else 0.0) * 0.2 +
            (1.0 if has_experience else 0.0) * 0.2 +
            (1.0 if has_recommendations else 0.0) * 0.15 +
            (1.0 if has_verification else 0.0) * 0.05
        )
        
        return {
            "helpfulness_score": helpfulness_score,
            "has_ratings": has_ratings,
            "has_pros": has_pros,
            "has_cons": has_cons,
            "has_experience": has_experience,
            "has_recommendations": has_recommendations,
            "has_verification": has_verification,
            "helpfulness_level": (
                "high" if helpfulness_score > 0.7 else
                "medium" if helpfulness_score > 0.4 else
                "low"
            )
        }


