"""
Marketing Content Analyzer - Sistema de análisis de contenido de marketing
"""

import logging
import re
from typing import Dict, Any, Optional, List
from collections import Counter

logger = logging.getLogger(__name__)


class MarketingContentAnalyzer:
    """Analizador de contenido de marketing"""

    def __init__(self):
        """Inicializar analizador"""
        # Elementos de marketing
        self.marketing_elements = {
            "brand_mentions": [
                r'@\w+',  # Menciones
                r'(?:marca|brand|producto|product)',
            ],
            "ctas": [
                r'(?:compra|buy|descarga|download|regístrate|register|suscríbete|subscribe)\s+ahora',
                r'(?:haz clic|click|únete|join|aprovecha|take advantage)',
                r'[¡!]\s*[A-Z][^!]*!'
            ],
            "benefits": [
                r'(?:beneficio|benefit|ventaja|advantage|mejora|improvement)',
                r'(?:gratis|free|descuento|discount|oferta|offer)'
            ],
            "social_proof": [
                r'(?:millones|millions|miles|thousands|muchos|many)',
                r'(?:recomendado|recommended|popular|éxito|success)'
            ],
            "urgency": [
                r'(?:limitado|limited|últimas unidades|last units|solo hoy|only today)',
                r'(?:oferta por tiempo limitado|limited time offer)'
            ],
            "testimonials": [
                r'"[^"]*"',  # Citas
                r'(?:testimonio|testimonial|reseña|review|opinión|opinion)'
            ]
        }

    def analyze_marketing_content(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar contenido de marketing.

        Args:
            content: Contenido

        Returns:
            Análisis de contenido de marketing
        """
        element_counts = {}
        
        # Contar elementos de marketing
        for element_type, patterns in self.marketing_elements.items():
            count = 0
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                count += len(matches)
            element_counts[element_type] = count
        
        # Calcular score de marketing
        total_elements = sum(element_counts.values())
        marketing_score = min(1.0, total_elements / 20)  # Normalizar
        
        # Verificar si es contenido de marketing
        is_marketing = (
            element_counts.get("ctas", 0) > 0 or
            element_counts.get("benefits", 0) > 0 or
            element_counts.get("brand_mentions", 0) > 0
        )
        
        return {
            "element_counts": element_counts,
            "marketing_score": marketing_score,
            "total_elements": total_elements,
            "is_marketing": is_marketing
        }

    def analyze_marketing_effectiveness(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar efectividad de marketing del contenido.

        Args:
            content: Contenido

        Returns:
            Análisis de efectividad de marketing
        """
        marketing_analysis = self.analyze_marketing_content(content)
        element_counts = marketing_analysis["element_counts"]
        
        # Verificar elementos de efectividad
        has_ctas = element_counts.get("ctas", 0) > 0
        has_benefits = element_counts.get("benefits", 0) > 0
        has_social_proof = element_counts.get("social_proof", 0) > 0
        has_urgency = element_counts.get("urgency", 0) > 0
        has_testimonials = element_counts.get("testimonials", 0) > 0
        
        # Calcular score de efectividad
        effectiveness_score = (
            (1.0 if has_ctas else 0.0) * 0.3 +
            (1.0 if has_benefits else 0.0) * 0.25 +
            (1.0 if has_social_proof else 0.0) * 0.2 +
            (1.0 if has_urgency else 0.0) * 0.15 +
            (1.0 if has_testimonials else 0.0) * 0.1
        )
        
        return {
            "effectiveness_score": effectiveness_score,
            "has_ctas": has_ctas,
            "has_benefits": has_benefits,
            "has_social_proof": has_social_proof,
            "has_urgency": has_urgency,
            "has_testimonials": has_testimonials,
            "effectiveness_level": (
                "high" if effectiveness_score > 0.7 else
                "medium" if effectiveness_score > 0.4 else
                "low"
            )
        }






