"""
Email Marketing Analyzer - Sistema de análisis de contenido de email marketing
"""

import logging
import re
from typing import Dict, Any, Optional, List
from collections import Counter

logger = logging.getLogger(__name__)


class EmailMarketingAnalyzer:
    """Analizador de contenido de email marketing"""

    def __init__(self):
        """Inicializar analizador"""
        # Elementos de email marketing
        self.email_elements = {
            "subject_line": [
                r'(?:asunto|subject|título|title)',
            ],
            "personalization": [
                r'\{\{.*?\}\}',  # Variables de personalización
                r'(?:nombre|name|usuario|user|cliente|customer)',
            ],
            "ctas": [
                r'(?:haz clic|click|compra|buy|descarga|download|regístrate|register)',
                r'(?:botón|button|enlace|link|acción|action)',
            ],
            "urgency": [
                r'(?:limitado|limited|últimas horas|last hours|solo hoy|only today)',
                r'(?:oferta por tiempo limitado|limited time offer)'
            ],
            "benefits": [
                r'(?:beneficio|benefit|ventaja|advantage|ahorra|save)',
                r'(?:gratis|free|descuento|discount|oferta|offer)'
            ],
            "social_proof": [
                r'(?:millones|millions|miles|thousands|muchos|many)',
                r'(?:recomendado|recommended|popular|éxito|success)'
            ],
            "unsubscribe": [
                r'(?:cancelar suscripción|unsubscribe|darse de baja|opt out)',
            ],
            "images": [
                r'<img[^>]*>',  # HTML images
                r'!\[([^\]]*)\]\(([^)]+)\)',  # Markdown images
            ]
        }

    def analyze_email_content(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar contenido de email marketing.

        Args:
            content: Contenido

        Returns:
            Análisis de contenido de email
        """
        element_counts = {}
        
        # Contar elementos de email
        for element_type, patterns in self.email_elements.items():
            count = 0
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                count += len(matches)
            element_counts[element_type] = count
        
        # Calcular score de email
        total_elements = sum(element_counts.values())
        email_score = min(1.0, total_elements / 20)  # Normalizar
        
        # Verificar si es contenido de email
        is_email = (
            element_counts.get("ctas", 0) > 0 or
            element_counts.get("benefits", 0) > 0 or
            element_counts.get("personalization", 0) > 0
        )
        
        return {
            "element_counts": element_counts,
            "email_score": email_score,
            "total_elements": total_elements,
            "is_email": is_email
        }

    def analyze_email_effectiveness(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar efectividad del email marketing.

        Args:
            content: Contenido

        Returns:
            Análisis de efectividad del email
        """
        email_analysis = self.analyze_email_content(content)
        element_counts = email_analysis["element_counts"]
        
        # Verificar elementos de efectividad
        has_personalization = element_counts.get("personalization", 0) > 0
        has_ctas = element_counts.get("ctas", 0) > 0
        has_benefits = element_counts.get("benefits", 0) > 0
        has_urgency = element_counts.get("urgency", 0) > 0
        has_social_proof = element_counts.get("social_proof", 0) > 0
        has_unsubscribe = element_counts.get("unsubscribe", 0) > 0
        
        # Calcular score de efectividad
        effectiveness_score = (
            (1.0 if has_personalization else 0.0) * 0.2 +
            (1.0 if has_ctas else 0.0) * 0.25 +
            (1.0 if has_benefits else 0.0) * 0.2 +
            (1.0 if has_urgency else 0.0) * 0.15 +
            (1.0 if has_social_proof else 0.0) * 0.15 +
            (1.0 if has_unsubscribe else 0.0) * 0.05  # Requerido legalmente
        )
        
        return {
            "effectiveness_score": effectiveness_score,
            "has_personalization": has_personalization,
            "has_ctas": has_ctas,
            "has_benefits": has_benefits,
            "has_urgency": has_urgency,
            "has_social_proof": has_social_proof,
            "has_unsubscribe": has_unsubscribe,
            "effectiveness_level": (
                "high" if effectiveness_score > 0.7 else
                "medium" if effectiveness_score > 0.4 else
                "low"
            )
        }


