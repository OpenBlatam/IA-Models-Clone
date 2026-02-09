"""
Landing Page Analyzer - Sistema de análisis de contenido de landing pages
"""

import logging
import re
from typing import Dict, Any, Optional, List
from collections import Counter

logger = logging.getLogger(__name__)


class LandingPageAnalyzer:
    """Analizador de contenido de landing pages"""

    def __init__(self):
        """Inicializar analizador"""
        # Elementos de landing pages
        self.landing_elements = {
            "headlines": [
                r'<h1[^>]*>',  # HTML H1
                r'^#\s+[A-Z]',  # Markdown H1
            ],
            "hero_section": [
                r'(?:hero|sección principal|main section)',
                r'(?:banner|encabezado principal|main header)'
            ],
            "ctas": [
                r'(?:compra|buy|descarga|download|regístrate|register|suscríbete|subscribe)',
                r'(?:haz clic|click|únete|join|aprovecha|take advantage)',
                r'<button[^>]*>',  # HTML buttons
            ],
            "benefits": [
                r'(?:beneficio|benefit|ventaja|advantage|mejora|improvement)',
                r'(?:por qué|why|razones|reasons)'
            ],
            "testimonials": [
                r'"[^"]*"',  # Citas
                r'(?:testimonio|testimonial|reseña|review|opinión|opinion)',
                r'(?:dice|says|afirma|states)'
            ],
            "features": [
                r'(?:característica|feature|función|function|capacidad|capability)',
                r'(?:incluye|includes|ofrece|offers|proporciona|provides)'
            ],
            "social_proof": [
                r'(?:millones|millions|miles|thousands|muchos|many)',
                r'(?:recomendado|recommended|popular|éxito|success)',
                r'(?:clientes|customers|usuarios|users)'
            ],
            "forms": [
                r'<form[^>]*>',  # HTML forms
                r'(?:formulario|form|campo|field|input)',
            ],
            "trust_signals": [
                r'(?:garantía|guarantee|devolución|return|seguro|secure)',
                r'(?:certificado|certified|verificado|verified)'
            ]
        }

    def analyze_landing_content(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar contenido de landing page.

        Args:
            content: Contenido

        Returns:
            Análisis de contenido de landing page
        """
        element_counts = {}
        
        # Contar elementos de landing page
        for element_type, patterns in self.landing_elements.items():
            count = 0
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                count += len(matches)
            element_counts[element_type] = count
        
        # Calcular score de landing page
        total_elements = sum(element_counts.values())
        landing_score = min(1.0, total_elements / 30)  # Normalizar
        
        # Verificar si es contenido de landing page
        is_landing = (
            element_counts.get("headlines", 0) > 0 or
            element_counts.get("ctas", 0) > 0 or
            element_counts.get("benefits", 0) > 0 or
            element_counts.get("forms", 0) > 0
        )
        
        return {
            "element_counts": element_counts,
            "landing_score": landing_score,
            "total_elements": total_elements,
            "is_landing": is_landing
        }

    def analyze_landing_conversion(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar potencial de conversión de la landing page.

        Args:
            content: Contenido

        Returns:
            Análisis de conversión
        """
        landing_analysis = self.analyze_landing_content(content)
        element_counts = landing_analysis["element_counts"]
        
        # Verificar elementos de conversión
        has_headline = element_counts.get("headlines", 0) > 0
        has_ctas = element_counts.get("ctas", 0) > 0
        has_benefits = element_counts.get("benefits", 0) > 0
        has_testimonials = element_counts.get("testimonials", 0) > 0
        has_features = element_counts.get("features", 0) > 0
        has_social_proof = element_counts.get("social_proof", 0) > 0
        has_forms = element_counts.get("forms", 0) > 0
        has_trust = element_counts.get("trust_signals", 0) > 0
        
        # Calcular score de conversión
        conversion_score = (
            (1.0 if has_headline else 0.0) * 0.15 +
            (1.0 if has_ctas else 0.0) * 0.2 +
            (1.0 if has_benefits else 0.0) * 0.15 +
            (1.0 if has_testimonials else 0.0) * 0.1 +
            (1.0 if has_features else 0.0) * 0.1 +
            (1.0 if has_social_proof else 0.0) * 0.1 +
            (1.0 if has_forms else 0.0) * 0.15 +
            (1.0 if has_trust else 0.0) * 0.05
        )
        
        return {
            "conversion_score": conversion_score,
            "has_headline": has_headline,
            "has_ctas": has_ctas,
            "has_benefits": has_benefits,
            "has_testimonials": has_testimonials,
            "has_features": has_features,
            "has_social_proof": has_social_proof,
            "has_forms": has_forms,
            "has_trust": has_trust,
            "conversion_level": (
                "high" if conversion_score > 0.7 else
                "medium" if conversion_score > 0.4 else
                "low"
            )
        }


