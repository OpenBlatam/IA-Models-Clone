"""
Sales Content Analyzer - Sistema de análisis de contenido de ventas
"""

import logging
import re
from typing import Dict, Any, Optional, List
from collections import Counter

logger = logging.getLogger(__name__)


class SalesContentAnalyzer:
    """Analizador de contenido de ventas"""

    def __init__(self):
        """Inicializar analizador"""
        # Elementos de ventas
        self.sales_elements = {
            "value_propositions": [
                r'(?:valor|value|propuesta|proposition|solución|solution)',
                r'(?:resuelve|solves|mejora|improves|ayuda|helps)'
            ],
            "objections": [
                r'(?:pero|but|sin embargo|however|aunque|although)',
                r'(?:preocupación|concern|duda|doubt|objeción|objection)'
            ],
            "closing": [
                r'(?:cierra|close|finaliza|finalize|completa|complete)',
                r'(?:ahora|now|inmediatamente|immediately|hoy|today)'
            ],
            "pricing": [
                r'\$\s*\d+',
                r'\d+\s*(?:USD|EUR|GBP|MXN|ARS)',
                r'(?:precio|price|costo|cost|tarifa|rate)',
                r'(?:descuento|discount|oferta|offer|promoción|promotion)'
            ],
            "features": [
                r'(?:característica|feature|función|function|capacidad|capability)',
                r'(?:incluye|includes|ofrece|offers|proporciona|provides)'
            ],
            "testimonials": [
                r'"[^"]*"',
                r'(?:cliente|customer|usuario|user|testimonio|testimonial)'
            ]
        }

    def analyze_sales_content(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar contenido de ventas.

        Args:
            content: Contenido

        Returns:
            Análisis de contenido de ventas
        """
        element_counts = {}
        
        # Contar elementos de ventas
        for element_type, patterns in self.sales_elements.items():
            count = 0
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                count += len(matches)
            element_counts[element_type] = count
        
        # Calcular score de ventas
        total_elements = sum(element_counts.values())
        sales_score = min(1.0, total_elements / 20)  # Normalizar
        
        # Verificar si es contenido de ventas
        is_sales = (
            element_counts.get("value_propositions", 0) > 0 or
            element_counts.get("pricing", 0) > 0 or
            element_counts.get("closing", 0) > 0
        )
        
        return {
            "element_counts": element_counts,
            "sales_score": sales_score,
            "total_elements": total_elements,
            "is_sales": is_sales
        }

    def analyze_sales_potential(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar potencial de ventas del contenido.

        Args:
            content: Contenido

        Returns:
            Análisis de potencial de ventas
        """
        sales_analysis = self.analyze_sales_content(content)
        element_counts = sales_analysis["element_counts"]
        
        # Verificar elementos de potencial
        has_value_prop = element_counts.get("value_propositions", 0) > 0
        has_pricing = element_counts.get("pricing", 0) > 0
        has_features = element_counts.get("features", 0) > 0
        has_testimonials = element_counts.get("testimonials", 0) > 0
        has_closing = element_counts.get("closing", 0) > 0
        
        # Calcular score de potencial
        potential_score = (
            (1.0 if has_value_prop else 0.0) * 0.3 +
            (1.0 if has_pricing else 0.0) * 0.25 +
            (1.0 if has_features else 0.0) * 0.2 +
            (1.0 if has_testimonials else 0.0) * 0.15 +
            (1.0 if has_closing else 0.0) * 0.1
        )
        
        return {
            "potential_score": potential_score,
            "has_value_proposition": has_value_prop,
            "has_pricing": has_pricing,
            "has_features": has_features,
            "has_testimonials": has_testimonials,
            "has_closing": has_closing,
            "potential_level": (
                "high" if potential_score > 0.7 else
                "medium" if potential_score > 0.4 else
                "low"
            )
        }






