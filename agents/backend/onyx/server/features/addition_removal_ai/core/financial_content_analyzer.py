"""
Financial Content Analyzer - Sistema de análisis de contenido financiero
"""

import logging
import re
from typing import Dict, Any, Optional, List
from collections import Counter

logger = logging.getLogger(__name__)


class FinancialContentAnalyzer:
    """Analizador de contenido financiero"""

    def __init__(self):
        """Inicializar analizador"""
        # Elementos financieros
        self.financial_elements = {
            "financial_terms": {
                "es": ["inversión", "capital", "dividendo", "interés", "rentabilidad", "beneficio", "pérdida", "activo", "pasivo"],
                "en": ["investment", "capital", "dividend", "interest", "profitability", "profit", "loss", "asset", "liability"]
            },
            "currencies": [
                r'\$\s*\d+',
                r'€\s*\d+',
                r'£\s*\d+',
                r'\d+\s*(?:USD|EUR|GBP|MXN|ARS)',
                r'(?:dólar|euro|libra|peso)'
            ],
            "percentages": [
                r'\d+\.?\d*\s*%',
                r'(?:por ciento|percent)'
            ],
            "financial_metrics": [
                r'(?:ROI|ROE|ROA|EBITDA|P\/E|EPS)',
                r'(?:retorno|return|rendimiento|yield)'
            ],
            "dates": [
                r'\d{1,2}\/\d{1,2}\/\d{4}',
                r'\d{4}-\d{2}-\d{2}',
                r'(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\s+\d+'
            ],
            "risks": [
                r'(?:riesgo|risk|volatilidad|volatility|incertidumbre|uncertainty)',
                r'(?:advertencia|warning|precaución|caution)'
            ]
        }

    def analyze_financial_content(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar contenido financiero.

        Args:
            content: Contenido

        Returns:
            Análisis de contenido financiero
        """
        content_lower = content.lower()
        element_counts = {}
        
        # Contar términos financieros
        financial_terms_count = 0
        for lang_words in self.financial_elements["financial_terms"].values():
            financial_terms_count += sum(1 for word in lang_words if word in content_lower)
        element_counts["financial_terms"] = financial_terms_count
        
        # Contar otros elementos financieros
        for element_type, patterns in self.financial_elements.items():
            if element_type != "financial_terms":
                count = 0
                for pattern in patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    count += len(matches)
                element_counts[element_type] = count
        
        # Calcular score financiero
        total_elements = sum(element_counts.values())
        financial_score = min(1.0, total_elements / 25)  # Normalizar
        
        # Verificar si es contenido financiero
        is_financial = (
            element_counts.get("financial_terms", 0) > 3 or
            element_counts.get("currencies", 0) > 0 or
            element_counts.get("financial_metrics", 0) > 0
        )
        
        return {
            "element_counts": element_counts,
            "financial_score": financial_score,
            "total_elements": total_elements,
            "is_financial": is_financial
        }

    def analyze_financial_accuracy(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar precisión financiera del contenido.

        Args:
            content: Contenido

        Returns:
            Análisis de precisión financiera
        """
        financial_analysis = self.analyze_financial_content(content)
        element_counts = financial_analysis["element_counts"]
        
        # Verificar elementos de precisión
        has_currencies = element_counts.get("currencies", 0) > 0
        has_percentages = element_counts.get("percentages", 0) > 0
        has_dates = element_counts.get("dates", 0) > 0
        has_metrics = element_counts.get("financial_metrics", 0) > 0
        has_risks = element_counts.get("risks", 0) > 0
        
        # Calcular score de precisión
        accuracy_score = (
            (1.0 if has_currencies else 0.0) * 0.2 +
            (1.0 if has_percentages else 0.0) * 0.2 +
            (1.0 if has_dates else 0.0) * 0.2 +
            (1.0 if has_metrics else 0.0) * 0.2 +
            (1.0 if has_risks else 0.0) * 0.2
        )
        
        return {
            "accuracy_score": accuracy_score,
            "has_currencies": has_currencies,
            "has_percentages": has_percentages,
            "has_dates": has_dates,
            "has_metrics": has_metrics,
            "has_risks": has_risks,
            "accuracy_level": (
                "high" if accuracy_score > 0.8 else
                "medium" if accuracy_score > 0.5 else
                "low"
            )
        }






