"""
Persuasive Content Analyzer - Sistema de análisis de contenido persuasivo
"""

import logging
import re
from typing import Dict, Any, Optional, List
from collections import Counter

logger = logging.getLogger(__name__)


class PersuasiveContentAnalyzer:
    """Analizador de contenido persuasivo"""

    def __init__(self):
        """Inicializar analizador"""
        # Técnicas persuasivas
        self.persuasive_techniques = {
            "authority": {
                "es": ["experto", "profesional", "certificado", "reconocido", "autoridad"],
                "en": ["expert", "professional", "certified", "recognized", "authority"]
            },
            "social_proof": {
                "es": ["millones", "miles", "muchos", "todos", "popular", "recomendado"],
                "en": ["millions", "thousands", "many", "everyone", "popular", "recommended"]
            },
            "scarcity": {
                "es": ["limitado", "últimas unidades", "oferta por tiempo limitado", "exclusivo"],
                "en": ["limited", "last units", "limited time offer", "exclusive"]
            },
            "reciprocity": {
                "es": ["gratis", "regalo", "bonificación", "sin costo"],
                "en": ["free", "gift", "bonus", "no cost"]
            },
            "commitment": {
                "es": ["compromiso", "garantía", "promesa", "asegurado"],
                "en": ["commitment", "guarantee", "promise", "assured"]
            },
            "liking": {
                "es": ["como tú", "similar", "compartimos", "juntos"],
                "en": ["like you", "similar", "we share", "together"]
            }
        }

    def analyze_persuasive_elements(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar elementos persuasivos en el contenido.

        Args:
            content: Contenido

        Returns:
            Análisis de elementos persuasivos
        """
        content_lower = content.lower()
        technique_scores = {}
        
        # Analizar cada técnica persuasiva
        for technique, languages in self.persuasive_techniques.items():
            total_count = 0
            for lang_words in languages.values():
                count = sum(1 for word in lang_words if word in content_lower)
                total_count += count
            
            technique_scores[technique] = {
                "count": total_count,
                "intensity": min(1.0, total_count / 5)  # Normalizar
            }
        
        # Calcular score persuasivo general
        total_persuasive_elements = sum(score["count"] for score in technique_scores.values())
        persuasive_score = min(1.0, total_persuasive_elements / 20)  # Normalizar
        
        # Determinar técnica dominante
        dominant_technique = max(
            technique_scores.items(),
            key=lambda x: x[1]["count"]
        )[0] if technique_scores else None
        
        return {
            "technique_scores": technique_scores,
            "dominant_technique": dominant_technique,
            "persuasive_score": persuasive_score,
            "total_persuasive_elements": total_persuasive_elements
        }

    def analyze_persuasion_strength(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar fuerza persuasiva del contenido.

        Args:
            content: Contenido

        Returns:
            Análisis de fuerza persuasiva
        """
        persuasive_analysis = self.analyze_persuasive_elements(content)
        
        # Detectar llamadas a la acción
        cta_patterns = [
            r'[¡!]\s*[A-Z][^!]*!',
            r'(?:compra|descarga|regístrate|suscríbete|únete)\s+ahora',
            r'(?:buy|download|register|subscribe|join)\s+now'
        ]
        
        cta_count = 0
        for pattern in cta_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            cta_count += len(matches)
        
        # Análisis de beneficios
        benefit_words = {
            "es": ["beneficio", "ventaja", "mejora", "ahorro", "ganancia"],
            "en": ["benefit", "advantage", "improvement", "save", "gain"]
        }
        
        content_lower = content.lower()
        benefit_count = sum(
            sum(1 for word in words if word in content_lower)
            for words in benefit_words.values()
        )
        
        # Calcular fuerza persuasiva
        persuasion_strength = (
            persuasive_analysis["persuasive_score"] * 0.5 +
            min(1.0, cta_count / 3) * 0.3 +
            min(1.0, benefit_count / 10) * 0.2
        )
        
        return {
            "persuasion_strength": persuasion_strength,
            "persuasive_techniques": persuasive_analysis["technique_scores"],
            "cta_count": cta_count,
            "benefit_count": benefit_count,
            "is_persuasive": persuasion_strength > 0.6
        }






