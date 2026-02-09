"""
Creative Content Analyzer - Sistema de análisis de contenido creativo
"""

import logging
import re
from typing import Dict, Any, Optional, List
from collections import Counter

logger = logging.getLogger(__name__)


class CreativeContentAnalyzer:
    """Analizador de contenido creativo"""

    def __init__(self):
        """Inicializar analizador"""
        # Elementos creativos
        self.creative_elements = {
            "metaphors": [
                r'(?:como|like|as)\s+[a-z]+\s+(?:es|is|are)',
                r'(?:es|is|are)\s+(?:como|like|as)\s+[a-z]+'
            ],
            "similes": [
                r'[a-z]+\s+(?:como|like|as)\s+[a-z]+',
                r'(?:tan|as)\s+[a-z]+\s+(?:como|as)\s+[a-z]+'
            ],
            "alliteration": r'\b([a-z])\w*\s+\1\w*',  # Palabras que empiezan con la misma letra
            "rhyme": r'\b\w+[aeiou]\b\s+\w+[aeiou]\b',  # Rimas simples
            "imagery": {
                "es": ["brillante", "oscuro", "colorido", "vibrante", "resplandeciente"],
                "en": ["bright", "dark", "colorful", "vibrant", "glowing"]
            },
            "personification": [
                r'[A-Z][a-z]+\s+(?:sonríe|ríe|llora|grita|habla)',
                r'[A-Z][a-z]+\s+(?:smiles|laughs|cries|shouts|speaks)'
            ]
        }

    def analyze_creative_elements(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar elementos creativos en el contenido.

        Args:
            content: Contenido

        Returns:
            Análisis de elementos creativos
        """
        element_counts = {}
        
        # Contar metáforas
        metaphors = sum(
            len(re.findall(pattern, content, re.IGNORECASE))
            for pattern in self.creative_elements["metaphors"]
        )
        element_counts["metaphors"] = metaphors
        
        # Contar símiles
        similes = sum(
            len(re.findall(pattern, content, re.IGNORECASE))
            for pattern in self.creative_elements["similes"]
        )
        element_counts["similes"] = similes
        
        # Contar aliteración
        alliteration = len(re.findall(self.creative_elements["alliteration"], content, re.IGNORECASE))
        element_counts["alliteration"] = alliteration
        
        # Contar imágenes
        content_lower = content.lower()
        imagery_count = sum(
            sum(1 for word in words if word in content_lower)
            for words in self.creative_elements["imagery"].values()
        )
        element_counts["imagery"] = imagery_count
        
        # Contar personificación
        personification = sum(
            len(re.findall(pattern, content))
            for pattern in self.creative_elements["personification"]
        )
        element_counts["personification"] = personification
        
        # Calcular score creativo
        total_elements = sum(element_counts.values())
        creative_score = min(1.0, total_elements / 15)  # Normalizar
        
        return {
            "element_counts": element_counts,
            "creative_score": creative_score,
            "total_elements": total_elements,
            "is_creative": creative_score > 0.4
        }

    def analyze_creativity_level(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar nivel de creatividad del contenido.

        Args:
            content: Contenido

        Returns:
            Análisis de nivel de creatividad
        """
        creative_analysis = self.analyze_creative_elements(content)
        
        # Análisis de originalidad (palabras únicas)
        words = content.split()
        unique_words = len(set(words))
        total_words = len(words)
        originality = unique_words / total_words if total_words > 0 else 0
        
        # Análisis de variación en estructura
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        sentence_lengths = [len(s.split()) for s in sentences]
        if sentence_lengths:
            avg_length = sum(sentence_lengths) / len(sentence_lengths)
            variance = sum((l - avg_length) ** 2 for l in sentence_lengths) / len(sentence_lengths)
            variation = min(1.0, variance / 100)  # Normalizar
        else:
            variation = 0
        
        # Calcular nivel de creatividad
        creativity_level = (
            creative_analysis["creative_score"] * 0.4 +
            originality * 0.3 +
            variation * 0.3
        )
        
        return {
            "creativity_level": creativity_level,
            "creative_elements": creative_analysis["element_counts"],
            "originality": originality,
            "variation": variation,
            "level": (
                "very_creative" if creativity_level > 0.7 else
                "creative" if creativity_level > 0.4 else
                "moderate" if creativity_level > 0.2 else
                "low"
            )
        }






