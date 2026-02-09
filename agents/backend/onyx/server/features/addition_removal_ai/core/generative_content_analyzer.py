"""
Generative Content Analyzer - Sistema de análisis de contenido generativo
"""

import logging
import re
from typing import Dict, Any, Optional, List
from collections import Counter

logger = logging.getLogger(__name__)


class GenerativeContentAnalyzer:
    """Analizador de contenido generativo"""

    def __init__(self):
        """Inicializar analizador"""
        # Patrones que pueden indicar contenido generado
        self.generative_patterns = {
            "repetitive_phrases": r'\b(\w+)\s+\1\b',  # Palabras repetidas consecutivas
            "generic_introductions": [
                r'^en\s+este\s+artículo',
                r'^en\s+conclusión',
                r'^en\s+resumen',
                r'^this\s+article',
                r'^in\s+conclusion',
                r'^in\s+summary'
            ],
            "transition_words": {
                "es": ["además", "por otro lado", "sin embargo", "no obstante", "en consecuencia"],
                "en": ["furthermore", "moreover", "however", "nevertheless", "consequently"]
            }
        }

    def analyze_generative_indicators(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar indicadores de contenido generativo.

        Args:
            content: Contenido

        Returns:
            Análisis de indicadores generativos
        """
        indicators = {
            "repetitive_phrases": 0,
            "generic_introductions": 0,
            "transition_word_density": 0.0,
            "sentence_structure_variety": 0.0,
            "vocabulary_diversity": 0.0
        }
        
        # Detectar frases repetitivas
        repetitive_matches = re.findall(
            self.generative_patterns["repetitive_phrases"],
            content,
            re.IGNORECASE
        )
        indicators["repetitive_phrases"] = len(repetitive_matches)
        
        # Detectar introducciones genéricas
        sentences = content.split('.')
        for sentence in sentences[:5]:  # Primeras 5 oraciones
            for pattern in self.generative_patterns["generic_introductions"]:
                if re.search(pattern, sentence, re.IGNORECASE):
                    indicators["generic_introductions"] += 1
                    break
        
        # Calcular densidad de palabras de transición
        words = content.lower().split()
        all_transitions = []
        for lang_transitions in self.generative_patterns["transition_words"].values():
            all_transitions.extend(lang_transitions)
        
        transition_count = sum(1 for word in words if word in all_transitions)
        indicators["transition_word_density"] = transition_count / len(words) if words else 0
        
        # Calcular variedad de estructura de oraciones
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
        if sentence_lengths:
            avg_length = sum(sentence_lengths) / len(sentence_lengths)
            variance = sum((l - avg_length) ** 2 for l in sentence_lengths) / len(sentence_lengths)
            indicators["sentence_structure_variety"] = min(1.0, variance / 100)  # Normalizar
        
        # Calcular diversidad de vocabulario
        unique_words = len(set(words))
        total_words = len(words)
        indicators["vocabulary_diversity"] = unique_words / total_words if total_words > 0 else 0
        
        # Calcular score de generatividad (0-1, donde 1 es más probable que sea generado)
        generative_score = (
            min(1.0, indicators["repetitive_phrases"] / 10) * 0.2 +
            min(1.0, indicators["generic_introductions"] / 3) * 0.2 +
            min(1.0, indicators["transition_word_density"] * 20) * 0.2 +
            (1.0 - indicators["sentence_structure_variety"]) * 0.2 +
            (1.0 - indicators["vocabulary_diversity"]) * 0.2
        )
        
        return {
            "indicators": indicators,
            "generative_score": generative_score,
            "likely_generated": generative_score > 0.6,
            "confidence": abs(generative_score - 0.5) * 2  # Mayor confianza cuanto más alejado de 0.5
        }

    def compare_with_human_content(
        self,
        generative_content: str,
        human_content: str
    ) -> Dict[str, Any]:
        """
        Comparar contenido generativo con contenido humano.

        Args:
            generative_content: Contenido generativo
            human_content: Contenido humano

        Returns:
            Comparación
        """
        gen_analysis = self.analyze_generative_indicators(generative_content)
        human_analysis = self.analyze_generative_indicators(human_content)
        
        # Calcular diferencias
        differences = {}
        for key in gen_analysis["indicators"]:
            gen_val = gen_analysis["indicators"][key]
            human_val = human_analysis["indicators"][key]
            
            if isinstance(gen_val, (int, float)) and isinstance(human_val, (int, float)):
                differences[key] = abs(gen_val - human_val)
        
        return {
            "generative_analysis": gen_analysis,
            "human_analysis": human_analysis,
            "differences": differences,
            "generative_score_difference": abs(
                gen_analysis["generative_score"] - human_analysis["generative_score"]
            )
        }

    def detect_generative_sections(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Detectar secciones que parecen generadas.

        Args:
            content: Contenido

        Returns:
            Secciones detectadas
        """
        paragraphs = content.split('\n\n')
        sections = []
        
        for i, paragraph in enumerate(paragraphs):
            if paragraph.strip():
                analysis = self.analyze_generative_indicators(paragraph)
                
                if analysis["likely_generated"]:
                    sections.append({
                        "section_index": i,
                        "section_text": paragraph[:200],  # Primeros 200 caracteres
                        "generative_score": analysis["generative_score"],
                        "confidence": analysis["confidence"]
                    })
        
        return {
            "total_sections": len(paragraphs),
            "generative_sections": sections,
            "generative_section_count": len(sections),
            "generative_percentage": (len(sections) / len(paragraphs) * 100) if paragraphs else 0
        }

    def suggest_improvements(
        self,
        content: str
    ) -> List[Dict[str, Any]]:
        """
        Sugerir mejoras para hacer el contenido más natural.

        Args:
            content: Contenido

        Returns:
            Sugerencias de mejora
        """
        analysis = self.analyze_generative_indicators(content)
        suggestions = []
        
        if analysis["indicators"]["repetitive_phrases"] > 5:
            suggestions.append({
                "type": "repetition",
                "priority": "high",
                "issue": "Demasiadas frases repetitivas",
                "suggestion": "Varía el lenguaje y evita repeticiones innecesarias"
            })
        
        if analysis["indicators"]["generic_introductions"] > 2:
            suggestions.append({
                "type": "generic_intro",
                "priority": "medium",
                "issue": "Introducciones genéricas",
                "suggestion": "Usa introducciones más creativas y específicas"
            })
        
        if analysis["indicators"]["transition_word_density"] > 0.1:
            suggestions.append({
                "type": "transitions",
                "priority": "low",
                "issue": "Demasiadas palabras de transición",
                "suggestion": "Reduce el uso excesivo de palabras de transición"
            })
        
        if analysis["indicators"]["sentence_structure_variety"] < 0.3:
            suggestions.append({
                "type": "structure",
                "priority": "medium",
                "issue": "Poca variedad en estructura de oraciones",
                "suggestion": "Varía la longitud y estructura de las oraciones"
            })
        
        if analysis["indicators"]["vocabulary_diversity"] < 0.5:
            suggestions.append({
                "type": "vocabulary",
                "priority": "high",
                "issue": "Vocabulario limitado",
                "suggestion": "Amplía el vocabulario y usa sinónimos"
            })
        
        return suggestions






