"""
Narrative Analyzer - Sistema de análisis de contenido narrativo
"""

import logging
import re
from typing import Dict, Any, Optional, List
from collections import Counter

logger = logging.getLogger(__name__)


class NarrativeAnalyzer:
    """Analizador de contenido narrativo"""

    def __init__(self):
        """Inicializar analizador"""
        # Elementos narrativos
        self.narrative_elements = {
            "characters": r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',  # Nombres propios
            "dialogue": [
                r'"[^"]*"',
                r''[^']*'',
                r'—[^—]*—'
            ],
            "action_verbs": [
                'corrió', 'saltó', 'gritó', 'rió', 'lloró', 'caminó',
                'ran', 'jumped', 'shouted', 'laughed', 'cried', 'walked'
            ],
            "descriptive_words": [
                'grande', 'pequeño', 'hermoso', 'feo', 'rápido', 'lento',
                'big', 'small', 'beautiful', 'ugly', 'fast', 'slow'
            ]
        }

    def analyze_narrative_structure(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar estructura narrativa del contenido.

        Args:
            content: Contenido

        Returns:
            Análisis de estructura narrativa
        """
        # Detectar diálogos
        dialogues = []
        for pattern in self.narrative_elements["dialogue"]:
            matches = re.findall(pattern, content)
            dialogues.extend(matches)
        
        # Detectar personajes (nombres propios)
        characters = re.findall(self.narrative_elements["characters"], content)
        # Filtrar palabras comunes
        common_words = {'The', 'This', 'That', 'El', 'La', 'Los', 'Las', 'A', 'An'}
        characters = [c for c in characters if c not in common_words]
        unique_characters = list(set(characters))
        
        # Contar verbos de acción
        content_lower = content.lower()
        action_verbs = sum(1 for verb in self.narrative_elements["action_verbs"] if verb in content_lower)
        
        # Contar palabras descriptivas
        descriptive_words = sum(1 for word in self.narrative_elements["descriptive_words"] if word in content_lower)
        
        # Analizar estructura (introducción, desarrollo, conclusión)
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        structure = {
            "introduction": len(paragraphs[0].split()) if paragraphs else 0,
            "development": sum(len(p.split()) for p in paragraphs[1:-1]) if len(paragraphs) > 2 else 0,
            "conclusion": len(paragraphs[-1].split()) if len(paragraphs) > 1 else 0
        }
        
        # Calcular score narrativo (0-1)
        narrative_score = (
            min(1.0, len(dialogues) / 5) * 0.3 +
            min(1.0, len(unique_characters) / 5) * 0.2 +
            min(1.0, action_verbs / 10) * 0.25 +
            min(1.0, descriptive_words / 15) * 0.25
        )
        
        return {
            "narrative_elements": {
                "dialogues": len(dialogues),
                "characters": len(unique_characters),
                "unique_characters": unique_characters[:10],
                "action_verbs": action_verbs,
                "descriptive_words": descriptive_words
            },
            "structure": structure,
            "narrative_score": narrative_score,
            "is_narrative": narrative_score > 0.4
        }

    def analyze_story_flow(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar flujo narrativo del contenido.

        Args:
            content: Contenido

        Returns:
            Análisis de flujo narrativo
        """
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        
        if len(sentences) < 3:
            return {"error": "Contenido demasiado corto para análisis de flujo"}
        
        # Palabras de transición
        transitions = {
            'entonces', 'después', 'luego', 'más tarde', 'finalmente',
            'then', 'after', 'later', 'finally', 'meanwhile', 'however'
        }
        
        transition_count = 0
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(trans in sentence_lower for trans in transitions):
                transition_count += 1
        
        transition_ratio = transition_count / len(sentences) if sentences else 0
        
        # Variación en longitud de oraciones (indica ritmo)
        sentence_lengths = [len(s.split()) for s in sentences]
        if sentence_lengths:
            avg_length = sum(sentence_lengths) / len(sentence_lengths)
            variance = sum((l - avg_length) ** 2 for l in sentence_lengths) / len(sentence_lengths)
            rhythm_score = min(1.0, variance / 100)  # Normalizar
        else:
            rhythm_score = 0
        
        # Calcular score de flujo (0-1)
        flow_score = (
            transition_ratio * 0.5 +
            rhythm_score * 0.5
        )
        
        return {
            "transition_ratio": transition_ratio,
            "rhythm_score": rhythm_score,
            "flow_score": flow_score,
            "total_sentences": len(sentences),
            "average_sentence_length": avg_length if sentence_lengths else 0,
            "has_good_flow": flow_score > 0.6
        }

    def suggest_narrative_improvements(
        self,
        content: str
    ) -> List[Dict[str, Any]]:
        """
        Sugerir mejoras narrativas.

        Args:
            content: Contenido

        Returns:
            Sugerencias de mejora narrativa
        """
        narrative_analysis = self.analyze_narrative_structure(content)
        flow_analysis = self.analyze_story_flow(content)
        
        suggestions = []
        
        if narrative_analysis["narrative_score"] < 0.4:
            suggestions.append({
                "type": "narrative_elements",
                "priority": "high",
                "issue": "Faltan elementos narrativos",
                "suggestion": "Agrega diálogos, personajes y verbos de acción para crear una narrativa más rica"
            })
        
        if narrative_analysis["narrative_elements"]["dialogues"] == 0:
            suggestions.append({
                "type": "dialogue",
                "priority": "medium",
                "issue": "No hay diálogos",
                "suggestion": "Incluye diálogos para hacer el contenido más dinámico y atractivo"
            })
        
        if flow_analysis.get("flow_score", 0) < 0.5:
            suggestions.append({
                "type": "flow",
                "priority": "medium",
                "issue": "Flujo narrativo débil",
                "suggestion": "Usa más palabras de transición y varía la longitud de las oraciones"
            })
        
        return suggestions






