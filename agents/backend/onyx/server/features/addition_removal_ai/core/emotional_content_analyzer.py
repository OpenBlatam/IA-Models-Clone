"""
Emotional Content Analyzer - Sistema de análisis de contenido emocional
"""

import logging
import re
from typing import Dict, Any, Optional, List
from collections import Counter

logger = logging.getLogger(__name__)


class EmotionalContentAnalyzer:
    """Analizador de contenido emocional"""

    def __init__(self):
        """Inicializar analizador"""
        # Palabras emocionales por categoría
        self.emotional_words = {
            "joy": {
                "es": ["alegría", "felicidad", "gozo", "diversión", "risa", "sonrisa"],
                "en": ["joy", "happiness", "delight", "fun", "laugh", "smile"]
            },
            "sadness": {
                "es": ["tristeza", "dolor", "llanto", "melancolía", "desánimo"],
                "en": ["sadness", "pain", "cry", "melancholy", "discouragement"]
            },
            "anger": {
                "es": ["ira", "rabia", "enojo", "furia", "molestia"],
                "en": ["anger", "rage", "fury", "annoyance", "irritation"]
            },
            "fear": {
                "es": ["miedo", "temor", "ansiedad", "preocupación", "pánico"],
                "en": ["fear", "dread", "anxiety", "worry", "panic"]
            },
            "surprise": {
                "es": ["sorpresa", "asombro", "increíble", "impresionante"],
                "en": ["surprise", "amazement", "incredible", "impressive"]
            },
            "trust": {
                "es": ["confianza", "seguridad", "certeza", "garantía"],
                "en": ["trust", "confidence", "certainty", "guarantee"]
            },
            "anticipation": {
                "es": ["anticipación", "esperanza", "expectativa", "anhelo"],
                "en": ["anticipation", "hope", "expectation", "longing"]
            }
        }

    def analyze_emotional_content(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar contenido emocional.

        Args:
            content: Contenido

        Returns:
            Análisis de contenido emocional
        """
        content_lower = content.lower()
        emotion_scores = {}
        
        # Analizar cada emoción
        for emotion, languages in self.emotional_words.items():
            total_count = 0
            for lang_words in languages.values():
                count = sum(1 for word in lang_words if word in content_lower)
                total_count += count
            
            emotion_scores[emotion] = {
                "count": total_count,
                "intensity": min(1.0, total_count / 10)  # Normalizar
            }
        
        # Determinar emoción dominante
        dominant_emotion = max(
            emotion_scores.items(),
            key=lambda x: x[1]["count"]
        )[0] if emotion_scores else None
        
        # Calcular score emocional general
        total_emotional_words = sum(score["count"] for score in emotion_scores.values())
        emotional_score = min(1.0, total_emotional_words / 30)  # Normalizar
        
        # Análisis de polaridad (positivo vs negativo)
        positive_emotions = ["joy", "surprise", "trust", "anticipation"]
        negative_emotions = ["sadness", "anger", "fear"]
        
        positive_score = sum(emotion_scores[e]["count"] for e in positive_emotions)
        negative_score = sum(emotion_scores[e]["count"] for e in negative_emotions)
        
        polarity = "positive" if positive_score > negative_score else "negative" if negative_score > positive_score else "neutral"
        
        return {
            "emotion_scores": emotion_scores,
            "dominant_emotion": dominant_emotion,
            "emotional_score": emotional_score,
            "polarity": polarity,
            "positive_score": positive_score,
            "negative_score": negative_score,
            "total_emotional_words": total_emotional_words
        }

    def analyze_emotional_arc(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar arco emocional del contenido.

        Args:
            content: Contenido

        Returns:
            Análisis de arco emocional
        """
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        if len(paragraphs) < 3:
            return {"error": "Contenido demasiado corto para análisis de arco emocional"}
        
        # Analizar emociones por párrafo
        paragraph_emotions = []
        for paragraph in paragraphs:
            analysis = self.analyze_emotional_content(paragraph)
            paragraph_emotions.append({
                "emotion": analysis["dominant_emotion"],
                "polarity": analysis["polarity"],
                "score": analysis["emotional_score"]
            })
        
        # Detectar cambios emocionales
        emotional_changes = []
        for i in range(len(paragraph_emotions) - 1):
            if paragraph_emotions[i]["emotion"] != paragraph_emotions[i+1]["emotion"]:
                emotional_changes.append({
                    "from": paragraph_emotions[i]["emotion"],
                    "to": paragraph_emotions[i+1]["emotion"],
                    "position": i
                })
        
        # Calcular variabilidad emocional
        unique_emotions = len(set(p["emotion"] for p in paragraph_emotions))
        variability = unique_emotions / len(paragraphs) if paragraphs else 0
        
        return {
            "paragraph_emotions": paragraph_emotions,
            "emotional_changes": emotional_changes,
            "variability": variability,
            "has_emotional_arc": len(emotional_changes) > 0
        }






