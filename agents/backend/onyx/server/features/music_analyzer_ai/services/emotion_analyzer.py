"""
Analizador de emociones y sentimiento en música
"""

import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class EmotionAnalyzer:
    """Analizador de emociones basado en características de audio"""
    
    # Mapeo de emociones basado en características
    EMOTION_PROFILES = {
        "happy": {
            "valence_range": (0.6, 1.0),
            "energy_range": (0.5, 1.0),
            "danceability_range": (0.5, 1.0),
            "tempo_range": (100, 180)
        },
        "sad": {
            "valence_range": (0.0, 0.4),
            "energy_range": (0.0, 0.5),
            "danceability_range": (0.0, 0.5),
            "tempo_range": (40, 100)
        },
        "energetic": {
            "valence_range": (0.4, 1.0),
            "energy_range": (0.7, 1.0),
            "danceability_range": (0.6, 1.0),
            "tempo_range": (120, 200)
        },
        "calm": {
            "valence_range": (0.4, 0.8),
            "energy_range": (0.0, 0.4),
            "danceability_range": (0.0, 0.5),
            "tempo_range": (40, 100)
        },
        "angry": {
            "valence_range": (0.0, 0.3),
            "energy_range": (0.8, 1.0),
            "danceability_range": (0.2, 0.6),
            "tempo_range": (140, 200)
        },
        "romantic": {
            "valence_range": (0.5, 0.8),
            "energy_range": (0.2, 0.6),
            "danceability_range": (0.3, 0.7),
            "tempo_range": (60, 120),
            "acousticness_range": (0.3, 1.0)
        },
        "nostalgic": {
            "valence_range": (0.3, 0.6),
            "energy_range": (0.2, 0.6),
            "danceability_range": (0.2, 0.6),
            "tempo_range": (60, 120)
        },
        "mysterious": {
            "valence_range": (0.2, 0.5),
            "energy_range": (0.2, 0.6),
            "danceability_range": (0.1, 0.5),
            "tempo_range": (50, 110)
        }
    }
    
    def __init__(self):
        self.logger = logger
    
    def analyze_emotions(self, audio_features: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza las emociones en la música"""
        scores = {}
        
        valence = audio_features.get("valence", 0.5)
        energy = audio_features.get("energy", 0.5)
        danceability = audio_features.get("danceability", 0.5)
        tempo = audio_features.get("tempo", 120)
        acousticness = audio_features.get("acousticness", 0.5)
        
        # Calcular score para cada emoción
        for emotion, profile in self.EMOTION_PROFILES.items():
            score = 0.0
            matches = 0
            
            # Valence
            if "valence_range" in profile:
                min_v, max_v = profile["valence_range"]
                if min_v <= valence <= max_v:
                    score += 1.5
                    matches += 1
                else:
                    distance = min(abs(valence - min_v), abs(valence - max_v))
                    score += max(0, 1.5 - distance)
            
            # Energy
            if "energy_range" in profile:
                min_e, max_e = profile["energy_range"]
                if min_e <= energy <= max_e:
                    score += 1.5
                    matches += 1
                else:
                    distance = min(abs(energy - min_e), abs(energy - max_e))
                    score += max(0, 1.5 - distance)
            
            # Danceability
            if "danceability_range" in profile:
                min_d, max_d = profile["danceability_range"]
                if min_d <= danceability <= max_d:
                    score += 1.0
                    matches += 1
                else:
                    distance = min(abs(danceability - min_d), abs(danceability - max_d))
                    score += max(0, 1.0 - distance)
            
            # Tempo
            if "tempo_range" in profile:
                min_t, max_t = profile["tempo_range"]
                if min_t <= tempo <= max_t:
                    score += 1.0
                    matches += 1
                else:
                    distance = min(abs(tempo - min_t), abs(tempo - max_t))
                    score += max(0, 1.0 - (distance / 50))
            
            # Acousticness (para algunas emociones)
            if "acousticness_range" in profile:
                min_a, max_a = profile["acousticness_range"]
                if min_a <= acousticness <= max_a:
                    score += 0.5
                    matches += 1
            
            # Normalizar
            if matches > 0:
                scores[emotion] = score / (matches + 1)
        
        # Ordenar por score
        sorted_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Obtener emociones principales
        primary_emotion = sorted_emotions[0][0] if sorted_emotions else "neutral"
        confidence = sorted_emotions[0][1] if sorted_emotions else 0.0
        
        # Obtener top 3
        top_emotions = sorted_emotions[:3]
        
        return {
            "primary_emotion": primary_emotion,
            "confidence": round(confidence, 3),
            "all_emotions": {emotion: round(score, 3) for emotion, score in sorted_emotions},
            "top_3": [
                {
                    "emotion": emotion,
                    "score": round(score, 3),
                    "confidence": round(score / max(scores.values()) if scores else 0, 3)
                }
                for emotion, score in top_emotions
            ],
            "emotional_profile": self._get_emotional_profile(valence, energy)
        }
    
    def _get_emotional_profile(self, valence: float, energy: float) -> Dict[str, Any]:
        """Obtiene un perfil emocional basado en valence y energy"""
        # Mapeo de cuadrantes emocionales
        if valence > 0.5 and energy > 0.5:
            quadrant = "High Valence, High Energy"
            description = "Música positiva y energética"
        elif valence > 0.5 and energy <= 0.5:
            quadrant = "High Valence, Low Energy"
            description = "Música positiva y relajante"
        elif valence <= 0.5 and energy > 0.5:
            quadrant = "Low Valence, High Energy"
            description = "Música intensa y poderosa"
        else:
            quadrant = "Low Valence, Low Energy"
            description = "Música melancólica y suave"
        
        return {
            "quadrant": quadrant,
            "description": description,
            "valence": round(valence, 3),
            "energy": round(energy, 3)
        }
    
    def get_emotion_description(self, emotion: str) -> str:
        """Obtiene una descripción de la emoción"""
        descriptions = {
            "happy": "Música alegre y positiva que eleva el ánimo",
            "sad": "Música melancólica y emotiva",
            "energetic": "Música llena de energía y vitalidad",
            "calm": "Música tranquila y relajante",
            "angry": "Música intensa y agresiva",
            "romantic": "Música romántica y sentimental",
            "nostalgic": "Música nostálgica que evoca recuerdos",
            "mysterious": "Música misteriosa y enigmática",
            "neutral": "Música con perfil emocional neutro"
        }
        return descriptions.get(emotion, "Emoción musical")

