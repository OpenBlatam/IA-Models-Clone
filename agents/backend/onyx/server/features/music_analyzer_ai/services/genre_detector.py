"""
Detector de género musical basado en características de audio
"""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class GenreDetector:
    """Detector de género musical basado en características de audio"""
    
    # Definiciones de géneros basadas en características
    GENRE_PROFILES = {
        "Rock": {
            "energy_range": (0.6, 1.0),
            "danceability_range": (0.3, 0.7),
            "valence_range": (0.3, 0.8),
            "acousticness_range": (0.0, 0.3),
            "tempo_range": (100, 180)
        },
        "Pop": {
            "energy_range": (0.5, 0.9),
            "danceability_range": (0.6, 1.0),
            "valence_range": (0.5, 1.0),
            "acousticness_range": (0.0, 0.4),
            "tempo_range": (90, 150)
        },
        "Electronic/Dance": {
            "energy_range": (0.7, 1.0),
            "danceability_range": (0.7, 1.0),
            "valence_range": (0.4, 0.9),
            "acousticness_range": (0.0, 0.2),
            "tempo_range": (120, 180)
        },
        "Hip-Hop": {
            "energy_range": (0.4, 0.9),
            "danceability_range": (0.6, 1.0),
            "valence_range": (0.2, 0.8),
            "acousticness_range": (0.0, 0.3),
            "tempo_range": (70, 120),
            "speechiness_range": (0.3, 1.0)
        },
        "Jazz": {
            "energy_range": (0.2, 0.6),
            "danceability_range": (0.3, 0.6),
            "valence_range": (0.3, 0.7),
            "acousticness_range": (0.4, 1.0),
            "tempo_range": (60, 140),
            "instrumentalness_range": (0.3, 1.0)
        },
        "Classical": {
            "energy_range": (0.1, 0.5),
            "danceability_range": (0.1, 0.4),
            "valence_range": (0.2, 0.6),
            "acousticness_range": (0.7, 1.0),
            "tempo_range": (40, 120),
            "instrumentalness_range": (0.7, 1.0)
        },
        "Country": {
            "energy_range": (0.3, 0.7),
            "danceability_range": (0.4, 0.7),
            "valence_range": (0.4, 0.8),
            "acousticness_range": (0.3, 0.8),
            "tempo_range": (70, 130)
        },
        "R&B": {
            "energy_range": (0.4, 0.8),
            "danceability_range": (0.5, 0.9),
            "valence_range": (0.4, 0.8),
            "acousticness_range": (0.0, 0.4),
            "tempo_range": (70, 120)
        },
        "Metal": {
            "energy_range": (0.8, 1.0),
            "danceability_range": (0.2, 0.5),
            "valence_range": (0.1, 0.5),
            "acousticness_range": (0.0, 0.2),
            "tempo_range": (120, 200)
        },
        "Folk": {
            "energy_range": (0.2, 0.6),
            "danceability_range": (0.3, 0.6),
            "valence_range": (0.3, 0.7),
            "acousticness_range": (0.5, 1.0),
            "tempo_range": (60, 120)
        },
        "Blues": {
            "energy_range": (0.3, 0.7),
            "danceability_range": (0.3, 0.6),
            "valence_range": (0.2, 0.6),
            "acousticness_range": (0.3, 0.9),
            "tempo_range": (60, 120)
        },
        "Reggae": {
            "energy_range": (0.3, 0.7),
            "danceability_range": (0.5, 0.8),
            "valence_range": (0.5, 0.9),
            "acousticness_range": (0.2, 0.6),
            "tempo_range": (70, 100)
        }
    }
    
    def __init__(self):
        self.logger = logger
    
    def detect_genre(self, audio_features: Dict[str, Any], 
                    audio_analysis: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Detecta el género musical basado en características de audio"""
        scores = {}
        
        energy = audio_features.get("energy", 0.5)
        danceability = audio_features.get("danceability", 0.5)
        valence = audio_features.get("valence", 0.5)
        acousticness = audio_features.get("acousticness", 0.5)
        tempo = audio_features.get("tempo", 120)
        speechiness = audio_features.get("speechiness", 0.0)
        instrumentalness = audio_features.get("instrumentalness", 0.0)
        
        # Calcular score para cada género
        for genre, profile in self.GENRE_PROFILES.items():
            score = 0.0
            matches = 0
            
            # Energy
            if "energy_range" in profile:
                min_e, max_e = profile["energy_range"]
                if min_e <= energy <= max_e:
                    score += 1.0
                    matches += 1
                else:
                    # Penalizar si está fuera del rango
                    distance = min(abs(energy - min_e), abs(energy - max_e))
                    score += max(0, 1.0 - distance)
            
            # Danceability
            if "danceability_range" in profile:
                min_d, max_d = profile["danceability_range"]
                if min_d <= danceability <= max_d:
                    score += 1.0
                    matches += 1
                else:
                    distance = min(abs(danceability - min_d), abs(danceability - max_d))
                    score += max(0, 1.0 - distance)
            
            # Valence
            if "valence_range" in profile:
                min_v, max_v = profile["valence_range"]
                if min_v <= valence <= max_v:
                    score += 1.0
                    matches += 1
                else:
                    distance = min(abs(valence - min_v), abs(valence - max_v))
                    score += max(0, 1.0 - distance)
            
            # Acousticness
            if "acousticness_range" in profile:
                min_a, max_a = profile["acousticness_range"]
                if min_a <= acousticness <= max_a:
                    score += 1.0
                    matches += 1
                else:
                    distance = min(abs(acousticness - min_a), abs(acousticness - max_a))
                    score += max(0, 1.0 - distance)
            
            # Tempo
            if "tempo_range" in profile:
                min_t, max_t = profile["tempo_range"]
                if min_t <= tempo <= max_t:
                    score += 1.0
                    matches += 1
                else:
                    distance = min(abs(tempo - min_t), abs(tempo - max_t))
                    score += max(0, 1.0 - (distance / 50))  # Normalizar por 50 BPM
            
            # Speechiness (para Hip-Hop)
            if "speechiness_range" in profile:
                min_s, max_s = profile["speechiness_range"]
                if min_s <= speechiness <= max_s:
                    score += 1.5  # Peso extra para speechiness
                    matches += 1
            
            # Instrumentalness (para Jazz, Classical)
            if "instrumentalness_range" in profile:
                min_i, max_i = profile["instrumentalness_range"]
                if min_i <= instrumentalness <= max_i:
                    score += 1.5  # Peso extra para instrumentalness
                    matches += 1
            
            # Normalizar score
            if matches > 0:
                scores[genre] = score / (matches + 1)  # +1 para normalizar mejor
        
        # Ordenar por score
        sorted_genres = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Obtener top 3
        top_genres = sorted_genres[:3]
        
        primary_genre = top_genres[0][0] if top_genres else "Unknown"
        confidence = top_genres[0][1] if top_genres else 0.0
        
        return {
            "primary_genre": primary_genre,
            "confidence": round(confidence, 3),
            "all_scores": {genre: round(score, 3) for genre, score in sorted_genres},
            "top_3": [
                {"genre": genre, "score": round(score, 3), "confidence": round(score / max(scores.values()) if scores else 0, 3)}
                for genre, score in top_genres
            ]
        }
    
    def get_genre_description(self, genre: str) -> str:
        """Obtiene una descripción del género"""
        descriptions = {
            "Rock": "Caracterizado por guitarras eléctricas, batería fuerte y energía alta",
            "Pop": "Música popular con melodías pegajosas y alta bailabilidad",
            "Electronic/Dance": "Música electrónica con ritmos de baile y sintetizadores",
            "Hip-Hop": "Rap y música urbana con énfasis en ritmo y letras",
            "Jazz": "Música improvisada con instrumentos acústicos y complejidad armónica",
            "Classical": "Música clásica orquestal con instrumentos acústicos",
            "Country": "Música country con guitarras acústicas y temáticas rurales",
            "R&B": "Rhythm and Blues con influencias de soul y funk",
            "Metal": "Rock pesado con distorsión extrema y tempos rápidos",
            "Folk": "Música folclórica acústica con narrativa",
            "Blues": "Música blues con progresiones de acordes características",
            "Reggae": "Música reggae con ritmos relajados y bajo prominente"
        }
        return descriptions.get(genre, "Género musical")

