"""
Servicio avanzado de Machine Learning para análisis musical
"""

import logging
from typing import Dict, List, Any, Optional
import numpy as np
from collections import Counter

from .spotify_service import SpotifyService
from .genre_detector import GenreDetector
from .emotion_analyzer import EmotionAnalyzer
from .harmonic_analyzer import HarmonicAnalyzer
from .intelligent_recommender import IntelligentRecommender

logger = logging.getLogger(__name__)


class AdvancedMLService:
    """Servicio avanzado de ML para análisis musical profundo"""
    
    def __init__(self):
        self.spotify = SpotifyService()
        self.genre_detector = GenreDetector()
        self.emotion_analyzer = EmotionAnalyzer()
        self.harmonic_analyzer = HarmonicAnalyzer()
        self.recommender = IntelligentRecommender()
        self.logger = logger
    
    def analyze_music_style(self, audio_features: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza el estilo musical completo"""
        genre = self.genre_detector.detect_genre(audio_features)
        emotion = self.emotion_analyzer.analyze_emotions(audio_features)
        
        # Determinar estilo basado en combinación de género y emoción
        style_mapping = {
            ("Rock", "energetic"): "Hard Rock",
            ("Rock", "calm"): "Soft Rock",
            ("Pop", "happy"): "Upbeat Pop",
            ("Pop", "sad"): "Ballad Pop",
            ("Electronic", "energetic"): "EDM",
            ("Electronic", "calm"): "Ambient Electronic",
            ("Jazz", "calm"): "Smooth Jazz",
            ("Jazz", "energetic"): "Bebop Jazz",
            ("Hip-Hop", "energetic"): "Trap/Hip-Hop",
            ("Hip-Hop", "calm"): "Chill Hip-Hop",
            ("Classical", "calm"): "Classical",
            ("Classical", "energetic"): "Orchestral"
        }
        
        primary_genre = genre.get("primary_genre", "Unknown")
        primary_emotion = emotion.get("primary_emotion", "Unknown")
        style_key = (primary_genre, primary_emotion)
        
        detected_style = style_mapping.get(style_key, f"{primary_genre} - {primary_emotion}")
        
        return {
            "style": detected_style,
            "genre": primary_genre,
            "emotion": primary_emotion,
            "confidence": (genre.get("confidence", 0) + emotion.get("confidence", 0)) / 2,
            "characteristics": {
                "energy_level": self._categorize_energy(audio_features.get("energy", 0.5)),
                "danceability_level": self._categorize_danceability(audio_features.get("danceability", 0.5)),
                "mood": primary_emotion,
                "acoustic": audio_features.get("acousticness", 0.5) > 0.5
            }
        }
    
    def predict_musical_era(self, audio_features: Dict[str, Any],
                           track_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Predice la era musical basada en características"""
        # Si tenemos fecha de lanzamiento, usarla
        if track_info and track_info.get("release_date"):
            release_date = track_info["release_date"]
            try:
                year = int(release_date.split("-")[0])
                era = self._year_to_era(year)
                return {
                    "era": era,
                    "year": year,
                    "method": "release_date",
                    "confidence": 1.0
                }
            except:
                pass
        
        # Si no, predecir basado en características
        acousticness = audio_features.get("acousticness", 0.5)
        energy = audio_features.get("energy", 0.5)
        tempo = audio_features.get("tempo", 120)
        
        # Lógica simple basada en características
        if acousticness > 0.7 and tempo < 100:
            era = "Classical/Early 20th Century"
            confidence = 0.6
        elif acousticness > 0.5 and energy < 0.5:
            era = "1960s-1970s"
            confidence = 0.5
        elif energy > 0.7 and tempo > 140:
            era = "2000s-2010s (Electronic/Dance)"
            confidence = 0.6
        elif energy > 0.6:
            era = "1980s-1990s"
            confidence = 0.5
        else:
            era = "Modern (2010s-2020s)"
            confidence = 0.4
        
        return {
            "era": era,
            "method": "feature_based",
            "confidence": confidence
        }
    
    def analyze_musical_influences(self, audio_features: Dict[str, Any],
                                  genre_analysis: Dict[str, Any]) -> List[str]:
        """Analiza posibles influencias musicales"""
        influences = []
        
        genre = genre_analysis.get("primary_genre", "Unknown")
        energy = audio_features.get("energy", 0.5)
        acousticness = audio_features.get("acousticness", 0.5)
        
        # Influencias basadas en género
        genre_influences = {
            "Rock": ["Classic Rock", "Blues", "Folk"],
            "Pop": ["R&B", "Soul", "Disco"],
            "Electronic": ["Techno", "House", "Ambient"],
            "Jazz": ["Blues", "Classical", "Gospel"],
            "Hip-Hop": ["Funk", "Soul", "Reggae"],
            "Classical": ["Baroque", "Romantic", "Modern"]
        }
        
        if genre in genre_influences:
            influences.extend(genre_influences[genre])
        
        # Influencias basadas en características
        if acousticness > 0.7:
            influences.append("Acoustic/Folk")
        
        if energy > 0.8:
            influences.append("Punk/Hardcore")
        
        return list(set(influences))  # Eliminar duplicados
    
    def calculate_musical_diversity(self, tracks_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calcula la diversidad musical de un conjunto de tracks"""
        if not tracks_data:
            return {"diversity_score": 0, "description": "No hay datos"}
        
        # Extraer características
        genres = []
        emotions = []
        keys = []
        tempos = []
        
        for track in tracks_data:
            analysis = track.get("analysis", {})
            genres.append(analysis.get("genre_analysis", {}).get("primary_genre", "Unknown"))
            emotions.append(analysis.get("emotion_analysis", {}).get("primary_emotion", "Unknown"))
            keys.append(analysis.get("musical_analysis", {}).get("key_signature", "Unknown"))
            tempos.append(analysis.get("musical_analysis", {}).get("tempo", {}).get("bpm", 120))
        
        # Calcular diversidad
        genre_diversity = len(set(genres)) / len(genres) if genres else 0
        emotion_diversity = len(set(emotions)) / len(emotions) if emotions else 0
        key_diversity = len(set(keys)) / len(keys) if keys else 0
        
        # Diversidad de tempo (rango relativo)
        tempo_range = max(tempos) - min(tempos) if tempos else 0
        tempo_diversity = min(tempo_range / 200, 1.0)  # Normalizar por 200 BPM
        
        # Score combinado
        diversity_score = (
            genre_diversity * 0.3 +
            emotion_diversity * 0.3 +
            key_diversity * 0.2 +
            tempo_diversity * 0.2
        )
        
        if diversity_score < 0.3:
            level = "Low"
            description = "Música muy homogénea, mismo estilo"
        elif diversity_score < 0.6:
            level = "Medium"
            description = "Música con variación moderada"
        else:
            level = "High"
            description = "Música muy diversa, múltiples estilos"
        
        return {
            "diversity_score": round(diversity_score, 3),
            "level": level,
            "description": description,
            "breakdown": {
                "genre_diversity": round(genre_diversity, 3),
                "emotion_diversity": round(emotion_diversity, 3),
                "key_diversity": round(key_diversity, 3),
                "tempo_diversity": round(tempo_diversity, 3)
            },
            "statistics": {
                "unique_genres": len(set(genres)),
                "unique_emotions": len(set(emotions)),
                "unique_keys": len(set(keys)),
                "tempo_range": round(tempo_range, 1)
            }
        }
    
    def _categorize_energy(self, energy: float) -> str:
        """Categoriza el nivel de energía"""
        if energy < 0.3:
            return "Very Low"
        elif energy < 0.5:
            return "Low"
        elif energy < 0.7:
            return "Medium"
        elif energy < 0.9:
            return "High"
        else:
            return "Very High"
    
    def _categorize_danceability(self, danceability: float) -> str:
        """Categoriza el nivel de bailabilidad"""
        if danceability < 0.3:
            return "Not Danceable"
        elif danceability < 0.5:
            return "Slightly Danceable"
        elif danceability < 0.7:
            return "Moderately Danceable"
        elif danceability < 0.9:
            return "Very Danceable"
        else:
            return "Extremely Danceable"
    
    def _year_to_era(self, year: int) -> str:
        """Convierte un año a era musical"""
        if year < 1920:
            return "Early 20th Century"
        elif year < 1950:
            return "1940s-1950s (Big Band/Jazz)"
        elif year < 1965:
            return "1950s-1960s (Rock & Roll)"
        elif year < 1980:
            return "1960s-1970s (Classic Rock)"
        elif year < 1995:
            return "1980s-1990s (Pop/Rock)"
        elif year < 2010:
            return "2000s (Pop/Electronic)"
        else:
            return "2010s-2020s (Modern)"

