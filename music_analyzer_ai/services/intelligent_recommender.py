"""
Sistema de recomendaciones inteligentes basado en ML
"""

import logging
from typing import Dict, List, Any, Optional
from collections import Counter
import math

logger = logging.getLogger(__name__)


class IntelligentRecommender:
    """Sistema de recomendaciones inteligentes"""
    
    def __init__(self):
        self.logger = logger
    
    def recommend_similar_tracks(self, target_features: Dict[str, Any],
                                 all_tracks: List[Dict[str, Any]],
                                 limit: int = 10) -> List[Dict[str, Any]]:
        """Recomienda tracks similares basado en características"""
        if not all_tracks:
            return []
        
        # Calcular similitud para cada track
        similarities = []
        
        for track in all_tracks:
            track_features = track.get("audio_features", {})
            if not track_features:
                continue
            
            similarity = self._calculate_similarity(target_features, track_features)
            
            similarities.append({
                "track": track,
                "similarity": similarity
            })
        
        # Ordenar por similitud
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Retornar top tracks
        return [
            {
                **s["track"],
                "similarity_score": round(s["similarity"], 3)
            }
            for s in similarities[:limit]
        ]
    
    def _calculate_similarity(self, features1: Dict[str, Any],
                             features2: Dict[str, Any]) -> float:
        """Calcula similitud entre dos sets de características"""
        # Características a comparar
        attributes = [
            "energy", "danceability", "valence", "acousticness",
            "instrumentalness", "liveness", "speechiness"
        ]
        
        total_similarity = 0.0
        count = 0
        
        for attr in attributes:
            val1 = features1.get(attr, 0.5)
            val2 = features2.get(attr, 0.5)
            
            # Similitud basada en diferencia absoluta
            diff = abs(val1 - val2)
            similarity = 1.0 - diff
            total_similarity += similarity
            count += 1
        
        # Comparar tempo (con normalización)
        tempo1 = features1.get("tempo", 120)
        tempo2 = features2.get("tempo", 120)
        tempo_diff = abs(tempo1 - tempo2) / 200.0  # Normalizar por 200 BPM
        tempo_similarity = 1.0 - min(tempo_diff, 1.0)
        total_similarity += tempo_similarity
        count += 1
        
        # Comparar key (mismo key = más similar)
        key1 = features1.get("key", -1)
        key2 = features2.get("key", -1)
        if key1 >= 0 and key2 >= 0:
            if key1 == key2:
                key_similarity = 1.0
            else:
                # Penalizar diferencia de key
                key_similarity = 0.5
            total_similarity += key_similarity
            count += 1
        
        # Comparar mode
        mode1 = features1.get("mode", 0)
        mode2 = features2.get("mode", 0)
        mode_similarity = 1.0 if mode1 == mode2 else 0.7
        total_similarity += mode_similarity
        count += 1
        
        return total_similarity / count if count > 0 else 0.0
    
    def recommend_by_mood(self, target_mood: str,
                          all_tracks: List[Dict[str, Any]],
                          limit: int = 10) -> List[Dict[str, Any]]:
        """Recomienda tracks basado en mood/emoción"""
        # Filtrar tracks que coincidan con el mood
        mood_tracks = []
        
        for track in all_tracks:
            track_features = track.get("audio_features", {})
            if not track_features:
                continue
            
            # Analizar emoción del track
            from .emotion_analyzer import EmotionAnalyzer
            emotion_analyzer = EmotionAnalyzer()
            emotions = emotion_analyzer.analyze_emotions(track_features)
            
            # Verificar si coincide con el mood objetivo
            if emotions["primary_emotion"] == target_mood:
                mood_tracks.append({
                    **track,
                    "emotion_confidence": emotions["confidence"]
                })
        
        # Ordenar por confianza de emoción
        mood_tracks.sort(key=lambda x: x.get("emotion_confidence", 0), reverse=True)
        
        return mood_tracks[:limit]
    
    def recommend_by_genre(self, target_genre: str,
                           all_tracks: List[Dict[str, Any]],
                           limit: int = 10) -> List[Dict[str, Any]]:
        """Recomienda tracks basado en género"""
        genre_tracks = []
        
        for track in all_tracks:
            track_features = track.get("audio_features", {})
            if not track_features:
                continue
            
            # Analizar género del track
            from .genre_detector import GenreDetector
            genre_detector = GenreDetector()
            genre_analysis = genre_detector.detect_genre(track_features)
            
            # Verificar si coincide con el género objetivo
            if genre_analysis["primary_genre"] == target_genre:
                genre_tracks.append({
                    **track,
                    "genre_confidence": genre_analysis["confidence"]
                })
        
        # Ordenar por confianza de género
        genre_tracks.sort(key=lambda x: x.get("genre_confidence", 0), reverse=True)
        
        return genre_tracks[:limit]
    
    def recommend_playlist(self, user_preferences: Dict[str, Any],
                           available_tracks: List[Dict[str, Any]],
                           playlist_length: int = 20) -> List[Dict[str, Any]]:
        """Genera una playlist recomendada basada en preferencias del usuario"""
        # Preferencias del usuario
        preferred_genres = user_preferences.get("genres", [])
        preferred_moods = user_preferences.get("moods", [])
        preferred_energy = user_preferences.get("energy_range", (0.0, 1.0))
        preferred_tempo = user_preferences.get("tempo_range", (60, 180))
        
        scored_tracks = []
        
        for track in available_tracks:
            track_features = track.get("audio_features", {})
            if not track_features:
                continue
            
            score = 0.0
            
            # Score por género
            if preferred_genres:
                from .genre_detector import GenreDetector
                genre_detector = GenreDetector()
                genre_analysis = genre_detector.detect_genre(track_features)
                if genre_analysis["primary_genre"] in preferred_genres:
                    score += 2.0 * genre_analysis["confidence"]
            
            # Score por mood
            if preferred_moods:
                from .emotion_analyzer import EmotionAnalyzer
                emotion_analyzer = EmotionAnalyzer()
                emotions = emotion_analyzer.analyze_emotions(track_features)
                if emotions["primary_emotion"] in preferred_moods:
                    score += 1.5 * emotions["confidence"]
            
            # Score por energía
            energy = track_features.get("energy", 0.5)
            min_e, max_e = preferred_energy
            if min_e <= energy <= max_e:
                score += 1.0
            else:
                distance = min(abs(energy - min_e), abs(energy - max_e))
                score += max(0, 1.0 - distance)
            
            # Score por tempo
            tempo = track_features.get("tempo", 120)
            min_t, max_t = preferred_tempo
            if min_t <= tempo <= max_t:
                score += 0.5
            else:
                distance = min(abs(tempo - min_t), abs(tempo - max_t))
                score += max(0, 0.5 - (distance / 50))
            
            scored_tracks.append({
                **track,
                "recommendation_score": score
            })
        
        # Ordenar por score
        scored_tracks.sort(key=lambda x: x.get("recommendation_score", 0), reverse=True)
        
        return scored_tracks[:playlist_length]

