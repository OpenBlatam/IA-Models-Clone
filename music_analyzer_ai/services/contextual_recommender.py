"""
Servicio de recomendaciones contextuales inteligentes
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from .spotify_service import SpotifyService
from .intelligent_recommender import IntelligentRecommender
from .genre_detector import GenreDetector
from .emotion_analyzer import EmotionAnalyzer

logger = logging.getLogger(__name__)


class ContextualRecommender:
    """Recomendaciones basadas en contexto"""
    
    def __init__(self):
        self.spotify = SpotifyService()
        self.intelligent_recommender = IntelligentRecommender()
        self.genre_detector = GenreDetector()
        self.emotion_analyzer = EmotionAnalyzer()
        self.logger = logger
    
    def recommend_by_context(
        self,
        track_id: str,
        context: Dict[str, Any],
        limit: int = 10
    ) -> Dict[str, Any]:
        """Recomienda tracks basado en contexto"""
        try:
            # Obtener información del track
            track_info = self.spotify.get_track(track_id)
            audio_features = self.spotify.get_track_audio_features(track_id)
            
            # Analizar track
            genre = self.genre_detector.detect_genre(audio_features)
            emotion = self.emotion_analyzer.analyze_emotions(audio_features)
            
            # Aplicar filtros contextuales
            recommendations = []
            
            # Obtener recomendaciones base
            base_recommendations = self.spotify.get_recommendations(track_id, limit=limit * 2)
            
            # Filtrar y rankear por contexto
            for rec in base_recommendations:
                try:
                    rec_features = self.spotify.get_track_audio_features(rec.get("id"))
                    if not rec_features:
                        continue
                    
                    score = self._calculate_contextual_score(
                        audio_features, rec_features,
                        genre, emotion, context
                    )
                    
                    recommendations.append({
                        "track_id": rec.get("id"),
                        "track_name": rec.get("name"),
                        "artists": [a.get("name") for a in rec.get("artists", [])],
                        "contextual_score": round(score, 3),
                        "preview_url": rec.get("preview_url")
                    })
                except:
                    continue
            
            # Ordenar por score contextual
            recommendations.sort(key=lambda x: x["contextual_score"], reverse=True)
            
            return {
                "track_id": track_id,
                "context": context,
                "recommendations": recommendations[:limit],
                "total_found": len(recommendations)
            }
        except Exception as e:
            self.logger.error(f"Error in contextual recommendation: {e}")
            return {"error": str(e)}
    
    def recommend_by_time_of_day(self, track_id: str, time_of_day: str, limit: int = 10) -> Dict[str, Any]:
        """Recomienda basado en hora del día"""
        time_contexts = {
            "morning": {"energy_range": [0.4, 0.7], "valence_range": [0.5, 0.8], "tempo_range": [80, 120]},
            "afternoon": {"energy_range": [0.5, 0.8], "valence_range": [0.5, 0.9], "tempo_range": [100, 140]},
            "evening": {"energy_range": [0.6, 0.9], "valence_range": [0.4, 0.8], "tempo_range": [110, 160]},
            "night": {"energy_range": [0.3, 0.6], "valence_range": [0.3, 0.7], "tempo_range": [60, 100]}
        }
        
        context = time_contexts.get(time_of_day.lower(), time_contexts["afternoon"])
        return self.recommend_by_context(track_id, context, limit)
    
    def recommend_by_activity(self, track_id: str, activity: str, limit: int = 10) -> Dict[str, Any]:
        """Recomienda basado en actividad"""
        activity_contexts = {
            "workout": {"energy_range": [0.7, 1.0], "tempo_range": [120, 180], "danceability_range": [0.6, 1.0]},
            "study": {"energy_range": [0.2, 0.5], "tempo_range": [60, 100], "instrumentalness_range": [0.5, 1.0]},
            "party": {"energy_range": [0.7, 1.0], "danceability_range": [0.7, 1.0], "valence_range": [0.6, 1.0]},
            "relax": {"energy_range": [0.2, 0.5], "tempo_range": [60, 90], "valence_range": [0.4, 0.7]},
            "drive": {"energy_range": [0.5, 0.8], "tempo_range": [100, 140], "valence_range": [0.5, 0.8]}
        }
        
        context = activity_contexts.get(activity.lower(), activity_contexts["relax"])
        return self.recommend_by_context(track_id, context, limit)
    
    def recommend_by_mood(self, track_id: str, target_mood: str, limit: int = 10) -> Dict[str, Any]:
        """Recomienda basado en mood objetivo"""
        mood_contexts = {
            "happy": {"valence_range": [0.7, 1.0], "energy_range": [0.6, 1.0]},
            "sad": {"valence_range": [0.0, 0.4], "energy_range": [0.2, 0.6]},
            "energetic": {"energy_range": [0.7, 1.0], "tempo_range": [120, 180]},
            "calm": {"energy_range": [0.2, 0.5], "tempo_range": [60, 100]},
            "romantic": {"valence_range": [0.5, 0.8], "energy_range": [0.3, 0.6], "tempo_range": [70, 110]}
        }
        
        context = mood_contexts.get(target_mood.lower(), mood_contexts["calm"])
        return self.recommend_by_context(track_id, context, limit)
    
    def _calculate_contextual_score(
        self,
        source_features: Dict,
        target_features: Dict,
        source_genre: Dict,
        source_emotion: Dict,
        context: Dict
    ) -> float:
        """Calcula score contextual"""
        # Score base de similitud
        base_score = self.intelligent_recommender._calculate_similarity(
            source_features, target_features
        )
        
        # Ajustes contextuales
        context_score = 1.0
        
        # Filtro de energía
        if "energy_range" in context:
            energy = target_features.get("energy", 0.5)
            min_e, max_e = context["energy_range"]
            if min_e <= energy <= max_e:
                context_score *= 1.2
            else:
                context_score *= 0.8
        
        # Filtro de tempo
        if "tempo_range" in context:
            tempo = target_features.get("tempo", 120)
            min_t, max_t = context["tempo_range"]
            if min_t <= tempo <= max_t:
                context_score *= 1.2
            else:
                context_score *= 0.8
        
        # Filtro de valence
        if "valence_range" in context:
            valence = target_features.get("valence", 0.5)
            min_v, max_v = context["valence_range"]
            if min_v <= valence <= max_v:
                context_score *= 1.2
            else:
                context_score *= 0.8
        
        # Filtro de danceability
        if "danceability_range" in context:
            danceability = target_features.get("danceability", 0.5)
            min_d, max_d = context["danceability_range"]
            if min_d <= danceability <= max_d:
                context_score *= 1.1
            else:
                context_score *= 0.9
        
        # Filtro de instrumentalness
        if "instrumentalness_range" in context:
            instrumentalness = target_features.get("instrumentalness", 0.5)
            min_i, max_i = context["instrumentalness_range"]
            if min_i <= instrumentalness <= max_i:
                context_score *= 1.1
            else:
                context_score *= 0.9
        
        return base_score * context_score

