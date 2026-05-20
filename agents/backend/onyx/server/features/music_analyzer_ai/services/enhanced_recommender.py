"""
Servicio de recomendaciones mejorado con ML avanzado
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from .spotify_service import SpotifyService
from .intelligent_recommender import IntelligentRecommender
from .emotion_analyzer import EmotionAnalyzer
from .genre_detector import GenreDetector

logger = logging.getLogger(__name__)


class EnhancedRecommender:
    """Sistema de recomendaciones mejorado"""
    
    def __init__(self):
        self.spotify = SpotifyService()
        self.intelligent_recommender = IntelligentRecommender()
        self.emotion_analyzer = EmotionAnalyzer()
        self.genre_detector = GenreDetector()
        self.logger = logger
    
    def get_enhanced_recommendations(self, track_id: str, limit: int = 20, 
                                    include_factors: bool = True) -> Dict[str, Any]:
        """Obtiene recomendaciones mejoradas"""
        try:
            # Obtener datos del track
            track_info = self.spotify.get_track(track_id)
            audio_features = self.spotify.get_track_audio_features(track_id)
            
            if not track_info or not audio_features:
                return {"error": "No hay datos disponibles"}
            
            # Análisis del track
            genre_analysis = self.genre_detector.detect_genre(audio_features)
            emotion_analysis = self.emotion_analyzer.analyze_emotions(audio_features)
            
            # Obtener recomendaciones de Spotify
            spotify_recs = self.spotify.get_recommendations(
                seed_tracks=[track_id],
                limit=limit
            )
            
            if not spotify_recs:
                return {"error": "No se encontraron recomendaciones"}
            
            # Mejorar recomendaciones con análisis
            enhanced_recs = []
            for rec in spotify_recs:
                rec_id = rec.get("id")
                rec_features = self.spotify.get_track_audio_features(rec_id)
                
                if rec_features:
                    # Calcular similitud mejorada
                    similarity = self._calculate_enhanced_similarity(
                        audio_features, rec_features, genre_analysis, emotion_analysis
                    )
                    
                    rec_data = {
                        "track_id": rec_id,
                        "name": rec.get("name", "Unknown"),
                        "artists": [a.get("name") for a in rec.get("artists", [])],
                        "popularity": rec.get("popularity", 0),
                        "similarity_score": round(similarity["total_score"], 3),
                        "preview_url": rec.get("preview_url")
                    }
                    
                    if include_factors:
                        rec_data["similarity_factors"] = similarity["factors"]
                    
                    enhanced_recs.append(rec_data)
            
            # Ordenar por similitud
            enhanced_recs.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            return {
                "seed_track_id": track_id,
                "seed_track_name": track_info.get("name", "Unknown"),
                "recommendations": enhanced_recs[:limit],
                "total_recommendations": len(enhanced_recs),
                "analysis": {
                    "seed_genre": genre_analysis.get("primary_genre", "Unknown"),
                    "seed_emotion": emotion_analysis.get("primary_emotion", "Unknown")
                }
            }
        except Exception as e:
            self.logger.error(f"Error getting enhanced recommendations: {e}")
            return {"error": str(e)}
    
    def _calculate_enhanced_similarity(self, track1_features: Dict, track2_features: Dict,
                                       genre_analysis: Dict, emotion_analysis: Dict) -> Dict[str, Any]:
        """Calcula similitud mejorada"""
        factors = {}
        total_score = 0
        
        # Similitud de características básicas (40%)
        basic_similarity = self._calculate_basic_similarity(track1_features, track2_features)
        factors["audio_features"] = round(basic_similarity, 3)
        total_score += basic_similarity * 0.4
        
        # Similitud de género (30%)
        genre2 = self.genre_detector.detect_genre(track2_features)
        genre_similarity = 1.0 if genre2.get("primary_genre") == genre_analysis.get("primary_genre") else 0.5
        factors["genre"] = round(genre_similarity, 3)
        total_score += genre_similarity * 0.3
        
        # Similitud de emoción (30%)
        emotion2 = self.emotion_analyzer.analyze_emotions(track2_features)
        emotion_similarity = 1.0 if emotion2.get("primary_emotion") == emotion_analysis.get("primary_emotion") else 0.5
        factors["emotion"] = round(emotion_similarity, 3)
        total_score += emotion_similarity * 0.3
        
        return {
            "total_score": round(total_score, 3),
            "factors": factors
        }
    
    def _calculate_basic_similarity(self, features1: Dict, features2: Dict) -> float:
        """Calcula similitud básica de características"""
        keys = ["energy", "danceability", "valence", "acousticness", "instrumentalness"]
        
        similarities = []
        for key in keys:
            val1 = features1.get(key, 0.5)
            val2 = features2.get(key, 0.5)
            similarity = 1.0 - abs(val1 - val2)
            similarities.append(similarity)
        
        return sum(similarities) / len(similarities) if similarities else 0.5
    
    def get_contextual_playlist(self, context: Dict[str, Any], length: int = 20) -> Dict[str, Any]:
        """Genera playlist contextual mejorada"""
        try:
            # Extraer parámetros del contexto
            genres = context.get("genres", [])
            emotions = context.get("emotions", [])
            energy_range = context.get("energy_range", [0.0, 1.0])
            tempo_range = context.get("tempo_range", [60, 200])
            
            # Buscar tracks que coincidan
            tracks = []
            search_queries = []
            
            if genres:
                for genre in genres[:3]:
                    search_queries.append(f"genre:{genre}")
            
            # Buscar tracks
            for query in search_queries[:5]:
                results = self.spotify.search_tracks(query, limit=50)
                tracks.extend(results)
            
            # Filtrar y rankear
            filtered_tracks = []
            for track in tracks:
                features = self.spotify.get_track_audio_features(track.get("id"))
                if features:
                    energy = features.get("energy", 0.5)
                    tempo = features.get("tempo", 120)
                    
                    if energy_range[0] <= energy <= energy_range[1] and tempo_range[0] <= tempo <= tempo_range[1]:
                        # Analizar emoción
                        emotion = self.emotion_analyzer.analyze_emotions(features)
                        if not emotions or emotion.get("primary_emotion") in emotions:
                            filtered_tracks.append({
                                "track": track,
                                "features": features,
                                "emotion": emotion
                            })
            
            # Ordenar por popularidad
            filtered_tracks.sort(key=lambda x: x["track"].get("popularity", 0), reverse=True)
            
            playlist = []
            for item in filtered_tracks[:length]:
                playlist.append({
                    "track_id": item["track"].get("id"),
                    "name": item["track"].get("name", "Unknown"),
                    "artists": [a.get("name") for a in item["track"].get("artists", [])],
                    "popularity": item["track"].get("popularity", 0)
                })
            
            return {
                "context": context,
                "playlist": playlist,
                "playlist_length": len(playlist),
                "generated_at": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error generating contextual playlist: {e}")
            return {"error": str(e)}

