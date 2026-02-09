"""
Servicio de predicción de éxito mejorado
"""

import logging
from typing import Dict, List, Any, Optional

from .spotify_service import SpotifyService
from .genre_detector import GenreDetector
from .emotion_analyzer import EmotionAnalyzer
from .trends_analyzer import TrendsAnalyzer

logger = logging.getLogger(__name__)


class SuccessPredictor:
    """Predicción de éxito comercial mejorada"""
    
    def __init__(self):
        self.spotify = SpotifyService()
        self.genre_detector = GenreDetector()
        self.emotion_analyzer = EmotionAnalyzer()
        self.trends_analyzer = TrendsAnalyzer()
        self.logger = logger
    
    def predict_success(self, track_id: str) -> Dict[str, Any]:
        """Predice el éxito comercial de un track"""
        try:
            track_info = self.spotify.get_track(track_id)
            audio_features = self.spotify.get_track_audio_features(track_id)
            
            if not track_info or not audio_features:
                return {"error": "No hay datos disponibles"}
            
            popularity = track_info.get("popularity", 0)
            
            # Factores de éxito
            factors = {
                "commercial_appeal": self._calculate_commercial_appeal(audio_features),
                "trend_alignment": self._calculate_trend_alignment(audio_features),
                "genre_potential": self._calculate_genre_potential(audio_features),
                "emotional_appeal": self._calculate_emotional_appeal(audio_features),
                "current_popularity": popularity / 100.0
            }
            
            # Score de éxito
            success_score = (
                factors["commercial_appeal"] * 0.3 +
                factors["trend_alignment"] * 0.25 +
                factors["genre_potential"] * 0.2 +
                factors["emotional_appeal"] * 0.15 +
                factors["current_popularity"] * 0.1
            )
            
            # Predicción
            if success_score > 0.75:
                prediction = "Very High Success Potential"
                confidence = "High"
            elif success_score > 0.6:
                prediction = "High Success Potential"
                confidence = "Medium-High"
            elif success_score > 0.45:
                prediction = "Moderate Success Potential"
                confidence = "Medium"
            elif success_score > 0.3:
                prediction = "Low Success Potential"
                confidence = "Medium-Low"
            else:
                prediction = "Very Low Success Potential"
                confidence = "Low"
            
            # Recomendaciones
            recommendations = self._generate_recommendations(factors, success_score)
            
            return {
                "track_id": track_id,
                "track_name": track_info.get("name", "Unknown"),
                "success_score": round(success_score, 3),
                "prediction": prediction,
                "confidence": confidence,
                "factors": {k: round(v, 3) for k, v in factors.items()},
                "recommendations": recommendations
            }
        except Exception as e:
            self.logger.error(f"Error predicting success: {e}")
            return {"error": str(e)}
    
    def _calculate_commercial_appeal(self, audio_features: Dict) -> float:
        """Calcula atractivo comercial"""
        energy = audio_features.get("energy", 0.5)
        danceability = audio_features.get("danceability", 0.5)
        valence = audio_features.get("valence", 0.5)
        tempo = audio_features.get("tempo", 120)
        
        # Factores comerciales
        commercial_score = (
            (energy * 0.3) +
            (danceability * 0.3) +
            (valence * 0.2) +
            (1.0 if 100 <= tempo <= 140 else 0.5) * 0.2
        )
        
        return commercial_score
    
    def _calculate_trend_alignment(self, audio_features: Dict) -> float:
        """Calcula alineación con tendencias"""
        # Simplificado: basado en características actuales
        energy = audio_features.get("energy", 0.5)
        danceability = audio_features.get("danceability", 0.5)
        
        # Tendencias actuales (simplificado)
        # Alta energía y bailabilidad = más alineado
        trend_score = (energy * 0.5 + danceability * 0.5)
        
        return trend_score
    
    def _calculate_genre_potential(self, audio_features: Dict) -> float:
        """Calcula potencial del género"""
        genre_analysis = self.genre_detector.detect_genre(audio_features)
        confidence = genre_analysis.get("confidence", 0.5)
        
        # Géneros con alto potencial comercial
        commercial_genres = ["Pop", "Rock", "Electronic", "Hip-Hop", "R&B"]
        primary_genre = genre_analysis.get("primary_genre", "")
        
        genre_potential = confidence
        if primary_genre in commercial_genres:
            genre_potential += 0.2
        
        return min(1.0, genre_potential)
    
    def _calculate_emotional_appeal(self, audio_features: Dict) -> float:
        """Calcula atractivo emocional"""
        emotion_analysis = self.emotion_analyzer.analyze_emotions(audio_features)
        primary_emotion = emotion_analysis.get("primary_emotion", "Unknown")
        confidence = emotion_analysis.get("confidence", 0.5)
        
        # Emociones con alto atractivo comercial
        commercial_emotions = ["happy", "energetic", "romantic"]
        
        emotional_appeal = confidence
        if primary_emotion in commercial_emotions:
            emotional_appeal += 0.2
        
        return min(1.0, emotional_appeal)
    
    def _generate_recommendations(self, factors: Dict, success_score: float) -> List[str]:
        """Genera recomendaciones"""
        recommendations = []
        
        if factors["commercial_appeal"] < 0.6:
            recommendations.append("Consider increasing energy and danceability for broader appeal")
        
        if factors["trend_alignment"] < 0.5:
            recommendations.append("Consider aligning more with current musical trends")
        
        if factors["genre_potential"] < 0.5:
            recommendations.append("Consider genre positioning for better commercial potential")
        
        if success_score < 0.5:
            recommendations.append("Focus on marketing and promotion to increase visibility")
        
        if factors["current_popularity"] < 0.3 and success_score > 0.6:
            recommendations.append("High potential track - invest in promotion and marketing")
        
        return recommendations if recommendations else ["Track shows good commercial potential"]

