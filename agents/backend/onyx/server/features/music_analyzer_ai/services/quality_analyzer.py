"""
Servicio de análisis de calidad musical
"""

import logging
from typing import Dict, List, Any, Optional

from .spotify_service import SpotifyService

logger = logging.getLogger(__name__)


class QualityAnalyzer:
    """Analiza la calidad de producción musical"""
    
    def __init__(self):
        self.spotify = SpotifyService()
        self.logger = logger
    
    def analyze_production_quality(self, track_id: str) -> Dict[str, Any]:
        """Analiza la calidad de producción de un track"""
        try:
            track_info = self.spotify.get_track(track_id)
            audio_features = self.spotify.get_track_audio_features(track_id)
            audio_analysis = self.spotify.get_track_audio_analysis(track_id)
            
            # Factores de calidad
            factors = {
                "audio_quality": self._analyze_audio_quality(audio_features, audio_analysis),
                "production_technique": self._analyze_production_technique(audio_features, audio_analysis),
                "musical_quality": self._analyze_musical_quality(audio_features, track_info),
                "technical_quality": self._analyze_technical_quality(audio_features, audio_analysis)
            }
            
            # Score general de calidad
            quality_score = (
                factors["audio_quality"]["score"] * 0.3 +
                factors["production_technique"]["score"] * 0.3 +
                factors["musical_quality"]["score"] * 0.2 +
                factors["technical_quality"]["score"] * 0.2
            )
            
            # Categorizar calidad
            if quality_score >= 0.8:
                quality_level = "Excellent"
            elif quality_score >= 0.6:
                quality_level = "Good"
            elif quality_score >= 0.4:
                quality_level = "Average"
            elif quality_score >= 0.2:
                quality_level = "Below Average"
            else:
                quality_level = "Poor"
            
            return {
                "track_id": track_id,
                "overall_quality": {
                    "score": round(quality_score, 3),
                    "level": quality_level
                },
                "factors": factors,
                "recommendations": self._get_quality_recommendations(factors, quality_score)
            }
        except Exception as e:
            self.logger.error(f"Error analyzing production quality: {e}")
            return {"error": str(e)}
    
    def _analyze_audio_quality(self, audio_features: Dict, audio_analysis: Dict) -> Dict[str, Any]:
        """Analiza la calidad de audio"""
        # Factores: claridad, rango dinámico, etc.
        acousticness = audio_features.get("acousticness", 0.5)
        instrumentalness = audio_features.get("instrumentalness", 0.5)
        
        # Score basado en características
        score = (
            (1.0 - abs(acousticness - 0.5)) * 0.5 +  # Balance acústico
            (1.0 - abs(instrumentalness - 0.3)) * 0.5  # Balance instrumental
        )
        
        return {
            "score": round(score, 3),
            "acousticness": acousticness,
            "instrumentalness": instrumentalness,
            "assessment": "High" if score > 0.7 else "Medium" if score > 0.4 else "Low"
        }
    
    def _analyze_production_technique(self, audio_features: Dict, audio_analysis: Dict) -> Dict[str, Any]:
        """Analiza la técnica de producción"""
        # Factores: mezcla, masterización, etc.
        sections = audio_analysis.get("sections", [])
        
        if not sections:
            return {"score": 0.5, "assessment": "Insufficient Data"}
        
        # Analizar consistencia de loudness
        loudnesses = [s.get("loudness", -10) for s in sections]
        avg_loudness = sum(loudnesses) / len(loudnesses) if loudnesses else -10
        variance = sum((l - avg_loudness) ** 2 for l in loudnesses) / len(loudnesses) if loudnesses else 0
        
        # Score: menos variación = mejor masterización
        consistency_score = 1.0 / (1.0 + variance)
        
        # Analizar rango dinámico
        dynamic_range = max(loudnesses) - min(loudnesses) if loudnesses else 0
        dynamic_score = min(dynamic_range / 20, 1.0)  # Normalizar
        
        score = (consistency_score * 0.6 + dynamic_score * 0.4)
        
        return {
            "score": round(score, 3),
            "loudness_consistency": round(consistency_score, 3),
            "dynamic_range": round(dynamic_range, 2),
            "assessment": "Professional" if score > 0.7 else "Good" if score > 0.5 else "Needs Improvement"
        }
    
    def _analyze_musical_quality(self, audio_features: Dict, track_info: Dict) -> Dict[str, Any]:
        """Analiza la calidad musical"""
        # Factores: complejidad, originalidad, etc.
        danceability = audio_features.get("danceability", 0.5)
        energy = audio_features.get("energy", 0.5)
        valence = audio_features.get("valence", 0.5)
        tempo = audio_features.get("tempo", 120)
        
        # Score basado en balance de características
        balance_score = (
            (1.0 - abs(danceability - 0.6)) * 0.3 +
            (1.0 - abs(energy - 0.6)) * 0.3 +
            (1.0 - abs(valence - 0.5)) * 0.2 +
            (1.0 - abs((tempo - 120) / 120)) * 0.2
        )
        
        # Ajustar por popularidad (proxy de calidad percibida)
        popularity = track_info.get("popularity", 50) / 100
        
        score = (balance_score * 0.7 + popularity * 0.3)
        
        return {
            "score": round(score, 3),
            "balance_score": round(balance_score, 3),
            "popularity_factor": round(popularity, 3),
            "assessment": "High" if score > 0.7 else "Medium" if score > 0.4 else "Low"
        }
    
    def _analyze_technical_quality(self, audio_features: Dict, audio_analysis: Dict) -> Dict[str, Any]:
        """Analiza la calidad técnica"""
        # Factores: precisión, timing, etc.
        sections = audio_analysis.get("sections", [])
        
        if not sections:
            return {"score": 0.5, "assessment": "Insufficient Data"}
        
        # Analizar precisión de tempo
        tempos = [s.get("tempo", 120) for s in sections if s.get("tempo")]
        
        if tempos:
            avg_tempo = sum(tempos) / len(tempos)
            tempo_variance = sum((t - avg_tempo) ** 2 for t in tempos) / len(tempos)
            tempo_precision = 1.0 / (1.0 + tempo_variance / 100)
        else:
            tempo_precision = 0.5
        
        # Analizar key consistency
        keys = [s.get("key", -1) for s in sections if s.get("key", -1) >= 0]
        key_consistency = len(set(keys)) / len(keys) if keys else 0.5
        key_score = 1.0 - key_consistency  # Menos cambios = mejor
        
        score = (tempo_precision * 0.6 + key_score * 0.4)
        
        return {
            "score": round(score, 3),
            "tempo_precision": round(tempo_precision, 3),
            "key_consistency": round(key_consistency, 3),
            "assessment": "Precise" if score > 0.7 else "Good" if score > 0.5 else "Needs Improvement"
        }
    
    def _get_quality_recommendations(self, factors: Dict, overall_score: float) -> List[str]:
        """Genera recomendaciones para mejorar calidad"""
        recommendations = []
        
        if overall_score < 0.5:
            recommendations.append("Considerar revisión profesional de masterización")
            recommendations.append("Mejorar balance de mezcla")
        
        if factors["production_technique"]["score"] < 0.5:
            recommendations.append("Trabajar en consistencia de loudness")
            recommendations.append("Mejorar rango dinámico")
        
        if factors["technical_quality"]["score"] < 0.5:
            recommendations.append("Mejorar precisión de tempo")
            recommendations.append("Revisar consistencia de tonalidad")
        
        if factors["musical_quality"]["score"] < 0.5:
            recommendations.append("Revisar balance de características musicales")
            recommendations.append("Considerar feedback de audiencia")
        
        if not recommendations:
            recommendations.append("Calidad general buena - mantener estándares")
        
        return recommendations

