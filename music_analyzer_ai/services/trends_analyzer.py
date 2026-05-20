"""
Servicio de análisis de tendencias y popularidad musical
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import Counter

from .spotify_service import SpotifyService

logger = logging.getLogger(__name__)


class TrendsAnalyzer:
    """Analiza tendencias y popularidad en música"""
    
    def __init__(self):
        self.spotify = SpotifyService()
        self.logger = logger
    
    def analyze_popularity_trends(self, track_id: str, historical_data: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Analiza tendencias de popularidad de un track"""
        try:
            track_info = self.spotify.get_track(track_id)
            popularity = track_info.get("popularity", 0)
            
            # Categorizar popularidad
            if popularity >= 80:
                category = "Very Popular"
                trend = "High"
            elif popularity >= 60:
                category = "Popular"
                trend = "Medium-High"
            elif popularity >= 40:
                category = "Moderate"
                trend = "Medium"
            elif popularity >= 20:
                category = "Low"
                trend = "Low"
            else:
                category = "Very Low"
                trend = "Very Low"
            
            # Análisis de potencial
            potential = self._calculate_potential(popularity, track_info)
            
            return {
                "current_popularity": popularity,
                "category": category,
                "trend": trend,
                "potential": potential,
                "factors": {
                    "explicit": track_info.get("explicit", False),
                    "is_local": track_info.get("is_local", False),
                    "preview_available": track_info.get("preview_url") is not None
                },
                "recommendations": self._get_popularity_recommendations(popularity)
            }
        except Exception as e:
            self.logger.error(f"Error analyzing popularity trends: {e}")
            return {"error": str(e)}
    
    def analyze_artist_trends(self, artist_ids: List[str]) -> Dict[str, Any]:
        """Analiza tendencias de múltiples artistas"""
        try:
            artists_data = []
            
            for artist_id in artist_ids[:10]:  # Máximo 10 artistas
                try:
                    # Obtener información del artista (simulado, ya que no tenemos endpoint directo)
                    # Usaremos tracks del artista como proxy
                    tracks = self.spotify.search_tracks(f"artist:{artist_id}", limit=5)
                    
                    if tracks:
                        avg_popularity = sum(t.get("popularity", 0) for t in tracks) / len(tracks)
                        genres = []
                        for track in tracks:
                            # Extraer géneros de los artistas
                            for artist in track.get("artists", []):
                                if artist.get("id") == artist_id:
                                    # Géneros no disponibles directamente, usar análisis
                                    pass
                        
                        artists_data.append({
                            "artist_id": artist_id,
                            "average_popularity": round(avg_popularity, 2),
                            "tracks_analyzed": len(tracks)
                        })
                except:
                    continue
            
            if not artists_data:
                return {"error": "No se pudieron analizar artistas"}
            
            # Calcular estadísticas
            avg_popularity = sum(a["average_popularity"] for a in artists_data) / len(artists_data)
            most_popular = max(artists_data, key=lambda x: x["average_popularity"])
            
            return {
                "artists_analyzed": len(artists_data),
                "average_popularity": round(avg_popularity, 2),
                "most_popular": most_popular,
                "trend": "Rising" if avg_popularity > 50 else "Stable" if avg_popularity > 30 else "Declining",
                "artists": artists_data
            }
        except Exception as e:
            self.logger.error(f"Error analyzing artist trends: {e}")
            return {"error": str(e)}
    
    def predict_commercial_success(self, track_id: str) -> Dict[str, Any]:
        """Predice el éxito comercial de un track"""
        try:
            track_info = self.spotify.get_track(track_id)
            audio_features = self.spotify.get_track_audio_features(track_id)
            
            popularity = track_info.get("popularity", 0)
            
            # Factores de éxito
            factors = {
                "current_popularity": popularity,
                "danceability": audio_features.get("danceability", 0.5),
                "energy": audio_features.get("energy", 0.5),
                "valence": audio_features.get("valence", 0.5),
                "tempo": audio_features.get("tempo", 120),
                "has_preview": track_info.get("preview_url") is not None,
                "explicit": track_info.get("explicit", False)
            }
            
            # Calcular score de éxito
            success_score = (
                (popularity / 100) * 0.4 +
                factors["danceability"] * 0.2 +
                factors["energy"] * 0.15 +
                factors["valence"] * 0.15 +
                (1.0 if factors["has_preview"] else 0.5) * 0.1
            )
            
            # Ajustar por tempo (tempos comerciales típicos: 100-140 BPM)
            tempo = factors["tempo"]
            if 100 <= tempo <= 140:
                tempo_bonus = 0.1
            elif 80 <= tempo < 100 or 140 < tempo <= 160:
                tempo_bonus = 0.05
            else:
                tempo_bonus = 0
            
            success_score = min(success_score + tempo_bonus, 1.0)
            
            # Categorizar
            if success_score >= 0.8:
                prediction = "Very High Success Potential"
                confidence = "High"
            elif success_score >= 0.6:
                prediction = "High Success Potential"
                confidence = "Medium-High"
            elif success_score >= 0.4:
                prediction = "Moderate Success Potential"
                confidence = "Medium"
            elif success_score >= 0.2:
                prediction = "Low Success Potential"
                confidence = "Low"
            else:
                prediction = "Very Low Success Potential"
                confidence = "Very Low"
            
            return {
                "track_id": track_id,
                "success_score": round(success_score, 3),
                "prediction": prediction,
                "confidence": confidence,
                "factors": factors,
                "recommendations": self._get_success_recommendations(success_score, factors)
            }
        except Exception as e:
            self.logger.error(f"Error predicting commercial success: {e}")
            return {"error": str(e)}
    
    def analyze_rhythmic_patterns(self, audio_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza patrones rítmicos avanzados"""
        try:
            beats = audio_analysis.get("beats", [])
            sections = audio_analysis.get("sections", [])
            bars = audio_analysis.get("bars", [])
            
            if not beats:
                return {"error": "No hay datos de beats disponibles"}
            
            # Análisis de densidad rítmica
            total_duration = sections[-1].get("start", 0) + sections[-1].get("duration", 0) if sections else 0
            beat_count = len(beats)
            beat_density = beat_count / total_duration if total_duration > 0 else 0
            
            # Análisis de variación rítmica
            beat_intervals = []
            for i in range(1, len(beats)):
                interval = beats[i].get("start", 0) - beats[i-1].get("start", 0)
                beat_intervals.append(interval)
            
            if beat_intervals:
                avg_interval = sum(beat_intervals) / len(beat_intervals)
                interval_variance = sum((x - avg_interval) ** 2 for x in beat_intervals) / len(beat_intervals)
                rhythmic_consistency = 1.0 / (1.0 + interval_variance)  # Normalizar
            else:
                rhythmic_consistency = 0
            
            # Análisis de estructura rítmica
            bar_count = len(bars)
            beats_per_bar = beat_count / bar_count if bar_count > 0 else 0
            
            # Categorizar complejidad rítmica
            if rhythmic_consistency > 0.9 and beat_density < 2:
                complexity = "Simple"
            elif rhythmic_consistency > 0.7:
                complexity = "Moderate"
            else:
                complexity = "Complex"
            
            return {
                "beat_density": round(beat_density, 3),
                "beat_count": beat_count,
                "bar_count": bar_count,
                "beats_per_bar": round(beats_per_bar, 2),
                "rhythmic_consistency": round(rhythmic_consistency, 3),
                "complexity": complexity,
                "pattern": self._identify_rhythmic_pattern(beat_intervals)
            }
        except Exception as e:
            self.logger.error(f"Error analyzing rhythmic patterns: {e}")
            return {"error": str(e)}
    
    def _calculate_potential(self, popularity: int, track_info: Dict) -> Dict[str, Any]:
        """Calcula el potencial de crecimiento"""
        has_preview = track_info.get("preview_url") is not None
        is_explicit = track_info.get("explicit", False)
        
        potential_score = popularity / 100
        
        # Ajustes
        if has_preview:
            potential_score += 0.1
        
        if popularity < 50:
            growth_potential = "High" if potential_score < 0.4 else "Medium"
        elif popularity < 70:
            growth_potential = "Medium"
        else:
            growth_potential = "Low"
        
        return {
            "score": round(min(potential_score, 1.0), 3),
            "growth_potential": growth_potential,
            "factors": {
                "has_preview": has_preview,
                "explicit": is_explicit
            }
        }
    
    def _get_popularity_recommendations(self, popularity: int) -> List[str]:
        """Genera recomendaciones basadas en popularidad"""
        recommendations = []
        
        if popularity < 30:
            recommendations.append("Considerar promoción en redes sociales")
            recommendations.append("Mejorar metadata y tags")
            recommendations.append("Crear contenido visual atractivo")
        elif popularity < 50:
            recommendations.append("Colaborar con artistas más populares")
            recommendations.append("Incluir en playlists temáticas")
            recommendations.append("Optimizar para algoritmos de recomendación")
        elif popularity < 70:
            recommendations.append("Mantener engagement con audiencia")
            recommendations.append("Explorar nuevas plataformas")
            recommendations.append("Considerar remixes o versiones")
        else:
            recommendations.append("Mantener calidad y consistencia")
            recommendations.append("Explorar oportunidades de licenciamiento")
        
        return recommendations
    
    def _get_success_recommendations(self, success_score: float, factors: Dict) -> List[str]:
        """Genera recomendaciones para mejorar éxito comercial"""
        recommendations = []
        
        if factors["danceability"] < 0.5:
            recommendations.append("Aumentar bailabilidad para mayor atractivo comercial")
        
        if factors["energy"] < 0.5:
            recommendations.append("Aumentar energía para mayor impacto")
        
        if factors["valence"] < 0.4:
            recommendations.append("Considerar aumentar positividad (valence)")
        
        if not factors["has_preview"]:
            recommendations.append("Agregar preview para aumentar descubrimiento")
        
        tempo = factors["tempo"]
        if tempo < 80 or tempo > 160:
            recommendations.append("Considerar ajustar tempo a rango comercial (100-140 BPM)")
        
        return recommendations
    
    def _identify_rhythmic_pattern(self, intervals: List[float]) -> str:
        """Identifica patrones rítmicos comunes"""
        if not intervals:
            return "Unknown"
        
        avg_interval = sum(intervals) / len(intervals)
        
        # Detectar si es constante (patrón simple)
        variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)
        
        if variance < 0.01:
            return "Steady/Constant"
        elif variance < 0.1:
            return "Slightly Varied"
        else:
            return "Highly Varied"

