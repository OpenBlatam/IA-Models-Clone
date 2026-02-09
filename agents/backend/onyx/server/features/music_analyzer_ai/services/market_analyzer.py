"""
Servicio de análisis de mercado musical
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import Counter

from .spotify_service import SpotifyService
from .trends_analyzer import TrendsAnalyzer

logger = logging.getLogger(__name__)


class MarketAnalyzer:
    """Analiza el mercado musical y competencia"""
    
    def __init__(self):
        self.spotify = SpotifyService()
        self.trends_analyzer = TrendsAnalyzer()
        self.logger = logger
    
    def analyze_market_position(self, track_id: str) -> Dict[str, Any]:
        """Analiza la posición de mercado de un track"""
        try:
            track_info = self.spotify.get_track(track_id)
            audio_features = self.spotify.get_track_audio_features(track_id)
            
            if not track_info or not audio_features:
                return {"error": "No hay datos disponibles"}
            
            popularity = track_info.get("popularity", 0)
            
            # Análisis de posición
            market_position = {
                "popularity_score": popularity,
                "market_tier": self._determine_market_tier(popularity),
                "competitive_analysis": self._analyze_competitiveness(audio_features, popularity),
                "market_potential": self._calculate_market_potential(audio_features, popularity),
                "recommendations": self._generate_market_recommendations(audio_features, popularity)
            }
            
            return {
                "track_id": track_id,
                "track_name": track_info.get("name", "Unknown"),
                "market_position": market_position
            }
        except Exception as e:
            self.logger.error(f"Error analyzing market position: {e}")
            return {"error": str(e)}
    
    def analyze_competitor_landscape(self, genre: str, limit: int = 20) -> Dict[str, Any]:
        """Analiza el panorama competitivo de un género"""
        try:
            # Buscar tracks del género
            tracks = self.spotify.search_tracks(f"genre:{genre}", limit=limit)
            
            if not tracks:
                return {"error": f"No se encontraron tracks del género {genre}"}
            
            # Analizar características
            popularities = []
            energies = []
            danceabilities = []
            valences = []
            
            for track in tracks:
                popularities.append(track.get("popularity", 0))
                audio_features = self.spotify.get_track_audio_features(track.get("id"))
                if audio_features:
                    energies.append(audio_features.get("energy", 0.5))
                    danceabilities.append(audio_features.get("danceability", 0.5))
                    valences.append(audio_features.get("valence", 0.5))
            
            return {
                "genre": genre,
                "tracks_analyzed": len(tracks),
                "market_statistics": {
                    "average_popularity": round(sum(popularities) / len(popularities), 2) if popularities else 0,
                    "max_popularity": max(popularities) if popularities else 0,
                    "min_popularity": min(popularities) if popularities else 0,
                    "average_energy": round(sum(energies) / len(energies), 3) if energies else 0,
                    "average_danceability": round(sum(danceabilities) / len(danceabilities), 3) if danceabilities else 0,
                    "average_valence": round(sum(valences) / len(valences), 3) if valences else 0
                },
                "market_saturation": self._calculate_market_saturation(popularities),
                "opportunities": self._identify_market_opportunities(energies, danceabilities, valences)
            }
        except Exception as e:
            self.logger.error(f"Error analyzing competitor landscape: {e}")
            return {"error": str(e)}
    
    def _determine_market_tier(self, popularity: int) -> str:
        """Determina el tier de mercado basado en popularidad"""
        if popularity >= 80:
            return "Top Tier"
        elif popularity >= 60:
            return "High Tier"
        elif popularity >= 40:
            return "Mid Tier"
        elif popularity >= 20:
            return "Low Tier"
        else:
            return "Underground"
    
    def _analyze_competitiveness(self, audio_features: Dict, popularity: int) -> Dict[str, Any]:
        """Analiza la competitividad del track"""
        energy = audio_features.get("energy", 0.5)
        danceability = audio_features.get("danceability", 0.5)
        valence = audio_features.get("valence", 0.5)
        
        # Score de competitividad
        commercial_score = (energy * 0.3 + danceability * 0.3 + valence * 0.2 + (popularity / 100) * 0.2)
        
        if commercial_score > 0.7:
            competitiveness = "High"
        elif commercial_score > 0.5:
            competitiveness = "Medium"
        else:
            competitiveness = "Low"
        
        return {
            "commercial_score": round(commercial_score, 3),
            "competitiveness": competitiveness,
            "factors": {
                "energy": round(energy, 3),
                "danceability": round(danceability, 3),
                "valence": round(valence, 3),
                "popularity": popularity
            }
        }
    
    def _calculate_market_potential(self, audio_features: Dict, popularity: int) -> Dict[str, Any]:
        """Calcula el potencial de mercado"""
        energy = audio_features.get("energy", 0.5)
        danceability = audio_features.get("danceability", 0.5)
        tempo = audio_features.get("tempo", 120)
        
        # Potencial basado en características comerciales
        potential_score = (
            (energy * 0.3) +
            (danceability * 0.3) +
            (1.0 if 100 <= tempo <= 140 else 0.5) * 0.2 +
            (popularity / 100 * 0.2)
        )
        
        if potential_score > 0.75:
            potential = "Very High"
        elif potential_score > 0.6:
            potential = "High"
        elif potential_score > 0.4:
            potential = "Medium"
        else:
            potential = "Low"
        
        return {
            "potential_score": round(potential_score, 3),
            "potential": potential,
            "growth_opportunity": "High" if popularity < 50 and potential_score > 0.6 else "Medium" if popularity < 70 else "Low"
        }
    
    def _generate_market_recommendations(self, audio_features: Dict, popularity: int) -> List[str]:
        """Genera recomendaciones de mercado"""
        recommendations = []
        
        energy = audio_features.get("energy", 0.5)
        danceability = audio_features.get("danceability", 0.5)
        tempo = audio_features.get("tempo", 120)
        
        if popularity < 50:
            recommendations.append("Consider increasing marketing efforts")
        
        if energy < 0.5:
            recommendations.append("Consider increasing energy for broader appeal")
        
        if danceability < 0.5:
            recommendations.append("Consider improving danceability for commercial success")
        
        if not (100 <= tempo <= 140):
            recommendations.append("Consider adjusting tempo to commercial range (100-140 BPM)")
        
        if popularity < 30 and energy > 0.7 and danceability > 0.7:
            recommendations.append("High potential track - focus on promotion")
        
        return recommendations if recommendations else ["Track is well-positioned in the market"]
    
    def _calculate_market_saturation(self, popularities: List[int]) -> Dict[str, Any]:
        """Calcula la saturación del mercado"""
        if not popularities:
            return {"level": "Unknown", "score": 0}
        
        # Calcular distribución de popularidad
        high_popularity = sum(1 for p in popularities if p >= 70)
        medium_popularity = sum(1 for p in popularities if 40 <= p < 70)
        low_popularity = sum(1 for p in popularities if p < 40)
        
        total = len(popularities)
        high_ratio = high_popularity / total if total > 0 else 0
        
        if high_ratio > 0.5:
            saturation = "High"
            score = 0.8
        elif high_ratio > 0.3:
            saturation = "Medium"
            score = 0.5
        else:
            saturation = "Low"
            score = 0.2
        
        return {
            "level": saturation,
            "score": round(score, 3),
            "distribution": {
                "high_popularity": high_popularity,
                "medium_popularity": medium_popularity,
                "low_popularity": low_popularity
            }
        }
    
    def _identify_market_opportunities(self, energies: List[float], danceabilities: List[float], valences: List[float]) -> List[str]:
        """Identifica oportunidades de mercado"""
        opportunities = []
        
        if energies:
            avg_energy = sum(energies) / len(energies)
            if avg_energy < 0.5:
                opportunities.append("Low energy market - opportunity for high-energy tracks")
            elif avg_energy > 0.8:
                opportunities.append("High energy market - opportunity for calmer tracks")
        
        if danceabilities:
            avg_danceability = sum(danceabilities) / len(danceabilities)
            if avg_danceability < 0.5:
                opportunities.append("Low danceability market - opportunity for dance tracks")
        
        if valences:
            avg_valence = sum(valences) / len(valences)
            if avg_valence < 0.4:
                opportunities.append("Negative sentiment market - opportunity for positive tracks")
            elif avg_valence > 0.7:
                opportunities.append("Positive sentiment market - opportunity for emotional depth")
        
        return opportunities if opportunities else ["Market is balanced"]

