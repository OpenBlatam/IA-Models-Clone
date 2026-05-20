"""
Servicio de generación de reportes avanzados
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

from .spotify_service import SpotifyService
from .music_analyzer import MusicAnalyzer

logger = logging.getLogger(__name__)


class AdvancedReportGenerator:
    """Genera reportes avanzados de análisis musical"""
    
    def __init__(self):
        self.spotify = SpotifyService()
        self.music_analyzer = MusicAnalyzer()
        self.logger = logger
    
    def generate_comprehensive_report(self, track_id: str, include_all: bool = True) -> Dict[str, Any]:
        """Genera reporte comprehensivo"""
        try:
            track_info = self.spotify.get_track(track_id)
            spotify_data = self.spotify.get_track_full_analysis(track_id)
            analysis = self.music_analyzer.analyze_track(spotify_data)
            
            if not track_info or not analysis:
                return {"error": "No hay datos disponibles"}
            
            # Construir reporte
            report = {
                "report_metadata": {
                    "track_id": track_id,
                    "track_name": track_info.get("name", "Unknown"),
                    "artists": [a.get("name") for a in track_info.get("artists", [])],
                    "generated_at": datetime.now().isoformat(),
                    "report_version": "2.10.0"
                },
                "executive_summary": self._generate_executive_summary(analysis, track_info),
                "detailed_analysis": analysis if include_all else {},
                "insights": self._generate_insights(analysis),
                "recommendations": self._generate_recommendations(analysis, track_info)
            }
            
            return {
                "success": True,
                "report": report
            }
        except Exception as e:
            self.logger.error(f"Error generating comprehensive report: {e}")
            return {"error": str(e)}
    
    def _generate_executive_summary(self, analysis: Dict, track_info: Dict) -> Dict[str, Any]:
        """Genera resumen ejecutivo"""
        musical = analysis.get("musical_analysis", {})
        technical = analysis.get("technical_analysis", {})
        genre = analysis.get("genre_analysis", {})
        emotion = analysis.get("emotion_analysis", {})
        
        return {
            "key_signature": musical.get("key_signature", "Unknown"),
            "tempo": musical.get("tempo", {}).get("bpm", 0),
            "genre": genre.get("primary_genre", "Unknown"),
            "emotion": emotion.get("primary_emotion", "Unknown"),
            "energy": technical.get("energy", {}).get("value", 0),
            "danceability": technical.get("danceability", {}).get("value", 0),
            "popularity": track_info.get("popularity", 0),
            "duration_seconds": track_info.get("duration_ms", 0) / 1000
        }
    
    def _generate_insights(self, analysis: Dict) -> List[str]:
        """Genera insights del análisis"""
        insights = []
        
        musical = analysis.get("musical_analysis", {})
        technical = analysis.get("technical_analysis", {})
        genre = analysis.get("genre_analysis", {})
        emotion = analysis.get("emotion_analysis", {})
        
        # Insights musicales
        key = musical.get("key_signature", "Unknown")
        if key != "Unknown":
            insights.append(f"La canción está en {key}, lo que le da un carácter específico")
        
        tempo = musical.get("tempo", {}).get("bpm", 0)
        if tempo > 140:
            insights.append("Tempo rápido sugiere alta energía y movimiento")
        elif tempo < 80:
            insights.append("Tempo lento sugiere ambiente más relajado o emocional")
        
        # Insights técnicos
        energy = technical.get("energy", {}).get("value", 0)
        if energy > 0.7:
            insights.append("Alta energía indica track dinámico y potente")
        elif energy < 0.4:
            insights.append("Baja energía sugiere ambiente más íntimo o contemplativo")
        
        # Insights de género y emoción
        genre_name = genre.get("primary_genre", "Unknown")
        emotion_name = emotion.get("primary_emotion", "Unknown")
        if genre_name != "Unknown" and emotion_name != "Unknown":
            insights.append(f"Combinación de {genre_name} con emoción {emotion_name} crea un perfil único")
        
        return insights if insights else ["Análisis completo disponible en secciones detalladas"]
    
    def _generate_recommendations(self, analysis: Dict, track_info: Dict) -> List[str]:
        """Genera recomendaciones"""
        recommendations = []
        
        popularity = track_info.get("popularity", 0)
        technical = analysis.get("technical_analysis", {})
        energy = technical.get("energy", {}).get("value", 0)
        danceability = technical.get("danceability", {}).get("value", 0)
        
        if popularity < 50:
            recommendations.append("Considerar estrategia de marketing para aumentar visibilidad")
        
        if energy < 0.5 and danceability < 0.5:
            recommendations.append("Track podría beneficiarse de mayor energía y bailabilidad para audiencia más amplia")
        
        if popularity > 70 and energy > 0.7:
            recommendations.append("Track tiene potencial comercial alto - considerar promoción adicional")
        
        return recommendations if recommendations else ["Track está bien posicionado"]
    
    def generate_comparison_report(self, track_ids: List[str]) -> Dict[str, Any]:
        """Genera reporte comparativo"""
        try:
            if len(track_ids) > 10:
                return {"error": "Máximo 10 tracks para comparación"}
            
            tracks_data = []
            for track_id in track_ids:
                track_info = self.spotify.get_track(track_id)
                spotify_data = self.spotify.get_track_full_analysis(track_id)
                analysis = self.music_analyzer.analyze_track(spotify_data)
                
                if track_info and analysis:
                    tracks_data.append({
                        "track_id": track_id,
                        "track_info": track_info,
                        "analysis": analysis
                    })
            
            if not tracks_data:
                return {"error": "No se pudieron analizar tracks"}
            
            # Comparación
            comparison = {
                "tracks_compared": len(tracks_data),
                "comparison_summary": self._compare_tracks(tracks_data),
                "detailed_comparison": tracks_data
            }
            
            return {
                "success": True,
                "report": comparison
            }
        except Exception as e:
            self.logger.error(f"Error generating comparison report: {e}")
            return {"error": str(e)}
    
    def _compare_tracks(self, tracks_data: List[Dict]) -> Dict[str, Any]:
        """Compara tracks"""
        if not tracks_data:
            return {}
        
        # Extraer características
        energies = []
        danceabilities = []
        tempos = []
        popularities = []
        
        for track in tracks_data:
            analysis = track.get("analysis", {})
            track_info = track.get("track_info", {})
            
            technical = analysis.get("technical_analysis", {})
            musical = analysis.get("musical_analysis", {})
            
            energies.append(technical.get("energy", {}).get("value", 0.5))
            danceabilities.append(technical.get("danceability", {}).get("value", 0.5))
            tempos.append(musical.get("tempo", {}).get("bpm", 120))
            popularities.append(track_info.get("popularity", 0))
        
        return {
            "average_energy": round(sum(energies) / len(energies), 3) if energies else 0,
            "average_danceability": round(sum(danceabilities) / len(danceabilities), 3) if danceabilities else 0,
            "average_tempo": round(sum(tempos) / len(tempos), 2) if tempos else 0,
            "average_popularity": round(sum(popularities) / len(popularities), 2) if popularities else 0,
            "energy_range": round(max(energies) - min(energies), 3) if energies else 0,
            "danceability_range": round(max(danceabilities) - min(danceabilities), 3) if danceabilities else 0
        }

