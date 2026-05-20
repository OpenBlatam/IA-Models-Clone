"""
Servicio de visualización de datos musicales
"""

import logging
from typing import Dict, List, Any, Optional
import json

from .spotify_service import SpotifyService
from .music_analyzer import MusicAnalyzer

logger = logging.getLogger(__name__)


class DataVisualization:
    """Genera datos para visualización"""
    
    def __init__(self):
        self.spotify = SpotifyService()
        self.music_analyzer = MusicAnalyzer()
        self.logger = logger
    
    def generate_track_visualization_data(self, track_id: str) -> Dict[str, Any]:
        """Genera datos para visualización de un track"""
        try:
            audio_analysis = self.spotify.get_track_audio_analysis(track_id)
            audio_features = self.spotify.get_track_audio_features(track_id)
            track_info = self.spotify.get_track(track_id)
            
            if not audio_analysis or not audio_features:
                return {"error": "No hay datos disponibles"}
            
            # Datos para visualización
            visualization = {
                "track_info": {
                    "name": track_info.get("name", "Unknown") if track_info else "Unknown",
                    "artists": [a.get("name") for a in track_info.get("artists", [])] if track_info else [],
                    "duration_ms": track_info.get("duration_ms", 0) if track_info else 0
                },
                "energy_progression": self._generate_energy_progression(audio_analysis),
                "tempo_progression": self._generate_tempo_progression(audio_analysis),
                "loudness_progression": self._generate_loudness_progression(audio_analysis),
                "section_timeline": self._generate_section_timeline(audio_analysis),
                "feature_radar": self._generate_feature_radar(audio_features),
                "structure_map": self._generate_structure_map(audio_analysis)
            }
            
            return {
                "track_id": track_id,
                "visualization_data": visualization
            }
        except Exception as e:
            self.logger.error(f"Error generating visualization data: {e}")
            return {"error": str(e)}
    
    def _generate_energy_progression(self, audio_analysis: Dict) -> List[Dict[str, Any]]:
        """Genera datos de progresión de energía"""
        sections = audio_analysis.get("sections", [])
        
        progression = []
        for section in sections:
            if "energy" in section:
                progression.append({
                    "time": round(section.get("start", 0), 2),
                    "energy": round(section.get("energy", 0.5), 3),
                    "duration": round(section.get("duration", 0), 2)
                })
        
        return progression
    
    def _generate_tempo_progression(self, audio_analysis: Dict) -> List[Dict[str, Any]]:
        """Genera datos de progresión de tempo"""
        sections = audio_analysis.get("sections", [])
        
        progression = []
        for section in sections:
            if "tempo" in section:
                progression.append({
                    "time": round(section.get("start", 0), 2),
                    "tempo": round(section.get("tempo", 120), 2),
                    "duration": round(section.get("duration", 0), 2)
                })
        
        return progression
    
    def _generate_loudness_progression(self, audio_analysis: Dict) -> List[Dict[str, Any]]:
        """Genera datos de progresión de loudness"""
        sections = audio_analysis.get("sections", [])
        
        progression = []
        for section in sections:
            progression.append({
                "time": round(section.get("start", 0), 2),
                "loudness": round(section.get("loudness", -10), 2),
                "duration": round(section.get("duration", 0), 2)
            })
        
        return progression
    
    def _generate_section_timeline(self, audio_analysis: Dict) -> List[Dict[str, Any]]:
        """Genera timeline de secciones"""
        sections = audio_analysis.get("sections", [])
        
        timeline = []
        for i, section in enumerate(sections):
            timeline.append({
                "index": i,
                "start": round(section.get("start", 0), 2),
                "duration": round(section.get("duration", 0), 2),
                "end": round(section.get("start", 0) + section.get("duration", 0), 2),
                "loudness": round(section.get("loudness", -10), 2),
                "tempo": round(section.get("tempo", 120), 2) if "tempo" in section else None,
                "key": section.get("key", -1),
                "mode": section.get("mode", 0)
            })
        
        return timeline
    
    def _generate_feature_radar(self, audio_features: Dict) -> Dict[str, Any]:
        """Genera datos para gráfico radar de características"""
        return {
            "energy": round(audio_features.get("energy", 0.5), 3),
            "danceability": round(audio_features.get("danceability", 0.5), 3),
            "valence": round(audio_features.get("valence", 0.5), 3),
            "acousticness": round(audio_features.get("acousticness", 0.5), 3),
            "instrumentalness": round(audio_features.get("instrumentalness", 0.5), 3),
            "liveness": round(audio_features.get("liveness", 0.5), 3),
            "speechiness": round(audio_features.get("speechiness", 0.5), 3)
        }
    
    def _generate_structure_map(self, audio_analysis: Dict) -> Dict[str, Any]:
        """Genera mapa de estructura"""
        sections = audio_analysis.get("sections", [])
        
        structure = []
        for i, section in enumerate(sections):
            loudness = section.get("loudness", -10)
            
            # Clasificar sección
            if i == 0:
                section_type = "Intro"
            elif i == len(sections) - 1:
                section_type = "Outro"
            elif loudness > -5:
                section_type = "Chorus"
            else:
                section_type = "Verse"
            
            structure.append({
                "index": i,
                "type": section_type,
                "start": round(section.get("start", 0), 2),
                "duration": round(section.get("duration", 0), 2),
                "loudness": round(loudness, 2)
            })
        
        return {
            "sections": structure,
            "total_duration": round(sum(s.get("duration", 0) for s in sections), 2) if sections else 0,
            "section_count": len(sections)
        }
    
    def generate_comparison_visualization(self, track_ids: List[str]) -> Dict[str, Any]:
        """Genera datos para visualización comparativa"""
        try:
            if len(track_ids) > 10:
                return {"error": "Máximo 10 tracks para comparación"}
            
            comparison_data = {
                "tracks": [],
                "feature_comparison": {
                    "energy": [],
                    "danceability": [],
                    "valence": [],
                    "tempo": []
                }
            }
            
            for track_id in track_ids:
                track_info = self.spotify.get_track(track_id)
                audio_features = self.spotify.get_track_audio_features(track_id)
                
                if track_info and audio_features:
                    track_data = {
                        "track_id": track_id,
                        "name": track_info.get("name", "Unknown"),
                        "artists": [a.get("name") for a in track_info.get("artists", [])],
                        "features": {
                            "energy": round(audio_features.get("energy", 0.5), 3),
                            "danceability": round(audio_features.get("danceability", 0.5), 3),
                            "valence": round(audio_features.get("valence", 0.5), 3),
                            "tempo": round(audio_features.get("tempo", 120), 2)
                        }
                    }
                    
                    comparison_data["tracks"].append(track_data)
                    
                    # Agregar a comparación
                    comparison_data["feature_comparison"]["energy"].append(track_data["features"]["energy"])
                    comparison_data["feature_comparison"]["danceability"].append(track_data["features"]["danceability"])
                    comparison_data["feature_comparison"]["valence"].append(track_data["features"]["valence"])
                    comparison_data["feature_comparison"]["tempo"].append(track_data["features"]["tempo"])
            
            return {
                "track_count": len(comparison_data["tracks"]),
                "comparison_data": comparison_data
            }
        except Exception as e:
            self.logger.error(f"Error generating comparison visualization: {e}")
            return {"error": str(e)}

