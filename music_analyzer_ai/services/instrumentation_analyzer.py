"""
Servicio de análisis de instrumentación
"""

import logging
from typing import Dict, List, Any, Optional

from .spotify_service import SpotifyService

logger = logging.getLogger(__name__)


class InstrumentationAnalyzer:
    """Analiza la instrumentación de tracks"""
    
    def __init__(self):
        self.spotify = SpotifyService()
        self.logger = logger
    
    def analyze_instrumentation(self, track_id: str) -> Dict[str, Any]:
        """Analiza la instrumentación de un track"""
        try:
            audio_features = self.spotify.get_track_audio_features(track_id)
            audio_analysis = self.spotify.get_track_audio_analysis(track_id)
            
            if not audio_features or not audio_analysis:
                return {"error": "No hay datos de audio disponibles"}
            
            # Análisis basado en características de audio
            instrumentation = {
                "acoustic_vs_electric": self._analyze_acoustic_electric(audio_features),
                "instrumental_vs_vocal": self._analyze_instrumental_vocal(audio_features),
                "texture": self._analyze_texture(audio_features, audio_analysis),
                "arrangement": self._analyze_arrangement(audio_analysis),
                "estimated_instruments": self._estimate_instruments(audio_features, audio_analysis)
            }
            
            return {
                "track_id": track_id,
                "instrumentation": instrumentation
            }
        except Exception as e:
            self.logger.error(f"Error analyzing instrumentation: {e}")
            return {"error": str(e)}
    
    def _analyze_acoustic_electric(self, audio_features: Dict) -> Dict[str, Any]:
        """Analiza si es acústico o eléctrico"""
        acousticness = audio_features.get("acousticness", 0.5)
        
        if acousticness > 0.7:
            type_ = "Acoustic"
            confidence = "High"
        elif acousticness > 0.4:
            type_ = "Mixed Acoustic/Electric"
            confidence = "Medium"
        else:
            type_ = "Electric"
            confidence = "High"
        
        return {
            "type": type_,
            "acousticness": round(acousticness, 3),
            "confidence": confidence
        }
    
    def _analyze_instrumental_vocal(self, audio_features: Dict) -> Dict[str, Any]:
        """Analiza si es instrumental o vocal"""
        instrumentalness = audio_features.get("instrumentalness", 0.5)
        
        if instrumentalness > 0.7:
            type_ = "Instrumental"
            confidence = "High"
        elif instrumentalness > 0.4:
            type_ = "Mixed Instrumental/Vocal"
            confidence = "Medium"
        else:
            type_ = "Vocal"
            confidence = "High"
        
        return {
            "type": type_,
            "instrumentalness": round(instrumentalness, 3),
            "confidence": confidence
        }
    
    def _analyze_texture(self, audio_features: Dict, audio_analysis: Dict) -> Dict[str, Any]:
        """Analiza la textura del track"""
        # Factores de textura
        energy = audio_features.get("energy", 0.5)
        danceability = audio_features.get("danceability", 0.5)
        sections = audio_analysis.get("sections", [])
        
        # Densidad de secciones
        if sections:
            avg_section_duration = sum(s.get("duration", 0) for s in sections) / len(sections)
            density = 1.0 / (1.0 + avg_section_duration)  # Más secciones = mayor densidad
        else:
            density = 0.5
        
        # Calcular textura
        texture_score = (energy * 0.4 + danceability * 0.3 + density * 0.3)
        
        if texture_score > 0.7:
            texture = "Dense/Complex"
        elif texture_score > 0.4:
            texture = "Moderate"
        else:
            texture = "Sparse/Simple"
        
        return {
            "texture": texture,
            "score": round(texture_score, 3),
            "energy_factor": round(energy, 3),
            "danceability_factor": round(danceability, 3),
            "density_factor": round(density, 3)
        }
    
    def _analyze_arrangement(self, audio_analysis: Dict) -> Dict[str, Any]:
        """Analiza el arreglo"""
        sections = audio_analysis.get("sections", [])
        segments = audio_analysis.get("segments", [])
        
        if not sections:
            return {"error": "No hay datos de secciones"}
        
        # Análisis de estructura
        section_count = len(sections)
        avg_section_duration = sum(s.get("duration", 0) for s in sections) / len(sections)
        
        # Variación de loudness
        loudnesses = [s.get("loudness", -10) for s in sections]
        if loudnesses:
            loudness_variance = sum((l - sum(loudnesses)/len(loudnesses)) ** 2 for l in loudnesses) / len(loudnesses)
        else:
            loudness_variance = 0
        
        # Complejidad del arreglo
        complexity = (
            (section_count / 10) * 0.4 +  # Más secciones = más complejo
            (1.0 / (1.0 + avg_section_duration)) * 0.3 +  # Secciones cortas = más complejo
            (loudness_variance / 10) * 0.3  # Más variación = más complejo
        )
        
        if complexity > 0.6:
            arrangement_type = "Complex"
        elif complexity > 0.3:
            arrangement_type = "Moderate"
        else:
            arrangement_type = "Simple"
        
        return {
            "type": arrangement_type,
            "complexity_score": round(complexity, 3),
            "section_count": section_count,
            "average_section_duration": round(avg_section_duration, 2),
            "loudness_variance": round(loudness_variance, 2)
        }
    
    def _estimate_instruments(self, audio_features: Dict, audio_analysis: Dict) -> List[Dict[str, Any]]:
        """Estima instrumentos presentes"""
        instruments = []
        
        acousticness = audio_features.get("acousticness", 0.5)
        instrumentalness = audio_features.get("instrumentalness", 0.5)
        energy = audio_features.get("energy", 0.5)
        danceability = audio_features.get("danceability", 0.5)
        
        # Estimaciones basadas en características
        if acousticness > 0.6:
            instruments.append({
                "instrument": "Acoustic Guitar",
                "confidence": "Medium",
                "reason": "High acousticness"
            })
            instruments.append({
                "instrument": "Piano",
                "confidence": "Low",
                "reason": "Possible in acoustic tracks"
            })
        
        if energy > 0.7:
            instruments.append({
                "instrument": "Electric Guitar",
                "confidence": "Medium",
                "reason": "High energy"
            })
            instruments.append({
                "instrument": "Drums",
                "confidence": "High",
                "reason": "High energy tracks typically have drums"
            })
        
        if danceability > 0.7:
            instruments.append({
                "instrument": "Bass",
                "confidence": "Medium",
                "reason": "High danceability"
            })
            instruments.append({
                "instrument": "Synthesizer",
                "confidence": "Low",
                "reason": "Possible in dance tracks"
            })
        
        if instrumentalness > 0.5:
            instruments.append({
                "instrument": "Various Instruments",
                "confidence": "Medium",
                "reason": "High instrumentalness"
            })
        
        # Agregar instrumentos comunes
        if not instruments:
            instruments.append({
                "instrument": "Standard Band Setup",
                "confidence": "Low",
                "reason": "Typical instrumentation"
            })
        
        return instruments
