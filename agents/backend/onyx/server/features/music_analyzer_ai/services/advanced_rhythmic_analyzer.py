"""
Servicio de análisis rítmico avanzado
"""

import logging
from typing import Dict, List, Any, Optional
from collections import Counter

from .spotify_service import SpotifyService

logger = logging.getLogger(__name__)


class AdvancedRhythmicAnalyzer:
    """Análisis rítmico avanzado"""
    
    def __init__(self):
        self.spotify = SpotifyService()
        self.logger = logger
    
    def analyze_advanced_rhythm(self, track_id: str) -> Dict[str, Any]:
        """Análisis rítmico avanzado"""
        try:
            audio_analysis = self.spotify.get_track_audio_analysis(track_id)
            audio_features = self.spotify.get_track_audio_features(track_id)
            
            if not audio_analysis or not audio_features:
                return {"error": "No hay datos de audio disponibles"}
            
            beats = audio_analysis.get("beats", [])
            bars = audio_analysis.get("bars", [])
            sections = audio_analysis.get("sections", [])
            
            # Análisis avanzado
            analysis = {
                "rhythmic_patterns": self._analyze_rhythmic_patterns(beats, bars),
                "tempo_analysis": self._analyze_tempo_changes(sections, audio_features),
                "syncopation_analysis": self._analyze_syncopation(beats, bars),
                "rhythmic_complexity": self._calculate_rhythmic_complexity(beats, bars, sections),
                "groove_analysis": self._analyze_groove(beats, bars, audio_features)
            }
            
            return {
                "track_id": track_id,
                "advanced_rhythm": analysis
            }
        except Exception as e:
            self.logger.error(f"Error analyzing advanced rhythm: {e}")
            return {"error": str(e)}
    
    def _analyze_rhythmic_patterns(self, beats: List[Dict], bars: List[Dict]) -> Dict[str, Any]:
        """Analiza patrones rítmicos"""
        if not beats or not bars:
            return {"error": "No hay suficientes datos rítmicos"}
        
        # Calcular duraciones de beats
        beat_durations = []
        for i in range(1, len(beats)):
            duration = beats[i].get("start", 0) - beats[i-1].get("start", 0)
            beat_durations.append(duration)
        
        if not beat_durations:
            return {"error": "No se pudieron calcular duraciones"}
        
        # Análisis de consistencia
        avg_duration = sum(beat_durations) / len(beat_durations)
        variance = sum((d - avg_duration) ** 2 for d in beat_durations) / len(beat_durations)
        
        # Patrones
        pattern_type = "Consistent" if variance < 0.01 else "Variable" if variance < 0.05 else "Irregular"
        
        # Beats por bar
        beats_per_bar = len(beats) / len(bars) if bars else 0
        
        return {
            "beat_count": len(beats),
            "bar_count": len(bars),
            "beats_per_bar": round(beats_per_bar, 2),
            "average_beat_duration": round(avg_duration, 3),
            "rhythmic_variance": round(variance, 5),
            "pattern_type": pattern_type,
            "consistency": "High" if variance < 0.01 else "Medium" if variance < 0.05 else "Low"
        }
    
    def _analyze_tempo_changes(self, sections: List[Dict], audio_features: Dict) -> Dict[str, Any]:
        """Analiza cambios de tempo"""
        if not sections:
            return {"error": "No hay secciones disponibles"}
        
        base_tempo = audio_features.get("tempo", 120)
        section_tempos = [s.get("tempo", base_tempo) for s in sections if "tempo" in s]
        
        if not section_tempos:
            return {
                "base_tempo": round(base_tempo, 2),
                "tempo_changes": 0,
                "tempo_stability": "Stable"
            }
        
        # Contar cambios significativos
        tempo_changes = 0
        for i in range(1, len(section_tempos)):
            if abs(section_tempos[i] - section_tempos[i-1]) > 5:
                tempo_changes += 1
        
        tempo_variance = sum((t - base_tempo) ** 2 for t in section_tempos) / len(section_tempos) if section_tempos else 0
        
        return {
            "base_tempo": round(base_tempo, 2),
            "section_tempos": [round(t, 2) for t in section_tempos],
            "tempo_changes": tempo_changes,
            "tempo_variance": round(tempo_variance, 2),
            "tempo_stability": "Stable" if tempo_changes == 0 else "Variable" if tempo_changes <= 2 else "Unstable",
            "tempo_range": round(max(section_tempos) - min(section_tempos), 2) if section_tempos else 0
        }
    
    def _analyze_syncopation(self, beats: List[Dict], bars: List[Dict]) -> Dict[str, Any]:
        """Analiza sincopación"""
        if not beats or not bars:
            return {"error": "No hay suficientes datos"}
        
        # Simplificado: analizar distribución de beats dentro de bars
        syncopation_score = 0.0
        syncopated_beats = 0
        
        if len(bars) > 0:
            for bar in bars[:10]:  # Analizar primeras 10 bars
                bar_start = bar.get("start", 0)
                bar_duration = bar.get("duration", 0)
                
                # Beats en este bar
                bar_beats = [b for b in beats if bar_start <= b.get("start", 0) < bar_start + bar_duration]
                
                if len(bar_beats) > 1:
                    # Calcular si hay beats fuera de posiciones regulares
                    for i, beat in enumerate(bar_beats[1:], 1):
                        expected_position = bar_start + (bar_duration * i / len(bar_beats))
                        actual_position = beat.get("start", 0)
                        deviation = abs(actual_position - expected_position)
                        
                        if deviation > bar_duration * 0.1:  # Umbral de sincopación
                            syncopated_beats += 1
                            syncopation_score += deviation
        
        total_beats_analyzed = min(len(beats), 100)
        syncopation_ratio = syncopated_beats / total_beats_analyzed if total_beats_analyzed > 0 else 0
        
        return {
            "syncopation_score": round(syncopation_score, 3),
            "syncopated_beats": syncopated_beats,
            "syncopation_ratio": round(syncopation_ratio, 3),
            "syncopation_level": "High" if syncopation_ratio > 0.3 else "Medium" if syncopation_ratio > 0.1 else "Low"
        }
    
    def _calculate_rhythmic_complexity(self, beats: List[Dict], bars: List[Dict], sections: List[Dict]) -> Dict[str, Any]:
        """Calcula complejidad rítmica"""
        complexity_factors = []
        complexity_score = 0
        
        # Factor de variación de tempo
        if sections:
            tempos = [s.get("tempo", 120) for s in sections if "tempo" in s]
            if tempos:
                tempo_variance = sum((t - sum(tempos)/len(tempos)) ** 2 for t in tempos) / len(tempos)
                if tempo_variance > 50:
                    complexity_factors.append("Tempo variations")
                    complexity_score += 0.3
        
        # Factor de sincopación
        syncopation = self._analyze_syncopation(beats, bars)
        if isinstance(syncopation, dict) and "syncopation_level" in syncopation:
            if syncopation["syncopation_level"] == "High":
                complexity_factors.append("High syncopation")
                complexity_score += 0.3
            elif syncopation["syncopation_level"] == "Medium":
                complexity_factors.append("Moderate syncopation")
                complexity_score += 0.15
        
        # Factor de número de secciones
        if len(sections) > 8:
            complexity_factors.append("Many sections")
            complexity_score += 0.2
        
        # Factor de beats por bar
        if beats and bars:
            beats_per_bar = len(beats) / len(bars)
            if beats_per_bar > 4:
                complexity_factors.append("Complex time signature")
                complexity_score += 0.2
        
        if complexity_score > 0.7:
            level = "Very Complex"
        elif complexity_score > 0.4:
            level = "Moderate"
        else:
            level = "Simple"
        
        return {
            "level": level,
            "score": round(complexity_score, 3),
            "factors": complexity_factors
        }
    
    def _analyze_groove(self, beats: List[Dict], bars: List[Dict], audio_features: Dict) -> Dict[str, Any]:
        """Analiza groove (sensación rítmica)"""
        danceability = audio_features.get("danceability", 0.5)
        energy = audio_features.get("energy", 0.5)
        
        # Groove basado en características
        groove_score = (danceability * 0.6 + energy * 0.4)
        
        # Análisis de consistencia rítmica
        if beats and len(beats) > 1:
            beat_durations = [beats[i].get("start", 0) - beats[i-1].get("start", 0) 
                            for i in range(1, min(len(beats), 20))]
            if beat_durations:
                avg_duration = sum(beat_durations) / len(beat_durations)
                consistency = 1.0 - min(1.0, sum(abs(d - avg_duration) for d in beat_durations) / (len(beat_durations) * avg_duration))
                groove_score = (groove_score * 0.7 + consistency * 0.3)
        
        return {
            "groove_score": round(groove_score, 3),
            "groove_level": "High" if groove_score > 0.7 else "Medium" if groove_score > 0.4 else "Low",
            "danceability_factor": round(danceability, 3),
            "energy_factor": round(energy, 3)
        }

