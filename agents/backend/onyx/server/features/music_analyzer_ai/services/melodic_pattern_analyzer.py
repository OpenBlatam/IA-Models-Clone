"""
Servicio de análisis de patrones melódicos
"""

import logging
from typing import Dict, List, Any, Optional
from collections import Counter

from .spotify_service import SpotifyService

logger = logging.getLogger(__name__)


class MelodicPatternAnalyzer:
    """Analiza patrones melódicos en música"""
    
    def __init__(self):
        self.spotify = SpotifyService()
        self.logger = logger
    
    def analyze_melodic_patterns(self, track_id: str) -> Dict[str, Any]:
        """Analiza patrones melódicos de un track"""
        try:
            audio_analysis = self.spotify.get_track_audio_analysis(track_id)
            
            if not audio_analysis:
                return {"error": "No hay datos de audio disponibles"}
            
            segments = audio_analysis.get("segments", [])
            if not segments:
                return {"error": "No hay segmentos de audio disponibles"}
            
            # Análisis de patrones
            patterns = {
                "pitch_patterns": self._analyze_pitch_patterns(segments),
                "timbre_patterns": self._analyze_timbre_patterns(segments),
                "rhythmic_patterns": self._analyze_rhythmic_patterns(segments),
                "melodic_contour": self._analyze_melodic_contour(segments),
                "repetition_patterns": self._analyze_repetition_patterns(segments)
            }
            
            return {
                "track_id": track_id,
                "patterns": patterns,
                "complexity": self._calculate_melodic_complexity(patterns)
            }
        except Exception as e:
            self.logger.error(f"Error analyzing melodic patterns: {e}")
            return {"error": str(e)}
    
    def _analyze_pitch_patterns(self, segments: List[Dict]) -> Dict[str, Any]:
        """Analiza patrones de pitch"""
        pitches = []
        for segment in segments:
            if "pitches" in segment and segment["pitches"]:
                # Encontrar pitch dominante
                max_pitch = max(enumerate(segment["pitches"]), key=lambda x: x[1])
                pitches.append(max_pitch[0])
        
        if not pitches:
            return {"error": "No hay datos de pitch"}
        
        # Análisis de variación
        pitch_variance = self._calculate_variance(pitches)
        
        # Patrones comunes
        pitch_counter = Counter(pitches)
        most_common_pitches = dict(pitch_counter.most_common(5))
        
        return {
            "average_pitch": round(sum(pitches) / len(pitches), 2),
            "pitch_variance": round(pitch_variance, 3),
            "pitch_range": max(pitches) - min(pitches) if pitches else 0,
            "most_common_pitches": most_common_pitches,
            "pattern_type": "Ascending" if pitches[-1] > pitches[0] else "Descending" if pitches[-1] < pitches[0] else "Stable"
        }
    
    def _analyze_timbre_patterns(self, segments: List[Dict]) -> Dict[str, Any]:
        """Analiza patrones de timbre"""
        timbres = []
        for segment in segments:
            if "timbre" in segment and segment["timbre"]:
                timbres.append(segment["timbre"])
        
        if not timbres:
            return {"error": "No hay datos de timbre"}
        
        # Análisis de variación de timbre
        timbre_variance = sum(
            sum((t[i] - sum(t) / len(t)) ** 2 for i in range(len(t))) / len(t)
            for t in timbres
        ) / len(timbres) if timbres else 0
        
        return {
            "timbre_variance": round(timbre_variance, 3),
            "timbre_complexity": "High" if timbre_variance > 50 else "Medium" if timbre_variance > 20 else "Low",
            "segments_analyzed": len(timbres)
        }
    
    def _analyze_rhythmic_patterns(self, segments: List[Dict]) -> Dict[str, Any]:
        """Analiza patrones rítmicos"""
        durations = [s.get("duration", 0) for s in segments]
        
        if not durations:
            return {"error": "No hay datos de duración"}
        
        # Análisis de variación rítmica
        duration_variance = self._calculate_variance(durations)
        
        # Patrones de duración
        avg_duration = sum(durations) / len(durations)
        short_segments = sum(1 for d in durations if d < avg_duration * 0.7)
        long_segments = sum(1 for d in durations if d > avg_duration * 1.3)
        
        return {
            "average_duration": round(avg_duration, 3),
            "duration_variance": round(duration_variance, 3),
            "rhythmic_consistency": "High" if duration_variance < 0.1 else "Medium" if duration_variance < 0.3 else "Low",
            "short_segments_ratio": round(short_segments / len(durations), 3),
            "long_segments_ratio": round(long_segments / len(durations), 3)
        }
    
    def _analyze_melodic_contour(self, segments: List[Dict]) -> Dict[str, Any]:
        """Analiza el contorno melódico"""
        pitches = []
        for segment in segments:
            if "pitches" in segment and segment["pitches"]:
                max_pitch = max(enumerate(segment["pitches"]), key=lambda x: x[1])
                pitches.append(max_pitch[0])
        
        if len(pitches) < 2:
            return {"error": "No hay suficientes datos para análisis de contorno"}
        
        # Calcular dirección
        ascending = sum(1 for i in range(1, len(pitches)) if pitches[i] > pitches[i-1])
        descending = sum(1 for i in range(1, len(pitches)) if pitches[i] < pitches[i-1])
        stable = len(pitches) - 1 - ascending - descending
        
        total_changes = len(pitches) - 1
        ascending_ratio = ascending / total_changes if total_changes > 0 else 0
        descending_ratio = descending / total_changes if total_changes > 0 else 0
        
        # Determinar tipo de contorno
        if ascending_ratio > 0.5:
            contour_type = "Ascending"
        elif descending_ratio > 0.5:
            contour_type = "Descending"
        elif ascending_ratio > 0.3 or descending_ratio > 0.3:
            contour_type = "Wavy"
        else:
            contour_type = "Stable"
        
        return {
            "contour_type": contour_type,
            "ascending_ratio": round(ascending_ratio, 3),
            "descending_ratio": round(descending_ratio, 3),
            "stable_ratio": round(stable / total_changes, 3) if total_changes > 0 else 0,
            "pitch_range": max(pitches) - min(pitches) if pitches else 0
        }
    
    def _analyze_repetition_patterns(self, segments: List[Dict]) -> Dict[str, Any]:
        """Analiza patrones de repetición"""
        # Simplificado: analizar similitud entre segmentos
        if len(segments) < 2:
            return {"error": "No hay suficientes segmentos"}
        
        # Comparar segmentos adyacentes
        similar_segments = 0
        for i in range(1, len(segments)):
            seg1 = segments[i-1]
            seg2 = segments[i]
            
            # Comparar pitches si están disponibles
            if "pitches" in seg1 and "pitches" in seg2:
                if seg1["pitches"] and seg2["pitches"]:
                    similarity = sum(abs(a - b) for a, b in zip(seg1["pitches"], seg2["pitches"])) / len(seg1["pitches"])
                    if similarity < 0.1:  # Umbral de similitud
                        similar_segments += 1
        
        repetition_ratio = similar_segments / (len(segments) - 1) if len(segments) > 1 else 0
        
        return {
            "repetition_ratio": round(repetition_ratio, 3),
            "similar_segments": similar_segments,
            "pattern_type": "Highly Repetitive" if repetition_ratio > 0.5 else "Moderately Repetitive" if repetition_ratio > 0.2 else "Low Repetition"
        }
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calcula la varianza de una lista de valores"""
        if not values:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance
    
    def _calculate_melodic_complexity(self, patterns: Dict) -> Dict[str, Any]:
        """Calcula la complejidad melódica general"""
        complexity_factors = []
        complexity_score = 0
        
        # Factor de variación de pitch
        pitch_patterns = patterns.get("pitch_patterns", {})
        if isinstance(pitch_patterns, dict) and "pitch_variance" in pitch_patterns:
            if pitch_patterns["pitch_variance"] > 5:
                complexity_factors.append("High pitch variation")
                complexity_score += 0.3
        
        # Factor de timbre
        timbre_patterns = patterns.get("timbre_patterns", {})
        if isinstance(timbre_patterns, dict) and "timbre_complexity" in timbre_patterns:
            if timbre_patterns["timbre_complexity"] == "High":
                complexity_factors.append("Complex timbre")
                complexity_score += 0.2
        
        # Factor de contorno
        melodic_contour = patterns.get("melodic_contour", {})
        if isinstance(melodic_contour, dict) and "contour_type" in melodic_contour:
            if melodic_contour["contour_type"] == "Wavy":
                complexity_factors.append("Wavy melodic contour")
                complexity_score += 0.2
        
        # Factor de repetición
        repetition = patterns.get("repetition_patterns", {})
        if isinstance(repetition, dict) and "repetition_ratio" in repetition:
            if repetition["repetition_ratio"] < 0.2:
                complexity_factors.append("Low repetition")
                complexity_score += 0.3
        
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

