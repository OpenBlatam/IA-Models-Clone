"""
Servicio de análisis de dinámica musical
"""

import logging
from typing import Dict, List, Any, Optional

from .spotify_service import SpotifyService

logger = logging.getLogger(__name__)


class DynamicsAnalyzer:
    """Analiza la dinámica musical (volumen, intensidad, cambios)"""
    
    def __init__(self):
        self.spotify = SpotifyService()
        self.logger = logger
    
    def analyze_dynamics(self, track_id: str) -> Dict[str, Any]:
        """Analiza la dinámica de un track"""
        try:
            audio_analysis = self.spotify.get_track_audio_analysis(track_id)
            audio_features = self.spotify.get_track_audio_features(track_id)
            
            if not audio_analysis or not audio_features:
                return {"error": "No hay datos de audio disponibles"}
            
            sections = audio_analysis.get("sections", [])
            segments = audio_analysis.get("segments", [])
            
            # Análisis de dinámica
            dynamics = {
                "loudness_analysis": self._analyze_loudness(sections, segments),
                "energy_dynamics": self._analyze_energy_dynamics(sections, audio_features),
                "dynamic_range": self._calculate_dynamic_range(sections),
                "intensity_changes": self._analyze_intensity_changes(sections),
                "crescendo_decrescendo": self._detect_crescendo_decrescendo(sections)
            }
            
            return {
                "track_id": track_id,
                "dynamics": dynamics,
                "overall_dynamic_profile": self._determine_dynamic_profile(dynamics)
            }
        except Exception as e:
            self.logger.error(f"Error analyzing dynamics: {e}")
            return {"error": str(e)}
    
    def _analyze_loudness(self, sections: List[Dict], segments: List[Dict]) -> Dict[str, Any]:
        """Analiza la loudness"""
        if not sections:
            return {"error": "No hay secciones disponibles"}
        
        loudnesses = [s.get("loudness", -10) for s in sections]
        segment_loudnesses = [s.get("loudness", -10) for s in segments if "loudness" in s]
        
        avg_loudness = sum(loudnesses) / len(loudnesses) if loudnesses else -10
        max_loudness = max(loudnesses) if loudnesses else -10
        min_loudness = min(loudnesses) if loudnesses else -10
        
        # Variación de loudness
        loudness_variance = sum((l - avg_loudness) ** 2 for l in loudnesses) / len(loudnesses) if loudnesses else 0
        
        return {
            "average_loudness": round(avg_loudness, 2),
            "max_loudness": round(max_loudness, 2),
            "min_loudness": round(min_loudness, 2),
            "dynamic_range_db": round(max_loudness - min_loudness, 2),
            "loudness_variance": round(loudness_variance, 3),
            "consistency": "High" if loudness_variance < 5 else "Medium" if loudness_variance < 15 else "Low",
            "segments_analyzed": len(segment_loudnesses)
        }
    
    def _analyze_energy_dynamics(self, sections: List[Dict], audio_features: Dict) -> Dict[str, Any]:
        """Analiza la dinámica de energía"""
        if not sections:
            return {"error": "No hay secciones disponibles"}
        
        energies = [s.get("energy", 0.5) for s in sections if "energy" in s]
        overall_energy = audio_features.get("energy", 0.5)
        
        if not energies:
            return {
                "overall_energy": round(overall_energy, 3),
                "energy_variance": 0,
                "energy_progression": "Unknown"
            }
        
        avg_energy = sum(energies) / len(energies)
        energy_variance = sum((e - avg_energy) ** 2 for e in energies) / len(energies)
        
        # Progresión de energía
        if len(energies) > 1:
            if energies[-1] > energies[0] + 0.1:
                progression = "Increasing"
            elif energies[-1] < energies[0] - 0.1:
                progression = "Decreasing"
            else:
                progression = "Stable"
        else:
            progression = "Unknown"
        
        return {
            "overall_energy": round(overall_energy, 3),
            "average_section_energy": round(avg_energy, 3),
            "energy_variance": round(energy_variance, 3),
            "energy_progression": progression,
            "energy_consistency": "High" if energy_variance < 0.05 else "Medium" if energy_variance < 0.1 else "Low"
        }
    
    def _calculate_dynamic_range(self, sections: List[Dict]) -> Dict[str, Any]:
        """Calcula el rango dinámico"""
        if not sections:
            return {"error": "No hay secciones disponibles"}
        
        loudnesses = [s.get("loudness", -10) for s in sections]
        dynamic_range = max(loudnesses) - min(loudnesses) if loudnesses else 0
        
        if dynamic_range > 15:
            range_level = "Wide"
        elif dynamic_range > 8:
            range_level = "Moderate"
        else:
            range_level = "Narrow"
        
        return {
            "range_db": round(dynamic_range, 2),
            "level": range_level,
            "max_loudness": round(max(loudnesses), 2) if loudnesses else 0,
            "min_loudness": round(min(loudnesses), 2) if loudnesses else 0
        }
    
    def _analyze_intensity_changes(self, sections: List[Dict]) -> Dict[str, Any]:
        """Analiza cambios de intensidad"""
        if len(sections) < 2:
            return {"error": "No hay suficientes secciones"}
        
        intensity_changes = []
        for i in range(1, len(sections)):
            prev_loudness = sections[i-1].get("loudness", -10)
            curr_loudness = sections[i].get("loudness", -10)
            change = curr_loudness - prev_loudness
            intensity_changes.append(change)
        
        significant_changes = sum(1 for c in intensity_changes if abs(c) > 3)
        change_variance = sum((c - sum(intensity_changes) / len(intensity_changes)) ** 2 for c in intensity_changes) / len(intensity_changes) if intensity_changes else 0
        
        return {
            "total_changes": len(intensity_changes),
            "significant_changes": significant_changes,
            "average_change": round(sum(intensity_changes) / len(intensity_changes), 2) if intensity_changes else 0,
            "change_variance": round(change_variance, 3),
            "change_consistency": "High" if change_variance < 5 else "Medium" if change_variance < 15 else "Low"
        }
    
    def _detect_crescendo_decrescendo(self, sections: List[Dict]) -> Dict[str, Any]:
        """Detecta crescendos y decrescendos"""
        if len(sections) < 3:
            return {"error": "No hay suficientes secciones"}
        
        loudnesses = [s.get("loudness", -10) for s in sections]
        
        # Detectar crescendos (aumentos sostenidos)
        crescendos = 0
        decrescendos = 0
        
        for i in range(2, len(loudnesses)):
            # Crescendo: aumento sostenido
            if loudnesses[i] > loudnesses[i-1] > loudnesses[i-2]:
                crescendos += 1
            # Decrescendo: disminución sostenida
            elif loudnesses[i] < loudnesses[i-1] < loudnesses[i-2]:
                decrescendos += 1
        
        return {
            "crescendos_detected": crescendos,
            "decrescendos_detected": decrescendos,
            "overall_trend": "Crescendo" if loudnesses[-1] > loudnesses[0] + 2 else "Decrescendo" if loudnesses[-1] < loudnesses[0] - 2 else "Stable",
            "dynamic_activity": "High" if (crescendos + decrescendos) > len(loudnesses) * 0.3 else "Medium" if (crescendos + decrescendos) > len(loudnesses) * 0.1 else "Low"
        }
    
    def _determine_dynamic_profile(self, dynamics: Dict) -> Dict[str, Any]:
        """Determina el perfil dinámico general"""
        profile_factors = []
        complexity_score = 0
        
        loudness_analysis = dynamics.get("loudness_analysis", {})
        if isinstance(loudness_analysis, dict) and "consistency" in loudness_analysis:
            if loudness_analysis["consistency"] == "Low":
                profile_factors.append("Variable loudness")
                complexity_score += 0.3
        
        dynamic_range = dynamics.get("dynamic_range", {})
        if isinstance(dynamic_range, dict) and "level" in dynamic_range:
            if dynamic_range["level"] == "Wide":
                profile_factors.append("Wide dynamic range")
                complexity_score += 0.3
        
        crescendo_decrescendo = dynamics.get("crescendo_decrescendo", {})
        if isinstance(crescendo_decrescendo, dict) and "dynamic_activity" in crescendo_decrescendo:
            if crescendo_decrescendo["dynamic_activity"] == "High":
                profile_factors.append("High dynamic activity")
                complexity_score += 0.4
        
        if complexity_score > 0.7:
            profile = "Highly Dynamic"
        elif complexity_score > 0.4:
            profile = "Moderately Dynamic"
        else:
            profile = "Static"
        
        return {
            "profile": profile,
            "complexity_score": round(complexity_score, 3),
            "factors": profile_factors
        }

