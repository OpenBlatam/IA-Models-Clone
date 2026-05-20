"""
Servicio de análisis armónico avanzado
"""

import logging
from typing import Dict, List, Any, Optional

from .spotify_service import SpotifyService
from .harmonic_analyzer import HarmonicAnalyzer

logger = logging.getLogger(__name__)


class AdvancedHarmonicAnalyzer:
    """Análisis armónico avanzado con progresiones y modulaciones"""
    
    def __init__(self):
        self.spotify = SpotifyService()
        self.harmonic_analyzer = HarmonicAnalyzer()
        self.logger = logger
        
        # Progresiones comunes
        self.common_progressions = {
            "I-V-vi-IV": [0, 7, 9, 5],  # Pop progression
            "vi-IV-I-V": [9, 5, 0, 7],  # Pop progression variant
            "I-vi-IV-V": [0, 9, 5, 7],  # 50s progression
            "ii-V-I": [2, 7, 0],  # Jazz progression
            "I-IV-V": [0, 5, 7],  # Blues progression
            "vi-V-IV-V": [9, 7, 5, 7],  # Pop progression
        }
    
    def analyze_advanced_harmony(self, track_id: str) -> Dict[str, Any]:
        """Análisis armónico avanzado"""
        try:
            audio_analysis = self.spotify.get_track_audio_analysis(track_id)
            audio_features = self.spotify.get_track_audio_features(track_id)
            
            if not audio_analysis or not audio_features:
                return {"error": "No hay datos de audio disponibles"}
            
            key = audio_features.get("key", -1)
            mode = audio_features.get("mode", 0)
            
            if key < 0:
                return {"error": "No se pudo determinar la tonalidad"}
            
            # Análisis avanzado
            analysis = {
                "key_analysis": self._analyze_key_stability(audio_analysis, key, mode),
                "progression_analysis": self._analyze_progressions(audio_analysis, key, mode),
                "modulation_analysis": self._analyze_modulations(audio_analysis),
                "harmonic_complexity": self._calculate_harmonic_complexity(audio_analysis, key, mode),
                "cadence_analysis": self._analyze_cadences(audio_analysis, key, mode)
            }
            
            return {
                "track_id": track_id,
                "key": key,
                "mode": mode,
                "advanced_harmony": analysis
            }
        except Exception as e:
            self.logger.error(f"Error analyzing advanced harmony: {e}")
            return {"error": str(e)}
    
    def _analyze_key_stability(self, audio_analysis: Dict, key: int, mode: int) -> Dict[str, Any]:
        """Analiza la estabilidad de la tonalidad"""
        sections = audio_analysis.get("sections", [])
        
        if not sections:
            return {"error": "No hay secciones disponibles"}
        
        # Contar cambios de tonalidad
        key_changes = 0
        for i in range(1, len(sections)):
            prev_key = sections[i-1].get("key", -1)
            curr_key = sections[i].get("key", -1)
            if prev_key >= 0 and curr_key >= 0 and prev_key != curr_key:
                key_changes += 1
        
        stability = "Very Stable" if key_changes == 0 else "Stable" if key_changes <= 2 else "Unstable"
        
        return {
            "key_changes": key_changes,
            "stability": stability,
            "sections_in_key": len([s for s in sections if s.get("key", -1) == key]),
            "total_sections": len(sections)
        }
    
    def _analyze_progressions(self, audio_analysis: Dict, key: int, mode: int) -> Dict[str, Any]:
        """Analiza progresiones de acordes"""
        sections = audio_analysis.get("sections", [])
        
        if not sections:
            return {"error": "No hay secciones disponibles"}
        
        # Detectar progresiones comunes
        detected_progressions = []
        
        # Simplificado: buscar patrones en secciones
        for i in range(len(sections) - 3):
            section_keys = [s.get("key", -1) for s in sections[i:i+4]]
            if all(k >= 0 for k in section_keys):
                # Normalizar a la tonalidad principal
                normalized = [(k - key) % 12 for k in section_keys]
                
                # Comparar con progresiones conocidas
                for prog_name, prog_pattern in self.common_progressions.items():
                    if normalized == prog_pattern[:len(normalized)]:
                        detected_progressions.append({
                            "progression": prog_name,
                            "position": i,
                            "confidence": "High"
                        })
                        break
        
        return {
            "detected_progressions": detected_progressions,
            "progression_count": len(detected_progressions),
            "common_progressions_used": list(set([p["progression"] for p in detected_progressions]))
        }
    
    def _analyze_modulations(self, audio_analysis: Dict) -> Dict[str, Any]:
        """Analiza modulaciones (cambios de tonalidad)"""
        sections = audio_analysis.get("sections", [])
        
        if not sections:
            return {"error": "No hay secciones disponibles"}
        
        modulations = []
        for i in range(1, len(sections)):
            prev_key = sections[i-1].get("key", -1)
            curr_key = sections[i].get("key", -1)
            
            if prev_key >= 0 and curr_key >= 0 and prev_key != curr_key:
                # Calcular distancia de modulación
                distance = abs(curr_key - prev_key)
                if distance > 6:
                    distance = 12 - distance
                
                modulations.append({
                    "from_key": prev_key,
                    "to_key": curr_key,
                    "distance": distance,
                    "section": i,
                    "type": "Close" if distance <= 2 else "Distant"
                })
        
        return {
            "modulation_count": len(modulations),
            "modulations": modulations,
            "modulation_type": "Frequent" if len(modulations) > 3 else "Moderate" if len(modulations) > 0 else "None"
        }
    
    def _calculate_harmonic_complexity(self, audio_analysis: Dict, key: int, mode: int) -> Dict[str, Any]:
        """Calcula la complejidad armónica"""
        sections = audio_analysis.get("sections", [])
        
        if not sections:
            return {"error": "No hay secciones disponibles"}
        
        complexity_factors = []
        complexity_score = 0
        
        # Factor de cambios de tonalidad
        key_changes = sum(1 for i in range(1, len(sections)) 
                         if sections[i].get("key", -1) != sections[i-1].get("key", -1))
        if key_changes > 2:
            complexity_factors.append("Multiple key changes")
            complexity_score += 0.3
        elif key_changes > 0:
            complexity_factors.append("Some key changes")
            complexity_score += 0.15
        
        # Factor de variación de tempo
        tempos = [s.get("tempo", 120) for s in sections if "tempo" in s]
        if tempos:
            tempo_variance = sum((t - sum(tempos)/len(tempos)) ** 2 for t in tempos) / len(tempos)
            if tempo_variance > 100:
                complexity_factors.append("Tempo variations")
                complexity_score += 0.2
        
        # Factor de número de secciones
        if len(sections) > 8:
            complexity_factors.append("Many sections")
            complexity_score += 0.2
        
        if complexity_score > 0.6:
            level = "Very Complex"
        elif complexity_score > 0.3:
            level = "Moderate"
        else:
            level = "Simple"
        
        return {
            "level": level,
            "score": round(complexity_score, 3),
            "factors": complexity_factors,
            "key_changes": key_changes,
            "section_count": len(sections)
        }
    
    def _analyze_cadences(self, audio_analysis: Dict, key: int, mode: int) -> Dict[str, Any]:
        """Analiza cadencias armónicas"""
        sections = audio_analysis.get("sections", [])
        
        if not sections or len(sections) < 2:
            return {"error": "No hay suficientes secciones"}
        
        cadences = []
        
        # Buscar cadencias al final de secciones
        for i in range(len(sections) - 1):
            prev_key = sections[i].get("key", -1)
            curr_key = sections[i+1].get("key", -1)
            
            if prev_key >= 0 and curr_key >= 0:
                # Simplificado: detectar cadencias comunes
                if prev_key == (key + 7) % 12 and curr_key == key:
                    cadences.append({
                        "type": "Perfect Cadence (V-I)",
                        "position": i,
                        "confidence": "Medium"
                    })
                elif prev_key == (key + 5) % 12 and curr_key == key:
                    cadences.append({
                        "type": "Plagal Cadence (IV-I)",
                        "position": i,
                        "confidence": "Medium"
                    })
        
        return {
            "cadence_count": len(cadences),
            "cadences": cadences,
            "cadence_types": list(set([c["type"] for c in cadences]))
        }

