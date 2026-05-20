"""
Servicio de análisis temporal de música
"""

import logging
from typing import Dict, List, Any, Optional
from collections import defaultdict

from .spotify_service import SpotifyService

logger = logging.getLogger(__name__)


class TemporalAnalyzer:
    """Analiza cambios temporales en música"""
    
    def __init__(self):
        self.spotify = SpotifyService()
        self.logger = logger
    
    def analyze_temporal_structure(self, audio_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza la estructura temporal de una canción"""
        try:
            sections = audio_analysis.get("sections", [])
            segments = audio_analysis.get("segments", [])
            
            if not sections:
                return {"error": "No hay datos de secciones disponibles"}
            
            # Análisis de progresión temporal
            temporal_progression = []
            for i, section in enumerate(sections):
                start = section.get("start", 0)
                duration = section.get("duration", 0)
                energy = section.get("loudness", -10)  # Normalizar
                tempo = section.get("tempo", 120)
                key = section.get("key", -1)
                mode = section.get("mode", 0)
                
                temporal_progression.append({
                    "section_index": i,
                    "start_time": round(start, 2),
                    "duration": round(duration, 2),
                    "end_time": round(start + duration, 2),
                    "energy": round(energy, 2),
                    "tempo": round(tempo, 2) if tempo else None,
                    "key": key,
                    "mode": mode,
                    "time_signature": section.get("time_signature", 4)
                })
            
            # Detectar cambios significativos
            changes = self._detect_temporal_changes(temporal_progression)
            
            # Análisis de dinámica
            dynamics = self._analyze_dynamics(temporal_progression)
            
            # Análisis de estructura
            structure = self._analyze_structure(sections)
            
            return {
                "total_sections": len(sections),
                "total_duration": sum(s.get("duration", 0) for s in sections),
                "temporal_progression": temporal_progression,
                "changes": changes,
                "dynamics": dynamics,
                "structure": structure,
                "complexity": self._calculate_temporal_complexity(temporal_progression)
            }
        except Exception as e:
            self.logger.error(f"Error analyzing temporal structure: {e}")
            return {"error": str(e)}
    
    def analyze_energy_progression(self, audio_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza la progresión de energía a lo largo del tiempo"""
        try:
            sections = audio_analysis.get("sections", [])
            segments = audio_analysis.get("segments", [])
            
            if not sections:
                return {"error": "No hay datos disponibles"}
            
            # Progresión de energía por sección
            energy_progression = []
            for section in sections:
                energy_progression.append({
                    "time": section.get("start", 0),
                    "energy": section.get("loudness", -10),
                    "duration": section.get("duration", 0)
                })
            
            # Progresión de energía por segmento (más detallada)
            segment_energy = []
            for segment in segments[:100]:  # Limitar a 100 para rendimiento
                segment_energy.append({
                    "time": segment.get("start", 0),
                    "energy": segment.get("loudness", -10),
                    "pitch": segment.get("pitches", [])
                })
            
            # Calcular estadísticas
            energies = [e["energy"] for e in energy_progression]
            avg_energy = sum(energies) / len(energies) if energies else 0
            max_energy = max(energies) if energies else 0
            min_energy = min(energies) if energies else 0
            
            # Detectar picos
            peaks = self._detect_energy_peaks(energy_progression)
            
            # Detectar build-ups y drops
            build_ups = self._detect_build_ups(energy_progression)
            drops = self._detect_drops(energy_progression)
            
            return {
                "progression": energy_progression,
                "segment_details": segment_energy[:20],  # Primeros 20 para respuesta
                "statistics": {
                    "average_energy": round(avg_energy, 2),
                    "max_energy": round(max_energy, 2),
                    "min_energy": round(min_energy, 2),
                    "energy_range": round(max_energy - min_energy, 2)
                },
                "peaks": peaks,
                "build_ups": build_ups,
                "drops": drops,
                "pattern": self._identify_energy_pattern(energy_progression)
            }
        except Exception as e:
            self.logger.error(f"Error analyzing energy progression: {e}")
            return {"error": str(e)}
    
    def analyze_tempo_changes(self, audio_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza cambios de tempo a lo largo del tiempo"""
        try:
            sections = audio_analysis.get("sections", [])
            
            if not sections:
                return {"error": "No hay datos disponibles"}
            
            tempo_changes = []
            base_tempo = None
            
            for section in sections:
                tempo = section.get("tempo", None)
                if tempo:
                    if base_tempo is None:
                        base_tempo = tempo
                    
                    tempo_changes.append({
                        "time": section.get("start", 0),
                        "tempo": round(tempo, 2),
                        "change_from_base": round(tempo - base_tempo, 2) if base_tempo else 0,
                        "change_percent": round(((tempo - base_tempo) / base_tempo * 100), 2) if base_tempo else 0
                    })
            
            if not tempo_changes:
                return {"error": "No hay datos de tempo disponibles"}
            
            # Calcular estadísticas
            tempos = [t["tempo"] for t in tempo_changes]
            avg_tempo = sum(tempos) / len(tempos)
            max_tempo = max(tempos)
            min_tempo = min(tempos)
            
            # Detectar variaciones significativas
            significant_changes = [
                t for t in tempo_changes
                if abs(t["change_percent"]) > 5
            ]
            
            return {
                "base_tempo": round(base_tempo, 2) if base_tempo else None,
                "tempo_changes": tempo_changes,
                "statistics": {
                    "average_tempo": round(avg_tempo, 2),
                    "max_tempo": round(max_tempo, 2),
                    "min_tempo": round(min_tempo, 2),
                    "tempo_range": round(max_tempo - min_tempo, 2)
                },
                "significant_changes": significant_changes,
                "is_constant": len(significant_changes) == 0,
                "variation_level": self._calculate_variation_level(tempo_changes)
            }
        except Exception as e:
            self.logger.error(f"Error analyzing tempo changes: {e}")
            return {"error": str(e)}
    
    def _detect_temporal_changes(self, progression: List[Dict]) -> List[Dict]:
        """Detecta cambios significativos en la progresión temporal"""
        changes = []
        
        for i in range(1, len(progression)):
            prev = progression[i-1]
            curr = progression[i]
            
            # Cambio de energía significativo
            if abs(curr["energy"] - prev["energy"]) > 3:
                changes.append({
                    "type": "energy_change",
                    "time": curr["start_time"],
                    "from": round(prev["energy"], 2),
                    "to": round(curr["energy"], 2),
                    "magnitude": "high" if abs(curr["energy"] - prev["energy"]) > 5 else "medium"
                })
            
            # Cambio de tempo
            if prev["tempo"] and curr["tempo"]:
                if abs(curr["tempo"] - prev["tempo"]) > 10:
                    changes.append({
                        "type": "tempo_change",
                        "time": curr["start_time"],
                        "from": prev["tempo"],
                        "to": curr["tempo"]
                    })
            
            # Cambio de tonalidad
            if prev["key"] != curr["key"] and prev["key"] >= 0 and curr["key"] >= 0:
                changes.append({
                    "type": "key_change",
                    "time": curr["start_time"],
                    "from": prev["key"],
                    "to": curr["key"]
                })
        
        return changes
    
    def _analyze_dynamics(self, progression: List[Dict]) -> Dict[str, Any]:
        """Analiza la dinámica de la canción"""
        energies = [p["energy"] for p in progression]
        
        if not energies:
            return {}
        
        # Calcular variación
        avg_energy = sum(energies) / len(energies)
        variance = sum((e - avg_energy) ** 2 for e in energies) / len(energies)
        
        # Categorizar dinámica
        if variance < 1:
            dynamic_level = "Static"
        elif variance < 5:
            dynamic_level = "Moderate"
        else:
            dynamic_level = "Highly Dynamic"
        
        return {
            "average_energy": round(avg_energy, 2),
            "variance": round(variance, 2),
            "level": dynamic_level,
            "range": round(max(energies) - min(energies), 2)
        }
    
    def _analyze_structure(self, sections: List[Dict]) -> Dict[str, Any]:
        """Analiza la estructura de la canción"""
        if not sections:
            return {}
        
        # Identificar secciones por duración y energía
        section_types = []
        for section in sections:
            duration = section.get("duration", 0)
            energy = section.get("loudness", -10)
            
            # Clasificar sección
            if duration < 5:
                section_type = "Short"
            elif duration < 15:
                section_type = "Medium"
            else:
                section_type = "Long"
            
            if energy > -5:
                intensity = "High"
            elif energy > -10:
                intensity = "Medium"
            else:
                intensity = "Low"
            
            section_types.append({
                "type": section_type,
                "intensity": intensity,
                "duration": round(duration, 2)
            })
        
        return {
            "total_sections": len(sections),
            "section_types": section_types,
            "average_section_duration": round(
                sum(s.get("duration", 0) for s in sections) / len(sections), 2
            ) if sections else 0
        }
    
    def _calculate_temporal_complexity(self, progression: List[Dict]) -> Dict[str, Any]:
        """Calcula la complejidad temporal"""
        if not progression:
            return {"level": "Unknown", "score": 0}
        
        # Factores de complejidad
        energy_changes = sum(
            1 for i in range(1, len(progression))
            if abs(progression[i]["energy"] - progression[i-1]["energy"]) > 2
        )
        
        tempo_changes = sum(
            1 for i in range(1, len(progression))
            if progression[i]["tempo"] and progression[i-1]["tempo"]
            and abs(progression[i]["tempo"] - progression[i-1]["tempo"]) > 5
        )
        
        key_changes = sum(
            1 for i in range(1, len(progression))
            if progression[i]["key"] != progression[i-1]["key"]
            and progression[i]["key"] >= 0 and progression[i-1]["key"] >= 0
        )
        
        complexity_score = (
            (energy_changes / len(progression)) * 0.4 +
            (tempo_changes / len(progression)) * 0.3 +
            (key_changes / len(progression)) * 0.3
        )
        
        if complexity_score < 0.2:
            level = "Simple"
        elif complexity_score < 0.5:
            level = "Moderate"
        else:
            level = "Complex"
        
        return {
            "level": level,
            "score": round(complexity_score, 3),
            "factors": {
                "energy_changes": energy_changes,
                "tempo_changes": tempo_changes,
                "key_changes": key_changes
            }
        }
    
    def _detect_energy_peaks(self, progression: List[Dict]) -> List[Dict]:
        """Detecta picos de energía"""
        peaks = []
        
        for i in range(1, len(progression) - 1):
            prev = progression[i-1]["energy"]
            curr = progression[i]["energy"]
            next_e = progression[i+1]["energy"]
            
            if curr > prev and curr > next_e and curr > -5:  # Pico significativo
                peaks.append({
                    "time": progression[i]["time"],
                    "energy": round(curr, 2),
                    "magnitude": "high" if curr > -3 else "medium"
                })
        
        return peaks
    
    def _detect_build_ups(self, progression: List[Dict]) -> List[Dict]:
        """Detecta build-ups (aumentos progresivos de energía)"""
        build_ups = []
        
        for i in range(len(progression) - 2):
            window = progression[i:i+3]
            energies = [w["energy"] for w in window]
            
            # Verificar si hay aumento progresivo
            if energies[0] < energies[1] < energies[2]:
                build_ups.append({
                    "start_time": window[0]["time"],
                    "end_time": window[2]["time"],
                    "start_energy": round(energies[0], 2),
                    "end_energy": round(energies[2], 2),
                    "increase": round(energies[2] - energies[0], 2)
                })
        
        return build_ups
    
    def _detect_drops(self, progression: List[Dict]) -> List[Dict]:
        """Detecta drops (caídas de energía)"""
        drops = []
        
        for i in range(len(progression) - 2):
            window = progression[i:i+3]
            energies = [w["energy"] for w in window]
            
            # Verificar si hay caída significativa
            if energies[0] > energies[1] > energies[2] and (energies[0] - energies[2]) > 3:
                drops.append({
                    "start_time": window[0]["time"],
                    "end_time": window[2]["time"],
                    "start_energy": round(energies[0], 2),
                    "end_energy": round(energies[2], 2),
                    "decrease": round(energies[0] - energies[2], 2)
                })
        
        return drops
    
    def _identify_energy_pattern(self, progression: List[Dict]) -> str:
        """Identifica el patrón de energía"""
        if not progression:
            return "Unknown"
        
        energies = [p["energy"] for p in progression]
        avg_energy = sum(energies) / len(energies)
        
        # Verificar si es ascendente, descendente o constante
        first_half = energies[:len(energies)//2]
        second_half = energies[len(energies)//2:]
        
        first_avg = sum(first_half) / len(first_half) if first_half else 0
        second_avg = sum(second_half) / len(second_half) if second_half else 0
        
        if second_avg > first_avg + 2:
            return "Ascending"
        elif first_avg > second_avg + 2:
            return "Descending"
        else:
            return "Stable"
    
    def _calculate_variation_level(self, tempo_changes: List[Dict]) -> str:
        """Calcula el nivel de variación de tempo"""
        if not tempo_changes:
            return "Unknown"
        
        changes = [abs(t["change_percent"]) for t in tempo_changes]
        avg_change = sum(changes) / len(changes) if changes else 0
        
        if avg_change < 2:
            return "Very Stable"
        elif avg_change < 5:
            return "Stable"
        elif avg_change < 10:
            return "Moderate Variation"
        else:
            return "High Variation"

