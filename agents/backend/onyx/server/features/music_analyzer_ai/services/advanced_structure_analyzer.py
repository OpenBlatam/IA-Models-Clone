"""
Servicio de análisis avanzado de estructura de canciones
"""

import logging
from typing import Dict, List, Any, Optional

from .spotify_service import SpotifyService

logger = logging.getLogger(__name__)


class AdvancedStructureAnalyzer:
    """Análisis avanzado de estructura de canciones"""
    
    def __init__(self):
        self.spotify = SpotifyService()
        self.logger = logger
    
    def analyze_advanced_structure(self, track_id: str) -> Dict[str, Any]:
        """Análisis avanzado de estructura"""
        try:
            audio_analysis = self.spotify.get_track_audio_analysis(track_id)
            track_info = self.spotify.get_track(track_id)
            
            if not audio_analysis:
                return {"error": "No hay datos de audio disponibles"}
            
            sections = audio_analysis.get("sections", [])
            segments = audio_analysis.get("segments", [])
            
            # Análisis avanzado
            analysis = {
                "structure_mapping": self._map_structure(sections),
                "section_transitions": self._analyze_transitions(sections),
                "structural_complexity": self._calculate_structural_complexity(sections),
                "repetition_analysis": self._analyze_structural_repetition(sections),
                "build_up_analysis": self._analyze_build_ups(sections),
                "drop_analysis": self._analyze_drops(sections),
                "structural_pattern": self._identify_structural_pattern(sections)
            }
            
            return {
                "track_id": track_id,
                "track_name": track_info.get("name", "Unknown") if track_info else "Unknown",
                "duration_ms": track_info.get("duration_ms", 0) if track_info else 0,
                "advanced_structure": analysis
            }
        except Exception as e:
            self.logger.error(f"Error analyzing advanced structure: {e}")
            return {"error": str(e)}
    
    def _map_structure(self, sections: List[Dict]) -> Dict[str, Any]:
        """Mapea la estructura de la canción"""
        if not sections:
            return {"error": "No hay secciones disponibles"}
        
        # Identificar tipos de secciones
        section_types = []
        for i, section in enumerate(sections):
            loudness = section.get("loudness", -10)
            tempo = section.get("tempo", 120)
            duration = section.get("duration", 0)
            
            # Clasificar sección
            if i == 0:
                section_type = "Intro"
            elif i == len(sections) - 1:
                section_type = "Outro"
            elif loudness > -5:
                section_type = "Chorus" if duration > 15 else "Bridge"
            elif loudness < -10:
                section_type = "Verse"
            else:
                section_type = "Verse"
            
            section_types.append({
                "index": i,
                "type": section_type,
                "start": section.get("start", 0),
                "duration": duration,
                "loudness": round(loudness, 2),
                "tempo": round(tempo, 2)
            })
        
        # Contar tipos
        type_counts = {}
        for st in section_types:
            st_type = st["type"]
            type_counts[st_type] = type_counts.get(st_type, 0) + 1
        
        return {
            "sections": section_types,
            "section_count": len(sections),
            "type_distribution": type_counts,
            "structure_summary": " -> ".join([st["type"] for st in section_types])
        }
    
    def _analyze_transitions(self, sections: List[Dict]) -> Dict[str, Any]:
        """Analiza transiciones entre secciones"""
        if len(sections) < 2:
            return {"error": "No hay suficientes secciones"}
        
        transitions = []
        for i in range(1, len(sections)):
            prev = sections[i-1]
            curr = sections[i]
            
            loudness_change = curr.get("loudness", -10) - prev.get("loudness", -10)
            tempo_change = curr.get("tempo", 120) - prev.get("tempo", 120)
            
            transition_type = "Smooth"
            if abs(loudness_change) > 5:
                transition_type = "Abrupt"
            elif abs(loudness_change) > 2:
                transition_type = "Moderate"
            
            transitions.append({
                "from_section": i-1,
                "to_section": i,
                "loudness_change": round(loudness_change, 2),
                "tempo_change": round(tempo_change, 2),
                "transition_type": transition_type,
                "intensity": "High" if abs(loudness_change) > 5 else "Medium" if abs(loudness_change) > 2 else "Low"
            })
        
        return {
            "transitions": transitions,
            "transition_count": len(transitions),
            "smooth_transitions": sum(1 for t in transitions if t["transition_type"] == "Smooth"),
            "abrupt_transitions": sum(1 for t in transitions if t["transition_type"] == "Abrupt"),
            "average_loudness_change": round(sum(abs(t["loudness_change"]) for t in transitions) / len(transitions), 2) if transitions else 0
        }
    
    def _calculate_structural_complexity(self, sections: List[Dict]) -> Dict[str, Any]:
        """Calcula complejidad estructural"""
        if not sections:
            return {"error": "No hay secciones disponibles"}
        
        complexity_factors = []
        complexity_score = 0
        
        # Factor de número de secciones
        if len(sections) > 10:
            complexity_factors.append("Many sections")
            complexity_score += 0.3
        elif len(sections) > 6:
            complexity_factors.append("Moderate sections")
            complexity_score += 0.15
        
        # Factor de variación de duración
        durations = [s.get("duration", 0) for s in sections]
        if durations:
            duration_variance = sum((d - sum(durations)/len(durations)) ** 2 for d in durations) / len(durations)
            if duration_variance > 50:
                complexity_factors.append("Variable section lengths")
                complexity_score += 0.2
        
        # Factor de variación de loudness
        loudnesses = [s.get("loudness", -10) for s in sections]
        if loudnesses:
            loudness_variance = sum((l - sum(loudnesses)/len(loudnesses)) ** 2 for l in loudnesses) / len(loudnesses)
            if loudness_variance > 20:
                complexity_factors.append("Variable loudness")
                complexity_score += 0.25
        
        # Factor de cambios de tempo
        tempos = [s.get("tempo", 120) for s in sections if "tempo" in s]
        if tempos:
            tempo_variance = sum((t - sum(tempos)/len(tempos)) ** 2 for t in tempos) / len(tempos)
            if tempo_variance > 100:
                complexity_factors.append("Tempo changes")
                complexity_score += 0.25
        
        if complexity_score > 0.7:
            level = "Very Complex"
        elif complexity_score > 0.4:
            level = "Moderate"
        else:
            level = "Simple"
        
        return {
            "level": level,
            "score": round(complexity_score, 3),
            "factors": complexity_factors,
            "section_count": len(sections)
        }
    
    def _analyze_structural_repetition(self, sections: List[Dict]) -> Dict[str, Any]:
        """Analiza repetición estructural"""
        if len(sections) < 2:
            return {"error": "No hay suficientes secciones"}
        
        # Comparar secciones similares
        similar_sections = 0
        for i in range(len(sections)):
            for j in range(i+1, len(sections)):
                si = sections[i]
                sj = sections[j]
                
                # Comparar características
                loudness_diff = abs(si.get("loudness", -10) - sj.get("loudness", -10))
                tempo_diff = abs(si.get("tempo", 120) - sj.get("tempo", 120))
                duration_diff = abs(si.get("duration", 0) - sj.get("duration", 0))
                
                if loudness_diff < 2 and tempo_diff < 10 and duration_diff < 5:
                    similar_sections += 1
        
        total_comparisons = len(sections) * (len(sections) - 1) / 2
        repetition_ratio = similar_sections / total_comparisons if total_comparisons > 0 else 0
        
        return {
            "repetition_ratio": round(repetition_ratio, 3),
            "similar_sections": similar_sections,
            "repetition_level": "High" if repetition_ratio > 0.3 else "Medium" if repetition_ratio > 0.1 else "Low"
        }
    
    def _analyze_build_ups(self, sections: List[Dict]) -> Dict[str, Any]:
        """Analiza build-ups (aumentos de intensidad)"""
        if len(sections) < 3:
            return {"error": "No hay suficientes secciones"}
        
        build_ups = []
        for i in range(2, len(sections)):
            loudness1 = sections[i-2].get("loudness", -10)
            loudness2 = sections[i-1].get("loudness", -10)
            loudness3 = sections[i].get("loudness", -10)
            
            # Build-up: aumento sostenido
            if loudness1 < loudness2 < loudness3 and (loudness3 - loudness1) > 3:
                build_ups.append({
                    "start_section": i-2,
                    "end_section": i,
                    "loudness_increase": round(loudness3 - loudness1, 2),
                    "intensity": "High" if (loudness3 - loudness1) > 6 else "Medium"
                })
        
        return {
            "build_ups": build_ups,
            "build_up_count": len(build_ups),
            "has_build_ups": len(build_ups) > 0
        }
    
    def _analyze_drops(self, sections: List[Dict]) -> Dict[str, Any]:
        """Analiza drops (caídas de intensidad)"""
        if len(sections) < 3:
            return {"error": "No hay suficientes secciones"}
        
        drops = []
        for i in range(2, len(sections)):
            loudness1 = sections[i-2].get("loudness", -10)
            loudness2 = sections[i-1].get("loudness", -10)
            loudness3 = sections[i].get("loudness", -10)
            
            # Drop: caída sostenida
            if loudness1 > loudness2 > loudness3 and (loudness1 - loudness3) > 3:
                drops.append({
                    "start_section": i-2,
                    "end_section": i,
                    "loudness_decrease": round(loudness1 - loudness3, 2),
                    "intensity": "High" if (loudness1 - loudness3) > 6 else "Medium"
                })
        
        return {
            "drops": drops,
            "drop_count": len(drops),
            "has_drops": len(drops) > 0
        }
    
    def _identify_structural_pattern(self, sections: List[Dict]) -> Dict[str, Any]:
        """Identifica patrón estructural"""
        if not sections:
            return {"error": "No hay secciones disponibles"}
        
        # Mapear estructura
        structure_map = self._map_structure(sections)
        section_types = [s["type"] for s in structure_map.get("sections", [])]
        
        # Identificar patrones comunes
        pattern = "Custom"
        
        if len(section_types) >= 4:
            # Verse-Chorus pattern
            if "Verse" in section_types and "Chorus" in section_types:
                if section_types.count("Chorus") >= 2:
                    pattern = "Verse-Chorus"
            
            # Intro-Verse-Chorus-Outro
            if section_types[0] == "Intro" and section_types[-1] == "Outro":
                if "Verse" in section_types and "Chorus" in section_types:
                    pattern = "Intro-Verse-Chorus-Outro"
        
        return {
            "pattern": pattern,
            "section_sequence": " -> ".join(section_types),
            "pattern_confidence": "High" if pattern != "Custom" else "Low"
        }

