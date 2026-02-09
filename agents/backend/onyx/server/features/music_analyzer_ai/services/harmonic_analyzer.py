"""
Analizador avanzado de progresiones armónicas
"""

import logging
from typing import Dict, List, Any, Optional
from collections import Counter

logger = logging.getLogger(__name__)


class HarmonicAnalyzer:
    """Analizador avanzado de progresiones armónicas"""
    
    # Progresiones comunes
    COMMON_PROGRESSIONS = {
        "I-V-vi-IV": ["I", "V", "vi", "IV"],  # Pop progression
        "vi-IV-I-V": ["vi", "IV", "I", "V"],  # Pop progression variant
        "I-vi-IV-V": ["I", "vi", "IV", "V"],  # 50s progression
        "I-IV-V": ["I", "IV", "V"],  # Blues progression
        "ii-V-I": ["ii", "V", "I"],  # Jazz progression
        "I-vi-ii-V": ["I", "vi", "ii", "V"],  # Circle progression
        "vi-V-IV-V": ["vi", "V", "IV", "V"],  # Andalusian cadence
    }
    
    # Funciones de acordes
    CHORD_FUNCTIONS = {
        "I": "Tonic",
        "ii": "Supertonic",
        "iii": "Mediant",
        "IV": "Subdominant",
        "V": "Dominant",
        "vi": "Submediant",
        "vii°": "Leading tone"
    }
    
    def __init__(self):
        self.logger = logger
    
    def analyze_harmonic_progression(self, audio_analysis: Dict[str, Any],
                                    key: int, mode: int) -> Dict[str, Any]:
        """Analiza la progresión armónica de una canción"""
        segments = audio_analysis.get("segments", [])
        sections = audio_analysis.get("sections", [])
        
        # Extraer información de pitch de los segmentos
        pitch_data = []
        for segment in segments[:200]:  # Limitar a primeros 200 segmentos
            pitches = segment.get("pitches", [])
            if pitches:
                # Encontrar la nota más prominente
                max_pitch_idx = pitches.index(max(pitches))
                pitch_data.append({
                    "time": segment.get("start", 0),
                    "pitch_class": max_pitch_idx,
                    "confidence": max(pitches)
                })
        
        # Identificar acordes basados en pitch classes
        chords = self._identify_chords(pitch_data, key, mode)
        
        # Encontrar progresiones
        progressions = self._find_progressions(chords, key, mode)
        
        # Analizar cadencias
        cadences = self._analyze_cadences(chords)
        
        return {
            "chords": chords[:50],  # Primeros 50 acordes
            "progressions": progressions,
            "cadences": cadences,
            "harmonic_complexity": self._assess_harmonic_complexity(chords, progressions),
            "common_patterns": self._find_common_patterns(chords)
        }
    
    def _identify_chords(self, pitch_data: List[Dict[str, Any]],
                        key: int, mode: int) -> List[Dict[str, Any]]:
        """Identifica acordes basados en pitch data"""
        chords = []
        
        # Agrupar por tiempo (ventana de 1 segundo)
        time_windows = {}
        for data in pitch_data:
            window = int(data["time"])
            if window not in time_windows:
                time_windows[window] = []
            time_windows[window].append(data)
        
        # Identificar acorde para cada ventana
        for time, pitches in sorted(time_windows.items()):
            # Contar pitch classes
            pitch_counts = Counter(p["pitch_class"] for p in pitches)
            
            # Encontrar los 3 más comunes
            top_pitches = [p[0] for p in pitch_counts.most_common(3)]
            
            # Determinar función del acorde
            chord_function = self._pitch_to_chord_function(top_pitches[0], key, mode)
            
            chords.append({
                "time": time,
                "pitch_classes": top_pitches,
                "function": chord_function,
                "confidence": sum(p["confidence"] for p in pitches) / len(pitches)
            })
        
        return chords
    
    def _pitch_to_chord_function(self, pitch_class: int, key: int, mode: int) -> str:
        """Convierte un pitch class a función de acorde"""
        # Calcular intervalo desde la tónica
        interval = (pitch_class - key) % 12
        
        if mode == 1:  # Major
            functions = {
                0: "I",
                2: "ii",
                4: "iii",
                5: "IV",
                7: "V",
                9: "vi",
                11: "vii°"
            }
        else:  # Minor
            functions = {
                0: "i",
                2: "ii°",
                3: "III",
                5: "iv",
                7: "v",
                8: "VI",
                10: "VII"
            }
        
        return functions.get(interval, "?")
    
    def _find_progressions(self, chords: List[Dict[str, Any]],
                          key: int, mode: int) -> List[Dict[str, Any]]:
        """Encuentra progresiones comunes"""
        progressions_found = []
        
        # Extraer secuencia de funciones
        chord_sequence = [c["function"] for c in chords]
        
        # Buscar progresiones conocidas
        for name, pattern in self.COMMON_PROGRESSIONS.items():
            pattern_str = "-".join(pattern)
            
            # Buscar patrón en la secuencia
            for i in range(len(chord_sequence) - len(pattern) + 1):
                segment = chord_sequence[i:i+len(pattern)]
                segment_str = "-".join(segment)
                
                if segment_str == pattern_str:
                    progressions_found.append({
                        "name": name,
                        "pattern": pattern,
                        "start_time": chords[i]["time"],
                        "occurrences": 1
                    })
        
        return progressions_found
    
    def _analyze_cadences(self, chords: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analiza las cadencias en la progresión"""
        cadences = []
        
        if len(chords) < 2:
            return cadences
        
        for i in range(len(chords) - 1):
            current = chords[i]["function"]
            next_chord = chords[i+1]["function"]
            
            # Cadencia perfecta (V-I)
            if current == "V" and next_chord == "I":
                cadences.append({
                    "type": "Perfect Cadence",
                    "time": chords[i]["time"],
                    "chords": [current, next_chord],
                    "description": "V-I cadence (strong resolution)"
                })
            
            # Cadencia plagal (IV-I)
            elif current == "IV" and next_chord == "I":
                cadences.append({
                    "type": "Plagal Cadence",
                    "time": chords[i]["time"],
                    "chords": [current, next_chord],
                    "description": "IV-I cadence (Amen cadence)"
                })
            
            # Cadencia de medio (I-V)
            elif current == "I" and next_chord == "V":
                cadences.append({
                    "type": "Half Cadence",
                    "time": chords[i]["time"],
                    "chords": [current, next_chord],
                    "description": "I-V cadence (suspension)"
                })
        
        return cadences
    
    def _assess_harmonic_complexity(self, chords: List[Dict[str, Any]],
                                   progressions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evalúa la complejidad armónica"""
        unique_chords = len(set(c["function"] for c in chords))
        total_chords = len(chords)
        
        # Calcular diversidad
        diversity = unique_chords / total_chords if total_chords > 0 else 0
        
        # Calcular complejidad basada en progresiones
        progression_complexity = len(progressions) / max(total_chords / 4, 1)
        
        complexity_score = (diversity * 0.5) + (progression_complexity * 0.5)
        
        if complexity_score < 0.3:
            level = "Simple"
        elif complexity_score < 0.6:
            level = "Moderate"
        elif complexity_score < 0.8:
            level = "Complex"
        else:
            level = "Very Complex"
        
        return {
            "level": level,
            "score": round(complexity_score, 3),
            "unique_chords": unique_chords,
            "total_chords": total_chords,
            "diversity": round(diversity, 3),
            "progressions_found": len(progressions)
        }
    
    def _find_common_patterns(self, chords: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Encuentra patrones comunes en la progresión"""
        if len(chords) < 2:
            return []
        
        # Buscar secuencias de 2-4 acordes que se repiten
        patterns = []
        
        for length in [2, 3, 4]:
            sequences = {}
            chord_functions = [c["function"] for c in chords]
            
            for i in range(len(chord_functions) - length + 1):
                sequence = tuple(chord_functions[i:i+length])
                if sequence not in sequences:
                    sequences[sequence] = []
                sequences[sequence].append(i)
            
            # Encontrar secuencias que aparecen más de una vez
            for sequence, positions in sequences.items():
                if len(positions) > 1:
                    patterns.append({
                        "pattern": "-".join(sequence),
                        "length": length,
                        "occurrences": len(positions),
                        "positions": positions
                    })
        
        # Ordenar por número de ocurrencias
        patterns.sort(key=lambda x: x["occurrences"], reverse=True)
        
        return patterns[:10]  # Top 10 patrones

