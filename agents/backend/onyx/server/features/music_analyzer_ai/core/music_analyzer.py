"""
Analizador principal de música
"""

import logging
from typing import Dict, List, Any, Optional
import math

from ..services.genre_detector import GenreDetector
from ..services.harmonic_analyzer import HarmonicAnalyzer
from ..services.emotion_analyzer import EmotionAnalyzer

logger = logging.getLogger(__name__)


class MusicAnalyzer:
    """Analizador de música que procesa datos de Spotify y genera análisis detallado"""
    
    # Notas musicales
    NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    MODES = {
        0: "Minor",
        1: "Major"
    }
    
    # Escalas comunes
    SCALES = {
        "major": [0, 2, 4, 5, 7, 9, 11],
        "minor": [0, 2, 3, 5, 7, 8, 10],
        "pentatonic_major": [0, 2, 4, 7, 9],
        "pentatonic_minor": [0, 3, 5, 7, 10],
        "blues": [0, 3, 5, 6, 7, 10],
        "dorian": [0, 2, 3, 5, 7, 9, 10],
        "mixolydian": [0, 2, 4, 5, 7, 9, 10]
    }
    
    def __init__(self):
        self.logger = logger
        self.genre_detector = GenreDetector()
        self.harmonic_analyzer = HarmonicAnalyzer()
        self.emotion_analyzer = EmotionAnalyzer()
    
    def analyze_track(self, spotify_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza una canción completa con datos de Spotify"""
        track_info = spotify_data.get("track_info", {})
        audio_features = spotify_data.get("audio_features", {})
        audio_analysis = spotify_data.get("audio_analysis", {})
        
        # Detectar género
        genre_analysis = self.genre_detector.detect_genre(audio_features, audio_analysis)
        
        # Análisis de emociones
        emotion_analysis = self.emotion_analyzer.analyze_emotions(audio_features)
        
        # Análisis armónico avanzado
        key = audio_features.get("key", -1)
        mode = audio_features.get("mode", 0)
        harmonic_analysis = {}
        if key >= 0:
            harmonic_analysis = self.harmonic_analyzer.analyze_harmonic_progression(
                audio_analysis, key, mode
            )
        
        analysis = {
            "track_basic_info": self._extract_basic_info(track_info),
            "musical_analysis": self._analyze_musical_elements(audio_features, audio_analysis),
            "technical_analysis": self._analyze_technical_aspects(audio_features, audio_analysis),
            "composition_analysis": self._analyze_composition(audio_features, audio_analysis),
            "performance_analysis": self._analyze_performance(audio_analysis),
            "educational_insights": self._generate_educational_insights(audio_features, audio_analysis),
            "genre_analysis": genre_analysis,
            "emotion_analysis": emotion_analysis,
            "harmonic_analysis": harmonic_analysis
        }
        
        return analysis
    
    def _extract_basic_info(self, track_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extrae información básica de la canción"""
        artists = [artist["name"] for artist in track_info.get("artists", [])]
        
        return {
            "name": track_info.get("name", "Unknown"),
            "artists": artists,
            "album": track_info.get("album", {}).get("name", "Unknown"),
            "duration_ms": track_info.get("duration_ms", 0),
            "duration_seconds": track_info.get("duration_ms", 0) / 1000,
            "popularity": track_info.get("popularity", 0),
            "release_date": track_info.get("album", {}).get("release_date", "Unknown"),
            "external_urls": track_info.get("external_urls", {}),
            "preview_url": track_info.get("preview_url")
        }
    
    def _analyze_musical_elements(self, audio_features: Dict[str, Any], 
                                  audio_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza elementos musicales (notas, tonalidad, tempo, etc.)"""
        key = audio_features.get("key", -1)
        mode = audio_features.get("mode", 0)
        tempo = audio_features.get("tempo", 0)
        time_signature = audio_features.get("time_signature", 4)
        
        # Determinar nota y tonalidad
        note_name = self.NOTES[key] if key >= 0 else "Unknown"
        mode_name = self.MODES.get(mode, "Unknown")
        key_signature = f"{note_name} {mode_name.lower()}"
        
        # Análisis de secciones
        sections = audio_analysis.get("sections", [])
        key_changes = []
        tempo_changes = []
        
        for section in sections:
            section_key = section.get("key", -1)
            section_mode = section.get("mode", 0)
            section_tempo = section.get("tempo", 0)
            
            if section_key >= 0:
                key_changes.append({
                    "start": section.get("start", 0),
                    "key": self.NOTES[section_key],
                    "mode": self.MODES.get(section_mode, "Unknown"),
                    "confidence": section.get("key_confidence", 0)
                })
            
            if section_tempo > 0:
                tempo_changes.append({
                    "start": section.get("start", 0),
                    "tempo": section_tempo,
                    "confidence": section.get("tempo_confidence", 0)
                })
        
        # Análisis de beats y barras
        beats = audio_analysis.get("beats", [])
        bars = audio_analysis.get("bars", [])
        
        return {
            "key_signature": key_signature,
            "root_note": note_name,
            "mode": mode_name,
            "tempo": {
                "bpm": round(tempo, 2),
                "category": self._categorize_tempo(tempo)
            },
            "time_signature": f"{time_signature}/4",
            "key_changes": key_changes,
            "tempo_changes": tempo_changes,
            "structure": {
                "total_beats": len(beats),
                "total_bars": len(bars),
                "beats_per_bar": time_signature
            },
            "scale": self._identify_scale(key, mode)
        }
    
    def _analyze_technical_aspects(self, audio_features: Dict[str, Any],
                                   audio_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza aspectos técnicos del audio"""
        energy = audio_features.get("energy", 0)
        danceability = audio_features.get("danceability", 0)
        valence = audio_features.get("valence", 0)
        acousticness = audio_features.get("acousticness", 0)
        instrumentalness = audio_features.get("instrumentalness", 0)
        liveness = audio_features.get("liveness", 0)
        speechiness = audio_features.get("speechiness", 0)
        loudness = audio_features.get("loudness", 0)
        
        # Análisis de tatum, beats, sections
        tatums = audio_analysis.get("tatums", [])
        beats = audio_analysis.get("beats", [])
        sections = audio_analysis.get("sections", [])
        segments = audio_analysis.get("segments", [])
        
        return {
            "energy": {
                "value": round(energy, 3),
                "description": self._describe_energy(energy)
            },
            "danceability": {
                "value": round(danceability, 3),
                "description": self._describe_danceability(danceability)
            },
            "valence": {
                "value": round(valence, 3),
                "description": self._describe_valence(valence)
            },
            "acousticness": {
                "value": round(acousticness, 3),
                "description": "Acústico" if acousticness > 0.5 else "Electrónico"
            },
            "instrumentalness": {
                "value": round(instrumentalness, 3),
                "description": "Instrumental" if instrumentalness > 0.5 else "Con voces"
            },
            "liveness": {
                "value": round(liveness, 3),
                "description": "En vivo" if liveness > 0.8 else "Estudio"
            },
            "speechiness": {
                "value": round(speechiness, 3),
                "description": self._describe_speechiness(speechiness)
            },
            "loudness": {
                "value": round(loudness, 2),
                "unit": "dB",
                "description": self._describe_loudness(loudness)
            },
            "rhythm_structure": {
                "tatums": len(tatums),
                "beats": len(beats),
                "sections": len(sections),
                "segments": len(segments)
            }
        }
    
    def _analyze_composition(self, audio_features: Dict[str, Any],
                             audio_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza la composición musical"""
        sections = audio_analysis.get("sections", [])
        segments = audio_analysis.get("segments", [])
        
        # Identificar estructura (intro, verse, chorus, bridge, outro)
        section_types = []
        for section in sections:
            section_type = self._identify_section_type(section, sections)
            section_types.append({
                "start": section.get("start", 0),
                "duration": section.get("duration", 0),
                "type": section_type,
                "key": self.NOTES[section.get("key", -1)] if section.get("key", -1) >= 0 else "Unknown",
                "tempo": section.get("tempo", 0)
            })
        
        # Análisis de progresiones armónicas
        harmonic_progressions = []
        for i, segment in enumerate(segments[:50]):  # Limitar a primeros 50 segmentos
            pitch_data = segment.get("pitches", [])
            if pitch_data:
                dominant_pitch = pitch_data.index(max(pitch_data))
                harmonic_progressions.append({
                    "time": segment.get("start", 0),
                    "dominant_pitch": self.NOTES[dominant_pitch % 12],
                    "confidence": max(pitch_data)
                })
        
        return {
            "structure": section_types,
            "harmonic_progressions": harmonic_progressions[:20],  # Primeros 20
            "composition_style": self._identify_composition_style(audio_features),
            "complexity": self._assess_complexity(audio_features, audio_analysis)
        }
    
    def _analyze_performance(self, audio_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza aspectos de la interpretación"""
        segments = audio_analysis.get("segments", [])
        
        timbre_analysis = []
        for segment in segments[:30]:  # Primeros 30 segmentos
            timbre = segment.get("timbre", [])
            if timbre:
                timbre_analysis.append({
                    "time": segment.get("start", 0),
                    "brightness": timbre[0] if len(timbre) > 0 else 0,
                    "flatness": timbre[1] if len(timbre) > 1 else 0
                })
        
        return {
            "timbre_analysis": timbre_analysis,
            "dynamic_range": self._calculate_dynamic_range(segments),
            "performance_characteristics": self._identify_performance_characteristics(segments)
        }
    
    def _generate_educational_insights(self, audio_features: Dict[str, Any],
                                      audio_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Genera insights educativos sobre la música"""
        key = audio_features.get("key", -1)
        mode = audio_features.get("mode", 0)
        tempo = audio_features.get("tempo", 0)
        
        insights = {
            "key_analysis": {
                "note": self.NOTES[key] if key >= 0 else "Unknown",
                "mode": self.MODES.get(mode, "Unknown"),
                "scale_notes": self._get_scale_notes(key, mode),
                "common_chords": self._get_common_chords(key, mode)
            },
            "tempo_analysis": {
                "bpm": tempo,
                "category": self._categorize_tempo(tempo),
                "musical_style": self._suggest_musical_style(tempo, mode)
            },
            "learning_points": self._generate_learning_points(audio_features, audio_analysis),
            "practice_suggestions": self._generate_practice_suggestions(audio_features)
        }
        
        return insights
    
    # Métodos auxiliares
    
    def _categorize_tempo(self, tempo: float) -> str:
        """Categoriza el tempo"""
        if tempo < 60:
            return "Muy lento (Largo)"
        elif tempo < 72:
            return "Lento (Adagio)"
        elif tempo < 96:
            return "Moderado-lento (Andante)"
        elif tempo < 120:
            return "Moderado (Moderato)"
        elif tempo < 144:
            return "Moderado-rápido (Allegretto)"
        elif tempo < 168:
            return "Rápido (Allegro)"
        elif tempo < 200:
            return "Muy rápido (Presto)"
        else:
            return "Extremadamente rápido (Prestissimo)"
    
    def _identify_scale(self, key: int, mode: int) -> Dict[str, Any]:
        """Identifica la escala"""
        if key < 0:
            return {"name": "Unknown", "notes": []}
        
        scale_type = "major" if mode == 1 else "minor"
        scale_intervals = self.SCALES.get(scale_type, [])
        
        notes = [self.NOTES[(key + interval) % 12] for interval in scale_intervals]
        
        return {
            "name": f"{self.NOTES[key]} {scale_type}",
            "notes": notes,
            "intervals": scale_intervals
        }
    
    def _get_scale_notes(self, key: int, mode: int) -> List[str]:
        """Obtiene las notas de la escala"""
        if key < 0:
            return []
        scale_type = "major" if mode == 1 else "minor"
        scale_intervals = self.SCALES.get(scale_type, [])
        return [self.NOTES[(key + interval) % 12] for interval in scale_intervals]
    
    def _get_common_chords(self, key: int, mode: int) -> List[str]:
        """Obtiene acordes comunes en la tonalidad"""
        if key < 0:
            return []
        
        scale_notes = self._get_scale_notes(key, mode)
        
        # Acordes triada básicos
        if mode == 1:  # Major
            chords = [
                f"{scale_notes[0]}maj",  # I
                f"{scale_notes[1]}m",    # ii
                f"{scale_notes[2]}m",    # iii
                f"{scale_notes[3]}maj",  # IV
                f"{scale_notes[4]}maj",  # V
                f"{scale_notes[5]}m",    # vi
                f"{scale_notes[6]}dim"   # vii°
            ]
        else:  # Minor
            chords = [
                f"{scale_notes[0]}m",    # i
                f"{scale_notes[1]}dim",  # ii°
                f"{scale_notes[2]}maj",  # III
                f"{scale_notes[3]}m",    # iv
                f"{scale_notes[4]}m",    # v
                f"{scale_notes[5]}maj",  # VI
                f"{scale_notes[6]}maj"   # VII
            ]
        
        return chords
    
    def _describe_energy(self, energy: float) -> str:
        """Describe el nivel de energía"""
        if energy < 0.3:
            return "Muy baja energía"
        elif energy < 0.5:
            return "Baja energía"
        elif energy < 0.7:
            return "Energía moderada"
        elif energy < 0.9:
            return "Alta energía"
        else:
            return "Energía muy alta"
    
    def _describe_danceability(self, danceability: float) -> str:
        """Describe la bailabilidad"""
        if danceability < 0.3:
            return "No bailable"
        elif danceability < 0.5:
            return "Poco bailable"
        elif danceability < 0.7:
            return "Moderadamente bailable"
        elif danceability < 0.9:
            return "Muy bailable"
        else:
            return "Extremadamente bailable"
    
    def _describe_valence(self, valence: float) -> str:
        """Describe la valencia (positividad)"""
        if valence < 0.2:
            return "Muy triste/negativo"
        elif valence < 0.4:
            return "Triste"
        elif valence < 0.6:
            return "Neutral"
        elif valence < 0.8:
            return "Feliz"
        else:
            return "Muy feliz/positivo"
    
    def _describe_speechiness(self, speechiness: float) -> str:
        """Describe el contenido de habla"""
        if speechiness < 0.33:
            return "Música"
        elif speechiness < 0.66:
            return "Música y habla"
        else:
            return "Principalmente habla"
    
    def _describe_loudness(self, loudness: float) -> str:
        """Describe el nivel de volumen"""
        if loudness < -20:
            return "Muy silencioso"
        elif loudness < -10:
            return "Silencioso"
        elif loudness < -5:
            return "Moderado"
        elif loudness < 0:
            return "Fuerte"
        else:
            return "Muy fuerte"
    
    def _identify_section_type(self, section: Dict[str, Any], all_sections: List[Dict[str, Any]]) -> str:
        """Identifica el tipo de sección"""
        # Análisis básico basado en posición y características
        start = section.get("start", 0)
        duration = section.get("duration", 0)
        loudness = section.get("loudness", 0)
        
        # Lógica simple para identificar secciones
        if start < 10:
            return "Intro"
        elif loudness > -8:
            return "Chorus"
        elif duration < 20:
            return "Bridge"
        else:
            return "Verse"
    
    def _identify_composition_style(self, audio_features: Dict[str, Any]) -> str:
        """Identifica el estilo de composición"""
        energy = audio_features.get("energy", 0)
        danceability = audio_features.get("danceability", 0)
        acousticness = audio_features.get("acousticness", 0)
        
        if acousticness > 0.7:
            return "Acústico"
        elif energy > 0.7 and danceability > 0.7:
            return "Dance/Electronic"
        elif energy > 0.8:
            return "Rock/Metal"
        else:
            return "Pop/General"
    
    def _assess_complexity(self, audio_features: Dict[str, Any],
                          audio_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Evalúa la complejidad de la composición"""
        sections = audio_analysis.get("sections", [])
        segments = audio_analysis.get("segments", [])
        
        key_changes = len([s for s in sections if s.get("key", -1) >= 0])
        tempo_changes = len([s for s in sections if s.get("tempo", 0) > 0])
        
        complexity_score = (
            (len(sections) / 10) * 0.3 +
            (key_changes / 5) * 0.3 +
            (tempo_changes / 5) * 0.2 +
            (len(segments) / 100) * 0.2
        )
        
        if complexity_score < 0.3:
            level = "Simple"
        elif complexity_score < 0.6:
            level = "Moderada"
        elif complexity_score < 0.8:
            level = "Compleja"
        else:
            level = "Muy compleja"
        
        return {
            "level": level,
            "score": round(complexity_score, 2),
            "factors": {
                "sections": len(sections),
                "key_changes": key_changes,
                "tempo_changes": tempo_changes,
                "segments": len(segments)
            }
        }
    
    def _calculate_dynamic_range(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calcula el rango dinámico"""
        if not segments:
            return {"range": "Unknown", "description": "No hay datos suficientes"}
        
        loudnesses = [s.get("loudness_max", 0) for s in segments if s.get("loudness_max")]
        
        if not loudnesses:
            return {"range": "Unknown", "description": "No hay datos suficientes"}
        
        min_loud = min(loudnesses)
        max_loud = max(loudnesses)
        range_value = max_loud - min_loud
        
        if range_value < 5:
            description = "Rango dinámico limitado"
        elif range_value < 10:
            description = "Rango dinámico moderado"
        else:
            description = "Rango dinámico amplio"
        
        return {
            "min": round(min_loud, 2),
            "max": round(max_loud, 2),
            "range": round(range_value, 2),
            "description": description
        }
    
    def _identify_performance_characteristics(self, segments: List[Dict[str, Any]]) -> List[str]:
        """Identifica características de la interpretación"""
        characteristics = []
        
        if not segments:
            return characteristics
        
        # Analizar variaciones en timbre
        timbres = [s.get("timbre", []) for s in segments if s.get("timbre")]
        if timbres:
            brightness_values = [t[0] for t in timbres if len(t) > 0]
            if brightness_values:
                avg_brightness = sum(brightness_values) / len(brightness_values)
                if avg_brightness > 0:
                    characteristics.append("Timbre brillante")
                else:
                    characteristics.append("Timbre oscuro")
        
        return characteristics
    
    def _suggest_musical_style(self, tempo: float, mode: int) -> str:
        """Sugiere el estilo musical basado en tempo y modo"""
        if tempo < 60:
            return "Balada lenta"
        elif tempo < 90:
            return "Balada"
        elif tempo < 120:
            return mode == 1 and "Pop suave" or "Rock alternativo"
        elif tempo < 140:
            return "Pop/Rock"
        elif tempo < 180:
            return "Dance/Electronic"
        else:
            return "EDM/Hardcore"
    
    def _generate_learning_points(self, audio_features: Dict[str, Any],
                                 audio_analysis: Dict[str, Any]) -> List[str]:
        """Genera puntos de aprendizaje"""
        points = []
        
        key = audio_features.get("key", -1)
        mode = audio_features.get("mode", 0)
        tempo = audio_features.get("tempo", 0)
        
        if key >= 0:
            note = self.NOTES[key]
            mode_name = self.MODES.get(mode, "Unknown")
            points.append(f"La canción está en {note} {mode_name.lower()}")
            points.append(f"Escala principal: {', '.join(self._get_scale_notes(key, mode))}")
            points.append(f"Acordes comunes: {', '.join(self._get_common_chords(key, mode)[:5])}")
        
        points.append(f"Tempo: {round(tempo, 0)} BPM ({self._categorize_tempo(tempo)})")
        
        time_sig = audio_features.get("time_signature", 4)
        points.append(f"Compás: {time_sig}/4")
        
        return points
    
    def _generate_practice_suggestions(self, audio_features: Dict[str, Any]) -> List[str]:
        """Genera sugerencias de práctica"""
        suggestions = []
        
        tempo = audio_features.get("tempo", 0)
        key = audio_features.get("key", -1)
        mode = audio_features.get("mode", 0)
        
        if key >= 0:
            scale_notes = self._get_scale_notes(key, mode)
            suggestions.append(f"Practica la escala de {self.NOTES[key]} {self.MODES.get(mode, 'Unknown').lower()}: {', '.join(scale_notes)}")
            
            chords = self._get_common_chords(key, mode)
            suggestions.append(f"Practica estos acordes: {', '.join(chords[:4])}")
        
        if tempo > 0:
            suggestions.append(f"Practica con metrónomo a {round(tempo * 0.75, 0)} BPM primero, luego aumenta gradualmente")
        
        suggestions.append("Escucha la canción varias veces para identificar la estructura")
        suggestions.append("Intenta tocar la melodía principal en tu instrumento")
        
        return suggestions

