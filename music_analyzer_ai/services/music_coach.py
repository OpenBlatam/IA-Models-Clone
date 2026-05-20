"""
Servicio de coaching musical
"""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class MusicCoach:
    """Coach musical que proporciona recomendaciones y guías educativas"""
    
    def __init__(self):
        self.logger = logger
    
    def generate_coaching_analysis(self, music_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Genera un análisis completo de coaching"""
        return {
            "overview": self._generate_overview(music_analysis),
            "technical_breakdown": self._generate_technical_breakdown(music_analysis),
            "learning_path": self._generate_learning_path(music_analysis),
            "practice_exercises": self._generate_practice_exercises(music_analysis),
            "composition_insights": self._generate_composition_insights(music_analysis),
            "performance_tips": self._generate_performance_tips(music_analysis),
            "recommendations": self._generate_recommendations(music_analysis)
        }
    
    def _generate_overview(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Genera un resumen general"""
        musical = analysis.get("musical_analysis", {})
        technical = analysis.get("technical_analysis", {})
        basic_info = analysis.get("track_basic_info", {})
        
        return {
            "summary": f"Análisis de '{basic_info.get('name', 'Unknown')}'",
            "key_findings": [
                f"Tonalidad: {musical.get('key_signature', 'Unknown')}",
                f"Tempo: {musical.get('tempo', {}).get('bpm', 0)} BPM",
                f"Compás: {musical.get('time_signature', 'Unknown')}",
                f"Energía: {technical.get('energy', {}).get('description', 'Unknown')}",
                f"Estilo: {analysis.get('composition_analysis', {}).get('composition_style', 'Unknown')}"
            ],
            "difficulty_level": self._assess_difficulty(analysis),
            "suitable_for": self._suggest_skill_level(analysis)
        }
    
    def _generate_technical_breakdown(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Genera un desglose técnico detallado"""
        musical = analysis.get("musical_analysis", {})
        technical = analysis.get("technical_analysis", {})
        composition = analysis.get("composition_analysis", {})
        
        return {
            "harmony": {
                "key": musical.get("key_signature", "Unknown"),
                "scale": musical.get("scale", {}).get("name", "Unknown"),
                "scale_notes": musical.get("scale", {}).get("notes", []),
                "chord_progressions": self._analyze_chord_progressions(composition)
            },
            "rhythm": {
                "tempo": musical.get("tempo", {}),
                "time_signature": musical.get("time_signature", "Unknown"),
                "rhythm_pattern": self._analyze_rhythm_pattern(technical)
            },
            "structure": {
                "sections": composition.get("structure", []),
                "complexity": composition.get("complexity", {})
            },
            "dynamics": {
                "energy": technical.get("energy", {}),
                "loudness": technical.get("loudness", {}),
                "dynamic_range": analysis.get("performance_analysis", {}).get("dynamic_range", {})
            }
        }
    
    def _generate_learning_path(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Genera una ruta de aprendizaje"""
        musical = analysis.get("musical_analysis", {})
        educational = analysis.get("educational_insights", {})
        
        steps = [
            {
                "step": 1,
                "title": "Familiarización",
                "description": "Escucha la canción varias veces para familiarizarte con la melodía y estructura",
                "duration": "15-30 minutos"
            },
            {
                "step": 2,
                "title": "Identificar la tonalidad",
                "description": f"La canción está en {musical.get('key_signature', 'Unknown')}. Practica la escala: {', '.join(educational.get('key_analysis', {}).get('scale_notes', []))}",
                "duration": "20-30 minutos"
            },
            {
                "step": 3,
                "title": "Aprender los acordes",
                "description": f"Practica los acordes principales: {', '.join(educational.get('key_analysis', {}).get('common_chords', [])[:4])}",
                "duration": "30-45 minutos"
            },
            {
                "step": 4,
                "title": "Trabajar el tempo",
                "description": f"Practica con metrónomo empezando a {int(musical.get('tempo', {}).get('bpm', 120) * 0.7)} BPM y aumenta gradualmente",
                "duration": "20-30 minutos"
            },
            {
                "step": 5,
                "title": "Tocar la melodía",
                "description": "Intenta tocar la melodía principal siguiendo la estructura de la canción",
                "duration": "30-60 minutos"
            },
            {
                "step": 6,
                "title": "Refinamiento",
                "description": "Trabaja en la dinámica, expresión y detalles de interpretación",
                "duration": "Ongoing"
            }
        ]
        
        return steps
    
    def _generate_practice_exercises(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Genera ejercicios de práctica"""
        musical = analysis.get("musical_analysis", {})
        educational = analysis.get("educational_insights", {})
        
        exercises = []
        
        # Ejercicio de escala
        scale_notes = educational.get("key_analysis", {}).get("scale_notes", [])
        if scale_notes:
            exercises.append({
                "type": "Scale Practice",
                "title": f"Practicar escala de {musical.get('key_signature', 'Unknown')}",
                "description": f"Toca la escala: {', '.join(scale_notes)} en diferentes octavas",
                "tempo": int(musical.get("tempo", {}).get("bpm", 120) * 0.7),
                "repetitions": 5
            })
        
        # Ejercicio de acordes
        chords = educational.get("key_analysis", {}).get("common_chords", [])
        if chords:
            exercises.append({
                "type": "Chord Practice",
                "title": "Progresión de acordes básicos",
                "description": f"Practica la progresión: {', '.join(chords[:4])}",
                "tempo": int(musical.get("tempo", {}).get("bpm", 120) * 0.8),
                "repetitions": 10
            })
        
        # Ejercicio de ritmo
        time_sig = musical.get("time_signature", "4/4")
        exercises.append({
            "type": "Rhythm Practice",
            "title": f"Practicar compás {time_sig}",
            "description": f"Practica patrones rítmicos en compás {time_sig}",
            "tempo": int(musical.get("tempo", {}).get("bpm", 120) * 0.75),
            "repetitions": 8
        })
        
        # Ejercicio de melodía
        exercises.append({
            "type": "Melody Practice",
            "title": "Tocar la melodía principal",
            "description": "Toca la melodía principal de la canción lentamente",
            "tempo": int(musical.get("tempo", {}).get("bpm", 120) * 0.6),
            "repetitions": 5
        })
        
        return exercises
    
    def _generate_composition_insights(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Genera insights sobre la composición"""
        composition = analysis.get("composition_analysis", {})
        musical = analysis.get("musical_analysis", {})
        
        return {
            "structure_analysis": {
                "sections": composition.get("structure", []),
                "total_sections": len(composition.get("structure", [])),
                "common_pattern": self._identify_structure_pattern(composition)
            },
            "harmonic_analysis": {
                "key_changes": len(musical.get("key_changes", [])),
                "modulation": "Sí" if len(musical.get("key_changes", [])) > 1 else "No",
                "harmonic_complexity": composition.get("complexity", {}).get("level", "Unknown")
            },
            "composition_techniques": self._identify_composition_techniques(composition, musical)
        }
    
    def _generate_performance_tips(self, analysis: Dict[str, Any]) -> List[str]:
        """Genera tips de interpretación"""
        technical = analysis.get("technical_analysis", {})
        performance = analysis.get("performance_analysis", {})
        musical = analysis.get("musical_analysis", {})
        
        tips = []
        
        # Tips basados en energía
        energy = technical.get("energy", {}).get("value", 0.5)
        if energy > 0.7:
            tips.append("Esta canción requiere mucha energía. Calienta bien antes de tocar.")
        elif energy < 0.4:
            tips.append("Mantén un tono suave y controlado para esta pieza.")
        
        # Tips basados en tempo
        tempo = musical.get("tempo", {}).get("bpm", 120)
        if tempo > 140:
            tips.append("El tempo es rápido. Practica lentamente primero y aumenta gradualmente.")
        elif tempo < 80:
            tips.append("El tempo es lento. Enfócate en mantener el tiempo constante.")
        
        # Tips de dinámica
        dynamic_range = performance.get("dynamic_range", {})
        if dynamic_range.get("range", 0) > 10:
            tips.append("Hay un amplio rango dinámico. Practica los cambios de volumen.")
        
        # Tips generales
        tips.append("Escucha la canción original varias veces para captar el estilo y la expresión.")
        tips.append("Graba tu interpretación y compárala con la original.")
        tips.append("Practica sección por sección antes de tocar la canción completa.")
        
        return tips
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Genera recomendaciones generales"""
        composition = analysis.get("composition_analysis", {})
        technical = analysis.get("technical_analysis", {})
        
        recommendations = {
            "for_beginners": [],
            "for_intermediate": [],
            "for_advanced": [],
            "general": []
        }
        
        difficulty = self._assess_difficulty(analysis)
        
        if difficulty == "Beginner":
            recommendations["for_beginners"].append("Esta canción es adecuada para principiantes.")
            recommendations["for_beginners"].append("Enfócate en aprender la melodía principal primero.")
        elif difficulty == "Intermediate":
            recommendations["for_intermediate"].append("Esta canción requiere habilidades intermedias.")
            recommendations["for_intermediate"].append("Trabaja en la técnica y la expresión.")
        else:
            recommendations["for_advanced"].append("Esta canción es desafiante y requiere habilidades avanzadas.")
            recommendations["for_advanced"].append("Enfócate en los detalles técnicos y la interpretación.")
        
        # Recomendaciones basadas en complejidad
        complexity = composition.get("complexity", {}).get("level", "Moderada")
        if complexity in ["Compleja", "Muy compleja"]:
            recommendations["general"].append("La composición es compleja. Estudia la estructura primero.")
        
        # Recomendaciones basadas en estilo
        style = composition.get("composition_style", "")
        if style == "Acústico":
            recommendations["general"].append("Enfócate en la técnica acústica y el toque.")
        elif "Electronic" in style:
            recommendations["general"].append("Presta atención a los elementos rítmicos y la sincronización.")
        
        return recommendations
    
    # Métodos auxiliares
    
    def _assess_difficulty(self, analysis: Dict[str, Any]) -> str:
        """Evalúa la dificultad de la canción"""
        composition = analysis.get("composition_analysis", {})
        musical = analysis.get("musical_analysis", {})
        technical = analysis.get("technical_analysis", {})
        
        complexity = composition.get("complexity", {}).get("level", "Moderada")
        key_changes = len(musical.get("key_changes", []))
        tempo = musical.get("tempo", {}).get("bpm", 120)
        time_sig = musical.get("time_signature", "4/4")
        
        score = 0
        
        # Complejidad
        if complexity == "Simple":
            score += 1
        elif complexity == "Moderada":
            score += 2
        elif complexity == "Compleja":
            score += 3
        else:
            score += 4
        
        # Cambios de tonalidad
        if key_changes > 2:
            score += 2
        elif key_changes > 0:
            score += 1
        
        # Tempo
        if tempo > 160:
            score += 2
        elif tempo > 140:
            score += 1
        elif tempo < 70:
            score += 1
        
        # Compás
        if time_sig != "4/4":
            score += 1
        
        if score <= 2:
            return "Beginner"
        elif score <= 4:
            return "Intermediate"
        else:
            return "Advanced"
    
    def _suggest_skill_level(self, analysis: Dict[str, Any]) -> List[str]:
        """Sugiere niveles de habilidad apropiados"""
        difficulty = self._assess_difficulty(analysis)
        
        if difficulty == "Beginner":
            return ["Principiante", "Principiante-Intermedio"]
        elif difficulty == "Intermediate":
            return ["Intermedio", "Principiante-Intermedio", "Intermedio-Avanzado"]
        else:
            return ["Avanzado", "Intermedio-Avanzado"]
    
    def _analyze_chord_progressions(self, composition: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza las progresiones de acordes"""
        progressions = composition.get("harmonic_progressions", [])
        
        if not progressions:
            return {"analysis": "No hay datos suficientes de progresiones armónicas"}
        
        # Contar notas dominantes
        note_counts = {}
        for prog in progressions:
            note = prog.get("dominant_pitch", "Unknown")
            note_counts[note] = note_counts.get(note, 0) + 1
        
        most_common = max(note_counts.items(), key=lambda x: x[1]) if note_counts else ("Unknown", 0)
        
        return {
            "total_changes": len(progressions),
            "most_common_note": most_common[0],
            "note_distribution": note_counts
        }
    
    def _analyze_rhythm_pattern(self, technical: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza el patrón rítmico"""
        rhythm = technical.get("rhythm_structure", {})
        
        return {
            "beats_per_section": rhythm.get("beats", 0) / max(rhythm.get("sections", 1), 1),
            "segments_per_beat": rhythm.get("segments", 0) / max(rhythm.get("beats", 1), 1),
            "complexity": "Simple" if rhythm.get("beats", 0) < 100 else "Moderada" if rhythm.get("beats", 0) < 300 else "Compleja"
        }
    
    def _identify_structure_pattern(self, composition: Dict[str, Any]) -> str:
        """Identifica el patrón de estructura"""
        sections = composition.get("structure", [])
        
        if len(sections) < 3:
            return "Estructura simple"
        
        section_types = [s.get("type", "Unknown") for s in sections]
        
        # Patrones comunes
        if "Intro" in section_types and "Chorus" in section_types and "Verse" in section_types:
            return "Estructura estándar (Intro-Verse-Chorus)"
        elif section_types.count("Verse") > section_types.count("Chorus"):
            return "Estructura basada en versos"
        else:
            return "Estructura variada"
    
    def _identify_composition_techniques(self, composition: Dict[str, Any],
                                        musical: Dict[str, Any]) -> List[str]:
        """Identifica técnicas de composición"""
        techniques = []
        
        # Modulación
        if len(musical.get("key_changes", [])) > 1:
            techniques.append("Modulación de tonalidad")
        
        # Cambios de tempo
        if len(musical.get("tempo_changes", [])) > 1:
            techniques.append("Cambios de tempo")
        
        # Complejidad
        complexity = composition.get("complexity", {}).get("level", "Moderada")
        if complexity in ["Compleja", "Muy compleja"]:
            techniques.append("Composición compleja")
        
        # Estructura
        sections = composition.get("structure", [])
        if len(sections) > 6:
            techniques.append("Estructura extensa")
        
        return techniques


