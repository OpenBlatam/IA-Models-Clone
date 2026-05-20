"""
Servicio para exportar análisis musical
"""

import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ExportService:
    """Servicio para exportar análisis a diferentes formatos"""
    
    def __init__(self):
        self.logger = logger
    
    def export_to_json(self, analysis: Dict[str, Any], 
                      include_coaching: bool = True) -> str:
        """Exporta análisis a JSON"""
        export_data = {
            "export_date": datetime.now().isoformat(),
            "analysis": analysis
        }
        
        if include_coaching and "coaching" in analysis:
            export_data["coaching"] = analysis["coaching"]
        
        return json.dumps(export_data, indent=2, ensure_ascii=False)
    
    def export_to_text(self, analysis: Dict[str, Any],
                      include_coaching: bool = True) -> str:
        """Exporta análisis a texto plano"""
        lines = []
        lines.append("=" * 60)
        lines.append("ANÁLISIS MUSICAL")
        lines.append("=" * 60)
        lines.append("")
        
        # Información básica
        basic_info = analysis.get("track_basic_info", {})
        lines.append(f"Canción: {basic_info.get('name', 'Unknown')}")
        lines.append(f"Artista(s): {', '.join(basic_info.get('artists', []))}")
        lines.append(f"Álbum: {basic_info.get('album', 'Unknown')}")
        lines.append(f"Duración: {basic_info.get('duration_seconds', 0):.1f} segundos")
        lines.append("")
        
        # Análisis musical
        musical = analysis.get("musical_analysis", {})
        lines.append("-" * 60)
        lines.append("ANÁLISIS MUSICAL")
        lines.append("-" * 60)
        lines.append(f"Tonalidad: {musical.get('key_signature', 'Unknown')}")
        lines.append(f"Tempo: {musical.get('tempo', {}).get('bpm', 0):.1f} BPM")
        lines.append(f"Compás: {musical.get('time_signature', 'Unknown')}")
        lines.append(f"Escala: {musical.get('scale', {}).get('name', 'Unknown')}")
        lines.append("")
        
        # Análisis técnico
        technical = analysis.get("technical_analysis", {})
        lines.append("-" * 60)
        lines.append("ANÁLISIS TÉCNICO")
        lines.append("-" * 60)
        lines.append(f"Energía: {technical.get('energy', {}).get('value', 0):.2f}")
        lines.append(f"Bailabilidad: {technical.get('danceability', {}).get('value', 0):.2f}")
        lines.append(f"Valencia: {technical.get('valence', {}).get('value', 0):.2f}")
        lines.append("")
        
        # Insights educativos
        educational = analysis.get("educational_insights", {})
        if educational:
            lines.append("-" * 60)
            lines.append("INSIGHTS EDUCATIVOS")
            lines.append("-" * 60)
            for point in educational.get("learning_points", []):
                lines.append(f"• {point}")
            lines.append("")
        
        # Coaching
        if include_coaching and "coaching" in analysis:
            coaching = analysis["coaching"]
            lines.append("-" * 60)
            lines.append("COACHING MUSICAL")
            lines.append("-" * 60)
            lines.append(f"Dificultad: {coaching.get('overview', {}).get('difficulty_level', 'Unknown')}")
            lines.append("")
            lines.append("Ruta de Aprendizaje:")
            for step in coaching.get("learning_path", []):
                lines.append(f"  {step.get('step')}. {step.get('title')}")
                lines.append(f"     {step.get('description')}")
                lines.append(f"     Duración: {step.get('duration')}")
                lines.append("")
        
        lines.append("=" * 60)
        lines.append(f"Generado el: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def export_to_markdown(self, analysis: Dict[str, Any],
                          include_coaching: bool = True) -> str:
        """Exporta análisis a Markdown"""
        lines = []
        lines.append("# Análisis Musical")
        lines.append("")
        
        # Información básica
        basic_info = analysis.get("track_basic_info", {})
        lines.append("## Información de la Canción")
        lines.append("")
        lines.append(f"- **Canción**: {basic_info.get('name', 'Unknown')}")
        lines.append(f"- **Artista(s)**: {', '.join(basic_info.get('artists', []))}")
        lines.append(f"- **Álbum**: {basic_info.get('album', 'Unknown')}")
        lines.append(f"- **Duración**: {basic_info.get('duration_seconds', 0):.1f} segundos")
        lines.append("")
        
        # Análisis musical
        musical = analysis.get("musical_analysis", {})
        lines.append("## Análisis Musical")
        lines.append("")
        lines.append(f"- **Tonalidad**: {musical.get('key_signature', 'Unknown')}")
        lines.append(f"- **Tempo**: {musical.get('tempo', {}).get('bpm', 0):.1f} BPM")
        lines.append(f"- **Compás**: {musical.get('time_signature', 'Unknown')}")
        lines.append(f"- **Escala**: {musical.get('scale', {}).get('name', 'Unknown')}")
        lines.append("")
        
        # Análisis técnico
        technical = analysis.get("technical_analysis", {})
        lines.append("## Análisis Técnico")
        lines.append("")
        lines.append(f"- **Energía**: {technical.get('energy', {}).get('value', 0):.2f}")
        lines.append(f"- **Bailabilidad**: {technical.get('danceability', {}).get('value', 0):.2f}")
        lines.append(f"- **Valencia**: {technical.get('valence', {}).get('value', 0):.2f}")
        lines.append("")
        
        # Insights educativos
        educational = analysis.get("educational_insights", {})
        if educational:
            lines.append("## Insights Educativos")
            lines.append("")
            for point in educational.get("learning_points", []):
                lines.append(f"- {point}")
            lines.append("")
        
        # Coaching
        if include_coaching and "coaching" in analysis:
            coaching = analysis["coaching"]
            lines.append("## Coaching Musical")
            lines.append("")
            lines.append(f"**Dificultad**: {coaching.get('overview', {}).get('difficulty_level', 'Unknown')}")
            lines.append("")
            lines.append("### Ruta de Aprendizaje")
            lines.append("")
            for step in coaching.get("learning_path", []):
                lines.append(f"#### {step.get('step')}. {step.get('title')}")
                lines.append("")
                lines.append(f"{step.get('description')}")
                lines.append("")
                lines.append(f"**Duración**: {step.get('duration')}")
                lines.append("")
        
        lines.append("---")
        lines.append(f"*Generado el {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        
        return "\n".join(lines)
    
    def save_to_file(self, content: str, filename: str, 
                    output_dir: Optional[Path] = None) -> Path:
        """Guarda el contenido en un archivo"""
        if output_dir is None:
            output_dir = Path("./exports")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / filename
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        self.logger.info(f"Archivo exportado: {file_path}")
        return file_path
    
    def export_to_csv(self, analyses: List[Dict[str, Any]]) -> str:
        """Exporta múltiples análisis a CSV"""
        import csv
        from io import StringIO
        
        output = StringIO()
        writer = csv.writer(output)
        
        # Headers
        writer.writerow([
            "Track Name", "Artists", "Genre", "Emotion", "Tempo", "Energy",
            "Danceability", "Valence", "Key", "Mode", "Popularity"
        ])
        
        # Data rows
        for analysis in analyses:
            basic_info = analysis.get("track_basic_info", {})
            musical = analysis.get("musical_analysis", {})
            technical = analysis.get("technical_analysis", {})
            genre = analysis.get("genre_analysis", {}).get("primary_genre", "Unknown")
            emotion = analysis.get("emotion_analysis", {}).get("primary_emotion", "Unknown")
            
            writer.writerow([
                basic_info.get("name", "Unknown"),
                ", ".join(basic_info.get("artists", [])),
                genre,
                emotion,
                musical.get("tempo", {}).get("bpm", 0),
                technical.get("energy", {}).get("value", 0),
                technical.get("danceability", {}).get("value", 0),
                technical.get("valence", {}).get("value", 0),
                musical.get("root_note", "Unknown"),
                musical.get("mode", "Unknown"),
                basic_info.get("popularity", 0)
            ])
        
        return output.getvalue()
    
    def export_comprehensive_report(self, analysis: Dict[str, Any]) -> str:
        """Exporta un reporte comprehensivo en Markdown"""
        lines = []
        lines.append("# Reporte Comprehensivo de Análisis Musical\n")
        lines.append(f"**Fecha de Exportación:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Información básica
        basic_info = analysis.get("track_basic_info", {})
        lines.append("## Información Básica\n")
        lines.append(f"- **Canción:** {basic_info.get('name', 'Unknown')}")
        lines.append(f"- **Artista(s):** {', '.join(basic_info.get('artists', []))}")
        lines.append(f"- **Álbum:** {basic_info.get('album', 'Unknown')}")
        lines.append(f"- **Duración:** {basic_info.get('duration_seconds', 0):.1f} segundos\n")
        
        # Análisis musical
        musical = analysis.get("musical_analysis", {})
        lines.append("## Análisis Musical\n")
        lines.append(f"- **Tonalidad:** {musical.get('key_signature', 'Unknown')}")
        lines.append(f"- **Tempo:** {musical.get('tempo', {}).get('bpm', 0)} BPM ({musical.get('tempo', {}).get('category', 'Unknown')})")
        lines.append(f"- **Compás:** {musical.get('time_signature', 'Unknown')}\n")
        
        # Análisis técnico
        technical = analysis.get("technical_analysis", {})
        lines.append("## Análisis Técnico\n")
        lines.append(f"- **Energía:** {technical.get('energy', {}).get('value', 0):.3f} ({technical.get('energy', {}).get('description', 'Unknown')})")
        lines.append(f"- **Bailabilidad:** {technical.get('danceability', {}).get('value', 0):.3f}")
        lines.append(f"- **Valencia:** {technical.get('valence', {}).get('value', 0):.3f}\n")
        
        # Género y emoción
        genre = analysis.get("genre_analysis", {})
        emotion = analysis.get("emotion_analysis", {})
        lines.append("## Clasificación\n")
        lines.append(f"- **Género Principal:** {genre.get('primary_genre', 'Unknown')} (Confianza: {genre.get('confidence', 0):.2%})")
        lines.append(f"- **Emoción Principal:** {emotion.get('primary_emotion', 'Unknown')} (Confianza: {emotion.get('confidence', 0):.2%})\n")
        
        # Insights educativos
        insights = analysis.get("educational_insights", {})
        if insights:
            lines.append("## Insights Educativos\n")
            learning_points = insights.get("learning_points", [])
            for point in learning_points:
                lines.append(f"- {point}")
            lines.append("")
        
        return "\n".join(lines)

