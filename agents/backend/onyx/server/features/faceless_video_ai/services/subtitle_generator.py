"""
Subtitle Generator Service
Generates and embeds subtitles in video
"""

import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SubtitleGenerator:
    """Generates subtitles from script segments"""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("/tmp/faceless_video/subtitles")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_subtitles(
        self,
        segments: List[Dict[str, Any]],
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate subtitle entries from segments
        
        Args:
            segments: Script segments with timing
            config: Subtitle configuration
            
        Returns:
            List of subtitle entries with text, timing, and styling
        """
        if not config.get("enabled", True):
            logger.info("Subtitles disabled")
            return []

        logger.info(f"Generating subtitles for {len(segments)} segments")
        
        subtitles = []
        for segment in segments:
            text = segment.get("text", "")
            start_time = segment.get("start_time", 0.0)
            end_time = segment.get("end_time", 0.0)
            
            # Split long text into multiple subtitle lines
            subtitle_lines = self._split_text_for_subtitles(
                text,
                max_chars=config.get("max_chars_per_line", 42)
            )
            
            # Create subtitle entries
            for i, line in enumerate(subtitle_lines):
                # Distribute time across lines
                line_duration = (end_time - start_time) / len(subtitle_lines)
                line_start = start_time + (i * line_duration)
                line_end = line_start + line_duration
                
                subtitle = {
                    "index": len(subtitles),
                    "text": line,
                    "start_time": line_start,
                    "end_time": line_end,
                    "duration": line_end - line_start,
                    "style": config.get("style", "modern"),
                    "font_size": config.get("font_size", 48),
                    "font_color": config.get("font_color", "#FFFFFF"),
                    "background_color": config.get("background_color"),
                    "position": config.get("position", "bottom"),
                    "animation": config.get("animation", True),
                    "fade_in": config.get("fade_in", True),
                    "fade_out": config.get("fade_out", True),
                }
                subtitles.append(subtitle)
        
        logger.info(f"Generated {len(subtitles)} subtitle entries")
        return subtitles

    def _split_text_for_subtitles(self, text: str, max_chars: int = 42) -> List[str]:
        """Split text into subtitle lines respecting max characters"""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0

        for word in words:
            word_length = len(word) + 1  # +1 for space
            
            if current_length + word_length > max_chars and current_line:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)
            else:
                current_line.append(word)
                current_length += word_length

        if current_line:
            lines.append(" ".join(current_line))

        return lines

    def export_srt(self, subtitles: List[Dict[str, Any]], output_path: Path) -> Path:
        """Export subtitles to SRT format"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for subtitle in subtitles:
                index = subtitle["index"] + 1
                start = self._format_timestamp(subtitle["start_time"])
                end = self._format_timestamp(subtitle["end_time"])
                text = subtitle["text"]
                
                f.write(f"{index}\n")
                f.write(f"{start} --> {end}\n")
                f.write(f"{text}\n\n")
        
        logger.info(f"Exported SRT file: {output_path}")
        return output_path

    def export_vtt(self, subtitles: List[Dict[str, Any]], output_path: Path) -> Path:
        """Export subtitles to WebVTT format"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n\n")
            
            for subtitle in subtitles:
                start = self._format_timestamp(subtitle["start_time"], vtt=True)
                end = self._format_timestamp(subtitle["end_time"], vtt=True)
                text = subtitle["text"]
                
                f.write(f"{start} --> {end}\n")
                f.write(f"{text}\n\n")
        
        logger.info(f"Exported VTT file: {output_path}")
        return output_path

    def _format_timestamp(self, seconds: float, vtt: bool = False) -> str:
        """Format seconds to SRT/VTT timestamp format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        
        if vtt:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
        else:
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def create_subtitle_overlay_data(
        self,
        subtitles: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Create subtitle overlay data for video composition with enhanced styles
        
        Returns:
            List of subtitle overlay configurations
        """
        overlay_data = []
        
        for subtitle in subtitles:
            style = subtitle.get("style", "modern")
            style_config = self._get_style_config(style)
            
            overlay_data.append({
                "text": subtitle["text"],
                "start_time": subtitle["start_time"],
                "end_time": subtitle["end_time"],
                "x": "center",
                "y": self._get_y_position(subtitle["position"]),
                "font_size": subtitle["font_size"],
                "font_color": subtitle["font_color"],
                "background_color": subtitle.get("background_color") or style_config.get("background_color"),
                "border_color": style_config.get("border_color"),
                "border_width": style_config.get("border_width", 0),
                "shadow_color": style_config.get("shadow_color"),
                "shadow_offset": style_config.get("shadow_offset", 0),
                "animation": subtitle["animation"],
                "fade_in": subtitle["fade_in"],
                "fade_out": subtitle["fade_out"],
                "style": style,
            })
        
        return overlay_data
    
    def _get_style_config(self, style: str) -> Dict[str, Any]:
        """Get style configuration for subtitle style"""
        styles = {
            "simple": {
                "background_color": None,
                "border_color": None,
                "border_width": 0,
                "shadow_color": None,
                "shadow_offset": 0,
            },
            "modern": {
                "background_color": "#00000080",
                "border_color": None,
                "border_width": 0,
                "shadow_color": "#000000",
                "shadow_offset": 2,
            },
            "bold": {
                "background_color": "#000000CC",
                "border_color": "#FFFFFF",
                "border_width": 2,
                "shadow_color": "#000000",
                "shadow_offset": 3,
            },
            "elegant": {
                "background_color": "#1a1a1aB3",
                "border_color": "#FFD700",
                "border_width": 1,
                "shadow_color": "#000000",
                "shadow_offset": 2,
            },
            "minimal": {
                "background_color": None,
                "border_color": None,
                "border_width": 0,
                "shadow_color": "#000000",
                "shadow_offset": 1,
            },
            "neon": {
                "background_color": "#000000CC",
                "border_color": "#00FFFF",
                "border_width": 3,
                "shadow_color": "#00FFFF",
                "shadow_offset": 5,
            },
            "glass": {
                "background_color": "#FFFFFF40",
                "border_color": "#FFFFFF",
                "border_width": 1,
                "shadow_color": "#000000",
                "shadow_offset": 2,
            },
            "outline": {
                "background_color": None,
                "border_color": "#000000",
                "border_width": 4,
                "shadow_color": None,
                "shadow_offset": 0,
            },
            "shadow": {
                "background_color": "#00000080",
                "border_color": None,
                "border_width": 0,
                "shadow_color": "#000000",
                "shadow_offset": 5,
            },
        }
        
        return styles.get(style, styles["modern"])

    def _get_y_position(self, position: str) -> str:
        """Get Y position string for subtitle placement"""
        positions = {
            "top": "10%",
            "center": "50%",
            "bottom": "90%",
        }
        return positions.get(position, "90%")

