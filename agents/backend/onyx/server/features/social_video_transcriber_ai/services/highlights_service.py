"""
Highlights Service for Social Video Transcriber AI
Detects key moments and highlights in transcriptions
"""

import json
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

from ..config.settings import get_settings
from ..core.models import TranscriptionSegment
from .openrouter_client import get_openrouter_client

logger = logging.getLogger(__name__)


class HighlightType(str, Enum):
    """Types of highlights"""
    HOOK = "hook"
    KEY_POINT = "key_point"
    QUOTE = "quote"
    CALL_TO_ACTION = "call_to_action"
    EMOTIONAL_PEAK = "emotional_peak"
    FUNNY_MOMENT = "funny_moment"
    INSIGHT = "insight"
    STORY_CLIMAX = "story_climax"
    QUESTION = "question"
    REVELATION = "revelation"


@dataclass
class Highlight:
    """A detected highlight in the transcription"""
    id: int
    highlight_type: HighlightType
    text: str
    start_time: float
    end_time: float
    importance_score: float  # 0.0 - 1.0
    reason: str
    clip_worthy: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.highlight_type.value,
            "text": self.text,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.end_time - self.start_time,
            "importance_score": self.importance_score,
            "reason": self.reason,
            "clip_worthy": self.clip_worthy,
            "formatted_time": f"{int(self.start_time // 60):02d}:{int(self.start_time % 60):02d}",
        }


@dataclass
class HighlightsSummary:
    """Summary of all highlights"""
    total_highlights: int
    clip_worthy_count: int
    average_importance: float
    highlights_by_type: Dict[str, int]
    best_clip_start: Optional[float]
    best_clip_end: Optional[float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_highlights": self.total_highlights,
            "clip_worthy_count": self.clip_worthy_count,
            "average_importance": round(self.average_importance, 2),
            "highlights_by_type": self.highlights_by_type,
            "best_clip": {
                "start": self.best_clip_start,
                "end": self.best_clip_end,
            } if self.best_clip_start is not None else None,
        }


class HighlightsService:
    """Service for detecting highlights in transcriptions"""
    
    DETECT_HIGHLIGHTS_PROMPT = """Analiza la siguiente transcripción y detecta los momentos más destacados.

TRANSCRIPCIÓN CON TIMESTAMPS:
{text}

CRITERIOS DE DETECCIÓN:
1. Hook inicial: ¿Hay un gancho que capture la atención?
2. Puntos clave: Información importante o valiosa
3. Citas memorables: Frases que podrían ser compartidas
4. Llamadas a la acción: Momentos donde se pide algo al espectador
5. Picos emocionales: Momentos de emoción o intensidad
6. Momentos graciosos: Si el contenido tiene humor
7. Insights: Revelaciones o ideas profundas
8. Clímax de historia: Si hay narrativa, el punto culminante
9. Preguntas importantes: Preguntas retóricas o relevantes
10. Revelaciones: Información sorprendente o inesperada

Para cada highlight detectado (máximo 10), indica:
- El tipo de highlight
- El texto exacto
- Los timestamps (inicio y fin)
- Puntuación de importancia (0.0-1.0)
- Si es digno de clip (material que funcionaría como video corto independiente)
- Razón por la que es destacable

Responde SOLO en JSON válido:
{{
    "highlights": [
        {{
            "type": "hook",
            "text": "texto del highlight",
            "start_time": 0.0,
            "end_time": 5.0,
            "importance_score": 0.95,
            "clip_worthy": true,
            "reason": "razón por la que es destacable"
        }}
    ],
    "best_clip": {{
        "start": 10.0,
        "end": 30.0,
        "reason": "mejor fragmento para clip viral"
    }}
}}"""

    SUGGEST_CLIPS_PROMPT = """Basándote en la transcripción, sugiere los mejores fragmentos para crear clips cortos virales.

TRANSCRIPCIÓN:
{text}

CRITERIOS PARA CLIPS:
1. Duración ideal: 15-60 segundos
2. Debe ser auto-contenido (entendible sin contexto)
3. Debe tener un gancho inicial
4. Preferiblemente con un cierre satisfactorio
5. Alto potencial de engagement

Sugiere 3-5 clips con:
- Timestamps de inicio y fin
- Título sugerido para el clip
- Plataforma ideal (TikTok, Reels, Shorts)
- Puntuación de viralidad potencial

Responde en JSON:
{{
    "clips": [
        {{
            "start_time": 10.0,
            "end_time": 40.0,
            "duration": 30.0,
            "title": "título sugerido",
            "platform": "tiktok",
            "virality_score": 0.85,
            "hook": "texto del gancho inicial",
            "reason": "por qué funcionaría"
        }}
    ]
}}"""

    def __init__(self):
        self.settings = get_settings()
        self.client = get_openrouter_client()
    
    async def detect_highlights(
        self,
        text_with_timestamps: str,
        max_highlights: int = 10,
    ) -> List[Highlight]:
        """
        Detect highlights in a transcription
        
        Args:
            text_with_timestamps: Transcription with timestamps
            max_highlights: Maximum number of highlights to return
            
        Returns:
            List of Highlight objects
        """
        if not text_with_timestamps or len(text_with_timestamps) < 50:
            return []
        
        logger.info("Detecting highlights")
        
        try:
            response = await self.client.complete(
                prompt=self.DETECT_HIGHLIGHTS_PROMPT.format(text=text_with_timestamps),
                system_prompt="Eres un experto en contenido viral y detección de momentos destacados. Analiza transcripciones para encontrar los mejores fragmentos.",
                max_tokens=2500,
                temperature=0.4,
            )
            
            data = self._parse_json(response)
            highlights = []
            
            for i, h in enumerate(data.get("highlights", [])[:max_highlights]):
                try:
                    highlight_type = HighlightType(h.get("type", "key_point"))
                except ValueError:
                    highlight_type = HighlightType.KEY_POINT
                
                highlights.append(Highlight(
                    id=i,
                    highlight_type=highlight_type,
                    text=h.get("text", ""),
                    start_time=float(h.get("start_time", 0)),
                    end_time=float(h.get("end_time", 0)),
                    importance_score=float(h.get("importance_score", 0.5)),
                    reason=h.get("reason", ""),
                    clip_worthy=h.get("clip_worthy", False),
                ))
            
            highlights.sort(key=lambda h: h.importance_score, reverse=True)
            
            logger.info(f"Detected {len(highlights)} highlights")
            return highlights
            
        except Exception as e:
            logger.error(f"Highlight detection failed: {e}")
            return []
    
    async def detect_highlights_from_segments(
        self,
        segments: List[TranscriptionSegment],
        max_highlights: int = 10,
    ) -> List[Highlight]:
        """
        Detect highlights from transcription segments
        
        Args:
            segments: List of transcription segments
            max_highlights: Maximum highlights
            
        Returns:
            List of Highlight objects
        """
        text_with_timestamps = "\n".join(
            f"{s.formatted_timestamp} {s.text}"
            for s in segments
        )
        
        return await self.detect_highlights(text_with_timestamps, max_highlights)
    
    async def suggest_clips(
        self,
        text_with_timestamps: str,
        max_clips: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Suggest video clips from transcription
        
        Args:
            text_with_timestamps: Transcription with timestamps
            max_clips: Maximum number of clip suggestions
            
        Returns:
            List of clip suggestions
        """
        if not text_with_timestamps or len(text_with_timestamps) < 100:
            return []
        
        logger.info("Suggesting clips")
        
        try:
            response = await self.client.complete(
                prompt=self.SUGGEST_CLIPS_PROMPT.format(text=text_with_timestamps),
                system_prompt="Eres un experto en contenido viral para redes sociales. Identifica los mejores fragmentos para clips cortos.",
                max_tokens=1500,
                temperature=0.5,
            )
            
            data = self._parse_json(response)
            clips = data.get("clips", [])[:max_clips]
            
            for clip in clips:
                clip["formatted_start"] = f"{int(clip.get('start_time', 0) // 60):02d}:{int(clip.get('start_time', 0) % 60):02d}"
                clip["formatted_end"] = f"{int(clip.get('end_time', 0) // 60):02d}:{int(clip.get('end_time', 0) % 60):02d}"
            
            return clips
            
        except Exception as e:
            logger.error(f"Clip suggestion failed: {e}")
            return []
    
    def get_highlights_summary(self, highlights: List[Highlight]) -> HighlightsSummary:
        """
        Generate summary of highlights
        
        Args:
            highlights: List of detected highlights
            
        Returns:
            HighlightsSummary object
        """
        if not highlights:
            return HighlightsSummary(
                total_highlights=0,
                clip_worthy_count=0,
                average_importance=0,
                highlights_by_type={},
                best_clip_start=None,
                best_clip_end=None,
            )
        
        by_type = {}
        for h in highlights:
            type_name = h.highlight_type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1
        
        clip_worthy = [h for h in highlights if h.clip_worthy]
        best_clip = max(clip_worthy, key=lambda h: h.importance_score) if clip_worthy else None
        
        return HighlightsSummary(
            total_highlights=len(highlights),
            clip_worthy_count=len(clip_worthy),
            average_importance=sum(h.importance_score for h in highlights) / len(highlights),
            highlights_by_type=by_type,
            best_clip_start=best_clip.start_time if best_clip else None,
            best_clip_end=best_clip.end_time if best_clip else None,
        )
    
    def _parse_json(self, response: str) -> Dict[str, Any]:
        """Parse JSON from response"""
        response = response.strip()
        
        if response.startswith('```'):
            lines = response.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            response = '\n'.join(lines)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except:
                    pass
            return {"highlights": []}


_highlights_service: Optional[HighlightsService] = None


def get_highlights_service() -> HighlightsService:
    """Get highlights service singleton"""
    global _highlights_service
    if _highlights_service is None:
        _highlights_service = HighlightsService()
    return _highlights_service












