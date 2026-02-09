"""
Transcription Service
Uses Whisper for audio transcription with timestamp support
"""

import os
import logging
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any
from uuid import UUID

from ..config.settings import get_settings
from ..core.models import TranscriptionSegment

logger = logging.getLogger(__name__)


class TranscriptionService:
    """Service for transcribing audio to text with timestamps"""
    
    def __init__(self):
        self.settings = get_settings()
        self._model = None
        self._model_loaded = False
    
    def _load_model(self):
        """Lazy load Whisper model"""
        if not self._model_loaded:
            try:
                import whisper
                logger.info(f"Loading Whisper model: {self.settings.whisper_model}")
                self._model = whisper.load_model(self.settings.whisper_model)
                self._model_loaded = True
                logger.info("Whisper model loaded successfully")
            except ImportError:
                logger.error("Whisper not installed. Install with: pip install openai-whisper")
                raise
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                raise
    
    async def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        include_timestamps: bool = True,
    ) -> Dict[str, Any]:
        """
        Transcribe audio file to text
        
        Args:
            audio_path: Path to audio file
            language: Language code (None for auto-detect)
            include_timestamps: Whether to include word/segment timestamps
            
        Returns:
            Dict with transcription results
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        logger.info(f"Transcribing: {audio_path}")
        
        def _transcribe():
            self._load_model()
            
            options = {
                'language': language or self.settings.whisper_language,
                'task': 'transcribe',
                'verbose': False,
            }
            
            # Remove None values
            options = {k: v for k, v in options.items() if v is not None}
            
            result = self._model.transcribe(str(audio_path), **options)
            return result
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _transcribe)
        
        # Process segments
        segments = []
        for i, seg in enumerate(result.get('segments', [])):
            segment = TranscriptionSegment(
                id=i,
                start_time=seg['start'],
                end_time=seg['end'],
                text=seg['text'].strip(),
                confidence=seg.get('avg_logprob'),
            )
            segments.append(segment)
        
        # Build full text
        full_text = result.get('text', '').strip()
        
        # Build text with timestamps
        full_text_with_timestamps = self._build_timestamped_text(segments)
        
        return {
            'full_text': full_text,
            'full_text_with_timestamps': full_text_with_timestamps,
            'segments': segments,
            'language_detected': result.get('language', 'unknown'),
        }
    
    def _build_timestamped_text(self, segments: List[TranscriptionSegment]) -> str:
        """Build text with timestamps from segments"""
        lines = []
        for segment in segments:
            timestamp = segment.formatted_timestamp
            lines.append(f"{timestamp} {segment.text}")
        return '\n'.join(lines)
    
    async def transcribe_with_chunks(
        self,
        audio_path: Path,
        chunk_duration: Optional[int] = None,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Transcribe long audio in chunks
        
        Args:
            audio_path: Path to audio file
            chunk_duration: Duration of each chunk in seconds
            language: Language code
            
        Returns:
            Combined transcription results
        """
        chunk_duration = chunk_duration or self.settings.chunk_duration
        
        # For now, just use regular transcription
        # Whisper handles long audio well
        return await self.transcribe(audio_path, language, include_timestamps=True)
    
    def format_timestamp(self, seconds: float) -> str:
        """Format seconds to timestamp string"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"
    
    def segments_to_srt(self, segments: List[TranscriptionSegment]) -> str:
        """Convert segments to SRT subtitle format"""
        srt_lines = []
        for i, segment in enumerate(segments, 1):
            start = self._format_srt_time(segment.start_time)
            end = self._format_srt_time(segment.end_time)
            srt_lines.append(f"{i}")
            srt_lines.append(f"{start} --> {end}")
            srt_lines.append(segment.text)
            srt_lines.append("")
        return '\n'.join(srt_lines)
    
    def _format_srt_time(self, seconds: float) -> str:
        """Format seconds to SRT time format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def segments_to_vtt(self, segments: List[TranscriptionSegment]) -> str:
        """Convert segments to WebVTT subtitle format"""
        vtt_lines = ["WEBVTT", ""]
        for segment in segments:
            start = self._format_vtt_time(segment.start_time)
            end = self._format_vtt_time(segment.end_time)
            vtt_lines.append(f"{start} --> {end}")
            vtt_lines.append(segment.text)
            vtt_lines.append("")
        return '\n'.join(vtt_lines)
    
    def _format_vtt_time(self, seconds: float) -> str:
        """Format seconds to VTT time format (HH:MM:SS.mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


_transcription_service: Optional[TranscriptionService] = None


def get_transcription_service() -> TranscriptionService:
    """Get transcription service singleton"""
    global _transcription_service
    if _transcription_service is None:
        _transcription_service = TranscriptionService()
    return _transcription_service












