"""
Transcription Service
Transcribes audio to text
"""

from typing import Optional, Dict, Any, List
from pathlib import Path
import logging
import os

logger = logging.getLogger(__name__)


class TranscriptionService:
    """Transcribes audio to text"""
    
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.whisper_available = False
        
        # Check if Whisper is available
        try:
            import whisper
            self.whisper_available = True
            logger.info("Whisper available for transcription")
        except ImportError:
            logger.warning("Whisper not installed, using OpenAI API")
    
    async def transcribe_audio(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        model: str = "base"
    ) -> Dict[str, Any]:
        """
        Transcribe audio to text
        
        Args:
            audio_path: Path to audio file
            language: Language code (optional, auto-detect if None)
            model: Whisper model size (tiny, base, small, medium, large)
            
        Returns:
            Transcription result with text and segments
        """
        # Try OpenAI Whisper API first
        if self.openai_api_key:
            try:
                return await self._transcribe_with_openai(audio_path, language)
            except Exception as e:
                logger.warning(f"OpenAI transcription failed: {str(e)}")
        
        # Fallback to local Whisper
        if self.whisper_available:
            try:
                return await self._transcribe_with_whisper(audio_path, language, model)
            except Exception as e:
                logger.warning(f"Whisper transcription failed: {str(e)}")
        
        # Fallback: return placeholder
        logger.warning("No transcription service available")
        return {
            "text": "",
            "language": language or "unknown",
            "segments": [],
        }
    
    async def _transcribe_with_openai(
        self,
        audio_path: Path,
        language: Optional[str]
    ) -> Dict[str, Any]:
        """Transcribe using OpenAI Whisper API"""
        import httpx
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            with open(audio_path, "rb") as f:
                files = {"file": f}
                data = {"model": "whisper-1"}
                if language:
                    data["language"] = language
                
                response = await client.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {self.openai_api_key}"},
                    files=files,
                    data=data
                )
                response.raise_for_status()
                result = response.json()
                
                return {
                    "text": result.get("text", ""),
                    "language": result.get("language", language or "unknown"),
                    "segments": [],  # OpenAI API doesn't return segments
                }
    
    async def _transcribe_with_whisper(
        self,
        audio_path: Path,
        language: Optional[str],
        model: str
    ) -> Dict[str, Any]:
        """Transcribe using local Whisper"""
        import whisper
        
        # Load model
        whisper_model = whisper.load_model(model)
        
        # Transcribe
        result = whisper_model.transcribe(
            str(audio_path),
            language=language
        )
        
        return {
            "text": result["text"],
            "language": result.get("language", language or "unknown"),
            "segments": [
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                }
                for seg in result.get("segments", [])
            ],
        }
    
    async def transcribe_video(
        self,
        video_path: Path,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe video by extracting audio first
        
        Args:
            video_path: Path to video file
            language: Language code
            
        Returns:
            Transcription result
        """
        # Extract audio from video
        audio_path = Path("/tmp") / f"audio_{video_path.stem}.wav"
        
        import asyncio
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            "-y",
            str(audio_path)
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            
            if process.returncode == 0 and audio_path.exists():
                # Transcribe audio
                result = await self.transcribe_audio(audio_path, language)
                
                # Cleanup
                audio_path.unlink()
                
                return result
        except Exception as e:
            logger.error(f"Video transcription failed: {str(e)}")
        
        return {
            "text": "",
            "language": language or "unknown",
            "segments": [],
        }


_transcription_service: Optional[TranscriptionService] = None


def get_transcription_service() -> TranscriptionService:
    """Get transcription service instance (singleton)"""
    global _transcription_service
    if _transcription_service is None:
        _transcription_service = TranscriptionService()
    return _transcription_service

