"""
Audio Generator Service
Generates text-to-speech audio from script
"""

import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

from .ai_providers.audio_providers import get_audio_provider, AudioProvider

logger = logging.getLogger(__name__)


class AudioGenerator:
    """Generates audio from text using TTS"""

    def __init__(self, output_dir: Optional[str] = None, audio_provider: Optional[AudioProvider] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("/tmp/faceless_video/audio")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.audio_provider = audio_provider or get_audio_provider()
        self.max_retries = 3
        self.retry_delay = 2.0

    async def generate_audio(
        self,
        script_text: str,
        segments: List[Dict[str, Any]],
        voice: str = "neutral",
        speed: float = 1.0,
        pitch: float = 1.0,
        language: str = "es"
    ) -> Dict[str, Any]:
        """
        Generate audio from script text
        
        Args:
            script_text: Full script text
            segments: Script segments with timing
            voice: Voice type (male_1, female_1, neutral, etc.)
            speed: Speech speed multiplier
            pitch: Pitch adjustment
            language: Language code
            
        Returns:
            Dictionary with audio file path and metadata
        """
        logger.info(f"Generating audio with voice '{voice}', speed {speed}, language '{language}'")
        
        # Generate audio for full script
        audio_path = await self._generate_tts(
            text=script_text,
            voice=voice,
            speed=speed,
            pitch=pitch,
            language=language
        )
        
        # Calculate total duration
        total_duration = segments[-1].get("end_time", 0.0) if segments else 0.0
        
        result = {
            "audio_path": str(audio_path),
            "duration": total_duration,
            "voice": voice,
            "speed": speed,
            "pitch": pitch,
            "language": language,
            "format": "mp3",
        }
        
        logger.info(f"Audio generated: {audio_path} ({total_duration:.2f}s)")
        return result

    async def _generate_tts(
        self,
        text: str,
        voice: str,
        speed: float,
        pitch: float,
        language: str
    ) -> Path:
        """
        Generate TTS audio file using configured audio provider
        """
        # Split long text into chunks if needed (some TTS services have limits)
        max_chunk_length = 5000  # Characters
        if len(text) > max_chunk_length:
            return await self._generate_long_audio(text, voice, speed, pitch, language)
        
        # Generate with retry logic
        audio_path = None
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                audio_path = await self.audio_provider.generate_speech(
                    text=text,
                    voice=voice,
                    speed=speed,
                    language=language
                )
                break  # Success
            except Exception as e:
                last_error = e
                logger.warning(f"TTS generation attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
        
        if audio_path is None:
            logger.error(f"Failed to generate audio after {self.max_retries} attempts")
            raise RuntimeError(f"TTS generation failed: {str(last_error)}")
        
        # Verify audio file exists
        if not audio_path.exists():
            raise FileNotFoundError(f"Generated audio not found: {audio_path}")
        
        logger.debug(f"Generated audio: {audio_path}")
        return audio_path
    
    async def _generate_long_audio(
        self,
        text: str,
        voice: str,
        speed: float,
        pitch: float,
        language: str
    ) -> Path:
        """Generate audio for long text by splitting into chunks"""
        import subprocess
        from pathlib import Path
        
        # Split text into chunks
        chunks = [text[i:i+5000] for i in range(0, len(text), 5000)]
        chunk_files = []
        
        # Generate audio for each chunk
        for i, chunk in enumerate(chunks):
            chunk_audio = await self.audio_provider.generate_speech(
                text=chunk,
                voice=voice,
                speed=speed,
                language=language
            )
            chunk_files.append(chunk_audio)
        
        # Concatenate audio files using ffmpeg
        concat_file = self.output_dir / "concat_list.txt"
        with open(concat_file, 'w') as f:
            for chunk_file in chunk_files:
                f.write(f"file '{chunk_file.absolute()}'\n")
        
        output_path = self.output_dir / f"combined_{hash(text) % 100000}.mp3"
        
        # Use ffmpeg to concatenate
        cmd = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c", "copy",
            "-y",
            str(output_path)
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            
            if process.returncode != 0:
                raise RuntimeError("Failed to concatenate audio chunks")
            
            return output_path
        except FileNotFoundError:
            raise RuntimeError("FFmpeg not found. Required for long audio generation.")

    async def add_background_music(
        self,
        audio_path: Path,
        music_style: Optional[str] = None,
        volume: float = 0.3
    ) -> Path:
        """
        Add background music to audio using music library
        
        Args:
            audio_path: Path to main audio file
            music_style: Style of background music
            volume: Music volume (0.0 to 1.0)
            
        Returns:
            Path to audio file with background music
        """
        try:
            from .music_library import get_music_library, MusicStyle
            import asyncio
            
            music_lib = get_music_library()
            
            # Find appropriate track
            if music_style:
                try:
                    style_enum = MusicStyle(music_style)
                    tracks = music_lib.get_tracks_by_style(style_enum)
                    if tracks:
                        track = tracks[0]  # Use first available
                    else:
                        track = music_lib.find_track()
                except ValueError:
                    track = music_lib.find_track()
            else:
                track = music_lib.find_track()
            
            if not track or not Path(track.file_path).exists():
                logger.warning("Music track not found, returning original audio")
                return audio_path
            
            # Mix audio using ffmpeg
            output_path = self.output_dir / f"with_music_{audio_path.stem}.mp3"
            
            cmd = [
                "ffmpeg",
                "-i", str(audio_path),
                "-i", track.file_path,
                "-filter_complex", f"[1:a]volume={volume}[music];[0:a][music]amix=inputs=2:duration=first:dropout_transition=2",
                "-c:a", "libmp3lame",
                "-y",
                str(output_path)
            ]
            
            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await process.communicate()
                
                if process.returncode == 0 and output_path.exists():
                    logger.info(f"Added background music: {output_path}")
                    return output_path
                else:
                    logger.warning("Music mixing failed, returning original audio")
                    return audio_path
            except FileNotFoundError:
                logger.warning("FFmpeg not found, returning original audio")
                return audio_path
                
        except Exception as e:
            logger.warning(f"Failed to add background music: {str(e)}, returning original audio")
            return audio_path

