"""
Audio Generation Providers (TTS)
Supports multiple text-to-speech services
"""

import asyncio
import os
from abc import ABC, abstractmethod
from typing import Optional
from pathlib import Path
import logging
import httpx

logger = logging.getLogger(__name__)


class AudioProvider(ABC):
    """Base class for audio generation providers"""
    
    @abstractmethod
    async def generate_speech(
        self,
        text: str,
        voice: str,
        speed: float,
        language: str
    ) -> Path:
        """Generate speech from text"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available"""
        pass


class OpenAITTSProvider(AudioProvider):
    """OpenAI TTS provider"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = "https://api.openai.com/v1"
        
        # Voice mapping
        self.voice_map = {
            "male_1": "alloy",
            "male_2": "echo",
            "female_1": "nova",
            "female_2": "shimmer",
            "neutral": "alloy",
        }
    
    def is_available(self) -> bool:
        return self.api_key is not None
    
    async def generate_speech(
        self,
        text: str,
        voice: str,
        speed: float,
        language: str
    ) -> Path:
        """Generate speech using OpenAI TTS"""
        if not self.is_available():
            raise ValueError("OpenAI API key not configured")
        
        openai_voice = self.voice_map.get(voice, "alloy")
        
        # Clamp speed to OpenAI's supported range (0.25 to 4.0)
        clamped_speed = max(0.25, min(4.0, speed))
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.base_url}/audio/speech",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "tts-1",
                        "input": text,
                        "voice": openai_voice,
                        "speed": clamped_speed,
                    }
                )
                response.raise_for_status()
                
                # Save audio
                output_dir = Path("/tmp/faceless_video/audio")
                output_dir.mkdir(parents=True, exist_ok=True)
                audio_path = output_dir / f"openai_tts_{hash(text) % 100000}.mp3"
                
                with open(audio_path, "wb") as f:
                    f.write(response.content)
                
                logger.info(f"Generated audio with OpenAI TTS: {audio_path}")
                return audio_path
                
        except Exception as e:
            logger.error(f"OpenAI TTS generation failed: {str(e)}")
            raise


class GoogleTTSProvider(AudioProvider):
    """Google Text-to-Speech provider (gTTS)"""
    
    def __init__(self):
        self.available = True  # gTTS doesn't need API key
    
    def is_available(self) -> bool:
        return self.available
    
    async def generate_speech(
        self,
        text: str,
        voice: str,
        speed: float,
        language: str
    ) -> Path:
        """Generate speech using Google TTS"""
        try:
            from gtts import gTTS
            import io
            
            # Create TTS
            tts = gTTS(text=text, lang=language, slow=False)
            
            # Save to bytes
            audio_bytes = io.BytesIO()
            tts.write_to_fp(audio_bytes)
            audio_bytes.seek(0)
            
            # Save file
            output_dir = Path("/tmp/faceless_video/audio")
            output_dir.mkdir(parents=True, exist_ok=True)
            audio_path = output_dir / f"gtts_{hash(text) % 100000}.mp3"
            
            with open(audio_path, "wb") as f:
                f.write(audio_bytes.read())
            
            logger.info(f"Generated audio with Google TTS: {audio_path}")
            return audio_path
            
        except ImportError:
            logger.error("gTTS not installed. Install with: pip install gtts")
            raise
        except Exception as e:
            logger.error(f"Google TTS generation failed: {str(e)}")
            raise


class ElevenLabsProvider(AudioProvider):
    """ElevenLabs TTS provider"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        self.base_url = "https://api.elevenlabs.io/v1"
    
    def is_available(self) -> bool:
        return self.api_key is not None
    
    async def generate_speech(
        self,
        text: str,
        voice: str,
        speed: float,
        language: str
    ) -> Path:
        """Generate speech using ElevenLabs TTS"""
        if not self.is_available():
            raise ValueError("ElevenLabs API key not configured")
        
        # Get default voice ID (you can customize this)
        voice_id = "21m00Tcm4TlvDq8ikWAM"  # Default voice
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.base_url}/text-to-speech/{voice_id}",
                    headers={
                        "xi-api-key": self.api_key,
                        "Content-Type": "application/json",
                    },
                    json={
                        "text": text,
                        "model_id": "eleven_monolingual_v1",
                        "voice_settings": {
                            "stability": 0.5,
                            "similarity_boost": 0.75,
                            "speed": speed,
                        }
                    }
                )
                response.raise_for_status()
                
                # Save audio
                output_dir = Path("/tmp/faceless_video/audio")
                output_dir.mkdir(parents=True, exist_ok=True)
                audio_path = output_dir / f"elevenlabs_{hash(text) % 100000}.mp3"
                
                with open(audio_path, "wb") as f:
                    f.write(response.content)
                
                logger.info(f"Generated audio with ElevenLabs: {audio_path}")
                return audio_path
                
        except Exception as e:
            logger.error(f"ElevenLabs TTS generation failed: {str(e)}")
            raise


class PlaceholderAudioProvider(AudioProvider):
    """Placeholder audio provider (fallback)"""
    
    def is_available(self) -> bool:
        return True
    
    async def generate_speech(
        self,
        text: str,
        voice: str,
        speed: float,
        language: str
    ) -> Path:
        """Generate placeholder audio file"""
        output_dir = Path("/tmp/faceless_video/audio")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create empty/silent audio file
        audio_path = output_dir / f"placeholder_{hash(text) % 100000}.mp3"
        
        # Create a minimal silent MP3 (1 second)
        # In production, use pydub or similar to create actual audio
        with open(audio_path, "wb") as f:
            # Minimal MP3 header (silent)
            f.write(b'\xff\xfb\x90\x00')  # MP3 header
            f.write(b'\x00' * 1000)  # Silent data
        
        logger.debug(f"Generated placeholder audio: {audio_path}")
        return audio_path


def get_audio_provider() -> AudioProvider:
    """Get the best available audio provider"""
    # Try OpenAI TTS first
    openai_provider = OpenAITTSProvider()
    if openai_provider.is_available():
        return openai_provider
    
    # Try ElevenLabs
    elevenlabs_provider = ElevenLabsProvider()
    if elevenlabs_provider.is_available():
        return elevenlabs_provider
    
    # Try Google TTS (free, no API key needed)
    try:
        google_provider = GoogleTTSProvider()
        if google_provider.is_available():
            return google_provider
    except:
        pass
    
    # Fallback to placeholder
    logger.warning("No TTS provider configured, using placeholder")
    return PlaceholderAudioProvider()

