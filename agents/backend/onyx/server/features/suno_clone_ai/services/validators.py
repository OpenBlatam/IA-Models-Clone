import logging
from typing import Optional

logger = logging.getLogger(__name__)


class GenerationValidator:
    @staticmethod
    def validate_prompt(prompt: str) -> None:
        if not prompt:
            raise ValueError("Prompt must be at least 3 characters")
        prompt_stripped = prompt.strip()
        if len(prompt_stripped) < 3:
            raise ValueError("Prompt must be at least 3 characters")
    
    @staticmethod
    def validate_duration(duration: int) -> None:
        if duration < 5 or duration > 600:
            raise ValueError("Duration must be between 5 and 600 seconds")
    
    @staticmethod
    def validate_guidance_scale(guidance_scale: float) -> None:
        if guidance_scale < 1.0 or guidance_scale > 10.0:
            raise ValueError("Guidance scale must be between 1.0 and 10.0")
    
    @staticmethod
    def validate_temperature(temperature: float) -> None:
        if temperature < 0.1 or temperature > 2.0:
            raise ValueError("Temperature must be between 0.1 and 2.0")
    
    @staticmethod
    def validate_sample_rate(sample_rate: Optional[int]) -> int:
        if sample_rate is None:
            return 44100
        if sample_rate in (16000, 22050, 32000, 44100, 48000):
            return sample_rate
        logger.warning(f"Invalid sample rate {sample_rate}, using 44100")
        return 44100
    
    @staticmethod
    def validate_num_variants(num_variants: int) -> None:
        if num_variants < 1 or num_variants > 10:
            raise ValueError("num_variants must be between 1 and 10")
    
    @staticmethod
    def validate_voice_volume(voice_volume: float) -> None:
        if voice_volume < 0.0 or voice_volume > 1.0:
            raise ValueError("voice_volume must be between 0.0 and 1.0")
    
    @staticmethod
    def validate_lyrics(lyrics: str) -> None:
        if not lyrics:
            raise ValueError("Lyrics cannot be empty")
        if not lyrics.strip():
            raise ValueError("Lyrics cannot be empty")

