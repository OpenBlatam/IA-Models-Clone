import logging
import numpy as np
from typing import Optional, Dict, Any, List
from pathlib import Path

from .model_loader import ModelLoader
from .audio_processor import AudioProcessor
from .audio_processors import MasteringProcessor, EffectsProcessor
from .validators import GenerationValidator
from .processing_pipeline import ProcessingPipeline
from .variant_generator import VariantGenerator

logger = logging.getLogger(__name__)


class RealisticMusicGenerator:
    def __init__(
        self,
        model_name: str = "facebook/musicgen-medium",
        use_post_processing: bool = True,
        use_noise_reduction: bool = True,
        fast_mode: bool = False,
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.use_post_processing = use_post_processing
        self.use_noise_reduction = use_noise_reduction
        self.fast_mode = fast_mode
        
        self.model_loader = ModelLoader(model_name, device)
        self.audio_processor = AudioProcessor(fast_mode)
        self.mastering_processor = MasteringProcessor(fast_mode)
        self.effects_processor = EffectsProcessor() if use_post_processing else None
        self.validator = GenerationValidator()
        self.pipeline = ProcessingPipeline(
            self.audio_processor,
            self.mastering_processor,
            self.effects_processor,
            fast_mode
        )
        self.tts = self._init_tts()
    
    def _init_tts(self):
        try:
            from TTS.api import TTS
            return TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        except (ImportError, Exception) as e:
            logger.warning(f"TTS not available: {e}")
            return None
    
    def generate(
        self,
        prompt: str,
        duration: int = 30,
        sample_rate: Optional[int] = None,
        guidance_scale: float = 3.5,
        temperature: float = 0.9
    ) -> np.ndarray:
        self.validator.validate_prompt(prompt)
        self.validator.validate_duration(duration)
        self.validator.validate_guidance_scale(guidance_scale)
        self.validator.validate_temperature(temperature)
        
        if sample_rate is None:
            sample_rate = 32000 if self.fast_mode else 44100
        else:
            sample_rate = self.validator.validate_sample_rate(sample_rate)
        
        try:
            audio = self.model_loader.generate_audio(
                prompt, duration, guidance_scale, temperature, self.fast_mode
            )
        except Exception as e:
            logger.error(f"Error generating audio: {e}", exc_info=True)
            raise
        
        if audio.ndim > 1:
            if audio.shape[0] == 1:
                audio = audio[0]
            else:
                audio = np.mean(audio, axis=0, dtype=np.float32)
        
        if not np.all(np.isfinite(audio)):
            logger.warning("Generated audio contains NaN or Inf, cleaning...")
            np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0, out=audio)
        
        try:
            audio = self.pipeline.process(
                audio, sample_rate, duration, self.use_noise_reduction
            )
            logger.info(f"Generated music: {duration}s")
            return audio
        except Exception as e:
            logger.error(f"Error in post-processing: {e}", exc_info=True)
            raise
    
    async def generate_async(
        self,
        prompt: str,
        duration: int = 30,
        sample_rate: Optional[int] = None,
        guidance_scale: float = 3.5,
        temperature: float = 0.9
    ) -> np.ndarray:
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.generate, prompt, duration, sample_rate, guidance_scale, temperature
        )
    
    def generate_variants(
        self,
        prompt: str,
        num_variants: int = 4,
        duration: int = 30,
        sample_rate: Optional[int] = None,
        base_guidance_scale: float = 3.5,
        base_temperature: float = 0.9
    ) -> List[Dict[str, Any]]:
        self.validator.validate_num_variants(num_variants)
        return VariantGenerator.generate_variants_sync(
            self.generate,
            prompt,
            num_variants,
            duration,
            sample_rate,
            base_guidance_scale,
            base_temperature
        )
    
    async def generate_variants_async(
        self,
        prompt: str,
        num_variants: int = 4,
        duration: int = 30,
        sample_rate: Optional[int] = None,
        base_guidance_scale: float = 3.5,
        base_temperature: float = 0.9,
        max_concurrent: int = 2
    ) -> List[Dict[str, Any]]:
        self.validator.validate_num_variants(num_variants)
        return await VariantGenerator.generate_variants_async(
            self.generate_async,
            prompt,
            num_variants,
            duration,
            sample_rate,
            base_guidance_scale,
            base_temperature,
            max_concurrent
        )
    
    def add_voice(
        self,
        audio: np.ndarray,
        lyrics: str,
        reference_voice_path: Optional[str] = None,
        sample_rate: int = 32000,
        voice_volume: float = 0.7
    ) -> np.ndarray:
        self.validator.validate_lyrics(lyrics)
        self.validator.validate_voice_volume(voice_volume)
        
        if not self.tts:
            logger.warning("TTS not available")
            return audio
        
        try:
            if reference_voice_path and Path(reference_voice_path).exists():
                voice_audio = self.tts.tts(
                    text=lyrics, speaker_wav=reference_voice_path, language="en"
                )
            else:
                voice_audio = self.tts.tts(text=lyrics, language="en")
            
            import torch
            if isinstance(voice_audio, torch.Tensor):
                voice_audio = voice_audio.cpu().numpy()
                if voice_audio.dtype != np.float32:
                    voice_audio = voice_audio.astype(np.float32, copy=False)
            elif not isinstance(voice_audio, np.ndarray):
                voice_audio = np.array(voice_audio, dtype=np.float32)
            elif voice_audio.dtype != np.float32:
                voice_audio = voice_audio.astype(np.float32, copy=False)
            
            if voice_audio.ndim > 1:
                if voice_audio.shape[0] == 1:
                    voice_audio = voice_audio[0]
                else:
                    voice_audio = np.mean(voice_audio, axis=0, dtype=np.float32)
            
            target_length = len(audio) if audio.ndim == 1 else audio.shape[1]
            if len(voice_audio) > target_length:
                voice_audio = voice_audio[:target_length]
            elif len(voice_audio) < target_length:
                result = np.zeros(target_length, dtype=np.float32)
                result[:len(voice_audio)] = voice_audio
                voice_audio = result
            
            if audio.ndim > 1:
                voice_stereo = np.stack([voice_audio, voice_audio], axis=0)
                mixed = np.empty_like(audio, dtype=np.float32)
                np.multiply(audio, 1 - voice_volume, out=mixed)
                np.add(mixed, voice_stereo * voice_volume, out=mixed)
            else:
                mixed = np.empty_like(audio, dtype=np.float32)
                np.multiply(audio, 1 - voice_volume, out=mixed)
                np.add(mixed, voice_audio * voice_volume, out=mixed)
            
            return self.audio_processor.normalize(mixed)
        except Exception as e:
            logger.error(f"Error adding voice: {e}", exc_info=True)
            return audio
    
    def enhance_quality(
        self, audio: np.ndarray, sample_rate: int = 32000
    ) -> np.ndarray:
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32, copy=False)
        
        if self.use_noise_reduction:
            audio = self.audio_processor.reduce_noise(audio, sample_rate)
        
        effects_processor = self.effects_processor
        if effects_processor and effects_processor.is_available():
            audio = effects_processor.apply(audio, sample_rate)
        
        return self.audio_processor.normalize(audio)
    
    def optimize_memory(self):
        self.model_loader.optimize_memory()
    
    def clear_cache(self):
        self.model_loader.clear_cache()
    
    def get_stats(self) -> Dict[str, Any]:
        effects_available = self.effects_processor.is_available() if self.effects_processor else False
        return {
            "model_name": self.model_name,
            "device": self.model_loader.device,
            "model_loaded": self.model_loader.model is not None,
            "post_processing_enabled": effects_available,
            "noise_reduction_enabled": self.use_noise_reduction,
            "tts_available": self.tts is not None,
            "fast_mode": self.fast_mode
        }


_realistic_generator: Optional[RealisticMusicGenerator] = None


def get_realistic_generator(
    model_name: str = "facebook/musicgen-medium",
    use_post_processing: bool = True,
    use_noise_reduction: bool = True,
    fast_mode: bool = False
) -> RealisticMusicGenerator:
    global _realistic_generator
    if _realistic_generator is None:
        _realistic_generator = RealisticMusicGenerator(
            model_name=model_name,
            use_post_processing=use_post_processing,
            use_noise_reduction=use_noise_reduction,
            fast_mode=fast_mode
        )
    return _realistic_generator
