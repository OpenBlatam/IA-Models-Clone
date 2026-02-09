import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class ProcessingPipeline:
    def __init__(
        self,
        audio_processor,
        mastering_processor,
        effects_processor: Optional[object],
        fast_mode: bool
    ):
        self.audio_processor = audio_processor
        self.mastering_processor = mastering_processor
        self.effects_processor = effects_processor
        self.fast_mode = fast_mode
    
    def process(
        self,
        audio: np.ndarray,
        sample_rate: int,
        duration: int,
        use_noise_reduction: bool
    ) -> np.ndarray:
        audio_len = len(audio)
        if audio_len == 0:
            raise ValueError("Generated audio is empty")
        
        if not isinstance(audio, np.ndarray):
            audio = np.asarray(audio, dtype=np.float32)
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32, copy=False)
        
        if not np.all(np.isfinite(audio)):
            logger.warning("Input audio contains NaN or Inf, cleaning...")
            np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0, out=audio)
        
        audio, final_sr = self.audio_processor.upsample(audio, sample_rate)
        
        if use_noise_reduction and not self.fast_mode:
            audio = self.audio_processor.reduce_noise(audio, final_sr)
        
        if not self.fast_mode:
            audio = self.audio_processor.remove_artifacts(audio, final_sr)
        
        if self.effects_processor and self.effects_processor.is_available():
            audio = self.effects_processor.apply(audio, final_sr)
        
        if not self.fast_mode:
            audio = self.audio_processor.apply_eq(audio, final_sr)
            audio = self.audio_processor.enhance_dynamics(audio, final_sr)
            audio = self.audio_processor.apply_saturation(audio)
        
        audio = self.audio_processor.convert_to_stereo(audio)
        
        if not self.fast_mode:
            audio = self.mastering_processor.master(audio, final_sr)
        
        audio = self.audio_processor.normalize(audio)
        
        if not self.fast_mode:
            audio = self.audio_processor.apply_dithering(audio)
        
        audio = self.audio_processor.trim_to_duration(audio, duration, final_sr)
        
        if not np.all(np.isfinite(audio)):
            logger.warning("Audio contains NaN or Inf, cleaning...")
            np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0, out=audio)
        
        if audio.dtype != np.float32:
            return audio.astype(np.float32, copy=False)
        return audio

