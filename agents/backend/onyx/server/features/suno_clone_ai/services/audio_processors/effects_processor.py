import logging
import numpy as np

logger = logging.getLogger(__name__)


class EffectsProcessor:
    def __init__(self):
        self.pedalboard = None
        self._initialize()
    
    def _initialize(self):
        try:
            import pedalboard
            from pedalboard import (
                Reverb, Compressor, Gain, HighpassFilter,
                LowpassFilter, Limiter, Chorus
            )
            
            self.pedalboard = pedalboard.Pedalboard([
                HighpassFilter(cutoff_frequency_hz=40),
                Compressor(
                    threshold_db=-18,
                    ratio=3.5,
                    attack_ms=3,
                    release_ms=100,
                    makeup_gain_db=2
                ),
                Gain(gain_db=1.2),
                Chorus(rate_hz=1.5, depth=0.25, centre_delay_ms=7, feedback=0.1),
                Reverb(
                    room_size=0.5,
                    damping=0.4,
                    wet_level=0.25,
                    dry_level=0.75,
                    width=0.8,
                    freeze_mode=0
                ),
                LowpassFilter(cutoff_frequency_hz=18000),
                Limiter(threshold_db=-1.0, release_ms=50)
            ])
            logger.info("Pedalboard initialized")
        except ImportError:
            logger.warning("Pedalboard not available")
            self.pedalboard = None
        except Exception as e:
            logger.warning(f"Error initializing pedalboard: {e}")
            self.pedalboard = None
    
    def apply(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if not self.pedalboard:
            return audio
        
        try:
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32, copy=False)
            
            if audio.ndim == 1:
                audio = audio.reshape(1, -1)
            
            processed = self.pedalboard(audio, sample_rate)
            
            if processed.ndim > 1:
                if processed.shape[0] == 1:
                    processed = processed[0]
                else:
                    processed = np.mean(processed, axis=0, dtype=np.float32)
            
            if processed.dtype != np.float32:
                return processed.astype(np.float32, copy=False)
            return processed
        except Exception as e:
            logger.warning(f"Error applying effects: {e}")
            return audio
    
    def is_available(self) -> bool:
        return self.pedalboard is not None

