import logging
import numpy as np

logger = logging.getLogger(__name__)


class StereoConverter:
    def __init__(self, fast_mode: bool = False):
        self.fast_mode = fast_mode
    
    def convert(self, audio: np.ndarray) -> np.ndarray:
        if audio.ndim > 1:
            if audio.dtype != np.float32:
                return audio.astype(np.float32, copy=False)
            return audio
        
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32, copy=False)
        
        if self.fast_mode:
            return np.stack([audio, audio], axis=0)
        
        try:
            import scipy.signal
            sample_rate = 44100
            delay_samples = int(0.03 * sample_rate)
            haas_delay = int(0.015 * sample_rate)
            
            audio_len = len(audio)
            delayed = np.pad(audio, (delay_samples, 0), mode='constant')[:audio_len]
            
            left = np.empty_like(audio, dtype=np.float32)
            right = np.empty_like(audio, dtype=np.float32)
            
            np.multiply(audio, 0.92, out=left)
            np.multiply(audio, 0.92, out=right)
            
            delayed_left = np.roll(delayed, -haas_delay)
            delayed_right = np.roll(delayed, haas_delay)
            
            np.add(left, delayed_left * 0.08, out=left)
            np.add(right, delayed_right * 0.08, out=right)
            
            mid = np.empty_like(audio, dtype=np.float32)
            side = np.empty_like(audio, dtype=np.float32)
            np.add(left, right, out=mid)
            np.multiply(mid, 0.5, out=mid)
            np.subtract(left, right, out=side)
            np.multiply(side, 0.5, out=side)
            
            width = 0.3
            left_out = np.empty_like(audio, dtype=np.float32)
            right_out = np.empty_like(audio, dtype=np.float32)
            np.add(mid, side * width, out=left_out)
            np.subtract(mid, side * width, out=right_out)
            
            return np.stack([left_out, right_out], axis=0)
        except Exception as e:
            logger.warning(f"Error converting to stereo: {e}")
            return np.stack([audio, audio], axis=0)

