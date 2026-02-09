import logging
import numpy as np

logger = logging.getLogger(__name__)


class NoiseReducer:
    def reduce(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        try:
            import noisereduce as nr
            result = nr.reduce_noise(
                y=audio.astype(np.float32, copy=False),
                sr=sample_rate,
                stationary=False,
                prop_decrease=0.7,
                n_fft=2048,
                win_length=2048,
                hop_length=512
            )
            return result.astype(np.float32, copy=False) if result.dtype != np.float32 else result
        except ImportError:
            logger.warning("noisereduce not available")
            return audio
        except Exception as e:
            logger.warning(f"Error reducing noise: {e}")
            return audio

