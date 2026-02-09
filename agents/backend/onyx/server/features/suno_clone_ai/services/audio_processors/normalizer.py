import logging
import numpy as np

logger = logging.getLogger(__name__)


class Normalizer:
    def normalize(self, audio: np.ndarray) -> np.ndarray:
        result = np.empty_like(audio, dtype=np.float32)
        
        if audio.ndim > 1:
            num_channels = audio.shape[0]
            for i in range(num_channels):
                channel = audio[i]
                if channel.dtype != np.float32:
                    channel = channel.astype(np.float32, copy=False)
                
                channel_len = len(channel)
                rms = np.sqrt(np.dot(channel, channel) / channel_len)
                if rms > 1e-8:
                    np.multiply(channel, 0.1 / rms, out=result[i])
                else:
                    np.copyto(result[i], channel)
                
                abs_result = np.abs(result[i])
                max_val = np.max(abs_result)
                if max_val > 0.95:
                    np.multiply(result[i], 0.95 / max_val, out=result[i])
        else:
            channel = audio
            if channel.dtype != np.float32:
                channel = channel.astype(np.float32, copy=False)
            
            channel_len = len(channel)
            rms = np.sqrt(np.dot(channel, channel) / channel_len)
            if rms > 1e-8:
                np.multiply(channel, 0.1 / rms, out=result)
            else:
                np.copyto(result, channel)
            
            abs_result = np.abs(result)
            max_val = np.max(abs_result)
            if max_val > 0.95:
                np.multiply(result, 0.95 / max_val, out=result)
        
        return result

