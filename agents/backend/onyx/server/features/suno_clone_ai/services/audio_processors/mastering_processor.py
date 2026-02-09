import logging
import numpy as np

logger = logging.getLogger(__name__)


class MasteringProcessor:
    def __init__(self, fast_mode: bool = False):
        self.fast_mode = fast_mode
        self._filter_cache = {}
    
    def master(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if self.fast_mode:
            return audio
        
        try:
            import scipy.signal
            
            if audio.ndim == 1:
                audio = np.stack([audio, audio], axis=0)
            
            audio = audio.astype(np.float32, copy=False)
            
            num_channels = audio.shape[0]
            for ch in range(num_channels):
                channel = audio[ch]
                multiband_comp = self._multiband_compressor(channel, sample_rate)
                result = np.empty_like(channel, dtype=np.float32)
                np.multiply(channel, 0.7, out=result)
                np.add(result, multiband_comp * 0.3, out=result)
                result = self._apply_exciter(result, sample_rate)
                audio[ch] = result
            
            return audio
        except Exception as e:
            logger.warning(f"Error mastering audio: {e}")
            return audio
    
    def _multiband_compressor(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        try:
            import scipy.signal
            
            cache_key = (sample_rate, 'multiband')
            if cache_key not in self._filter_cache:
                self._filter_cache[cache_key] = {
                    'low': scipy.signal.iirfilter(4, 200, btype='low', ftype='butter', output='sos', fs=sample_rate),
                    'high': scipy.signal.iirfilter(4, 200, btype='high', ftype='butter', output='sos', fs=sample_rate)
                }
            
            filters = self._filter_cache[cache_key]
            low = scipy.signal.sosfilt(filters['low'], audio)
            high = scipy.signal.sosfilt(filters['high'], audio)
            
            temp = np.empty_like(low, dtype=np.float32)
            np.multiply(low, 1.2, out=temp)
            np.tanh(temp, out=temp)
            low_comp = np.multiply(temp, 0.9, out=temp)
            
            high_comp = np.multiply(high, 0.95)
            
            result = np.empty_like(audio, dtype=np.float32)
            np.add(low_comp, high_comp, out=result)
            return result
        except Exception:
            return audio
    
    def _apply_exciter(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        try:
            import scipy.signal
            
            cache_key = (sample_rate, 'exciter')
            if cache_key not in self._filter_cache:
                self._filter_cache[cache_key] = scipy.signal.iirfilter(
                    2, 5000, btype='high', ftype='butter', output='sos', fs=sample_rate
                )
            
            high = scipy.signal.sosfilt(self._filter_cache[cache_key], audio)
            
            temp = np.empty_like(high, dtype=np.float32)
            np.multiply(high, 2.0, out=temp)
            np.tanh(temp, out=temp)
            harmonic = np.multiply(temp, 0.3, out=temp)
            
            result = np.empty_like(audio, dtype=np.float32)
            np.add(audio, harmonic, out=result)
            return result
        except Exception:
            return audio

