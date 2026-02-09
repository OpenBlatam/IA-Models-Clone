import logging
import numpy as np
from typing import Optional
from .audio_processors.normalizer import Normalizer
from .audio_processors.stereo_converter import StereoConverter
from .audio_processors.noise_reducer import NoiseReducer

logger = logging.getLogger(__name__)


class AudioProcessor:
    def __init__(self, fast_mode: bool = False):
        self.fast_mode = fast_mode
        self.normalizer = Normalizer()
        self.stereo_converter = StereoConverter(fast_mode)
        self.noise_reducer = NoiseReducer()
        self._eq_filters_cache = {}
        self._dither_rng = np.random.default_rng()
    
    def upsample(self, audio: np.ndarray, target_sr: int) -> tuple:
        if self.fast_mode:
            if audio.dtype != np.float32:
                return audio.astype(np.float32, copy=False), target_sr
            return audio, target_sr
        
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32, copy=False)
        
        if target_sr >= 44100:
            return audio, target_sr
        
        try:
            import librosa
            audio = librosa.resample(
                audio, orig_sr=target_sr, target_sr=44100, res_type='kaiser_fast'
            )
            if audio.dtype != np.float32:
                return audio.astype(np.float32, copy=False), 44100
            return audio, 44100
        except ImportError:
            return audio, target_sr
        except Exception as e:
            logger.warning(f"Error upsampling: {e}")
            return audio, target_sr
    
    def reduce_noise(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        return self.noise_reducer.reduce(audio, sample_rate)
    
    def remove_artifacts(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        audio_len = len(audio)
        if audio_len < 1000 or self.fast_mode:
            return audio
        try:
            import scipy.signal
            cache_key = (sample_rate, 'artifacts')
            if cache_key not in self._eq_filters_cache:
                self._eq_filters_cache[cache_key] = scipy.signal.butter(
                    4, [20, sample_rate//2 - 1000], btype='band', fs=sample_rate, output='sos'
                )
            result = scipy.signal.sosfiltfilt(self._eq_filters_cache[cache_key], audio)
            return result.astype(np.float32, copy=False) if result.dtype != np.float32 else result
        except Exception as e:
            logger.warning(f"Error removing artifacts: {e}")
            return audio
    
    def apply_eq(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        audio_len = len(audio)
        if audio_len < 1000 or self.fast_mode:
            return audio
        try:
            import scipy.signal
            
            cache_key = sample_rate
            if cache_key not in self._eq_filters_cache:
                self._eq_filters_cache[cache_key] = {
                    'low': scipy.signal.iirfilter(4, [60, 200], btype='band', ftype='butter', output='sos', fs=sample_rate),
                    'mid': scipy.signal.iirfilter(4, [200, 2000], btype='band', ftype='butter', output='sos', fs=sample_rate),
                    'high': scipy.signal.iirfilter(2, 8000, btype='high', ftype='butter', output='sos', fs=sample_rate),
                    'presence': scipy.signal.iirfilter(4, [2000, 5000], btype='band', ftype='butter', output='sos', fs=sample_rate)
                }
            
            filters = self._eq_filters_cache[cache_key]
            low_boost = scipy.signal.sosfilt(filters['low'], audio)
            mid_boost = scipy.signal.sosfilt(filters['mid'], audio)
            high_boost = scipy.signal.sosfilt(filters['high'], audio)
            presence_boost = scipy.signal.sosfilt(filters['presence'], audio)
            
            result = np.empty_like(audio, dtype=np.float32)
            np.multiply(audio, 0.5, out=result)
            np.add(result, low_boost * 0.1725, out=result)
            np.add(result, mid_boost * 0.162, out=result)
            np.add(result, high_boost * 0.11, out=result)
            np.add(result, presence_boost * 0.112, out=result)
            return result
        except Exception as e:
            logger.warning(f"Error applying EQ: {e}")
            return audio
    
    def enhance_dynamics(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        audio_len = len(audio)
        if self.fast_mode or audio_len < 1000:
            return audio
        try:
            import scipy.signal
            envelope = np.abs(scipy.signal.hilbert(audio))
            envelope_len = len(envelope)
            win_len = min(101, envelope_len//4*2+1)
            if win_len < 5:
                return audio
            smoothed = scipy.signal.savgol_filter(envelope, window_length=win_len, polyorder=3)
            result = np.empty_like(audio, dtype=np.float32)
            temp = smoothed * 0.1
            np.multiply(audio, temp, out=result)
            np.add(audio, result, out=result)
            return result
        except Exception as e:
            logger.warning(f"Error enhancing dynamics: {e}")
            return audio
    
    def apply_saturation(self, audio: np.ndarray) -> np.ndarray:
        try:
            temp = np.empty_like(audio, dtype=np.float32)
            np.multiply(audio, 0.8, out=temp)
            np.tanh(temp, out=temp)
            np.multiply(temp, 0.95, out=temp)
            result = np.empty_like(audio, dtype=np.float32)
            np.multiply(audio, 0.85, out=result)
            np.add(result, temp * 0.15, out=result)
            return result
        except Exception as e:
            logger.warning(f"Error applying saturation: {e}")
            return audio
    
    def convert_to_stereo(self, audio: np.ndarray) -> np.ndarray:
        return self.stereo_converter.convert(audio)
    
    def normalize(self, audio: np.ndarray) -> np.ndarray:
        return self.normalizer.normalize(audio)
    
    def apply_dithering(self, audio: np.ndarray) -> np.ndarray:
        try:
            dither = self._dither_rng.normal(0, 1e-6, size=audio.shape, dtype=np.float32)
            result = np.empty_like(audio)
            np.add(audio, dither, out=result)
            return result
        except Exception:
            return audio
    
    def trim_to_duration(
        self, audio: np.ndarray, duration: int, sample_rate: int
    ) -> np.ndarray:
        target_length = int(duration * sample_rate)
        ndim = audio.ndim
        
        if ndim > 1:
            audio_len = audio.shape[1]
            if audio_len == target_length:
                return audio
            if audio_len > target_length:
                return audio[:, :target_length]
            result = np.zeros((audio.shape[0], target_length), dtype=audio.dtype)
            result[:, :audio_len] = audio
            return result
        else:
            audio_len = len(audio)
            if audio_len == target_length:
                return audio
            if audio_len > target_length:
                return audio[:target_length]
            result = np.zeros(target_length, dtype=audio.dtype)
            result[:audio_len] = audio
            return result

