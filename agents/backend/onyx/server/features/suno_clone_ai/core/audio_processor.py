"""
Procesador avanzado de audio
Incluye funciones de edición, mezcla, efectos y análisis
"""

import logging
import numpy as np
from typing import Optional, List, Tuple, Dict
from pathlib import Path
import soundfile as sf
import librosa
from scipy import signal
from numba import jit

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Procesador avanzado de audio"""
    
    def __init__(self, sample_rate: int = 32000):
        self.sample_rate = sample_rate
    
    def normalize(self, audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
        """Normaliza el audio a un nivel objetivo en dB"""
        try:
            # Calcular RMS
            rms = np.sqrt(np.mean(audio**2))
            if rms == 0:
                return audio
            
            # Calcular factor de normalización
            current_db = 20 * np.log10(rms)
            gain_db = target_db - current_db
            gain_linear = 10 ** (gain_db / 20)
            
            # Aplicar ganancia
            normalized = audio * gain_linear
            
            # Limitar a [-1, 1]
            normalized = np.clip(normalized, -1.0, 1.0)
            
            return normalized
        except Exception as e:
            logger.error(f"Error normalizing audio: {e}")
            return audio
    
    def apply_fade(self, audio: np.ndarray, fade_in: float = 0.5, 
                   fade_out: float = 0.5) -> np.ndarray:
        """Aplica fade in y fade out"""
        try:
            fade_in_samples = int(fade_in * self.sample_rate)
            fade_out_samples = int(fade_out * self.sample_rate)
            
            # Fade in
            if fade_in_samples > 0:
                fade_in_curve = np.linspace(0, 1, fade_in_samples)
                audio[:fade_in_samples] *= fade_in_curve
            
            # Fade out
            if fade_out_samples > 0:
                fade_out_curve = np.linspace(1, 0, fade_out_samples)
                audio[-fade_out_samples:] *= fade_out_curve
            
            return audio
        except Exception as e:
            logger.error(f"Error applying fade: {e}")
            return audio
    
    def mix_audio(self, audio_tracks: List[np.ndarray], 
                  volumes: Optional[List[float]] = None) -> np.ndarray:
        """Mezcla múltiples pistas de audio"""
        try:
            if not audio_tracks:
                raise ValueError("No audio tracks provided")
            
            # Normalizar longitudes
            max_length = max(len(track) for track in audio_tracks)
            normalized_tracks = []
            
            for i, track in enumerate(audio_tracks):
                if len(track) < max_length:
                    # Rellenar con ceros
                    padded = np.pad(track, (0, max_length - len(track)), 'constant')
                else:
                    padded = track[:max_length]
                
                # Aplicar volumen si se especifica
                if volumes and i < len(volumes):
                    padded *= volumes[i]
                
                normalized_tracks.append(padded)
            
            # Mezclar
            mixed = np.sum(normalized_tracks, axis=0)
            
            # Normalizar para evitar clipping
            max_val = np.max(np.abs(mixed))
            if max_val > 1.0:
                mixed = mixed / max_val
            
            return mixed
        except Exception as e:
            logger.error(f"Error mixing audio: {e}")
            raise
    
    def apply_reverb(self, audio: np.ndarray, room_size: float = 0.5,
                     damping: float = 0.5) -> np.ndarray:
        """Aplica efecto de reverb"""
        try:
            # Crear impulso de reverb simple
            reverb_length = int(0.5 * self.sample_rate)  # 0.5 segundos
            t = np.linspace(0, 0.5, reverb_length)
            
            # Decaimiento exponencial
            decay = np.exp(-t * damping * 10)
            
            # Aplicar convolución
            reverb_audio = signal.convolve(audio, decay, mode='same')
            
            # Mezclar con original
            mixed = audio + reverb_audio * room_size * 0.3
            
            # Normalizar
            max_val = np.max(np.abs(mixed))
            if max_val > 1.0:
                mixed = mixed / max_val
            
            return mixed
        except Exception as e:
            logger.error(f"Error applying reverb: {e}")
            return audio
    
    def apply_eq(self, audio: np.ndarray, low_gain: float = 0.0,
                 mid_gain: float = 0.0, high_gain: float = 0.0) -> np.ndarray:
        """Aplica ecualización"""
        try:
            # Filtros simples
            # Low pass (bajos)
            if low_gain != 0.0:
                b, a = signal.butter(4, 200 / (self.sample_rate / 2), 'low')
                low = signal.filtfilt(b, a, audio)
                audio = audio + low * low_gain
            
            # Band pass (medios)
            if mid_gain != 0.0:
                b, a = signal.butter(4, [200 / (self.sample_rate / 2), 
                                       2000 / (self.sample_rate / 2)], 'band')
                mid = signal.filtfilt(b, a, audio)
                audio = audio + mid * mid_gain
            
            # High pass (agudos)
            if high_gain != 0.0:
                b, a = signal.butter(4, 2000 / (self.sample_rate / 2), 'high')
                high = signal.filtfilt(b, a, audio)
                audio = audio + high * high_gain
            
            # Normalizar
            max_val = np.max(np.abs(audio))
            if max_val > 1.0:
                audio = audio / max_val
            
            return audio
        except Exception as e:
            logger.error(f"Error applying EQ: {e}")
            return audio
    
    def analyze_audio(self, audio: np.ndarray) -> Dict:
        """Analiza características del audio"""
        try:
            # RMS
            rms = np.sqrt(np.mean(audio**2))
            
            # Peak
            peak = np.max(np.abs(audio))
            
            # Zero crossing rate
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio, 
                                                             hop_length=512)[0])
            
            # Spectral centroid
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio, sr=self.sample_rate
            )[0]
            spectral_centroid = np.mean(spectral_centroids)
            
            # Tempo estimation
            tempo, _ = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
            
            return {
                "rms": float(rms),
                "peak": float(peak),
                "zero_crossing_rate": float(zcr),
                "spectral_centroid": float(spectral_centroid),
                "tempo": float(tempo),
                "duration": len(audio) / self.sample_rate
            }
        except Exception as e:
            logger.error(f"Error analyzing audio: {e}")
            return {}
    
    def trim_silence(self, audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """Elimina silencio al inicio y final"""
        try:
            # Encontrar índices donde el audio supera el threshold
            indices = np.where(np.abs(audio) > threshold)[0]
            
            if len(indices) == 0:
                return audio
            
            start = indices[0]
            end = indices[-1] + 1
            
            return audio[start:end]
        except Exception as e:
            logger.error(f"Error trimming silence: {e}")
            return audio
    
    def change_tempo(self, audio: np.ndarray, factor: float) -> np.ndarray:
        """Cambia el tempo sin cambiar el pitch"""
        try:
            # Usar librosa para time stretching
            stretched = librosa.effects.time_stretch(audio, rate=factor)
            return stretched
        except Exception as e:
            logger.error(f"Error changing tempo: {e}")
            return audio
    
    def change_pitch(self, audio: np.ndarray, semitones: float) -> np.ndarray:
        """Cambia el pitch sin cambiar el tempo"""
        try:
            # Usar librosa para pitch shifting
            shifted = librosa.effects.pitch_shift(
                audio, 
                sr=self.sample_rate, 
                n_steps=semitones
            )
            return shifted
        except Exception as e:
            logger.error(f"Error changing pitch: {e}")
            return audio


# Instancia global
_audio_processor: Optional[AudioProcessor] = None


def get_audio_processor(sample_rate: int = 32000) -> AudioProcessor:
    """Obtiene la instancia global del procesador de audio"""
    global _audio_processor
    if _audio_processor is None:
        _audio_processor = AudioProcessor(sample_rate=sample_rate)
    return _audio_processor

