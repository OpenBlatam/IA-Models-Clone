"""
Audio Optimizer
Optimizaciones específicas para procesamiento de audio
"""

import logging
from typing import Optional, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


class AudioOptimizer:
    """Optimizador para procesamiento de audio"""
    
    def __init__(self):
        self._optimized = False
    
    def optimize(self):
        """Aplica optimizaciones de audio"""
        if self._optimized:
            return
        
        logger.info("Applying audio processing optimizations...")
        
        # Configurar numpy para audio
        self._configure_numpy_audio()
        
        # Pre-compilar funciones comunes
        self._precompile_audio_functions()
        
        self._optimized = True
        logger.info("Audio optimizations applied")
    
    def _configure_numpy_audio(self):
        """Configura numpy para procesamiento de audio"""
        # Usar float32 en lugar de float64 para audio (más rápido, menos memoria)
        np.seterr(all='ignore')  # Ignorar warnings de overflow en audio
    
    def _precompile_audio_functions(self):
        """Pre-compila funciones de audio comunes"""
        try:
            from numba import jit
            
            # Pre-compilar funciones de procesamiento de audio
            @jit(nopython=True, cache=True)
            def normalize_audio(audio_array):
                """Normaliza array de audio"""
                max_val = np.max(np.abs(audio_array))
                if max_val > 0:
                    return audio_array / max_val
                return audio_array
            
            @jit(nopython=True, cache=True)
            def apply_fade(audio_array, fade_samples):
                """Aplica fade in/out"""
                length = len(audio_array)
                if fade_samples > 0:
                    fade_in = np.linspace(0, 1, min(fade_samples, length))
                    fade_out = np.linspace(1, 0, min(fade_samples, length))
                    audio_array[:len(fade_in)] *= fade_in
                    audio_array[-len(fade_out):] *= fade_out
                return audio_array
            
            self._normalize_audio = normalize_audio
            self._apply_fade = apply_fade
            
            logger.debug("Audio functions pre-compiled")
        except ImportError:
            logger.warning("numba not available, skipping JIT compilation")
    
    def normalize_audio(self, audio_array: np.ndarray) -> np.ndarray:
        """Normaliza audio optimizado"""
        if hasattr(self, '_normalize_audio'):
            return self._normalize_audio(audio_array)
        # Fallback
        max_val = np.max(np.abs(audio_array))
        return audio_array / max_val if max_val > 0 else audio_array
    
    def apply_fade(self, audio_array: np.ndarray, fade_samples: int) -> np.ndarray:
        """Aplica fade optimizado"""
        if hasattr(self, '_apply_fade'):
            return self._apply_fade(audio_array, fade_samples)
        # Fallback
        length = len(audio_array)
        if fade_samples > 0:
            fade_in = np.linspace(0, 1, min(fade_samples, length))
            fade_out = np.linspace(1, 0, min(fade_samples, length))
            audio_array = audio_array.copy()
            audio_array[:len(fade_in)] *= fade_in
            audio_array[-len(fade_out):] *= fade_out
        return audio_array
    
    def optimize_audio_format(self, sample_rate: int, channels: int) -> Dict[str, Any]:
        """
        Optimiza formato de audio para procesamiento
        
        Args:
            sample_rate: Sample rate
            channels: Número de canales
            
        Returns:
            Configuración optimizada
        """
        # Usar float32 para procesamiento (más rápido que float64)
        # Reducir sample rate si es muy alto (para procesamiento)
        optimal_sample_rate = min(sample_rate, 44100)  # 44.1kHz es suficiente para la mayoría
        
        return {
            "dtype": np.float32,
            "sample_rate": optimal_sample_rate,
            "channels": min(channels, 2),  # Mono o stereo
            "chunk_size": 4096  # Tamaño óptimo de chunks
        }


# Instancia global
_audio_optimizer: Optional[AudioOptimizer] = None


def get_audio_optimizer() -> AudioOptimizer:
    """Obtiene el optimizador de audio"""
    global _audio_optimizer
    if _audio_optimizer is None:
        _audio_optimizer = AudioOptimizer()
    return _audio_optimizer















