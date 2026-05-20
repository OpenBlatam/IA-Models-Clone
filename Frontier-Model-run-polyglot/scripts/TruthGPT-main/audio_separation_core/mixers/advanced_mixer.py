"""
Advanced Mixer - Mezclador de audio avanzado con efectos y procesamiento.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union, Any

from .simple_mixer import SimpleMixer
from ..core.config import MixingConfig
from ..core.exceptions import AudioProcessingError


class AdvancedMixer(SimpleMixer):
    """
    Mezclador de audio avanzado.
    
    Extiende SimpleMixer con soporte para efectos avanzados,
    compresor, EQ, reverb, etc.
    """
    
    def __init__(
        self,
        config: Optional[MixingConfig] = None,
        **kwargs
    ):
        """
        Inicializa el mezclador avanzado.
        
        Args:
            config: Configuración
            **kwargs: Parámetros adicionales
        """
        if config is None:
            config = MixingConfig(mixer_type="advanced")
        super().__init__(config, **kwargs)
    
    def _perform_mixing(
        self,
        audio_files: Dict[str, Path],
        output_path: Path,
        volumes: Dict[str, float],
        effects: Optional[Dict[str, Dict[str, Any]]],
        **kwargs
    ) -> str:
        """
        Realiza la mezcla con efectos avanzados.
        
        Args:
            audio_files: Diccionario de archivos de audio
            output_path: Ruta de salida
            volumes: Volúmenes por componente
            effects: Efectos por componente
            **kwargs: Parámetros adicionales
        
        Returns:
            Ruta al archivo mezclado
        """
        try:
            import librosa
            import soundfile as sf
            import numpy as np
        except ImportError:
            raise AudioProcessingError(
                "librosa and soundfile are required",
                component=self.name
            )
        
        try:
            # Cargar y procesar cada archivo con efectos
            processed_audio = {}
            sample_rate = None
            max_length = 0
            
            for name, path in audio_files.items():
                # Cargar audio
                y, sr = librosa.load(str(path), sr=None, mono=False)
                
                if len(y.shape) > 1:
                    y = librosa.to_mono(y)
                
                # Aplicar efectos específicos del componente
                if effects and name in effects:
                    y = self._apply_effects_to_audio(y, sr, effects[name])
                
                # Aplicar efectos globales de configuración
                if self._config.apply_reverb:
                    y = self._apply_reverb(y, sr, self._config.reverb_params)
                
                if self._config.apply_eq:
                    y = self._apply_eq(y, sr, self._config.eq_params)
                
                if self._config.apply_compressor:
                    y = self._apply_compressor(y, sr, self._config.compressor_params)
                
                # Aplicar volumen
                volume = volumes.get(name, self._config.default_volume)
                y = y * volume
                
                processed_audio[name] = y
                
                if sample_rate is None:
                    sample_rate = sr
                elif sr != sample_rate:
                    y = librosa.resample(y, orig_sr=sr, target_sr=sample_rate)
                    processed_audio[name] = y
                
                max_length = max(max_length, len(y))
            
            # Alinear longitudes
            for name in processed_audio:
                current_length = len(processed_audio[name])
                if current_length < max_length:
                    padding = np.zeros(max_length - current_length)
                    processed_audio[name] = np.concatenate([processed_audio[name], padding])
            
            # Mezclar
            mixed = np.zeros(max_length)
            for y in processed_audio.values():
                mixed = mixed + y
            
            # Normalizar
            if self._config.normalize_output:
                max_val = np.abs(mixed).max()
                if max_val > 0:
                    mixed = mixed / max_val * 0.95
            
            # Fade in/out
            if self._config.fade_in > 0:
                fade_samples = int(self._config.fade_in * sample_rate)
                fade_curve = np.linspace(0, 1, fade_samples)
                mixed[:fade_samples] = mixed[:fade_samples] * fade_curve
            
            if self._config.fade_out > 0:
                fade_samples = int(self._config.fade_out * sample_rate)
                fade_curve = np.linspace(1, 0, fade_samples)
                mixed[-fade_samples:] = mixed[-fade_samples:] * fade_curve
            
            # Guardar
            sf.write(str(output_path), mixed, sample_rate)
            return str(output_path)
        except Exception as e:
            raise AudioProcessingError(
                f"Advanced mixing failed: {e}",
                component=self.name
            ) from e
    
    def _apply_effects_to_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        effects: Dict[str, Any]
    ) -> np.ndarray:
        """
        Aplica múltiples efectos a un audio.
        
        Args:
            audio: Señal de audio
            sample_rate: Sample rate
            effects: Diccionario de efectos
        
        Returns:
            Audio procesado
        """
        result = audio.copy()
        
        for effect_type, params in effects.items():
            if effect_type == "reverb":
                result = self._apply_reverb(result, sample_rate, params)
            elif effect_type == "eq":
                result = self._apply_eq(result, sample_rate, params)
            elif effect_type == "compressor":
                result = self._apply_compressor(result, sample_rate, params)
            elif effect_type == "gain":
                gain_db = params.get("gain_db", 0.0)
                result = result * (10 ** (gain_db / 20))
        
        return result
    
    def _apply_reverb(
        self,
        audio: np.ndarray,
        sample_rate: int,
        params: Dict[str, Any]
    ) -> np.ndarray:
        """
        Aplica reverb al audio.
        
        Args:
            audio: Señal de audio
            sample_rate: Sample rate
            params: Parámetros de reverb
        
        Returns:
            Audio con reverb
        """
        # Implementación simple de reverb usando delay y feedback
        delay_ms = params.get("delay_ms", 50)
        feedback = params.get("feedback", 0.3)
        mix = params.get("mix", 0.5)
        
        delay_samples = int(delay_ms * sample_rate / 1000)
        delayed = np.zeros_like(audio)
        delayed[delay_samples:] = audio[:-delay_samples]
        
        reverb_signal = delayed * feedback
        result = audio * (1 - mix) + reverb_signal * mix
        
        return result
    
    def _apply_eq(
        self,
        audio: np.ndarray,
        sample_rate: int,
        params: Dict[str, Any]
    ) -> np.ndarray:
        """
        Aplica EQ al audio.
        
        Args:
            audio: Señal de audio
            sample_rate: Sample rate
            params: Parámetros de EQ
        
        Returns:
            Audio con EQ aplicado
        """
        try:
            from scipy import signal
        except ImportError:
            return audio  # Sin EQ si scipy no está disponible
        
        # EQ simple usando filtros
        result = audio.copy()
        
        # Low shelf
        if "low_gain" in params:
            gain = params["low_gain"]
            freq = params.get("low_freq", 200)
            b, a = signal.iirfilter(2, freq / (sample_rate / 2), btype="low", ftype="butter")
            filtered = signal.filtfilt(b, a, result)
            result = result + filtered * (gain - 1)
        
        # High shelf
        if "high_gain" in params:
            gain = params["high_gain"]
            freq = params.get("high_freq", 5000)
            b, a = signal.iirfilter(2, freq / (sample_rate / 2), btype="high", ftype="butter")
            filtered = signal.filtfilt(b, a, result)
            result = result + filtered * (gain - 1)
        
        return result
    
    def _apply_compressor(
        self,
        audio: np.ndarray,
        sample_rate: int,
        params: Dict[str, Any]
    ) -> np.ndarray:
        """
        Aplica compresor al audio.
        
        Args:
            audio: Señal de audio
            sample_rate: Sample rate
            params: Parámetros de compresor
        
        Returns:
            Audio comprimido
        """
        threshold = params.get("threshold", -12.0)  # dB
        ratio = params.get("ratio", 4.0)
        attack = params.get("attack", 0.003)  # segundos
        release = params.get("release", 0.1)  # segundos
        
        # Implementación simple de compresor
        threshold_linear = 10 ** (threshold / 20)
        
        # Calcular envolvente
        envelope = np.abs(audio)
        
        # Aplicar compresión
        compressed = np.zeros_like(audio)
        for i in range(len(audio)):
            if envelope[i] > threshold_linear:
                # Compresión
                excess = envelope[i] - threshold_linear
                compressed_level = threshold_linear + excess / ratio
                gain = compressed_level / envelope[i] if envelope[i] > 0 else 1.0
            else:
                gain = 1.0
            
            compressed[i] = audio[i] * gain
        
        return compressed




