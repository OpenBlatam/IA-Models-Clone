"""
Advanced Mixer Refactorizado - Versión optimizada que reutiliza SimpleMixer.

Mejoras:
- Reutiliza métodos de SimpleMixer
- Efectos en métodos separados
- Menos duplicación
- Mejor organización
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union, Any

from .simple_mixer import SimpleMixer
from ..core.config import MixingConfig
from ..core.exceptions import AudioProcessingError


class AdvancedMixer(SimpleMixer):
    """
    Mezclador de audio avanzado con efectos.
    
    Refactorizado para reutilizar SimpleMixer y agregar solo efectos.
    """
    
    def __init__(
        self,
        config: Optional[MixingConfig] = None,
        **kwargs
    ):
        """Inicializa el mezclador avanzado."""
        if config is None:
            config = MixingConfig()
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
        
        Reutiliza SimpleMixer pero aplica efectos antes de mezclar.
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
            # Paso 1: Cargar y procesar con efectos
            audio_data, sample_rate = self._load_and_process_with_effects(
                audio_files, volumes, effects, librosa, np
            )
            
            # Paso 2-5: Reutilizar métodos de SimpleMixer
            aligned_data = self._align_audio_lengths(audio_data, np)
            mixed = self._mix_tracks(aligned_data, np)
            processed = self._post_process(mixed, sample_rate, np)
            
            # Guardar
            sf.write(str(output_path), processed, sample_rate)
            return str(output_path)
        except Exception as e:
            raise AudioProcessingError(
                f"Advanced mixing failed: {e}",
                component=self.name
            ) from e
    
    # ════════════════════════════════════════════════════════════════════════════
    # MÉTODOS HELPER (específicos de AdvancedMixer)
    # ════════════════════════════════════════════════════════════════════════════
    
    def _load_and_process_with_effects(
        self,
        audio_files: Dict[str, Path],
        volumes: Dict[str, float],
        effects: Optional[Dict[str, Dict[str, Any]]],
        librosa,
        np
    ) -> tuple[Dict[str, Any], int]:
        """
        Carga y procesa archivos aplicando efectos.
        
        Similar a SimpleMixer pero con efectos adicionales.
        """
        audio_data = {}
        sample_rate = None
        max_length = 0
        
        for name, path in audio_files.items():
            # Cargar audio (reutilizar lógica base)
            y, sr = librosa.load(str(path), sr=None, mono=False)
            if len(y.shape) > 1:
                y = librosa.to_mono(y)
            
            # Aplicar efectos específicos del componente
            if effects and name in effects:
                y = self._apply_effects_to_audio(y, sr, effects[name], np)
            
            # Aplicar efectos globales de configuración
            y = self._apply_global_effects(y, sr, np)
            
            # Aplicar volumen
            volume = volumes.get(name, self._config.default_volume)
            y = y * volume
            
            # Resamplear si es necesario
            if sample_rate is None:
                sample_rate = sr
            elif sr != sample_rate:
                y = librosa.resample(y, orig_sr=sr, target_sr=sample_rate)
            
            audio_data[name] = y
            max_length = max(max_length, len(y))
        
        return audio_data, sample_rate
    
    def _apply_global_effects(
        self,
        audio: Any,
        sample_rate: int,
        np
    ) -> Any:
        """
        Aplica efectos globales de configuración.
        
        Args:
            audio: Array de audio
            sample_rate: Sample rate
            np: Módulo numpy
        
        Returns:
            Audio con efectos aplicados
        """
        processed = audio.copy()
        
        if self._config.apply_reverb:
            processed = self._apply_reverb(processed, sample_rate, self._config.reverb_params, np)
        
        if self._config.apply_eq:
            processed = self._apply_eq(processed, sample_rate, self._config.eq_params, np)
        
        if self._config.apply_compressor:
            processed = self._apply_compressor(processed, sample_rate, self._config.compressor_params, np)
        
        return processed
    
    def _apply_effects_to_audio(
        self,
        audio: Any,
        sample_rate: int,
        effects: Dict[str, Any],
        np
    ) -> Any:
        """
        Aplica múltiples efectos a un audio.
        
        Args:
            audio: Array de audio
            sample_rate: Sample rate
            effects: Diccionario de efectos
            np: Módulo numpy
        
        Returns:
            Audio procesado
        """
        result = audio.copy()
        
        for effect_type, params in effects.items():
            if effect_type == "reverb":
                result = self._apply_reverb(result, sample_rate, params, np)
            elif effect_type == "eq":
                result = self._apply_eq(result, sample_rate, params, np)
            elif effect_type == "compressor":
                result = self._apply_compressor(result, sample_rate, params, np)
            elif effect_type == "gain":
                gain_db = params.get("gain_db", 0.0)
                result = result * (10 ** (gain_db / 20))
        
        return result
    
    def _apply_reverb(
        self,
        audio: Any,
        sample_rate: int,
        params: Dict[str, Any],
        np
    ) -> Any:
        """Aplica reverb al audio."""
        delay_ms = params.get("delay_ms", 50)
        feedback = params.get("feedback", 0.3)
        mix = params.get("mix", 0.5)
        
        delay_samples = int(delay_ms * sample_rate / 1000)
        if delay_samples >= len(audio):
            return audio
        
        delayed = np.zeros_like(audio)
        delayed[delay_samples:] = audio[:-delay_samples]
        
        reverb_signal = delayed * feedback
        return audio * (1 - mix) + reverb_signal * mix
    
    def _apply_eq(
        self,
        audio: Any,
        sample_rate: int,
        params: Dict[str, Any],
        np
    ) -> Any:
        """Aplica EQ al audio."""
        try:
            from scipy import signal
        except ImportError:
            return audio  # Sin EQ si scipy no está disponible
        
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
        audio: Any,
        sample_rate: int,
        params: Dict[str, Any],
        np
    ) -> Any:
        """Aplica compresor al audio."""
        threshold = params.get("threshold", -12.0)  # dB
        ratio = params.get("ratio", 4.0)
        
        threshold_linear = 10 ** (threshold / 20)
        envelope = np.abs(audio)
        
        compressed = np.zeros_like(audio)
        for i in range(len(audio)):
            if envelope[i] > threshold_linear:
                excess = envelope[i] - threshold_linear
                compressed_level = threshold_linear + excess / ratio
                gain = compressed_level / envelope[i] if envelope[i] > 0 else 1.0
            else:
                gain = 1.0
            
            compressed[i] = audio[i] * gain
        
        return compressed

