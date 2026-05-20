"""
Simple Mixer Refactorizado - Versión simplificada y optimizada.

Mejoras:
- Método _perform_mixing dividido en métodos más pequeños
- Imports consolidados
- Lógica de procesamiento separada
- Mejor legibilidad
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union, Any

from .base_mixer import BaseMixer
from ..core.config import MixingConfig
from ..core.exceptions import AudioProcessingError


class SimpleMixer(BaseMixer):
    """
    Mezclador de audio simple usando librosa.
    
    Refactorizado para mejor legibilidad y mantenibilidad.
    """
    
    def __init__(
        self,
        config: Optional[MixingConfig] = None,
        **kwargs
    ):
        """Inicializa el mezclador simple."""
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
        Realiza la mezcla usando librosa.
        
        Dividido en pasos claros para mejor legibilidad.
        """
        try:
            import librosa
            import soundfile as sf
            import numpy as np
        except ImportError:
            raise AudioProcessingError(
                "librosa and soundfile are required. Install with: "
                "pip install librosa soundfile",
                component=self.name
            )
        
        try:
            # Paso 1: Cargar y procesar archivos
            audio_data, sample_rate = self._load_and_process_files(
                audio_files, volumes, librosa, np
            )
            
            # Paso 2: Alinear longitudes
            aligned_data = self._align_audio_lengths(audio_data, np)
            
            # Paso 3: Mezclar pistas
            mixed = self._mix_tracks(aligned_data, np)
            
            # Paso 4: Post-procesamiento
            processed = self._post_process(mixed, sample_rate, np)
            
            # Paso 5: Guardar
            sf.write(str(output_path), processed, sample_rate)
            return str(output_path)
        except Exception as e:
            raise AudioProcessingError(
                f"Simple mixing failed: {e}",
                component=self.name
            ) from e
    
    def _apply_effect_impl(
        self,
        audio_path: Path,
        effect_type: str,
        effect_params: Dict[str, Any],
        output_path: Path
    ) -> str:
        """Aplica un efecto simple."""
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
            y, sr = librosa.load(str(audio_path), sr=None)
            y = self._apply_single_effect(y, sr, effect_type, effect_params, np)
            sf.write(str(output_path), y, sr)
            return str(output_path)
        except Exception as e:
            raise AudioProcessingError(
                f"Failed to apply effect: {e}",
                component=self.name
            ) from e
    
    # ════════════════════════════════════════════════════════════════════════════
    # MÉTODOS HELPER (dividen la lógica en pasos claros)
    # ════════════════════════════════════════════════════════════════════════════
    
    def _load_and_process_files(
        self,
        audio_files: Dict[str, Path],
        volumes: Dict[str, float],
        librosa,
        np
    ) -> tuple[Dict[str, Any], int]:
        """
        Carga y procesa todos los archivos de audio.
        
        Returns:
            Tupla de (audio_data dict, sample_rate)
        """
        audio_data = {}
        sample_rate = None
        max_length = 0
        
        for name, path in audio_files.items():
            # Cargar audio
            y, sr = librosa.load(str(path), sr=None, mono=False)
            
            # Convertir a mono si es necesario
            if len(y.shape) > 1:
                y = librosa.to_mono(y)
            
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
    
    def _align_audio_lengths(
        self,
        audio_data: Dict[str, Any],
        np
    ) -> Dict[str, Any]:
        """
        Alinea las longitudes de todos los audios (rellena con ceros).
        
        Returns:
            Diccionario con audios alineados
        """
        if not audio_data:
            return {}
        
        max_length = max(len(y) for y in audio_data.values())
        aligned = {}
        
        for name, y in audio_data.items():
            current_length = len(y)
            if current_length < max_length:
                padding = np.zeros(max_length - current_length)
                aligned[name] = np.concatenate([y, padding])
            else:
                aligned[name] = y
        
        return aligned
    
    def _mix_tracks(
        self,
        audio_data: Dict[str, Any],
        np
    ) -> Any:
        """
        Mezcla todas las pistas sumándolas.
        
        Returns:
            Array numpy con audio mezclado
        """
        if not audio_data:
            raise AudioProcessingError(
                "No audio data to mix",
                component=self.name
            )
        
        # Obtener longitud de la primera pista
        first_track = next(iter(audio_data.values()))
        mixed = np.zeros_like(first_track)
        
        # Sumar todas las pistas
        for y in audio_data.values():
            mixed = mixed + y
        
        return mixed
    
    def _post_process(
        self,
        audio: Any,
        sample_rate: int,
        np
    ) -> Any:
        """
        Aplica post-procesamiento (normalización, fade in/out).
        
        Returns:
            Audio procesado
        """
        processed = audio.copy()
        
        # Normalizar
        if self._config.normalize_output:
            max_val = np.abs(processed).max()
            if max_val > 0:
                processed = processed / max_val * 0.95  # Headroom
        
        # Fade in
        if self._config.fade_in > 0:
            fade_samples = int(self._config.fade_in * sample_rate)
            fade_curve = np.linspace(0, 1, fade_samples)
            processed[:fade_samples] = processed[:fade_samples] * fade_curve
        
        # Fade out
        if self._config.fade_out > 0:
            fade_samples = int(self._config.fade_out * sample_rate)
            fade_curve = np.linspace(1, 0, fade_samples)
            processed[-fade_samples:] = processed[-fade_samples:] * fade_curve
        
        return processed
    
    def _apply_single_effect(
        self,
        audio: Any,
        sample_rate: int,
        effect_type: str,
        effect_params: Dict[str, Any],
        np
    ) -> Any:
        """
        Aplica un efecto individual a un audio.
        
        Returns:
            Audio con efecto aplicado
        """
        if effect_type == "gain":
            gain_db = effect_params.get("gain_db", 0.0)
            return audio * (10 ** (gain_db / 20))
        
        elif effect_type in ["lowpass", "highpass"]:
            try:
                from scipy import signal
            except ImportError:
                raise AudioProcessingError(
                    "scipy is required for filter effects",
                    component=self.name
                )
            
            cutoff = effect_params.get("cutoff", 3000 if effect_type == "lowpass" else 300)
            btype = "low" if effect_type == "lowpass" else "high"
            b, a = signal.butter(4, cutoff / (sample_rate / 2), btype)
            return signal.filtfilt(b, a, audio)
        
        else:
            raise AudioProcessingError(
                f"Unsupported effect type: {effect_type}",
                component=self.name
            )

