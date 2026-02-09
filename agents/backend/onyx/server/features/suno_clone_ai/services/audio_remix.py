"""
Sistema de Remix y Mashup

Proporciona:
- Remix automático de canciones
- Mashup de múltiples canciones
- Sincronización de BPM
- Mezcla de pistas
- Efectos de DJ
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

try:
    import librosa
    import soundfile as sf
    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    AUDIO_LIBS_AVAILABLE = False
    logger.warning("Audio libraries not available, remix limited")


@dataclass
class RemixConfig:
    """Configuración de remix"""
    target_bpm: Optional[float] = None
    key: Optional[str] = None
    fade_in: float = 0.0  # segundos
    fade_out: float = 0.0  # segundos
    crossfade: float = 0.0  # segundos para mashup
    volume: float = 1.0  # 0.0 a 1.0


@dataclass
class RemixResult:
    """Resultado de remix"""
    output_path: str
    original_bpm: float
    new_bpm: float
    duration: float
    timestamp: datetime = field(default_factory=datetime.now)


class AudioRemixer:
    """Remixer de audio"""
    
    def __init__(self):
        logger.info("AudioRemixer initialized")
    
    def remix(
        self,
        audio_path: str,
        output_path: str,
        config: Optional[RemixConfig] = None
    ) -> RemixResult:
        """
        Crea un remix de un archivo de audio
        
        Args:
            audio_path: Ruta del archivo original
            output_path: Ruta de salida
            config: Configuración del remix
        
        Returns:
            RemixResult
        """
        if not AUDIO_LIBS_AVAILABLE:
            raise Exception("Audio libraries not available")
        
        if config is None:
            config = RemixConfig()
        
        try:
            # Cargar audio
            y, sr = librosa.load(audio_path, sr=None)
            original_bpm = librosa.beat.beat_track(y=y, sr=sr)[0]
            
            # Cambiar BPM si se especifica
            if config.target_bpm and config.target_bpm != original_bpm:
                # Time stretching
                rate = original_bpm / config.target_bpm
                y = librosa.effects.time_stretch(y, rate=rate)
                new_bpm = config.target_bpm
            else:
                new_bpm = original_bpm
            
            # Cambiar key si se especifica (simplificado)
            # En producción usar librosa.effects.pitch_shift
            
            # Aplicar fade in/out
            if config.fade_in > 0:
                fade_samples = int(config.fade_in * sr)
                fade_curve = np.linspace(0, 1, fade_samples)
                y[:fade_samples] *= fade_curve
            
            if config.fade_out > 0:
                fade_samples = int(config.fade_out * sr)
                fade_curve = np.linspace(1, 0, fade_samples)
                y[-fade_samples:] *= fade_curve
            
            # Ajustar volumen
            y *= config.volume
            
            # Guardar
            sf.write(output_path, y, sr)
            
            duration = len(y) / sr
            
            logger.info(f"Remix created: {original_bpm} BPM -> {new_bpm} BPM")
            
            return RemixResult(
                output_path=output_path,
                original_bpm=float(original_bpm),
                new_bpm=float(new_bpm),
                duration=duration
            )
        
        except Exception as e:
            logger.error(f"Error creating remix: {e}")
            raise
    
    def mashup(
        self,
        audio_paths: List[str],
        output_path: str,
        config: Optional[RemixConfig] = None
    ) -> RemixResult:
        """
        Crea un mashup de múltiples canciones
        
        Args:
            audio_paths: Lista de rutas de archivos
            output_path: Ruta de salida
            config: Configuración
        
        Returns:
            RemixResult
        """
        if not AUDIO_LIBS_AVAILABLE:
            raise Exception("Audio libraries not available")
        
        if config is None:
            config = RemixConfig()
        
        try:
            # Cargar todos los audios
            audio_data = []
            sample_rates = []
            bpms = []
            
            for path in audio_paths:
                y, sr = librosa.load(path, sr=None)
                bpm = librosa.beat.beat_track(y=y, sr=sr)[0]
                
                audio_data.append(y)
                sample_rates.append(sr)
                bpms.append(bpm)
            
            # Normalizar sample rate
            target_sr = max(sample_rates)
            normalized_audio = []
            
            for y, sr in zip(audio_data, sample_rates):
                if sr != target_sr:
                    y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
                normalized_audio.append(y)
            
            # Sincronizar BPM si se especifica
            if config.target_bpm:
                synced_audio = []
                for y, bpm in zip(normalized_audio, bpms):
                    if bpm != config.target_bpm:
                        rate = bpm / config.target_bpm
                        y = librosa.effects.time_stretch(y, rate=rate)
                    synced_audio.append(y)
                normalized_audio = synced_audio
            
            # Mezclar (promedio)
            min_length = min(len(y) for y in normalized_audio)
            trimmed_audio = [y[:min_length] for y in normalized_audio]
            
            # Mezclar con crossfade si se especifica
            if config.crossfade > 0:
                mixed = self._crossfade_mix(trimmed_audio, target_sr, config.crossfade)
            else:
                mixed = np.mean(trimmed_audio, axis=0)
            
            # Aplicar fade in/out
            if config.fade_in > 0:
                fade_samples = int(config.fade_in * target_sr)
                fade_curve = np.linspace(0, 1, fade_samples)
                mixed[:fade_samples] *= fade_curve
            
            if config.fade_out > 0:
                fade_samples = int(config.fade_out * target_sr)
                fade_curve = np.linspace(1, 0, fade_samples)
                mixed[-fade_samples:] *= fade_curve
            
            # Ajustar volumen
            mixed *= config.volume
            
            # Normalizar para evitar clipping
            if np.max(np.abs(mixed)) > 1.0:
                mixed = mixed / np.max(np.abs(mixed))
            
            # Guardar
            sf.write(output_path, mixed, target_sr)
            
            duration = len(mixed) / target_sr
            avg_bpm = np.mean(bpms) if not config.target_bpm else config.target_bpm
            
            logger.info(f"Mashup created: {len(audio_paths)} tracks, {avg_bpm} BPM")
            
            return RemixResult(
                output_path=output_path,
                original_bpm=float(np.mean(bpms)),
                new_bpm=float(avg_bpm),
                duration=duration
            )
        
        except Exception as e:
            logger.error(f"Error creating mashup: {e}")
            raise
    
    def _crossfade_mix(
        self,
        audio_list: List[np.ndarray],
        sr: int,
        crossfade_time: float
    ) -> np.ndarray:
        """Mezcla con crossfade entre pistas"""
        crossfade_samples = int(crossfade_time * sr)
        
        # Para simplificar, solo mezclamos las primeras dos pistas con crossfade
        if len(audio_list) < 2:
            return audio_list[0] if audio_list else np.array([])
        
        result = audio_list[0].copy()
        
        for i in range(1, len(audio_list)):
            prev_audio = audio_list[i - 1]
            curr_audio = audio_list[i]
            
            # Crossfade en la intersección
            if len(prev_audio) > crossfade_samples:
                fade_out = np.linspace(1, 0, crossfade_samples)
                fade_in = np.linspace(0, 1, crossfade_samples)
                
                # Aplicar fade out al final del anterior
                result[-crossfade_samples:] *= fade_out
                
                # Aplicar fade in al inicio del actual
                if len(curr_audio) >= crossfade_samples:
                    curr_audio[:crossfade_samples] *= fade_in
                
                # Mezclar en la zona de crossfade
                result[-crossfade_samples:] += curr_audio[:crossfade_samples]
                
                # Agregar el resto
                if len(curr_audio) > crossfade_samples:
                    result = np.concatenate([result, curr_audio[crossfade_samples:]])
            else:
                result = np.concatenate([result, curr_audio])
        
        return result


# Instancia global
_audio_remixer: Optional[AudioRemixer] = None


def get_audio_remixer() -> AudioRemixer:
    """Obtiene la instancia global del remixer"""
    global _audio_remixer
    if _audio_remixer is None:
        _audio_remixer = AudioRemixer()
    return _audio_remixer

