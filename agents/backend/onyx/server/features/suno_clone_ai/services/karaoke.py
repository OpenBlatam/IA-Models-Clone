"""
Sistema de Karaoke

Proporciona:
- Generación de pistas de karaoke
- Eliminación de voces
- Visualización de letras sincronizadas
- Sistema de puntuación
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
    logger.warning("Audio libraries not available, karaoke limited")


@dataclass
class KaraokeTrack:
    """Pista de karaoke"""
    audio_path: str
    lyrics_path: Optional[str] = None
    synced_lyrics: Optional[Any] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class KaraokeScore:
    """Puntuación de karaoke"""
    accuracy: float = 0.0  # 0.0 a 1.0
    timing_score: float = 0.0
    pitch_score: float = 0.0
    total_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class KaraokeService:
    """Servicio de karaoke"""
    
    def __init__(self):
        logger.info("KaraokeService initialized")
    
    def create_karaoke_track(
        self,
        audio_path: str,
        output_path: str,
        method: str = "center"
    ) -> KaraokeTrack:
        """
        Crea una pista de karaoke eliminando voces
        
        Args:
            audio_path: Ruta del archivo original
            output_path: Ruta de salida
            method: Método de eliminación (center, spectral, ml)
        
        Returns:
            KaraokeTrack
        """
        if not AUDIO_LIBS_AVAILABLE:
            raise Exception("Audio libraries not available")
        
        try:
            # Cargar audio
            y, sr = librosa.load(audio_path, sr=None, mono=False)
            
            # Si es mono, convertir a estéreo
            if len(y.shape) == 1:
                y = np.column_stack([y, y])
            
            # Eliminar voces
            if method == "center":
                # Método simple: cancelar canal central (voces suelen estar centradas)
                karaoke = y[:, 0] - y[:, 1]
                # Convertir a estéreo
                karaoke = np.column_stack([karaoke, karaoke])
            elif method == "spectral":
                # Método espectral (simplificado)
                # En producción usar Spleeter o similar
                karaoke = self._spectral_vocal_removal(y, sr)
            else:
                # Por defecto, método center
                karaoke = y[:, 0] - y[:, 1]
                karaoke = np.column_stack([karaoke, karaoke])
            
            # Normalizar
            if np.max(np.abs(karaoke)) > 1.0:
                karaoke = karaoke / np.max(np.abs(karaoke))
            
            # Guardar
            sf.write(output_path, karaoke, sr)
            
            logger.info(f"Karaoke track created: {output_path}")
            
            return KaraokeTrack(audio_path=output_path)
        
        except Exception as e:
            logger.error(f"Error creating karaoke track: {e}")
            raise
    
    def _spectral_vocal_removal(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Eliminación de voces usando método espectral (simplificado)"""
        # STFT
        stft = librosa.stft(y[:, 0] if len(y.shape) > 1 else y, n_fft=2048)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Filtrar frecuencias vocales (aproximadamente 85-255 Hz para voces masculinas,
        # 165-255 Hz para voces femeninas)
        # Simplificado: reducir energía en rango vocal
        freq_bins = librosa.fft_frequencies(sr=sr, n_fft=2048)
        vocal_range = (freq_bins >= 85) & (freq_bins <= 255)
        magnitude[vocal_range, :] *= 0.3  # Reducir 70%
        
        # Reconstruir
        stft_filtered = magnitude * np.exp(1j * phase)
        y_filtered = librosa.istft(stft_filtered)
        
        # Convertir a estéreo
        return np.column_stack([y_filtered, y_filtered])
    
    def score_performance(
        self,
        original_audio_path: str,
        user_audio_path: str,
        synced_lyrics: Optional[Any] = None
    ) -> KaraokeScore:
        """
        Evalúa el rendimiento del usuario
        
        Args:
            original_audio_path: Audio original de referencia
            user_audio_path: Audio del usuario
            synced_lyrics: Letras sincronizadas (opcional)
        
        Returns:
            KaraokeScore
        """
        if not AUDIO_LIBS_AVAILABLE:
            return KaraokeScore()
        
        try:
            # Cargar audios
            y_original, sr_orig = librosa.load(original_audio_path, sr=None)
            y_user, sr_user = librosa.load(user_audio_path, sr=None)
            
            # Normalizar sample rate
            if sr_orig != sr_user:
                y_user = librosa.resample(y_user, orig_sr=sr_user, target_sr=sr_orig)
            
            # Ajustar duración
            min_len = min(len(y_original), len(y_user))
            y_original = y_original[:min_len]
            y_user = y_user[:min_len]
            
            # Calcular similitud de pitch
            pitch_score = self._calculate_pitch_score(y_original, y_user, sr_orig)
            
            # Calcular similitud de timing (si hay letras)
            timing_score = 0.5  # Por defecto
            if synced_lyrics:
                timing_score = self._calculate_timing_score(y_user, sr_orig, synced_lyrics)
            
            # Calcular precisión general
            accuracy = (pitch_score + timing_score) / 2
            
            # Score total (0-100)
            total_score = accuracy * 100
            
            return KaraokeScore(
                accuracy=accuracy,
                timing_score=timing_score,
                pitch_score=pitch_score,
                total_score=total_score
            )
        
        except Exception as e:
            logger.error(f"Error scoring performance: {e}")
            return KaraokeScore()
    
    def _calculate_pitch_score(
        self,
        y_original: np.ndarray,
        y_user: np.ndarray,
        sr: int
    ) -> float:
        """Calcula similitud de pitch"""
        try:
            # Extraer pitch
            pitches_orig = librosa.yin(y_original, fmin=80, fmax=400)
            pitches_user = librosa.yin(y_user, fmin=80, fmax=400)
            
            # Filtrar valores inválidos
            pitches_orig = pitches_orig[pitches_orig > 0]
            pitches_user = pitches_user[pitches_user > 0]
            
            if len(pitches_orig) == 0 or len(pitches_user) == 0:
                return 0.5
            
            # Ajustar longitudes
            min_len = min(len(pitches_orig), len(pitches_user))
            pitches_orig = pitches_orig[:min_len]
            pitches_user = pitches_user[:min_len]
            
            # Calcular diferencia relativa
            diff = np.abs(pitches_orig - pitches_user) / pitches_orig
            similarity = 1.0 - np.mean(diff)
            
            return max(0.0, min(1.0, similarity))
        
        except Exception as e:
            logger.warning(f"Error calculating pitch score: {e}")
            return 0.5
    
    def _calculate_timing_score(
        self,
        y_user: np.ndarray,
        sr: int,
        synced_lyrics: Any
    ) -> float:
        """Calcula similitud de timing con letras"""
        # Detectar energía en momentos de palabras
        rms = librosa.feature.rms(y=y_user)[0]
        frame_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
        
        # Verificar si hay energía en momentos esperados de palabras
        word_times = [(w.start_time + w.end_time) / 2 for w in synced_lyrics.words]
        
        if not word_times:
            return 0.5
        
        # Encontrar energía en esos momentos
        scores = []
        for word_time in word_times:
            # Encontrar frame más cercano
            idx = np.argmin(np.abs(frame_times - word_time))
            energy = rms[idx]
            scores.append(energy)
        
        # Normalizar score
        if scores:
            avg_energy = np.mean(scores)
            max_energy = np.max(rms)
            score = avg_energy / max_energy if max_energy > 0 else 0.5
            return float(score)
        
        return 0.5


# Instancia global
_karaoke_service: Optional[KaraokeService] = None


def get_karaoke_service() -> KaraokeService:
    """Obtiene la instancia global del servicio de karaoke"""
    global _karaoke_service
    if _karaoke_service is None:
        _karaoke_service = KaraokeService()
    return _karaoke_service

