"""
Sistema de Sincronización de Letras con Audio

Proporciona:
- Sincronización automática de letras con audio
- Timestamps por palabra
- Visualización de letras en tiempo real
- Detección de sílabas
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import librosa
    import numpy as np
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("Librosa not available, lyrics sync limited")


@dataclass
class WordTiming:
    """Timing de una palabra"""
    word: str
    start_time: float
    end_time: float
    confidence: float = 0.0


@dataclass
class SyncedLyrics:
    """Letras sincronizadas"""
    words: List[WordTiming]
    total_duration: float
    timestamp: datetime = field(default_factory=datetime.now)


class LyricsSynchronizer:
    """Sincronizador de letras con audio"""
    
    def __init__(self):
        logger.info("LyricsSynchronizer initialized")
    
    def sync_lyrics(
        self,
        audio_path: str,
        lyrics_text: str,
        method: str = "energy"
    ) -> SyncedLyrics:
        """
        Sincroniza letras con audio
        
        Args:
            audio_path: Ruta del archivo de audio
            lyrics_text: Texto de las letras
            method: Método de sincronización (energy, beats, manual)
        
        Returns:
            SyncedLyrics
        """
        if not LIBROSA_AVAILABLE:
            # Fallback: distribución uniforme
            return self._uniform_sync(lyrics_text, 180.0)  # 3 minutos por defecto
        
        try:
            # Cargar audio
            y, sr = librosa.load(audio_path, sr=None)
            duration = len(y) / sr
            
            # Dividir letras en palabras
            words = lyrics_text.split()
            
            if method == "energy":
                return self._energy_based_sync(y, sr, words, duration)
            elif method == "beats":
                return self._beat_based_sync(y, sr, words, duration)
            else:
                return self._uniform_sync(words, duration)
        
        except Exception as e:
            logger.error(f"Error syncing lyrics: {e}")
            return self._uniform_sync(lyrics_text.split(), 180.0)
    
    def _energy_based_sync(
        self,
        y: np.ndarray,
        sr: int,
        words: List[str],
        duration: float
    ) -> SyncedLyrics:
        """Sincronización basada en energía del audio"""
        # Calcular energía por frame
        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Encontrar picos de energía (donde hay palabras)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(rms, height=np.mean(rms), distance=len(rms) // len(words))
        
        # Mapear palabras a tiempos
        word_timings = []
        frame_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
        
        # Si hay menos picos que palabras, distribuir uniformemente
        if len(peaks) < len(words):
            time_per_word = duration / len(words)
            for i, word in enumerate(words):
                start_time = i * time_per_word
                end_time = (i + 1) * time_per_word
                word_timings.append(WordTiming(
                    word=word,
                    start_time=start_time,
                    end_time=end_time,
                    confidence=0.5
                ))
        else:
            # Usar picos para sincronizar
            peak_times = [frame_times[p] for p in peaks[:len(words)]]
            
            for i, word in enumerate(words):
                if i < len(peak_times) - 1:
                    start_time = peak_times[i]
                    end_time = peak_times[i + 1]
                else:
                    start_time = peak_times[i] if i < len(peak_times) else duration
                    end_time = duration
                
                word_timings.append(WordTiming(
                    word=word,
                    start_time=start_time,
                    end_time=end_time,
                    confidence=0.7
                ))
        
        return SyncedLyrics(
            words=word_timings,
            total_duration=duration
        )
    
    def _beat_based_sync(
        self,
        y: np.ndarray,
        sr: int,
        words: List[str],
        duration: float
    ) -> SyncedLyrics:
        """Sincronización basada en beats"""
        # Detectar beats
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr)
        
        # Distribuir palabras en beats
        word_timings = []
        
        if len(beat_times) >= len(words):
            # Una palabra por beat
            for i, word in enumerate(words):
                start_time = beat_times[i]
                end_time = beat_times[i + 1] if i + 1 < len(beat_times) else duration
                word_timings.append(WordTiming(
                    word=word,
                    start_time=start_time,
                    end_time=end_time,
                    confidence=0.8
                ))
        else:
            # Múltiples palabras por beat
            words_per_beat = len(words) / len(beat_times)
            word_idx = 0
            
            for i in range(len(beat_times)):
                start_time = beat_times[i]
                end_time = beat_times[i + 1] if i + 1 < len(beat_times) else duration
                
                # Calcular cuántas palabras en este beat
                num_words = int((i + 1) * words_per_beat) - int(i * words_per_beat)
                
                if num_words > 0:
                    time_per_word = (end_time - start_time) / num_words
                    
                    for j in range(num_words):
                        if word_idx < len(words):
                            word_start = start_time + j * time_per_word
                            word_end = start_time + (j + 1) * time_per_word
                            word_timings.append(WordTiming(
                                word=words[word_idx],
                                start_time=word_start,
                                end_time=word_end,
                                confidence=0.7
                            ))
                            word_idx += 1
        
        return SyncedLyrics(
            words=word_timings,
            total_duration=duration
        )
    
    def _uniform_sync(
        self,
        words: List[str],
        duration: float
    ) -> SyncedLyrics:
        """Sincronización uniforme (fallback)"""
        time_per_word = duration / len(words) if words else duration
        
        word_timings = [
            WordTiming(
                word=word,
                start_time=i * time_per_word,
                end_time=(i + 1) * time_per_word,
                confidence=0.3
            )
            for i, word in enumerate(words)
        ]
        
        return SyncedLyrics(
            words=word_timings,
            total_duration=duration
        )
    
    def get_words_at_time(
        self,
        synced_lyrics: SyncedLyrics,
        time: float
    ) -> List[WordTiming]:
        """
        Obtiene palabras activas en un tiempo específico
        
        Args:
            synced_lyrics: Letras sincronizadas
            time: Tiempo en segundos
        
        Returns:
            Lista de palabras activas
        """
        return [
            word for word in synced_lyrics.words
            if word.start_time <= time <= word.end_time
        ]


# Instancia global
_lyrics_synchronizer: Optional[LyricsSynchronizer] = None


def get_lyrics_synchronizer() -> LyricsSynchronizer:
    """Obtiene la instancia global del sincronizador de letras"""
    global _lyrics_synchronizer
    if _lyrics_synchronizer is None:
        _lyrics_synchronizer = LyricsSynchronizer()
    return _lyrics_synchronizer

