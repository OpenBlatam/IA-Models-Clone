"""
Sistema de Análisis de Audio Avanzado

Proporciona:
- Detección de BPM (tempo)
- Detección de key (tonalidad)
- Análisis de energía
- Detección de beats
- Análisis espectral
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
    logger.warning("Librosa not available, audio analysis limited")


@dataclass
class AudioAnalysis:
    """Resultado de análisis de audio"""
    bpm: float = 0.0
    key: str = "unknown"
    energy: float = 0.0
    beats: List[float] = field(default_factory=list)
    tempo: float = 0.0
    spectral_centroid: float = 0.0
    zero_crossing_rate: float = 0.0
    mfcc: List[float] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class AdvancedAudioAnalyzer:
    """Analizador avanzado de audio"""
    
    def __init__(self):
        logger.info("AdvancedAudioAnalyzer initialized")
    
    def analyze(self, audio_path: str) -> AudioAnalysis:
        """
        Analiza un archivo de audio
        
        Args:
            audio_path: Ruta del archivo
        
        Returns:
            AudioAnalysis
        """
        if not LIBROSA_AVAILABLE:
            return AudioAnalysis()
        
        try:
            # Cargar audio
            y, sr = librosa.load(audio_path, sr=None)
            
            # BPM/Tempo
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            bpm = float(tempo)
            
            # Key detection (simplificado)
            key = self._detect_key(y, sr)
            
            # Energy
            energy = float(np.mean(librosa.feature.rms(y=y)[0]))
            
            # Spectral features
            spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0]))
            zero_crossing_rate = float(np.mean(librosa.feature.zero_crossing_rate(y)[0]))
            
            # MFCC (primeros 13 coeficientes)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = [float(x) for x in np.mean(mfcc, axis=1)]
            
            # Beat times
            beat_times = librosa.frames_to_time(beats, sr=sr)
            beat_times_list = [float(t) for t in beat_times]
            
            analysis = AudioAnalysis(
                bpm=bpm,
                key=key,
                energy=energy,
                beats=beat_times_list,
                tempo=bpm,
                spectral_centroid=spectral_centroid,
                zero_crossing_rate=zero_crossing_rate,
                mfcc=mfcc_mean
            )
            
            logger.info(f"Audio analyzed: BPM={bpm}, Key={key}")
            return analysis
        
        except Exception as e:
            logger.error(f"Error analyzing audio: {e}")
            return AudioAnalysis()
    
    def _detect_key(self, y: np.ndarray, sr: int) -> str:
        """
        Detecta la tonalidad del audio (simplificado)
        
        Args:
            y: Audio data
            sr: Sample rate
        
        Returns:
            Key (ej: "C major", "A minor")
        """
        try:
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            
            # Keys (simplificado - solo mayores y menores básicas)
            major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
            minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
            
            keys = [
                "C major", "C# major", "D major", "D# major", "E major", "F major",
                "F# major", "G major", "G# major", "A major", "A# major", "B major",
                "C minor", "C# minor", "D minor", "D# minor", "E minor", "F minor",
                "F# minor", "G minor", "G# minor", "A minor", "A# minor", "B minor"
            ]
            
            correlations = []
            for i in range(12):
                # Major
                major_corr = np.corrcoef(
                    np.roll(major_profile, i),
                    chroma_mean
                )[0, 1]
                correlations.append(major_corr)
            
            for i in range(12):
                # Minor
                minor_corr = np.corrcoef(
                    np.roll(minor_profile, i),
                    chroma_mean
                )[0, 1]
                correlations.append(minor_corr)
            
            # Encontrar key con mayor correlación
            max_idx = np.argmax(correlations)
            return keys[max_idx]
        
        except Exception as e:
            logger.warning(f"Error detecting key: {e}")
            return "unknown"
    
    def analyze_segment(
        self,
        audio_path: str,
        start_time: float,
        end_time: float
    ) -> AudioAnalysis:
        """
        Analiza un segmento específico del audio
        
        Args:
            audio_path: Ruta del archivo
            start_time: Tiempo de inicio (segundos)
            end_time: Tiempo de fin (segundos)
        
        Returns:
            AudioAnalysis del segmento
        """
        if not LIBROSA_AVAILABLE:
            return AudioAnalysis()
        
        try:
            y, sr = librosa.load(audio_path, sr=None, offset=start_time, duration=end_time - start_time)
            
            # Guardar temporalmente y analizar
            import tempfile
            import soundfile as sf
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, y, sr)
                return self.analyze(tmp.name)
        
        except Exception as e:
            logger.error(f"Error analyzing segment: {e}")
            return AudioAnalysis()
    
    def compare_audio(self, audio1_path: str, audio2_path: str) -> Dict[str, Any]:
        """
        Compara dos archivos de audio
        
        Args:
            audio1_path: Ruta del primer archivo
            audio2_path: Ruta del segundo archivo
        
        Returns:
            Comparación
        """
        analysis1 = self.analyze(audio1_path)
        analysis2 = self.analyze(audio2_path)
        
        return {
            "audio1": {
                "bpm": analysis1.bpm,
                "key": analysis1.key,
                "energy": analysis1.energy
            },
            "audio2": {
                "bpm": analysis2.bpm,
                "key": analysis2.key,
                "energy": analysis2.energy
            },
            "similarity": {
                "bpm_diff": abs(analysis1.bpm - analysis2.bpm),
                "key_match": analysis1.key == analysis2.key,
                "energy_diff": abs(analysis1.energy - analysis2.energy)
            }
        }


# Instancia global
_audio_analyzer: Optional[AdvancedAudioAnalyzer] = None


def get_audio_analyzer() -> AdvancedAudioAnalyzer:
    """Obtiene la instancia global del analizador de audio"""
    global _audio_analyzer
    if _audio_analyzer is None:
        _audio_analyzer = AdvancedAudioAnalyzer()
    return _audio_analyzer

