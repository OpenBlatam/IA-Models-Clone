"""
Audio Analysis Module

Advanced audio analysis using Essentia, Madmom, and librosa.
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class AudioAnalyzer:
    """
    Comprehensive audio analysis using multiple libraries.
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self._essentia_available = False
        self._madmom_available = False
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check which analysis libraries are available."""
        try:
            import essentia
            import essentia.standard as es
            self._essentia_available = True
            self._essentia_std = es
        except ImportError:
            logger.debug("Essentia not available")
        
        try:
            import madmom
            self._madmom_available = True
        except ImportError:
            logger.debug("Madmom not available")
    
    def analyze_tempo(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Analyze tempo using multiple methods.
        
        Args:
            audio: Audio array
            
        Returns:
            Tempo analysis results
        """
        results = {}
        
        # Using librosa (always available)
        try:
            import librosa
            
            tempo, beats = librosa.beat.beat_track(
                y=audio,
                sr=self.sample_rate
            )
            results["librosa"] = {
                "tempo": float(tempo),
                "beats": beats.tolist()
            }
        except Exception as e:
            logger.warning(f"Librosa tempo analysis failed: {e}")
        
        # Using Essentia if available
        if self._essentia_available:
            try:
                rhythm_extractor = self._essentia_std.RhythmExtractor2013(
                    method="multifeature"
                )
                bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)
                
                results["essentia"] = {
                    "bpm": float(bpm),
                    "beats": beats.tolist() if hasattr(beats, 'tolist') else beats,
                    "confidence": float(beats_confidence)
                }
            except Exception as e:
                logger.warning(f"Essentia tempo analysis failed: {e}")
        
        # Using Madmom if available
        if self._madmom_available:
            try:
                from madmom.features.beats import RNNBeatProcessor
                from madmom.features.beats import BeatTrackingProcessor
                
                beat_processor = RNNBeatProcessor()
                beat_tracker = BeatTrackingProcessor(fps=100)
                
                act = beat_processor(audio)
                beats = beat_tracker(act)
                
                if len(beats) > 1:
                    intervals = np.diff(beats)
                    bpm = 60.0 / np.median(intervals)
                else:
                    bpm = 0.0
                
                results["madmom"] = {
                    "bpm": float(bpm),
                    "beats": beats.tolist()
                }
            except Exception as e:
                logger.warning(f"Madmom tempo analysis failed: {e}")
        
        return results
    
    def analyze_key(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Analyze musical key.
        
        Args:
            audio: Audio array
            
        Returns:
            Key analysis results
        """
        results = {}
        
        if self._essentia_available:
            try:
                key_extractor = self._essentia_std.KeyExtractor()
                key, scale, strength = key_extractor(audio)
                
                results["essentia"] = {
                    "key": key,
                    "scale": scale,
                    "strength": float(strength)
                }
            except Exception as e:
                logger.warning(f"Essentia key analysis failed: {e}")
        
        # Fallback using librosa chroma
        try:
            import librosa
            
            chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
            chroma_mean = np.mean(chroma, axis=1)
            key_idx = np.argmax(chroma_mean)
            
            keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            results["librosa"] = {
                "key": keys[key_idx],
                "chroma_strength": float(chroma_mean[key_idx])
            }
        except Exception as e:
            logger.warning(f"Librosa key analysis failed: {e}")
        
        return results
    
    def analyze_structure(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Analyze musical structure (verse, chorus, etc.).
        
        Args:
            audio: Audio array
            
        Returns:
            Structure analysis results
        """
        results = {}
        
        try:
            import librosa
            
            # Compute self-similarity matrix
            chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
            similarity = librosa.segment.cross_similarity(chroma, chroma)
            
            # Find segments
            boundaries = librosa.segment.agglomerative(
                similarity,
                k=5  # Number of segments
            )
            
            results["segments"] = boundaries.tolist()
            results["num_segments"] = len(boundaries)
            
            # Compute segment features
            segment_features = []
            for i in range(len(boundaries) - 1):
                start = boundaries[i]
                end = boundaries[i + 1]
                segment_audio = audio[start:end]
                
                # Basic features for each segment
                segment_features.append({
                    "start": int(start),
                    "end": int(end),
                    "duration": float((end - start) / self.sample_rate),
                    "energy": float(np.mean(np.abs(segment_audio)))
                })
            
            results["segment_features"] = segment_features
            
        except Exception as e:
            logger.warning(f"Structure analysis failed: {e}")
        
        return results
    
    def analyze_harmony(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Analyze harmonic content.
        
        Args:
            audio: Audio array
            
        Returns:
            Harmony analysis results
        """
        results = {}
        
        try:
            import librosa
            
            # Harmonic and percussive separation
            harmonic, percussive = librosa.effects.hpss(audio)
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=harmonic, sr=self.sample_rate)
            
            # Tonnetz (tonal centroid)
            tonnetz = librosa.feature.tonnetz(y=harmonic, sr=self.sample_rate)
            
            results["harmonic_ratio"] = float(np.mean(np.abs(harmonic)) / (np.mean(np.abs(audio)) + 1e-8))
            results["percussive_ratio"] = float(np.mean(np.abs(percussive)) / (np.mean(np.abs(audio)) + 1e-8))
            results["chroma_mean"] = np.mean(chroma, axis=1).tolist()
            results["tonnetz_mean"] = np.mean(tonnetz, axis=1).tolist()
            
        except Exception as e:
            logger.warning(f"Harmony analysis failed: {e}")
        
        return results
    
    def full_analysis(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Perform full audio analysis.
        
        Args:
            audio: Audio array
            
        Returns:
            Complete analysis results
        """
        return {
            "tempo": self.analyze_tempo(audio),
            "key": self.analyze_key(audio),
            "structure": self.analyze_structure(audio),
            "harmony": self.analyze_harmony(audio),
            "duration": len(audio) / self.sample_rate,
            "sample_rate": self.sample_rate
        }


class BeatTracker:
    """
    Advanced beat tracking using multiple algorithms.
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
    
    def track_beats(self, audio: np.ndarray, method: str = "librosa") -> np.ndarray:
        """
        Track beats in audio.
        
        Args:
            audio: Audio array
            method: Method to use (librosa, madmom, essentia)
            
        Returns:
            Beat times in seconds
        """
        if method == "librosa":
            try:
                import librosa
                tempo, beats = librosa.beat.beat_track(
                    y=audio,
                    sr=self.sample_rate
                )
                beat_times = librosa.frames_to_time(beats, sr=self.sample_rate)
                return beat_times
            except Exception as e:
                logger.error(f"Librosa beat tracking failed: {e}")
                return np.array([])
        
        elif method == "madmom":
            try:
                from madmom.features.beats import RNNBeatProcessor
                from madmom.features.beats import BeatTrackingProcessor
                
                beat_processor = RNNBeatProcessor()
                beat_tracker = BeatTrackingProcessor(fps=100)
                
                act = beat_processor(audio)
                beats = beat_tracker(act)
                
                return beats
            except ImportError:
                logger.warning("Madmom not available, falling back to librosa")
                return self.track_beats(audio, method="librosa")
            except Exception as e:
                logger.error(f"Madmom beat tracking failed: {e}")
                return np.array([])
        
        else:
            raise ValueError(f"Unknown method: {method}")















