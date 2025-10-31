"""
Gamma App - Audio Utilities
Advanced audio processing and manipulation utilities
"""

import io
import base64
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import librosa
import soundfile as sf
import numpy as np
from scipy import signal
from scipy.io import wavfile
import pydub
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range
import noisereduce as nr
import tempfile
import os

logger = logging.getLogger(__name__)

class AudioFormat(Enum):
    """Audio formats"""
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    AAC = "aac"
    OGG = "ogg"
    M4A = "m4a"

class AudioQuality(Enum):
    """Audio quality levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"

@dataclass
class AudioMetadata:
    """Audio metadata"""
    duration: float
    sample_rate: int
    channels: int
    format: str
    bitrate: int
    file_size: int
    has_metadata: bool
    title: Optional[str] = None
    artist: Optional[str] = None
    album: Optional[str] = None
    genre: Optional[str] = None

@dataclass
class AudioSegment:
    """Audio segment data"""
    start_time: float
    end_time: float
    duration: float
    amplitude: float
    frequency: float

class AudioProcessor:
    """Advanced audio processing class"""
    
    def __init__(self):
        self.supported_formats = ['wav', 'mp3', 'flac', 'aac', 'ogg', 'm4a']
        self.max_file_size = 100 * 1024 * 1024  # 100MB
    
    def get_audio_info(self, audio_path: str) -> AudioMetadata:
        """Get comprehensive audio information"""
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=None)
            
            # Get file info
            file_size = Path(audio_path).stat().st_size
            duration = len(y) / sr
            channels = 1 if y.ndim == 1 else y.shape[0]
            
            # Try to get metadata
            try:
                audio_segment = AudioSegment.from_file(audio_path)
                title = audio_segment.metadata.get('title') if hasattr(audio_segment, 'metadata') else None
                artist = audio_segment.metadata.get('artist') if hasattr(audio_segment, 'metadata') else None
                album = audio_segment.metadata.get('album') if hasattr(audio_segment, 'metadata') else None
                genre = audio_segment.metadata.get('genre') if hasattr(audio_segment, 'metadata') else None
                has_metadata = any([title, artist, album, genre])
            except:
                title = artist = album = genre = None
                has_metadata = False
            
            # Estimate bitrate
            bitrate = int((file_size * 8) / duration) if duration > 0 else 0
            
            return AudioMetadata(
                duration=duration,
                sample_rate=sr,
                channels=channels,
                format=Path(audio_path).suffix.lower(),
                bitrate=bitrate,
                file_size=file_size,
                has_metadata=has_metadata,
                title=title,
                artist=artist,
                album=album,
                genre=genre
            )
            
        except Exception as e:
            logger.error(f"Error getting audio info: {e}")
            raise
    
    def load_audio(self, audio_path: str, sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """Load audio file"""
        try:
            y, sr = librosa.load(audio_path, sr=sr)
            return y, sr
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            raise
    
    def save_audio(
        self,
        audio_data: np.ndarray,
        output_path: str,
        sr: int,
        format: AudioFormat = AudioFormat.WAV
    ) -> str:
        """Save audio data to file"""
        try:
            if format == AudioFormat.WAV:
                sf.write(output_path, audio_data, sr)
            else:
                # Use pydub for other formats
                audio_segment = pydub.AudioSegment(
                    audio_data.tobytes(),
                    frame_rate=sr,
                    sample_width=audio_data.dtype.itemsize,
                    channels=1 if audio_data.ndim == 1 else audio_data.shape[0]
                )
                audio_segment.export(output_path, format=format.value)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving audio: {e}")
            raise
    
    def normalize_audio(
        self,
        audio_path: str,
        output_path: str,
        target_lufs: float = -23.0
    ) -> str:
        """Normalize audio levels"""
        try:
            audio_segment = AudioSegment.from_file(audio_path)
            normalized_audio = normalize(audio_segment)
            normalized_audio.export(output_path, format=Path(output_path).suffix[1:])
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error normalizing audio: {e}")
            raise
    
    def compress_audio(
        self,
        audio_path: str,
        output_path: str,
        threshold: float = -20.0,
        ratio: float = 4.0,
        attack: float = 5.0,
        release: float = 50.0
    ) -> str:
        """Apply dynamic range compression"""
        try:
            audio_segment = AudioSegment.from_file(audio_path)
            compressed_audio = compress_dynamic_range(
                audio_segment,
                threshold=threshold,
                ratio=ratio,
                attack=attack,
                release=release
            )
            compressed_audio.export(output_path, format=Path(output_path).suffix[1:])
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error compressing audio: {e}")
            raise
    
    def reduce_noise(
        self,
        audio_path: str,
        output_path: str,
        noise_reduction_strength: float = 0.8
    ) -> str:
        """Reduce background noise"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=None)
            
            # Reduce noise
            reduced_noise = nr.reduce_noise(y=y, sr=sr, prop_decrease=noise_reduction_strength)
            
            # Save processed audio
            sf.write(output_path, reduced_noise, sr)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error reducing noise: {e}")
            raise
    
    def trim_silence(
        self,
        audio_path: str,
        output_path: str,
        silence_threshold: float = -40.0,
        min_silence_duration: int = 1000
    ) -> str:
        """Trim silence from beginning and end"""
        try:
            audio_segment = AudioSegment.from_file(audio_path)
            trimmed_audio = audio_segment.strip_silence(
                silence_thresh=silence_threshold,
                silence_len=min_silence_duration
            )
            trimmed_audio.export(output_path, format=Path(output_path).suffix[1:])
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error trimming silence: {e}")
            raise
    
    def fade_in_out(
        self,
        audio_path: str,
        output_path: str,
        fade_in_duration: int = 1000,
        fade_out_duration: int = 1000
    ) -> str:
        """Add fade in and fade out effects"""
        try:
            audio_segment = AudioSegment.from_file(audio_path)
            
            if fade_in_duration > 0:
                audio_segment = audio_segment.fade_in(fade_in_duration)
            
            if fade_out_duration > 0:
                audio_segment = audio_segment.fade_out(fade_out_duration)
            
            audio_segment.export(output_path, format=Path(output_path).suffix[1:])
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error adding fade effects: {e}")
            raise
    
    def change_speed(
        self,
        audio_path: str,
        output_path: str,
        speed_factor: float
    ) -> str:
        """Change audio playback speed"""
        try:
            audio_segment = AudioSegment.from_file(audio_path)
            speeded_audio = audio_segment.speedup(playback_speed=speed_factor)
            speeded_audio.export(output_path, format=Path(output_path).suffix[1:])
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error changing speed: {e}")
            raise
    
    def change_pitch(
        self,
        audio_path: str,
        output_path: str,
        pitch_shift: float
    ) -> str:
        """Change audio pitch"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=None)
            
            # Shift pitch
            y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_shift)
            
            # Save processed audio
            sf.write(output_path, y_shifted, sr)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error changing pitch: {e}")
            raise
    
    def apply_high_pass_filter(
        self,
        audio_path: str,
        output_path: str,
        cutoff_frequency: float
    ) -> str:
        """Apply high-pass filter"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=None)
            
            # Design filter
            nyquist = sr / 2
            normalized_cutoff = cutoff_frequency / nyquist
            b, a = signal.butter(4, normalized_cutoff, btype='high')
            
            # Apply filter
            filtered_audio = signal.filtfilt(b, a, y)
            
            # Save processed audio
            sf.write(output_path, filtered_audio, sr)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error applying high-pass filter: {e}")
            raise
    
    def apply_low_pass_filter(
        self,
        audio_path: str,
        output_path: str,
        cutoff_frequency: float
    ) -> str:
        """Apply low-pass filter"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=None)
            
            # Design filter
            nyquist = sr / 2
            normalized_cutoff = cutoff_frequency / nyquist
            b, a = signal.butter(4, normalized_cutoff, btype='low')
            
            # Apply filter
            filtered_audio = signal.filtfilt(b, a, y)
            
            # Save processed audio
            sf.write(output_path, filtered_audio, sr)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error applying low-pass filter: {e}")
            raise
    
    def apply_band_pass_filter(
        self,
        audio_path: str,
        output_path: str,
        low_cutoff: float,
        high_cutoff: float
    ) -> str:
        """Apply band-pass filter"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=None)
            
            # Design filter
            nyquist = sr / 2
            low_norm = low_cutoff / nyquist
            high_norm = high_cutoff / nyquist
            b, a = signal.butter(4, [low_norm, high_norm], btype='band')
            
            # Apply filter
            filtered_audio = signal.filtfilt(b, a, y)
            
            # Save processed audio
            sf.write(output_path, filtered_audio, sr)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error applying band-pass filter: {e}")
            raise
    
    def extract_spectral_features(
        self,
        audio_path: str,
        n_mfcc: int = 13,
        n_fft: int = 2048,
        hop_length: int = 512
    ) -> Dict[str, np.ndarray]:
        """Extract spectral features from audio"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=None)
            
            # Extract features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
            
            return {
                'mfccs': mfccs,
                'spectral_centroids': spectral_centroids,
                'spectral_rolloff': spectral_rolloff,
                'zero_crossing_rate': zero_crossing_rate,
                'chroma': chroma,
                'tonnetz': tonnetz
            }
            
        except Exception as e:
            logger.error(f"Error extracting spectral features: {e}")
            raise
    
    def detect_beats(
        self,
        audio_path: str,
        hop_length: int = 512
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detect beats and tempo"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=None)
            
            # Detect tempo and beats
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
            
            # Convert beat frames to time
            beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)
            
            return beat_times, tempo
            
        except Exception as e:
            logger.error(f"Error detecting beats: {e}")
            raise
    
    def segment_audio(
        self,
        audio_path: str,
        output_dir: str,
        segment_length: float = 30.0,
        overlap: float = 0.0
    ) -> List[str]:
        """Segment audio into smaller chunks"""
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            audio_segment = AudioSegment.from_file(audio_path)
            segment_length_ms = int(segment_length * 1000)
            overlap_ms = int(overlap * 1000)
            
            segments = []
            start = 0
            segment_count = 0
            
            while start < len(audio_segment):
                end = min(start + segment_length_ms, len(audio_segment))
                segment = audio_segment[start:end]
                
                output_file = output_dir / f"segment_{segment_count:04d}.wav"
                segment.export(str(output_file), format="wav")
                segments.append(str(output_file))
                
                start = end - overlap_ms
                segment_count += 1
            
            return segments
            
        except Exception as e:
            logger.error(f"Error segmenting audio: {e}")
            raise
    
    def merge_audio_files(
        self,
        audio_paths: List[str],
        output_path: str,
        crossfade_duration: int = 0
    ) -> str:
        """Merge multiple audio files"""
        try:
            if not audio_paths:
                raise ValueError("No audio files provided")
            
            # Load first audio file
            merged_audio = AudioSegment.from_file(audio_paths[0])
            
            # Merge with remaining files
            for audio_path in audio_paths[1:]:
                audio_segment = AudioSegment.from_file(audio_path)
                merged_audio = merged_audio.append(audio_segment, crossfade=crossfade_duration)
            
            # Export merged audio
            merged_audio.export(output_path, format=Path(output_path).suffix[1:])
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error merging audio files: {e}")
            raise
    
    def mix_audio_files(
        self,
        audio_paths: List[str],
        output_path: str,
        volumes: Optional[List[float]] = None
    ) -> str:
        """Mix multiple audio files"""
        try:
            if not audio_paths:
                raise ValueError("No audio files provided")
            
            # Load first audio file
            mixed_audio = AudioSegment.from_file(audio_paths[0])
            
            if volumes and len(volumes) > 0:
                mixed_audio = mixed_audio + volumes[0]
            
            # Mix with remaining files
            for i, audio_path in enumerate(audio_paths[1:], 1):
                audio_segment = AudioSegment.from_file(audio_path)
                
                if volumes and i < len(volumes):
                    audio_segment = audio_segment + volumes[i]
                
                mixed_audio = mixed_audio.overlay(audio_segment)
            
            # Export mixed audio
            mixed_audio.export(output_path, format=Path(output_path).suffix[1:])
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error mixing audio files: {e}")
            raise
    
    def convert_format(
        self,
        audio_path: str,
        output_path: str,
        target_format: AudioFormat,
        quality: AudioQuality = AudioQuality.MEDIUM
    ) -> str:
        """Convert audio to different format"""
        try:
            audio_segment = AudioSegment.from_file(audio_path)
            
            # Set quality parameters
            quality_params = {
                AudioQuality.LOW: {'bitrate': '128k'},
                AudioQuality.MEDIUM: {'bitrate': '192k'},
                AudioQuality.HIGH: {'bitrate': '320k'},
                AudioQuality.ULTRA: {'bitrate': '320k', 'parameters': ['-q:a', '0']}
            }
            
            params = quality_params.get(quality, {})
            audio_segment.export(output_path, format=target_format.value, **params)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error converting format: {e}")
            raise
    
    def create_spectrogram(
        self,
        audio_path: str,
        output_path: str,
        n_fft: int = 2048,
        hop_length: int = 512,
        cmap: str = 'viridis'
    ) -> str:
        """Create audio spectrogram"""
        try:
            import matplotlib.pyplot as plt
            
            # Load audio
            y, sr = librosa.load(audio_path, sr=None)
            
            # Create spectrogram
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
            
            # Plot spectrogram
            plt.figure(figsize=(12, 8))
            librosa.display.specshow(D, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', cmap=cmap)
            plt.colorbar(format='%+2.0f dB')
            plt.title('Spectrogram')
            plt.xlabel('Time')
            plt.ylabel('Frequency (Hz)')
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating spectrogram: {e}")
            raise
    
    def validate_audio(self, audio_path: str) -> Dict[str, Any]:
        """Validate audio file"""
        try:
            validation_result = {
                'valid': True,
                'errors': [],
                'warnings': [],
                'info': {}
            }
            
            # Check if file exists
            if not Path(audio_path).exists():
                validation_result['valid'] = False
                validation_result['errors'].append("File does not exist")
                return validation_result
            
            # Check file size
            file_size = Path(audio_path).stat().st_size
            if file_size > self.max_file_size:
                validation_result['warnings'].append(f"File size ({file_size} bytes) exceeds maximum ({self.max_file_size} bytes)")
            
            # Check file extension
            file_ext = Path(audio_path).suffix.lower()
            if file_ext not in ['.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a']:
                validation_result['warnings'].append(f"File extension {file_ext} may not be supported")
            
            # Try to load audio
            try:
                y, sr = librosa.load(audio_path, sr=None)
                validation_result['info']['duration'] = len(y) / sr
                validation_result['info']['sample_rate'] = sr
                validation_result['info']['channels'] = 1 if y.ndim == 1 else y.shape[0]
                validation_result['info']['samples'] = len(y)
                
            except Exception as e:
                validation_result['valid'] = False
                validation_result['errors'].append(f"Cannot load audio: {str(e)}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating audio: {e}")
            return {
                'valid': False,
                'errors': [f"Validation error: {str(e)}"],
                'warnings': [],
                'info': {}
            }

# Global audio processor instance
audio_processor = AudioProcessor()

def get_audio_info(audio_path: str) -> AudioMetadata:
    """Get audio info using global processor"""
    return audio_processor.get_audio_info(audio_path)

def normalize_audio_file(audio_path: str, output_path: str, target_lufs: float = -23.0) -> str:
    """Normalize audio using global processor"""
    return audio_processor.normalize_audio(audio_path, output_path, target_lufs)

def compress_audio_file(audio_path: str, output_path: str, threshold: float = -20.0, ratio: float = 4.0) -> str:
    """Compress audio using global processor"""
    return audio_processor.compress_audio(audio_path, output_path, threshold, ratio)

def reduce_audio_noise(audio_path: str, output_path: str, noise_reduction_strength: float = 0.8) -> str:
    """Reduce noise using global processor"""
    return audio_processor.reduce_noise(audio_path, output_path, noise_reduction_strength)

def trim_audio_silence(audio_path: str, output_path: str, silence_threshold: float = -40.0) -> str:
    """Trim silence using global processor"""
    return audio_processor.trim_silence(audio_path, output_path, silence_threshold)

def change_audio_speed(audio_path: str, output_path: str, speed_factor: float) -> str:
    """Change speed using global processor"""
    return audio_processor.change_speed(audio_path, output_path, speed_factor)

def change_audio_pitch(audio_path: str, output_path: str, pitch_shift: float) -> str:
    """Change pitch using global processor"""
    return audio_processor.change_pitch(audio_path, output_path, pitch_shift)

def merge_audio_files(audio_paths: List[str], output_path: str, crossfade_duration: int = 0) -> str:
    """Merge audio files using global processor"""
    return audio_processor.merge_audio_files(audio_paths, output_path, crossfade_duration)

def convert_audio_format(audio_path: str, output_path: str, target_format: AudioFormat, quality: AudioQuality = AudioQuality.MEDIUM) -> str:
    """Convert audio format using global processor"""
    return audio_processor.convert_format(audio_path, output_path, target_format, quality)

def create_audio_spectrogram(audio_path: str, output_path: str, n_fft: int = 2048, hop_length: int = 512) -> str:
    """Create spectrogram using global processor"""
    return audio_processor.create_spectrogram(audio_path, output_path, n_fft, hop_length)

























