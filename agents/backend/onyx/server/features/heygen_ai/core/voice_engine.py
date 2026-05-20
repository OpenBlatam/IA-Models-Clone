#!/usr/bin/env python3
"""
Voice Engine for HeyGen AI
==========================

Production-ready voice synthesis system using TTS models.
Follows best practices for audio processing and TTS.

Key Features:
- Multiple TTS engines (Coqui TTS, YourTTS, XTTS)
- Voice cloning support
- Audio post-processing
- Multi-language support
- Proper error handling and logging
"""

import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

# Third-party imports with proper error handling
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logging.warning("Librosa not available. Install with: pip install librosa")

try:
    import torch
    import torchaudio
    TORCH_AUDIO_AVAILABLE = True
except ImportError:
    TORCH_AUDIO_AVAILABLE = False
    logging.warning(
        "TorchAudio not available. Install with: pip install torchaudio"
    )

try:
    from TTS.api import TTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    logging.warning("TTS not available. Install with: pip install TTS")

try:
    from pydub import AudioSegment
    from pydub.effects import normalize
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    logging.warning("Pydub not available. Install with: pip install pydub")

logger = logging.getLogger(__name__)


# =============================================================================
# Imports from shared module
# =============================================================================

from shared import (
    VoiceQuality,
    AudioFormat,
    VoiceGenerationConfig,
    VoiceModel,
)

# =============================================================================
# Imports from utility helpers
# =============================================================================

from utils.gpu_error_handler import handle_gpu_errors

# =============================================================================
# Legacy Enums (deprecated - use shared module)
# =============================================================================

class _LegacyVoiceQuality(str, Enum):
    """Voice quality levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


class _LegacyAudioFormat(str, Enum):
    """Audio format options."""
    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"
    FLAC = "flac"


@dataclass
class _LegacyVoiceGenerationConfig:
    """Configuration for voice generation.
    
    Attributes:
        sample_rate: Audio sample rate in Hz
        bit_depth: Audio bit depth (16, 24, 32)
        channels: Number of audio channels (1=mono, 2=stereo)
        format: Output audio format
        quality: Generation quality level
        normalize: Normalize audio levels
        remove_silence: Remove leading/trailing silence
        speed: Speech speed multiplier (0.5-2.0)
        pitch: Pitch shift in semitones (-12 to 12)
        volume: Volume multiplier (0.0-2.0)
    """
    sample_rate: int = 22050
    bit_depth: int = 16
    channels: int = 1
    format: AudioFormat = AudioFormat.WAV
    quality: VoiceQuality = VoiceQuality.HIGH
    normalize: bool = True
    remove_silence: bool = True
    speed: float = 1.0
    pitch: float = 0.0
    volume: float = 1.0

    def get_sample_rate(self) -> int:
        """Get sample rate based on quality."""
        quality_rates = {
            VoiceQuality.LOW: 16000,
            VoiceQuality.MEDIUM: 22050,
            VoiceQuality.HIGH: 22050,
            VoiceQuality.ULTRA: 44100,
        }
        return quality_rates.get(self.quality, self.sample_rate)

    def validate(self) -> None:
        """Validate configuration parameters."""
        if not 0.5 <= self.speed <= 2.0:
            raise ValueError("Speed must be between 0.5 and 2.0")
        if not -12 <= self.pitch <= 12:
            raise ValueError("Pitch must be between -12 and 12 semitones")
        if not 0.0 <= self.volume <= 2.0:
            raise ValueError("Volume must be between 0.0 and 2.0")


@dataclass
class VoiceModel:
    """Voice model configuration.
    
    Attributes:
        id: Unique identifier
        name: Display name
        language: Language code (ISO 639-1)
        model_path: Path to model or HuggingFace model ID
        supports_cloning: Whether model supports voice cloning
        supports_emotion: Whether model supports emotion control
    """
    id: str
    name: str
    language: str
    model_path: str
    supports_cloning: bool = False
    supports_emotion: bool = False
    characteristics: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# TTS Engine Manager
# =============================================================================

class TTSEngineManager:
    """Manages TTS engines with proper initialization and error handling.
    
    Features:
    - Multiple TTS engine support
    - Automatic device detection
    - Model loading and caching
    - Proper error handling
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """Initialize TTS engine manager.
        
        Args:
            device: PyTorch device. Auto-detected if None.
        """
        if not TTS_AVAILABLE:
            raise RuntimeError(
                "TTS library not available. Install with: pip install TTS"
            )
        
        self.device = device or self._detect_device()
        self.models: Dict[str, Any] = {}
        self.logger = logging.getLogger(f"{__name__}.TTSEngineManager")
        
    def _detect_device(self) -> torch.device:
        """Detect and return appropriate device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def load_model(
        self,
        model_id: str,
        model_name: str = "coqui_tts",
    ) -> None:
        """Load a TTS model.
        
        Args:
            model_id: HuggingFace model ID or local path
            model_name: Internal name for the model
        
        Raises:
            RuntimeError: If loading fails
        """
        try:
            self.logger.info(f"Loading TTS model: {model_id} on {self.device}")
            
            tts_model = TTS(model_id)
            
            # Move to device if model supports it
            if hasattr(tts_model, "to"):
                tts_model = tts_model.to(self.device)
            
            self.models[model_name] = tts_model
            self.logger.info(f"TTS model loaded: {model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to load TTS model {model_id}: {e}")
            raise RuntimeError(f"TTS model loading failed: {e}") from e
    
    def get_model(self, model_name: str) -> Optional[Any]:
        """Get loaded model by name."""
        return self.models.get(model_name)
    
    def synthesize(
        self,
        text: str,
        model_name: str = "coqui_tts",
        speaker: Optional[str] = None,
        language: Optional[str] = None,
    ) -> np.ndarray:
        """Synthesize speech from text.
        
        Args:
            text: Input text to synthesize
            model_name: Name of model to use
            speaker: Optional speaker ID for multi-speaker models
            language: Optional language code
        
        Returns:
            Audio waveform as numpy array
        
        Raises:
            RuntimeError: If synthesis fails
        """
        model = self.get_model(model_name)
        if model is None:
            raise RuntimeError(f"Model {model_name} not loaded")
        
        try:
            # Prepare synthesis parameters
            kwargs = {}
            if speaker:
                kwargs["speaker"] = speaker
            if language:
                kwargs["language"] = language
            
            # Synthesize with mixed precision if on CUDA
            def _synthesize_audio():
                with torch.no_grad():
                    if self.device.type == "cuda":
                        with torch.cuda.amp.autocast():
                            return model.tts(text=text, **kwargs)
                    else:
                        return model.tts(text=text, **kwargs)
            
            # Use helper function for GPU error handling
            audio = handle_gpu_errors(
                _synthesize_audio,
                operation_name="Speech synthesis"
            )
            
            # Convert to numpy if needed and move to CPU
            if isinstance(audio, torch.Tensor):
                audio = audio.detach().cpu().numpy()
            
            return audio


# =============================================================================
# Audio Processing Utilities
# =============================================================================

class AudioProcessor:
    """Utility class for audio processing operations."""
    
    @staticmethod
    def preprocess_text(text: str, language: str = "en") -> str:
        """Preprocess text for TTS synthesis.
        
        Args:
            text: Input text
            language: Language code
        
        Returns:
            Preprocessed text
        """
        try:
            # Basic cleaning
            processed = text.strip()
            
            # Language-specific preprocessing
            if language == "en":
                processed = processed.replace("&", " and ")
                processed = processed.replace("@", " at ")
                processed = processed.replace("#", " number ")
            
            # Remove excessive whitespace
            processed = " ".join(processed.split())
            
            return processed
            
        except Exception as e:
            logging.warning(f"Text preprocessing failed: {e}")
            return text
    
    @staticmethod
    def post_process_audio(
        audio: np.ndarray,
        config: VoiceGenerationConfig,
    ) -> np.ndarray:
        """Post-process generated audio.
        
        Args:
            audio: Input audio waveform
            config: Processing configuration
        
        Returns:
            Processed audio waveform
        """
        try:
            processed = audio.copy()
            
            # Normalize if requested
            if config.normalize and PYDUB_AVAILABLE:
                # Convert to AudioSegment for processing
                temp_path = Path(f"/tmp/temp_{uuid.uuid4().hex[:8]}.wav")
                sf.write(str(temp_path), processed, config.sample_rate)
                
                audio_seg = AudioSegment.from_wav(str(temp_path))
                audio_seg = normalize(audio_seg)
                
                processed = np.array(audio_seg.get_array_of_samples())
                if audio_seg.channels == 2:
                    processed = processed.reshape((-1, 2))
                
                temp_path.unlink()
            
            # Remove silence if requested
            if config.remove_silence and LIBROSA_AVAILABLE:
                processed, _ = librosa.effects.trim(
                    processed,
                    top_db=20,
                    frame_length=2048,
                    hop_length=512,
                )
            
            # Apply speed change
            if config.speed != 1.0 and LIBROSA_AVAILABLE:
                processed = librosa.effects.time_stretch(
                    processed,
                    rate=config.speed,
                )
            
            # Apply pitch shift
            if config.pitch != 0.0 and LIBROSA_AVAILABLE:
                processed = librosa.effects.pitch_shift(
                    processed,
                    sr=config.sample_rate,
                    n_steps=config.pitch,
                )
            
            # Apply volume
            if config.volume != 1.0:
                processed = processed * config.volume
            
            return processed
            
        except Exception as e:
            logging.warning(f"Audio post-processing failed: {e}")
            return audio
    
    @staticmethod
    def save_audio(
        audio: np.ndarray,
        output_path: str,
        sample_rate: int,
        format: AudioFormat = AudioFormat.WAV,
    ) -> None:
        """Save audio to file.
        
        Args:
            audio: Audio waveform
            output_path: Output file path
            sample_rate: Sample rate in Hz
            format: Audio format
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format == AudioFormat.WAV:
                sf.write(str(output_path), audio, sample_rate)
            elif format == AudioFormat.MP3 and PYDUB_AVAILABLE:
                # Convert to MP3 using pydub
                temp_wav = output_path.with_suffix(".wav")
                sf.write(str(temp_wav), audio, sample_rate)
                
                audio_seg = AudioSegment.from_wav(str(temp_wav))
                audio_seg.export(str(output_path), format="mp3")
                temp_wav.unlink()
            else:
                # Default to WAV
                sf.write(str(output_path), audio, sample_rate)
            
        except Exception as e:
            logging.error(f"Failed to save audio: {e}")
            raise


# =============================================================================
# Voice Engine
# =============================================================================

class VoiceEngine:
    """Main voice synthesis engine.
    
    Features:
    - Text-to-speech synthesis
    - Voice cloning support
    - Audio post-processing
    - Multi-language support
    - Proper error handling
    """
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        default_model: str = "tts_models/en/ljspeech/tacotron2-DDC",
    ):
        """Initialize voice engine.
        
        Args:
            device: PyTorch device (auto-detected if None)
            default_model: Default TTS model ID
        """
        self.logger = logging.getLogger(f"{__name__}.VoiceEngine")
        
        # Initialize TTS engine manager
        try:
            self.tts_manager = TTSEngineManager(device=device)
            self.tts_manager.load_model(default_model, "default")
        except RuntimeError as e:
            self.logger.error(f"Failed to initialize TTS: {e}")
            raise
        
        # Initialize audio processor
        self.audio_processor = AudioProcessor()
        
        self.logger.info("Voice Engine initialized successfully")
    
    async def generate_voice(
        self,
        text: str,
        config: Optional[VoiceGenerationConfig] = None,
        speaker: Optional[str] = None,
        language: str = "en",
    ) -> str:
        """Generate voice from text.
        
        Args:
            text: Text to synthesize
            config: Generation configuration
            speaker: Optional speaker ID
            language: Language code
        
        Returns:
            Path to generated audio file
        
        Raises:
            RuntimeError: If generation fails
        """
        if config is None:
            config = VoiceGenerationConfig()
        
        try:
            config.validate()
            
            self.logger.info(f"Generating voice for text: {text[:50]}...")
            
            # Preprocess text
            processed_text = self.audio_processor.preprocess_text(text, language)
            
            # Synthesize speech
            audio = self.tts_manager.synthesize(
                text=processed_text,
                model_name="default",
                speaker=speaker,
                language=language,
            )
            
            # Post-process audio
            processed_audio = self.audio_processor.post_process_audio(
                audio, config
            )
            
            # Save audio
            output_dir = Path("./generated_audio")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_dir / f"voice_{uuid.uuid4().hex[:8]}.{config.format.value}"
            self.audio_processor.save_audio(
                processed_audio,
                str(output_path),
                config.get_sample_rate(),
                config.format,
            )
            
            self.logger.info(f"Voice generated: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Voice generation failed: {e}")
            raise RuntimeError(f"Generation failed: {e}") from e
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check.
        
        Returns:
            Health status dictionary
        """
        return {
            "status": "healthy",
            "device": str(self.tts_manager.device),
            "models_loaded": len(self.tts_manager.models),
            "tts_available": TTS_AVAILABLE,
            "librosa_available": LIBROSA_AVAILABLE,
            "pydub_available": PYDUB_AVAILABLE,
        }
