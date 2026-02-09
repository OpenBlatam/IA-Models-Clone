"""
Voice Synthesis Module

Text-to-Speech and voice cloning utilities.
"""

from typing import Optional, Union, List
import logging
import numpy as np

logger = logging.getLogger(__name__)


class VoiceSynthesizer:
    """
    Voice synthesis using Coqui TTS.
    """
    
    def __init__(
        self,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v3",
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.device = device
        self.tts = None
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize TTS model."""
        try:
            from TTS.api import TTS
            
            logger.info(f"Loading TTS model: {self.model_name}")
            self.tts = TTS(self.model_name)
            if self.device:
                self.tts.to(self.device)
            self._initialized = True
            logger.info("TTS model loaded successfully")
        except ImportError:
            raise ImportError(
                "TTS not installed. Install with: pip install TTS"
            )
        except Exception as e:
            logger.error(f"Error loading TTS model: {e}")
            raise
    
    def synthesize(
        self,
        text: str,
        output_path: Optional[str] = None,
        speaker_wav: Optional[str] = None,
        language: str = "en"
    ) -> Optional[np.ndarray]:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            output_path: Optional path to save audio file
            speaker_wav: Optional reference speaker audio for voice cloning
            language: Language code
            
        Returns:
            Audio array if output_path not provided, None otherwise
        """
        if not self._initialized:
            self.initialize()
        
        try:
            if output_path:
                self.tts.tts_to_file(
                    text=text,
                    file_path=output_path,
                    speaker_wav=speaker_wav,
                    language=language
                )
                return None
            else:
                # Generate to memory
                wav = self.tts.tts(
                    text=text,
                    speaker_wav=speaker_wav,
                    language=language
                )
                return np.array(wav)
        except Exception as e:
            logger.error(f"Error synthesizing speech: {e}")
            raise
    
    def synthesize_batch(
        self,
        texts: List[str],
        output_dir: Optional[str] = None,
        speaker_wav: Optional[str] = None,
        language: str = "en"
    ) -> List[np.ndarray]:
        """
        Synthesize speech for multiple texts.
        
        Args:
            texts: List of texts to synthesize
            output_dir: Optional directory to save audio files
            speaker_wav: Optional reference speaker audio
            language: Language code
            
        Returns:
            List of audio arrays
        """
        if not self._initialized:
            self.initialize()
        
        results = []
        for i, text in enumerate(texts):
            if output_dir:
                output_path = f"{output_dir}/output_{i}.wav"
                self.synthesize(text, output_path, speaker_wav, language)
                results.append(None)
            else:
                audio = self.synthesize(text, None, speaker_wav, language)
                results.append(audio)
        
        return results


class VoiceCloner:
    """
    Voice cloning using XTTS.
    """
    
    def __init__(
        self,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v3"
    ):
        self.model_name = model_name
        self.synthesizer = VoiceSynthesizer(model_name)
    
    def clone_voice(
        self,
        text: str,
        reference_audio: str,
        output_path: Optional[str] = None,
        language: str = "en"
    ) -> Optional[np.ndarray]:
        """
        Clone voice from reference audio.
        
        Args:
            text: Text to synthesize
            reference_audio: Path to reference audio file (min 6 seconds)
            output_path: Optional path to save output
            language: Language code
            
        Returns:
            Audio array with cloned voice
        """
        return self.synthesizer.synthesize(
            text=text,
            output_path=output_path,
            speaker_wav=reference_audio,
            language=language
        )















