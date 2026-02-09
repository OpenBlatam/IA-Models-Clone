"""
Main audio separator interface.

Refactored to:
- Use base separator for common functionality
- Extract methods for better organization
- Improve error handling and validation
- Better separation of concerns
"""

from pathlib import Path
from typing import Dict, Optional, Union, Any
import numpy as np
import torch

from .base_separator import BaseSeparator
from .constants import (
    DEFAULT_SAMPLE_RATE,
    DEFAULT_MODEL_TYPE,
    VALID_MODEL_TYPES,
    DEFAULT_OUTPUT_DIR_NAME,
    ERROR_CODE_SAVE_FAILED,
    ERROR_CODE_SEPARATION_FAILED,
    ERROR_CODE_SEPARATION_PIPELINE_FAILED
)
from ..model_builder import build_audio_separator_model
from ..model.base_separator import BaseSeparatorModel
from ..processor.audio_loader import AudioLoader
from ..processor.audio_saver import AudioSaver
from ..processor.preprocessor import AudioPreprocessor
from ..processor.postprocessor import AudioPostprocessor
from ..exceptions import (
    AudioIOError,
    AudioProcessingError,
    AudioValidationError,
    AudioInitializationError,
    AudioModelError
)
from ..logger import logger


class AudioSeparator(BaseSeparator):
    """
    High-level interface for audio source separation.
    
    This class provides a simple API for separating audio files.
    """
    
    # ════════════════════════════════════════════════════════════════════════════
    # INITIALIZATION
    # ════════════════════════════════════════════════════════════════════════════
    
    def __init__(
        self,
        model_type: str = DEFAULT_MODEL_TYPE,
        model_kwargs: Optional[Dict[str, Any]] = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE
    ):
        """
        Initialize audio separator.
        
        Args:
            model_type: Type of model ('demucs', 'spleeter', 'lalal', 'hybrid')
            model_kwargs: Additional arguments for model initialization
            sample_rate: Target sample rate
            
        Raises:
            AudioInitializationError: If initialization fails
            AudioValidationError: If parameters are invalid
        """
        super().__init__(sample_rate=sample_rate, name="AudioSeparator")
        
        if model_kwargs is None:
            model_kwargs = {}
        
        self.model_type = model_type.lower()
        self.model_kwargs = model_kwargs
        self.model: Optional[BaseSeparatorModel] = None
        self.loader: Optional[AudioLoader] = None
        self.saver: Optional[AudioSaver] = None
        self.preprocessor: Optional[AudioPreprocessor] = None
        self.postprocessor: Optional[AudioPostprocessor] = None
        
        self.initialize()
    
    def _validate_model_type(self, model_type: str) -> None:
        """
        Validate model type parameter.
        
        Args:
            model_type: Model type to validate
        
        Raises:
            AudioValidationError: If model_type is invalid
        """
        if model_type not in VALID_MODEL_TYPES:
            raise AudioValidationError(
                f"Invalid model_type '{model_type}'. Must be one of {VALID_MODEL_TYPES}",
                component=self.name,
                error_code="INVALID_MODEL_TYPE"
            )
    
    def _do_initialize(self, **kwargs):
        """Initialize separator components."""
        # Validate model type
        self._validate_model_type(self.model_type)
        
        logger.info(f"Initializing {self.name} with model_type={self.model_type}")
        
        # Build model
        self.model = build_audio_separator_model(
            model_type=self.model_type,
            sample_rate=self.sample_rate,
            **self.model_kwargs
        )
        self.register_resource("model", self.model)
        
        # Initialize processors
        self.loader = AudioLoader()
        self.saver = AudioSaver()
        self.preprocessor = AudioPreprocessor(sample_rate=self.sample_rate)
        self.postprocessor = AudioPostprocessor(sample_rate=self.sample_rate)
        
        self.register_resource("loader", self.loader)
        self.register_resource("saver", self.saver)
        self.register_resource("preprocessor", self.preprocessor)
        self.register_resource("postprocessor", self.postprocessor)
    
    def separate_file(
        self,
        audio_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        save_outputs: bool = True
    ) -> Dict[str, str]:
        """
        Separate an audio file into multiple sources.
        
        Args:
            audio_path: Path to input audio file
            output_dir: Directory to save separated sources
            save_outputs: Whether to save output files
            
        Returns:
            Dictionary mapping source names to output file paths
            
        Raises:
            AudioIOError: If file cannot be read or written
            AudioProcessingError: If separation fails
        """
        # Validate file
        audio_path = self.validate_audio_file(audio_path)
        
        try:
            logger.info(f"Separating audio file: {audio_path}")
            
            # Prepare output directory
            default_dir = audio_path.parent / DEFAULT_OUTPUT_DIR_NAME
            output_dir = self.prepare_output_dir(output_dir, default_dir)
            
            # Try model's separate method first
            result = self._try_model_separate_method(audio_path, output_dir)
            if result is not None:
                return result
            
            # Manual separation pipeline
            separated_audio = self._perform_separation_pipeline(audio_path)
            
            # Save outputs
            return self._save_separated_sources(
                separated_audio,
                audio_path,
                output_dir,
                save_outputs
            )
            
        except (AudioIOError, AudioValidationError):
            raise
        except Exception as e:
            raise AudioProcessingError(
                f"Error during audio separation: {str(e)}",
                component=self.name,
                error_code=ERROR_CODE_SEPARATION_FAILED,
                details={"audio_path": str(audio_path), "error": str(e)}
            ) from e
    
    def separate(
        self,
        audio_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> Dict[str, str]:
        """
        Separate audio file (implements base interface).
        
        Args:
            audio_path: Path to input audio file
            output_dir: Output directory
            **kwargs: Additional arguments (save_outputs, etc.)
            
        Returns:
            Dictionary mapping source names to output paths
        """
        save_outputs = kwargs.get("save_outputs", True)
        return self.separate_file(audio_path, output_dir, save_outputs=save_outputs)
    
    def _try_model_separate_method(
        self,
        audio_path: Path,
        output_dir: Path
    ) -> Optional[Dict[str, str]]:
        """
        Try to use model's separate method if available.
        
        Args:
            audio_path: Path to input audio file
            output_dir: Output directory
        
        Returns:
            Dictionary of separated sources or None if method not available/failed
        """
        if not hasattr(self.model, 'separate'):
            return None
        
        try:
            logger.debug("Using model's separate method")
            result = self.model.separate(str(audio_path), str(output_dir))
            logger.info(f"Successfully separated into {len(result)} sources")
            return result
        except Exception as e:
            logger.warning(
                f"Model's separate method failed: {str(e)}, "
                "falling back to manual separation"
            )
            return None
    
    def _perform_separation_pipeline(
        self,
        audio_path: Path
    ) -> Dict[str, np.ndarray]:
        """
        Perform the complete separation pipeline.
        
        Args:
            audio_path: Path to input audio file
        
        Returns:
            Dictionary of separated audio sources
        
        Raises:
            AudioProcessingError: If separation fails
        """
        try:
            logger.debug("Loading audio file")
            audio, original_sr = self.loader.load(str(audio_path), self.sample_rate)
            
            logger.debug("Preprocessing audio")
            audio_tensor = self.preprocessor.process(audio, original_sr)
            
            logger.debug("Running separation model")
            separated = self.model.forward(audio_tensor)
            
            logger.debug("Postprocessing separated sources")
            return self.postprocessor.process(separated)
        except Exception as e:
            raise AudioProcessingError(
                f"Error during separation pipeline: {str(e)}",
                component=self.name,
                error_code=ERROR_CODE_SEPARATION_PIPELINE_FAILED,
                details={"audio_path": str(audio_path), "error": str(e)}
            ) from e
    
    def _save_separated_sources(
        self,
        separated_audio: Dict[str, np.ndarray],
        audio_path: Path,
        output_dir: Path,
        save_outputs: bool
    ) -> Dict[str, str]:
        """
        Save separated audio sources to files.
        
        Args:
            separated_audio: Dictionary of separated audio sources
            audio_path: Original audio file path
            output_dir: Output directory
            save_outputs: Whether to save files or return audio data
        
        Returns:
            Dictionary mapping source names to output paths or audio data
        
        Raises:
            AudioIOError: If saving fails
        """
        result = {}
        audio_name = audio_path.stem
        
        logger.debug(f"Saving {len(separated_audio)} separated sources")
        for source_name, source_audio in separated_audio.items():
            if save_outputs:
                output_path = output_dir / f"{audio_name}_{source_name}.wav"
                try:
                    self.saver.save(
                        source_audio,
                        str(output_path),
                        sample_rate=self.sample_rate
                    )
                    result[source_name] = str(output_path)
                    logger.debug(f"Saved {source_name} to {output_path}")
                except Exception as e:
                    logger.error(f"Failed to save {source_name}: {str(e)}")
                    raise AudioIOError(
                        f"Failed to save source '{source_name}': {str(e)}",
                        component=self.name,
                        error_code=ERROR_CODE_SAVE_FAILED
                    ) from e
            else:
                result[source_name] = source_audio
        
        logger.info(f"Successfully separated {audio_path} into {len(result)} sources")
        return result
    
    def separate_audio(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        return_tensors: bool = False
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """
        Separate audio data directly (without file I/O).
        
        Args:
            audio: Audio data (numpy array or tensor)
            return_tensors: Return tensors instead of numpy arrays
            
        Returns:
            Dictionary of separated sources
            
        Raises:
            AudioValidationError: If audio format is invalid
            AudioProcessingError: If separation fails
        """
        # Use preprocessor validation
        self.preprocessor.validate_audio(audio, allow_empty=False)
        
        try:
            logger.debug("Separating audio data directly")
            
            # Convert to tensor using preprocessor
            if isinstance(audio, np.ndarray):
                audio_tensor = self.preprocessor.process(audio)
            else:  # torch.Tensor
                audio_tensor = self.preprocessor.process(audio.detach().cpu().numpy())
            
            # Separate
            logger.debug("Running separation model")
            separated = self.model.forward(audio_tensor)
            
            if return_tensors:
                logger.debug("Returning tensors")
                return separated
            
            # Postprocess to numpy
            logger.debug("Postprocessing to numpy arrays")
            return self.postprocessor.process(separated)
            
        except (AudioValidationError, AudioModelError):
            raise
        except Exception as e:
            raise AudioProcessingError(
                f"Error during audio separation: {str(e)}",
                component=self.name,
                error_code=ERROR_CODE_SEPARATION_FAILED,
                details={"error": str(e)}
            ) from e

