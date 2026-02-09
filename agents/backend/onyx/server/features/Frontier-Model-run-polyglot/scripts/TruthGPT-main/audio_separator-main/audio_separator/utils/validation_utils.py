"""
Validation utilities for audio separator.
Refactored to use constants.
"""

from pathlib import Path
from typing import List, Optional
import numpy as np

from .constants import (
    SUPPORTED_AUDIO_FORMATS,
    SUPPORTED_OUTPUT_FORMATS,
    COMMON_SAMPLE_RATES,
    MAX_NUM_SOURCES,
    ERROR_CODE_FILE_NOT_FOUND,
    ERROR_CODE_NOT_A_FILE,
    ERROR_CODE_UNSUPPORTED_FORMAT,
    ERROR_CODE_UNSUPPORTED_OUTPUT_FORMAT,
    ERROR_CODE_INVALID_SAMPLE_RATE,
    ERROR_CODE_INVALID_NUM_SOURCES,
    ERROR_CODE_INVALID_AUDIO_TYPE,
    ERROR_CODE_EMPTY_AUDIO,
    ERROR_CODE_INVALID_AUDIO_DIMENSIONS,
    ERROR_CODE_NAN_IN_AUDIO,
    ERROR_CODE_INF_IN_AUDIO,
    ERROR_CODE_NOT_A_DIRECTORY,
    ERROR_CODE_DIRECTORY_CREATION_FAILED
)
from ..exceptions import AudioValidationError, AudioFormatError
from ..logger import logger


def validate_audio_file(audio_path: Path) -> None:
    """
    Validate that audio file exists and has supported format.
    
    Args:
        audio_path: Path to audio file
        
    Raises:
        AudioValidationError: If file is invalid
        AudioFormatError: If format is not supported
    """
    if not audio_path.exists():
        raise AudioValidationError(
            f"Audio file not found: {audio_path}",
            component="ValidationUtils",
            error_code=ERROR_CODE_FILE_NOT_FOUND
        )
    
    if not audio_path.is_file():
        raise AudioValidationError(
            f"Path is not a file: {audio_path}",
            component="ValidationUtils",
            error_code=ERROR_CODE_NOT_A_FILE
        )
    
    suffix = audio_path.suffix.lower()
    if suffix not in SUPPORTED_AUDIO_FORMATS:
        raise AudioFormatError(
            f"Unsupported audio format: {suffix}. "
            f"Supported formats: {SUPPORTED_AUDIO_FORMATS}",
            component="ValidationUtils",
            error_code=ERROR_CODE_UNSUPPORTED_FORMAT
        )


def validate_output_format(format_str: str) -> None:
    """
    Validate output format.
    
    Args:
        format_str: Output format string
        
    Raises:
        AudioFormatError: If format is not supported
    """
    format_str = format_str.lower().lstrip(".")
    supported_formats_clean = [f.lstrip(".") for f in SUPPORTED_OUTPUT_FORMATS]
    if format_str not in supported_formats_clean:
        raise AudioFormatError(
            f"Unsupported output format: {format_str}. "
            f"Supported formats: {SUPPORTED_OUTPUT_FORMATS}",
            component="ValidationUtils",
            error_code=ERROR_CODE_UNSUPPORTED_OUTPUT_FORMAT
        )


def validate_sample_rate(sample_rate: int) -> None:
    """
    Validate sample rate.
    
    Args:
        sample_rate: Sample rate to validate
        
    Raises:
        AudioValidationError: If sample rate is invalid
    """
    if sample_rate <= 0:
        raise AudioValidationError(
            f"Sample rate must be > 0, got {sample_rate}",
            component="ValidationUtils",
            error_code=ERROR_CODE_INVALID_SAMPLE_RATE
        )
    
    # Warn about unusual sample rates
    if sample_rate not in COMMON_SAMPLE_RATES:
        logger.warning(
            f"Unusual sample rate: {sample_rate}. "
            f"Common rates: {COMMON_SAMPLE_RATES}"
        )


def validate_num_sources(num_sources: int) -> None:
    """
    Validate number of sources.
    
    Args:
        num_sources: Number of sources to validate
        
    Raises:
        AudioValidationError: If number of sources is invalid
    """
    if num_sources < 1:
        raise AudioValidationError(
            f"Number of sources must be >= 1, got {num_sources}",
            component="ValidationUtils",
            error_code=ERROR_CODE_INVALID_NUM_SOURCES
        )
    
    if num_sources > MAX_NUM_SOURCES:
        logger.warning(
            f"Large number of sources: {num_sources}. "
            f"This may impact performance. (Max recommended: {MAX_NUM_SOURCES})"
        )


def validate_audio_array(audio: np.ndarray) -> None:
    """
    Validate audio array.
    
    Args:
        audio: Audio array to validate
        
    Raises:
        AudioValidationError: If audio array is invalid
    """
    if not isinstance(audio, np.ndarray):
        raise AudioValidationError(
            f"Audio must be numpy array, got {type(audio)}",
            component="ValidationUtils",
            error_code=ERROR_CODE_INVALID_AUDIO_TYPE
        )
    
    if audio.size == 0:
        raise AudioValidationError(
            "Audio array is empty",
            component="ValidationUtils",
            error_code=ERROR_CODE_EMPTY_AUDIO
        )
    
    if audio.ndim > 2:
        raise AudioValidationError(
            f"Audio must have 1 or 2 dimensions, got {audio.ndim}",
            component="ValidationUtils",
            error_code=ERROR_CODE_INVALID_AUDIO_DIMENSIONS
        )
    
    # Check for NaN or Inf
    if np.any(np.isnan(audio)):
        raise AudioValidationError(
            "Audio array contains NaN values",
            component="ValidationUtils",
            error_code=ERROR_CODE_NAN_IN_AUDIO
        )
    
    if np.any(np.isinf(audio)):
        raise AudioValidationError(
            "Audio array contains Inf values",
            component="ValidationUtils",
            error_code=ERROR_CODE_INF_IN_AUDIO
        )


def validate_output_dir(output_dir: Path, create: bool = True) -> None:
    """
    Validate and optionally create output directory.
    
    Args:
        output_dir: Output directory path
        create: Whether to create directory if it doesn't exist
        
    Raises:
        AudioValidationError: If directory is invalid
    """
    if output_dir.exists() and not output_dir.is_dir():
        raise AudioValidationError(
            f"Output path exists but is not a directory: {output_dir}",
            component="ValidationUtils",
            error_code=ERROR_CODE_NOT_A_DIRECTORY
        )
    
    if create and not output_dir.exists():
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created output directory: {output_dir}")
        except Exception as e:
            raise AudioValidationError(
                f"Failed to create output directory: {str(e)}",
                component="ValidationUtils",
                error_code=ERROR_CODE_DIRECTORY_CREATION_FAILED
            ) from e

