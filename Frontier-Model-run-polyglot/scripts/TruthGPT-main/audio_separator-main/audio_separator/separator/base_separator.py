"""
Base separator class with common functionality.

Refactored to:
- Extract constants and error codes
- Improve type hints and documentation
- Enhance validation and error handling
- Better separation of concerns
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Union, Any

from ..core.base_component import BaseComponent
from ..exceptions import AudioIOError
from ..logger import logger
from .constants import (
    DEFAULT_SAMPLE_RATE,
    ERROR_CODE_FILE_NOT_FOUND,
    ERROR_CODE_NOT_A_FILE,
    ERROR_CODE_INVALID_OUTPUT_DIR,
    SUPPORTED_AUDIO_EXTENSIONS
)

# ════════════════════════════════════════════════════════════════════════════
# BASE SEPARATOR
# ════════════════════════════════════════════════════════════════════════════

class BaseSeparator(BaseComponent, ABC):
    """
    Base class for audio separators.
    
    Provides common functionality:
    - File validation and path handling
    - Output directory management
    - Error handling with specific error codes
    - Resource management (inherited from BaseComponent)
    
    Subclasses must implement:
    - _do_initialize(): Separator-specific initialization
    - separate(): Actual separation logic
    """
    
    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        name: Optional[str] = None
    ):
        """
        Initialize base separator.
        
        Args:
            sample_rate: Target sample rate in Hz
            name: Separator name (defaults to class name)
        
        Raises:
            AudioValidationError: If sample_rate is invalid
        """
        super().__init__(name=name)
        self._validate_sample_rate(sample_rate)
        self.sample_rate = sample_rate
    
    def _validate_sample_rate(self, sample_rate: int) -> None:
        """
        Validate sample rate parameter.
        
        Args:
            sample_rate: Sample rate to validate
        
        Raises:
            AudioValidationError: If sample_rate is invalid
        """
        if not isinstance(sample_rate, int):
            raise AudioIOError(
                f"sample_rate must be an integer, got {type(sample_rate).__name__}",
                component=self.name or self.__class__.__name__,
                error_code="INVALID_SAMPLE_RATE_TYPE"
            )
        if sample_rate <= 0:
            raise AudioIOError(
                f"sample_rate must be positive, got {sample_rate}",
                component=self.name or self.__class__.__name__,
                error_code="INVALID_SAMPLE_RATE"
            )
    
    def validate_audio_file(self, audio_path: Union[str, Path]) -> Path:
        """
        Validate audio file path and return Path object.
        
        Args:
            audio_path: Path to audio file (string or Path)
            
        Returns:
            Validated Path object
            
        Raises:
            AudioIOError: If file is invalid or not found
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise AudioIOError(
                f"Audio file not found: {audio_path}",
                component=self.name,
                error_code=ERROR_CODE_FILE_NOT_FOUND
            )
        
        if not audio_path.is_file():
            raise AudioIOError(
                f"Path is not a file: {audio_path}",
                component=self.name,
                error_code=ERROR_CODE_NOT_A_FILE
            )
        
        # Optional: Check file extension
        if audio_path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            logger.warning(
                f"File extension '{audio_path.suffix}' may not be supported. "
                f"Supported extensions: {SUPPORTED_AUDIO_EXTENSIONS}"
            )
        
        return audio_path
    
    def prepare_output_dir(
        self,
        output_dir: Optional[Union[str, Path]],
        default_dir: Path
    ) -> Path:
        """
        Prepare output directory, creating it if necessary.
        
        Args:
            output_dir: Desired output directory (None to use default)
            default_dir: Default directory if output_dir is None
            
        Returns:
            Prepared output directory Path
            
        Raises:
            AudioIOError: If output directory cannot be created
        """
        if output_dir is None:
            output_dir = default_dir
        else:
            output_dir = Path(output_dir)
        
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Prepared output directory: {output_dir}")
            return output_dir
        except (OSError, PermissionError) as e:
            raise AudioIOError(
                f"Cannot create output directory '{output_dir}': {str(e)}",
                component=self.name,
                error_code=ERROR_CODE_INVALID_OUTPUT_DIR
            ) from e
    
    @abstractmethod
    def _do_initialize(self, **kwargs) -> None:
        """
        Perform separator-specific initialization.
        
        This method should be implemented by subclasses to perform
        their specific initialization logic (e.g., loading models).
        
        Args:
            **kwargs: Initialization parameters
        """
        pass
    
    @abstractmethod
    def separate(
        self,
        audio_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> Dict[str, str]:
        """
        Separate audio file into multiple sources.
        
        This is the main entry point for audio separation.
        
        Args:
            audio_path: Path to input audio file
            output_dir: Output directory (None for default)
            **kwargs: Additional arguments for separation
            
        Returns:
            Dictionary mapping source names to output file paths
            
        Raises:
            AudioIOError: If input file is invalid
            AudioProcessingError: If separation fails
        """
        pass
