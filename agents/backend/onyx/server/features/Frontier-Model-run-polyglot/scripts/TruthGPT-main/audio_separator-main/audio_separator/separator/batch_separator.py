"""
Batch audio separation utilities.
Refactored to use constants and file utilities.
"""

from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

from .audio_separator import AudioSeparator
from .base_separator import DEFAULT_SAMPLE_RATE
from .constants import DEFAULT_MODEL_TYPE
from .file_utils import find_audio_files, DEFAULT_AUDIO_EXTENSIONS
from ..exceptions import AudioIOError, AudioProcessingError
from ..logger import logger


# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

DEFAULT_BATCH_MODEL_TYPE = DEFAULT_MODEL_TYPE


class BatchSeparator:
    """
    Batch processing for multiple audio files.
    """
    
    def __init__(
        self,
        model_type: str = DEFAULT_BATCH_MODEL_TYPE,
        model_kwargs: Optional[Dict] = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE
    ):
        """
        Initialize batch separator.
        
        Args:
            model_type: Type of model to use
            model_kwargs: Model initialization arguments
            sample_rate: Target sample rate
        """
        self.separator = AudioSeparator(
            model_type=model_type,
            model_kwargs=model_kwargs,
            sample_rate=sample_rate
        )
    
    def separate_files(
        self,
        audio_paths: List[str],
        output_dir: Optional[str] = None,
        show_progress: bool = True
    ) -> Dict[str, Dict[str, str]]:
        """
        Separate multiple audio files.
        
        Args:
            audio_paths: List of audio file paths
            output_dir: Base output directory
            show_progress: Show progress bar
            
        Returns:
            Dictionary mapping input file paths to separated sources
        """
        results = {}
        errors = []
        
        iterator = tqdm(audio_paths, desc="Separating audio files") if show_progress else audio_paths
        
        for audio_path in iterator:
            try:
                # Create subdirectory for each file
                file_output_dir = None
                if output_dir:
                    file_output_dir = Path(output_dir) / Path(audio_path).stem
                
                separated = self.separator.separate_file(
                    audio_path,
                    output_dir=str(file_output_dir) if file_output_dir else None
                )
                results[audio_path] = separated
                
            except (AudioIOError, AudioProcessingError) as e:
                error_msg = f"Error processing {audio_path}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                errors.append({"file": audio_path, "error": str(e), "type": type(e).__name__})
                results[audio_path] = {"error": str(e), "error_type": type(e).__name__}
            except Exception as e:
                error_msg = f"Unexpected error processing {audio_path}: {str(e)}"
                logger.exception(error_msg)
                errors.append({"file": audio_path, "error": str(e), "type": "UnexpectedError"})
                results[audio_path] = {"error": str(e), "error_type": "UnexpectedError"}
        
        if errors:
            logger.warning(f"Completed with {len(errors)} errors out of {len(audio_paths)} files")
        
        return results
    
    def separate_directory(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        extensions: Optional[List[str]] = None,
        recursive: bool = False
    ) -> Dict[str, Dict[str, str]]:
        """
        Separate all audio files in a directory.
        
        Args:
            input_dir: Input directory
            output_dir: Output directory
            extensions: Audio file extensions to process (default: DEFAULT_AUDIO_EXTENSIONS)
            recursive: Search recursively in subdirectories
            
        Returns:
            Dictionary mapping input file paths to separated sources
        """
        if extensions is None:
            extensions = DEFAULT_AUDIO_EXTENSIONS
        
        # Find all audio files using utility function
        audio_files = find_audio_files(
            input_path=Path(input_dir),
            extensions=extensions,
            recursive=recursive
        )
        
        if not audio_files:
            logger.warning(f"No audio files found in {input_dir}")
            return {}
        
        # Convert Path objects to strings
        audio_paths = [str(f) for f in audio_files]
        
        logger.info(f"Found {len(audio_paths)} audio file(s) to process")
        return self.separate_files(audio_paths, output_dir)

