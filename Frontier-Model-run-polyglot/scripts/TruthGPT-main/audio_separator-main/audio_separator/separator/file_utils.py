"""
File utilities for audio separation.
====================================

Common utilities for finding and handling audio files:
- Finding audio files in directories
- File extension validation
- Path normalization

Single Responsibility: Provide reusable file handling utilities.
"""

from pathlib import Path
from typing import List, Set, Optional

from ..logger import logger

# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

# Supported audio file extensions
SUPPORTED_AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}

# Default extensions for batch processing
DEFAULT_AUDIO_EXTENSIONS = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']


# ════════════════════════════════════════════════════════════════════════════
# FILE FINDING UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def find_audio_files(
    input_path: Path,
    extensions: Optional[List[str]] = None,
    recursive: bool = False
) -> List[Path]:
    """
    Find all audio files in a directory or return single file.
    
    Args:
        input_path: Path to file or directory
        extensions: List of extensions to search for (None for default)
        recursive: Search recursively in subdirectories
        
    Returns:
        List of audio file paths
        
    Raises:
        ValueError: If input_path is neither a file nor a directory
    """
    input_path = Path(input_path)
    
    # Single file
    if input_path.is_file():
        if extensions is None or input_path.suffix.lower() in extensions:
            return [input_path]
        return []
    
    # Directory
    if not input_path.is_dir():
        raise ValueError(f"Path is neither a file nor a directory: {input_path}")
    
    if extensions is None:
        extensions = DEFAULT_AUDIO_EXTENSIONS
    
    # Normalize extensions (lowercase, with dot)
    normalized_extensions = []
    for ext in extensions:
        ext = ext.lower()
        if not ext.startswith('.'):
            ext = '.' + ext
        normalized_extensions.append(ext)
    
    # Find files
    audio_files = []
    pattern = "**/*" if recursive else "*"
    
    for ext in normalized_extensions:
        # Search both lowercase and uppercase
        audio_files.extend(input_path.glob(f"{pattern}{ext}"))
        audio_files.extend(input_path.glob(f"{pattern}{ext.upper()}"))
    
    # Remove duplicates and sort
    audio_files = sorted(set(audio_files))
    
    logger.debug(f"Found {len(audio_files)} audio files in {input_path}")
    return audio_files


def is_audio_file(file_path: Path) -> bool:
    """
    Check if file has a supported audio extension.
    
    Args:
        file_path: Path to file
        
    Returns:
        True if file has supported audio extension
    """
    return file_path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS


def normalize_audio_path(audio_path: str) -> Path:
    """
    Normalize audio file path.
    
    Args:
        audio_path: Path as string
        
    Returns:
        Normalized Path object
        
    Raises:
        ValueError: If path is invalid
    """
    path = Path(audio_path)
    
    if not path.exists():
        raise ValueError(f"Path does not exist: {path}")
    
    if path.is_file():
        if not is_audio_file(path):
            logger.warning(
                f"File extension '{path.suffix}' may not be supported. "
                f"Supported: {SUPPORTED_AUDIO_EXTENSIONS}"
            )
        return path
    
    if path.is_dir():
        raise ValueError(f"Path is a directory, expected a file: {path}")
    
    raise ValueError(f"Path is neither a file nor a directory: {path}")


# ════════════════════════════════════════════════════════════════════════════
# OUTPUT DIRECTORY UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def prepare_output_directory(
    output_dir: Optional[Path],
    default_dir: Path,
    component_name: str = "Separator"
) -> Path:
    """
    Prepare output directory, creating it if necessary.
    
    Args:
        output_dir: Desired output directory (None to use default)
        default_dir: Default directory if output_dir is None
        component_name: Name of component for error messages
        
    Returns:
        Prepared output directory Path
        
    Raises:
        OSError: If output directory cannot be created
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
        raise OSError(
            f"Cannot create output directory '{output_dir}': {str(e)}"
        ) from e


def get_output_path_for_file(
    audio_file: Path,
    output_dir: Path,
    source_name: str,
    format: str = "wav"
) -> Path:
    """
    Generate output path for a separated source.
    
    Args:
        audio_file: Original audio file path
        output_dir: Output directory
        source_name: Name of separated source
        format: Audio format (wav, mp3, etc.)
        
    Returns:
        Output file path
    """
    audio_name = audio_file.stem
    output_path = output_dir / f"{audio_name}_{source_name}.{format}"
    return output_path

