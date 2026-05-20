"""
Audio format conversion utilities.
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np

from ..exceptions import AudioFormatError, AudioIOError
from ..logger import logger
from .audio_utils import get_audio_info


def convert_format(
    input_path: str,
    output_path: str,
    output_format: Optional[str] = None,
    sample_rate: Optional[int] = None,
    channels: Optional[int] = None
) -> str:
    """
    Convert audio file to different format.
    
    Args:
        input_path: Input audio file path
        output_path: Output audio file path
        output_format: Output format (auto-detected from output_path if None)
        sample_rate: Target sample rate (None to keep original)
        channels: Target channels (None to keep original)
        
    Returns:
        Path to converted file
        
    Raises:
        AudioIOError: If conversion fails
        AudioFormatError: If format is not supported
    """
    from ..processor.audio_loader import AudioLoader
    from ..processor.audio_saver import AudioSaver
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if not input_path.exists():
        raise AudioIOError(
            f"Input file not found: {input_path}",
            component="FormatConverter",
            error_code="INPUT_NOT_FOUND"
        )
    
    # Determine output format
    if output_format is None:
        output_format = output_path.suffix.lower().lstrip(".")
    
    if output_format not in ["wav", "mp3", "flac", "ogg"]:
        raise AudioFormatError(
            f"Unsupported output format: {output_format}",
            component="FormatConverter",
            error_code="UNSUPPORTED_FORMAT"
        )
    
    logger.info(f"Converting {input_path} to {output_format}")
    
    try:
        # Load audio
        loader = AudioLoader()
        audio, original_sr = loader.load(
            str(input_path),
            sample_rate=sample_rate,
            mono=(channels == 1) if channels else False
        )
        
        # Get target sample rate
        target_sr = sample_rate if sample_rate else original_sr
        
        # Save audio
        saver = AudioSaver()
        saver.save(
            audio,
            str(output_path),
            sample_rate=target_sr,
            format=output_format
        )
        
        logger.info(f"Successfully converted to {output_path}")
        return str(output_path)
        
    except Exception as e:
        raise AudioIOError(
            f"Conversion failed: {str(e)}",
            component="FormatConverter",
            error_code="CONVERSION_FAILED"
        ) from e


def batch_convert(
    input_dir: str,
    output_dir: str,
    output_format: str = "wav",
    sample_rate: Optional[int] = None,
    recursive: bool = False
) -> dict:
    """
    Batch convert audio files.
    
    Args:
        input_dir: Input directory
        output_dir: Output directory
        output_format: Target format
        sample_rate: Target sample rate
        recursive: Process subdirectories
        
    Returns:
        Dictionary mapping input files to output files
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find audio files
    extensions = [".mp3", ".wav", ".flac", ".m4a", ".ogg"]
    pattern = "**/*" if recursive else "*"
    
    audio_files = []
    for ext in extensions:
        audio_files.extend(input_dir.glob(f"{pattern}{ext}"))
        audio_files.extend(input_dir.glob(f"{pattern}{ext.upper()}"))
    
    audio_files = list(set(audio_files))
    
    results = {}
    for audio_file in audio_files:
        try:
            # Create output path
            relative_path = audio_file.relative_to(input_dir)
            output_path = output_dir / relative_path.with_suffix(f".{output_format}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert
            converted_path = convert_format(
                str(audio_file),
                str(output_path),
                output_format=output_format,
                sample_rate=sample_rate
            )
            
            results[str(audio_file)] = converted_path
            
        except Exception as e:
            logger.error(f"Failed to convert {audio_file}: {str(e)}")
            results[str(audio_file)] = {"error": str(e)}
    
    return results

