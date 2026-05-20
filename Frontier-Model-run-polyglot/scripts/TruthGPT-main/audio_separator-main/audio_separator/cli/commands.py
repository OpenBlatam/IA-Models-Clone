"""
CLI command implementations.
Refactored to separate command logic from parser creation.
"""

import sys
from pathlib import Path
from typing import Dict, Any

from ..separator.audio_separator import AudioSeparator
from ..separator.batch_separator import BatchSeparator
from ..separator.constants import DEFAULT_MODEL_TYPE
from ..separator.base_separator import DEFAULT_SAMPLE_RATE
from ..utils.device_utils import get_device_info
from ..exceptions import AudioIOError, AudioProcessingError
from ..logger import logger
from .constants import (
    DEFAULT_MODEL,
    DEFAULT_SAMPLE_RATE as CLI_DEFAULT_SAMPLE_RATE,
    DEFAULT_EXTENSIONS,
    MSG_FILE_NOT_FOUND,
    MSG_SUCCESS_SEPARATED,
    MSG_NO_FILES_FOUND,
    MSG_PROCESSING_COMPLETE
)


def cmd_separate(args) -> int:
    """
    Execute separate command.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        input_path = Path(args.input)
        if not input_path.exists():
            print(MSG_FILE_NOT_FOUND.format(path=input_path), file=sys.stderr)
            return 1
        
        output_dir = Path(args.output) if args.output else None
        
        logger.info(f"Separating: {input_path}")
        logger.info(f"Model: {args.model}")
        
        separator = AudioSeparator(
            model_type=args.model or DEFAULT_MODEL_TYPE,
            sample_rate=args.sample_rate or DEFAULT_SAMPLE_RATE
        )
        
        results = separator.separate_file(
            str(input_path),
            output_dir=str(output_dir) if output_dir else None,
            save_outputs=not args.no_save
        )
        
        print(MSG_SUCCESS_SEPARATED.format(count=len(results)))
        for source_name, output_path in results.items():
            print(f"  - {source_name}: {output_path}")
        
        return 0
        
    except (AudioIOError, AudioProcessingError) as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        logger.exception("Separation failed")
        return 1
    except Exception as e:
        print(f"Unexpected error: {str(e)}", file=sys.stderr)
        logger.exception("Unexpected error during separation")
        return 1


def cmd_batch(args) -> int:
    """
    Execute batch command.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        input_path = Path(args.input)
        
        if not input_path.exists():
            print(f"Error: Path not found: {input_path}", file=sys.stderr)
            return 1
        
        output_dir = Path(args.output) if args.output else None
        
        logger.info(f"Batch processing: {input_path}")
        logger.info(f"Model: {args.model}")
        
        batch_separator = BatchSeparator(
            model_type=args.model or DEFAULT_MODEL_TYPE,
            sample_rate=args.sample_rate or DEFAULT_SAMPLE_RATE
        )
        
        if input_path.is_file():
            # Single file
            results = batch_separator.separate_files(
                [str(input_path)],
                output_dir=str(output_dir) if output_dir else None,
                show_progress=not args.no_progress
            )
        else:
            # Directory
            extensions = args.extensions or DEFAULT_EXTENSIONS
            results = batch_separator.separate_directory(
                str(input_path),
                output_dir=str(output_dir) if output_dir else None,
                extensions=extensions,
                recursive=args.recursive
            )
        
        if not results:
            print(MSG_NO_FILES_FOUND.format(dir=input_path), file=sys.stderr)
            return 1
        
        # Count successful separations
        successful = sum(1 for r in results.values() if isinstance(r, dict) and "error" not in r)
        total = len(results)
        
        print(MSG_PROCESSING_COMPLETE.format(success=successful, total=total))
        
        return 0
        
    except (AudioIOError, AudioProcessingError) as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        logger.exception("Batch processing failed")
        return 1
    except Exception as e:
        print(f"Unexpected error: {str(e)}", file=sys.stderr)
        logger.exception("Unexpected error during batch processing")
        return 1


def cmd_info(args) -> int:
    """
    Execute info command.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        if args.device:
            device_info = get_device_info()
            print("\nDevice Information:")
            print("=" * 50)
            for key, value in device_info.items():
                print(f"  {key}: {value}")
            print()
        else:
            print("\nAudio Separator - System Information")
            print("=" * 50)
            print(f"Version: {__import__('audio_separator').__version__}")
            print(f"Default Model: {DEFAULT_MODEL_TYPE}")
            print(f"Default Sample Rate: {DEFAULT_SAMPLE_RATE} Hz")
            print()
        
        return 0
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        logger.exception("Info command failed")
        return 1

