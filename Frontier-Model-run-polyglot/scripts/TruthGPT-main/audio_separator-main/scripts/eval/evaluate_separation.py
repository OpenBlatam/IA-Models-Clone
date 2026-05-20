"""
Evaluation script for audio separation models.
Refactored to use constants and improve maintainability.
"""

import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from audio_separator import AudioSeparator
from audio_separator.utils.audio_utils import get_audio_info
from audio_separator.logger import logger
from audio_separator.exceptions import AudioProcessingError
from audio_separator.separator.constants import DEFAULT_MODEL_TYPE, SUPPORTED_AUDIO_EXTENSIONS
from audio_separator.utils.constants import SUPPORTED_AUDIO_FORMATS

# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

DEFAULT_OUTPUT_DIR = "evaluation_output"
SEPARATOR_LINE = "=" * 60
SEPARATOR_WIDTH = 60


def _process_audio_file(
    separator: AudioSeparator,
    audio_path: Path,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Process a single audio file for evaluation.
    
    Args:
        separator: AudioSeparator instance
        audio_path: Path to audio file
        output_dir: Output directory
        
    Returns:
        Dictionary with processing results
    """
    logger.info(f"Processing: {audio_path.name}")
    
    try:
        # Get audio info
        info = get_audio_info(str(audio_path))
        logger.debug(f"Audio info - Duration: {info['duration']:.2f}s, "
                    f"Sample Rate: {info['sample_rate']} Hz, "
                    f"Channels: {info['channels']}")
        
        # Separate
        separated = separator.separate_file(
            str(audio_path),
            output_dir=str(output_dir / audio_path.stem)
        )
        
        logger.info(f"Separated into {len(separated)} sources: {list(separated.keys())}")
        
        return {
            "file": str(audio_path),
            "sources": list(separated.keys()),
            "success": True,
            "info": info
        }
        
    except Exception as e:
        logger.error(f"Error processing {audio_path.name}: {str(e)}")
        return {
            "file": str(audio_path),
            "error": str(e),
            "success": False
        }


def _print_evaluation_summary(results: List[Dict[str, Any]]) -> None:
    """
    Print evaluation summary.
    
    Args:
        results: List of evaluation results
    """
    logger.info("=" * SEPARATOR_WIDTH)
    logger.info("Evaluation Summary")
    logger.info("=" * SEPARATOR_WIDTH)
    
    successful = sum(1 for r in results if r.get("success", False))
    failed = len(results) - successful
    
    logger.info(f"Successfully processed: {successful}/{len(results)}")
    logger.info(f"Failed: {failed}/{len(results)}")
    
    if failed > 0:
        logger.warning("Failed files:")
        for result in results:
            if not result.get("success", False):
                logger.warning(f"  - {result['file']}: {result.get('error', 'Unknown error')}")


def evaluate_model(
    model_type: str,
    test_files: List[str],
    output_dir: str,
    model_kwargs: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Evaluate a separation model on test files.
    
    Args:
        model_type: Type of model to evaluate
        test_files: List of test audio files
        output_dir: Output directory for separated files
        model_kwargs: Additional model arguments
        
    Returns:
        List of evaluation results
    """
    logger.info(f"Evaluating {model_type} model")
    logger.info("=" * SEPARATOR_WIDTH)
    
    separator = AudioSeparator(
        model_type=model_type,
        model_kwargs=model_kwargs or {}
    )
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for audio_file in test_files:
        audio_path = Path(audio_file)
        if not audio_path.exists():
            logger.warning(f"{audio_file} not found, skipping...")
            continue
        
        result = _process_audio_file(separator, audio_path, output_path)
        results.append(result)
    
    _print_evaluation_summary(results)
    return results


def _find_audio_files(input_path: Path, recursive: bool = False) -> List[str]:
    """
    Find audio files in the given path.
    
    Args:
        input_path: Path to file or directory
        recursive: Whether to search recursively
        
    Returns:
        List of audio file paths
        
    Raises:
        ValueError: If input path is invalid
    """
    if input_path.is_file():
        return [str(input_path)]
    
    if not input_path.is_dir():
        raise ValueError(f"{input_path} is not a valid file or directory")
    
    pattern = "**/*" if recursive else "*"
    test_files = set()
    
    # Use constants from the main package
    extensions = list(SUPPORTED_AUDIO_FORMATS)
    
    for ext in extensions:
        # Search for both lowercase and uppercase extensions
        test_files.update(input_path.glob(f"{pattern}{ext}"))
        test_files.update(input_path.glob(f"{pattern}{ext.upper()}"))
    
    return sorted([str(f) for f in test_files])


def _create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure argument parser.
    
    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(description="Evaluate audio separation models")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_TYPE,
        choices=["demucs", "spleeter", "hybrid", "lalal"],
        help="Model type to evaluate"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input audio file or directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process directories recursively"
    )
    return parser


def main():
    """Main entry point for evaluation script."""
    parser = _create_argument_parser()
    args = parser.parse_args()
    
    try:
        # Get input files
        input_path = Path(args.input)
        test_files = _find_audio_files(input_path, args.recursive)
        
        if not test_files:
            logger.warning(f"No audio files found in {args.input}")
            return
        
        logger.info(f"Found {len(test_files)} audio file(s)")
        
        # Evaluate
        evaluate_model(
            model_type=args.model,
            test_files=test_files,
            output_dir=args.output
        )
    except ValueError as e:
        logger.error(str(e))
        return
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

