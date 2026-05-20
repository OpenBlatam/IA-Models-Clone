"""
Main CLI entry point.
Refactored to separate parser creation from command execution.
"""

import argparse
import sys

from .commands import cmd_separate, cmd_batch, cmd_info
from .constants import DEFAULT_MODEL, DEFAULT_SAMPLE_RATE, DEFAULT_EXTENSIONS, VALID_MODELS


def create_parser() -> argparse.ArgumentParser:
    """
    Create command-line argument parser.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Audio Separator - Advanced Audio Source Separation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Separate a single file
  audio-separator separate song.mp3 -o output/

  # Separate with specific model
  audio-separator separate song.mp3 -m spleeter -o output/

  # Batch process directory
  audio-separator batch input_dir/ -o output_dir/

  # Check device info
  audio-separator info
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Separate command
    _add_separate_parser(subparsers)
    
    # Batch command
    _add_batch_parser(subparsers)
    
    # Info command
    _add_info_parser(subparsers)
    
    return parser


def _add_separate_parser(subparsers) -> None:
    """Add separate command parser."""
    separate_parser = subparsers.add_parser("separate", help="Separate a single audio file")
    separate_parser.add_argument("input", type=str, help="Input audio file")
    separate_parser.add_argument("-o", "--output", type=str, help="Output directory")
    separate_parser.add_argument(
        "-m", "--model",
        type=str,
        default=DEFAULT_MODEL,
        choices=VALID_MODELS,
        help="Model type to use"
    )
    separate_parser.add_argument(
        "-sr", "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="Target sample rate"
    )
    separate_parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save output files"
    )


def _add_batch_parser(subparsers) -> None:
    """Add batch command parser."""
    batch_parser = subparsers.add_parser("batch", help="Batch process multiple files")
    batch_parser.add_argument("input", type=str, help="Input directory or file list")
    batch_parser.add_argument("-o", "--output", type=str, help="Output directory")
    batch_parser.add_argument(
        "-m", "--model",
        type=str,
        default=DEFAULT_MODEL,
        choices=VALID_MODELS,
        help="Model type to use"
    )
    batch_parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Process subdirectories recursively"
    )
    batch_parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=DEFAULT_EXTENSIONS,
        help="File extensions to process"
    )
    batch_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Don't show progress bar"
    )


def _add_info_parser(subparsers) -> None:
    """Add info command parser."""
    info_parser = subparsers.add_parser("info", help="Show system information")
    info_parser.add_argument(
        "--device",
        action="store_true",
        help="Show device information"
    )


def main() -> int:
    """
    Main CLI entry point.
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Route to appropriate command handler
    command_handlers = {
        "separate": cmd_separate,
        "batch": cmd_batch,
        "info": cmd_info,
    }
    
    handler = command_handlers.get(args.command)
    if handler:
        return handler(args)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

