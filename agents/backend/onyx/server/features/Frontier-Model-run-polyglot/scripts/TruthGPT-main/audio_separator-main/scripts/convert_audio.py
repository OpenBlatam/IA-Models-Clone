"""
Script to convert audio files between formats.
Refactored to use constants.
"""

import argparse
import sys
from pathlib import Path
from audio_separator.utils.format_converter import convert_format, batch_convert
from audio_separator.logger import logger
from audio_separator.processor.constants import DEFAULT_AUDIO_FORMAT


def main():
    parser = argparse.ArgumentParser(
        description="Convert audio files between formats"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input audio file or directory"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output file or directory"
    )
    parser.add_argument(
        "-f", "--format",
        type=str,
        default=DEFAULT_AUDIO_FORMAT,
        choices=["wav", "mp3", "flac", "ogg"],
        help="Output format"
    )
    parser.add_argument(
        "-sr", "--sample-rate",
        type=int,
        help="Target sample rate"
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Process directories recursively"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: Path not found: {input_path}", file=sys.stderr)
        return 1
    
    try:
        if input_path.is_file():
            # Single file conversion
            if args.output:
                output_path = Path(args.output)
            else:
                output_path = input_path.with_suffix(f".{args.format}")
            
            convert_format(
                str(input_path),
                str(output_path),
                output_format=args.format,
                sample_rate=args.sample_rate
            )
            print(f"Converted: {output_path}")
            
        else:
            # Batch conversion
            if not args.output:
                print("Error: Output directory required for batch conversion", file=sys.stderr)
                return 1
            
            results = batch_convert(
                str(input_path),
                args.output,
                output_format=args.format,
                sample_rate=args.sample_rate,
                recursive=args.recursive
            )
            
            successful = sum(1 for r in results.values() if isinstance(r, str))
            print(f"Converted {successful}/{len(results)} files")
        
        return 0
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        logger.exception("Conversion failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

