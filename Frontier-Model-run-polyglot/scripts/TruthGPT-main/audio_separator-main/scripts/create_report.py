"""
Script to create visual reports of separation results.
Refactored to use constants.
"""

import argparse
import sys
from pathlib import Path
from audio_separator.utils.visualization import create_separation_report
from audio_separator.logger import logger
from audio_separator.separator.constants import DEFAULT_SAMPLE_RATE, DEFAULT_4_STEM_SOURCES
from audio_separator.utils.constants import SUPPORTED_AUDIO_FORMATS


def main():
    parser = argparse.ArgumentParser(
        description="Create visual report of audio separation"
    )
    parser.add_argument(
        "audio_file",
        type=str,
        help="Original audio file"
    )
    parser.add_argument(
        "separated_dir",
        type=str,
        help="Directory containing separated audio files"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output directory for report (default: separated_dir/report)"
    )
    parser.add_argument(
        "-sr", "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="Sample rate"
    )
    
    args = parser.parse_args()
    
    audio_file = Path(args.audio_file)
    separated_dir = Path(args.separated_dir)
    
    if not audio_file.exists():
        print(f"Error: Audio file not found: {audio_file}", file=sys.stderr)
        return 1
    
    if not separated_dir.exists():
        print(f"Error: Separated directory not found: {separated_dir}", file=sys.stderr)
        return 1
    
    # Find separated files
    audio_name = audio_file.stem
    separated_files = {}
    
    # Use constants for supported formats and source names
    source_names = list(DEFAULT_4_STEM_SOURCES) + ["accompaniment"]
    
    for ext in SUPPORTED_AUDIO_FORMATS:
        for source_name in source_names:
            file_path = separated_dir / f"{audio_name}_{source_name}{ext}"
            if file_path.exists():
                separated_files[source_name] = str(file_path)
                break
    
    if not separated_files:
        print(f"Error: No separated files found in {separated_dir}", file=sys.stderr)
        return 1
    
    output_dir = args.output or str(separated_dir / "report")
    
    try:
        create_separation_report(
            str(audio_file),
            separated_files,
            output_dir,
            sample_rate=args.sample_rate
        )
        print(f"Report created in {output_dir}")
        return 0
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        logger.exception("Report creation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

