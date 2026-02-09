"""
Script to analyze audio files.
"""

import argparse
import sys
import json
from pathlib import Path
from audio_separator.processor.audio_loader import AudioLoader
from audio_separator.utils.audio_analysis import (
    analyze_audio,
    detect_silence,
    calculate_loudness
)
from audio_separator.logger import logger


def main():
    parser = argparse.ArgumentParser(
        description="Analyze audio files"
    )
    parser.add_argument(
        "audio_file",
        type=str,
        help="Audio file to analyze"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--silence",
        action="store_true",
        help="Detect silence regions"
    )
    parser.add_argument(
        "--loudness",
        action="store_true",
        help="Calculate loudness metrics"
    )
    
    args = parser.parse_args()
    
    audio_file = Path(args.audio_file)
    
    if not audio_file.exists():
        print(f"Error: Audio file not found: {audio_file}", file=sys.stderr)
        return 1
    
    try:
        loader = AudioLoader()
        audio, sr = loader.load(str(audio_file))
        
        # Analyze audio
        analysis = analyze_audio(audio, sample_rate=sr)
        
        # Add silence detection if requested
        if args.silence:
            silence_regions = detect_silence(audio, sample_rate=sr)
            analysis["silence_regions"] = [
                {"start": start, "end": end}
                for start, end in silence_regions
            ]
        
        # Add loudness if requested
        if args.loudness:
            loudness = calculate_loudness(audio, sample_rate=sr)
            analysis.update(loudness)
        
        # Print results
        print(f"\nAudio Analysis: {audio_file.name}")
        print("=" * 60)
        for key, value in analysis.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            elif isinstance(value, (list, dict)):
                print(f"{key}: {json.dumps(value, indent=2)}")
            else:
                print(f"{key}: {value}")
        
        # Save to file if requested
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                json.dump(analysis, f, indent=2)
            print(f"\nResults saved to {output_path}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        logger.exception("Analysis failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

