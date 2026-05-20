"""
Compare different separation models.
"""

from pathlib import Path
from audio_separator import AudioSeparator
from audio_separator.eval.metrics import evaluate_separation
import numpy as np


def compare_models(audio_path: str, output_dir: str = "comparison_output"):
    """
    Compare different separation models on the same audio file.
    
    Args:
        audio_path: Path to audio file
        output_dir: Output directory for results
    """
    print("Comparing Separation Models")
    print("=" * 60)
    
    models = ["demucs", "spleeter"]
    results = {}
    
    for model_type in models:
        print(f"\nProcessing with {model_type}...")
        try:
            separator = AudioSeparator(model_type=model_type)
            separated = separator.separate_file(
                audio_path,
                output_dir=f"{output_dir}/{model_type}"
            )
            results[model_type] = separated
            print(f"  Successfully separated into {len(separated)} sources")
        except Exception as e:
            print(f"  Error: {str(e)}")
            results[model_type] = None
    
    # Compare results
    print("\n" + "=" * 60)
    print("Comparison Summary")
    print("=" * 60)
    
    for model_type, separated in results.items():
        if separated:
            print(f"\n{model_type.upper()}:")
            for source_name in separated.keys():
                print(f"  - {source_name}")
        else:
            print(f"\n{model_type.upper()}: Failed")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python comparison.py <audio_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    compare_models(audio_file)

