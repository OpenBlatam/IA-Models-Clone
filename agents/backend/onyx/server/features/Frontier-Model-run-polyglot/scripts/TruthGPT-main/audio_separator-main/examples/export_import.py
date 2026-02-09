"""
Examples of export and import features.
"""

from audio_separator import AudioSeparator
from audio_separator.utils.export_utils import (
    export_separation_metadata,
    export_separation_report,
    import_separation_metadata,
    create_separation_summary
)
from audio_separator.utils.audio_analysis import analyze_audio
from audio_separator.processor.audio_loader import AudioLoader


def example_export_metadata():
    """Example of exporting separation metadata."""
    print("Example 1: Export Metadata")
    print("-" * 50)
    
    separator = AudioSeparator(model_type="demucs")
    results = separator.separate_file("input.wav", output_dir="separated")
    
    # Export metadata
    export_separation_metadata(
        results,
        "separation_metadata.json",
        metadata={"model": "demucs", "sample_rate": 44100}
    )
    print("Exported metadata")
    print()


def example_export_report():
    """Example of exporting comprehensive report."""
    print("Example 2: Export Report")
    print("-" * 50)
    
    separator = AudioSeparator(model_type="demucs")
    results = separator.separate_file("input.wav", output_dir="separated")
    
    # Analyze original audio
    loader = AudioLoader()
    audio, sr = loader.load("input.wav")
    analysis = analyze_audio(audio, sample_rate=sr)
    
    # Export comprehensive report
    export_separation_report(
        results,
        analysis=analysis,
        output_path="separation_report.json"
    )
    print("Exported comprehensive report")
    print()


def example_import_metadata():
    """Example of importing metadata."""
    print("Example 3: Import Metadata")
    print("-" * 50)
    
    metadata = import_separation_metadata("separation_metadata.json")
    
    print("Imported Metadata:")
    print(f"  Timestamp: {metadata.get('timestamp')}")
    print(f"  Number of Sources: {metadata.get('num_sources')}")
    print(f"  Sources: {list(metadata.get('sources', {}).keys())}")
    print()


def example_create_summary():
    """Example of creating text summary."""
    print("Example 4: Create Summary")
    print("-" * 50)
    
    separator = AudioSeparator(model_type="demucs")
    results = separator.separate_file("input.wav", output_dir="separated")
    
    create_separation_summary(results, "separation_summary.txt")
    print("Created text summary")
    print()


if __name__ == "__main__":
    example_export_metadata()
    example_export_report()
    example_import_metadata()
    example_create_summary()

