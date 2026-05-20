"""
Examples of parallel processing.
"""

from pathlib import Path
from audio_separator import AudioSeparator
from audio_separator.utils.parallel_processing import (
    process_parallel,
    batch_process_files
)


def example_parallel_separation():
    """Example of parallel file processing."""
    print("Example 1: Parallel File Processing")
    print("-" * 50)
    
    audio_files = [
        "song1.mp3",
        "song2.mp3",
        "song3.mp3"
    ]
    
    separator = AudioSeparator(model_type="demucs")
    
    def process_file(file_path: str):
        try:
            results = separator.separate_file(file_path, output_dir=f"output/{Path(file_path).stem}")
            return {"success": True, "sources": len(results)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # Process files in parallel
    results = process_parallel(
        audio_files,
        process_file,
        max_workers=2,
        use_processes=False  # Use threads for I/O-bound tasks
    )
    
    for file_path, result in zip(audio_files, results):
        if result.get("success"):
            print(f"{file_path}: {result['sources']} sources")
        else:
            print(f"{file_path}: Error - {result.get('error')}")
    print()


def example_batch_parallel():
    """Example of batch parallel processing."""
    print("Example 2: Batch Parallel Processing")
    print("-" * 50)
    
    input_dir = Path("input")
    audio_files = list(input_dir.glob("*.mp3"))
    
    separator = AudioSeparator(model_type="demucs")
    
    def process_audio_file(file_path: str):
        return separator.separate_file(file_path, output_dir="output")
    
    results = batch_process_files(
        audio_files,
        process_audio_file,
        max_workers=3
    )
    
    successful = sum(1 for r in results.values() if isinstance(r, dict) and "error" not in r)
    print(f"Processed {successful}/{len(results)} files successfully")
    print()


if __name__ == "__main__":
    example_parallel_separation()
    example_batch_parallel()

