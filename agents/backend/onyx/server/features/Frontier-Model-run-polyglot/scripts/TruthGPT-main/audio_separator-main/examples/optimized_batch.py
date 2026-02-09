"""
Examples of optimized batch processing.
"""

from pathlib import Path
from audio_separator import AudioSeparator
from audio_separator.utils.batch_optimizer import BatchOptimizer


def example_optimized_batch():
    """Example of optimized batch processing with caching."""
    print("Example: Optimized Batch Processing")
    print("-" * 50)
    
    # Initialize optimizer
    optimizer = BatchOptimizer(
        cache_enabled=True,
        max_workers=2
    )
    
    # Find audio files
    audio_files = list(Path("input").glob("*.mp3"))
    
    # Define processing function
    separator = AudioSeparator(model_type="demucs")
    
    def process_file(file_path: str):
        return separator.separate_file(file_path, output_dir="output")
    
    # Process with optimization
    results = optimizer.optimize_batch(
        audio_files,
        process_file,
        skip_existing=True
    )
    
    # Get statistics
    stats = optimizer.get_statistics()
    print(f"\nProcessing Statistics:")
    print(f"  Cache size: {stats['cache_size'] / 1024 / 1024:.2f} MB")
    print(f"  Performance metrics available")
    
    successful = sum(1 for r in results.values() if isinstance(r, dict) and "error" not in r)
    print(f"\nProcessed {successful}/{len(results)} files successfully")
    print()


if __name__ == "__main__":
    example_optimized_batch()

