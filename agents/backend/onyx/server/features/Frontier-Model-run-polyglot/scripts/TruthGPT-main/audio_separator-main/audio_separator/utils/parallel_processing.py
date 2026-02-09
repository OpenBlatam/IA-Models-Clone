"""
Parallel processing utilities for audio separation.
"""

from typing import List, Callable, Any, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path

from ..logger import logger
from .progress_utils import ProgressTracker


def process_parallel(
    items: List[Any],
    func: Callable,
    max_workers: Optional[int] = None,
    use_processes: bool = False,
    show_progress: bool = True
) -> List[Any]:
    """
    Process items in parallel.
    
    Args:
        items: List of items to process
        func: Function to apply to each item
        max_workers: Maximum number of workers (None for auto)
        use_processes: Use processes instead of threads
        show_progress: Show progress bar
        
    Returns:
        List of results in same order as items
    """
    executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    
    results = [None] * len(items)
    
    with executor_class(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(func, item): i
            for i, item in enumerate(items)
        }
        
        # Track progress
        with ProgressTracker(len(items), "Processing", show_progress) as tracker:
            # Collect results as they complete
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                    tracker.update(1)
                except Exception as e:
                    logger.error(f"Error processing item {index}: {str(e)}")
                    results[index] = {"error": str(e)}
                    tracker.update(1)
    
    return results


def batch_process_files(
    files: List[Path],
    process_func: Callable,
    max_workers: Optional[int] = None,
    use_processes: bool = False
) -> dict:
    """
    Process multiple files in parallel.
    
    Args:
        files: List of file paths
        process_func: Function to process each file
        max_workers: Maximum number of workers
        use_processes: Use processes instead of threads
        
    Returns:
        Dictionary mapping file paths to results
    """
    def process_file(file_path: Path):
        try:
            return str(file_path), process_func(str(file_path))
        except Exception as e:
            return str(file_path), {"error": str(e)}
    
    results = process_parallel(
        files,
        process_file,
        max_workers=max_workers,
        use_processes=use_processes
    )
    
    return dict(results)


def chunk_audio_processing(
    audio: Any,
    chunk_size: int,
    process_func: Callable,
    overlap: float = 0.0,
    max_workers: Optional[int] = None
) -> Any:
    """
    Process audio in chunks in parallel.
    
    Args:
        audio: Audio data to process
        chunk_size: Size of each chunk
        process_func: Function to process each chunk
        overlap: Overlap between chunks (0.0-1.0)
        max_workers: Maximum number of workers
        
    Returns:
        Processed audio
    """
    import numpy as np
    
    if isinstance(audio, np.ndarray):
        total_length = len(audio)
        overlap_samples = int(chunk_size * overlap)
        step_size = chunk_size - overlap_samples
        
        chunks = []
        indices = []
        
        for start in range(0, total_length, step_size):
            end = min(start + chunk_size, total_length)
            chunk = audio[start:end]
            chunks.append(chunk)
            indices.append((start, end))
        
        # Process chunks in parallel
        processed_chunks = process_parallel(
            chunks,
            process_func,
            max_workers=max_workers,
            show_progress=False
        )
        
        # Reconstruct audio
        result = np.zeros(total_length, dtype=audio.dtype)
        for (start, end), processed_chunk in zip(indices, processed_chunks):
            if processed_chunk is not None:
                chunk_length = min(len(processed_chunk), end - start)
                result[start:start + chunk_length] = processed_chunk[:chunk_length]
        
        return result
    
    else:
        # For non-array audio, process sequentially
        return process_func(audio)

