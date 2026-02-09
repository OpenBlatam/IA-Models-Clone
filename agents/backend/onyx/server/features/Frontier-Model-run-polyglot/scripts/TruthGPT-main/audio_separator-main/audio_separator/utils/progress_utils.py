"""
Progress tracking utilities.
"""

from typing import Optional, Callable
from tqdm import tqdm
from ..logger import logger


class ProgressTracker:
    """Track progress of audio separation operations."""
    
    def __init__(
        self,
        total: int,
        description: str = "Processing",
        show_progress: bool = True
    ):
        """
        Initialize progress tracker.
        
        Args:
            total: Total number of items to process
            description: Description for progress bar
            show_progress: Whether to show progress bar
        """
        self.total = total
        self.description = description
        self.show_progress = show_progress
        self.current = 0
        self.pbar: Optional[tqdm] = None
        
        if self.show_progress:
            self.pbar = tqdm(
                total=total,
                desc=description,
                unit="file",
                ncols=100
            )
    
    def update(self, n: int = 1, description: Optional[str] = None):
        """
        Update progress.
        
        Args:
            n: Number of items completed
            description: Optional description update
        """
        self.current += n
        if self.pbar:
            if description:
                self.pbar.set_description(description)
            self.pbar.update(n)
        else:
            logger.info(f"{self.description}: {self.current}/{self.total}")
    
    def close(self):
        """Close progress tracker."""
        if self.pbar:
            self.pbar.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def track_progress(
    items: list,
    description: str = "Processing",
    show_progress: bool = True,
    callback: Optional[Callable] = None
):
    """
    Generator that tracks progress while iterating over items.
    
    Args:
        items: Items to iterate over
        description: Description for progress bar
        show_progress: Whether to show progress bar
        callback: Optional callback function called for each item
        
    Yields:
        Items from the list
    """
    with ProgressTracker(len(items), description, show_progress) as tracker:
        for item in items:
            if callback:
                callback(item)
            yield item
            tracker.update(1)

