"""
Development Helpers
==================

Utilities for development and debugging.
"""

import logging
import json
import time
from typing import Dict, Any, Optional
from pathlib import Path

# Import common decorators
from ..core.decorators import measure_time, log_calls

logger = logging.getLogger(__name__)

# Re-export decorators with aliases for backward compatibility
timing_decorator = measure_time()
log_function_call = log_calls(include_args=True, include_result=True)


def save_debug_info(data: Dict[str, Any], filename: str, output_dir: str = "debug"):
    """
    Save debug information to JSON file.
    
    Args:
        data: Data to save
        filename: Output filename
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    file_path = output_path / f"{filename}.json"
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    
    logger.debug(f"Saved debug info to {file_path}")


def load_debug_info(filename: str, output_dir: str = "debug") -> Optional[Dict[str, Any]]:
    """
    Load debug information from JSON file.
    
    Args:
        filename: Input filename
        output_dir: Input directory
        
    Returns:
        Loaded data or None
    """
    file_path = Path(output_dir) / f"{filename}.json"
    
    if not file_path.exists():
        logger.warning(f"Debug file not found: {file_path}")
        return None
    
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_size(size_bytes: int) -> str:
    """
    Format bytes to human-readable size.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def format_duration(seconds: float) -> str:
    """
    Format seconds to human-readable duration.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


class PerformanceProfiler:
    """
    Performance profiler for tracking execution times.
    """
    
    def __init__(self):
        self.timings: Dict[str, list] = {}
    
    def start(self, name: str):
        """Start timing an operation."""
        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(time.time())
    
    def end(self, name: str):
        """End timing an operation."""
        if name in self.timings and self.timings[name]:
            start = self.timings[name][-1]
            elapsed = time.time() - start
            self.timings[name][-1] = elapsed
            return elapsed
        return None
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """
        Get statistics for an operation.
        
        Args:
            name: Operation name
            
        Returns:
            Statistics dictionary
        """
        if name not in self.timings:
            return {}
        
        timings = [t for t in self.timings[name] if isinstance(t, float)]
        if not timings:
            return {}
        
        return {
            "count": len(timings),
            "total": sum(timings),
            "average": sum(timings) / len(timings),
            "min": min(timings),
            "max": max(timings)
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all operations."""
        return {name: self.get_stats(name) for name in self.timings.keys()}

