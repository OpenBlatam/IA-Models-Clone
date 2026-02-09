"""
Sliding Window
==============

Sliding window rate limiting algorithm.
"""

import logging
import time
from typing import Dict, Any, List
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class WindowState:
    """Sliding window state."""
    requests: deque
    window_size: float  # seconds
    max_requests: int


class SlidingWindow:
    """Sliding window rate limiter."""
    
    def __init__(self, max_requests: int, window_size: float):
        self.max_requests = max_requests
        self.window_size = window_size
        self._windows: Dict[str, WindowState] = {}
    
    def _get_window(self, key: str) -> WindowState:
        """Get or create window for key."""
        if key not in self._windows:
            self._windows[key] = WindowState(
                requests=deque(),
                window_size=self.window_size,
                max_requests=self.max_requests
            )
        
        return self._windows[key]
    
    def _clean_old_requests(self, window: WindowState):
        """Remove requests outside window."""
        now = time.time()
        cutoff = now - window.window_size
        
        while window.requests and window.requests[0] < cutoff:
            window.requests.popleft()
    
    async def acquire(self, key: str) -> bool:
        """Acquire request slot."""
        window = self._get_window(key)
        self._clean_old_requests(window)
        
        if len(window.requests) < window.max_requests:
            window.requests.append(time.time())
            return True
        
        return False
    
    def get_remaining_requests(self, key: str) -> int:
        """Get remaining requests in window."""
        window = self._get_window(key)
        self._clean_old_requests(window)
        return max(0, window.max_requests - len(window.requests))
    
    def get_window_stats(self) -> Dict[str, Any]:
        """Get window statistics."""
        return {
            "total_windows": len(self._windows),
            "windows": {
                key: {
                    "current_requests": len(window.requests),
                    "max_requests": window.max_requests,
                    "window_size": window.window_size,
                    "remaining": self.get_remaining_requests(key)
                }
                for key, window in self._windows.items()
            }
        }















