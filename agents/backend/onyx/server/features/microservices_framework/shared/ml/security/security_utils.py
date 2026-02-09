"""
Security Utilities
Security and validation utilities for ML operations.
"""

import re
import time
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class InputSanitizer:
    """Sanitize user inputs for security."""
    
    @staticmethod
    def sanitize_text(text: str, max_length: int = 10000) -> str:
        """Sanitize text input."""
        # Truncate if too long
        if len(text) > max_length:
            text = text[:max_length]
            logger.warning(f"Text truncated to {max_length} characters")
        
        # Remove null bytes
        text = text.replace("\x00", "")
        
        # Remove control characters (except newline and tab)
        text = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]", "", text)
        
        return text
    
    @staticmethod
    def sanitize_prompt(prompt: str, max_length: int = 2048) -> str:
        """Sanitize generation prompt."""
        sanitized = InputSanitizer.sanitize_text(prompt, max_length)
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r"<script.*?>",
            r"javascript:",
            r"on\w+\s*=",
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, sanitized, re.IGNORECASE):
                logger.warning(f"Suspicious pattern detected: {pattern}")
                sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)
        
        return sanitized
    
    @staticmethod
    def validate_model_name(model_name: str) -> bool:
        """Validate model name format."""
        # Allow alphanumeric, hyphens, underscores, and slashes
        pattern = r"^[a-zA-Z0-9_\-/]+$"
        return bool(re.match(pattern, model_name))


class RateLimiter:
    """Rate limiting for API endpoints."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = {}
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed."""
        now = time.time()
        
        if identifier not in self.requests:
            self.requests[identifier] = []
        
        # Remove old requests
        self.requests[identifier] = [
            req_time
            for req_time in self.requests[identifier]
            if now - req_time < self.window_seconds
        ]
        
        # Check limit
        if len(self.requests[identifier]) >= self.max_requests:
            return False
        
        # Record request
        self.requests[identifier].append(now)
        return True
    
    def get_remaining(self, identifier: str) -> int:
        """Get remaining requests."""
        now = time.time()
        
        if identifier not in self.requests:
            return self.max_requests
        
        # Remove old requests
        self.requests[identifier] = [
            req_time
            for req_time in self.requests[identifier]
            if now - req_time < self.window_seconds
        ]
        
        return max(0, self.max_requests - len(self.requests[identifier]))


class ResourceLimiter:
    """Limit resource usage."""
    
    def __init__(
        self,
        max_memory_mb: Optional[float] = None,
        max_gpu_memory_mb: Optional[float] = None,
    ):
        self.max_memory_mb = max_memory_mb
        self.max_gpu_memory_mb = max_gpu_memory_mb
    
    def check_memory(self) -> bool:
        """Check if memory usage is within limits."""
        if self.max_memory_mb is None:
            return True
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / (1024 * 1024)
        
        if memory_mb > self.max_memory_mb:
            logger.warning(f"Memory limit exceeded: {memory_mb:.2f}MB > {self.max_memory_mb}MB")
            return False
        
        return True
    
    def check_gpu_memory(self) -> bool:
        """Check if GPU memory usage is within limits."""
        if self.max_gpu_memory_mb is None:
            return True
        
        try:
            import torch
            if torch.cuda.is_available():
                memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                if memory_mb > self.max_gpu_memory_mb:
                    logger.warning(
                        f"GPU memory limit exceeded: {memory_mb:.2f}MB > {self.max_gpu_memory_mb}MB"
                    )
                    return False
        except Exception:
            pass
        
        return True

