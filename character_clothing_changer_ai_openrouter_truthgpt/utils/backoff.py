"""
Backoff Strategies
==================

Different backoff strategies for retry operations.
"""

import time
import random
from typing import Callable, Optional
from enum import Enum


class BackoffStrategy(str, Enum):
    """Backoff strategy types"""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIXED = "fixed"
    RANDOM = "random"
    EXPONENTIAL_JITTER = "exponential_jitter"


def calculate_backoff(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL,
    multiplier: float = 2.0
) -> float:
    """
    Calculate backoff delay based on strategy.
    
    Args:
        attempt: Current attempt number (0-indexed)
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        strategy: Backoff strategy to use
        multiplier: Multiplier for exponential strategies
        
    Returns:
        Delay in seconds
    """
    if strategy == BackoffStrategy.FIXED:
        delay = base_delay
    elif strategy == BackoffStrategy.LINEAR:
        delay = base_delay * (attempt + 1)
    elif strategy == BackoffStrategy.EXPONENTIAL:
        delay = base_delay * (multiplier ** attempt)
    elif strategy == BackoffStrategy.RANDOM:
        delay = base_delay * random.uniform(0.5, 1.5) * (attempt + 1)
    elif strategy == BackoffStrategy.EXPONENTIAL_JITTER:
        delay = base_delay * (multiplier ** attempt)
        # Add jitter (random component)
        jitter = delay * random.uniform(0.0, 0.3)
        delay = delay + jitter
    else:
        delay = base_delay
    
    # Clamp to max_delay
    return min(delay, max_delay)


def exponential_backoff(
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    multiplier: float = 2.0
) -> Callable[[int], float]:
    """
    Create exponential backoff function.
    
    Args:
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        multiplier: Multiplier for each attempt
        
    Returns:
        Backoff function
    """
    def backoff(attempt: int) -> float:
        return calculate_backoff(
            attempt,
            base_delay=base_delay,
            max_delay=max_delay,
            strategy=BackoffStrategy.EXPONENTIAL,
            multiplier=multiplier
        )
    return backoff


def linear_backoff(
    base_delay: float = 1.0,
    max_delay: float = 60.0
) -> Callable[[int], float]:
    """
    Create linear backoff function.
    
    Args:
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        
    Returns:
        Backoff function
    """
    def backoff(attempt: int) -> float:
        return calculate_backoff(
            attempt,
            base_delay=base_delay,
            max_delay=max_delay,
            strategy=BackoffStrategy.LINEAR
        )
    return backoff


def fixed_backoff(
    delay: float = 1.0
) -> Callable[[int], float]:
    """
    Create fixed backoff function.
    
    Args:
        delay: Fixed delay in seconds
        
    Returns:
        Backoff function
    """
    def backoff(attempt: int) -> float:
        return delay
    return backoff

