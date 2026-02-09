"""
Positional Encoding Submodule
Aggregates various positional encoding components.
"""

from .base import PositionalEncoding
from .sinusoidal import SinusoidalPositionalEncoding
from .learned import LearnedPositionalEncoding

# Alias for backward compatibility
PositionalEncoding = SinusoidalPositionalEncoding

__all__ = [
    "PositionalEncoding",
    "SinusoidalPositionalEncoding",
    "LearnedPositionalEncoding",
]



