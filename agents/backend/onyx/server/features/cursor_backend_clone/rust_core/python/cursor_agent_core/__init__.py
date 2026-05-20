"""
Cursor Agent Core - High Performance Rust Module

This module provides high-performance implementations of common operations
using Rust for optimal speed and efficiency.

Modules:
    - compression: LZ4, Zstd, Snappy, Brotli, Gzip (5-10x faster)
    - crypto: AES-GCM, ChaCha20, Blake3, Argon2 (secure & fast)
    - batch: Parallel processing with Rayon (10-100x faster)
    - serialization: simd-json, MessagePack, Bincode (3x faster)
    - ids: UUID, ULID, Nanoid, Snowflake generation
    - text: Regex, Aho-Corasick, text analysis
    - utils: Timer, SystemInfo, formatters

Example:
    >>> from cursor_agent_core import CryptoService, CompressionService
    >>> crypto = CryptoService()
    >>> hash_result = crypto.hash_blake3(b"Hello World")
    >>> print(hash_result.hex)
"""

from typing import Dict, List, Optional, Tuple, Any

__version__ = "0.1.0"
__author__ = "Cursor Agent Team"

# Try to import the Rust module
try:
    from cursor_agent_core.cursor_agent_core import (
        # Core services
        CompressionService,
        CryptoService,
        BatchProcessor,
        SerializationService,
        IdGenerator,
        TextProcessor,
        # Compression types
        CompressionResult,
        CompressionStats,
        # Crypto types
        HashResult,
        EncryptionResult,
        KeyPair,
        # Batch types
        BatchResult,
        BatchStats,
        ProgressInfo,
        StreamProcessor,
        # Text types
        TextMatch,
        TextStats,
        # Utils
        Timer,
        SystemInfo,
        SizeFormatter,
        DurationFormatter,
        # Functions
        get_system_info,
        create_timer,
        # Module info
        __version__ as _rust_version,
    )

    RUST_AVAILABLE = True

    # Submodule aliases
    from cursor_agent_core.cursor_agent_core import compression
    from cursor_agent_core.cursor_agent_core import crypto
    from cursor_agent_core.cursor_agent_core import batch
    from cursor_agent_core.cursor_agent_core import serialization
    from cursor_agent_core.cursor_agent_core import ids
    from cursor_agent_core.cursor_agent_core import text
    from cursor_agent_core.cursor_agent_core import utils

except ImportError as e:
    RUST_AVAILABLE = False
    _import_error = str(e)

    # Fallback stubs
    class CompressionService:
        """Compression service (Rust module not available)"""
        def __init__(self, *args, **kwargs):
            raise ImportError(f"Rust module not available: {_import_error}. Build with: maturin develop --release")

    class CryptoService:
        """Crypto service (Rust module not available)"""
        def __init__(self, *args, **kwargs):
            raise ImportError(f"Rust module not available: {_import_error}. Build with: maturin develop --release")

    class BatchProcessor:
        """Batch processor (Rust module not available)"""
        def __init__(self, *args, **kwargs):
            raise ImportError(f"Rust module not available: {_import_error}. Build with: maturin develop --release")

    class SerializationService:
        """Serialization service (Rust module not available)"""
        def __init__(self, *args, **kwargs):
            raise ImportError(f"Rust module not available: {_import_error}. Build with: maturin develop --release")

    class IdGenerator:
        """ID generator (Rust module not available)"""
        def __init__(self, *args, **kwargs):
            raise ImportError(f"Rust module not available: {_import_error}. Build with: maturin develop --release")

    class TextProcessor:
        """Text processor (Rust module not available)"""
        def __init__(self, *args, **kwargs):
            raise ImportError(f"Rust module not available: {_import_error}. Build with: maturin develop --release")

    class Timer:
        """Timer (Rust module not available)"""
        def __init__(self, *args, **kwargs):
            raise ImportError(f"Rust module not available: {_import_error}. Build with: maturin develop --release")

    def get_system_info() -> Dict[str, str]:
        """Get system information (Rust module not available)"""
        raise ImportError(f"Rust module not available: {_import_error}. Build with: maturin develop --release")

    def create_timer() -> Timer:
        """Create a timer (Rust module not available)"""
        raise ImportError(f"Rust module not available: {_import_error}. Build with: maturin develop --release")


def is_rust_available() -> bool:
    """Check if the Rust module is available.
    
    Returns:
        bool: True if Rust module is loaded, False otherwise.
    """
    return RUST_AVAILABLE


__all__ = [
    # Core services
    "CompressionService",
    "CryptoService",
    "BatchProcessor",
    "SerializationService",
    "IdGenerator",
    "TextProcessor",
    # Utility classes
    "Timer",
    "SystemInfo",
    "SizeFormatter", 
    "DurationFormatter",
    # Functions
    "get_system_info",
    "create_timer",
    "is_rust_available",
    # Constants
    "RUST_AVAILABLE",
    "__version__",
]

# Export compression types if available
if RUST_AVAILABLE:
    __all__.extend([
        "CompressionResult",
        "CompressionStats",
        "HashResult",
        "EncryptionResult",
        "KeyPair",
        "BatchResult",
        "BatchStats",
        "ProgressInfo",
        "StreamProcessor",
        "TextMatch",
        "TextStats",
        "compression",
        "crypto",
        "batch",
        "serialization",
        "ids",
        "text",
        "utils",
    ])
