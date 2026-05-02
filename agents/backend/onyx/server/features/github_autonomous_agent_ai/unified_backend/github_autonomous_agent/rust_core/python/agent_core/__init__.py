"""
Agent Core - High Performance Rust Core for GitHub Autonomous Agent

Este módulo proporciona bindings de Python para el núcleo de alto rendimiento
escrito en Rust, incluyendo:

- BatchProcessor: Procesamiento paralelo de tareas
- CacheService: Caché de alto rendimiento con TTL
- SearchEngine: Motor de búsqueda con filtros y regex
- TextProcessor: Procesamiento de texto e instrucciones
- TaskQueue: Cola de prioridad para tareas
- HashService: Hashing y utilidades criptográficas
- Timer, DateUtils, StringUtils, JsonUtils: Utilidades comunes

Example:
    >>> from agent_core import BatchProcessor, CacheService
    >>> batch = BatchProcessor(max_concurrent=4, batch_size=10)
    >>> cache = CacheService(max_size=1000, default_ttl=300)
"""

__version__ = "0.1.0"
__author__ = "GitHub Autonomous Agent Team"

try:
    from agent_core.agent_core import (
        # Batch processing
        BatchProcessor,
        BatchJob,
        BatchResult,
        BatchStats,
        # Cache
        CacheService,
        CacheEntry,
        CacheStats,
        # Crypto
        HashService,
        HashResult,
        # Queue
        TaskQueue,
        QueuedTask,
        QueueStats,
        # Search
        SearchEngine,
        SearchFilter,
        SearchResult,
        # Text
        TextProcessor,
        InstructionParams,
        ParsedInstruction,
        # Utils
        Timer,
        DateUtils,
        StringUtils,
        JsonUtils,
        # Functions
        get_system_info,
        create_timer,
    )

    RUST_AVAILABLE = True

except ImportError as e:
    import warnings

    warnings.warn(
        f"Rust core not available ({e}). Using Python fallback. "
        "For best performance, compile the Rust extension with `maturin develop`",
        RuntimeWarning,
    )

    RUST_AVAILABLE = False

    class _FallbackBase:
        """Base class for fallback implementations."""

        def __init__(self, *args, **kwargs):
            raise NotImplementedError(
                "Rust core not available. Install with `maturin develop`"
            )

    BatchProcessor = _FallbackBase
    BatchJob = _FallbackBase
    BatchResult = _FallbackBase
    BatchStats = _FallbackBase
    CacheService = _FallbackBase
    CacheEntry = _FallbackBase
    CacheStats = _FallbackBase
    HashService = _FallbackBase
    HashResult = _FallbackBase
    TaskQueue = _FallbackBase
    QueuedTask = _FallbackBase
    QueueStats = _FallbackBase
    SearchEngine = _FallbackBase
    SearchFilter = _FallbackBase
    SearchResult = _FallbackBase
    TextProcessor = _FallbackBase
    InstructionParams = _FallbackBase
    ParsedInstruction = _FallbackBase
    Timer = _FallbackBase
    DateUtils = _FallbackBase
    StringUtils = _FallbackBase
    JsonUtils = _FallbackBase

    def get_system_info():
        return {"rust_available": False}

    def create_timer():
        raise NotImplementedError("Rust core not available")


def is_rust_available() -> bool:
    """Check if the Rust module is available."""
    return RUST_AVAILABLE


def get_version() -> str:
    """Get module version."""
    return __version__


__all__ = [
    # Module info
    "__version__",
    "__author__",
    "is_rust_available",
    "get_version",
    "RUST_AVAILABLE",
    # Batch
    "BatchProcessor",
    "BatchJob",
    "BatchResult",
    "BatchStats",
    # Cache
    "CacheService",
    "CacheEntry",
    "CacheStats",
    # Crypto
    "HashService",
    "HashResult",
    # Queue
    "TaskQueue",
    "QueuedTask",
    "QueueStats",
    # Search
    "SearchEngine",
    "SearchFilter",
    "SearchResult",
    # Text
    "TextProcessor",
    "InstructionParams",
    "ParsedInstruction",
    # Utils
    "Timer",
    "DateUtils",
    "StringUtils",
    "JsonUtils",
    # Functions
    "get_system_info",
    "create_timer",
]
