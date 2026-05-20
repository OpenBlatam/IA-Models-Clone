from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import asyncio
import os
from typing import Any, List, Dict, Optional
import logging
"""
Executor Pool Singleton
======================

Provides a shared ThreadPoolExecutor and asyncio.Semaphore so that all NLP
modules reuse the same worker threads, preventing thread explosion.
"""


_DEFAULT_MAX_WORKERS = max(4, (os.cpu_count() or 4))

@lru_cache(maxsize=1)
def get_pool(max_workers: int = _DEFAULT_MAX_WORKERS) -> ThreadPoolExecutor:
    """Return global ThreadPoolExecutor singleton."""
    return ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="nlp-worker")

@lru_cache(maxsize=1)
def get_semaphore(max_concurrency: int = 20) -> asyncio.Semaphore:
    """Return global asyncio.Semaphore singleton to limit concurrent tasks."""
    return asyncio.Semaphore(max_concurrency) 