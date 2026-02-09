from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import time
import logging
from functools import wraps
from typing import Callable, Any, TypeVar

from typing import Any, List, Dict, Optional
import asyncio
F = TypeVar("F", bound=Callable[..., Any])
logger = logging.getLogger("centralized.decorator")

def centralized_logging_metrics_exception(func: F) -> F:
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            process_time = time.time() - start_time
            logger.info(f"{func.__name__} - SUCCESS - {process_time:.3f}s")
            return result
        except Exception as exc:
            process_time = time.time() - start_time
            logger.error(f"{func.__name__} - ERROR: {exc} - {process_time:.3f}s")
            return {
                "error": "Exception in CLI/core logic",
                "details": str(exc),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
    return wrapper  # type: ignore 