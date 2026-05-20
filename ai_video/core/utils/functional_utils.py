from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import Dict, List, Optional, Callable, Any, Tuple, Union
from functools import partial, reduce, wraps
import torch
import numpy as np
import logging
from pathlib import Path
import json
import asyncio
from datetime import datetime
from typing import Any, List, Dict, Optional
"""
Functional Utilities for AI Video Processing
===========================================

Pure functions for common operations in AI video generation.
"""


logger = logging.getLogger(__name__)

# Type aliases
Tensor = torch.Tensor
Array = np.ndarray
PathLike = Union[str, Path]

# Pure functions for data processing
def normalize_tensor(tensor: Tensor, min_val: float = 0.0, max_val: float = 1.0) -> Tensor:
    """Normalize tensor to specified range."""
    t_min, t_max = tensor.min(), tensor.max()
    return (tensor - t_min) / (t_max - t_min) * (max_val - min_val) + min_val

def resize_tensor(tensor: Tensor, size: Tuple[int, int]) -> Tensor:
    """Resize tensor using interpolation."""
    return F.interpolate(
        tensor.unsqueeze(0), 
        size=size, 
        mode='bilinear', 
        align_corners=False
    ).squeeze(0)

def apply_transform(tensor: Tensor, transform_fn: Callable) -> Tensor:
    """Apply transformation function to tensor."""
    return transform_fn(tensor)

def batch_apply(func: Callable, items: List[Any]) -> List[Any]:
    """Apply function to list of items."""
    return list(map(func, items))

def filter_items(items: List[Any], predicate: Callable) -> List[Any]:
    """Filter items using predicate function."""
    return list(filter(predicate, items))

def reduce_items(items: List[Any], reducer: Callable, initial: Any = None) -> Any:
    """Reduce items using reducer function."""
    if initial is not None:
        return reduce(reducer, items, initial)
    return reduce(reducer, items)

# Function composition utilities
def compose(*functions: Callable) -> Callable:
    """Compose multiple functions from right to left."""
    return reduce(lambda f, g: lambda x: f(g(x)), functions)

def pipe(*functions: Callable) -> Callable:
    """Pipe functions from left to right."""
    return reduce(lambda f, g: lambda x: g(f(x)), functions)

def curry(func: Callable, *args, **kwargs) -> Callable:
    """Curry a function with partial arguments."""
    return partial(func, *args, **kwargs)

# Decorators for functional programming
def memoize(func: Callable) -> Callable:
    """Memoize function results."""
    cache = {}
    
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        key = str(args) + str(sorted(kwargs.items()))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    
    return wrapper

def retry(max_attempts: int = 3, delay: float = 1.0):
    """Retry decorator for functions that may fail."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

def with_logging(func: Callable) -> Callable:
    """Log function calls and results."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        logger.info(f"Calling {func.__name__}")
        result = func(*args, **kwargs)
        logger.info(f"Completed {func.__name__}")
        return result
    return wrapper

def with_timing(func: Callable) -> Callable:
    """Time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} took {end_time - start_time:.2f}s")
        return result
    return wrapper

# File system utilities
def ensure_dir(path: PathLike) -> Path:
    """Ensure directory exists, return Path object."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_json(data: Dict, path: PathLike) -> Path:
    """Save data as JSON file."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        json.dump(data, f, indent=2, default=str)
    return path

def load_json(path: PathLike) -> Dict:
    """Load data from JSON file."""
    with open(path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        return json.load(f)

def list_files(directory: PathLike, pattern: str = "*") -> List[Path]:
    """List files matching pattern in directory."""
    return list(Path(directory).glob(pattern))

def get_file_info(path: PathLike) -> Dict[str, Any]:
    """Get file information."""
    path = Path(path)
    return {
        "name": path.name,
        "size": path.stat().st_size,
        "modified": datetime.fromtimestamp(path.stat().st_mtime),
        "exists": path.exists()
    }

# Tensor operations
def stack_tensors(tensors: List[Tensor], dim: int = 0) -> Tensor:
    """Stack list of tensors along specified dimension."""
    return torch.stack(tensors, dim=dim)

def concat_tensors(tensors: List[Tensor], dim: int = 0) -> Tensor:
    """Concatenate tensors along specified dimension."""
    return torch.cat(tensors, dim=dim)

def split_tensor(tensor: Tensor, sizes: List[int], dim: int = 0) -> List[Tensor]:
    """Split tensor into chunks of specified sizes."""
    return torch.split(tensor, sizes, dim=dim)

def chunk_tensor(tensor: Tensor, chunks: int, dim: int = 0) -> List[Tensor]:
    """Split tensor into specified number of chunks."""
    return torch.chunk(tensor, chunks, dim=dim)

# Validation functions
def validate_tensor_shape(tensor: Tensor, expected_shape: Tuple[int, ...]) -> bool:
    """Validate tensor has expected shape."""
    return tensor.shape == expected_shape

def validate_tensor_dtype(tensor: Tensor, expected_dtype: torch.dtype) -> bool:
    """Validate tensor has expected dtype."""
    return tensor.dtype == expected_dtype

def validate_range(value: float, min_val: float, max_val: float) -> bool:
    """Validate value is within range."""
    return min_val <= value <= max_val

def validate_positive(value: Union[int, float]) -> bool:
    """Validate value is positive."""
    return value > 0

# Configuration utilities
def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple configuration dictionaries."""
    return reduce(lambda acc, config: {**acc, **config}, configs, {})

def filter_config(config: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    """Filter configuration to include only specified keys."""
    return {k: v for k, v in config.items() if k in keys}

def transform_config(config: Dict[str, Any], transforms: Dict[str, Callable]) -> Dict[str, Any]:
    """Apply transformations to configuration values."""
    return {
        k: transforms.get(k, lambda x: x)(v) 
        for k, v in config.items()
    }

def validate_config(config: Dict[str, Any], validators: Dict[str, Callable]) -> bool:
    """Validate configuration using validator functions."""
    for key, validator in validators.items():
        if key in config and not validator(config[key]):
            raise ValueError(f"Invalid value for {key}: {config[key]}")
    return True

# Async utilities
async def run_in_executor(func: Callable, *args, **kwargs) -> Any:
    """Run function in executor to avoid blocking."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, func, *args, **kwargs)

async def batch_process_async(
    items: List[Any],
    processor: Callable,
    max_concurrent: int = 4
) -> List[Any]:
    """Process items asynchronously with concurrency limit."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(item: Any) -> Any:
        async with semaphore:
            return await run_in_executor(processor, item)
    
    tasks = [process_with_semaphore(item) for item in items]
    return await asyncio.gather(*tasks)

def create_async_pipeline(*functions: Callable) -> Callable:
    """Create async pipeline from functions."""
    async def pipeline(input_data: Any) -> Any:
        result = input_data
        for func in functions:
            result = await run_in_executor(func, result)
        return result
    return pipeline

# Error handling utilities
def safe_execute(func: Callable, default: Any = None) -> Any:
    """Execute function safely, return default on error."""
    try:
        return func()
    except Exception as e:
        logger.error(f"Error in {func.__name__}: {e}")
        return default

def with_fallback(func: Callable, fallback: Callable) -> Callable:
    """Create function with fallback."""
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception:
            return fallback(*args, **kwargs)
    return wrapper

def validate_and_transform(
    data: Any,
    validator: Callable,
    transformer: Callable,
    error_msg: str = "Validation failed"
) -> bool:
    """Validate data and apply transformation."""
    if not validator(data):
        raise ValueError(error_msg)
    return transformer(data)

# Example usage
def example_functional_pipeline():
    """Example of functional pipeline composition."""
    # Define pure functions
    def add_one(x: int) -> int:
        return x + 1
    
    def multiply_by_two(x: int) -> int:
        return x * 2
    
    def square(x: int) -> int:
        return x ** 2
    
    # Compose pipeline
    pipeline = compose(square, multiply_by_two, add_one)
    
    # Apply to data
    result = pipeline(5)  # ((5 + 1) * 2) ^ 2 = 144
    print(f"Pipeline result: {result}")

def example_batch_processing():
    """Example of batch processing with functional utilities."""
    data = [1, 2, 3, 4, 5]
    
    # Apply transformations
    doubled = batch_apply(lambda x: x * 2, data)
    filtered = filter_items(doubled, lambda x: x > 5)
    summed = reduce_items(filtered, lambda x, y: x + y, 0)
    
    print(f"Original: {data}")
    print(f"Doubled: {doubled}")
    print(f"Filtered: {filtered}")
    print(f"Sum: {summed}")

if __name__ == "__main__":
    example_functional_pipeline()
    example_batch_processing() 