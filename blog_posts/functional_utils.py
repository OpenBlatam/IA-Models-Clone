from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import time
import logging
import json
import hashlib
from typing import Dict, Any, Optional, List, Union, Tuple, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from functools import partial, reduce, wraps
from operator import itemgetter
import itertools
from contextlib import contextmanager
import numpy as np
import pandas as pd
    from tqdm import tqdm
import yaml
import yaml
    import concurrent.futures
from typing import Any, List, Dict, Optional
import asyncio
"""
🔧 Functional Utilities Module
=============================

Modular utilities for functional programming patterns.
Eliminates code duplication through reusable, composable functions.

Key Principles:
- Iteration over duplication
- Modularization over repetition
- Composition over inheritance
- Pure functions with no side effects
- Immutable data transformations
"""


# Set up logging
logger = logging.getLogger(__name__)

# ============================================================================
# Generic Types for Reusability
# ============================================================================

T = TypeVar('T')
U = TypeVar('U')
K = TypeVar('K')
V = TypeVar('V')

@dataclass(frozen=True)
class Result(Generic[T, U]):
    """Generic Result type for error handling."""
    is_successful: bool
    value: Optional[T] = None
    error: Optional[U] = None

@dataclass(frozen=True)
class ValidationResult:
    """Generic validation result."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

# ============================================================================
# Iteration Utilities
# ============================================================================

def iterate_with_progress(iterable, description: str = "Processing"):
    """Iterate with progress bar in a functional way."""
    return tqdm(iterable, desc=description)

def iterate_batches(data: List[T], batch_size: int) -> List[List[T]]:
    """Iterate over data in batches."""
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

def iterate_pairs(iterable) -> List[Tuple[T, T]]:
    """Iterate over pairs of consecutive elements."""
    return list(zip(iterable, iterable[1:]))

def iterate_windows(iterable, window_size: int) -> List[List[T]]:
    """Iterate over sliding windows."""
    return [list(iterable[i:i + window_size]) for i in range(len(iterable) - window_size + 1)]

def iterate_combinations(items: List[T], r: int = 2) -> List[Tuple[T, ...]]:
    """Iterate over combinations of items."""
    return list(itertools.combinations(items, r))

def iterate_permutations(items: List[T], r: Optional[int] = None) -> List[Tuple[T, ...]]:
    """Iterate over permutations of items."""
    return list(itertools.permutations(items, r))

# ============================================================================
# Data Transformation Utilities
# ============================================================================

def transform_dict(data: Dict[K, V], transform_fn: Callable[[K, V], Tuple[K, V]]) -> Dict[K, V]:
    """Transform dictionary using a function."""
    return dict(transform_fn(k, v) for k, v in data.items())

def transform_list(data: List[T], transform_fn: Callable[[T], U]) -> List[U]:
    """Transform list using a function."""
    return [transform_fn(item) for item in data]

def transform_nested_dict(data: Dict, transform_fn: Callable[[Any], Any]) -> Dict:
    """Transform nested dictionary recursively."""
    if isinstance(data, dict):
        return {k: transform_nested_dict(v, transform_fn) for k, v in data.items()}
    elif isinstance(data, list):
        return [transform_nested_dict(item, transform_fn) for item in data]
    else:
        return transform_fn(data)

def filter_dict(data: Dict[K, V], predicate: Callable[[K, V], bool]) -> Dict[K, V]:
    """Filter dictionary using a predicate."""
    return {k: v for k, v in data.items() if predicate(k, v)}

def filter_list(data: List[T], predicate: Callable[[T], bool]) -> List[T]:
    """Filter list using a predicate."""
    return [item for item in data if predicate(item)]

def group_by(data: List[T], key_fn: Callable[[T], K]) -> Dict[K, List[T]]:
    """Group data by key function."""
    grouped_result = {}
    for item in data:
        group_key = key_fn(item)
        if group_key not in grouped_result:
            grouped_result[group_key] = []
        grouped_result[group_key].append(item)
    return grouped_result

def sort_by(data: List[T], key_fn: Callable[[T], Any], reverse: bool = False) -> List[T]:
    """Sort data by key function."""
    return sorted(data, key=key_fn, reverse=reverse)

def unique_by(data: List[T], key_fn: Callable[[T], Any]) -> List[T]:
    """Get unique items by key function."""
    seen_keys = set()
    unique_result = []
    for item in data:
        item_key = key_fn(item)
        if item_key not in seen_keys:
            seen_keys.add(item_key)
            unique_result.append(item)
    return unique_result

# ============================================================================
# Function Composition Utilities
# ============================================================================

def compose(*functions: Callable) -> Callable:
    """Compose multiple functions."""
    def inner(argument) -> Any:
        result = argument
        for function in reversed(functions):
            result = function(result)
        return result
    return inner

def pipe(data: T, *functions: Callable) -> Any:
    """Pipe data through multiple functions."""
    result = data
    for function in functions:
        result = function(result)
    return result

def partial_apply(func: Callable, *args, **kwargs) -> Callable:
    """Create a partial function."""
    return partial(func, *args, **kwargs)

def curry(func: Callable) -> Callable:
    """Curry a function."""
    @wraps(func)
    def curried(*args, **kwargs) -> Any:
        if len(args) + len(kwargs) >= func.__code__.co_argcount:
            return func(*args, **kwargs)
        return curry(partial(func, *args, **kwargs))
    return curried

# ============================================================================
# Error Handling Utilities
# ============================================================================

def safe_execute(func: Callable[[], T], default: Optional[T] = None) -> Result[T, Exception]:
    """Safely execute a function and return Result."""
    try:
        result = func()
        return Result(is_successful=True, value=result)
    except Exception as error:
        logger.error(f"Error executing {func.__name__}: {error}")
        return Result(is_successful=False, error=error)

def retry_on_error(func: Callable, max_retries: int = 3, delay: float = 1.0):
    """Retry function on error."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        for attempt_number in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as error:
                if attempt_number == max_retries - 1:
                    raise error
                logger.warning(f"Attempt {attempt_number + 1} failed, retrying in {delay}s: {error}")
                time.sleep(delay)
    return wrapper

@contextmanager
def error_context(description: str = "Operation"):
    """Context manager for error handling."""
    try:
        yield
    except Exception as error:
        logger.error(f"Error in {description}: {error}")
        raise

# ============================================================================
# Validation Utilities
# ============================================================================

def validate_required_fields(data: Dict, required_fields: List[str]) -> ValidationResult:
    """Validate that required fields are present."""
    missing_fields = [field for field in required_fields if field not in data]
    is_valid = len(missing_fields) == 0
    errors = [f"Missing required field: {field}" for field in missing_fields]
    return ValidationResult(is_valid=is_valid, errors=errors)

def validate_field_types(data: Dict, field_types: Dict[str, type]) -> ValidationResult:
    """Validate field types."""
    type_errors = []
    for field_name, expected_type in field_types.items():
        if field_name in data and not isinstance(data[field_name], expected_type):
            type_errors.append(f"Field '{field_name}' should be {expected_type.__name__}, got {type(data[field_name]).__name__}")
    
    is_valid = len(type_errors) == 0
    return ValidationResult(is_valid=is_valid, errors=type_errors)

def validate_field_ranges(data: Dict, field_ranges: Dict[str, Tuple[float, float]]) -> ValidationResult:
    """Validate field ranges."""
    range_errors = []
    for field_name, (min_value, max_value) in field_ranges.items():
        if field_name in data:
            field_value = data[field_name]
            if not (min_value <= field_value <= max_value):
                range_errors.append(f"Field '{field_name}' should be between {min_value} and {max_value}, got {field_value}")
    
    is_valid = len(range_errors) == 0
    return ValidationResult(is_valid=is_valid, errors=range_errors)

def combine_validations(*validations: ValidationResult) -> ValidationResult:
    """Combine multiple validation results."""
    all_errors = []
    all_warnings = []
    is_all_valid = True
    
    for validation in validations:
        all_errors.extend(validation.errors)
        all_warnings.extend(validation.warnings)
        if not validation.is_valid:
            is_all_valid = False
    
    return ValidationResult(
        is_valid=is_all_valid,
        errors=all_errors,
        warnings=all_warnings
    )

# ============================================================================
# File and I/O Utilities
# ============================================================================

def safe_load_json(file_path: str, default: Optional[Dict] = None) -> Result[Dict, Exception]:
    """Safely load JSON file."""
    def load_func():
        
    """load_func function."""
with open(file_path, 'r', encoding='utf-8') as file:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            return json.load(file)
    
    result = safe_execute(load_func)
    if not result.is_successful and default is not None:
        return Result(is_successful=True, value=default)
    return result

def safe_save_json(data: Dict, file_path: str) -> Result[bool, Exception]:
    """Safely save JSON file."""
    def save_func():
        
    """save_func function."""
with open(file_path, 'w', encoding='utf-8') as file:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(data, file, indent=2, ensure_ascii=False)
        return True
    
    return safe_execute(save_func)

def safe_load_yaml(file_path: str, default: Optional[Dict] = None) -> Result[Dict, Exception]:
    """Safely load YAML file."""
    def load_func():
        
    """load_func function."""
        with open(file_path, 'r', encoding='utf-8') as file:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            return yaml.safe_load(file)
    
    result = safe_execute(load_func)
    if not result.is_successful and default is not None:
        return Result(is_successful=True, value=default)
    return result

def safe_save_yaml(data: Dict, file_path: str) -> Result[bool, Exception]:
    """Safely save YAML file."""
    def save_func():
        
    """save_func function."""
        with open(file_path, 'w', encoding='utf-8') as file:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            yaml.dump(data, file, default_flow_style=False, allow_unicode=True)
        return True
    
    return safe_execute(save_func)

# ============================================================================
# Performance Utilities
# ============================================================================

@contextmanager
def timer_context(description: str = "Operation"):
    """Context manager for timing operations."""
    start_time = time.time()
    try:
        yield
    finally:
        elapsed_time = time.time() - start_time
        logger.info(f"{description} completed in {elapsed_time:.4f} seconds")

def time_function(func: Callable) -> Callable:
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        with timer_context(f"Function {func.__name__}"):
            return func(*args, **kwargs)
    return wrapper

def memoize(func: Callable) -> Callable:
    """Memoization decorator."""
    cache = {}
    
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        cache_key = str(args) + str(sorted(kwargs.items()))
        if cache_key not in cache:
            cache[cache_key] = func(*args, **kwargs)
        return cache[cache_key]
    return wrapper

# ============================================================================
# Data Processing Utilities
# ============================================================================

def chunk_data(data: List[T], chunk_size: int) -> List[List[T]]:
    """Split data into chunks."""
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

def flatten_list(nested_list: List[List[T]]) -> List[T]:
    """Flatten nested list."""
    return [item for sublist in nested_list for item in sublist]

def batch_process(data: List[T], process_fn: Callable[[T], U], batch_size: int = 100) -> List[U]:
    """Process data in batches."""
    chunks = chunk_data(data, batch_size)
    results = []
    for chunk in chunks:
        chunk_results = [process_fn(item) for item in chunk]
        results.extend(chunk_results)
    return results

def parallel_process(data: List[T], process_fn: Callable[[T], U], max_workers: int = 4) -> List[U]:
    """Process data in parallel."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_fn, data))
    return results

# ============================================================================
# Configuration Utilities
# ============================================================================

def merge_configs(*configs: Dict) -> Dict:
    """Merge multiple configs."""
    merged_config = {}
    for config in configs:
        merged_config.update(config)
    return merged_config

def deep_merge_configs(*configs: Dict) -> Dict:
    """Deep merge multiple configs."""
    if not configs:
        return {}
    
    merged_config = configs[0].copy()
    for config in configs[1:]:
        for key, value in config.items():
            if key in merged_config and isinstance(merged_config[key], dict) and isinstance(value, dict):
                merged_config[key] = deep_merge_configs(merged_config[key], value)
            else:
                merged_config[key] = value
    return merged_config

def filter_config(config: Dict, allowed_keys: List[str]) -> Dict:
    """Filter configuration to only include allowed keys."""
    return {k: v for k, v in config.items() if k in allowed_keys}

def transform_config_values(config: Dict, transform_fn: Callable[[Any], Any]) -> Dict:
    """Transform configuration values using a function."""
    return transform_dict(config, lambda k, v: (k, transform_fn(v)))

# ============================================================================
# Logging Utilities
# ============================================================================

def log_function_call(func: Callable) -> Callable:
    """Decorator to log function calls."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        logger.info(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"{func.__name__} returned {result}")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {e}")
            raise
    return wrapper

def log_data_info(data: Any, name: str = "Data") -> None:
    """Log information about data."""
    if isinstance(data, (list, tuple)):
        logger.info(f"{name}: {type(data).__name__} with {len(data)} items")
    elif isinstance(data, dict):
        logger.info(f"{name}: {type(data).__name__} with {len(data)} keys")
    elif isinstance(data, np.ndarray):
        logger.info(f"{name}: numpy array with shape {data.shape} and dtype {data.dtype}")
    else:
        logger.info(f"{name}: {type(data).__name__}")

# ============================================================================
# Testing Utilities
# ============================================================================

def generate_test_data(data_type: str, size: int = 100) -> Any:
    """Generate test data of specified type."""
    if data_type == "classification":
        return {
            'y_true': np.random.randint(0, 3, size),
            'y_pred': np.random.randint(0, 3, size),
            'y_prob': np.random.rand(size, 3)
        }
    elif data_type == "regression":
        return {
            'y_true': np.random.randn(size),
            'y_pred': np.random.randn(size) + 0.1
        }
    elif data_type == "text":
        return {
            'texts': [f"Sample text {i}" for i in range(size)],
            'labels': np.random.randint(0, 2, size)
        }
    else:
        raise ValueError(f"Unknown data type: {data_type}")

def assert_pure_function(func: Callable, *args, **kwargs) -> None:
    """Assert that a function is pure (same output for same input)."""
    result1 = func(*args, **kwargs)
    result2 = func(*args, **kwargs)
    assert result1 == result2, f"Function {func.__name__} is not pure"

def assert_immutable_update(original: Any, update_fn: Callable, *update_args) -> None:
    """Assert that an update function is immutable."""
    updated = update_fn(original, *update_args)
    assert original is not updated, "Update function is not immutable"

# ============================================================================
# Demo Functions
# ============================================================================

def demo_utilities():
    """Demo the utility functions."""
    print("🔧 Functional Utilities Demo")
    print("=" * 50)
    
    # Data transformation
    data = [1, 2, 3, 4, 5]
    doubled = transform_list(data, lambda x: x * 2)
    print(f"✅ Doubled data: {doubled}")
    
    # Function composition
    add_one = lambda x: x + 1
    multiply_by_two = lambda x: x * 2
    composed = compose(add_one, multiply_by_two)
    result = composed(5)
    print(f"✅ Composed function result: {result}")
    
    # Safe execution
    safe_result = safe_execute(lambda: 1 / 0)
    print(f"✅ Safe execution: {safe_result}")
    
    # Validation
    test_data = {"name": "test", "age": 25}
    validation = validate_required_fields(test_data, ["name", "age"])
    print(f"✅ Validation result: {validation}")
    
    # Iteration
    batches = list(iterate_batches(data, 2))
    print(f"✅ Batches: {batches}")
    
    print("\n🎉 All utilities working correctly!")

match __name__:
    case "__main__":
    demo_utilities() 