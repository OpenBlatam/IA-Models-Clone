"""
Test Helpers
Additional helper functions and decorators for tests
"""

import functools
import time
import logging
from typing import Callable, Any, Optional
import traceback

logger = logging.getLogger(__name__)

def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry a test on failure"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(f"Test {func.__name__} failed (attempt {attempt + 1}/{max_retries}), retrying...")
                        time.sleep(delay)
                    else:
                        logger.error(f"Test {func.__name__} failed after {max_retries} attempts")
            raise last_exception
        return wrapper
    return decorator

def skip_if_no_cuda(func: Callable) -> Callable:
    """Decorator to skip test if CUDA is not available"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import torch
        if not torch.cuda.is_available():
            import unittest
            raise unittest.SkipTest("CUDA not available")
        return func(*args, **kwargs)
    return wrapper

def skip_if_slow(func: Callable) -> Callable:
    """Decorator to skip slow tests unless explicitly enabled"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import os
        if os.environ.get('RUN_SLOW_TESTS', '0') != '1':
            import unittest
            raise unittest.SkipTest("Slow test skipped (set RUN_SLOW_TESTS=1 to enable)")
        return func(*args, **kwargs)
    return wrapper

def performance_test(max_duration: float = 10.0):
    """Decorator to mark and validate performance tests"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                if duration > max_duration:
                    logger.warning(f"Performance test {func.__name__} took {duration:.2f}s (max: {max_duration}s)")
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Performance test {func.__name__} failed after {duration:.2f}s: {e}")
                raise
        return wrapper
    return decorator

def validate_output_shape(expected_shape: tuple):
    """Decorator to validate function output shape"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if hasattr(result, 'shape'):
                if result.shape != expected_shape:
                    raise AssertionError(f"Expected shape {expected_shape}, got {result.shape}")
            return result
        return wrapper
    return decorator

def log_test_execution(func: Callable) -> Callable:
    """Decorator to log test execution"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Starting test: {func.__name__}")
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"Completed test: {func.__name__} ({duration:.3f}s)")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Failed test: {func.__name__} ({duration:.3f}s): {e}")
            raise
    return wrapper

def memory_profiler(func: Callable) -> Callable:
    """Decorator to profile memory usage"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            result = func(*args, **kwargs)
            
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = peak_memory - initial_memory
            
            logger.info(f"Memory usage for {func.__name__}: {memory_used:.2f}MB")
            return result
        except ImportError:
            logger.warning("psutil not available, skipping memory profiling")
            return func(*args, **kwargs)
    return wrapper

class TestContext:
    """Context manager for test setup and teardown"""
    def __init__(self, setup_func: Optional[Callable] = None, teardown_func: Optional[Callable] = None):
        self.setup_func = setup_func
        self.teardown_func = teardown_func
    
    def __enter__(self):
        if self.setup_func:
            self.setup_func()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.teardown_func:
            self.teardown_func()
        return False

def assert_tensor_equal(tensor1, tensor2, rtol=1e-5, atol=1e-8):
    """Assert two tensors are equal within tolerance"""
    import torch
    if not isinstance(tensor1, torch.Tensor):
        tensor1 = torch.tensor(tensor1)
    if not isinstance(tensor2, torch.Tensor):
        tensor2 = torch.tensor(tensor2)
    
    if tensor1.shape != tensor2.shape:
        raise AssertionError(f"Shapes don't match: {tensor1.shape} vs {tensor2.shape}")
    
    if not torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol):
        max_diff = (tensor1 - tensor2).abs().max().item()
        raise AssertionError(f"Tensors not equal (max diff: {max_diff}, rtol={rtol}, atol={atol})")

def assert_model_outputs_valid(model, input_shape, output_shape=None):
    """Assert model produces valid outputs"""
    import torch
    test_input = torch.randn(*input_shape)
    
    model.eval()
    with torch.no_grad():
        output = model(test_input)
    
    assert output is not None, "Model output is None"
    assert output.shape[0] == input_shape[0], "Batch size mismatch"
    
    if output_shape:
        assert output.shape == output_shape, f"Output shape mismatch: {output.shape} vs {output_shape}"
    
    return output

def create_test_summary(test_results: dict) -> str:
    """Create a summary of test results"""
    total = test_results.get('total', 0)
    passed = test_results.get('passed', 0)
    failed = test_results.get('failed', 0)
    skipped = test_results.get('skipped', 0)
    
    success_rate = (passed / total * 100) if total > 0 else 0
    
    summary = f"""
Test Summary:
  Total:    {total}
  Passed:   {passed} ({success_rate:.1f}%)
  Failed:   {failed}
  Skipped:  {skipped}
"""
    return summary

def benchmark_function(func: Callable, iterations: int = 10, *args, **kwargs) -> dict:
    """Benchmark a function"""
    times = []
    for _ in range(iterations):
        start = time.time()
        func(*args, **kwargs)
        times.append(time.time() - start)
    
    return {
        'mean': sum(times) / len(times),
        'min': min(times),
        'max': max(times),
        'std': (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5,
        'total': sum(times)
    }








