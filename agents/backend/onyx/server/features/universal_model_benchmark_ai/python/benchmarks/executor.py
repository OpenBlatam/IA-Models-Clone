"""
Benchmark Executor - Enhanced execution logic for benchmarks.

This module handles the actual execution of benchmarks with:
- Improved error handling and retry logic
- Progress tracking and callbacks
- Memory monitoring
- Timeout support
- Result validation
"""

import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from contextlib import contextmanager
from .types import (
    BenchmarkResult,
    BenchmarkStatus,
    SampleResult,
    BenchmarkProgress,
    BenchmarkConfig,
)
from core.model_loader import ModelLoader
from core.utils import get_memory_usage, get_gpu_memory_usage, timer
from core.resilience.timeout_utils import timeout_context, TimeoutException
from core.error_handling import ErrorCollector, error_context, InferenceError

logger = logging.getLogger(__name__)


class BenchmarkExecutor:
    """
    Executes benchmarks on models with enhanced features.
    
    Features:
    - Automatic retry on failure
    - Timeout support
    - Progress tracking
    - Memory monitoring
    - Error collection
    - Result validation
    """
    
    def __init__(
        self,
        timeout: Optional[float] = None,
        retry_count: int = 0,
        retry_delay: float = 1.0,
    ):
        """
        Initialize executor.
        
        Args:
            timeout: Timeout in seconds (None = no timeout)
            retry_count: Number of retries on failure
            retry_delay: Delay between retries in seconds
        """
        self.timeout = timeout
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self._memory_baseline: Optional[Dict[str, float]] = None
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        memory = get_memory_usage()
        gpu_memory = get_gpu_memory_usage()
        
        return {
            "cpu_mb": memory.get("rss_mb", 0.0),
            "gpu_mb": gpu_memory.get("allocated_mb", 0.0),
        }
    
    @contextmanager
    def _timeout_context(self):
        """Context manager for timeout support."""
        if self.timeout is None:
            yield
            return
        
        with timeout_context(self.timeout, f"Benchmark {self.__class__.__name__}"):
            yield
    
    def _execute_with_retry(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function with retry logic.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            Function result
        
        Raises:
            Exception: If all retries fail
        """
        last_exception = None
        
        for attempt in range(self.retry_count + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.retry_count:
                    logger.warning(
                        f"Attempt {attempt + 1}/{self.retry_count + 1} failed: {e}. "
                        f"Retrying in {self.retry_delay}s..."
                    )
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"All {self.retry_count + 1} attempts failed")
        
        raise last_exception
    
    def execute_benchmark(
        self,
        benchmark_instance: Any,
        model_loader: ModelLoader,
        progress_callback: Optional[Callable[[BenchmarkProgress], None]] = None,
        save_results: bool = True,
        config: Optional[BenchmarkConfig] = None,
    ) -> BenchmarkResult:
        """
        Execute a benchmark with enhanced features.
        
        Args:
            benchmark_instance: Benchmark instance (BaseBenchmark)
            model_loader: Model loader instance
            progress_callback: Optional progress callback
            save_results: Whether to save individual results
            config: Optional benchmark configuration
        
        Returns:
            BenchmarkResult with metrics
        """
        # Validate benchmark instance
        if not hasattr(benchmark_instance, 'name'):
            raise ValueError("Benchmark instance must have a 'name' attribute")
        
        if not hasattr(benchmark_instance, 'format_prompt'):
            raise ValueError("Benchmark instance must have a 'format_prompt' method")
        
        if not hasattr(benchmark_instance, 'evaluate_answer'):
            raise ValueError("Benchmark instance must have an 'evaluate_answer' method")
        
        # Load dataset if needed
        if benchmark_instance.dataset is None:
            benchmark_instance.load_dataset()
        
        if benchmark_instance.dataset is None or len(benchmark_instance.dataset) == 0:
            raise ValueError("Dataset not loaded or empty")
        
        # Get timeout from config or use default
        timeout = config.timeout if config else self.timeout
        
        logger.info(f"Running benchmark: {benchmark_instance.name}")
        logger.info(f"Dataset size: {len(benchmark_instance.dataset)}")
        logger.info(f"Shots: {benchmark_instance.shots}, Batch size: {benchmark_instance.batch_size}")
        if timeout:
            logger.info(f"Timeout: {timeout}s")
        
        # Measure baseline memory
        self._memory_baseline = self._get_memory_usage()
        
        # Initialize tracking variables
        latencies = []
        correct = 0
        total = 0
        total_tokens = 0
        errors = []
        warnings = []
        start_time = time.time()
        
        # Load model if not loaded
        if not model_loader.is_loaded:
            logger.info("Loading model...")
            try:
                model_info = model_loader.load()
                model = model_info["model"]
                tokenizer = model_info.get("tokenizer")
            except Exception as e:
                error_msg = f"Failed to load model: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
                return BenchmarkResult(
                    benchmark_name=benchmark_instance.name,
                    model_name=model_loader.config.model_name,
                    accuracy=0.0,
                    latency_p50=0.0,
                    latency_p95=0.0,
                    latency_p99=0.0,
                    throughput=0.0,
                    memory_usage={},
                    total_samples=0,
                    correct_samples=0,
                    status=BenchmarkStatus.FAILED,
                    errors=errors,
                )
        else:
            model_info = model_loader._loaded_components
            model = model_info["model"]
            tokenizer = model_info.get("tokenizer")
        
        max_samples = config.max_samples if config else None
        dataset_size = min(len(benchmark_instance.dataset), max_samples) if max_samples else len(benchmark_instance.dataset)
        
        try:
            with self._timeout_context() if timeout else contextmanager(lambda: iter([None]))():
                for i, example in enumerate(benchmark_instance.dataset):
                    if max_samples and i >= max_samples:
                        break
                    
                    try:
                        prompt = benchmark_instance.format_prompt(example)
                        
                        # Measure latency
                        step_start = time.time()
                        prediction = model_loader.generate(
                            prompt,
                            max_tokens=512,
                            temperature=0.0  # Deterministic for evaluation
                        )
                        latency = time.time() - step_start
                        latencies.append(latency)
                        
                        # Count tokens
                        if tokenizer and isinstance(prediction, str):
                            try:
                                tokens = len(tokenizer.encode(prediction, add_special_tokens=False))
                                total_tokens += tokens
                            except Exception as e:
                                logger.debug(f"Failed to count tokens: {e}")
                                # Estimate tokens (rough approximation)
                                total_tokens += len(prediction.split())
                        
                        # Evaluate answer
                        try:
                            is_correct = benchmark_instance.evaluate_answer(prediction, example)
                            if is_correct:
                                correct += 1
                        except Exception as e:
                            warning_msg = f"Error evaluating answer for example {i}: {e}"
                            logger.warning(warning_msg)
                            warnings.append(warning_msg)
                            is_correct = False
                        
                        total += 1
                        
                        # Save individual result
                        if save_results:
                            sample_result = SampleResult(
                                example_id=i,
                                prompt=prompt,
                                prediction=prediction,
                                correct=is_correct,
                                latency=latency,
                                tokens=total_tokens if i == 0 else None,
                            )
                            if not hasattr(benchmark_instance, 'results'):
                                benchmark_instance.results = []
                            benchmark_instance.results.append(sample_result.to_dict())
                        
                        # Progress callback
                        if progress_callback:
                            elapsed = time.time() - start_time
                            avg_latency = np.mean(latencies) if latencies else 0.0
                            accuracy = correct / total if total > 0 else 0.0
                            
                            # Estimate remaining time
                            estimated_remaining = None
                            if total > 0 and elapsed > 0:
                                rate = total / elapsed
                                remaining = dataset_size - total
                                estimated_remaining = remaining / rate if rate > 0 else None
                            
                            progress = BenchmarkProgress(
                                current=total,
                                total=dataset_size,
                                correct=correct,
                                accuracy=accuracy,
                                avg_latency=avg_latency,
                                elapsed_time=elapsed,
                                estimated_remaining=estimated_remaining,
                            )
                            progress_callback(progress)
                        
                        # Periodic logging
                        if (i + 1) % 10 == 0:
                            current_accuracy = correct / total if total > 0 else 0.0
                            logger.info(
                                f"Progress: {i + 1}/{dataset_size} "
                                f"(Accuracy: {current_accuracy:.2%}, "
                                f"Avg Latency: {np.mean(latencies):.3f}s)"
                            )
                    
                     except TimeoutException as e:
                         error_msg = f"Benchmark timed out after {timeout}s"
                         logger.error(error_msg)
                         errors.append(error_msg)
                         break
                    except Exception as e:
                        error_msg = f"Error processing example {i}: {e}"
                        logger.error(error_msg, exc_info=True)
                        errors.append(error_msg)
                        # Continue with next example
                        continue
        
        finally:
            # Unload model
            if model_loader.is_loaded:
                try:
                    model_loader.unload()
                except Exception as e:
                    logger.warning(f"Error unloading model: {e}")
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        if total == 0:
            accuracy = 0.0
            latencies_sorted = []
        else:
            accuracy = correct / total
            latencies_sorted = sorted(latencies)
        
        latency_p50 = np.percentile(latencies_sorted, 50) if latencies_sorted else 0.0
        latency_p95 = np.percentile(latencies_sorted, 95) if latencies_sorted else 0.0
        latency_p99 = np.percentile(latencies_sorted, 99) if latencies_sorted else 0.0
        throughput = total_tokens / total_time if total_time > 0 else 0.0
        
        # Memory usage
        current_memory = self._get_memory_usage()
        memory_usage = {
            "cpu_mb": current_memory.get("cpu_mb", 0.0) - self._memory_baseline.get("cpu_mb", 0.0),
            "gpu_mb": current_memory.get("gpu_mb", 0.0) - self._memory_baseline.get("gpu_mb", 0.0),
        }
        
        # Determine status
        status = BenchmarkStatus.COMPLETED
        if errors:
            status = BenchmarkStatus.FAILED
        elif total == 0:
            status = BenchmarkStatus.FAILED
        
        return BenchmarkResult(
            benchmark_name=benchmark_instance.name,
            model_name=model_loader.config.model_name,
            accuracy=accuracy,
            latency_p50=latency_p50,
            latency_p95=latency_p95,
            latency_p99=latency_p99,
            throughput=throughput,
            memory_usage=memory_usage,
            total_samples=total,
            correct_samples=correct,
            status=status,
            errors=errors,
            warnings=warnings,
            metadata={
                "total_time": total_time,
                "total_tokens": total_tokens,
                "shots": benchmark_instance.shots,
                "dataset_size": len(benchmark_instance.dataset),
            }
        )


__all__ = [
    "BenchmarkExecutor",
    "TimeoutError",
]
