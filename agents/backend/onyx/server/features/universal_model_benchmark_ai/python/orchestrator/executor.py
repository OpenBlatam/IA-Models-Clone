"""
Orchestrator Executor - Benchmark execution logic.

This module handles the actual execution of benchmarks,
including sequential and parallel execution strategies.
"""

import logging
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
)

from .types import ExecutionResult, ExecutionStatus, ExecutionPlan
from core import get_device
from core.model_loader import ModelLoader, ModelType, QuantizationType, BackendType

console = Console()
logger = logging.getLogger(__name__)


class BenchmarkExecutor:
    """
    Executes benchmarks on models.
    
    Handles:
    - Sequential execution
    - Parallel execution
    - Progress tracking
    - Error handling
    """
    
    def __init__(
        self,
        benchmark_registry: Dict[str, type],
        max_workers: int = 1,
    ):
        """
        Initialize executor.
        
        Args:
            benchmark_registry: Registry of available benchmarks
            max_workers: Maximum workers for parallel execution
        """
        self.benchmark_registry = benchmark_registry
        self.max_workers = max_workers
    
    def execute_benchmark(
        self,
        model_config: Any,
        benchmark_config: Any,
    ) -> ExecutionResult:
        """
        Execute a single benchmark on a model.
        
        Args:
            model_config: Model configuration
            benchmark_config: Benchmark configuration
        
        Returns:
            ExecutionResult with results or error
        """
        start_time = datetime.now()
        benchmark_name = benchmark_config.name
        
        try:
            if benchmark_name not in self.benchmark_registry:
                raise ValueError(f"Unknown benchmark: {benchmark_name}")
            
            # Create model loader
            device = (
                get_device().value
                if model_config.device.value == "auto"
                else model_config.device.value
            )
            
            model_loader = ModelLoader(
                model_name=model_config.name,
                model_path=model_config.path or model_config.name,
                model_type=(
                    ModelType[model_config.model_type.value.upper()]
                    if hasattr(model_config, 'model_type')
                    else ModelType.CAUSAL_LM
                ),
                quantization=QuantizationType[model_config.quantization.value.upper()],
                device=device,
                backend=BackendType.AUTO,
            )
            
            # Create benchmark
            BenchmarkClass = self.benchmark_registry[benchmark_name]
            benchmark = BenchmarkClass(
                shots=benchmark_config.shots,
                max_samples=benchmark_config.max_samples
            )
            
            # Execute benchmark
            result = benchmark.run(
                model_loader,
                save_results=True,
                results_dir="results"
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ExecutionResult(
                model_name=model_config.name,
                benchmark_name=benchmark_name,
                result=result,
                execution_time=execution_time,
                success=True,
                status=ExecutionStatus.COMPLETED,
                metadata={
                    "model_type": getattr(model_config, 'model_type', None),
                    "quantization": getattr(model_config, 'quantization', None),
                }
            )
            
        except TimeoutError as e:
            logger.error(
                f"Timeout running {benchmark_name} on {model_config.name}: {e}",
                exc_info=True
            )
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ExecutionResult(
                model_name=model_config.name,
                benchmark_name=benchmark_name,
                error=f"Timeout: {str(e)}",
                execution_time=execution_time,
                success=False,
                status=ExecutionStatus.TIMEOUT,
            )
            
        except Exception as e:
            logger.error(
                f"Error running {benchmark_name} on {model_config.name}: {e}",
                exc_info=True
            )
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ExecutionResult(
                model_name=model_config.name,
                benchmark_name=benchmark_name,
                error=str(e),
                execution_time=execution_time,
                success=False,
                status=ExecutionStatus.FAILED,
            )
    
    def run_sequential(
        self,
        models: List[Any],
        benchmarks: List[Any],
        progress_callback: Optional[Callable] = None,
    ) -> List[ExecutionResult]:
        """
        Run benchmarks sequentially.
        
        Args:
            models: List of model configurations
            benchmarks: List of benchmark configurations
            progress_callback: Optional callback for progress updates
        
        Returns:
            List of execution results
        """
        results = []
        total = len(models) * len(benchmarks)
        current = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Running benchmarks...", total=total)
            
            for model_config in models:
                for benchmark_config in benchmarks:
                    current += 1
                    progress.update(
                        task,
                        description=f"[cyan]{model_config.name} - {benchmark_config.name}",
                        advance=1
                    )
                    
                    result = self.execute_benchmark(model_config, benchmark_config)
                    results.append(result)
                    
                    if progress_callback:
                        progress_callback(current, total, result)
        
        return results
    
    def run_parallel(
        self,
        models: List[Any],
        benchmarks: List[Any],
        progress_callback: Optional[Callable] = None,
        timeout: Optional[float] = None,
    ) -> List[ExecutionResult]:
        """
        Run benchmarks in parallel.
        
        Args:
            models: List of model configurations
            benchmarks: List of benchmark configurations
            progress_callback: Optional callback for progress updates
        
        Returns:
            List of execution results
        """
        results = []
        total = len(models) * len(benchmarks)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Running benchmarks...", total=total)
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                futures = {}
                for model_config in models:
                    for benchmark_config in benchmarks:
                        # Apply timeout if specified
                        if timeout:
                            future = executor.submit(
                                self._execute_with_timeout,
                                model_config,
                                benchmark_config,
                                timeout
                            )
                        else:
                            future = executor.submit(
                                self.execute_benchmark,
                                model_config,
                                benchmark_config
                            )
                        futures[future] = (model_config.name, benchmark_config.name)
                
                # Collect results
                for future in as_completed(futures):
                    model_name, benchmark_name = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                        progress.update(
                            task,
                            description=f"[cyan]{model_name} - {benchmark_name}",
                            advance=1
                        )
                        
                        if progress_callback:
                            progress.update(task, advance=0)  # Trigger callback
                    except Exception as e:
                        logger.error(f"Error in parallel execution: {e}")
                        results.append(ExecutionResult(
                            model_name=model_name,
                            benchmark_name=benchmark_name,
                            error=str(e),
                            success=False,
                            status=ExecutionStatus.FAILED,
                        ))
        
        return results
    
    def _execute_with_timeout(
        self,
        model_config: Any,
        benchmark_config: Any,
        timeout: float,
    ) -> ExecutionResult:
        """
        Execute benchmark with timeout using threading.
        
        Args:
            model_config: Model configuration
            benchmark_config: Benchmark configuration
            timeout: Timeout in seconds
        
        Returns:
            ExecutionResult
        """
        import threading
        
        result_container = [None]
        exception_container = [None]
        
        def execute():
            try:
                result_container[0] = self.execute_benchmark(
                    model_config,
                    benchmark_config
                )
            except Exception as e:
                exception_container[0] = e
        
        thread = threading.Thread(target=execute)
        thread.daemon = True
        thread.start()
        thread.join(timeout=timeout)
        
        if thread.is_alive():
            # Thread is still running, timeout occurred
            return ExecutionResult(
                model_name=model_config.name,
                benchmark_name=benchmark_config.name,
                error=f"Execution timed out after {timeout}s",
                success=False,
                status=ExecutionStatus.TIMEOUT,
            )
        
        if exception_container[0]:
            raise exception_container[0]
        
        return result_container[0]
    
    def run_with_plan(
        self,
        plan: ExecutionPlan,
        progress_callback: Optional[Callable] = None,
    ) -> List[ExecutionResult]:
        """
        Run benchmarks according to an execution plan.
        
        Args:
            plan: Execution plan
            progress_callback: Optional progress callback
        
        Returns:
            List of execution results
        """
        if plan.parallel:
            return self.run_parallel(
                plan.models,
                plan.benchmarks,
                progress_callback,
                timeout=plan.timeout,
            )
        else:
            return self.run_sequential(
                plan.models,
                plan.benchmarks,
                progress_callback,
            )


__all__ = [
    "BenchmarkExecutor",
]

