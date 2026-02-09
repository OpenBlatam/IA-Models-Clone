"""
Orchestrator Registry - Benchmark registration and management.

This module handles the registration and management of benchmarks
in the orchestrator system.
"""

import logging
from typing import Dict, List, Type

from benchmarks.base_benchmark import BaseBenchmark

logger = logging.getLogger(__name__)


class BenchmarkRegistry:
    """
    Registry for benchmark classes.
    
    Manages registration and retrieval of benchmark implementations.
    """
    
    def __init__(self):
        """Initialize registry."""
        self._benchmarks: Dict[str, Type[BaseBenchmark]] = {}
        self._register_default_benchmarks()
    
    def _register_default_benchmarks(self):
        """Register default benchmarks with lazy loading."""
        # Lazy import mapping
        benchmark_imports = {
            "mmlu": ("benchmarks.mmlu_benchmark", "MMLUBenchmark"),
            "hellaswag": ("benchmarks.hellaswag_benchmark", "HellaSwagBenchmark"),
            "gsm8k": ("benchmarks.gsm8k_benchmark", "GSM8KBenchmark"),
            "truthfulqa": ("benchmarks.truthfulqa_benchmark", "TruthfulQABenchmark"),
            "humaneval": ("benchmarks.humaneval_benchmark", "HumanEvalBenchmark"),
            "arc": ("benchmarks.arc_benchmark", "ARCBenchmark"),
            "winogrande": ("benchmarks.winogrande_benchmark", "WinoGrandeBenchmark"),
            "lambada": ("benchmarks.lambada_benchmark", "LAMBADABenchmark"),
        }
        
        for name, (module_name, class_name) in benchmark_imports.items():
            try:
                module = __import__(module_name, fromlist=[class_name])
                benchmark_class = getattr(module, class_name)
                self.register(name, benchmark_class)
            except (ImportError, AttributeError) as e:
                logger.debug(f"Benchmark {name} not available: {e}")
    
    def register(self, name: str, benchmark_class: Type[BaseBenchmark]) -> None:
        """
        Register a benchmark class.
        
        Args:
            name: Benchmark name
            benchmark_class: Benchmark class (subclass of BaseBenchmark)
        
        Raises:
            ValueError: If benchmark class is invalid
        """
        if not issubclass(benchmark_class, BaseBenchmark):
            raise ValueError("Benchmark class must inherit from BaseBenchmark")
        
        self._benchmarks[name] = benchmark_class
        logger.info(f"Registered benchmark: {name}")
    
    def get(self, name: str) -> Type[BaseBenchmark]:
        """
        Get benchmark class by name.
        
        Args:
            name: Benchmark name
        
        Returns:
            Benchmark class
        
        Raises:
            KeyError: If benchmark not found
        """
        if name not in self._benchmarks:
            # Try lazy loading
            self._try_lazy_load(name)
            if name not in self._benchmarks:
                available = ", ".join(self.list())
                raise KeyError(
                    f"Unknown benchmark: {name}. "
                    f"Available benchmarks: {available}"
                )
        return self._benchmarks[name]
    
    def _try_lazy_load(self, name: str) -> None:
        """Try to lazy load a benchmark."""
        benchmark_imports = {
            "mmlu": ("benchmarks.mmlu_benchmark", "MMLUBenchmark"),
            "hellaswag": ("benchmarks.hellaswag_benchmark", "HellaSwagBenchmark"),
            "gsm8k": ("benchmarks.gsm8k_benchmark", "GSM8KBenchmark"),
            "truthfulqa": ("benchmarks.truthfulqa_benchmark", "TruthfulQABenchmark"),
            "humaneval": ("benchmarks.humaneval_benchmark", "HumanEvalBenchmark"),
            "arc": ("benchmarks.arc_benchmark", "ARCBenchmark"),
            "winogrande": ("benchmarks.winogrande_benchmark", "WinoGrandeBenchmark"),
            "lambada": ("benchmarks.lambada_benchmark", "LAMBADABenchmark"),
        }
        
        if name in benchmark_imports:
            module_name, class_name = benchmark_imports[name]
            try:
                module = __import__(module_name, fromlist=[class_name])
                benchmark_class = getattr(module, class_name)
                self.register(name, benchmark_class)
            except (ImportError, AttributeError) as e:
                logger.debug(f"Failed to lazy load benchmark {name}: {e}")
    
    def list(self) -> List[str]:
        """
        List all registered benchmark names.
        
        Returns:
            List of benchmark names
        """
        return list(self._benchmarks.keys())
    
    def has(self, name: str) -> bool:
        """
        Check if benchmark is registered.
        
        Args:
            name: Benchmark name
        
        Returns:
            True if registered
        """
        return name in self._benchmarks
    
    def count(self) -> int:
        """
        Get number of registered benchmarks.
        
        Returns:
            Number of benchmarks
        """
        return len(self._benchmarks)
    
    def get_info(self, name: str) -> Dict[str, Any]:
        """
        Get information about a benchmark.
        
        Args:
            name: Benchmark name
        
        Returns:
            Dictionary with benchmark information
        """
        if name not in self._benchmarks:
            self._try_lazy_load(name)
        
        if name not in self._benchmarks:
            return {}
        
        benchmark_class = self._benchmarks[name]
        return {
            "name": name,
            "class": benchmark_class.__name__,
            "module": benchmark_class.__module__,
            "doc": benchmark_class.__doc__ or "",
        }
    
    def list_all_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all registered benchmarks.
        
        Returns:
            Dictionary mapping benchmark names to their info
        """
        return {
            name: self.get_info(name)
            for name in self.list()
        }


__all__ = [
    "BenchmarkRegistry",
]

