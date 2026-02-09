"""
Benchmarks Package - All benchmark implementations.

This package provides:
- BaseBenchmark: Base class for all benchmarks
- MMLUBenchmark: Massive Multitask Language Understanding
- HellaSwagBenchmark: Commonsense reasoning
- GSM8KBenchmark: Mathematical reasoning
- TruthfulQABenchmark: Truthfulness evaluation
"""

from .base_benchmark import BaseBenchmark
from .types import BenchmarkResult, BenchmarkConfig
from .executor import BenchmarkExecutor
from .mmlu_benchmark import MMLUBenchmark

# Lazy imports for other benchmarks
_LAZY_IMPORTS = {
    'HellaSwagBenchmark': '.hellaswag_benchmark',
    'GSM8KBenchmark': '.gsm8k_benchmark',
    'TruthfulQABenchmark': '.truthfulqa_benchmark',
    'HumanEvalBenchmark': '.humaneval_benchmark',
    'ARCBenchmark': '.arc_benchmark',
    'WinoGrandeBenchmark': '.winogrande_benchmark',
    'LAMBADABenchmark': '.lambada_benchmark',
}

_import_cache = {}


def __getattr__(name: str):
    """Lazy import system for benchmarks."""
    if name.startswith('_'):
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    
    if name in _import_cache:
        return _import_cache[name]
    
    module_path = _LAZY_IMPORTS[name]
    try:
        module = __import__(module_path, fromlist=[name], level=1)
        _import_cache[name] = getattr(module, name)
        return _import_cache[name]
    except (ImportError, AttributeError) as e:
        raise AttributeError(
            f"module '{__name__}' has no attribute '{name}'. "
            f"Failed to import: {e}"
        ) from e


__all__ = [
    "BaseBenchmark",
    "BenchmarkResult",
    "BenchmarkConfig",
    "BenchmarkExecutor",
    "MMLUBenchmark",
    "HellaSwagBenchmark",
    "GSM8KBenchmark",
    "TruthfulQABenchmark",
    "HumanEvalBenchmark",
    "ARCBenchmark",
    "WinoGrandeBenchmark",
    "LAMBADABenchmark",
]
