"""
Benchmark Types - Comprehensive data classes and types for benchmarks.

This module defines all types, data classes, and structures
used by the benchmark system, with enhanced functionality.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum
import json


class BenchmarkStatus(str, Enum):
    """Status of a benchmark execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BenchmarkResult:
    """
    Result of a benchmark execution.
    
    Contains comprehensive metrics and metadata about the benchmark run.
    """
    benchmark_name: str
    model_name: str
    accuracy: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    throughput: float  # tokens/second
    memory_usage: Dict[str, float]  # GPU/CPU memory
    total_samples: int
    correct_samples: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    status: BenchmarkStatus = BenchmarkStatus.COMPLETED
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "benchmark_name": self.benchmark_name,
            "model_name": self.model_name,
            "accuracy": self.accuracy,
            "latency_p50": self.latency_p50,
            "latency_p95": self.latency_p95,
            "latency_p99": self.latency_p99,
            "throughput": self.throughput,
            "memory_usage": self.memory_usage,
            "total_samples": self.total_samples,
            "correct_samples": self.correct_samples,
            "timestamp": self.timestamp,
            "status": self.status.value,
            "metadata": self.metadata,
            "errors": self.errors,
            "warnings": self.warnings,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert result to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def summary(self) -> str:
        """Get human-readable summary string."""
        return (
            f"Benchmark: {self.benchmark_name}\n"
            f"Model: {self.model_name}\n"
            f"Status: {self.status.value}\n"
            f"Accuracy: {self.accuracy:.2%}\n"
            f"Latency P50: {self.latency_p50:.3f}s\n"
            f"Latency P95: {self.latency_p95:.3f}s\n"
            f"Latency P99: {self.latency_p99:.3f}s\n"
            f"Throughput: {self.throughput:.2f} tokens/s\n"
            f"Samples: {self.correct_samples}/{self.total_samples}\n"
            f"Memory: {self.memory_usage.get('cpu_mb', 0):.1f} MB CPU, "
            f"{self.memory_usage.get('gpu_mb', 0):.1f} MB GPU"
        )
    
    def is_successful(self) -> bool:
        """Check if benchmark completed successfully."""
        return self.status == BenchmarkStatus.COMPLETED
    
    def has_errors(self) -> bool:
        """Check if benchmark has errors."""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if benchmark has warnings."""
        return len(self.warnings) > 0
    
    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)
        self.status = BenchmarkStatus.FAILED
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)
    
    def get_metrics_dict(self) -> Dict[str, float]:
        """Get metrics as dictionary."""
        return {
            "accuracy": self.accuracy,
            "latency_p50": self.latency_p50,
            "latency_p95": self.latency_p95,
            "latency_p99": self.latency_p99,
            "throughput": self.throughput,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkResult":
        """Create BenchmarkResult from dictionary."""
        status = data.get("status", BenchmarkStatus.COMPLETED.value)
        if isinstance(status, str):
            status = BenchmarkStatus(status)
        
        return cls(
            benchmark_name=data["benchmark_name"],
            model_name=data["model_name"],
            accuracy=data["accuracy"],
            latency_p50=data["latency_p50"],
            latency_p95=data["latency_p95"],
            latency_p99=data["latency_p99"],
            throughput=data["throughput"],
            memory_usage=data.get("memory_usage", {}),
            total_samples=data["total_samples"],
            correct_samples=data["correct_samples"],
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            status=status,
            metadata=data.get("metadata", {}),
            errors=data.get("errors", []),
            warnings=data.get("warnings", []),
        )


@dataclass
class BenchmarkConfig:
    """
    Configuration for a benchmark.
    
    Contains all settings needed to run a benchmark.
    """
    name: str
    dataset_name: Optional[str] = None
    dataset_config: Optional[str] = None
    shots: int = 0
    batch_size: int = 1
    max_samples: Optional[int] = None
    cache_dir: Optional[str] = None
    timeout: Optional[float] = None
    retry_count: int = 0
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "name": self.name,
            "dataset_name": self.dataset_name,
            "dataset_config": self.dataset_config,
            "shots": self.shots,
            "batch_size": self.batch_size,
            "max_samples": self.max_samples,
            "cache_dir": self.cache_dir,
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "extra_kwargs": self.extra_kwargs,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkConfig":
        """Create BenchmarkConfig from dictionary."""
        return cls(
            name=data["name"],
            dataset_name=data.get("dataset_name"),
            dataset_config=data.get("dataset_config"),
            shots=data.get("shots", 0),
            batch_size=data.get("batch_size", 1),
            max_samples=data.get("max_samples"),
            cache_dir=data.get("cache_dir"),
            timeout=data.get("timeout"),
            retry_count=data.get("retry_count", 0),
            extra_kwargs=data.get("extra_kwargs", {}),
        )
    
    def validate(self) -> List[str]:
        """Validate configuration. Returns list of error messages."""
        errors = []
        if not self.name:
            errors.append("Benchmark name is required")
        if self.shots < 0:
            errors.append("Shots must be non-negative")
        if self.batch_size < 1:
            errors.append("Batch size must be at least 1")
        if self.max_samples is not None and self.max_samples < 1:
            errors.append("Max samples must be at least 1 if specified")
        if self.timeout is not None and self.timeout <= 0:
            errors.append("Timeout must be positive if specified")
        if self.retry_count < 0:
            errors.append("Retry count must be non-negative")
        return errors


@dataclass
class SampleResult:
    """Result for a single sample in a benchmark."""
    example_id: int
    prompt: str
    prediction: str
    correct: bool
    latency: float
    tokens: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "example_id": self.example_id,
            "prompt": self.prompt,
            "prediction": self.prediction,
            "correct": self.correct,
            "latency": self.latency,
            "tokens": self.tokens,
            "metadata": self.metadata,
        }


@dataclass
class BenchmarkProgress:
    """Progress information for a running benchmark."""
    current: int
    total: int
    correct: int
    accuracy: float
    avg_latency: float
    elapsed_time: float
    estimated_remaining: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "current": self.current,
            "total": self.total,
            "correct": self.correct,
            "accuracy": self.accuracy,
            "avg_latency": self.avg_latency,
            "elapsed_time": self.elapsed_time,
            "estimated_remaining": self.estimated_remaining,
        }
    
    @property
    def percent_complete(self) -> float:
        """Get completion percentage."""
        if self.total == 0:
            return 0.0
        return (self.current / self.total) * 100.0
    
    @property
    def throughput(self) -> float:
        """Get current throughput (samples/second)."""
        if self.elapsed_time == 0:
            return 0.0
        return self.current / self.elapsed_time


__all__ = [
    "BenchmarkStatus",
    "BenchmarkResult",
    "BenchmarkConfig",
    "SampleResult",
    "BenchmarkProgress",
]

