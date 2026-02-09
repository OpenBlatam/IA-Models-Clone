"""
Orchestrator Types - Comprehensive data classes and types for the orchestrator.

This module defines all types, data classes, and structures
used by the orchestrator system.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

from benchmarks.types import BenchmarkResult


class ExecutionStatus(str, Enum):
    """Status of an execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class ExecutionResult:
    """
    Result of a benchmark execution.
    
    Contains comprehensive information about the execution,
    including results, errors, timing, and metadata.
    """
    model_name: str
    benchmark_name: str
    result: Optional[BenchmarkResult] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    success: bool = False
    status: ExecutionStatus = ExecutionStatus.PENDING
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "benchmark_name": self.benchmark_name,
            "result": self.result.to_dict() if self.result else None,
            "error": self.error,
            "execution_time": self.execution_time,
            "success": self.success,
            "status": self.status.value,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "warnings": self.warnings,
        }
    
    def is_successful(self) -> bool:
        """Check if execution was successful."""
        return self.success and self.status == ExecutionStatus.COMPLETED
    
    def has_error(self) -> bool:
        """Check if execution has an error."""
        return self.error is not None or self.status == ExecutionStatus.FAILED
    
    def get_summary(self) -> str:
        """Get human-readable summary."""
        status_icon = "✓" if self.success else "✗"
        summary = f"{status_icon} {self.model_name} - {self.benchmark_name}"
        
        if self.result:
            summary += f"\n  Accuracy: {self.result.accuracy:.2%}"
            summary += f"\n  Throughput: {self.result.throughput:.2f} tokens/s"
        
        if self.error:
            summary += f"\n  Error: {self.error}"
        
        summary += f"\n  Time: {self.execution_time:.2f}s"
        
        return summary
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionResult":
        """
        Create ExecutionResult from dictionary.
        
        Args:
            data: Dictionary with execution result data
        
        Returns:
            ExecutionResult instance
        """
        # Handle status enum
        if "status" in data and isinstance(data["status"], str):
            data["status"] = ExecutionStatus(data["status"])
        
        # Handle BenchmarkResult
        if "result" in data and isinstance(data["result"], dict):
            from benchmarks.types import BenchmarkResult
            data["result"] = BenchmarkResult.from_dict(data["result"])
        
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ExecutionPlan:
    """
    Plan for executing multiple benchmarks.
    
    Contains the list of models and benchmarks to run,
    along with execution strategy and configuration.
    """
    models: List[Any]
    benchmarks: List[Any]
    parallel: bool = False
    max_workers: int = 1
    timeout: Optional[float] = None
    retry_count: int = 0
    
    def get_total_tasks(self) -> int:
        """Get total number of tasks."""
        return len(self.models) * len(self.benchmarks)
    
    def get_tasks(self) -> List[tuple]:
        """Get list of (model, benchmark) tuples."""
        tasks = []
        for model in self.models:
            for benchmark in self.benchmarks:
                tasks.append((model, benchmark))
        return tasks


__all__ = [
    "ExecutionStatus",
    "ExecutionResult",
    "ExecutionPlan",
]

