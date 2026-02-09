"""
Results Management Module - Advanced result handling and comparison.

Provides:
- Result storage and retrieval
- Result comparison
- Result aggregation
- Result filtering and querying
"""

import logging
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
import json
import sqlite3
from collections import defaultdict

logger = logging.getLogger(__name__)


class ResultStatus(str, Enum):
    """Result status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BenchmarkResult:
    """
    Enhanced benchmark result with additional metadata.
    
    Extends the base BenchmarkResult with:
    - Status tracking
    - Error information
    - Execution metadata
    - Comparison data
    """
    benchmark_name: str
    model_name: str
    accuracy: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    throughput: float
    memory_usage: Dict[str, float]
    total_samples: int
    correct_samples: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    status: ResultStatus = ResultStatus.COMPLETED
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkResult":
        """Create from dictionary."""
        if "status" in data and isinstance(data["status"], str):
            data["status"] = ResultStatus(data["status"])
        return cls(**data)
    
    def get_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate composite score.
        
        Args:
            weights: Optional weights for different metrics
                    Default: accuracy (50%), throughput (30%), latency (20%)
        
        Returns:
            Composite score
        """
        if weights is None:
            weights = {
                "accuracy": 0.5,
                "throughput": 0.3,
                "latency": 0.2,
            }
        
        accuracy_score = self.accuracy * weights.get("accuracy", 0.5)
        throughput_score = (self.throughput / 1000.0) * weights.get("throughput", 0.3)
        latency_score = (1.0 / (self.latency_p50 + 0.001)) * weights.get("latency", 0.2)
        
        return accuracy_score + throughput_score + latency_score


@dataclass
class ModelResults:
    """Results for a single model across multiple benchmarks."""
    model_name: str
    results: List[BenchmarkResult] = field(default_factory=list)
    average_accuracy: float = 0.0
    average_throughput: float = 0.0
    total_benchmarks: int = 0
    
    def add_result(self, result: BenchmarkResult) -> None:
        """Add a benchmark result."""
        self.results.append(result)
        self._update_averages()
    
    def _update_averages(self) -> None:
        """Update average metrics."""
        if not self.results:
            return
        
        self.total_benchmarks = len(self.results)
        self.average_accuracy = sum(r.accuracy for r in self.results) / self.total_benchmarks
        self.average_throughput = sum(r.throughput for r in self.results) / self.total_benchmarks
    
    def get_best_benchmark(self) -> Optional[BenchmarkResult]:
        """Get best performing benchmark."""
        if not self.results:
            return None
        return max(self.results, key=lambda r: r.accuracy)
    
    def get_worst_benchmark(self) -> Optional[BenchmarkResult]:
        """Get worst performing benchmark."""
        if not self.results:
            return None
        return min(self.results, key=lambda r: r.accuracy)


@dataclass
class ComparisonResults:
    """Comparison results between multiple models."""
    benchmark_name: str
    model_results: List[BenchmarkResult] = field(default_factory=list)
    rankings: Dict[str, int] = field(default_factory=dict)
    best_model: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def add_result(self, result: BenchmarkResult) -> None:
        """Add a model result."""
        self.model_results.append(result)
        self._update_rankings()
    
    def _update_rankings(self) -> None:
        """Update model rankings."""
        if not self.model_results:
            return
        
        # Sort by composite score
        sorted_results = sorted(
            self.model_results,
            key=lambda r: r.get_score(),
            reverse=True
        )
        
        self.rankings = {
            result.model_name: rank + 1
            for rank, result in enumerate(sorted_results)
        }
        
        if sorted_results:
            self.best_model = sorted_results[0].model_name
    
    def get_winner(self) -> Optional[BenchmarkResult]:
        """Get winning model result."""
        if not self.model_results:
            return None
        
        return max(self.model_results, key=lambda r: r.get_score())


class ResultsManager:
    """
    Manager for benchmark results.
    
    Provides:
    - Storage and retrieval
    - Querying and filtering
    - Comparison generation
    - Export functionality
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize results manager.
        
        Args:
            storage_path: Path to SQLite database (optional)
        """
        self.storage_path = Path(storage_path) if storage_path else Path("results.db")
        self.results: List[BenchmarkResult] = []
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                benchmark_name TEXT NOT NULL,
                model_name TEXT NOT NULL,
                accuracy REAL,
                latency_p50 REAL,
                latency_p95 REAL,
                latency_p99 REAL,
                throughput REAL,
                total_samples INTEGER,
                correct_samples INTEGER,
                timestamp TEXT,
                status TEXT,
                error TEXT,
                execution_time REAL,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_benchmark_model 
            ON results(benchmark_name, model_name)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON results(timestamp)
        """)
        
        conn.commit()
        conn.close()
    
    def save_result(self, result: BenchmarkResult) -> None:
        """
        Save a benchmark result.
        
        Args:
            result: Benchmark result to save
        """
        self.results.append(result)
        
        # Save to database
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO results (
                benchmark_name, model_name, accuracy,
                latency_p50, latency_p95, latency_p99,
                throughput, total_samples, correct_samples,
                timestamp, status, error, execution_time, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result.benchmark_name,
            result.model_name,
            result.accuracy,
            result.latency_p50,
            result.latency_p95,
            result.latency_p99,
            result.throughput,
            result.total_samples,
            result.correct_samples,
            result.timestamp,
            result.status.value,
            result.error,
            result.execution_time,
            json.dumps(result.metadata),
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Saved result: {result.model_name} - {result.benchmark_name}")
    
    def get_results(
        self,
        model_name: Optional[str] = None,
        benchmark_name: Optional[str] = None,
        status: Optional[ResultStatus] = None,
        limit: Optional[int] = None,
    ) -> List[BenchmarkResult]:
        """
        Query results with filters.
        
        Args:
            model_name: Filter by model name
            benchmark_name: Filter by benchmark name
            status: Filter by status
            limit: Maximum number of results
            
        Returns:
            List of matching results
        """
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM results WHERE 1=1"
        params = []
        
        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)
        
        if benchmark_name:
            query += " AND benchmark_name = ?"
            params.append(benchmark_name)
        
        if status:
            query += " AND status = ?"
            params.append(status.value)
        
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        # Convert to BenchmarkResult objects
        results = []
        for row in rows:
            result = BenchmarkResult(
                benchmark_name=row[1],
                model_name=row[2],
                accuracy=row[3] or 0.0,
                latency_p50=row[4] or 0.0,
                latency_p95=row[5] or 0.0,
                latency_p99=row[6] or 0.0,
                throughput=row[7] or 0.0,
                total_samples=row[8] or 0,
                correct_samples=row[9] or 0,
                timestamp=row[10] or "",
                status=ResultStatus(row[11]) if row[11] else ResultStatus.COMPLETED,
                error=row[12],
                execution_time=row[13] or 0.0,
                memory_usage={},
                metadata=json.loads(row[14]) if row[14] else {},
            )
            results.append(result)
        
        return results
    
    def get_model_results(self, model_name: str) -> ModelResults:
        """
        Get all results for a model.
        
        Args:
            model_name: Model name
            
        Returns:
            ModelResults object
        """
        results = self.get_results(model_name=model_name)
        model_results = ModelResults(model_name=model_name)
        
        for result in results:
            model_results.add_result(result)
        
        return model_results
    
    def get_comparison(
        self,
        benchmark_name: str,
        model_names: Optional[List[str]] = None,
    ) -> ComparisonResults:
        """
        Get comparison results for a benchmark.
        
        Args:
            benchmark_name: Benchmark name
            model_names: Optional list of model names to compare
            
        Returns:
            ComparisonResults object
        """
        results = self.get_results(benchmark_name=benchmark_name)
        
        if model_names:
            results = [r for r in results if r.model_name in model_names]
        
        comparison = ComparisonResults(benchmark_name=benchmark_name)
        
        for result in results:
            comparison.add_result(result)
        
        return comparison
    
    def get_best_models(
        self,
        benchmark_name: str,
        top_k: int = 5,
    ) -> List[BenchmarkResult]:
        """
        Get top K models for a benchmark.
        
        Args:
            benchmark_name: Benchmark name
            top_k: Number of top models to return
            
        Returns:
            List of top K results
        """
        results = self.get_results(benchmark_name=benchmark_name)
        results.sort(key=lambda r: r.get_score(), reverse=True)
        return results[:top_k]
    
    def export_results(
        self,
        output_path: Path,
        format: str = "json",
        filters: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Export results to file.
        
        Args:
            output_path: Output file path
            format: Export format (json, csv)
            filters: Optional filters
            
        Returns:
            Path to exported file
        """
        results = self.get_results(
            model_name=filters.get("model_name") if filters else None,
            benchmark_name=filters.get("benchmark_name") if filters else None,
        )
        
        if format == "json":
            with open(output_path, 'w') as f:
                json.dump([r.to_dict() for r in results], f, indent=2, default=str)
        elif format == "csv":
            import csv
            with open(output_path, 'w', newline='') as f:
                if results:
                    writer = csv.DictWriter(f, fieldnames=results[0].to_dict().keys())
                    writer.writeheader()
                    for result in results:
                        writer.writerow(result.to_dict())
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Exported {len(results)} results to {output_path}")
        return output_path
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics."""
        all_results = self.get_results()
        
        if not all_results:
            return {}
        
        return {
            "total_results": len(all_results),
            "unique_models": len(set(r.model_name for r in all_results)),
            "unique_benchmarks": len(set(r.benchmark_name for r in all_results)),
            "average_accuracy": sum(r.accuracy for r in all_results) / len(all_results),
            "average_throughput": sum(r.throughput for r in all_results) / len(all_results),
            "completed": sum(1 for r in all_results if r.status == ResultStatus.COMPLETED),
            "failed": sum(1 for r in all_results if r.status == ResultStatus.FAILED),
        }
