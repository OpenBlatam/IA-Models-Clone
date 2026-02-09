"""
Orchestrator Results - Enhanced result management and reporting.

This module handles result aggregation, filtering, analysis, and reporting
for the orchestrator system with comprehensive features.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from collections import defaultdict

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from .types import ExecutionResult, ExecutionStatus
from core import save_results, format_duration, load_results

console = Console()
logger = logging.getLogger(__name__)


class ResultsManager:
    """
    Manages benchmark execution results with enhanced features.
    
    Features:
    - Result aggregation and filtering
    - Statistical analysis
    - Comparison and ranking
    - Export in multiple formats
    - Result validation
    """
    
    def __init__(self):
        """Initialize results manager."""
        self.results: List[ExecutionResult] = []
    
    def add(self, result: ExecutionResult) -> None:
        """
        Add a result.
        
        Args:
            result: Execution result to add
        """
        self.results.append(result)
    
    def add_all(self, results: List[ExecutionResult]) -> None:
        """
        Add multiple results.
        
        Args:
            results: List of execution results
        """
        self.results.extend(results)
    
    def get_by_model(self, model_name: str) -> List[ExecutionResult]:
        """
        Get results for a specific model.
        
        Args:
            model_name: Model name
        
        Returns:
            List of execution results
        """
        return [r for r in self.results if r.model_name == model_name]
    
    def get_by_benchmark(self, benchmark_name: str) -> List[ExecutionResult]:
        """
        Get results for a specific benchmark.
        
        Args:
            benchmark_name: Benchmark name
        
        Returns:
            List of execution results
        """
        return [r for r in self.results if r.benchmark_name == benchmark_name]
    
    def get_successful(self) -> List[ExecutionResult]:
        """Get all successful results."""
        return [r for r in self.results if r.is_successful()]
    
    def get_failed(self) -> List[ExecutionResult]:
        """Get all failed results."""
        return [r for r in self.results if r.has_error()]
    
    def filter(
        self,
        model_name: Optional[str] = None,
        benchmark_name: Optional[str] = None,
        status: Optional[ExecutionStatus] = None,
        min_accuracy: Optional[float] = None,
        max_execution_time: Optional[float] = None,
    ) -> List[ExecutionResult]:
        """
        Filter results by multiple criteria.
        
        Args:
            model_name: Filter by model name
            benchmark_name: Filter by benchmark name
            status: Filter by execution status
            min_accuracy: Minimum accuracy threshold
            max_execution_time: Maximum execution time
        
        Returns:
            Filtered list of results
        """
        filtered = self.results
        
        if model_name:
            filtered = [r for r in filtered if r.model_name == model_name]
        
        if benchmark_name:
            filtered = [r for r in filtered if r.benchmark_name == benchmark_name]
        
        if status:
            filtered = [r for r in filtered if r.status == status]
        
        if min_accuracy is not None:
            filtered = [
                r for r in filtered
                if r.result and r.result.accuracy >= min_accuracy
            ]
        
        if max_execution_time is not None:
            filtered = [
                r for r in filtered
                if r.execution_time <= max_execution_time
            ]
        
        return filtered
    
    def get_best_model(self, benchmark_name: str) -> Optional[ExecutionResult]:
        """
        Get best model for a benchmark (highest accuracy).
        
        Args:
            benchmark_name: Benchmark name
        
        Returns:
            Best execution result or None
        """
        benchmark_results = self.get_by_benchmark(benchmark_name)
        successful = [r for r in benchmark_results if r.is_successful() and r.result]
        
        if not successful:
            return None
        
        return max(successful, key=lambda r: r.result.accuracy)
    
    def get_ranking(
        self,
        benchmark_name: str,
        metric: str = "accuracy"
    ) -> List[ExecutionResult]:
        """
        Get ranking of models for a benchmark.
        
        Args:
            benchmark_name: Benchmark name
            metric: Metric to rank by (accuracy, throughput, latency_p50)
        
        Returns:
            List of results sorted by metric (descending)
        """
        benchmark_results = self.get_by_benchmark(benchmark_name)
        successful = [r for r in benchmark_results if r.is_successful() and r.result]
        
        if not successful:
            return []
        
        def get_metric_value(result: ExecutionResult) -> float:
            if metric == "accuracy":
                return result.result.accuracy
            elif metric == "throughput":
                return result.result.throughput
            elif metric == "latency_p50":
                return -result.result.latency_p50  # Negative for descending
            else:
                return 0.0
        
        return sorted(successful, key=get_metric_value, reverse=True)
    
    def compare_models(
        self,
        model_names: List[str],
        benchmark_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple models.
        
        Args:
            model_names: List of model names to compare
            benchmark_name: Optional benchmark name to filter
        
        Returns:
            Dictionary with comparison data
        """
        comparison = {
            "models": model_names,
            "benchmarks": {},
        }
        
        benchmarks = set(r.benchmark_name for r in self.results)
        if benchmark_name:
            benchmarks = {benchmark_name}
        
        for bench_name in benchmarks:
            bench_results = []
            for model_name in model_names:
                results = [
                    r for r in self.results
                    if r.model_name == model_name and r.benchmark_name == bench_name
                ]
                if results:
                    best = max(
                        [r for r in results if r.is_successful() and r.result],
                        key=lambda r: r.result.accuracy,
                        default=None
                    )
                    if best:
                        bench_results.append({
                            "model": model_name,
                            "accuracy": best.result.accuracy,
                            "throughput": best.result.throughput,
                            "latency_p50": best.result.latency_p50,
                            "execution_time": best.execution_time,
                        })
            
            if bench_results:
                comparison["benchmarks"][bench_name] = bench_results
        
        return comparison
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary statistics.
        
        Returns:
            Dictionary with summary statistics
        """
        total = len(self.results)
        successful = sum(1 for r in self.results if r.is_successful())
        failed = sum(1 for r in self.results if r.has_error())
        
        total_time = sum(r.execution_time for r in self.results)
        avg_time = total_time / total if total > 0 else 0.0
        
        models = set(r.model_name for r in self.results)
        benchmarks = set(r.benchmark_name for r in self.results)
        
        # Calculate accuracy statistics
        successful_results = [r for r in self.results if r.is_successful() and r.result]
        accuracies = [r.result.accuracy for r in successful_results]
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
        
        # Status breakdown
        status_counts = defaultdict(int)
        for r in self.results:
            status_counts[r.status.value] += 1
        
        return {
            "total": total,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total if total > 0 else 0.0,
            "total_time": total_time,
            "average_time": avg_time,
            "models": len(models),
            "benchmarks": len(benchmarks),
            "average_accuracy": avg_accuracy,
            "status_breakdown": dict(status_counts),
        }
    
    def print_summary(self) -> None:
        """Print comprehensive summary table."""
        summary = self.get_summary()
        
        # Main summary table
        table = Table(title="Benchmark Summary", box=box.ROUNDED)
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        
        table.add_row("Total Runs", str(summary["total"]))
        table.add_row("Successful", f"{summary['successful']} ({summary['success_rate']:.1%})")
        table.add_row("Failed", str(summary["failed"]))
        table.add_row(
            "Total Time",
            format_duration(summary["total_time"])
        )
        table.add_row(
            "Average Time",
            format_duration(summary["average_time"])
        )
        table.add_row("Models", str(summary["models"]))
        table.add_row("Benchmarks", str(summary["benchmarks"]))
        if summary["average_accuracy"] > 0:
            table.add_row(
                "Average Accuracy",
                f"{summary['average_accuracy']:.2%}"
            )
        
        console.print(table)
        
        # Status breakdown
        if summary["status_breakdown"]:
            status_table = Table(title="Status Breakdown", box=box.ROUNDED)
            status_table.add_column("Status", style="cyan")
            status_table.add_column("Count", style="green")
            
            for status, count in summary["status_breakdown"].items():
                status_table.add_row(status, str(count))
            
            console.print("\n")
            console.print(status_table)
        
        # Best models per benchmark
        if summary["benchmarks"] > 0:
            console.print("\n[bold]Best Models per Benchmark:[/bold]")
            benchmarks = set(r.benchmark_name for r in self.results)
            for benchmark_name in sorted(benchmarks):
                best = self.get_best_model(benchmark_name)
                if best:
                    console.print(
                        f"  [cyan]{benchmark_name}[/cyan]: "
                        f"[green]{best.model_name}[/green] "
                        f"({best.result.accuracy:.2%})"
                    )
    
    def save(
        self,
        output_dir: str = "results",
        format: str = "json"
    ) -> tuple[Path, Path]:
        """
        Save results to disk in multiple formats.
        
        Args:
            output_dir: Output directory
            format: Output format (json, csv)
        
        Returns:
            Tuple of (results_path, summary_path)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save results
        results_data = [r.to_dict() for r in self.results]
        
        if format == "json":
            results_file = output_path / f"results_{timestamp}.json"
            save_results(results_data, results_file, format="json")
        elif format == "csv":
            # Flatten results for CSV
            csv_data = []
            for r in self.results:
                row = {
                    "model_name": r.model_name,
                    "benchmark_name": r.benchmark_name,
                    "success": r.success,
                    "status": r.status.value,
                    "execution_time": r.execution_time,
                    "error": r.error or "",
                }
                if r.result:
                    row.update({
                        "accuracy": r.result.accuracy,
                        "throughput": r.result.throughput,
                        "latency_p50": r.result.latency_p50,
                        "latency_p95": r.result.latency_p95,
                        "latency_p99": r.result.latency_p99,
                        "total_samples": r.result.total_samples,
                        "correct_samples": r.result.correct_samples,
                    })
                csv_data.append(row)
            
            results_file = output_path / f"results_{timestamp}.csv"
            save_results(csv_data, results_file, format="csv")
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Save summary
        summary_path = output_path / f"summary_{timestamp}.json"
        save_results(self.get_summary(), summary_path, format="json")
        
        logger.info(f"Results saved to {results_file}")
        logger.info(f"Summary saved to {summary_path}")
        
        return results_file, summary_path
    
    def load(self, results_path: Path) -> None:
        """
        Load results from file.
        
        Args:
            results_path: Path to results file
        """
        data = load_results(results_path)
        
        if isinstance(data, list):
            for item in data:
                # Convert dict to ExecutionResult
                if isinstance(item, dict):
                    result = ExecutionResult.from_dict(item)
                else:
                    result = item
                self.results.append(result)
        else:
            logger.warning("Loaded data is not a list")
    
    def clear(self) -> None:
        """Clear all results."""
        self.results.clear()
    
    def __len__(self) -> int:
        """Get number of results."""
        return len(self.results)
    
    def __iter__(self):
        """Iterate over results."""
        return iter(self.results)


__all__ = [
    "ResultsManager",
]
