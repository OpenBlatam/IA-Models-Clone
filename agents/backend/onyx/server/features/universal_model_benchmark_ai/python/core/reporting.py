"""
Reporting Module - Enhanced report generation and visualization.

Provides:
- Comprehensive report generation
- Comparison reports with multiple scoring strategies
- Export to multiple formats (JSON, Markdown, HTML, CSV)
- Statistical analysis in reports
- Report templates and customization
"""

import json
import logging
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import statistics

from benchmarks.types import BenchmarkResult, BenchmarkStatus

logger = logging.getLogger(__name__)


class ScoringStrategy(str, Enum):
    """Scoring strategies for model comparison."""
    ACCURACY_ONLY = "accuracy_only"
    BALANCED = "balanced"
    PERFORMANCE = "performance"
    CUSTOM = "custom"


@dataclass
class ReportConfig:
    """
    Configuration for report generation.
    
    Controls what information is included and how reports are formatted.
    """
    include_detailed_results: bool = True
    include_percentiles: bool = True
    include_memory_stats: bool = True
    include_statistics: bool = True
    include_errors: bool = True
    include_warnings: bool = True
    format: str = "json"  # json, markdown, html, csv
    output_dir: str = "reports"
    scoring_strategy: ScoringStrategy = ScoringStrategy.BALANCED
    custom_scorer: Optional[Callable[[BenchmarkResult], float]] = None


class ReportGenerator:
    """
    Generate comprehensive benchmark reports.
    
    Features:
    - Single and comparison reports
    - Multiple export formats
    - Configurable scoring strategies
    - Statistical analysis
    - Custom templates
    """
    
    def __init__(self, config: Optional[ReportConfig] = None):
        """
        Initialize report generator.
        
        Args:
            config: Report configuration
        """
        self.config = config or ReportConfig()
    
    def generate_report(
        self,
        result: BenchmarkResult,
        model_name: str,
        benchmark_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a single benchmark report with comprehensive information.
        
        Args:
            result: Benchmark result
            model_name: Model name
            benchmark_name: Benchmark name
            metadata: Additional metadata
        
        Returns:
            Report dictionary
        """
        report = {
            "report_id": f"{model_name}_{benchmark_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "benchmark_name": benchmark_name,
            "status": result.status.value if hasattr(result, 'status') else "completed",
            "metrics": {
                "accuracy": result.accuracy,
                "latency_p50": result.latency_p50,
                "latency_p95": result.latency_p95,
                "latency_p99": result.latency_p99,
                "throughput": result.throughput,
            },
            "samples": {
                "total": result.total_samples,
                "correct": result.correct_samples,
                "incorrect": result.total_samples - result.correct_samples,
                "accuracy_rate": result.accuracy,
            },
        }
        
        # Add memory stats if configured
        if self.config.include_memory_stats:
            report["memory"] = result.memory_usage
        
        # Add percentiles if configured
        if self.config.include_percentiles:
            report["latency_percentiles"] = {
                "p50": result.latency_p50,
                "p95": result.latency_p95,
                "p99": result.latency_p99,
            }
        
        # Add statistics if configured
        if self.config.include_statistics:
            report["statistics"] = self._calculate_statistics(result)
        
        # Add errors and warnings if configured
        if self.config.include_errors and hasattr(result, 'errors'):
            report["errors"] = result.errors
        
        if self.config.include_warnings and hasattr(result, 'warnings'):
            report["warnings"] = result.warnings
        
        # Add metadata
        if metadata:
            report["metadata"] = metadata
        
        if self.config.include_detailed_results:
            report["detailed_metadata"] = result.metadata
        
        return report
    
    def _calculate_statistics(self, result: BenchmarkResult) -> Dict[str, Any]:
        """
        Calculate statistical information for a result.
        
        Args:
            result: Benchmark result
        
        Returns:
            Dictionary with statistical measures
        """
        return {
            "accuracy": {
                "value": result.accuracy,
                "percentage": result.accuracy * 100.0,
                "correct_samples": result.correct_samples,
                "total_samples": result.total_samples,
            },
            "latency": {
                "p50": result.latency_p50,
                "p95": result.latency_p95,
                "p99": result.latency_p99,
                "range": result.latency_p99 - result.latency_p50,
            },
            "throughput": {
                "value": result.throughput,
                "tokens_per_second": result.throughput,
            },
        }
    
    def _calculate_score(
        self,
        result: BenchmarkResult,
        strategy: Optional[ScoringStrategy] = None,
    ) -> float:
        """
        Calculate composite score for a result.
        
        Args:
            result: Benchmark result
            strategy: Scoring strategy (defaults to config)
        
        Returns:
            Composite score
        """
        strategy = strategy or self.config.scoring_strategy
        
        if strategy == ScoringStrategy.ACCURACY_ONLY:
            return result.accuracy
        
        elif strategy == ScoringStrategy.BALANCED:
            # Balanced: accuracy (50%) + throughput (30%) + latency (20%)
            return (
                result.accuracy * 0.5 +
                min(result.throughput / 1000.0, 1.0) * 0.3 +
                min(1.0 / (result.latency_p50 + 0.001), 1.0) * 0.2
            )
        
        elif strategy == ScoringStrategy.PERFORMANCE:
            # Performance-focused: accuracy (40%) + throughput (40%) + latency (20%)
            return (
                result.accuracy * 0.4 +
                min(result.throughput / 1000.0, 1.0) * 0.4 +
                min(1.0 / (result.latency_p50 + 0.001), 1.0) * 0.2
            )
        
        elif strategy == ScoringStrategy.CUSTOM:
            if self.config.custom_scorer:
                return self.config.custom_scorer(result)
            else:
                # Fallback to balanced
                return self._calculate_score(result, ScoringStrategy.BALANCED)
        
        else:
            return result.accuracy
    
    def generate_comparison_report(
        self,
        results: List[BenchmarkResult],
        model_names: List[str],
        benchmark_name: str,
        scoring_strategy: Optional[ScoringStrategy] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive comparison report for multiple models.
        
        Args:
            results: List of benchmark results
            model_names: List of model names
            benchmark_name: Benchmark name
            scoring_strategy: Optional scoring strategy override
        
        Returns:
            Comparison report dictionary
        
        Raises:
            ValueError: If results and model names don't match
        """
        if len(results) != len(model_names):
            raise ValueError("Results and model names must have same length")
        
        if not results:
            return {
                "benchmark_name": benchmark_name,
                "timestamp": datetime.now().isoformat(),
                "models": [],
                "rankings": {},
                "best_model": None,
            }
        
        # Calculate scores and rankings
        model_scores = []
        for result, model_name in zip(results, model_names):
            score = self._calculate_score(result, scoring_strategy)
            model_scores.append((model_name, result, score))
        
        # Sort by score (descending)
        model_scores.sort(key=lambda x: x[2], reverse=True)
        
        # Build comprehensive comparison report
        comparison = {
            "benchmark_name": benchmark_name,
            "timestamp": datetime.now().isoformat(),
            "scoring_strategy": (scoring_strategy or self.config.scoring_strategy).value,
            "models": [],
            "rankings": {},
            "statistics": {},
        }
        
        # Add model details
        for rank, (model_name, result, score) in enumerate(model_scores, 1):
            comparison["models"].append({
                "rank": rank,
                "model_name": model_name,
                "score": score,
                "metrics": {
                    "accuracy": result.accuracy,
                    "latency_p50": result.latency_p50,
                    "latency_p95": result.latency_p95,
                    "latency_p99": result.latency_p99,
                    "throughput": result.throughput,
                },
                "samples": {
                    "total": result.total_samples,
                    "correct": result.correct_samples,
                },
            })
            comparison["rankings"][model_name] = rank
        
        # Add statistics
        if len(results) > 1:
            accuracies = [r.accuracy for r in results]
            throughputs = [r.throughput for r in results]
            latencies = [r.latency_p50 for r in results]
            
            comparison["statistics"] = {
                "accuracy": {
                    "mean": statistics.mean(accuracies),
                    "median": statistics.median(accuracies),
                    "std_dev": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
                    "min": min(accuracies),
                    "max": max(accuracies),
                },
                "throughput": {
                    "mean": statistics.mean(throughputs),
                    "median": statistics.median(throughputs),
                    "std_dev": statistics.stdev(throughputs) if len(throughputs) > 1 else 0.0,
                    "min": min(throughputs),
                    "max": max(throughputs),
                },
                "latency": {
                    "mean": statistics.mean(latencies),
                    "median": statistics.median(latencies),
                    "std_dev": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
                    "min": min(latencies),
                    "max": max(latencies),
                },
            }
        
        comparison["best_model"] = model_scores[0][0] if model_scores else None
        
        return comparison
    
    def export_report(
        self,
        report: Dict[str, Any],
        output_path: Optional[Path] = None,
        format: Optional[str] = None,
    ) -> Path:
        """
        Export report to file in specified format.
        
        Args:
            report: Report dictionary
            output_path: Output path (optional)
            format: Format override (optional)
        
        Returns:
            Path to exported file
        
        Raises:
            ValueError: If format is unsupported
        """
        format = format or self.config.format
        
        if output_path is None:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            report_id = report.get("report_id", "report")
            output_path = output_dir / f"{report_id}.{format}"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format == "json":
                self._export_json(report, output_path)
            elif format == "markdown":
                self._export_markdown(report, output_path)
            elif format == "html":
                self._export_html(report, output_path)
            elif format == "csv":
                self._export_csv(report, output_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Report exported to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to export report: {e}", exc_info=True)
            raise
    
    def _export_json(self, report: Dict[str, Any], path: Path) -> None:
        """Export report as JSON."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)
    
    def _export_markdown(self, report: Dict[str, Any], path: Path) -> None:
        """Export report as Markdown with enhanced formatting."""
        md_lines = [
            f"# Benchmark Report: {report.get('benchmark_name', 'Unknown')}",
            "",
            f"**Model**: {report.get('model_name', 'Unknown')}",
            f"**Timestamp**: {report.get('timestamp', 'Unknown')}",
            f"**Status**: {report.get('status', 'completed')}",
            "",
            "## Metrics",
            "",
        ]
        
        metrics = report.get('metrics', {})
        md_lines.extend([
            f"- **Accuracy**: {metrics.get('accuracy', 0):.2%}",
            f"- **Latency P50**: {metrics.get('latency_p50', 0):.3f}s",
            f"- **Latency P95**: {metrics.get('latency_p95', 0):.3f}s",
            f"- **Latency P99**: {metrics.get('latency_p99', 0):.3f}s",
            f"- **Throughput**: {metrics.get('throughput', 0):.2f} tokens/s",
            "",
        ])
        
        # Add samples
        samples = report.get('samples', {})
        md_lines.extend([
            "## Samples",
            "",
            f"- **Total**: {samples.get('total', 0)}",
            f"- **Correct**: {samples.get('correct', 0)}",
            f"- **Incorrect**: {samples.get('incorrect', 0)}",
            f"- **Accuracy Rate**: {samples.get('accuracy_rate', 0):.2%}",
            "",
        ])
        
        # Add statistics if available
        if 'statistics' in report:
            md_lines.extend([
                "## Statistics",
                "",
            ])
            stats = report['statistics']
            if 'accuracy' in stats:
                md_lines.append(f"- **Accuracy**: {stats['accuracy'].get('value', 0):.2%}")
            if 'latency' in stats:
                md_lines.append(f"- **Latency Range**: {stats['latency'].get('range', 0):.3f}s")
        
        # Add errors and warnings if available
        if 'errors' in report and report['errors']:
            md_lines.extend([
                "",
                "## Errors",
                "",
            ])
            for error in report['errors']:
                md_lines.append(f"- {error}")
        
        if 'warnings' in report and report['warnings']:
            md_lines.extend([
                "",
                "## Warnings",
                "",
            ])
            for warning in report['warnings']:
                md_lines.append(f"- {warning}")
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_lines))
    
    def _export_html(self, report: Dict[str, Any], path: Path) -> None:
        """Export report as HTML with enhanced styling."""
        metrics = report.get('metrics', {})
        samples = report.get('samples', {})
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Benchmark Report: {report.get('benchmark_name', 'Unknown')}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            margin: 0;
            padding: 40px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        .metrics {{
            background: #f9f9f9;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .status {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 4px;
            font-weight: bold;
        }}
        .status.completed {{
            background: #4CAF50;
            color: white;
        }}
        .status.failed {{
            background: #f44336;
            color: white;
        }}
        .badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 0.85em;
            margin-left: 8px;
        }}
        .badge.success {{
            background: #4CAF50;
            color: white;
        }}
        .badge.warning {{
            background: #ff9800;
            color: white;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Benchmark Report: {report.get('benchmark_name', 'Unknown')}</h1>
        <p><strong>Model:</strong> {report.get('model_name', 'Unknown')}</p>
        <p><strong>Timestamp:</strong> {report.get('timestamp', 'Unknown')}</p>
        <p><strong>Status:</strong> <span class="status {report.get('status', 'completed')}">{report.get('status', 'completed')}</span></p>
        
        <div class="metrics">
            <h2>Metrics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Accuracy</td><td>{metrics.get('accuracy', 0):.2%}</td></tr>
                <tr><td>Latency P50</td><td>{metrics.get('latency_p50', 0):.3f}s</td></tr>
                <tr><td>Latency P95</td><td>{metrics.get('latency_p95', 0):.3f}s</td></tr>
                <tr><td>Latency P99</td><td>{metrics.get('latency_p99', 0):.3f}s</td></tr>
                <tr><td>Throughput</td><td>{metrics.get('throughput', 0):.2f} tokens/s</td></tr>
            </table>
        </div>
        
        <h2>Samples</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total</td><td>{samples.get('total', 0)}</td></tr>
            <tr><td>Correct</td><td>{samples.get('correct', 0)}</td></tr>
            <tr><td>Incorrect</td><td>{samples.get('incorrect', 0)}</td></tr>
            <tr><td>Accuracy Rate</td><td>{samples.get('accuracy_rate', 0):.2%}</td></tr>
        </table>
"""
        
        # Add errors and warnings if available
        if 'errors' in report and report['errors']:
            html += """
        <h2>Errors</h2>
        <ul>
"""
            for error in report['errors']:
                html += f"            <li>{error}</li>\n"
            html += "        </ul>\n"
        
        if 'warnings' in report and report['warnings']:
            html += """
        <h2>Warnings</h2>
        <ul>
"""
            for warning in report['warnings']:
                html += f"            <li>{warning}</li>\n"
            html += "        </ul>\n"
        
        html += """
    </div>
</body>
</html>"""
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html)
    
    def _export_csv(self, report: Dict[str, Any], path: Path) -> None:
        """Export report as CSV."""
        import csv
        
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Model', report.get('model_name', 'Unknown')])
            writer.writerow(['Benchmark', report.get('benchmark_name', 'Unknown')])
            writer.writerow(['Status', report.get('status', 'completed')])
            writer.writerow(['Timestamp', report.get('timestamp', 'Unknown')])
            
            # Metrics
            metrics = report.get('metrics', {})
            writer.writerow(['Accuracy', f"{metrics.get('accuracy', 0):.4f}"])
            writer.writerow(['Latency P50', f"{metrics.get('latency_p50', 0):.4f}"])
            writer.writerow(['Latency P95', f"{metrics.get('latency_p95', 0):.4f}"])
            writer.writerow(['Latency P99', f"{metrics.get('latency_p99', 0):.4f}"])
            writer.writerow(['Throughput', f"{metrics.get('throughput', 0):.4f}"])
            
            # Samples
            samples = report.get('samples', {})
            writer.writerow(['Total Samples', samples.get('total', 0)])
            writer.writerow(['Correct Samples', samples.get('correct', 0)])
            writer.writerow(['Incorrect Samples', samples.get('incorrect', 0)])


def generate_summary_report(
    all_results: List[Dict[str, Any]],
    output_path: Optional[Path] = None,
    config: Optional[ReportConfig] = None,
) -> Path:
    """
    Generate comprehensive summary report from multiple benchmark results.
    
    Args:
        all_results: List of result dictionaries
        output_path: Output path (optional)
        config: Optional report configuration
    
    Returns:
        Path to generated report
    """
    generator = ReportGenerator(config)
    
    summary = {
        "report_type": "summary",
        "timestamp": datetime.now().isoformat(),
        "total_benchmarks": len(all_results),
        "results": all_results,
        "statistics": _calculate_summary_stats(all_results),
    }
    
    if output_path is None:
        output_dir = Path(config.output_dir if config else "reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    return generator.export_report(summary, output_path)


def _calculate_summary_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate comprehensive summary statistics.
    
    Args:
        results: List of result dictionaries
    
    Returns:
        Dictionary with statistical measures
    """
    if not results:
        return {}
    
    accuracies = [r.get("metrics", {}).get("accuracy", 0.0) for r in results]
    throughputs = [r.get("metrics", {}).get("throughput", 0.0) for r in results]
    latencies_p50 = [r.get("metrics", {}).get("latency_p50", 0.0) for r in results]
    
    return {
        "accuracy": {
            "mean": statistics.mean(accuracies) if accuracies else 0.0,
            "median": statistics.median(accuracies) if accuracies else 0.0,
            "std_dev": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
            "min": min(accuracies) if accuracies else 0.0,
            "max": max(accuracies) if accuracies else 0.0,
        },
        "throughput": {
            "mean": statistics.mean(throughputs) if throughputs else 0.0,
            "median": statistics.median(throughputs) if throughputs else 0.0,
            "std_dev": statistics.stdev(throughputs) if len(throughputs) > 1 else 0.0,
            "min": min(throughputs) if throughputs else 0.0,
            "max": max(throughputs) if throughputs else 0.0,
        },
        "latency_p50": {
            "mean": statistics.mean(latencies_p50) if latencies_p50 else 0.0,
            "median": statistics.median(latencies_p50) if latencies_p50 else 0.0,
            "std_dev": statistics.stdev(latencies_p50) if len(latencies_p50) > 1 else 0.0,
            "min": min(latencies_p50) if latencies_p50 else 0.0,
            "max": max(latencies_p50) if latencies_p50 else 0.0,
        },
    }


__all__ = [
    "ScoringStrategy",
    "ReportConfig",
    "ReportGenerator",
    "generate_summary_report",
]
