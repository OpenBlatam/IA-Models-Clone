"""
Visualization Module - Enhanced data preparation for visualization.

Provides:
- Multiple chart types (bar, line, radar, scatter, heatmap)
- Comparison visualizations
- Performance graphs
- Export for visualization tools (Chart.js, Plotly, etc.)
- Interactive chart configurations
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import statistics

from benchmarks.types import BenchmarkResult

logger = logging.getLogger(__name__)


class ChartType(str, Enum):
    """Chart types."""
    BAR = "bar"
    LINE = "line"
    RADAR = "radar"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    PIE = "pie"


@dataclass
class ChartData:
    """
    Chart data structure for visualization.
    
    Compatible with Chart.js, Plotly, and other visualization libraries.
    """
    labels: List[str] = field(default_factory=list)
    datasets: List[Dict[str, Any]] = field(default_factory=list)
    options: Dict[str, Any] = field(default_factory=dict)
    chart_type: ChartType = ChartType.BAR
    title: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.chart_type.value,
            "title": self.title,
            "labels": self.labels,
            "datasets": self.datasets,
            "options": self.options,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class VisualizationGenerator:
    """
    Generate visualization data from benchmark results.
    
    Supports multiple chart types and export formats.
    """
    
    def __init__(self, default_chart_type: ChartType = ChartType.BAR):
        """
        Initialize visualization generator.
        
        Args:
            default_chart_type: Default chart type
        """
        self.default_chart_type = default_chart_type
    
    def accuracy_chart(
        self,
        results: List[BenchmarkResult],
        model_names: List[str],
        chart_type: Optional[ChartType] = None,
    ) -> ChartData:
        """
        Generate accuracy comparison chart data.
        
        Args:
            results: List of benchmark results
            model_names: List of model names
            chart_type: Optional chart type override
        
        Returns:
            ChartData for accuracy comparison
        """
        chart_type = chart_type or self.default_chart_type
        
        return ChartData(
            chart_type=chart_type,
            title="Accuracy Comparison",
            labels=model_names,
            datasets=[{
                "label": "Accuracy (%)",
                "data": [r.accuracy * 100.0 for r in results],
                "backgroundColor": "rgba(54, 162, 235, 0.5)",
                "borderColor": "rgba(54, 162, 235, 1)",
                "borderWidth": 2,
            }],
            options={
                "scales": {
                    "y": {
                        "beginAtZero": True,
                        "max": 100,
                        "title": {
                            "display": True,
                            "text": "Accuracy (%)"
                        },
                        "ticks": {
                            "callback": "function(value) { return value + '%'; }"
                        }
                    }
                },
                "plugins": {
                    "legend": {
                        "display": True,
                        "position": "top"
                    }
                }
            }
        )
    
    def latency_chart(
        self,
        results: List[BenchmarkResult],
        model_names: List[str],
        chart_type: Optional[ChartType] = None,
    ) -> ChartData:
        """
        Generate latency comparison chart data.
        
        Args:
            results: List of benchmark results
            model_names: List of model names
            chart_type: Optional chart type override
        
        Returns:
            ChartData for latency comparison
        """
        chart_type = chart_type or ChartType.BAR
        
        return ChartData(
            chart_type=chart_type,
            title="Latency Comparison",
            labels=model_names,
            datasets=[
                {
                    "label": "P50",
                    "data": [r.latency_p50 for r in results],
                    "backgroundColor": "rgba(255, 99, 132, 0.5)",
                    "borderColor": "rgba(255, 99, 132, 1)",
                    "borderWidth": 2,
                },
                {
                    "label": "P95",
                    "data": [r.latency_p95 for r in results],
                    "backgroundColor": "rgba(75, 192, 192, 0.5)",
                    "borderColor": "rgba(75, 192, 192, 1)",
                    "borderWidth": 2,
                },
                {
                    "label": "P99",
                    "data": [r.latency_p99 for r in results],
                    "backgroundColor": "rgba(255, 206, 86, 0.5)",
                    "borderColor": "rgba(255, 206, 86, 1)",
                    "borderWidth": 2,
                },
            ],
            options={
                "scales": {
                    "y": {
                        "beginAtZero": True,
                        "title": {
                            "display": True,
                            "text": "Latency (seconds)"
                        }
                    }
                },
                "plugins": {
                    "legend": {
                        "display": True,
                        "position": "top"
                    }
                }
            }
        )
    
    def throughput_chart(
        self,
        results: List[BenchmarkResult],
        model_names: List[str],
        chart_type: Optional[ChartType] = None,
    ) -> ChartData:
        """
        Generate throughput comparison chart data.
        
        Args:
            results: List of benchmark results
            model_names: List of model names
            chart_type: Optional chart type override
        
        Returns:
            ChartData for throughput comparison
        """
        chart_type = chart_type or self.default_chart_type
        
        return ChartData(
            chart_type=chart_type,
            title="Throughput Comparison",
            labels=model_names,
            datasets=[{
                "label": "Throughput (tokens/s)",
                "data": [r.throughput for r in results],
                "backgroundColor": "rgba(153, 102, 255, 0.5)",
                "borderColor": "rgba(153, 102, 255, 1)",
                "borderWidth": 2,
            }],
            options={
                "scales": {
                    "y": {
                        "beginAtZero": True,
                        "title": {
                            "display": True,
                            "text": "Tokens per second"
                        }
                    }
                },
                "plugins": {
                    "legend": {
                        "display": True,
                        "position": "top"
                    }
                }
            }
        )
    
    def radar_chart(
        self,
        results: List[BenchmarkResult],
        model_names: List[str],
    ) -> ChartData:
        """
        Generate radar chart for multi-metric comparison.
        
        Args:
            results: List of benchmark results
            model_names: List of model names
        
        Returns:
            ChartData for radar chart
        """
        datasets = []
        colors = [
            ("rgba(255, 99, 132, 0.5)", "rgba(255, 99, 132, 1)"),
            ("rgba(54, 162, 235, 0.5)", "rgba(54, 162, 235, 1)"),
            ("rgba(255, 206, 86, 0.5)", "rgba(255, 206, 86, 1)"),
            ("rgba(75, 192, 192, 0.5)", "rgba(75, 192, 192, 1)"),
            ("rgba(153, 102, 255, 0.5)", "rgba(153, 102, 255, 1)"),
        ]
        
        for idx, (result, model_name) in enumerate(zip(results, model_names)):
            bg_color, border_color = colors[idx % len(colors)]
            
            # Normalize metrics to 0-1 scale for radar chart
            datasets.append({
                "label": model_name,
                "data": [
                    result.accuracy,  # Already 0-1
                    min(result.throughput / 1000.0, 1.0),  # Normalize
                    1.0 / (result.latency_p50 + 0.001),  # Inverse latency
                ],
                "backgroundColor": bg_color,
                "borderColor": border_color,
                "borderWidth": 2,
            })
        
        return ChartData(
            chart_type=ChartType.RADAR,
            title="Multi-Metric Comparison",
            labels=["Accuracy", "Throughput", "Speed (1/latency)"],
            datasets=datasets,
            options={
                "scales": {
                    "r": {
                        "beginAtZero": True,
                        "max": 1.0,
                        "title": {
                            "display": True,
                            "text": "Normalized Score"
                        }
                    }
                },
                "plugins": {
                    "legend": {
                        "display": True,
                        "position": "top"
                    }
                }
            }
        )
    
    def scatter_chart(
        self,
        results: List[BenchmarkResult],
        model_names: List[str],
        x_metric: str = "latency_p50",
        y_metric: str = "accuracy",
    ) -> ChartData:
        """
        Generate scatter chart for two-metric comparison.
        
        Args:
            results: List of benchmark results
            model_names: List of model names
            x_metric: Metric for X axis
            y_metric: Metric for Y axis
        
        Returns:
            ChartData for scatter chart
        """
        data_points = []
        for result, model_name in zip(results, model_names):
            x_value = getattr(result, x_metric, 0.0)
            y_value = getattr(result, y_metric, 0.0)
            data_points.append({
                "x": x_value,
                "y": y_value,
                "label": model_name,
            })
        
        return ChartData(
            chart_type=ChartType.SCATTER,
            title=f"{y_metric} vs {x_metric}",
            labels=model_names,
            datasets=[{
                "label": "Models",
                "data": data_points,
                "backgroundColor": "rgba(54, 162, 235, 0.5)",
                "borderColor": "rgba(54, 162, 235, 1)",
            }],
            options={
                "scales": {
                    "x": {
                        "title": {
                            "display": True,
                            "text": x_metric.replace("_", " ").title()
                        }
                    },
                    "y": {
                        "title": {
                            "display": True,
                            "text": y_metric.replace("_", " ").title()
                        }
                    }
                },
                "plugins": {
                    "legend": {
                        "display": True,
                        "position": "top"
                    },
                    "tooltip": {
                        "callbacks": {
                            "label": "function(context) { return context.raw.label + ': (' + context.raw.x + ', ' + context.raw.y + ')'; }"
                        }
                    }
                }
            }
        )
    
    def heatmap_chart(
        self,
        results: List[BenchmarkResult],
        model_names: List[str],
        benchmark_names: Optional[List[str]] = None,
    ) -> ChartData:
        """
        Generate heatmap for multi-model, multi-benchmark comparison.
        
        Args:
            results: List of benchmark results
            model_names: List of model names
            benchmark_names: Optional list of benchmark names
        
        Returns:
            ChartData for heatmap
        """
        # Group results by benchmark
        if benchmark_names is None:
            benchmark_names = list(set(r.benchmark_name for r in results if hasattr(r, 'benchmark_name')))
        
        # Create matrix: models x benchmarks
        matrix = []
        for model_name in model_names:
            row = []
            for benchmark_name in benchmark_names:
                # Find result for this model and benchmark
                result = next(
                    (r for r in results if 
                     (hasattr(r, 'model_name') and r.model_name == model_name) and
                     (hasattr(r, 'benchmark_name') and r.benchmark_name == benchmark_name)),
                    None
                )
                if result:
                    row.append(result.accuracy)
                else:
                    row.append(0.0)
            matrix.append(row)
        
        return ChartData(
            chart_type=ChartType.HEATMAP,
            title="Model Performance Heatmap",
            labels=benchmark_names,
            datasets=[{
                "label": model_names[i],
                "data": matrix[i],
            } for i in range(len(model_names))],
            options={
                "scales": {
                    "x": {
                        "title": {
                            "display": True,
                            "text": "Benchmarks"
                        }
                    },
                    "y": {
                        "title": {
                            "display": True,
                            "text": "Models"
                        }
                    }
                }
            }
        )
    
    def export_chart_data(
        self,
        chart_data: ChartData,
        output_path: str,
        format: str = "json",
    ) -> None:
        """
        Export chart data to file.
        
        Args:
            chart_data: Chart data to export
            output_path: Output file path
            format: Export format (json, csv)
        
        Raises:
            ValueError: If format is unsupported
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(chart_data.to_dict(), f, indent=2, ensure_ascii=False)
        elif format == "csv":
            import csv
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Model"] + chart_data.labels)
                for dataset in chart_data.datasets:
                    label = dataset.get("label", "Unknown")
                    data = dataset.get("data", [])
                    if isinstance(data[0], dict):  # Scatter chart
                        row = [label] + [f"({d['x']}, {d['y']})" for d in data]
                    else:
                        row = [label] + data
                    writer.writerow(row)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Chart data exported to {path}")


def prepare_dashboard_data(
    all_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Prepare comprehensive data for dashboard visualization.
    
    Args:
        all_results: List of result dictionaries with 'model' and 'result' keys
    
    Returns:
        Dashboard data dictionary with all chart types
    """
    generator = VisualizationGenerator()
    
    # Extract results and model names
    results = [r.get("result") for r in all_results if r.get("result")]
    model_names = [r.get("model") for r in all_results if r.get("model")]
    
    if not results or not model_names:
        return {
            "charts": {},
            "summary": {
                "total_models": 0,
                "total_benchmarks": 0,
            }
        }
    
    # Generate all chart types
    dashboard_data = {
        "charts": {
            "accuracy": generator.accuracy_chart(results, model_names).to_dict(),
            "latency": generator.latency_chart(results, model_names).to_dict(),
            "throughput": generator.throughput_chart(results, model_names).to_dict(),
            "radar": generator.radar_chart(results, model_names).to_dict(),
            "scatter": generator.scatter_chart(results, model_names).to_dict(),
        },
        "summary": {
            "total_models": len(model_names),
            "total_benchmarks": len(results),
            "best_accuracy": max(r.accuracy for r in results) if results else 0.0,
            "best_throughput": max(r.throughput for r in results) if results else 0.0,
            "avg_accuracy": statistics.mean([r.accuracy for r in results]) if results else 0.0,
            "avg_throughput": statistics.mean([r.throughput for r in results]) if results else 0.0,
        }
    }
    
    return dashboard_data


__all__ = [
    "ChartType",
    "ChartData",
    "VisualizationGenerator",
    "prepare_dashboard_data",
]
