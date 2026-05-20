from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import numpy as np
import json
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import time
from collections import defaultdict, deque
import statistics
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
            import psutil
            import GPUtil
from typing import Any, List, Dict, Optional
import asyncio
"""
Metrics Tracking System
======================

This module provides comprehensive metrics tracking for AI video generation experiments.

Features:
- Multiple metric types (scalar, vector, image, video)
- Metric aggregation and statistics
- Custom metric definitions
- Metric visualization
- Performance monitoring
- Metric export and analysis
"""


# Optional imports
try:
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class MetricValue:
    """Single metric value with metadata."""
    
    value: float
    step: int
    epoch: int
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "value": self.value,
            "step": self.step,
            "epoch": self.epoch,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


@dataclass
class MetricDefinition:
    """Definition of a metric."""
    
    name: str
    metric_type: str  # "scalar", "vector", "image", "video"
    description: str = ""
    unit: str = ""
    lower_is_better: bool = True
    aggregation_methods: List[str] = field(default_factory=lambda: ["mean", "min", "max"])
    window_size: int = 100  # For moving averages
    
    def __post_init__(self) -> Any:
        """Validate metric definition."""
        valid_types = ["scalar", "vector", "image", "video", "histogram"]
        if self.metric_type not in valid_types:
            raise ValueError(f"Invalid metric type: {self.metric_type}. Must be one of {valid_types}")


class MetricsTracker:
    """Main metrics tracking class."""
    
    def __init__(self, max_history: int = 10000):
        
    """__init__ function."""
self.max_history = max_history
        self.metrics: Dict[str, List[MetricValue]] = defaultdict(list)
        self.metric_definitions: Dict[str, MetricDefinition] = {}
        self.aggregation_cache: Dict[str, Dict[str, float]] = {}
        self.last_update = time.time()
        
        logger.info("Metrics tracker initialized")
    
    def register_metric(self, metric_def: MetricDefinition):
        """Register a new metric."""
        self.metric_definitions[metric_def.name] = metric_def
        logger.info(f"Registered metric: {metric_def.name} ({metric_def.metric_type})")
    
    def log_metric(
        self,
        name: str,
        value: float,
        step: int,
        epoch: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log a metric value."""
        if metadata is None:
            metadata = {}
        
        metric_value = MetricValue(
            value=value,
            step=step,
            epoch=epoch,
            timestamp=datetime.now().isoformat(),
            metadata=metadata
        )
        
        self.metrics[name].append(metric_value)
        
        # Limit history size
        if len(self.metrics[name]) > self.max_history:
            self.metrics[name] = self.metrics[name][-self.max_history:]
        
        # Clear aggregation cache
        if name in self.aggregation_cache:
            del self.aggregation_cache[name]
        
        self.last_update = time.time()
    
    def log_metrics(self, metrics: Dict[str, float], step: int, epoch: int = 0):
        """Log multiple metrics at once."""
        for name, value in metrics.items():
            self.log_metric(name, value, step, epoch)
    
    def get_metric_history(self, name: str) -> List[MetricValue]:
        """Get metric history."""
        return self.metrics.get(name, [])
    
    def get_latest_value(self, name: str) -> Optional[float]:
        """Get the latest value for a metric."""
        history = self.metrics.get(name, [])
        if not history:
            return None
        return history[-1].value
    
    def get_metric_statistics(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric."""
        history = self.metrics.get(name, [])
        if not history:
            return {}
        
        values = [mv.value for mv in history]
        
        stats = {
            "count": len(values),
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "median": np.median(values),
            "latest": values[-1]
        }
        
        # Add percentiles
        for p in [25, 50, 75, 90, 95, 99]:
            stats[f"p{p}"] = np.percentile(values, p)
        
        return stats
    
    def get_moving_average(self, name: str, window_size: Optional[int] = None) -> List[float]:
        """Get moving average for a metric."""
        history = self.metrics.get(name, [])
        if not history:
            return []
        
        if window_size is None:
            metric_def = self.metric_definitions.get(name)
            window_size = metric_def.window_size if metric_def else 100
        
        values = [mv.value for mv in history]
        
        if len(values) < window_size:
            return values
        
        moving_avg = []
        for i in range(len(values)):
            start_idx = max(0, i - window_size + 1)
            window_values = values[start_idx:i + 1]
            moving_avg.append(np.mean(window_values))
        
        return moving_avg
    
    def get_best_value(self, name: str) -> Optional[float]:
        """Get the best value for a metric."""
        metric_def = self.metric_definitions.get(name)
        if not metric_def:
            return None
        
        history = self.metrics.get(name, [])
        if not history:
            return None
        
        values = [mv.value for mv in history]
        
        if metric_def.lower_is_better:
            return min(values)
        else:
            return max(values)
    
    def get_best_step(self, name: str) -> Optional[int]:
        """Get the step with the best value for a metric."""
        metric_def = self.metric_definitions.get(name)
        if not metric_def:
            return None
        
        history = self.metrics.get(name, [])
        if not history:
            return None
        
        if metric_def.lower_is_better:
            best_idx = np.argmin([mv.value for mv in history])
        else:
            best_idx = np.argmax([mv.value for mv in history])
        
        return history[best_idx].step
    
    def get_metric_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all metrics."""
        summary = {}
        
        for name in self.metrics:
            stats = self.get_metric_statistics(name)
            best_value = self.get_best_value(name)
            best_step = self.get_best_step(name)
            
            summary[name] = {
                "statistics": stats,
                "best_value": best_value,
                "best_step": best_step,
                "total_logs": len(self.metrics[name])
            }
        
        return summary
    
    def export_metrics(self, filepath: str, format: str = "json"):
        """Export metrics to file."""
        try:
            if format.lower() == "json":
                # Export as JSON
                export_data = {
                    "metrics": {
                        name: [mv.to_dict() for mv in values]
                        for name, values in self.metrics.items()
                    },
                    "definitions": {
                        name: {
                            "name": def_.name,
                            "type": def_.metric_type,
                            "description": def_.description,
                            "unit": def_.unit,
                            "lower_is_better": def_.lower_is_better
                        }
                        for name, def_ in self.metric_definitions.items()
                    },
                    "summary": self.get_metric_summary(),
                    "export_timestamp": datetime.now().isoformat()
                }
                
                with open(filepath, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    json.dump(export_data, f, indent=2)
            
            elif format.lower() == "csv" and PANDAS_AVAILABLE:
                # Export as CSV
                all_data = []
                for name, values in self.metrics.items():
                    for mv in values:
                        all_data.append({
                            "metric_name": name,
                            "value": mv.value,
                            "step": mv.step,
                            "epoch": mv.epoch,
                            "timestamp": mv.timestamp
                        })
                
                df = pd.DataFrame(all_data)
                df.to_csv(filepath, index=False)
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"Metrics exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
    
    def plot_metric(self, name: str, save_path: Optional[str] = None):
        """Plot a metric."""
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib not available for plotting")
            return
        
        history = self.metrics.get(name, [])
        if not history:
            logger.warning(f"No data for metric: {name}")
            return
        
        steps = [mv.step for mv in history]
        values = [mv.value for mv in history]
        
        plt.figure(figsize=(12, 6))
        
        # Main plot
        plt.subplot(2, 1, 1)
        plt.plot(steps, values, 'b-', alpha=0.7, label='Raw values')
        
        # Moving average
        moving_avg = self.get_moving_average(name)
        if len(moving_avg) == len(steps):
            plt.plot(steps, moving_avg, 'r-', linewidth=2, label='Moving average')
        
        plt.title(f'Metric: {name}')
        plt.xlabel('Step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Statistics subplot
        plt.subplot(2, 1, 2)
        stats = self.get_metric_statistics(name)
        
        # Create bar plot of statistics
        stat_names = ['mean', 'std', 'min', 'max', 'median']
        stat_values = [stats.get(name, 0) for name in stat_names]
        
        plt.bar(stat_names, stat_values, alpha=0.7)
        plt.title('Statistics')
        plt.ylabel('Value')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Metric plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_all_metrics(self, save_dir: str = "metric_plots"):
        """Plot all metrics."""
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib not available for plotting")
            return
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for name in self.metrics:
            plot_path = save_path / f"{name}.png"
            self.plot_metric(name, str(plot_path))
        
        logger.info(f"All metric plots saved to {save_path}")


class VideoMetricsTracker:
    """Specialized tracker for video generation metrics."""
    
    def __init__(self, base_tracker: MetricsTracker):
        
    """__init__ function."""
self.base_tracker = base_tracker
        self._register_video_metrics()
    
    def _register_video_metrics(self) -> Any:
        """Register common video generation metrics."""
        video_metrics = [
            MetricDefinition("psnr", "scalar", "Peak Signal-to-Noise Ratio", "dB", lower_is_better=False),
            MetricDefinition("ssim", "scalar", "Structural Similarity Index", "", lower_is_better=False),
            MetricDefinition("lpips", "scalar", "Learned Perceptual Image Patch Similarity", "", lower_is_better=True),
            MetricDefinition("fid", "scalar", "Fréchet Inception Distance", "", lower_is_better=True),
            MetricDefinition("inception_score", "scalar", "Inception Score", "", lower_is_better=False),
            MetricDefinition("generation_time", "scalar", "Video Generation Time", "seconds", lower_is_better=True),
            MetricDefinition("memory_usage", "scalar", "GPU Memory Usage", "MB", lower_is_better=True),
            MetricDefinition("fps", "scalar", "Frames Per Second", "fps", lower_is_better=False),
            MetricDefinition("video_length", "scalar", "Generated Video Length", "frames", lower_is_better=False),
            MetricDefinition("quality_score", "scalar", "Overall Quality Score", "", lower_is_better=False)
        ]
        
        for metric in video_metrics:
            self.base_tracker.register_metric(metric)
    
    def log_video_metrics(
        self,
        psnr: Optional[float] = None,
        ssim: Optional[float] = None,
        lpips: Optional[float] = None,
        fid: Optional[float] = None,
        inception_score: Optional[float] = None,
        generation_time: Optional[float] = None,
        memory_usage: Optional[float] = None,
        fps: Optional[float] = None,
        video_length: Optional[int] = None,
        quality_score: Optional[float] = None,
        step: int = 0,
        epoch: int = 0
    ):
        """Log video generation metrics."""
        metrics = {}
        
        if psnr is not None:
            metrics["psnr"] = psnr
        if ssim is not None:
            metrics["ssim"] = ssim
        if lpips is not None:
            metrics["lpips"] = lpips
        if fid is not None:
            metrics["fid"] = fid
        if inception_score is not None:
            metrics["inception_score"] = inception_score
        if generation_time is not None:
            metrics["generation_time"] = generation_time
        if memory_usage is not None:
            metrics["memory_usage"] = memory_usage
        if fps is not None:
            metrics["fps"] = fps
        if video_length is not None:
            metrics["video_length"] = video_length
        if quality_score is not None:
            metrics["quality_score"] = quality_score
        
        self.base_tracker.log_metrics(metrics, step, epoch)
    
    def get_video_quality_summary(self) -> Dict[str, Any]:
        """Get summary of video quality metrics."""
        quality_metrics = ["psnr", "ssim", "lpips", "fid", "inception_score", "quality_score"]
        summary = {}
        
        for metric in quality_metrics:
            stats = self.base_tracker.get_metric_statistics(metric)
            if stats:
                summary[metric] = {
                    "latest": stats.get("latest"),
                    "best": self.base_tracker.get_best_value(metric),
                    "mean": stats.get("mean"),
                    "std": stats.get("std")
                }
        
        return summary


class PerformanceMonitor:
    """Monitor system performance during training."""
    
    def __init__(self, metrics_tracker: MetricsTracker):
        
    """__init__ function."""
self.metrics_tracker = metrics_tracker
        self.start_time = time.time()
        self.last_check = time.time()
        
        # Register performance metrics
        self._register_performance_metrics()
    
    def _register_performance_metrics(self) -> Any:
        """Register performance monitoring metrics."""
        perf_metrics = [
            MetricDefinition("gpu_utilization", "scalar", "GPU Utilization", "%", lower_is_better=False),
            MetricDefinition("gpu_memory_used", "scalar", "GPU Memory Used", "MB", lower_is_better=True),
            MetricDefinition("gpu_memory_total", "scalar", "GPU Memory Total", "MB", lower_is_better=False),
            MetricDefinition("cpu_utilization", "scalar", "CPU Utilization", "%", lower_is_better=False),
            MetricDefinition("memory_usage", "scalar", "System Memory Usage", "MB", lower_is_better=True),
            MetricDefinition("training_time", "scalar", "Training Time", "seconds", lower_is_better=True),
            MetricDefinition("throughput", "scalar", "Samples per Second", "samples/s", lower_is_better=False)
        ]
        
        for metric in perf_metrics:
            self.metrics_tracker.register_metric(metric)
    
    def check_performance(self, step: int):
        """Check and log current performance metrics."""
        try:
            
            # System metrics
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            # GPU metrics
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                gpu_util = gpu.load * 100
                gpu_memory_used = gpu.memoryUsed
                gpu_memory_total = gpu.memoryTotal
            else:
                gpu_util = 0
                gpu_memory_used = 0
                gpu_memory_total = 0
            
            # Training time
            current_time = time.time()
            training_time = current_time - self.start_time
            
            # Calculate throughput (samples per second)
            time_diff = current_time - self.last_check
            throughput = 1 / time_diff if time_diff > 0 else 0
            
            # Log metrics
            metrics = {
                "gpu_utilization": gpu_util,
                "gpu_memory_used": gpu_memory_used,
                "gpu_memory_total": gpu_memory_total,
                "cpu_utilization": cpu_percent,
                "memory_usage": memory.used / (1024 * 1024),  # Convert to MB
                "training_time": training_time,
                "throughput": throughput
            }
            
            self.metrics_tracker.log_metrics(metrics, step)
            self.last_check = current_time
            
        except ImportError:
            logger.warning("psutil or GPUtil not available for performance monitoring")
        except Exception as e:
            logger.warning(f"Failed to check performance: {e}")


# Convenience functions
def create_metrics_tracker(max_history: int = 10000) -> MetricsTracker:
    """Create a metrics tracker."""
    return MetricsTracker(max_history)


def create_video_metrics_tracker(base_tracker: Optional[MetricsTracker] = None) -> VideoMetricsTracker:
    """Create a video metrics tracker."""
    if base_tracker is None:
        base_tracker = create_metrics_tracker()
    return VideoMetricsTracker(base_tracker)


def create_performance_monitor(metrics_tracker: MetricsTracker) -> PerformanceMonitor:
    """Create a performance monitor."""
    return PerformanceMonitor(metrics_tracker)


if __name__ == "__main__":
    # Example usage
    print("📊 Metrics Tracking System")
    print("=" * 40)
    
    # Create metrics tracker
    tracker = create_metrics_tracker()
    
    # Register metrics
    tracker.register_metric(MetricDefinition("loss", "scalar", "Training Loss", "", True))
    tracker.register_metric(MetricDefinition("accuracy", "scalar", "Accuracy", "%", False))
    
    # Simulate training
    for step in range(100):
        loss = 1.0 / (step + 1) + np.random.normal(0, 0.01)
        accuracy = 0.5 + 0.4 * (1 - np.exp(-step / 20)) + np.random.normal(0, 0.02)
        
        tracker.log_metrics({
            "loss": loss,
            "accuracy": accuracy
        }, step)
    
    # Get statistics
    loss_stats = tracker.get_metric_statistics("loss")
    print(f"Loss statistics: {loss_stats}")
    
    # Get best values
    best_loss = tracker.get_best_value("loss")
    best_accuracy = tracker.get_best_value("accuracy")
    print(f"Best loss: {best_loss}")
    print(f"Best accuracy: {best_accuracy}")
    
    # Export metrics
    tracker.export_metrics("metrics.json", "json")
    
    # Create video metrics tracker
    video_tracker = create_video_metrics_tracker(tracker)
    
    # Log video metrics
    video_tracker.log_video_metrics(
        psnr=25.5,
        ssim=0.85,
        lpips=0.12,
        generation_time=2.5,
        step=100
    )
    
    print("✅ Metrics tracking example completed!") 