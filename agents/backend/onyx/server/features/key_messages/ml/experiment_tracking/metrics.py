from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import time
import statistics
from typing import Dict, Any, Optional, List, Union, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, field
import structlog
import numpy as np
import torch
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Metrics Tracking System for Key Messages ML Pipeline
Provides comprehensive metric logging and aggregation capabilities
"""


logger = structlog.get_logger(__name__)

@dataclass
class MetricValue:
    """Represents a single metric value."""
    value: float
    step: int
    timestamp: float
    
    def __post_init__(self) -> Any:
        if self.timestamp is None:
            self.timestamp = time.time()

@dataclass
class MetricAggregator:
    """Aggregates metric values over time."""
    window_size: int = 100
    values: deque = field(default_factory=deque)
    
    def add_value(self, value: float, step: int):
        """Add a new value to the aggregator."""
        metric_value = MetricValue(value=value, step=step, timestamp=time.time())
        self.values.append(metric_value)
        
        # Remove old values if window is exceeded
        while len(self.values) > self.window_size:
            self.values.popleft()
    
    def get_mean(self) -> Optional[float]:
        """Get mean of values in window."""
        if not self.values:
            return None
        return statistics.mean(v.value for v in self.values)
    
    def get_median(self) -> Optional[float]:
        """Get median of values in window."""
        if not self.values:
            return None
        return statistics.median(v.value for v in self.values)
    
    def get_std(self) -> Optional[float]:
        """Get standard deviation of values in window."""
        if len(self.values) < 2:
            return None
        return statistics.stdev(v.value for v in self.values)
    
    def get_min(self) -> Optional[float]:
        """Get minimum value in window."""
        if not self.values:
            return None
        return min(v.value for v in self.values)
    
    def get_max(self) -> Optional[float]:
        """Get maximum value in window."""
        if not self.values:
            return None
        return max(v.value for v in self.values)
    
    def get_latest(self) -> Optional[float]:
        """Get the most recent value."""
        if not self.values:
            return None
        return self.values[-1].value
    
    def get_summary(self) -> Dict[str, Optional[float]]:
        """Get summary statistics."""
        return {
            'mean': self.get_mean(),
            'median': self.get_median(),
            'std': self.get_std(),
            'min': self.get_min(),
            'max': self.get_max(),
            'latest': self.get_latest(),
            'count': len(self.values)
        }
    
    def clear(self) -> Any:
        """Clear all values."""
        self.values.clear()

class MetricLogger:
    """Logs metrics with different aggregation strategies."""
    
    def __init__(self, log_frequency: int = 1):
        
    """__init__ function."""
self.log_frequency = log_frequency
        self.metrics: Dict[str, MetricAggregator] = defaultdict(lambda: MetricAggregator())
        self.last_log_step = -1
        
    def log_metric(self, name: str, value: float, step: int):
        """Log a single metric."""
        self.metrics[name].add_value(value, step)
        
        # Log if frequency is met
        if step - self.last_log_step >= self.log_frequency:
            self._log_aggregated_metrics(step)
            self.last_log_step = step
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log multiple metrics."""
        for name, value in metrics.items():
            self.metrics[name].add_value(value, step)
        
        # Log if frequency is met
        if step - self.last_log_step >= self.log_frequency:
            self._log_aggregated_metrics(step)
            self.last_log_step = step
    
    def _log_aggregated_metrics(self, step: int):
        """Log aggregated metrics."""
        aggregated = {}
        for name, aggregator in self.metrics.items():
            summary = aggregator.get_summary()
            aggregated[f"{name}_mean"] = summary['mean']
            aggregated[f"{name}_std"] = summary['std']
            aggregated[f"{name}_min"] = summary['min']
            aggregated[f"{name}_max"] = summary['max']
            aggregated[f"{name}_latest"] = summary['latest']
        
        logger.info("Metrics logged", step=step, metrics=aggregated)
    
    def get_metric_summary(self, name: str) -> Dict[str, Optional[float]]:
        """Get summary for a specific metric."""
        if name not in self.metrics:
            return {}
        return self.metrics[name].get_summary()
    
    def get_all_summaries(self) -> Dict[str, Dict[str, Optional[float]]]:
        """Get summaries for all metrics."""
        return {name: aggregator.get_summary() for name, aggregator in self.metrics.items()}
    
    def clear_metrics(self) -> Any:
        """Clear all metrics."""
        for aggregator in self.metrics.values():
            aggregator.clear()

class MetricsTracker:
    """Comprehensive metrics tracking system."""
    
    def __init__(self, log_frequency: int = 1, window_size: int = 100):
        
    """__init__ function."""
self.log_frequency = log_frequency
        self.window_size = window_size
        self.metric_logger = MetricLogger(log_frequency)
        self.metrics: Dict[str, MetricAggregator] = defaultdict(lambda: MetricAggregator(window_size))
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.text_logs: Dict[str, List[str]] = defaultdict(list)
        self.image_logs: Dict[str, List[Any]] = defaultdict(list)
        self.custom_metrics: Dict[str, Callable] = {}
        self.step = 0
        
        logger.info("MetricsTracker initialized", 
                   log_frequency=log_frequency,
                   window_size=window_size)
    
    def log_scalar(self, name: str, value: float, step: Optional[int] = None):
        """Log a scalar metric."""
        if step is None:
            step = self.step
        
        self.metrics[name].add_value(value, step)
        self.metric_logger.log_metric(name, value, step)
        
        logger.debug("Scalar metric logged", name=name, value=value, step=step)
    
    def log_scalars(self, scalars: Dict[str, float], step: Optional[int] = None):
        """Log multiple scalar metrics."""
        if step is None:
            step = self.step
        
        for name, value in scalars.items():
            self.metrics[name].add_value(value, step)
        
        self.metric_logger.log_metrics(scalars, step)
        
        logger.debug("Scalar metrics logged", metrics=scalars, step=step)
    
    def log_histogram(self, name: str, values: Union[List[float], torch.Tensor], step: Optional[int] = None):
        """Log histogram data."""
        if step is None:
            step = self.step
        
        # Convert tensor to list if necessary
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy().flatten().tolist()
        
        # Store histogram data
        self.histograms[name].extend(values)
        
        # Keep only recent values
        if len(self.histograms[name]) > self.window_size * 10:
            self.histograms[name] = self.histograms[name][-self.window_size * 10:]
        
        # Also log summary statistics
        if values:
            self.log_scalars({
                f"{name}_mean": np.mean(values),
                f"{name}_std": np.std(values),
                f"{name}_min": np.min(values),
                f"{name}_max": np.max(values),
                f"{name}_count": len(values)
            }, step)
        
        logger.debug("Histogram logged", name=name, count=len(values), step=step)
    
    def log_text(self, name: str, text: str, step: Optional[int] = None):
        """Log text data."""
        if step is None:
            step = self.step
        
        # Store text log
        self.text_logs[name].append(f"Step {step}: {text}")
        
        # Keep only recent logs
        if len(self.text_logs[name]) > 100:
            self.text_logs[name] = self.text_logs[name][-100:]
        
        logger.debug("Text logged", name=name, text=text[:100], step=step)
    
    def log_image(self, name: str, image_data: Any, step: Optional[int] = None):
        """Log image data."""
        if step is None:
            step = self.step
        
        # Store image log (just metadata for now)
        self.image_logs[name].append({
            'step': step,
            'timestamp': time.time(),
            'shape': getattr(image_data, 'shape', None),
            'dtype': getattr(image_data, 'dtype', None)
        })
        
        # Keep only recent logs
        if len(self.image_logs[name]) > 50:
            self.image_logs[name] = self.image_logs[name][-50:]
        
        logger.debug("Image logged", name=name, step=step)
    
    def log_custom_metric(self, name: str, metric_func: Callable, *args, **kwargs):
        """Log a custom metric computed by a function."""
        try:
            value = metric_func(*args, **kwargs)
            if isinstance(value, (int, float)):
                self.log_scalar(name, value)
            else:
                logger.warning(f"Custom metric {name} returned non-scalar value: {value}")
        except Exception as e:
            logger.error(f"Failed to compute custom metric {name}", error=str(e))
    
    def register_custom_metric(self, name: str, metric_func: Callable):
        """Register a custom metric function."""
        self.custom_metrics[name] = metric_func
        logger.info("Custom metric registered", name=name)
    
    def get_metric(self, name: str) -> Optional[MetricAggregator]:
        """Get metric aggregator by name."""
        return self.metrics.get(name)
    
    def get_average(self, name: str, window: Optional[int] = None) -> Optional[float]:
        """Get average of a metric over a window."""
        aggregator = self.metrics.get(name)
        if aggregator is None:
            return None
        
        if window is not None:
            # Create temporary aggregator with custom window
            temp_aggregator = MetricAggregator(window_size=window)
            for value in list(aggregator.values)[-window:]:
                temp_aggregator.add_value(value.value, value.step)
            return temp_aggregator.get_mean()
        
        return aggregator.get_mean()
    
    def get_latest(self, name: str) -> Optional[float]:
        """Get the latest value of a metric."""
        aggregator = self.metrics.get(name)
        if aggregator is None:
            return None
        return aggregator.get_latest()
    
    def get_summary(self, name: str) -> Dict[str, Optional[float]]:
        """Get summary statistics for a metric."""
        aggregator = self.metrics.get(name)
        if aggregator is None:
            return {}
        return aggregator.get_summary()
    
    def get_all_summaries(self) -> Dict[str, Dict[str, Optional[float]]]:
        """Get summaries for all metrics."""
        return {name: aggregator.get_summary() for name, aggregator in self.metrics.items()}
    
    def get_histogram_data(self, name: str) -> List[float]:
        """Get histogram data for a metric."""
        return self.histograms.get(name, [])
    
    def get_text_logs(self, name: str) -> List[str]:
        """Get text logs for a metric."""
        return self.text_logs.get(name, [])
    
    def get_image_logs(self, name: str) -> List[Dict[str, Any]]:
        """Get image logs for a metric."""
        return self.image_logs.get(name, [])
    
    def step_metrics(self) -> Any:
        """Increment the step counter."""
        self.step += 1
    
    def set_step(self, step: int):
        """Set the current step."""
        self.step = step
    
    def clear_metrics(self) -> Any:
        """Clear all metrics."""
        for aggregator in self.metrics.values():
            aggregator.clear()
        self.histograms.clear()
        self.text_logs.clear()
        self.image_logs.clear()
        self.metric_logger.clear_metrics()
        
        logger.info("All metrics cleared")
    
    def export_metrics(self) -> Dict[str, Any]:
        """Export all metrics data."""
        export_data = {
            'step': self.step,
            'timestamp': time.time(),
            'metrics': {},
            'histograms': {},
            'text_logs': {},
            'image_logs': {}
        }
        
        # Export scalar metrics
        for name, aggregator in self.metrics.items():
            export_data['metrics'][name] = {
                'values': [(v.value, v.step, v.timestamp) for v in aggregator.values],
                'summary': aggregator.get_summary()
            }
        
        # Export histograms
        for name, values in self.histograms.items():
            export_data['histograms'][name] = values
        
        # Export text logs
        for name, logs in self.text_logs.items():
            export_data['text_logs'][name] = logs
        
        # Export image logs
        for name, logs in self.image_logs.items():
            export_data['image_logs'][name] = logs
        
        return export_data
    
    def import_metrics(self, data: Dict[str, Any]):
        """Import metrics data."""
        self.step = data.get('step', 0)
        
        # Import scalar metrics
        for name, metric_data in data.get('metrics', {}).items():
            aggregator = MetricAggregator(window_size=self.window_size)
            for value, step, timestamp in metric_data.get('values', []):
                metric_value = MetricValue(value=value, step=step, timestamp=timestamp)
                aggregator.values.append(metric_value)
            self.metrics[name] = aggregator
        
        # Import histograms
        for name, values in data.get('histograms', {}).items():
            self.histograms[name] = values
        
        # Import text logs
        for name, logs in data.get('text_logs', {}).items():
            self.text_logs[name] = logs
        
        # Import image logs
        for name, logs in data.get('image_logs', {}).items():
            self.image_logs[name] = logs
        
        logger.info("Metrics imported", step=self.step)

class TrainingMetricsTracker(MetricsTracker):
    """Specialized metrics tracker for training."""
    
    def __init__(self, log_frequency: int = 1, window_size: int = 100):
        
    """__init__ function."""
super().__init__(log_frequency, window_size)
        self.epoch = 0
        self.best_metrics: Dict[str, float] = {}
        
        # Register common training metrics
        self._register_training_metrics()
    
    def _register_training_metrics(self) -> Any:
        """Register common training metrics."""
        # Learning rate tracking
        self.register_custom_metric("learning_rate", lambda optimizer: optimizer.param_groups[0]['lr'])
        
        # Gradient norm tracking
        def compute_grad_norm(model) -> Any:
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            return total_norm ** (1. / 2)
        
        self.register_custom_metric("grad_norm", compute_grad_norm)
    
    def log_training_step(self, loss: float, optimizer: Optional[torch.optim.Optimizer] = None,
                         model: Optional[torch.nn.Module] = None, **kwargs):
        """Log metrics for a training step."""
        self.log_scalar("train/loss", loss)
        
        # Log learning rate if optimizer is provided
        if optimizer is not None:
            lr = optimizer.param_groups[0]['lr']
            self.log_scalar("train/learning_rate", lr)
        
        # Log gradient norm if model is provided
        if model is not None:
            grad_norm = self._compute_grad_norm(model)
            if grad_norm is not None:
                self.log_scalar("train/grad_norm", grad_norm)
        
        # Log additional metrics
        for name, value in kwargs.items():
            if isinstance(value, (int, float)):
                self.log_scalar(f"train/{name}", value)
        
        self.step_metrics()
    
    def log_validation_step(self, loss: float, **kwargs):
        """Log metrics for a validation step."""
        self.log_scalar("val/loss", loss)
        
        # Log additional metrics
        for name, value in kwargs.items():
            if isinstance(value, (int, float)):
                self.log_scalar(f"val/{name}", value)
    
    def log_epoch(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Log metrics for an epoch."""
        self.epoch += 1
        
        # Log training metrics
        for name, value in train_metrics.items():
            self.log_scalar(f"epoch/train_{name}", value)
        
        # Log validation metrics
        for name, value in val_metrics.items():
            self.log_scalar(f"epoch/val_{name}", value)
            
            # Track best metrics
            if name not in self.best_metrics or value < self.best_metrics[name]:
                self.best_metrics[name] = value
                self.log_scalar(f"best/val_{name}", value)
        
        logger.info("Epoch completed", 
                   epoch=self.epoch,
                   train_metrics=train_metrics,
                   val_metrics=val_metrics)
    
    def _compute_grad_norm(self, model: torch.nn.Module) -> Optional[float]:
        """Compute gradient norm for a model."""
        try:
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            return total_norm ** (1. / 2)
        except Exception:
            return None
    
    def get_best_metrics(self) -> Dict[str, float]:
        """Get the best metrics achieved."""
        return self.best_metrics.copy()
    
    def reset_epoch(self) -> Any:
        """Reset epoch counter."""
        self.epoch = 0
        self.best_metrics.clear() 