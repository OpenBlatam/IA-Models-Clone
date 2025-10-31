from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from pathlib import Path
import json
import pickle
import logging
import time
from dataclasses import dataclass, asdict
import yaml
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd
import psutil
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Deep Learning Framework Utilities
Utility functions and helper classes for the deep learning framework.
"""



@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    train_loss: float
    val_loss: float
    train_metric: float
    val_metric: float
    epoch: int
    step: int
    learning_rate: float
    timestamp: float = None
    
    def __post_init__(self) -> Any:
        if self.timestamp is None:
            self.timestamp = time.time()


class MetricsTracker:
    """Track and manage training metrics."""
    
    def __init__(self, save_path: str = "metrics"):
        
    """__init__ function."""
self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.metrics: List[TrainingMetrics] = []
        self.best_metric = float('inf')
        self.best_epoch = 0
    
    def add_metrics(self, metrics: TrainingMetrics):
        """Add new metrics."""
        self.metrics.append(metrics)
        
        # Update best metric
        if metrics.val_metric < self.best_metric:
            self.best_metric = metrics.val_metric
            self.best_epoch = metrics.epoch
    
    def get_latest_metrics(self) -> Optional[TrainingMetrics]:
        """Get the latest metrics."""
        return self.metrics[-1] if self.metrics else None
    
    def get_best_metrics(self) -> Optional[TrainingMetrics]:
        """Get the best metrics."""
        if not self.metrics:
            return None
        
        best_idx = np.argmin([m.val_metric for m in self.metrics])
        return self.metrics[best_idx]
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training curves."""
        if not self.metrics:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = [m.epoch for m in self.metrics]
        train_losses = [m.train_loss for m in self.metrics]
        val_losses = [m.val_loss for m in self.metrics]
        train_metrics = [m.train_metric for m in self.metrics]
        val_metrics = [m.val_metric for m in self.metrics]
        
        # Loss curves
        axes[0, 0].plot(epochs, train_losses, label='Train Loss', color='blue')
        axes[0, 0].plot(epochs, val_losses, label='Val Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Metric curves
        axes[0, 1].plot(epochs, train_metrics, label='Train Metric', color='blue')
        axes[0, 1].plot(epochs, val_metrics, label='Val Metric', color='red')
        axes[0, 1].set_title('Training and Validation Metric')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Metric')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate curve
        lrs = [m.learning_rate for m in self.metrics]
        axes[1, 0].plot(epochs, lrs, color='green')
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True)
        
        # Loss difference
        loss_diff = [abs(t - v) for t, v in zip(train_losses, val_losses)]
        axes[1, 1].plot(epochs, loss_diff, color='orange')
        axes[1, 1].set_title('Train-Val Loss Difference')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('|Train Loss - Val Loss|')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.save_path / "training_curves.png", dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def save_metrics(self, filename: str = "metrics.json"):
        """Save metrics to JSON file."""
        metrics_dict = [asdict(m) for m in self.metrics]
        
        with open(self.save_path / filename, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(metrics_dict, f, indent=2)
    
    def load_metrics(self, filename: str = "metrics.json"):
        """Load metrics from JSON file."""
        filepath = self.save_path / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                metrics_dict = json.load(f)
            
            self.metrics = [TrainingMetrics(**m) for m in metrics_dict]
            
            if self.metrics:
                self.best_metric = min(m.val_metric for m in self.metrics)
                best_idx = np.argmin([m.val_metric for m in self.metrics])
                self.best_epoch = self.metrics[best_idx].epoch


class ModelAnalyzer:
    """Analyze model performance and characteristics."""
    
    def __init__(self, model: nn.Module, device: torch.device):
        
    """__init__ function."""
self.model = model
        self.device = device
        self.model.to(device)
    
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        total_params = 0
        trainable_params = 0
        
        for param in self.model.parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params
        }
    
    def get_model_size_mb(self) -> float:
        """Get model size in MB."""
        param_size = 0
        buffer_size = 0
        
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    def analyze_layer_distribution(self) -> Dict[str, Any]:
        """Analyze layer distribution in the model."""
        layer_info = {}
        
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                layer_type = type(module).__name__
                if layer_type not in layer_info:
                    layer_info[layer_type] = {'count': 0, 'parameters': 0}
                
                layer_info[layer_type]['count'] += 1
                layer_info[layer_type]['parameters'] += sum(p.numel() for p in module.parameters())
        
        return layer_info
    
    def measure_inference_time(self, input_shape: Tuple[int, ...], num_runs: int = 100) -> Dict[str, float]:
        """Measure model inference time."""
        self.model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(dummy_input)
        
        # Measure inference time
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = self.model(dummy_input)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                times.append(end_time - start_time)
        
        return {
            'mean_inference_time_ms': np.mean(times) * 1000,
            'std_inference_time_ms': np.std(times) * 1000,
            'min_inference_time_ms': np.min(times) * 1000,
            'max_inference_time_ms': np.max(times) * 1000
        }


class DataAnalyzer:
    """Analyze dataset characteristics."""
    
    def __init__(self, dataset) -> Any:
        self.dataset = dataset
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get basic dataset statistics."""
        dataset_size = len(self.dataset)
        
        # Sample a few items to analyze structure
        sample_items = []
        for i in range(min(10, dataset_size)):
            try:
                sample_items.append(self.dataset[i])
            except Exception as e:
                print(f"Error sampling item {i}: {e}")
                break
        
        stats = {
            'dataset_size': dataset_size,
            'sample_items': len(sample_items)
        }
        
        if sample_items:
            # Analyze first sample
            first_sample = sample_items[0]
            if isinstance(first_sample, (tuple, list)):
                stats['num_outputs'] = len(first_sample)
                stats['output_types'] = [type(item).__name__ for item in first_sample]
                
                # Analyze tensor shapes if present
                tensor_shapes = []
                for item in first_sample:
                    if isinstance(item, torch.Tensor):
                        tensor_shapes.append(list(item.shape))
                    elif isinstance(item, np.ndarray):
                        tensor_shapes.append(list(item.shape))
                
                if tensor_shapes:
                    stats['tensor_shapes'] = tensor_shapes
            
            elif isinstance(first_sample, dict):
                stats['output_keys'] = list(first_sample.keys())
                stats['output_types'] = {k: type(v).__name__ for k, v in first_sample.items()}
                
                # Analyze tensor shapes in dict
                tensor_shapes = {}
                for k, v in first_sample.items():
                    if isinstance(v, torch.Tensor):
                        tensor_shapes[k] = list(v.shape)
                    elif isinstance(v, np.ndarray):
                        tensor_shapes[k] = list(v.shape)
                
                if tensor_shapes:
                    stats['tensor_shapes'] = tensor_shapes
        
        return stats
    
    def analyze_class_distribution(self, label_key: str = 'labels') -> Dict[str, Any]:
        """Analyze class distribution in the dataset."""
        class_counts = {}
        total_samples = 0
        
        for i in range(min(1000, len(self.dataset))):  # Sample up to 1000 items
            try:
                item = self.dataset[i]
                
                if isinstance(item, dict) and label_key in item:
                    label = item[label_key]
                elif isinstance(item, (tuple, list)) and len(item) > 1:
                    label = item[1]  # Assume second element is label
                else:
                    continue
                
                if isinstance(label, torch.Tensor):
                    label = label.item()
                elif isinstance(label, np.ndarray):
                    label = label.item()
                
                class_counts[label] = class_counts.get(label, 0) + 1
                total_samples += 1
                
            except Exception as e:
                continue
        
        if class_counts:
            class_distribution = {k: v / total_samples for k, v in class_counts.items()}
            return {
                'class_counts': class_counts,
                'class_distribution': class_distribution,
                'num_classes': len(class_counts),
                'total_samples_analyzed': total_samples
            }
        
        return {}


class PerformanceMonitor:
    """Monitor system performance during training."""
    
    def __init__(self) -> Any:
        self.memory_usage = []
        self.gpu_memory_usage = []
        self.timestamps = []
    
    def record_metrics(self) -> Any:
        """Record current system metrics."""
        
        # CPU and memory usage
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        # GPU memory usage
        gpu_memory = 0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        self.memory_usage.append({
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used_gb': memory.used / 1024 / 1024 / 1024,
            'memory_available_gb': memory.available / 1024 / 1024 / 1024
        })
        
        self.gpu_memory_usage.append(gpu_memory)
        self.timestamps.append(time.time())
    
    def plot_performance(self, save_path: Optional[str] = None):
        """Plot performance metrics."""
        if not self.memory_usage:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        timestamps = np.array(self.timestamps) - self.timestamps[0]
        
        # CPU usage
        cpu_usage = [m['cpu_percent'] for m in self.memory_usage]
        axes[0, 0].plot(timestamps, cpu_usage, color='blue')
        axes[0, 0].set_title('CPU Usage')
        axes[0, 0].set_xlabel('Time (seconds)')
        axes[0, 0].set_ylabel('CPU %')
        axes[0, 0].grid(True)
        
        # Memory usage
        memory_usage = [m['memory_percent'] for m in self.memory_usage]
        axes[0, 1].plot(timestamps, memory_usage, color='red')
        axes[0, 1].set_title('Memory Usage')
        axes[0, 1].set_xlabel('Time (seconds)')
        axes[0, 1].set_ylabel('Memory %')
        axes[0, 1].grid(True)
        
        # GPU memory usage
        if self.gpu_memory_usage and any(m > 0 for m in self.gpu_memory_usage):
            axes[1, 0].plot(timestamps, self.gpu_memory_usage, color='green')
            axes[1, 0].set_title('GPU Memory Usage')
            axes[1, 0].set_xlabel('Time (seconds)')
            axes[1, 0].set_ylabel('GPU Memory (MB)')
            axes[1, 0].grid(True)
        
        # Memory used
        memory_used = [m['memory_used_gb'] for m in self.memory_usage]
        axes[1, 1].plot(timestamps, memory_used, color='orange')
        axes[1, 1].set_title('Memory Used')
        axes[1, 1].set_xlabel('Time (seconds)')
        axes[1, 1].set_ylabel('Memory (GB)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig("performance_monitor.png", dpi=300, bbox_inches='tight')
        
        plt.close()


class ExperimentLogger:
    """Comprehensive experiment logging."""
    
    def __init__(self, experiment_name: str, log_dir: str = "experiments"):
        
    """__init__ function."""
self.experiment_name = experiment_name
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / "experiment.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(experiment_name)
        
        # Initialize components
        self.metrics_tracker = MetricsTracker(str(self.log_dir))
        self.performance_monitor = PerformanceMonitor()
    
    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration."""
        config_path = self.log_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        self.logger.info(f"Configuration saved to {config_path}")
    
    def log_model_info(self, model: nn.Module, device: torch.device):
        """Log model information."""
        analyzer = ModelAnalyzer(model, device)
        
        # Parameter count
        param_info = analyzer.count_parameters()
        self.logger.info(f"Model parameters: {param_info}")
        
        # Model size
        model_size = analyzer.get_model_size_mb()
        self.logger.info(f"Model size: {model_size:.2f} MB")
        
        # Layer distribution
        layer_info = analyzer.analyze_layer_distribution()
        self.logger.info(f"Layer distribution: {layer_info}")
        
        # Save model info
        model_info = {
            'parameters': param_info,
            'model_size_mb': model_size,
            'layer_distribution': layer_info
        }
        
        with open(self.log_dir / "model_info.json", 'w') as f:
            json.dump(model_info, f, indent=2)
    
    def log_dataset_info(self, dataset) -> Any:
        """Log dataset information."""
        analyzer = DataAnalyzer(dataset)
        
        # Dataset stats
        stats = analyzer.get_dataset_stats()
        self.logger.info(f"Dataset stats: {stats}")
        
        # Class distribution
        class_info = analyzer.analyze_class_distribution()
        if class_info:
            self.logger.info(f"Class distribution: {class_info}")
        
        # Save dataset info
        dataset_info = {
            'stats': stats,
            'class_distribution': class_info
        }
        
        with open(self.log_dir / "dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)
    
    def log_training_step(self, step: int, epoch: int, train_loss: float, 
                         val_loss: float, train_metric: float, val_metric: float,
                         learning_rate: float):
        """Log training step metrics."""
        metrics = TrainingMetrics(
            train_loss=train_loss,
            val_loss=val_loss,
            train_metric=train_metric,
            val_metric=val_metric,
            epoch=epoch,
            step=step,
            learning_rate=learning_rate
        )
        
        self.metrics_tracker.add_metrics(metrics)
        self.performance_monitor.record_metrics()
        
        # Log to console
        self.logger.info(
            f"Epoch {epoch}, Step {step} - "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"Train Metric: {train_metric:.4f}, Val Metric: {val_metric:.4f}"
        )
    
    def save_experiment_summary(self) -> Any:
        """Save comprehensive experiment summary."""
        summary = {
            'experiment_name': self.experiment_name,
            'best_metrics': self.metrics_tracker.get_best_metrics(),
            'total_steps': len(self.metrics_tracker.metrics),
            'final_metrics': self.metrics_tracker.get_latest_metrics()
        }
        
        # Save summary
        with open(self.log_dir / "experiment_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Generate plots
        self.metrics_tracker.plot_training_curves()
        self.performance_monitor.plot_performance()
        
        # Save metrics
        self.metrics_tracker.save_metrics()
        
        self.logger.info(f"Experiment summary saved to {self.log_dir}")


def create_experiment_config(task_type: str, model_name: str, **kwargs) -> Dict[str, Any]:
    """Create experiment configuration."""
    base_config = {
        'task_type': task_type,
        'model_name': model_name,
        'learning_rate': 1e-4,
        'batch_size': 32,
        'num_epochs': 100,
        'gradient_accumulation_steps': 4,
        'mixed_precision': True,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_steps': 1000,
        'eval_steps': 500,
        'logging_steps': 100,
        'weight_decay': 0.01,
        'warmup_steps': 1000,
        'max_grad_norm': 1.0,
        'early_stopping_patience': 10
    }
    
    base_config.update(kwargs)
    return base_config


def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration from file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_experiment_config(config: Dict[str, Any], config_path: str):
    """Save experiment configuration to file."""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


if __name__ == "__main__":
    # Example usage of utilities
    print("Deep Learning Framework Utilities")
    print("=" * 40)
    
    # Create example metrics
    metrics_tracker = MetricsTracker()
    
    # Add some dummy metrics
    for epoch in range(10):
        metrics = TrainingMetrics(
            train_loss=1.0 - epoch * 0.1,
            val_loss=1.2 - epoch * 0.08,
            train_metric=0.5 + epoch * 0.05,
            val_metric=0.4 + epoch * 0.04,
            epoch=epoch,
            step=epoch * 100,
            learning_rate=1e-4 * (0.9 ** epoch)
        )
        metrics_tracker.add_metrics(metrics)
    
    # Plot training curves
    metrics_tracker.plot_training_curves()
    print("Training curves saved to metrics/training_curves.png")
    
    # Save metrics
    metrics_tracker.save_metrics()
    print("Metrics saved to metrics/metrics.json")
    
    print("\nUtility functions ready for use!") 