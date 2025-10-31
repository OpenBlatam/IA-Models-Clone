#!/usr/bin/env python3
"""
Performance Monitoring and Metrics System for Frontier Model Training
Provides comprehensive performance tracking, metrics collection, and analysis.
"""

import os
import time
import json
import threading
import queue
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import statistics
import numpy as np
import pandas as pd
import psutil
import torch
import torch.profiler
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import wandb
import mlflow

console = Console()

class MetricType(Enum):
    """Types of metrics to track."""
    TRAINING_LOSS = "training_loss"
    VALIDATION_LOSS = "validation_loss"
    LEARNING_RATE = "learning_rate"
    BATCH_TIME = "batch_time"
    THROUGHPUT = "throughput"
    MEMORY_USAGE = "memory_usage"
    GPU_MEMORY = "gpu_memory"
    GPU_UTILIZATION = "gpu_utilization"
    CPU_USAGE = "cpu_usage"
    CUSTOM = "custom"

class AlertLevel(Enum):
    """Alert levels for performance issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class Metric:
    """Individual metric measurement."""
    name: str
    value: float
    timestamp: datetime
    metric_type: MetricType
    step: Optional[int] = None
    epoch: Optional[int] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceAlert:
    """Performance alert."""
    level: AlertLevel
    message: str
    metric_name: str
    current_value: float
    threshold: float
    timestamp: datetime
    component: str = "performance_monitor"

@dataclass
class TrainingMetrics:
    """Training-specific metrics."""
    step: int
    epoch: int
    training_loss: float
    validation_loss: Optional[float] = None
    learning_rate: Optional[float] = None
    batch_time: Optional[float] = None
    throughput: Optional[float] = None
    gradient_norm: Optional[float] = None
    memory_usage: Optional[float] = None
    gpu_memory_used: Optional[float] = None
    gpu_memory_total: Optional[float] = None
    gpu_utilization: Optional[float] = None
    cpu_usage: Optional[float] = None

@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available: float
    memory_used: float
    disk_usage_percent: float
    disk_free: float
    network_sent: float
    network_recv: float
    gpu_memory_used: float
    gpu_memory_total: float
    gpu_utilization: float
    gpu_temperature: Optional[float] = None
    gpu_power: Optional[float] = None

class MetricsCollector:
    """Collects and manages performance metrics."""
    
    def __init__(self, 
                 log_dir: str = "./metrics",
                 enable_tensorboard: bool = True,
                 enable_wandb: bool = False,
                 enable_mlflow: bool = False,
                 wandb_project: Optional[str] = None,
                 mlflow_experiment: Optional[str] = None):
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_tensorboard = enable_tensorboard
        self.enable_wandb = enable_wandb
        self.enable_mlflow = enable_mlflow
        
        # Initialize logging backends
        self.tensorboard_writer = None
        self.wandb_run = None
        self.mlflow_run = None
        
        if self.enable_tensorboard:
            self.tensorboard_writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))
        
        if self.enable_wandb and wandb_project:
            self.wandb_run = wandb.init(project=wandb_project, resume="allow")
        
        if self.enable_mlflow and mlflow_experiment:
            mlflow.set_experiment(mlflow_experiment)
            self.mlflow_run = mlflow.start_run()
        
        # Metrics storage
        self.metrics: List[Metric] = []
        self.training_metrics: List[TrainingMetrics] = []
        self.system_metrics: List[SystemMetrics] = []
        
        # Alerting
        self.alerts: List[PerformanceAlert] = []
        self.alert_thresholds: Dict[str, Dict[str, float]] = {
            "memory_usage": {"warning": 80.0, "error": 90.0, "critical": 95.0},
            "gpu_memory": {"warning": 80.0, "error": 90.0, "critical": 95.0},
            "gpu_utilization": {"warning": 90.0, "error": 95.0, "critical": 98.0},
            "cpu_usage": {"warning": 80.0, "error": 90.0, "critical": 95.0},
            "batch_time": {"warning": 10.0, "error": 20.0, "critical": 30.0},
            "training_loss": {"warning": 10.0, "error": 20.0, "critical": 50.0}
        }
        
        # Background monitoring
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_active = False
        self.monitoring_interval = 5.0
        
        # Performance profiling
        self.profiler: Optional[torch.profiler.profile] = None
        self.profiling_active = False
        
    def start_monitoring(self, interval: float = 5.0):
        """Start background system monitoring."""
        if self.monitoring_active:
            console.print("[yellow]Monitoring already active[/yellow]")
            return
            
        self.monitoring_interval = interval
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitor_system,
            daemon=True
        )
        self.monitoring_thread.start()
        console.print(f"[green]System monitoring started with {interval}s interval[/green]")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        console.print("[yellow]System monitoring stopped[/yellow]")
    
    def _monitor_system(self):
        """Background system monitoring loop."""
        while self.monitoring_active:
            try:
                metrics = self._collect_system_metrics()
                self.system_metrics.append(metrics)
                
                # Check for alerts
                self._check_system_alerts(metrics)
                
                # Keep only last 1000 metrics
                if len(self.system_metrics) > 1000:
                    self.system_metrics = self.system_metrics[-1000:]
                
                time.sleep(self.monitoring_interval)
            except Exception as e:
                console.print(f"[red]Error in system monitoring: {e}[/red]")
                time.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        # Disk usage
        disk = psutil.disk_usage('/')
        
        # Network
        network = psutil.net_io_counters()
        
        # GPU metrics
        gpu_memory_used = 0.0
        gpu_memory_total = 0.0
        gpu_utilization = 0.0
        gpu_temperature = None
        gpu_power = None
        
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated() / 1024**2  # MB
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**2  # MB
            gpu_utilization = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0.0
            
            # Try to get temperature and power if available
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                gpu_power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Watts
            except ImportError:
                pass
        
        return SystemMetrics(
            timestamp=datetime.now(timezone.utc),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_available=memory.available / 1024**3,  # GB
            memory_used=memory.used / 1024**3,  # GB
            disk_usage_percent=disk.percent,
            disk_free=disk.free / 1024**3,  # GB
            network_sent=network.bytes_sent / 1024**2,  # MB
            network_recv=network.bytes_recv / 1024**2,  # MB
            gpu_memory_used=gpu_memory_used,
            gpu_memory_total=gpu_memory_total,
            gpu_utilization=gpu_utilization,
            gpu_temperature=gpu_temperature,
            gpu_power=gpu_power
        )
    
    def _check_system_alerts(self, metrics: SystemMetrics):
        """Check for system performance alerts."""
        # Memory usage alert
        if metrics.memory_percent >= self.alert_thresholds["memory_usage"]["critical"]:
            self._create_alert(
                AlertLevel.CRITICAL,
                f"Critical memory usage: {metrics.memory_percent:.1f}%",
                "memory_usage",
                metrics.memory_percent,
                self.alert_thresholds["memory_usage"]["critical"]
            )
        elif metrics.memory_percent >= self.alert_thresholds["memory_usage"]["error"]:
            self._create_alert(
                AlertLevel.ERROR,
                f"High memory usage: {metrics.memory_percent:.1f}%",
                "memory_usage",
                metrics.memory_percent,
                self.alert_thresholds["memory_usage"]["error"]
            )
        elif metrics.memory_percent >= self.alert_thresholds["memory_usage"]["warning"]:
            self._create_alert(
                AlertLevel.WARNING,
                f"Memory usage warning: {metrics.memory_percent:.1f}%",
                "memory_usage",
                metrics.memory_percent,
                self.alert_thresholds["memory_usage"]["warning"]
            )
        
        # GPU memory alert
        if metrics.gpu_memory_total > 0:
            gpu_memory_percent = (metrics.gpu_memory_used / metrics.gpu_memory_total) * 100
            if gpu_memory_percent >= self.alert_thresholds["gpu_memory"]["critical"]:
                self._create_alert(
                    AlertLevel.CRITICAL,
                    f"Critical GPU memory usage: {gpu_memory_percent:.1f}%",
                    "gpu_memory",
                    gpu_memory_percent,
                    self.alert_thresholds["gpu_memory"]["critical"]
                )
            elif gpu_memory_percent >= self.alert_thresholds["gpu_memory"]["error"]:
                self._create_alert(
                    AlertLevel.ERROR,
                    f"High GPU memory usage: {gpu_memory_percent:.1f}%",
                    "gpu_memory",
                    gpu_memory_percent,
                    self.alert_thresholds["gpu_memory"]["error"]
                )
            elif gpu_memory_percent >= self.alert_thresholds["gpu_memory"]["warning"]:
                self._create_alert(
                    AlertLevel.WARNING,
                    f"GPU memory usage warning: {gpu_memory_percent:.1f}%",
                    "gpu_memory",
                    gpu_memory_percent,
                    self.alert_thresholds["gpu_memory"]["warning"]
                )
    
    def _create_alert(self, level: AlertLevel, message: str, metric_name: str, 
                     current_value: float, threshold: float):
        """Create a performance alert."""
        alert = PerformanceAlert(
            level=level,
            message=message,
            metric_name=metric_name,
            current_value=current_value,
            threshold=threshold,
            timestamp=datetime.now(timezone.utc)
        )
        
        self.alerts.append(alert)
        
        # Log alert
        if level == AlertLevel.CRITICAL:
            console.print(f"[red]CRITICAL: {message}[/red]")
        elif level == AlertLevel.ERROR:
            console.print(f"[red]ERROR: {message}[/red]")
        elif level == AlertLevel.WARNING:
            console.print(f"[yellow]WARNING: {message}[/yellow]")
        else:
            console.print(f"[blue]INFO: {message}[/blue]")
    
    def log_training_metrics(self, metrics: TrainingMetrics):
        """Log training metrics."""
        self.training_metrics.append(metrics)
        
        # Log to TensorBoard
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar("Loss/Train", metrics.training_loss, metrics.step)
            if metrics.validation_loss is not None:
                self.tensorboard_writer.add_scalar("Loss/Validation", metrics.validation_loss, metrics.step)
            if metrics.learning_rate is not None:
                self.tensorboard_writer.add_scalar("Learning_Rate", metrics.learning_rate, metrics.step)
            if metrics.batch_time is not None:
                self.tensorboard_writer.add_scalar("Time/Batch", metrics.batch_time, metrics.step)
            if metrics.throughput is not None:
                self.tensorboard_writer.add_scalar("Throughput", metrics.throughput, metrics.step)
            if metrics.gradient_norm is not None:
                self.tensorboard_writer.add_scalar("Gradient_Norm", metrics.gradient_norm, metrics.step)
        
        # Log to Weights & Biases
        if self.wandb_run:
            wandb.log({
                "train_loss": metrics.training_loss,
                "val_loss": metrics.validation_loss,
                "learning_rate": metrics.learning_rate,
                "batch_time": metrics.batch_time,
                "throughput": metrics.throughput,
                "gradient_norm": metrics.gradient_norm,
                "memory_usage": metrics.memory_usage,
                "gpu_memory_used": metrics.gpu_memory_used,
                "gpu_utilization": metrics.gpu_utilization,
                "cpu_usage": metrics.cpu_usage,
                "step": metrics.step,
                "epoch": metrics.epoch
            })
        
        # Log to MLflow
        if self.mlflow_run:
            mlflow.log_metrics({
                "train_loss": metrics.training_loss,
                "val_loss": metrics.validation_loss or 0.0,
                "learning_rate": metrics.learning_rate or 0.0,
                "batch_time": metrics.batch_time or 0.0,
                "throughput": metrics.throughput or 0.0,
                "gradient_norm": metrics.gradient_norm or 0.0,
                "memory_usage": metrics.memory_usage or 0.0,
                "gpu_memory_used": metrics.gpu_memory_used or 0.0,
                "gpu_utilization": metrics.gpu_utilization or 0.0,
                "cpu_usage": metrics.cpu_usage or 0.0
            }, step=metrics.step)
        
        # Check for training alerts
        self._check_training_alerts(metrics)
    
    def _check_training_alerts(self, metrics: TrainingMetrics):
        """Check for training performance alerts."""
        # Batch time alert
        if metrics.batch_time and metrics.batch_time >= self.alert_thresholds["batch_time"]["critical"]:
            self._create_alert(
                AlertLevel.CRITICAL,
                f"Critical batch time: {metrics.batch_time:.2f}s",
                "batch_time",
                metrics.batch_time,
                self.alert_thresholds["batch_time"]["critical"]
            )
        
        # Training loss alert
        if metrics.training_loss >= self.alert_thresholds["training_loss"]["critical"]:
            self._create_alert(
                AlertLevel.CRITICAL,
                f"Critical training loss: {metrics.training_loss:.4f}",
                "training_loss",
                metrics.training_loss,
                self.alert_thresholds["training_loss"]["critical"]
            )
    
    def start_profiling(self, activities: List[torch.profiler.ProfilerActivity] = None):
        """Start PyTorch profiling."""
        if self.profiling_active:
            console.print("[yellow]Profiling already active[/yellow]")
            return
            
        if activities is None:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
        
        self.profiler = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(self.log_dir / "profiler")),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        
        self.profiling_active = True
        console.print("[green]Profiling started[/green]")
    
    def stop_profiling(self):
        """Stop PyTorch profiling."""
        if not self.profiling_active or not self.profiler:
            console.print("[yellow]Profiling not active[/yellow]")
            return
            
        self.profiler.stop()
        self.profiling_active = False
        console.print("[green]Profiling stopped[/green]")
    
    def step_profiler(self):
        """Step the profiler."""
        if self.profiler and self.profiling_active:
            self.profiler.step()
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """Generate comprehensive performance report."""
        if output_path is None:
            output_path = str(self.log_dir / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        
        # Create plots
        self._create_performance_plots()
        
        # Generate HTML report
        html_content = self._generate_html_report()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        console.print(f"[green]Performance report generated: {output_path}[/green]")
        return output_path
    
    def _create_performance_plots(self):
        """Create performance visualization plots."""
        if not self.training_metrics and not self.system_metrics:
            console.print("[yellow]No metrics available for plotting[/yellow]")
            return
        
        plots_dir = self.log_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Training metrics plots
        if self.training_metrics:
            self._plot_training_metrics(plots_dir)
        
        # System metrics plots
        if self.system_metrics:
            self._plot_system_metrics(plots_dir)
    
    def _plot_training_metrics(self, plots_dir: Path):
        """Create training metrics plots."""
        df = pd.DataFrame([asdict(m) for m in self.training_metrics])
        
        # Loss plot
        fig = make_subplots(rows=2, cols=2, subplot_titles=('Training Loss', 'Learning Rate', 'Batch Time', 'Throughput'))
        
        fig.add_trace(go.Scatter(x=df['step'], y=df['training_loss'], name='Training Loss'), row=1, col=1)
        if 'validation_loss' in df.columns:
            fig.add_trace(go.Scatter(x=df['step'], y=df['validation_loss'], name='Validation Loss'), row=1, col=1)
        
        if 'learning_rate' in df.columns:
            fig.add_trace(go.Scatter(x=df['step'], y=df['learning_rate'], name='Learning Rate'), row=1, col=2)
        
        if 'batch_time' in df.columns:
            fig.add_trace(go.Scatter(x=df['step'], y=df['batch_time'], name='Batch Time'), row=1, col=3)
        
        if 'throughput' in df.columns:
            fig.add_trace(go.Scatter(x=df['step'], y=df['throughput'], name='Throughput'), row=1, col=4)
        
        fig.update_layout(height=800, title_text="Training Metrics")
        fig.write_html(str(plots_dir / "training_metrics.html"))
    
    def _plot_system_metrics(self, plots_dir: Path):
        """Create system metrics plots."""
        df = pd.DataFrame([asdict(m) for m in self.system_metrics])
        
        # System performance plot
        fig = make_subplots(rows=2, cols=2, subplot_titles=('CPU Usage', 'Memory Usage', 'GPU Memory', 'GPU Utilization'))
        
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['cpu_percent'], name='CPU %'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['memory_percent'], name='Memory %'), row=1, col=2)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['gpu_memory_used'], name='GPU Memory Used'), row=1, col=3)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['gpu_utilization'], name='GPU Utilization'), row=1, col=4)
        
        fig.update_layout(height=800, title_text="System Performance")
        fig.write_html(str(plots_dir / "system_metrics.html"))
    
    def _generate_html_report(self) -> str:
        """Generate HTML performance report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Frontier Model Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e8f4f8; border-radius: 5px; }}
                .alert {{ padding: 10px; margin: 5px 0; border-radius: 5px; }}
                .alert-critical {{ background-color: #ffebee; border-left: 5px solid #f44336; }}
                .alert-error {{ background-color: #fff3e0; border-left: 5px solid #ff9800; }}
                .alert-warning {{ background-color: #fff8e1; border-left: 5px solid #ffc107; }}
                .alert-info {{ background-color: #e3f2fd; border-left: 5px solid #2196f3; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Frontier Model Performance Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Summary Statistics</h2>
                {self._generate_summary_stats()}
            </div>
            
            <div class="section">
                <h2>Alerts</h2>
                {self._generate_alerts_html()}
            </div>
            
            <div class="section">
                <h2>Training Metrics</h2>
                {self._generate_training_metrics_html()}
            </div>
            
            <div class="section">
                <h2>System Metrics</h2>
                {self._generate_system_metrics_html()}
            </div>
        </body>
        </html>
        """
        return html
    
    def _generate_summary_stats(self) -> str:
        """Generate summary statistics HTML."""
        if not self.training_metrics and not self.system_metrics:
            return "<p>No metrics available</p>"
        
        html = "<div class='metric'>"
        
        if self.training_metrics:
            losses = [m.training_loss for m in self.training_metrics]
            html += f"<strong>Training Loss:</strong> Min: {min(losses):.4f}, Max: {max(losses):.4f}, Avg: {statistics.mean(losses):.4f}<br>"
        
        if self.system_metrics:
            cpu_values = [m.cpu_percent for m in self.system_metrics]
            memory_values = [m.memory_percent for m in self.system_metrics]
            html += f"<strong>CPU Usage:</strong> Avg: {statistics.mean(cpu_values):.1f}%<br>"
            html += f"<strong>Memory Usage:</strong> Avg: {statistics.mean(memory_values):.1f}%<br>"
        
        html += "</div>"
        return html
    
    def _generate_alerts_html(self) -> str:
        """Generate alerts HTML."""
        if not self.alerts:
            return "<p>No alerts</p>"
        
        html = ""
        for alert in self.alerts[-20:]:  # Show last 20 alerts
            level_class = f"alert-{alert.level.value}"
            html += f"""
            <div class="alert {level_class}">
                <strong>{alert.level.value.upper()}:</strong> {alert.message}<br>
                <small>Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</small>
            </div>
            """
        return html
    
    def _generate_training_metrics_html(self) -> str:
        """Generate training metrics HTML."""
        if not self.training_metrics:
            return "<p>No training metrics available</p>"
        
        html = "<table border='1' style='border-collapse: collapse; width: 100%;'>"
        html += "<tr><th>Step</th><th>Epoch</th><th>Training Loss</th><th>Validation Loss</th><th>Learning Rate</th><th>Batch Time</th></tr>"
        
        for metric in self.training_metrics[-50:]:  # Show last 50 metrics
            html += f"""
            <tr>
                <td>{metric.step}</td>
                <td>{metric.epoch}</td>
                <td>{metric.training_loss:.4f}</td>
                <td>{metric.validation_loss:.4f if metric.validation_loss else 'N/A'}</td>
                <td>{metric.learning_rate:.6f if metric.learning_rate else 'N/A'}</td>
                <td>{metric.batch_time:.3f if metric.batch_time else 'N/A'}</td>
            </tr>
            """
        
        html += "</table>"
        return html
    
    def _generate_system_metrics_html(self) -> str:
        """Generate system metrics HTML."""
        if not self.system_metrics:
            return "<p>No system metrics available</p>"
        
        html = "<table border='1' style='border-collapse: collapse; width: 100%;'>"
        html += "<tr><th>Timestamp</th><th>CPU %</th><th>Memory %</th><th>GPU Memory</th><th>GPU Utilization</th></tr>"
        
        for metric in self.system_metrics[-50:]:  # Show last 50 metrics
            html += f"""
            <tr>
                <td>{metric.timestamp.strftime('%H:%M:%S')}</td>
                <td>{metric.cpu_percent:.1f}</td>
                <td>{metric.memory_percent:.1f}</td>
                <td>{metric.gpu_memory_used:.1f}MB</td>
                <td>{metric.gpu_utilization:.1f}%</td>
            </tr>
            """
        
        html += "</table>"
        return html
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        summary = {
            "total_training_steps": len(self.training_metrics),
            "total_system_samples": len(self.system_metrics),
            "total_alerts": len(self.alerts),
            "critical_alerts": len([a for a in self.alerts if a.level == AlertLevel.CRITICAL]),
            "error_alerts": len([a for a in self.alerts if a.level == AlertLevel.ERROR]),
            "warning_alerts": len([a for a in self.alerts if a.level == AlertLevel.WARNING])
        }
        
        if self.training_metrics:
            losses = [m.training_loss for m in self.training_metrics]
            summary["training_loss"] = {
                "min": min(losses),
                "max": max(losses),
                "mean": statistics.mean(losses),
                "std": statistics.stdev(losses) if len(losses) > 1 else 0
            }
        
        if self.system_metrics:
            cpu_values = [m.cpu_percent for m in self.system_metrics]
            memory_values = [m.memory_percent for m in self.system_metrics]
            summary["system_performance"] = {
                "cpu_mean": statistics.mean(cpu_values),
                "memory_mean": statistics.mean(memory_values),
                "cpu_max": max(cpu_values),
                "memory_max": max(memory_values)
            }
        
        return summary
    
    def cleanup(self):
        """Cleanup resources."""
        self.stop_monitoring()
        self.stop_profiling()
        
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        if self.wandb_run:
            wandb.finish()
        
        if self.mlflow_run:
            mlflow.end_run()

def create_metrics_collector(log_dir: str = "./metrics",
                           enable_tensorboard: bool = True,
                           enable_wandb: bool = False,
                           enable_mlflow: bool = False,
                           wandb_project: Optional[str] = None,
                           mlflow_experiment: Optional[str] = None) -> MetricsCollector:
    """Create a metrics collector instance."""
    return MetricsCollector(
        log_dir=log_dir,
        enable_tensorboard=enable_tensorboard,
        enable_wandb=enable_wandb,
        enable_mlflow=enable_mlflow,
        wandb_project=wandb_project,
        mlflow_experiment=mlflow_experiment
    )

if __name__ == "__main__":
    # Example usage
    collector = create_metrics_collector(
        log_dir="./metrics",
        enable_tensorboard=True,
        enable_wandb=False,
        enable_mlflow=False
    )
    
    # Start monitoring
    collector.start_monitoring(interval=2.0)
    
    # Simulate some training metrics
    for step in range(10):
        training_metrics = TrainingMetrics(
            step=step,
            epoch=step // 5,
            training_loss=1.0 - step * 0.1,
            validation_loss=1.2 - step * 0.08,
            learning_rate=0.001 * (0.9 ** step),
            batch_time=0.5 + step * 0.1,
            throughput=100 - step * 5,
            memory_usage=50 + step * 2,
            gpu_memory_used=1000 + step * 100,
            gpu_memory_total=8000,
            gpu_utilization=60 + step * 3,
            cpu_usage=40 + step * 2
        )
        collector.log_training_metrics(training_metrics)
        time.sleep(1)
    
    # Wait for system monitoring
    time.sleep(10)
    
    # Stop monitoring
    collector.stop_monitoring()
    
    # Generate report
    report_path = collector.generate_report()
    console.print(f"Report generated: {report_path}")
    
    # Show summary
    summary = collector.get_summary()
    console.print(f"Summary: {summary}")
    
    # Cleanup
    collector.cleanup()
