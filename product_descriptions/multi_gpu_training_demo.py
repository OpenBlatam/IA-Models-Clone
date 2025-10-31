from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import structlog
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from multi_gpu_training import (
from typing import Any, List, Dict, Optional
"""
Multi-GPU Training Demo

This demo showcases the comprehensive multi-GPU training system with:
- DataParallel vs DistributedDataParallel comparison
- Performance benchmarking
- Memory usage monitoring
- Fault tolerance demonstration
- Real-world training scenarios
"""



    MultiGPUConfig, MultiGPUTrainingManager, TrainingMode,
    setup_distributed_training, launch_distributed_training
)

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class DemoModel(nn.Module):
    """Demo model for multi-GPU training demonstration."""
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 512, num_classes: int = 10):
        
    """__init__ function."""
super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
        self.regressor = nn.Linear(hidden_dim // 2, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module) -> Any:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x, task="classification") -> Any:
        features = self.encoder(x)
        
        if task == "classification":
            logits = self.classifier(features)
            return {
                'logits': logits,
                'features': features,
                'task': task
            }
        elif task == "regression":
            prediction = self.regressor(features)
            return {
                'prediction': prediction,
                'features': features,
                'task': task
            }
        else:
            logits = self.classifier(features)
            prediction = self.regressor(features)
            return {
                'logits': logits,
                'prediction': prediction,
                'features': features,
                'task': 'multitask'
            }


class MultiTaskLoss(nn.Module):
    """Multi-task loss function."""
    
    def __init__(self, classification_weight: float = 1.0, regression_weight: float = 0.5):
        
    """__init__ function."""
super().__init__()
        self.classification_weight = classification_weight
        self.regression_weight = regression_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, outputs: Dict, targets: Dict) -> Dict[str, torch.Tensor]:
        losses = {}
        
        if 'logits' in outputs:
            ce_loss = self.ce_loss(outputs['logits'], targets['labels'])
            losses['classification_loss'] = ce_loss * self.classification_weight
        
        if 'prediction' in outputs:
            mse_loss = self.mse_loss(outputs['prediction'].squeeze(), targets['values'])
            losses['regression_loss'] = mse_loss * self.regression_weight
        
        total_loss = sum(losses.values())
        losses['total_loss'] = total_loss
        
        return losses


class DemoDataset(Dataset):
    """Demo dataset with multi-task capabilities."""
    
    def __init__(self, num_samples: int = 10000, input_dim: int = 768, num_classes: int = 10):
        
    """__init__ function."""
self.num_samples = num_samples
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Generate synthetic data
        self.data = torch.randn(num_samples, input_dim)
        
        # Generate classification labels
        self.labels = torch.randint(0, num_classes, (num_samples,))
        
        # Generate regression values
        self.values = torch.randn(num_samples)
        
        logger.info(f"Created demo dataset with {num_samples} samples")
    
    def __len__(self) -> Any:
        return self.num_samples
    
    def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
        return {
            'input_ids': self.data[idx],
            'labels': self.labels[idx],
            'values': self.values[idx]
        }


class PerformanceMonitor:
    """Monitor training performance and resource usage."""
    
    def __init__(self) -> Any:
        self.metrics = {
            'training_time': [],
            'memory_usage': [],
            'gpu_utilization': [],
            'throughput': [],
            'loss_history': []
        }
        self.start_time = None
    
    def start_monitoring(self) -> Any:
        """Start performance monitoring."""
        self.start_time = time.time()
        logger.info("Performance monitoring started")
    
    def record_metrics(self, loss: float, batch_size: int, num_gpus: int):
        """Record training metrics."""
        if self.start_time is None:
            return
        
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # Record metrics
        self.metrics['training_time'].append(elapsed_time)
        self.metrics['loss_history'].append(loss)
        
        # Calculate throughput (samples per second)
        total_samples = len(self.metrics['training_time']) * batch_size * num_gpus
        throughput = total_samples / elapsed_time if elapsed_time > 0 else 0
        self.metrics['throughput'].append(throughput)
        
        # Record GPU memory usage
        if torch.cuda.is_available():
            memory_usage = torch.cuda.memory_allocated() / 1024**3  # GB
            self.metrics['memory_usage'].append(memory_usage)
            
            # GPU utilization (simplified)
            gpu_util = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
            self.metrics['gpu_utilization'].append(gpu_util)
    
    def get_summary(self) -> Dict:
        """Get performance summary."""
        if not self.metrics['training_time']:
            return {}
        
        return {
            'total_training_time': self.metrics['training_time'][-1],
            'avg_loss': np.mean(self.metrics['loss_history']),
            'final_loss': self.metrics['loss_history'][-1],
            'max_throughput': max(self.metrics['throughput']),
            'avg_throughput': np.mean(self.metrics['throughput']),
            'max_memory_usage': max(self.metrics['memory_usage']) if self.metrics['memory_usage'] else 0,
            'avg_gpu_utilization': np.mean(self.metrics['gpu_utilization']) if self.metrics['gpu_utilization'] else 0
        }
    
    def plot_metrics(self, save_path: str = "training_metrics.png"):
        """Plot training metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss over time
        axes[0, 0].plot(self.metrics['loss_history'])
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        
        # Throughput over time
        axes[0, 1].plot(self.metrics['throughput'])
        axes[0, 1].set_title('Training Throughput')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Samples/Second')
        
        # Memory usage over time
        if self.metrics['memory_usage']:
            axes[1, 0].plot(self.metrics['memory_usage'])
            axes[1, 0].set_title('GPU Memory Usage')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Memory (GB)')
        
        # GPU utilization over time
        if self.metrics['gpu_utilization']:
            axes[1, 1].plot(self.metrics['gpu_utilization'])
            axes[1, 1].set_title('GPU Utilization')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Utilization (%)')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training metrics plot saved to {save_path}")


class MultiGPUTrainingDemo:
    """Comprehensive demo for multi-GPU training."""
    
    def __init__(self) -> Any:
        self.results = {}
        self.monitor = PerformanceMonitor()
    
    async def run_data_parallel_demo(self) -> Dict:
        """Demonstrate DataParallel training."""
        logger.info("Starting DataParallel Demo")
        
        # Configuration
        num_gpus = min(2, torch.cuda.device_count())
        config = MultiGPUConfig(
            training_mode=TrainingMode.DATA_PARALLEL,
            device_ids=list(range(num_gpus)),
            use_mixed_precision=True,
            gradient_accumulation_steps=2,
            batch_size=64,
            num_workers=4
        )
        
        # Setup
        manager = MultiGPUTrainingManager(config)
        model = DemoModel()
        dataset = DemoDataset(5000)
        
        # Setup training
        model, train_loader, _ = manager.setup_training(
            model, dataset,
            optimizer=optim.AdamW(model.parameters(), lr=1e-4),
            scheduler=optim.lr_scheduler.CosineAnnealingLR(
                optim.AdamW(model.parameters(), lr=1e-4), T_max=10
            )
        )
        
        # Training
        self.monitor.start_monitoring()
        num_epochs = 3
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            for batch_idx, batch in enumerate(train_loader):
                try:
                    outputs = manager.trainer.train_step(batch)
                    loss = outputs.get('total_loss', 0.0)
                    epoch_losses.append(loss)
                    
                    # Record metrics
                    self.monitor.record_metrics(loss, config.batch_size, num_gpus)
                    
                    if batch_idx % 50 == 0:
                        logger.info(
                            f"DataParallel Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.4f}"
                        )
                
                except Exception as e:
                    logger.error(f"Training step failed: {e}")
                    raise
            
            avg_loss = np.mean(epoch_losses)
            logger.info(f"DataParallel Epoch {epoch} completed, Avg Loss: {avg_loss:.4f}")
        
        # Results
        summary = self.monitor.get_summary()
        summary['training_mode'] = 'DataParallel'
        summary['num_gpus'] = num_gpus
        
        manager.cleanup()
        return summary
    
    async def run_distributed_demo(self) -> Dict:
        """Demonstrate DistributedDataParallel training."""
        logger.info("Starting DistributedDataParallel Demo")
        
        # Configuration
        world_size = min(2, torch.cuda.device_count())
        config = MultiGPUConfig(
            training_mode=TrainingMode.DISTRIBUTED_DATA_PARALLEL,
            world_size=world_size,
            rank=0,
            local_rank=0,
            use_mixed_precision=True,
            gradient_accumulation_steps=2,
            batch_size=32,  # Smaller batch size per GPU
            num_workers=2
        )
        
        # Setup
        manager = MultiGPUTrainingManager(config)
        model = DemoModel()
        dataset = DemoDataset(5000)
        
        # Setup training
        model, train_loader, _ = manager.setup_training(
            model, dataset,
            optimizer=optim.AdamW(model.parameters(), lr=1e-4)
        )
        
        # Training
        self.monitor.start_monitoring()
        num_epochs = 3
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            for batch_idx, batch in enumerate(train_loader):
                try:
                    outputs = manager.trainer.train_step(batch)
                    loss = outputs.get('total_loss', 0.0)
                    epoch_losses.append(loss)
                    
                    # Record metrics
                    self.monitor.record_metrics(loss, config.batch_size, world_size)
                    
                    if batch_idx % 50 == 0:
                        logger.info(
                            f"DistributedDataParallel Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.4f}"
                        )
                
                except Exception as e:
                    logger.error(f"Training step failed: {e}")
                    raise
            
            avg_loss = np.mean(epoch_losses)
            logger.info(f"DistributedDataParallel Epoch {epoch} completed, Avg Loss: {avg_loss:.4f}")
        
        # Results
        summary = self.monitor.get_summary()
        summary['training_mode'] = 'DistributedDataParallel'
        summary['world_size'] = world_size
        
        manager.cleanup()
        return summary
    
    async def run_memory_optimization_demo(self) -> Dict:
        """Demonstrate memory optimization techniques."""
        logger.info("Starting Memory Optimization Demo")
        
        config = MultiGPUConfig(
            training_mode=TrainingMode.DATA_PARALLEL,
            device_ids=[0],  # Single GPU for memory demo
            use_mixed_precision=True,
            gradient_accumulation_steps=4,
            batch_size=128,
            num_workers=2
        )
        
        manager = MultiGPUTrainingManager(config)
        model = DemoModel()
        dataset = DemoDataset(3000)
        
        # Setup training
        model, train_loader, _ = manager.setup_training(model, dataset)
        
        # Memory monitoring
        initial_memory = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        
        self.monitor.start_monitoring()
        num_epochs = 2
        
        for epoch in range(num_epochs):
            for batch_idx, batch in enumerate(train_loader):
                outputs = manager.trainer.train_step(batch)
                loss = outputs.get('total_loss', 0.0)
                
                self.monitor.record_metrics(loss, config.batch_size, 1)
                
                if batch_idx % 20 == 0:
                    current_memory = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                    logger.info(
                        f"Memory Demo - Epoch {epoch}, Batch {batch_idx}, "
                        f"Loss: {loss:.4f}, Memory: {current_memory:.2f}GB"
                    )
        
        final_memory = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        
        summary = self.monitor.get_summary()
        summary['training_mode'] = 'MemoryOptimized'
        summary['initial_memory_gb'] = initial_memory
        summary['final_memory_gb'] = final_memory
        summary['memory_increase_gb'] = final_memory - initial_memory
        
        manager.cleanup()
        return summary
    
    async def run_fault_tolerance_demo(self) -> Dict:
        """Demonstrate fault tolerance capabilities."""
        logger.info("Starting Fault Tolerance Demo")
        
        config = MultiGPUConfig(
            training_mode=TrainingMode.DATA_PARALLEL,
            device_ids=[0],
            enable_fault_tolerance=True,
            checkpoint_frequency=50,
            use_mixed_precision=True,
            batch_size=64
        )
        
        manager = MultiGPUTrainingManager(config)
        model = DemoModel()
        dataset = DemoDataset(2000)
        
        # Setup training
        model, train_loader, _ = manager.setup_training(model, dataset)
        
        self.monitor.start_monitoring()
        num_epochs = 2
        recovery_count = 0
        
        for epoch in range(num_epochs):
            for batch_idx, batch in enumerate(train_loader):
                try:
                    outputs = manager.trainer.train_step(batch)
                    loss = outputs.get('total_loss', 0.0)
                    
                    self.monitor.record_metrics(loss, config.batch_size, 1)
                    
                    # Simulate occasional failures
                    if batch_idx > 0 and batch_idx % 100 == 0 and np.random.random() < 0.1:
                        logger.warning(f"Simulating training failure at batch {batch_idx}")
                        raise RuntimeError("Simulated training failure")
                
                except Exception as e:
                    logger.warning(f"Training error occurred: {e}")
                    recovery_count += 1
                    
                    # Continue training (fault tolerance handled by manager)
                    continue
        
        summary = self.monitor.get_summary()
        summary['training_mode'] = 'FaultTolerant'
        summary['recovery_count'] = recovery_count
        
        manager.cleanup()
        return summary
    
    async def run_comprehensive_demo(self) -> Dict:
        """Run comprehensive multi-GPU training demo."""
        logger.info("Starting Comprehensive Multi-GPU Training Demo")
        
        results = {}
        
        # Check GPU availability
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, skipping GPU demos")
            return {'error': 'CUDA not available'}
        
        num_gpus = torch.cuda.device_count()
        logger.info(f"Found {num_gpus} GPU(s)")
        
        # Run different demos
        try:
            # DataParallel demo
            if num_gpus >= 1:
                results['data_parallel'] = await self.run_data_parallel_demo()
            
            # DistributedDataParallel demo (if multiple GPUs)
            if num_gpus >= 2:
                results['distributed_data_parallel'] = await self.run_distributed_demo()
            
            # Memory optimization demo
            results['memory_optimization'] = await self.run_memory_optimization_demo()
            
            # Fault tolerance demo
            results['fault_tolerance'] = await self.run_fault_tolerance_demo()
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            results['error'] = str(e)
        
        # Generate comparison report
        comparison = self._generate_comparison_report(results)
        results['comparison'] = comparison
        
        # Save results
        self._save_results(results)
        
        # Plot metrics
        self.monitor.plot_metrics("multi_gpu_training_metrics.png")
        
        return results
    
    def _generate_comparison_report(self, results: Dict) -> Dict:
        """Generate comparison report between different training modes."""
        comparison = {
            'performance_comparison': {},
            'recommendations': []
        }
        
        # Compare performance metrics
        for mode, data in results.items():
            if isinstance(data, dict) and 'training_mode' in data:
                comparison['performance_comparison'][mode] = {
                    'throughput': data.get('avg_throughput', 0),
                    'memory_usage': data.get('max_memory_usage', 0),
                    'training_time': data.get('total_training_time', 0),
                    'final_loss': data.get('final_loss', 0)
                }
        
        # Generate recommendations
        if 'data_parallel' in results and 'distributed_data_parallel' in results:
            dp_data = results['data_parallel']
            ddp_data = results['distributed_data_parallel']
            
            if dp_data.get('avg_throughput', 0) > ddp_data.get('avg_throughput', 0):
                comparison['recommendations'].append(
                    "DataParallel shows better throughput for single-node multi-GPU training"
                )
            else:
                comparison['recommendations'].append(
                    "DistributedDataParallel shows better throughput for distributed training"
                )
        
        if 'memory_optimization' in results:
            mem_data = results['memory_optimization']
            if mem_data.get('memory_increase_gb', 0) < 1.0:
                comparison['recommendations'].append(
                    "Memory optimization techniques effectively control memory usage"
                )
        
        return comparison
    
    def _save_results(self, results: Dict):
        """Save demo results to file."""
        output_path = Path("multi_gpu_training_results.json")
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj) -> Any:
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Recursively convert numpy types
        def recursive_convert(obj) -> Any:
            if isinstance(obj, dict):
                return {k: recursive_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recursive_convert(v) for v in obj]
            else:
                return convert_numpy(obj)
        
        serializable_results = recursive_convert(results)
        
        with open(output_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Demo results saved to {output_path}")


async def main():
    """Main demo function."""
    logger.info("Multi-GPU Training System Demo")
    
    # Create demo instance
    demo = MultiGPUTrainingDemo()
    
    # Run comprehensive demo
    results = await demo.run_comprehensive_demo()
    
    # Print summary
    logger.info("Demo completed successfully!")
    logger.info("Results summary:", results=results)


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main()) 