from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

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
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from advanced_gradient_accumulation import (
from typing import Any, List, Dict, Optional
"""
Large Batch Gradient Accumulation Demo

This demo showcases advanced gradient accumulation techniques specifically
designed for training with very large effective batch sizes:

- Memory-efficient large batch training
- Dynamic batch size scaling
- Performance optimization for large models
- Memory monitoring and optimization
- Comparison of different accumulation strategies
- Real-world training scenarios with large datasets
"""



    GradientAccumulationConfig, AccumulationStrategy,
    AdvancedGradientAccumulationTrainer, MemoryMonitor,
    PerformanceMetrics
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


class LargeModel(nn.Module):
    """Large model for testing gradient accumulation with big batch sizes."""
    
    def __init__(self, input_dim: int = 768, hidden_dims: List[int] = None, num_classes: int = 10):
        
    """__init__ function."""
super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [1024, 512, 256, 128]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.backbone = nn.Sequential(*layers)
        
        # Additional components for large model
        self.attention = nn.MultiheadAttention(768, 8, batch_first=True)
        self.layer_norm = nn.LayerNorm(768)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module) -> Any:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x, labels=None) -> Any:
        # Expand input if needed
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        # Apply attention
        if x.size(1) > 1:
            attn_output, _ = self.attention(x, x, x)
            x = self.layer_norm(x + attn_output)
        
        # Flatten and pass through backbone
        x = x.view(x.size(0), -1)
        logits = self.backbone(x)
        
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        
        return {
            'logits': logits,
            'loss': loss,
            'features': x
        }


class LargeDataset(Dataset):
    """Large dataset for testing gradient accumulation."""
    
    def __init__(self, num_samples: int = 10000, input_dim: int = 768, num_classes: int = 10):
        
    """__init__ function."""
self.num_samples = num_samples
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Generate synthetic data
        self.data = torch.randn(num_samples, input_dim)
        self.labels = torch.randint(0, num_classes, (num_samples,))
        
        # Add some structure to the data
        for i in range(num_classes):
            mask = (self.labels == i)
            if mask.sum() > 0:
                self.data[mask] += torch.randn_like(self.data[mask]) * 0.1
        
        logger.info(f"Created large dataset with {num_samples} samples")
    
    def __len__(self) -> Any:
        return self.num_samples
    
    def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
        return {
            'input_ids': self.data[idx],
            'labels': self.labels[idx]
        }


class LargeBatchGradientAccumulationDemo:
    """Demo for large batch gradient accumulation."""
    
    def __init__(self) -> Any:
        self.results = {}
        self.monitors = {}
        
    async def run_memory_efficient_large_batch_demo(self) -> Dict:
        """Demonstrate memory-efficient large batch training."""
        logger.info("Starting Memory-Efficient Large Batch Demo")
        
        config = GradientAccumulationConfig(
            strategy=AccumulationStrategy.MEMORY_AWARE,
            accumulation_steps=16,  # Large accumulation
            target_batch_size=2048,  # Very large target batch
            max_memory_usage_gb=12.0,  # Conservative memory limit
            use_mixed_precision=True,
            enable_monitoring=True,
            log_every_n_steps=5
        )
        
        trainer = AdvancedGradientAccumulationTrainer(config)
        model = LargeModel(input_dim=768, hidden_dims=[1024, 512, 256])
        dataset = LargeDataset(5000, input_dim=768)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        scaler = GradScaler()
        
        # Training loop
        epoch_losses = []
        memory_usage = []
        
        for epoch in range(3):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(dataloader):
                result = trainer.train_step(batch, model, optimizer, scaler)
                
                if result['should_optimize']:
                    epoch_loss += result['outputs']['loss'].item()
                    num_batches += 1
                
                # Record memory usage
                if torch.cuda.is_available():
                    memory_usage.append(torch.cuda.memory_allocated() / 1024**3)
                
                if batch_idx >= 50:  # Limit training steps
                    break
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            epoch_losses.append(avg_loss)
            
            logger.info(f"Epoch {epoch}: Avg Loss = {avg_loss:.4f}")
        
        # Get final stats
        stats = trainer.get_training_stats()
        stats['epoch_losses'] = epoch_losses
        stats['memory_usage'] = memory_usage
        
        trainer.cleanup()
        return stats
    
    async def run_dynamic_batch_scaling_demo(self) -> Dict:
        """Demonstrate dynamic batch size scaling."""
        logger.info("Starting Dynamic Batch Scaling Demo")
        
        config = GradientAccumulationConfig(
            strategy=AccumulationStrategy.DYNAMIC,
            accumulation_steps=8,
            target_batch_size=1024,
            min_accumulation_steps=4,
            max_accumulation_steps=32,
            automatic_scaling=True,
            use_mixed_precision=True,
            enable_monitoring=True
        )
        
        trainer = AdvancedGradientAccumulationTrainer(config)
        model = LargeModel(input_dim=512, hidden_dims=[768, 384, 192])
        dataset = LargeDataset(3000, input_dim=512)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        scaler = GradScaler()
        
        # Track batch size changes
        batch_size_history = []
        performance_history = []
        
        for batch_idx, batch in enumerate(dataloader):
            result = trainer.train_step(batch, model, optimizer, scaler)
            
            # Record batch size and performance
            batch_size_history.append(result['effective_batch_size'])
            performance_history.append(result['outputs']['loss'].item())
            
            if batch_idx >= 100:  # Train for 100 steps
                break
        
        stats = trainer.get_training_stats()
        stats['batch_size_history'] = batch_size_history
        stats['performance_history'] = performance_history
        
        trainer.cleanup()
        return stats
    
    async def run_performance_optimization_demo(self) -> Dict:
        """Demonstrate performance optimization for large batches."""
        logger.info("Starting Performance Optimization Demo")
        
        config = GradientAccumulationConfig(
            strategy=AccumulationStrategy.PERFORMANCE_OPTIMIZED,
            accumulation_steps=12,
            target_batch_size=1536,
            use_mixed_precision=True,
            gradient_scaling=True,
            use_gradient_accumulation_hooks=True,
            enable_monitoring=True
        )
        
        trainer = AdvancedGradientAccumulationTrainer(config)
        model = LargeModel(input_dim=1024, hidden_dims=[2048, 1024, 512, 256])
        dataset = LargeDataset(4000, input_dim=1024)
        dataloader = DataLoader(dataset, batch_size=48, shuffle=True)
        
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        scaler = GradScaler()
        
        # Performance tracking
        throughput_history = []
        memory_efficiency_history = []
        
        for batch_idx, batch in enumerate(dataloader):
            start_time = time.time()
            
            result = trainer.train_step(batch, model, optimizer, scaler)
            
            step_time = time.time() - start_time
            throughput = result['effective_batch_size'] / step_time
            throughput_history.append(throughput)
            
            # Calculate memory efficiency
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                memory_efficiency = memory_used / memory_total
                memory_efficiency_history.append(memory_efficiency)
            
            if batch_idx >= 80:  # Train for 80 steps
                break
        
        stats = trainer.get_training_stats()
        stats['throughput_history'] = throughput_history
        stats['memory_efficiency_history'] = memory_efficiency_history
        
        trainer.cleanup()
        return stats
    
    async def run_adaptive_strategy_demo(self) -> Dict:
        """Demonstrate adaptive gradient accumulation strategy."""
        logger.info("Starting Adaptive Strategy Demo")
        
        config = GradientAccumulationConfig(
            strategy=AccumulationStrategy.ADAPTIVE,
            accumulation_steps=6,
            target_batch_size=768,
            min_accumulation_steps=2,
            max_accumulation_steps=24,
            automatic_scaling=True,
            use_mixed_precision=True,
            enable_monitoring=True
        )
        
        trainer = AdvancedGradientAccumulationTrainer(config)
        model = LargeModel(input_dim=640, hidden_dims=[1280, 640, 320])
        dataset = LargeDataset(3500, input_dim=640)
        dataloader = DataLoader(dataset, batch_size=40, shuffle=True)
        
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        scaler = GradScaler()
        
        # Track adaptations
        adaptation_history = []
        effective_batch_sizes = []
        
        for batch_idx, batch in enumerate(dataloader):
            result = trainer.train_step(batch, model, optimizer, scaler)
            
            effective_batch_sizes.append(result['effective_batch_size'])
            
            # Record adaptations
            if hasattr(trainer.accumulator, 'adaptation_history'):
                adaptation_history.extend(trainer.accumulator.adaptation_history)
                trainer.accumulator.adaptation_history.clear()
            
            if batch_idx >= 120:  # Train for 120 steps
                break
        
        stats = trainer.get_training_stats()
        stats['effective_batch_sizes'] = effective_batch_sizes
        stats['adaptation_history'] = adaptation_history
        
        trainer.cleanup()
        return stats
    
    async def run_comparison_demo(self) -> Dict:
        """Compare different gradient accumulation strategies."""
        logger.info("Starting Strategy Comparison Demo")
        
        strategies = [
            (AccumulationStrategy.FIXED, "Fixed Strategy"),
            (AccumulationStrategy.DYNAMIC, "Dynamic Strategy"),
            (AccumulationStrategy.MEMORY_AWARE, "Memory-Aware Strategy"),
            (AccumulationStrategy.PERFORMANCE_OPTIMIZED, "Performance-Optimized Strategy"),
            (AccumulationStrategy.ADAPTIVE, "Adaptive Strategy")
        ]
        
        comparison_results = {}
        
        for strategy, strategy_name in strategies:
            logger.info(f"Testing {strategy_name}")
            
            config = GradientAccumulationConfig(
                strategy=strategy,
                accumulation_steps=8,
                target_batch_size=1024,
                use_mixed_precision=True,
                enable_monitoring=True
            )
            
            trainer = AdvancedGradientAccumulationTrainer(config)
            model = LargeModel(input_dim=512, hidden_dims=[768, 384])
            dataset = LargeDataset(2000, input_dim=512)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            optimizer = optim.AdamW(model.parameters(), lr=1e-4)
            scaler = GradScaler()
            
            # Training metrics
            losses = []
            throughputs = []
            memory_usage = []
            
            start_time = time.time()
            
            for batch_idx, batch in enumerate(dataloader):
                result = trainer.train_step(batch, model, optimizer, scaler)
                
                if result['should_optimize']:
                    losses.append(result['outputs']['loss'].item())
                
                # Calculate throughput
                step_time = time.time() - start_time
                throughput = result['effective_batch_size'] / step_time if step_time > 0 else 0
                throughputs.append(throughput)
                
                # Record memory usage
                if torch.cuda.is_available():
                    memory_usage.append(torch.cuda.memory_allocated() / 1024**3)
                
                if batch_idx >= 60:  # Train for 60 steps
                    break
            
            # Get final stats
            stats = trainer.get_training_stats()
            stats['losses'] = losses
            stats['throughputs'] = throughputs
            stats['memory_usage'] = memory_usage
            stats['training_time'] = time.time() - start_time
            
            comparison_results[strategy_name] = stats
            trainer.cleanup()
        
        return comparison_results
    
    def generate_performance_report(self, results: Dict) -> Dict:
        """Generate comprehensive performance report."""
        report = {
            'summary': {},
            'comparisons': {},
            'recommendations': []
        }
        
        # Analyze each demo
        for demo_name, demo_results in results.items():
            if isinstance(demo_results, dict) and 'performance_metrics' in demo_results:
                metrics = demo_results['performance_metrics']
                report['summary'][demo_name] = {
                    'throughput': metrics.get('throughput', 0),
                    'memory_efficiency': metrics.get('memory_efficiency', 0),
                    'effective_batch_size': demo_results.get('accumulator_stats', {}).get('effective_batch_size', 0)
                }
        
        # Generate comparisons
        if 'comparison_demo' in results:
            comparison = results['comparison_demo']
            best_throughput = max(
                comparison.items(),
                key=lambda x: x[1].get('performance_metrics', {}).get('throughput', 0)
            )
            best_memory = min(
                comparison.items(),
                key=lambda x: x[1].get('memory_stats', {}).get('current_memory_gb', float('inf'))
            )
            
            report['comparisons'] = {
                'best_throughput': best_throughput[0],
                'best_memory_efficiency': best_memory[0]
            }
        
        # Generate recommendations
        report['recommendations'] = [
            "Use Memory-Aware strategy for memory-constrained environments",
            "Use Performance-Optimized strategy for maximum throughput",
            "Use Adaptive strategy for dynamic environments",
            "Enable mixed precision for better memory efficiency",
            "Monitor memory usage and adjust accumulation steps accordingly"
        ]
        
        return report
    
    def plot_results(self, results: Dict, save_path: str = "large_batch_results.png"):
        """Plot comprehensive results."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Memory usage comparison
        if 'comparison_demo' in results:
            comparison = results['comparison_demo']
            for strategy_name, stats in comparison.items():
                memory_usage = stats.get('memory_usage', [])
                if memory_usage:
                    axes[0, 0].plot(memory_usage, label=strategy_name)
            
            axes[0, 0].set_title('Memory Usage Comparison')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Memory (GB)')
            axes[0, 0].legend()
        
        # Plot 2: Throughput comparison
        if 'comparison_demo' in results:
            comparison = results['comparison_demo']
            for strategy_name, stats in comparison.items():
                throughputs = stats.get('throughputs', [])
                if throughputs:
                    axes[0, 1].plot(throughputs, label=strategy_name)
            
            axes[0, 1].set_title('Throughput Comparison')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Samples/Second')
            axes[0, 1].legend()
        
        # Plot 3: Loss convergence
        if 'comparison_demo' in results:
            comparison = results['comparison_demo']
            for strategy_name, stats in comparison.items():
                losses = stats.get('losses', [])
                if losses:
                    axes[0, 2].plot(losses, label=strategy_name)
            
            axes[0, 2].set_title('Loss Convergence')
            axes[0, 2].set_xlabel('Optimization Step')
            axes[0, 2].set_ylabel('Loss')
            axes[0, 2].legend()
        
        # Plot 4: Dynamic batch scaling
        if 'dynamic_batch_scaling_demo' in results:
            batch_sizes = results['dynamic_batch_scaling_demo'].get('batch_size_history', [])
            if batch_sizes:
                axes[1, 0].plot(batch_sizes)
                axes[1, 0].set_title('Dynamic Batch Size Scaling')
                axes[1, 0].set_xlabel('Step')
                axes[1, 0].set_ylabel('Effective Batch Size')
        
        # Plot 5: Performance optimization
        if 'performance_optimization_demo' in results:
            throughputs = results['performance_optimization_demo'].get('throughput_history', [])
            if throughputs:
                axes[1, 1].plot(throughputs)
                axes[1, 1].set_title('Performance Optimization')
                axes[1, 1].set_xlabel('Step')
                axes[1, 1].set_ylabel('Throughput (samples/sec)')
        
        # Plot 6: Adaptive strategy
        if 'adaptive_strategy_demo' in results:
            batch_sizes = results['adaptive_strategy_demo'].get('effective_batch_sizes', [])
            if batch_sizes:
                axes[1, 2].plot(batch_sizes)
                axes[1, 2].set_title('Adaptive Strategy')
                axes[1, 2].set_xlabel('Step')
                axes[1, 2].set_ylabel('Effective Batch Size')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Results plot saved to {save_path}")
    
    async def run_comprehensive_demo(self) -> Dict:
        """Run comprehensive large batch gradient accumulation demo."""
        logger.info("Starting Comprehensive Large Batch Gradient Accumulation Demo")
        
        results = {}
        
        try:
            # Run individual demos
            results['memory_efficient_demo'] = await self.run_memory_efficient_large_batch_demo()
            results['dynamic_scaling_demo'] = await self.run_dynamic_batch_scaling_demo()
            results['performance_optimization_demo'] = await self.run_performance_optimization_demo()
            results['adaptive_strategy_demo'] = await self.run_adaptive_strategy_demo()
            results['comparison_demo'] = await self.run_comparison_demo()
            
            # Generate report
            report = self.generate_performance_report(results)
            results['performance_report'] = report
            
            # Plot results
            self.plot_results(results)
            
            # Save results
            self._save_results(results)
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def _save_results(self, results: Dict):
        """Save demo results to file."""
        output_path = Path("large_batch_gradient_accumulation_results.json")
        
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
    logger.info("Large Batch Gradient Accumulation Demo")
    
    # Create demo instance
    demo = LargeBatchGradientAccumulationDemo()
    
    # Run comprehensive demo
    results = await demo.run_comprehensive_demo()
    
    # Print summary
    logger.info("Demo completed successfully!")
    
    if 'performance_report' in results:
        report = results['performance_report']
        logger.info("Performance Summary:")
        for demo_name, metrics in report['summary'].items():
            logger.info(
                f"{demo_name}: Throughput = {metrics['throughput']:.2f} samples/sec, "
                f"Memory Efficiency = {metrics['memory_efficiency']:.2f}"
            )


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main()) 