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
from advanced_mixed_precision_training import (
from typing import Any, List, Dict, Optional
"""
Advanced Mixed Precision Training Demo

This demo showcases comprehensive mixed precision training capabilities using
torch.cuda.amp with advanced features:

- Performance comparison between FP32, FP16, and mixed precision
- Memory usage optimization
- Gradient scaling strategies
- Numerical stability monitoring
- Automatic fallback mechanisms
- Performance profiling and optimization
- Real-world training scenarios
"""



    MixedPrecisionConfig, PrecisionMode, ScalingStrategy,
    AdvancedMixedPrecisionManager, StandardMixedPrecisionTrainer,
    DynamicMixedPrecisionTrainer, PerformanceOptimizedMixedPrecisionTrainer
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


class LargeTransformerModel(nn.Module):
    """Large transformer model for testing mixed precision training."""
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 1024, num_layers: int = 6, num_classes: int = 10):
        
    """__init__ function."""
super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Output layers
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module) -> Any:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x, labels=None) -> Any:
        # Add sequence dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Embedding
        x = self.embedding(x)
        
        # Transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Output
        x = self.layer_norm(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        
        return {
            'logits': logits,
            'loss': loss,
            'features': x
        }


class LargeDataset(Dataset):
    """Large dataset for testing mixed precision training."""
    
    def __init__(self, num_samples: int = 5000, input_dim: int = 768, num_classes: int = 10):
        
    """__init__ function."""
self.num_samples = num_samples
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Generate synthetic data with some structure
        self.data = torch.randn(num_samples, input_dim)
        self.labels = torch.randint(0, num_classes, (num_samples,))
        
        # Add class-specific patterns
        for i in range(num_classes):
            mask = (self.labels == i)
            if mask.sum() > 0:
                self.data[mask] += torch.randn_like(self.data[mask]) * 0.2
        
        logger.info(f"Created large dataset with {num_samples} samples")
    
    def __len__(self) -> Any:
        return self.num_samples
    
    def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
        return {
            'input_ids': self.data[idx],
            'labels': self.labels[idx]
        }


class MixedPrecisionTrainingDemo:
    """Comprehensive demo for mixed precision training."""
    
    def __init__(self) -> Any:
        self.results = {}
        self.monitors = {}
        
    async def run_fp32_vs_fp16_comparison(self) -> Dict:
        """Compare FP32 vs FP16 vs Mixed Precision training."""
        logger.info("Starting FP32 vs FP16 vs Mixed Precision Comparison")
        
        precision_modes = [
            (PrecisionMode.FP32, "FP32 Training"),
            (PrecisionMode.MIXED, "Mixed Precision Training"),
            (PrecisionMode.FP16, "FP16 Training")
        ]
        
        comparison_results = {}
        
        for precision_mode, mode_name in precision_modes:
            logger.info(f"Testing {mode_name}")
            
            config = MixedPrecisionConfig(
                enabled=precision_mode != PrecisionMode.FP32,
                precision_mode=precision_mode,
                scaling_strategy=ScalingStrategy.CONSTANT,
                enable_monitoring=True,
                log_every_n_steps=50
            )
            
            manager = AdvancedMixedPrecisionManager(config)
            model = LargeTransformerModel(input_dim=512, hidden_dim=768, num_layers=4)
            dataset = LargeDataset(2000, input_dim=512)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
            
            # Training metrics
            losses = []
            step_times = []
            memory_usage = []
            
            start_time = time.time()
            
            # Train for one epoch
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= 100:  # Limit training steps
                    break
                
                batch_start_time = time.time()
                
                try:
                    result = manager.trainer.train_step(batch, model, optimizer)
                    
                    batch_time = time.time() - batch_start_time
                    losses.append(result['loss'])
                    step_times.append(batch_time)
                    
                    # Record memory usage
                    if torch.cuda.is_available():
                        memory_usage.append(torch.cuda.memory_allocated() / 1024**3)
                    
                except Exception as e:
                    logger.error(f"Training step failed in {mode_name}: {e}")
                    break
            
            training_time = time.time() - start_time
            
            # Get final stats
            stats = manager.get_training_stats()
            stats['losses'] = losses
            stats['step_times'] = step_times
            stats['memory_usage'] = memory_usage
            stats['training_time'] = training_time
            stats['mode_name'] = mode_name
            
            comparison_results[mode_name] = stats
            manager.cleanup()
        
        return comparison_results
    
    async def run_scaling_strategy_comparison(self) -> Dict:
        """Compare different gradient scaling strategies."""
        logger.info("Starting Scaling Strategy Comparison")
        
        strategies = [
            (ScalingStrategy.CONSTANT, "Constant Scaling"),
            (ScalingStrategy.DYNAMIC, "Dynamic Scaling"),
            (ScalingStrategy.PERFORMANCE_OPTIMIZED, "Performance Optimized")
        ]
        
        strategy_results = {}
        
        for strategy, strategy_name in strategies:
            logger.info(f"Testing {strategy_name}")
            
            config = MixedPrecisionConfig(
                enabled=True,
                precision_mode=PrecisionMode.MIXED,
                scaling_strategy=strategy,
                enable_monitoring=True,
                automatic_fallback=True
            )
            
            manager = AdvancedMixedPrecisionManager(config)
            model = LargeTransformerModel(input_dim=384, hidden_dim=512, num_layers=3)
            dataset = LargeDataset(1500, input_dim=384)
            dataloader = DataLoader(dataset, batch_size=48, shuffle=True)
            
            optimizer = optim.AdamW(model.parameters(), lr=1e-4)
            
            # Training metrics
            gradient_scales = []
            losses = []
            numerical_errors = []
            
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= 80:  # Limit training steps
                    break
                
                try:
                    result = manager.trainer.train_step(batch, model, optimizer)
                    
                    losses.append(result['loss'])
                    
                    # Record gradient scale
                    if manager.trainer.scaler:
                        gradient_scales.append(manager.trainer.scaler.get_scale())
                    
                    # Check for numerical errors
                    if torch.isnan(result['loss']) or torch.isinf(result['loss']):
                        numerical_errors.append(batch_idx)
                    
                except Exception as e:
                    logger.error(f"Training step failed in {strategy_name}: {e}")
                    numerical_errors.append(batch_idx)
            
            # Get final stats
            stats = manager.get_training_stats()
            stats['gradient_scales'] = gradient_scales
            stats['losses'] = losses
            stats['numerical_errors'] = numerical_errors
            stats['strategy_name'] = strategy_name
            
            strategy_results[strategy_name] = stats
            manager.cleanup()
        
        return strategy_results
    
    async def run_memory_optimization_demo(self) -> Dict:
        """Demonstrate memory optimization with mixed precision."""
        logger.info("Starting Memory Optimization Demo")
        
        config = MixedPrecisionConfig(
            enabled=True,
            precision_mode=PrecisionMode.MIXED,
            scaling_strategy=ScalingStrategy.PERFORMANCE_OPTIMIZED,
            memory_efficient=True,
            enable_monitoring=True
        )
        
        manager = AdvancedMixedPrecisionManager(config)
        model = LargeTransformerModel(input_dim=1024, hidden_dim=1536, num_layers=8)
        dataset = LargeDataset(3000, input_dim=1024)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        
        # Memory tracking
        memory_usage = []
        peak_memory = 0
        memory_savings = []
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 60:  # Limit training steps
                break
            
            # Record memory before training step
            if torch.cuda.is_available():
                memory_before = torch.cuda.memory_allocated() / 1024**3
                
                result = manager.trainer.train_step(batch, model, optimizer)
                
                memory_after = torch.cuda.memory_allocated() / 1024**3
                memory_usage.append(memory_after)
                peak_memory = max(peak_memory, memory_after)
                
                # Calculate memory savings compared to FP32
                fp32_estimate = memory_before * 2  # Rough estimate
                savings = (fp32_estimate - memory_after) / fp32_estimate
                memory_savings.append(savings)
        
        stats = manager.get_training_stats()
        stats['memory_usage'] = memory_usage
        stats['peak_memory_gb'] = peak_memory
        stats['memory_savings'] = memory_savings
        stats['avg_memory_savings'] = np.mean(memory_savings) if memory_savings else 0.0
        
        manager.cleanup()
        return stats
    
    async def run_numerical_stability_demo(self) -> Dict:
        """Demonstrate numerical stability with mixed precision."""
        logger.info("Starting Numerical Stability Demo")
        
        config = MixedPrecisionConfig(
            enabled=True,
            precision_mode=PrecisionMode.MIXED,
            scaling_strategy=ScalingStrategy.ADAPTIVE,
            automatic_fallback=True,
            fallback_threshold=1e-6,
            enable_monitoring=True
        )
        
        manager = AdvancedMixedPrecisionManager(config)
        model = LargeTransformerModel(input_dim=256, hidden_dim=384, num_layers=6)
        dataset = LargeDataset(1000, input_dim=256)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)  # Higher learning rate to test stability
        
        # Stability tracking
        losses = []
        gradient_scales = []
        fallback_events = []
        numerical_errors = []
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 120:  # More steps to test stability
                break
            
            try:
                result = manager.trainer.train_step(batch, model, optimizer)
                
                losses.append(result['loss'])
                
                if manager.trainer.scaler:
                    gradient_scales.append(manager.trainer.scaler.get_scale())
                
                # Check for numerical issues
                if torch.isnan(result['loss']) or torch.isinf(result['loss']):
                    numerical_errors.append(batch_idx)
                
            except Exception as e:
                logger.warning(f"Numerical error at step {batch_idx}: {e}")
                numerical_errors.append(batch_idx)
                fallback_events.append(batch_idx)
        
        stats = manager.get_training_stats()
        stats['losses'] = losses
        stats['gradient_scales'] = gradient_scales
        stats['numerical_errors'] = numerical_errors
        stats['fallback_events'] = fallback_events
        stats['stability_rate'] = 1.0 - (len(numerical_errors) / len(losses)) if losses else 0.0
        
        manager.cleanup()
        return stats
    
    async def run_performance_profiling_demo(self) -> Dict:
        """Demonstrate performance profiling with mixed precision."""
        logger.info("Starting Performance Profiling Demo")
        
        config = MixedPrecisionConfig(
            enabled=True,
            precision_mode=PrecisionMode.MIXED,
            scaling_strategy=ScalingStrategy.PERFORMANCE_OPTIMIZED,
            enable_monitoring=True,
            profile_precision=True
        )
        
        manager = AdvancedMixedPrecisionManager(config)
        model = LargeTransformerModel(input_dim=640, hidden_dim=1024, num_layers=5)
        dataset = LargeDataset(2500, input_dim=640)
        dataloader = DataLoader(dataset, batch_size=40, shuffle=True)
        
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        
        # Performance profiling
        step_times = []
        throughput_history = []
        memory_efficiency = []
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 100:  # Limit training steps
                break
            
            step_start = time.time()
            
            result = manager.trainer.train_step(batch, model, optimizer)
            
            step_time = time.time() - step_start
            step_times.append(step_time)
            
            # Calculate throughput (samples per second)
            batch_size = batch['input_ids'].size(0)
            throughput = batch_size / step_time
            throughput_history.append(throughput)
            
            # Calculate memory efficiency
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                efficiency = memory_used / memory_total
                memory_efficiency.append(efficiency)
        
        total_time = time.time() - start_time
        
        stats = manager.get_training_stats()
        stats['step_times'] = step_times
        stats['throughput_history'] = throughput_history
        stats['memory_efficiency'] = memory_efficiency
        stats['total_training_time'] = total_time
        stats['avg_throughput'] = np.mean(throughput_history) if throughput_history else 0.0
        stats['avg_memory_efficiency'] = np.mean(memory_efficiency) if memory_efficiency else 0.0
        
        manager.cleanup()
        return stats
    
    async def run_comprehensive_demo(self) -> Dict:
        """Run comprehensive mixed precision training demo."""
        logger.info("Starting Comprehensive Mixed Precision Training Demo")
        
        results = {}
        
        try:
            # Check CUDA availability
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, skipping GPU demos")
                return {'error': 'CUDA not available'}
            
            # Run individual demos
            results['precision_comparison'] = await self.run_fp32_vs_fp16_comparison()
            results['scaling_strategy_comparison'] = await self.run_scaling_strategy_comparison()
            results['memory_optimization'] = await self.run_memory_optimization_demo()
            results['numerical_stability'] = await self.run_numerical_stability_demo()
            results['performance_profiling'] = await self.run_performance_profiling_demo()
            
            # Generate comparison report
            comparison = self._generate_comparison_report(results)
            results['comparison'] = comparison
            
            # Save results
            self._save_results(results)
            
            # Plot results
            self.plot_results(results)
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def _generate_comparison_report(self, results: Dict) -> Dict:
        """Generate comparison report between different approaches."""
        report = {
            'precision_comparison': {},
            'scaling_strategy_comparison': {},
            'recommendations': []
        }
        
        # Analyze precision comparison
        if 'precision_comparison' in results:
            precision_results = results['precision_comparison']
            for mode_name, stats in precision_results.items():
                report['precision_comparison'][mode_name] = {
                    'avg_loss': np.mean(stats.get('losses', [0])),
                    'avg_step_time': np.mean(stats.get('step_times', [0])),
                    'memory_usage': np.mean(stats.get('memory_usage', [0])),
                    'training_time': stats.get('training_time', 0)
                }
        
        # Analyze scaling strategy comparison
        if 'scaling_strategy_comparison' in results:
            strategy_results = results['scaling_strategy_comparison']
            for strategy_name, stats in strategy_results.items():
                report['scaling_strategy_comparison'][strategy_name] = {
                    'avg_loss': np.mean(stats.get('losses', [0])),
                    'numerical_errors': len(stats.get('numerical_errors', [])),
                    'avg_gradient_scale': np.mean(stats.get('gradient_scales', [1]))
                }
        
        # Generate recommendations
        if 'precision_comparison' in results:
            precision_results = results['precision_comparison']
            if 'Mixed Precision Training' in precision_results:
                mixed_stats = precision_results['Mixed Precision Training']
                if mixed_stats.get('avg_step_time', float('inf')) < 0.1:
                    report['recommendations'].append(
                        "Mixed precision training shows significant performance improvements"
                    )
        
        if 'memory_optimization' in results:
            memory_stats = results['memory_optimization']
            if memory_stats.get('avg_memory_savings', 0) > 0.3:
                report['recommendations'].append(
                    "Mixed precision provides substantial memory savings (>30%)"
                )
        
        if 'numerical_stability' in results:
            stability_stats = results['numerical_stability']
            if stability_stats.get('stability_rate', 0) > 0.95:
                report['recommendations'].append(
                    "Mixed precision maintains excellent numerical stability"
                )
        
        return report
    
    def plot_results(self, results: Dict, save_path: str = "mixed_precision_results.png"):
        """Plot comprehensive results."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Precision comparison - Loss convergence
        if 'precision_comparison' in results:
            precision_results = results['precision_comparison']
            for mode_name, stats in precision_results.items():
                losses = stats.get('losses', [])
                if losses:
                    axes[0, 0].plot(losses, label=mode_name)
            
            axes[0, 0].set_title('Loss Convergence Comparison')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
        
        # Plot 2: Precision comparison - Step times
        if 'precision_comparison' in results:
            precision_results = results['precision_comparison']
            for mode_name, stats in precision_results.items():
                step_times = stats.get('step_times', [])
                if step_times:
                    axes[0, 1].plot(step_times, label=mode_name)
            
            axes[0, 1].set_title('Step Time Comparison')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Time (seconds)')
            axes[0, 1].legend()
        
        # Plot 3: Memory usage comparison
        if 'precision_comparison' in results:
            precision_results = results['precision_comparison']
            for mode_name, stats in precision_results.items():
                memory_usage = stats.get('memory_usage', [])
                if memory_usage:
                    axes[0, 2].plot(memory_usage, label=mode_name)
            
            axes[0, 2].set_title('Memory Usage Comparison')
            axes[0, 2].set_xlabel('Step')
            axes[0, 2].set_ylabel('Memory (GB)')
            axes[0, 2].legend()
        
        # Plot 4: Gradient scaling strategies
        if 'scaling_strategy_comparison' in results:
            strategy_results = results['scaling_strategy_comparison']
            for strategy_name, stats in strategy_results.items():
                gradient_scales = stats.get('gradient_scales', [])
                if gradient_scales:
                    axes[1, 0].plot(gradient_scales, label=strategy_name)
            
            axes[1, 0].set_title('Gradient Scaling Strategies')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Gradient Scale')
            axes[1, 0].legend()
        
        # Plot 5: Performance profiling
        if 'performance_profiling' in results:
            profiling_stats = results['performance_profiling']
            throughput = profiling_stats.get('throughput_history', [])
            if throughput:
                axes[1, 1].plot(throughput)
                axes[1, 1].set_title('Training Throughput')
                axes[1, 1].set_xlabel('Step')
                axes[1, 1].set_ylabel('Samples/Second')
        
        # Plot 6: Memory efficiency
        if 'performance_profiling' in results:
            profiling_stats = results['performance_profiling']
            memory_efficiency = profiling_stats.get('memory_efficiency', [])
            if memory_efficiency:
                axes[1, 2].plot(memory_efficiency)
                axes[1, 2].set_title('Memory Efficiency')
                axes[1, 2].set_xlabel('Step')
                axes[1, 2].set_ylabel('Efficiency')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Results plot saved to {save_path}")
    
    def _save_results(self, results: Dict):
        """Save demo results to file."""
        output_path = Path("mixed_precision_training_results.json")
        
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
    logger.info("Advanced Mixed Precision Training Demo")
    
    # Create demo instance
    demo = MixedPrecisionTrainingDemo()
    
    # Run comprehensive demo
    results = await demo.run_comprehensive_demo()
    
    # Print summary
    logger.info("Demo completed successfully!")
    
    if 'comparison' in results:
        comparison = results['comparison']
        logger.info("Performance Summary:")
        
        if 'precision_comparison' in comparison:
            for mode_name, metrics in comparison['precision_comparison'].items():
                logger.info(
                    f"{mode_name}: Avg Loss = {metrics['avg_loss']:.4f}, "
                    f"Avg Step Time = {metrics['avg_step_time']:.4f}s, "
                    f"Memory = {metrics['memory_usage']:.2f}GB"
                )
        
        if 'recommendations' in comparison:
            logger.info("Recommendations:")
            for rec in comparison['recommendations']:
                logger.info(f"- {rec}")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main()) 