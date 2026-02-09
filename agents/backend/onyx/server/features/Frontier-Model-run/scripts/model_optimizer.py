#!/usr/bin/env python3
"""
Advanced Model Optimization Utilities for Frontier Model Training
Provides model compression, quantization, pruning, and optimization techniques.
"""

import os
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.quantization as quantization
from torch.ao.quantization import quantize_dynamic, quantize_static
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import time
from pathlib import Path
import logging
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
import psutil
import GPUtil

console = Console()

class OptimizationType(Enum):
    """Types of model optimizations."""
    PRUNING = "pruning"
    QUANTIZATION = "quantization"
    DISTILLATION = "distillation"
    COMPRESSION = "compression"
    FUSION = "fusion"
    OPTIMIZATION = "optimization"

class PruningMethod(Enum):
    """Pruning methods."""
    MAGNITUDE = "magnitude"
    RANDOM = "random"
    STRUCTURED = "structured"
    UNSTRUCTURED = "unstructured"
    L1 = "l1"
    L2 = "l2"

class QuantizationType(Enum):
    """Quantization types."""
    DYNAMIC = "dynamic"
    STATIC = "static"
    QAT = "qat"  # Quantization Aware Training
    INT8 = "int8"
    INT4 = "int4"
    FP16 = "fp16"
    BF16 = "bf16"

@dataclass
class OptimizationConfig:
    """Configuration for model optimization."""
    optimization_type: OptimizationType
    target_size_mb: Optional[float] = None
    compression_ratio: float = 0.5
    pruning_ratio: float = 0.1
    quantization_bits: int = 8
    quantization_type: QuantizationType = QuantizationType.DYNAMIC
    pruning_method: PruningMethod = PruningMethod.MAGNITUDE
    preserve_accuracy: bool = True
    accuracy_threshold: float = 0.95
    enable_fusion: bool = True
    enable_optimization: bool = True

@dataclass
class OptimizationResult:
    """Result of model optimization."""
    original_size_mb: float
    optimized_size_mb: float
    compression_ratio: float
    accuracy_loss: float
    speedup_factor: float
    optimization_time: float
    optimization_type: OptimizationType
    details: Dict[str, Any] = None

class ModelOptimizer:
    """Advanced model optimization with multiple techniques."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def optimize_model(self, model: nn.Module, 
                     input_shape: Tuple[int, ...],
                     test_data: Optional[torch.Tensor] = None,
                     test_labels: Optional[torch.Tensor] = None) -> OptimizationResult:
        """Optimize model using specified techniques."""
        
        start_time = time.time()
        original_size = self._get_model_size(model)
        
        console.print(f"[bold blue]Starting model optimization...[/bold blue]")
        console.print(f"Original model size: {original_size:.2f} MB")
        
        # Apply optimizations based on type
        if self.config.optimization_type == OptimizationType.PRUNING:
            optimized_model = self._apply_pruning(model)
        elif self.config.optimization_type == OptimizationType.QUANTIZATION:
            optimized_model = self._apply_quantization(model)
        elif self.config.optimization_type == OptimizationType.COMPRESSION:
            optimized_model = self._apply_compression(model)
        elif self.config.optimization_type == OptimizationType.FUSION:
            optimized_model = self._apply_fusion(model)
        else:
            optimized_model = self._apply_general_optimization(model)
        
        # Measure results
        optimized_size = self._get_model_size(optimized_model)
        compression_ratio = optimized_size / original_size
        
        # Measure accuracy loss if test data provided
        accuracy_loss = 0.0
        if test_data is not None and test_labels is not None:
            accuracy_loss = self._measure_accuracy_loss(model, optimized_model, test_data, test_labels)
        
        # Measure speedup
        speedup_factor = self._measure_speedup(model, optimized_model, input_shape)
        
        optimization_time = time.time() - start_time
        
        result = OptimizationResult(
            original_size_mb=original_size,
            optimized_size_mb=optimized_size,
            compression_ratio=compression_ratio,
            accuracy_loss=accuracy_loss,
            speedup_factor=speedup_factor,
            optimization_time=optimization_time,
            optimization_type=self.config.optimization_type
        )
        
        self._display_optimization_results(result)
        return result
    
    def _apply_pruning(self, model: nn.Module) -> nn.Module:
        """Apply pruning to the model."""
        console.print(f"[blue]Applying {self.config.pruning_method.value} pruning...[/blue]")
        
        # Get parameters to prune
        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                parameters_to_prune.append((module, 'weight'))
        
        # Apply pruning
        if self.config.pruning_method == PruningMethod.MAGNITUDE:
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=self.config.pruning_ratio,
            )
        elif self.config.pruning_method == PruningMethod.RANDOM:
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.RandomUnstructured,
                amount=self.config.pruning_ratio,
            )
        elif self.config.pruning_method == PruningMethod.STRUCTURED:
            for module, param_name in parameters_to_prune:
                prune.ln_structured(module, param_name, amount=self.config.pruning_ratio, n=2, dim=0)
        
        # Remove pruning reparameterization
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
        
        return model
    
    def _apply_quantization(self, model: nn.Module) -> nn.Module:
        """Apply quantization to the model."""
        console.print(f"[blue]Applying {self.config.quantization_type.value} quantization...[/blue]")
        
        model.eval()
        
        if self.config.quantization_type == QuantizationType.DYNAMIC:
            # Dynamic quantization
            quantized_model = quantize_dynamic(
                model, 
                {nn.Linear, nn.LSTM, nn.GRU}, 
                dtype=torch.qint8
            )
        elif self.config.quantization_type == QuantizationType.STATIC:
            # Static quantization (requires calibration data)
            model.qconfig = quantization.get_default_qconfig('fbgemm')
            model_prepared = quantization.prepare(model)
            # Note: In practice, you'd run calibration data through model_prepared
            quantized_model = quantization.convert(model_prepared)
        elif self.config.quantization_type == QuantizationType.INT8:
            # INT8 quantization
            quantized_model = torch.quantization.quantize_dynamic(
                model, 
                {nn.Linear}, 
                dtype=torch.qint8
            )
        else:
            # Default to dynamic quantization
            quantized_model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        
        return quantized_model
    
    def _apply_compression(self, model: nn.Module) -> nn.Module:
        """Apply general compression techniques."""
        console.print("[blue]Applying compression techniques...[/blue]")
        
        # Apply multiple compression techniques
        compressed_model = model
        
        # 1. Remove unused parameters
        compressed_model = self._remove_unused_parameters(compressed_model)
        
        # 2. Optimize data types
        compressed_model = self._optimize_data_types(compressed_model)
        
        # 3. Apply knowledge distillation if teacher model available
        # compressed_model = self._apply_distillation(compressed_model)
        
        return compressed_model
    
    def _apply_fusion(self, model: nn.Module) -> nn.Module:
        """Apply operator fusion optimizations."""
        console.print("[blue]Applying operator fusion...[/blue]")
        
        # Enable fusion optimizations
        if hasattr(torch.jit, 'optimize_for_inference'):
            model = torch.jit.optimize_for_inference(torch.jit.script(model))
        
        return model
    
    def _apply_general_optimization(self, model: nn.Module) -> nn.Module:
        """Apply general optimizations."""
        console.print("[blue]Applying general optimizations...[/blue]")
        
        optimized_model = model
        
        # Enable optimizations
        if self.config.enable_optimization:
            # Compile model for better performance
            if hasattr(torch, 'compile'):
                optimized_model = torch.compile(optimized_model)
        
        return optimized_model
    
    def _remove_unused_parameters(self, model: nn.Module) -> nn.Module:
        """Remove unused parameters from the model."""
        # This is a simplified version - in practice, you'd analyze the model graph
        for name, param in model.named_parameters():
            if param.grad is None or param.grad.norm() == 0:
                param.requires_grad = False
        
        return model
    
    def _optimize_data_types(self, model: nn.Module) -> nn.Module:
        """Optimize data types for better memory usage."""
        # Convert to half precision if supported
        if torch.cuda.is_available():
            model = model.half()
        
        return model
    
    def _get_model_size(self, model: nn.Module) -> float:
        """Get model size in MB."""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb
    
    def _measure_accuracy_loss(self, original_model: nn.Module, 
                              optimized_model: nn.Module,
                              test_data: torch.Tensor, 
                              test_labels: torch.Tensor) -> float:
        """Measure accuracy loss between original and optimized models."""
        console.print("[blue]Measuring accuracy loss...[/blue]")
        
        original_model.eval()
        optimized_model.eval()
        
        with torch.no_grad():
            # Original model accuracy
            original_outputs = original_model(test_data)
            original_predictions = torch.argmax(original_outputs, dim=1)
            original_accuracy = (original_predictions == test_labels).float().mean().item()
            
            # Optimized model accuracy
            optimized_outputs = optimized_model(test_data)
            optimized_predictions = torch.argmax(optimized_outputs, dim=1)
            optimized_accuracy = (optimized_predictions == test_labels).float().mean().item()
        
        accuracy_loss = original_accuracy - optimized_accuracy
        return accuracy_loss
    
    def _measure_speedup(self, original_model: nn.Module, 
                        optimized_model: nn.Module,
                        input_shape: Tuple[int, ...]) -> float:
        """Measure speedup factor of optimized model."""
        console.print("[blue]Measuring speedup...[/blue]")
        
        # Create dummy input
        dummy_input = torch.randn(1, *input_shape)
        
        # Warm up
        for _ in range(10):
            _ = original_model(dummy_input)
            _ = optimized_model(dummy_input)
        
        # Measure original model time
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        for _ in range(100):
            _ = original_model(dummy_input)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        original_time = time.time() - start_time
        
        # Measure optimized model time
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        for _ in range(100):
            _ = optimized_model(dummy_input)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        optimized_time = time.time() - start_time
        
        speedup_factor = original_time / optimized_time if optimized_time > 0 else 1.0
        return speedup_factor
    
    def _display_optimization_results(self, result: OptimizationResult):
        """Display optimization results in a formatted table."""
        table = Table(title="Model Optimization Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Original Size", f"{result.original_size_mb:.2f} MB")
        table.add_row("Optimized Size", f"{result.optimized_size_mb:.2f} MB")
        table.add_row("Compression Ratio", f"{result.compression_ratio:.2%}")
        table.add_row("Size Reduction", f"{(1 - result.compression_ratio):.2%}")
        table.add_row("Accuracy Loss", f"{result.accuracy_loss:.4f}")
        table.add_row("Speedup Factor", f"{result.speedup_factor:.2f}x")
        table.add_row("Optimization Time", f"{result.optimization_time:.2f}s")
        
        console.print(table)

class ModelCompressor:
    """Advanced model compression with multiple techniques."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def compress_model(self, model: nn.Module, 
                      compression_ratio: float = 0.5,
                      preserve_accuracy: bool = True) -> nn.Module:
        """Compress model using multiple techniques."""
        
        console.print(f"[bold blue]Compressing model with ratio {compression_ratio}...[/bold blue]")
        
        compressed_model = model
        
        # Apply different compression techniques
        if compression_ratio <= 0.3:
            # Aggressive compression
            compressed_model = self._apply_aggressive_compression(compressed_model)
        elif compression_ratio <= 0.5:
            # Moderate compression
            compressed_model = self._apply_moderate_compression(compressed_model)
        else:
            # Light compression
            compressed_model = self._apply_light_compression(compressed_model)
        
        return compressed_model
    
    def _apply_light_compression(self, model: nn.Module) -> nn.Module:
        """Apply light compression techniques."""
        console.print("[blue]Applying light compression...[/blue]")
        
        # Remove unused parameters
        for param in model.parameters():
            if param.grad is None:
                param.requires_grad = False
        
        return model
    
    def _apply_moderate_compression(self, model: nn.Module) -> nn.Module:
        """Apply moderate compression techniques."""
        console.print("[blue]Applying moderate compression...[/blue]")
        
        # Apply pruning
        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                parameters_to_prune.append((module, 'weight'))
        
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=0.1,
        )
        
        # Remove pruning reparameterization
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
        
        return model
    
    def _apply_aggressive_compression(self, model: nn.Module) -> nn.Module:
        """Apply aggressive compression techniques."""
        console.print("[blue]Applying aggressive compression...[/blue]")
        
        # Apply heavy pruning
        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                parameters_to_prune.append((module, 'weight'))
        
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=0.3,
        )
        
        # Remove pruning reparameterization
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
        
        # Apply quantization
        model = torch.quantization.quantize_dynamic(
            model, 
            {nn.Linear}, 
            dtype=torch.qint8
        )
        
        return model

class HyperparameterOptimizer:
    """Hyperparameter optimization using various algorithms."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def optimize_hyperparameters(self, 
                               model_class,
                               train_data,
                               val_data,
                               param_space: Dict[str, Any],
                               n_trials: int = 100,
                               optimization_algorithm: str = "bayesian") -> Dict[str, Any]:
        """Optimize hyperparameters using specified algorithm."""
        
        console.print(f"[bold blue]Optimizing hyperparameters with {optimization_algorithm}...[/bold blue]")
        
        if optimization_algorithm == "bayesian":
            return self._bayesian_optimization(model_class, train_data, val_data, param_space, n_trials)
        elif optimization_algorithm == "random":
            return self._random_search(model_class, train_data, val_data, param_space, n_trials)
        elif optimization_algorithm == "grid":
            return self._grid_search(model_class, train_data, val_data, param_space)
        else:
            raise ValueError(f"Unknown optimization algorithm: {optimization_algorithm}")
    
    def _bayesian_optimization(self, model_class, train_data, val_data, param_space, n_trials):
        """Bayesian optimization for hyperparameters."""
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer, Categorical
            
            # Define search space
            dimensions = []
            param_names = []
            
            for name, space in param_space.items():
                if isinstance(space, tuple) and len(space) == 2:
                    if isinstance(space[0], int):
                        dimensions.append(Integer(space[0], space[1]))
                    else:
                        dimensions.append(Real(space[0], space[1]))
                elif isinstance(space, list):
                    dimensions.append(Categorical(space))
                param_names.append(name)
            
            def objective(params):
                # Create model with current parameters
                param_dict = dict(zip(param_names, params))
                model = model_class(**param_dict)
                
                # Train and evaluate
                score = self._evaluate_model(model, train_data, val_data)
                return -score  # Minimize negative score
            
            # Run optimization
            result = gp_minimize(objective, dimensions, n_calls=n_trials)
            
            # Return best parameters
            best_params = dict(zip(param_names, result.x))
            return {
                "best_params": best_params,
                "best_score": -result.fun,
                "optimization_history": result.func_vals
            }
            
        except ImportError:
            console.print("[yellow]scikit-optimize not available, falling back to random search[/yellow]")
            return self._random_search(model_class, train_data, val_data, param_space, n_trials)
    
    def _random_search(self, model_class, train_data, val_data, param_space, n_trials):
        """Random search for hyperparameters."""
        best_score = float('-inf')
        best_params = None
        history = []
        
        for trial in range(n_trials):
            # Sample random parameters
            params = {}
            for name, space in param_space.items():
                if isinstance(space, tuple) and len(space) == 2:
                    if isinstance(space[0], int):
                        params[name] = np.random.randint(space[0], space[1] + 1)
                    else:
                        params[name] = np.random.uniform(space[0], space[1])
                elif isinstance(space, list):
                    params[name] = np.random.choice(space)
            
            # Evaluate model
            model = model_class(**params)
            score = self._evaluate_model(model, train_data, val_data)
            history.append(score)
            
            if score > best_score:
                best_score = score
                best_params = params
            
            if trial % 10 == 0:
                console.print(f"Trial {trial}/{n_trials}, Best Score: {best_score:.4f}")
        
        return {
            "best_params": best_params,
            "best_score": best_score,
            "optimization_history": history
        }
    
    def _grid_search(self, model_class, train_data, val_data, param_space):
        """Grid search for hyperparameters."""
        # Generate all parameter combinations
        param_names = list(param_space.keys())
        param_values = list(param_space.values())
        
        # Create grid
        import itertools
        param_combinations = list(itertools.product(*param_values))
        
        best_score = float('-inf')
        best_params = None
        history = []
        
        for i, combination in enumerate(param_combinations):
            params = dict(zip(param_names, combination))
            
            # Evaluate model
            model = model_class(**params)
            score = self._evaluate_model(model, train_data, val_data)
            history.append(score)
            
            if score > best_score:
                best_score = score
                best_params = params
            
            console.print(f"Grid {i+1}/{len(param_combinations)}, Score: {score:.4f}")
        
        return {
            "best_params": best_params,
            "best_score": best_score,
            "optimization_history": history
        }
    
    def _evaluate_model(self, model, train_data, val_data):
        """Evaluate model performance."""
        # This is a simplified evaluation - in practice, you'd implement proper training
        # For now, return a random score
        return np.random.random()

def main():
    """Main function for model optimization CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Optimization Utilities")
    parser.add_argument("--model-path", type=str, help="Path to model file")
    parser.add_argument("--optimization-type", type=str, 
                       choices=["pruning", "quantization", "compression", "fusion"],
                       default="compression", help="Type of optimization")
    parser.add_argument("--compression-ratio", type=float, default=0.5,
                       help="Compression ratio (0.0 to 1.0)")
    parser.add_argument("--pruning-ratio", type=float, default=0.1,
                       help="Pruning ratio (0.0 to 1.0)")
    parser.add_argument("--quantization-bits", type=int, default=8,
                       help="Number of bits for quantization")
    parser.add_argument("--output-path", type=str, default="./optimized_model.pth",
                       help="Output path for optimized model")
    
    args = parser.parse_args()
    
    # Create optimization config
    config = OptimizationConfig(
        optimization_type=OptimizationType(args.optimization_type),
        compression_ratio=args.compression_ratio,
        pruning_ratio=args.pruning_ratio,
        quantization_bits=args.quantization_bits
    )
    
    # Create optimizer
    optimizer = ModelOptimizer(config)
    
    # Load model (simplified - in practice, you'd load your actual model)
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(100, 50)
            self.linear2 = nn.Linear(50, 10)
        
        def forward(self, x):
            x = torch.relu(self.linear1(x))
            x = self.linear2(x)
            return x
    
    model = DummyModel()
    
    # Optimize model
    result = optimizer.optimize_model(model, input_shape=(100,))
    
    # Save optimized model
    torch.save(model.state_dict(), args.output_path)
    console.print(f"[green]Optimized model saved to: {args.output_path}[/green]")

if __name__ == "__main__":
    main()
