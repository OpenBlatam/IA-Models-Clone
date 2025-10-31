"""
Magnitude-based Pruning for TruthGPT API
========================================

TensorFlow-like magnitude-based pruning implementation.
"""

import torch
import torch.nn.utils.prune as prune
from typing import Any, Optional, Dict, List, Tuple
import numpy as np


class MagnitudePruning:
    """
    Magnitude-based pruning for models.
    
    Similar to tf.keras.pruning.magnitude_pruning, this class
    implements magnitude-based pruning for PyTorch models.
    """
    
    def __init__(self, 
                 sparsity: float = 0.5,
                 global_pruning: bool = True,
                 name: Optional[str] = None):
        """
        Initialize MagnitudePruning.
        
        Args:
            sparsity: Target sparsity (fraction of weights to prune)
            global_pruning: Whether to use global pruning
            name: Optional name for the pruner
        """
        self.sparsity = sparsity
        self.global_pruning = global_pruning
        self.name = name or "magnitude_pruning"
        
        self.pruned_model = None
        self.original_model = None
        self.pruning_masks = {}
    
    def prune(self, model: Any, 
              sparsity: Optional[float] = None) -> Any:
        """
        Prune a model.
        
        Args:
            model: Model to prune
            sparsity: Target sparsity (overrides initialization)
            
        Returns:
            Pruned model
        """
        if sparsity is not None:
            self.sparsity = sparsity
        
        print(f"🔧 Applying magnitude-based pruning...")
        print(f"   Target sparsity: {self.sparsity:.2%}")
        print(f"   Global pruning: {self.global_pruning}")
        
        # Store original model
        self.original_model = model
        
        # Create a copy for pruning
        pruned_model = model
        
        # Get parameters to prune
        parameters_to_prune = []
        for name, module in pruned_model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                parameters_to_prune.append((module, 'weight'))
        
        if not parameters_to_prune:
            print("⚠️ No parameters found for pruning")
            return model
        
        # Apply magnitude pruning
        if self.global_pruning:
            # Global pruning
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=self.sparsity
            )
        else:
            # Local pruning
            for module, param_name in parameters_to_prune:
                prune.l1_unstructured(module, param_name, amount=self.sparsity)
        
        # Make pruning permanent
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
        
        self.pruned_model = pruned_model
        
        # Calculate actual sparsity
        actual_sparsity = self._calculate_sparsity(pruned_model)
        
        print(f"✅ Magnitude-based pruning completed!")
        print(f"   Target sparsity: {self.sparsity:.2%}")
        print(f"   Actual sparsity: {actual_sparsity:.2%}")
        print(f"   Compression ratio: {1 / (1 - actual_sparsity):.2f}x")
        
        return pruned_model
    
    def _calculate_sparsity(self, model: Any) -> float:
        """Calculate actual sparsity of the model."""
        total_params = 0
        zero_params = 0
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                total_params += param.numel()
                zero_params += (param == 0).sum().item()
        
        return zero_params / total_params if total_params > 0 else 0.0
    
    def analyze_sparsity(self, model: Any) -> Dict[str, Any]:
        """
        Analyze sparsity of the model.
        
        Args:
            model: Model to analyze
            
        Returns:
            Sparsity analysis results
        """
        print(f"📊 Analyzing model sparsity...")
        
        layer_sparsity = {}
        total_params = 0
        total_zero_params = 0
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_count = param.numel()
                zero_count = (param == 0).sum().item()
                
                layer_sparsity[name] = {
                    'total_params': param_count,
                    'zero_params': zero_count,
                    'sparsity': zero_count / param_count if param_count > 0 else 0.0
                }
                
                total_params += param_count
                total_zero_params += zero_count
        
        overall_sparsity = total_zero_params / total_params if total_params > 0 else 0.0
        
        results = {
            'overall_sparsity': overall_sparsity,
            'total_params': total_params,
            'zero_params': total_zero_params,
            'layer_sparsity': layer_sparsity
        }
        
        print(f"✅ Sparsity analysis completed!")
        print(f"   Overall sparsity: {overall_sparsity:.2%}")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Zero parameters: {total_zero_params:,}")
        
        return results
    
    def benchmark(self, 
                 original_model: Any,
                 pruned_model: Any,
                 sample_input: torch.Tensor,
                 num_runs: int = 100) -> Dict[str, float]:
        """
        Benchmark pruned model.
        
        Args:
            original_model: Original model
            pruned_model: Pruned model
            sample_input: Sample input
            num_runs: Number of benchmark runs
            
        Returns:
            Benchmark results
        """
        print(f"📊 Benchmarking pruned model...")
        
        # Benchmark original model
        original_times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                _ = original_model(sample_input)
                end_time.record()
                
                torch.cuda.synchronize()
                original_times.append(start_time.elapsed_time(end_time))
        
        # Benchmark pruned model
        pruned_times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                _ = pruned_model(sample_input)
                end_time.record()
                
                torch.cuda.synchronize()
                pruned_times.append(start_time.elapsed_time(end_time))
        
        # Calculate statistics
        original_avg = sum(original_times) / len(original_times)
        pruned_avg = sum(pruned_times) / len(pruned_times)
        
        speedup = original_avg / pruned_avg if pruned_avg > 0 else 1.0
        
        # Calculate sparsity
        sparsity = self._calculate_sparsity(pruned_model)
        compression_ratio = 1 / (1 - sparsity) if sparsity < 1.0 else float('inf')
        
        results = {
            'original_time': original_avg,
            'pruned_time': pruned_avg,
            'speedup': speedup,
            'sparsity': sparsity,
            'compression_ratio': compression_ratio
        }
        
        print(f"✅ Benchmark completed!")
        print(f"   Original time: {original_avg:.2f} ms")
        print(f"   Pruned time: {pruned_avg:.2f} ms")
        print(f"   Speedup: {speedup:.2f}x")
        print(f"   Sparsity: {sparsity:.2%}")
        print(f"   Compression: {compression_ratio:.2f}x")
        
        return results
    
    def get_config(self) -> Dict[str, Any]:
        """Get pruning configuration."""
        return {
            'name': self.name,
            'sparsity': self.sparsity,
            'global_pruning': self.global_pruning,
            'type': 'magnitude'
        }
    
    def __repr__(self):
        return f"MagnitudePruning(sparsity={self.sparsity}, global_pruning={self.global_pruning})"









