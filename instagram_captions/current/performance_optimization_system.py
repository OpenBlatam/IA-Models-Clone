"""
Performance Optimization System
Implements comprehensive performance optimization with best practices for deep learning workflows
"""

import torch
import torch.nn as nn
import torch.nn.parallel as parallel
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.cuda.amp import autocast, GradScaler
import time
import cProfile
import pstats
import io
import psutil
import gc
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from pathlib import Path
import logging
import warnings
from dataclasses import dataclass
from contextlib import contextmanager
import numpy as np

@dataclass
class OptimizationConfig:
    """Configuration for performance optimization"""
    enable_mixed_precision: bool = True
    enable_gradient_accumulation: bool = True
    enable_data_parallel: bool = False
    enable_distributed: bool = False
    enable_gradient_checkpointing: bool = False
    enable_torch_compile: bool = False
    accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    memory_efficient_attention: bool = True
    compile_mode: str = "reduce-overhead"  # "reduce-overhead", "max-autotune"

class PerformanceOptimizer:
    """Comprehensive performance optimization for deep learning workflows"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.scaler = None
        self.optimization_stats = {}
        
        # Setup mixed precision if enabled
        if config.enable_mixed_precision:
            self._setup_mixed_precision()
        
        # Setup distributed training if enabled
        if config.enable_distributed:
            self._setup_distributed()
    
    def _setup_mixed_precision(self):
        """Setup mixed precision training"""
        if torch.cuda.is_available():
            self.scaler = GradScaler()
            self.logger.info("Mixed precision training enabled with GradScaler")
        else:
            self.logger.warning("Mixed precision requested but CUDA not available")
            self.config.enable_mixed_precision = False
    
    def _setup_distributed(self):
        """Setup distributed training"""
        if not dist.is_initialized():
            self.logger.warning("Distributed training requested but not initialized")
            self.config.enable_distributed = False
        else:
            self.logger.info(f"Distributed training enabled on rank {dist.get_rank()}")
    
    def optimize_model(self, model: nn.Module, device: str = "cuda") -> nn.Module:
        """Apply comprehensive model optimizations"""
        if not torch.cuda.is_available():
            device = "cpu"
            self.logger.warning("CUDA not available, using CPU")
        
        # Move model to device
        model = model.to(device)
        
        # Apply gradient checkpointing if enabled
        if self.config.enable_gradient_checkpointing:
            model.gradient_checkpointing_enable()
            self.logger.info("Gradient checkpointing enabled")
        
        # Apply torch.compile if enabled
        if self.config.enable_torch_compile and hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode=self.config.compile_mode)
                self.logger.info(f"Torch compile enabled with mode: {self.config.compile_mode}")
            except Exception as e:
                self.logger.warning(f"Torch compile failed: {e}")
        
        # Apply DataParallel if enabled
        if self.config.enable_data_parallel and torch.cuda.device_count() > 1:
            model = DataParallel(model)
            self.logger.info(f"DataParallel enabled on {torch.cuda.device_count()} GPUs")
        
        # Apply DistributedDataParallel if enabled
        if self.config.enable_distributed:
            model = DistributedDataParallel(model)
            self.logger.info("DistributedDataParallel enabled")
        
        return model
    
    def optimize_optimizer(self, optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
        """Apply optimizer optimizations"""
        # Enable automatic mixed precision for optimizer
        if self.config.enable_mixed_precision and self.scaler:
            # The scaler will handle mixed precision automatically
            pass
        
        return optimizer
    
    def training_step(self, model: nn.Module, data: torch.Tensor, target: torch.Tensor,
                     criterion: nn.Module, optimizer: torch.optim.Optimizer,
                     accumulation_step: int = 0) -> Dict[str, Any]:
        """Optimized training step with all optimizations"""
        start_time = time.time()
        
        # Forward pass with mixed precision if enabled
        if self.config.enable_mixed_precision and self.scaler:
            with autocast():
                output = model(data)
                loss = criterion(output, target)
            
            # Scale loss for gradient accumulation
            if self.config.enable_gradient_accumulation:
                loss = loss / self.config.accumulation_steps
            
            # Backward pass with scaler
            self.scaler.scale(loss).backward()
            
            # Gradient accumulation logic
            if self.config.enable_gradient_accumulation:
                if (accumulation_step + 1) % self.config.accumulation_steps == 0:
                    # Unscale gradients for clipping
                    self.scaler.unscale_(optimizer)
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                    
                    # Optimizer step with scaler
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad()
            else:
                # Unscale gradients for clipping
                self.scaler.unscale_(optimizer)
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                
                # Optimizer step with scaler
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()
        else:
            # Standard precision training
            output = model(data)
            loss = criterion(output, target)
            
            # Scale loss for gradient accumulation
            if self.config.enable_gradient_accumulation:
                loss = loss / self.config.accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation logic
            if self.config.enable_gradient_accumulation:
                if (accumulation_step + 1) % self.config.accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                    
                    # Optimizer step
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                
                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()
        
        # Calculate metrics
        step_time = time.time() - start_time
        accuracy = self._calculate_accuracy(output, target)
        
        # Update optimization stats
        self._update_stats(step_time, loss.item(), accuracy)
        
        return {
            "loss": loss.item(),
            "accuracy": accuracy,
            "step_time": step_time,
            "accumulation_step": accumulation_step
        }
    
    def _calculate_accuracy(self, output: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate accuracy for classification tasks"""
        if output.dim() > 1:
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            return 100. * correct / target.size(0)
        return 0.0
    
    def _update_stats(self, step_time: float, loss: float, accuracy: float):
        """Update optimization statistics"""
        if "step_times" not in self.optimization_stats:
            self.optimization_stats = {
                "step_times": [],
                "losses": [],
                "accuracies": []
            }
        
        self.optimization_stats["step_times"].append(step_time)
        self.optimization_stats["losses"].append(loss)
        self.optimization_stats["accuracies"].append(accuracy)
        
        # Keep only last 1000 entries
        max_entries = 1000
        for key in self.optimization_stats:
            if len(self.optimization_stats[key]) > max_entries:
                self.optimization_stats[key] = self.optimization_stats[key][-max_entries:]
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization performance"""
        if not self.optimization_stats:
            return {}
        
        summary = {}
        
        # Step time statistics
        if self.optimization_stats["step_times"]:
            times = self.optimization_stats["step_times"]
            summary["step_time"] = {
                "mean": np.mean(times),
                "std": np.std(times),
                "min": np.min(times),
                "max": np.max(times),
                "total": np.sum(times)
            }
        
        # Loss statistics
        if self.optimization_stats["losses"]:
            losses = self.optimization_stats["losses"]
            summary["loss"] = {
                "mean": np.mean(losses),
                "std": np.std(losses),
                "min": np.min(losses),
                "max": np.max(losses),
                "latest": losses[-1]
            }
        
        # Accuracy statistics
        if self.optimization_stats["accuracies"]:
            accuracies = self.optimization_stats["accuracies"]
            summary["accuracy"] = {
                "mean": np.mean(accuracies),
                "std": np.std(accuracies),
                "min": np.min(accuracies),
                "max": np.max(accuracies),
                "latest": accuracies[-1]
            }
        
        # System information
        summary["system"] = self._get_system_info()
        
        return summary
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get current system information"""
        info = {
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None
        }
        
        if torch.cuda.is_available():
            info.update({
                "device_name": torch.cuda.get_device_name(),
                "memory_allocated": torch.cuda.memory_allocated(),
                "memory_reserved": torch.cuda.memory_reserved(),
                "memory_cached": torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
            })
        
        # CPU and memory info
        try:
            info.update({
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_available": psutil.virtual_memory().available
            })
        except Exception as e:
            self.logger.warning(f"Failed to get system info: {e}")
        
        return info

class CodeProfiler:
    """Advanced code profiling for performance optimization"""
    
    def __init__(self, enable_profiling: bool = True):
        self.enable_profiling = enable_profiling
        self.logger = logging.getLogger(__name__)
        self.profiler = None
        self.profile_results = {}
    
    @contextmanager
    def profile_function(self, function_name: str):
        """Context manager for profiling functions"""
        if not self.enable_profiling:
            yield
            return
        
        # Start profiling
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            yield
        finally:
            # Stop profiling
            profiler.disable()
            
            # Get stats
            s = io.StringIO()
            stats = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            stats.print_stats(20)  # Top 20 functions
            
            # Store results
            self.profile_results[function_name] = {
                "stats": stats,
                "output": s.getvalue(),
                "total_calls": stats.total_calls,
                "total_time": stats.total_tt
            }
            
            self.logger.info(f"Profiled {function_name}: {stats.total_calls} calls, {stats.total_tt:.4f}s")
    
    def profile_torch_operations(self, model: nn.Module, input_tensor: torch.Tensor):
        """Profile PyTorch operations with torch.profiler"""
        if not torch.cuda.is_available():
            self.logger.warning("CUDA not available for torch.profiler")
            return
        
        try:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                with_stack=True,
                profile_memory=True
            ) as prof:
                # Forward pass
                with torch.no_grad():
                    output = model(input_tensor)
                
                # Backward pass (if needed)
                if input_tensor.requires_grad:
                    loss = output.sum()
                    loss.backward()
            
            # Print results
            prof.export_chrome_trace("trace.json")
            self.logger.info("PyTorch profiling completed, trace exported to trace.json")
            
            # Get key metrics
            key_averages = prof.key_averages()
            for avg in key_averages:
                if avg.key in ["aten::conv2d", "aten::linear", "aten::relu"]:
                    self.logger.info(f"{avg.key}: {avg.cpu_time_total/1000:.2f}ms CPU, "
                                   f"{avg.cuda_time_total/1000:.2f}ms CUDA")
        
        except Exception as e:
            self.logger.error(f"PyTorch profiling failed: {e}")
    
    def get_profile_summary(self) -> Dict[str, Any]:
        """Get summary of profiling results"""
        summary = {}
        
        for func_name, results in self.profile_results.items():
            summary[func_name] = {
                "total_calls": results["total_calls"],
                "total_time": results["total_time"],
                "avg_time_per_call": results["total_time"] / results["total_calls"] if results["total_calls"] > 0 else 0
            }
        
        return summary

class MemoryOptimizer:
    """Memory optimization utilities"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def clear_gpu_cache(self):
        """Clear GPU memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.info("GPU cache cleared")
    
    def clear_cpu_cache(self):
        """Clear CPU memory cache"""
        gc.collect()
        self.logger.info("CPU cache cleared")
    
    def clear_all_caches(self):
        """Clear all memory caches"""
        self.clear_gpu_cache()
        self.clear_cpu_cache()
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage"""
        memory_info = {}
        
        # GPU memory
        if torch.cuda.is_available():
            memory_info["gpu"] = {
                "allocated": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved(),
                "cached": torch.cuda.memory_reserved() - torch.cuda.memory_allocated(),
                "max_allocated": torch.cuda.max_memory_allocated()
            }
        
        # CPU memory
        try:
            vm = psutil.virtual_memory()
            memory_info["cpu"] = {
                "total": vm.total,
                "available": vm.available,
                "used": vm.used,
                "percent": vm.percent
            }
        except Exception as e:
            self.logger.warning(f"Failed to get CPU memory info: {e}")
        
        return memory_info
    
    def optimize_memory_usage(self, model: nn.Module, input_tensor: torch.Tensor):
        """Apply memory optimizations"""
        if not torch.cuda.is_available():
            return
        
        # Enable gradient checkpointing if not already enabled
        if not hasattr(model, 'gradient_checkpointing_enable'):
            return
        
        if not model.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            self.logger.info("Gradient checkpointing enabled for memory optimization")
        
        # Use memory efficient attention if available
        if hasattr(model, 'config') and hasattr(model.config, 'attention_mode'):
            if model.config.attention_mode != "flash_attention_2":
                try:
                    model.config.attention_mode = "flash_attention_2"
                    self.logger.info("Memory efficient attention enabled")
                except Exception as e:
                    self.logger.warning(f"Failed to enable memory efficient attention: {e}")

class TrainingOptimizer:
    """High-level training optimization orchestrator"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.performance_optimizer = PerformanceOptimizer(config)
        self.code_profiler = CodeProfiler(enable_profiling=True)
        self.memory_optimizer = MemoryOptimizer()
        self.logger = logging.getLogger(__name__)
    
    def optimize_training_loop(self, model: nn.Module, train_loader, val_loader,
                             criterion: nn.Module, optimizer: torch.optim.Optimizer,
                             num_epochs: int, device: str = "cuda") -> Dict[str, Any]:
        """Optimized training loop with all optimizations"""
        
        # Apply model optimizations
        model = self.performance_optimizer.optimize_model(model, device)
        optimizer = self.performance_optimizer.optimize_optimizer(optimizer)
        
        # Training loop
        training_stats = {
            "epochs": [],
            "train_losses": [],
            "val_losses": [],
            "train_accuracies": [],
            "val_accuracies": [],
            "epoch_times": []
        }
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_metrics = self._train_epoch(model, train_loader, criterion, optimizer)
            
            # Validation phase
            val_metrics = self._validate_epoch(model, val_loader, criterion)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Store metrics
            training_stats["epochs"].append(epoch + 1)
            training_stats["train_losses"].append(train_metrics["loss"])
            training_stats["val_losses"].append(val_metrics["loss"])
            training_stats["train_accuracies"].append(train_metrics["accuracy"])
            training_stats["val_accuracies"].append(val_metrics["accuracy"])
            training_stats["epoch_times"].append(epoch_time)
            
            # Log progress
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs} - "
                           f"Train Loss: {train_metrics['loss']:.4f}, "
                           f"Train Acc: {train_metrics['accuracy']:.2f}%, "
                           f"Val Loss: {val_metrics['loss']:.4f}, "
                           f"Val Acc: {val_metrics['accuracy']:.2f}%, "
                           f"Time: {epoch_time:.2f}s")
            
            # Memory optimization
            if epoch % 5 == 0:  # Every 5 epochs
                self.memory_optimizer.clear_all_caches()
        
        # Get optimization summary
        optimization_summary = self.performance_optimizer.get_optimization_summary()
        profile_summary = self.code_profiler.get_profile_summary()
        memory_summary = self.memory_optimizer.get_memory_usage()
        
        return {
            "training_stats": training_stats,
            "optimization_summary": optimization_summary,
            "profile_summary": profile_summary,
            "memory_summary": memory_summary
        }
    
    def _train_epoch(self, model: nn.Module, train_loader, criterion: nn.Module,
                     optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """Optimized training epoch"""
        model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Profile training step
            with self.code_profiler.profile_function(f"train_step_{batch_idx}"):
                metrics = self.performance_optimizer.training_step(
                    model, data, target, criterion, optimizer, batch_idx
                )
            
            total_loss += metrics["loss"]
            total_accuracy += metrics["accuracy"]
            num_batches += 1
        
        return {
            "loss": total_loss / num_batches,
            "accuracy": total_accuracy / num_batches
        }
    
    def _validate_epoch(self, model: nn.Module, val_loader, criterion: nn.Module) -> Dict[str, float]:
        """Validation epoch"""
        model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                loss = criterion(output, target)
                
                # Calculate accuracy
                if output.dim() > 1:
                    pred = output.argmax(dim=1, keepdim=True)
                    correct = pred.eq(target.view_as(pred)).sum().item()
                    accuracy = 100. * correct / target.size(0)
                else:
                    accuracy = 0.0
                
                total_loss += loss.item()
                total_accuracy += accuracy
                num_batches += 1
        
        return {
            "loss": total_loss / num_batches,
            "accuracy": total_accuracy / num_batches
        }

def main():
    """Example usage of the performance optimization system"""
    
    # Configuration
    config = OptimizationConfig(
        enable_mixed_precision=True,
        enable_gradient_accumulation=True,
        enable_data_parallel=True,
        enable_gradient_checkpointing=True,
        enable_torch_compile=True,
        accumulation_steps=4,
        max_grad_norm=1.0
    )
    
    # Initialize training optimizer
    training_optimizer = TrainingOptimizer(config)
    
    # Example model
    model = nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 10)
    )
    
    # Example data (dummy)
    class DummyDataset:
        def __init__(self, size=1000):
            self.data = torch.randn(size, 784)
            self.targets = torch.randint(0, 10, (size,))
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx], self.targets[idx]
    
    # Create data loaders
    train_dataset = DummyDataset(1000)
    val_dataset = DummyDataset(200)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Training components
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Run optimized training
    print("Starting optimized training...")
    results = training_optimizer.optimize_training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=5,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Print results
    print("\n=== Training Results ===")
    print(f"Final Train Loss: {results['training_stats']['train_losses'][-1]:.4f}")
    print(f"Final Train Accuracy: {results['training_stats']['train_accuracies'][-1]:.2f}%")
    print(f"Final Val Loss: {results['training_stats']['val_losses'][-1]:.4f}")
    print(f"Final Val Accuracy: {results['training_stats']['val_accuracies'][-1]:.2f}%")
    
    print("\n=== Optimization Summary ===")
    opt_summary = results['optimization_summary']
    if 'step_time' in opt_summary:
        print(f"Average Step Time: {opt_summary['step_time']['mean']:.4f}s")
        print(f"Total Training Time: {opt_summary['step_time']['total']:.2f}s")
    
    print("\n=== Memory Usage ===")
    mem_summary = results['memory_summary']
    if 'gpu' in mem_summary:
        gpu_mem = mem_summary['gpu']
        print(f"GPU Memory Allocated: {gpu_mem['allocated'] / 1024**3:.2f} GB")
        print(f"GPU Memory Reserved: {gpu_mem['reserved'] / 1024**3:.2f} GB")

if __name__ == "__main__":
    main()


