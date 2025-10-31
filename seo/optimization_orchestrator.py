#!/usr/bin/env python3
"""
Optimization Orchestrator for SEO Evaluation System
Comprehensive optimization management integrating all optimization modules
"""

import torch
import torch.nn as nn
import torch.cuda.amp as amp
import logging
import time
import asyncio
import warnings
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager, asynccontextmanager
import functools
from pathlib import Path
import json
import pickle
import hashlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import psutil
import gc

# Import optimization modules
from memory_optimizer import MemoryOptimizer, MemoryConfig
from async_data_loader import AsyncDataLoader, AsyncDataConfig
from model_compiler import ModelCompiler, CompilationConfig

warnings.filterwarnings("ignore")

@dataclass
class OptimizationConfig:
    """Comprehensive optimization configuration."""
    # Memory Optimization
    memory_config: MemoryConfig = field(default_factory=MemoryConfig)
    
    # Data Loading Optimization
    data_config: AsyncDataConfig = field(default_factory=AsyncDataConfig)
    
    # Model Compilation
    compilation_config: CompilationConfig = field(default_factory=CompilationConfig)
    
    # System Optimization
    enable_system_optimization: bool = True
    enable_process_priority: bool = True
    enable_cpu_affinity: bool = True
    enable_memory_optimization: bool = True
    enable_data_optimization: bool = True
    enable_model_optimization: bool = True
    
    # Performance Monitoring
    enable_performance_monitoring: bool = True
    monitoring_interval: float = 1.0
    enable_auto_optimization: bool = True
    
    # Advanced Settings
    enable_adaptive_optimization: bool = True
    enable_dynamic_configuration: bool = True
    optimization_threshold: float = 0.1  # 10% improvement threshold

class OptimizationOrchestrator:
    """Comprehensive optimization orchestrator for SEO evaluation system."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize optimization modules
        self.memory_optimizer = MemoryOptimizer(config.memory_config) if config.enable_memory_optimization else None
        self.data_loader = AsyncDataLoader(config.data_config) if config.enable_data_optimization else None
        self.model_compiler = ModelCompiler(config.compilation_config) if config.enable_model_optimization else None
        
        # Performance tracking
        self.performance_history = []
        self.optimization_stats = {}
        
        # Initialize system optimization
        if self.config.enable_system_optimization:
            self._setup_system_optimization()
    
    def _setup_system_optimization(self):
        """Setup system-level optimizations."""
        try:
            if self.config.enable_process_priority:
                # Set high process priority
                import os
                os.nice(-10)  # Higher priority
                
            if self.config.enable_cpu_affinity:
                # Set CPU affinity for better performance
                import psutil
                process = psutil.Process()
                # Use all available cores
                process.cpu_affinity(list(range(psutil.cpu_count())))
                
            self.logger.info("System optimization completed")
            
        except Exception as e:
            self.logger.warning(f"System optimization failed: {e}")
    
    @contextmanager
    def optimization_context(self, model: nn.Module = None, dataset = None):
        """Context manager for comprehensive optimization."""
        try:
            # Pre-optimization setup
            if model and self.memory_optimizer:
                model = self.memory_optimizer.optimize_model_memory(model)
                
            if model and self.model_compiler:
                model = self.model_compiler.compile_model(model)
                
            # Start monitoring
            if self.config.enable_performance_monitoring:
                self._start_performance_monitoring()
                
            yield model
            
        finally:
            # Post-optimization cleanup
            if self.config.enable_performance_monitoring:
                self._stop_performance_monitoring()
                
            # Cleanup memory
            if self.memory_optimizer:
                self.memory_optimizer._cleanup_memory()
    
    async def optimize_training_pipeline(self, model: nn.Module, dataset, training_config: Dict[str, Any]) -> nn.Module:
        """Optimize complete training pipeline."""
        try:
            self.logger.info("Starting comprehensive training pipeline optimization")
            
            # Phase 1: Memory Optimization
            if self.memory_optimizer:
                self.logger.info("Phase 1: Memory optimization")
                model = await self._optimize_memory(model, dataset)
            
            # Phase 2: Model Compilation
            if self.model_compiler:
                self.logger.info("Phase 2: Model compilation")
                model = await self._optimize_model(model, dataset)
            
            # Phase 3: Data Loading Optimization
            if self.data_loader:
                self.logger.info("Phase 3: Data loading optimization")
                await self._optimize_data_loading(dataset)
            
            # Phase 4: Training Configuration Optimization
            self.logger.info("Phase 4: Training configuration optimization")
            optimized_config = await self._optimize_training_config(training_config)
            
            self.logger.info("Training pipeline optimization completed")
            return model, optimized_config
            
        except Exception as e:
            self.logger.error(f"Training pipeline optimization failed: {e}")
            return model, training_config
    
    async def _optimize_memory(self, model: nn.Module, dataset) -> nn.Module:
        """Optimize memory usage."""
        try:
            # Get optimal batch size
            if hasattr(dataset, '__getitem__'):
                sample_input = dataset[0] if len(dataset) > 0 else torch.randn(1, 512)
                optimal_batch_size = self.memory_optimizer.get_optimal_batch_size(model, sample_input)
                
                # Update data config with optimal batch size
                self.config.data_config.batch_size = optimal_batch_size
                self.logger.info(f"Optimized batch size: {optimal_batch_size}")
            
            # Optimize model memory
            model = self.memory_optimizer.optimize_model_memory(model)
            
            return model
            
        except Exception as e:
            self.logger.error(f"Memory optimization failed: {e}")
            return model
    
    async def _optimize_model(self, model: nn.Module, dataset) -> nn.Module:
        """Optimize model performance."""
        try:
            # Get sample input for compilation
            sample_input = None
            if hasattr(dataset, '__getitem__') and len(dataset) > 0:
                sample_input = dataset[0]
                if isinstance(sample_input, (list, tuple)):
                    sample_input = sample_input[0]
            
            # Compile model
            compiled_model = self.model_compiler.compile_model(model, sample_input)
            
            return compiled_model
            
        except Exception as e:
            self.logger.error(f"Model optimization failed: {e}")
            return model
    
    async def _optimize_data_loading(self, dataset):
        """Optimize data loading."""
        try:
            # Configure async data loading
            if self.data_loader:
                # Update configuration based on dataset size
                if hasattr(dataset, '__len__'):
                    dataset_size = len(dataset)
                    if dataset_size > 10000:
                        self.config.data_config.num_workers = min(8, mp.cpu_count())
                        self.config.data_config.prefetch_batches = 8
                    elif dataset_size > 1000:
                        self.config.data_config.num_workers = min(4, mp.cpu_count())
                        self.config.data_config.prefetch_batches = 4
                    else:
                        self.config.data_config.num_workers = 2
                        self.config.data_config.prefetch_batches = 2
                        
                self.logger.info(f"Data loading optimized with {self.config.data_config.num_workers} workers")
                
        except Exception as e:
            self.logger.error(f"Data loading optimization failed: {e}")
    
    async def _optimize_training_config(self, training_config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize training configuration."""
        try:
            optimized_config = training_config.copy()
            
            # Optimize learning rate based on batch size
            if 'learning_rate' in optimized_config:
                batch_size = self.config.data_config.batch_size
                base_lr = optimized_config['learning_rate']
                
                # Scale learning rate with batch size (linear scaling rule)
                if batch_size > 32:
                    scale_factor = batch_size / 32
                    optimized_config['learning_rate'] = base_lr * scale_factor
                    self.logger.info(f"Scaled learning rate to {optimized_config['learning_rate']:.6f}")
            
            # Optimize gradient accumulation
            if 'gradient_accumulation_steps' not in optimized_config:
                # Calculate optimal gradient accumulation steps
                target_batch_size = 32  # Target effective batch size
                effective_batch_size = self.config.data_config.batch_size
                if effective_batch_size > target_batch_size:
                    optimized_config['gradient_accumulation_steps'] = effective_batch_size // target_batch_size
                    self.logger.info(f"Set gradient accumulation steps to {optimized_config['gradient_accumulation_steps']}")
            
            # Enable mixed precision if not already enabled
            if 'fp16' not in optimized_config:
                optimized_config['fp16'] = True
                self.logger.info("Enabled mixed precision training")
            
            return optimized_config
            
        except Exception as e:
            self.logger.error(f"Training configuration optimization failed: {e}")
            return training_config
    
    def _start_performance_monitoring(self):
        """Start performance monitoring."""
        self.monitoring_start_time = time.time()
        self.initial_memory = self._get_memory_usage()
        
        if self.config.enable_performance_monitoring:
            # Start monitoring thread
            import threading
            self.monitoring_thread = threading.Thread(target=self._monitor_performance, daemon=True)
            self.monitoring_thread.start()
    
    def _stop_performance_monitoring(self):
        """Stop performance monitoring."""
        if hasattr(self, 'monitoring_thread'):
            self.monitoring_thread.join(timeout=1.0)
    
    def _monitor_performance(self):
        """Monitor performance metrics."""
        while hasattr(self, 'monitoring_start_time'):
            try:
                current_memory = self._get_memory_usage()
                current_time = time.time()
                
                performance_metric = {
                    'timestamp': current_time,
                    'memory_usage': current_memory,
                    'elapsed_time': current_time - self.monitoring_start_time
                }
                
                self.performance_history.append(performance_metric)
                
                # Keep history manageable
                if len(self.performance_history) > 1000:
                    self.performance_history.pop(0)
                
                time.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")
                time.sleep(self.config.monitoring_interval)
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            gpu_memory = {}
            if torch.cuda.is_available():
                gpu_memory = {
                    'gpu_allocated': torch.cuda.memory_allocated() / 1024**3,
                    'gpu_reserved': torch.cuda.memory_reserved() / 1024**3
                }
            
            return {
                'rss': memory_info.rss / 1024**3,
                'vms': memory_info.vms / 1024**3,
                **gpu_memory
            }
        except Exception as e:
            self.logger.error(f"Error getting memory usage: {e}")
            return {}
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        stats = {
            'performance_history': self.performance_history,
            'optimization_config': self.config,
            'memory_optimizer_stats': self.memory_optimizer.memory_monitor.get_memory_stats() if self.memory_optimizer else {},
            'data_loader_stats': self.data_loader.cache_manager.get_cache_stats() if self.data_loader else {},
            'compilation_stats': {}  # Would need to be implemented in ModelCompiler
        }
        
        return stats
    
    def adaptive_optimize(self, model: nn.Module, performance_metric: float) -> nn.Module:
        """Adaptive optimization based on performance metrics."""
        if not self.config.enable_adaptive_optimization:
            return model
            
        try:
            # Check if optimization is needed
            if len(self.performance_history) < 2:
                return model
                
            recent_performance = self.performance_history[-1]
            previous_performance = self.performance_history[-2]
            
            # Calculate improvement
            improvement = (recent_performance['memory_usage']['rss'] - previous_performance['memory_usage']['rss']) / previous_performance['memory_usage']['rss']
            
            if abs(improvement) > self.config.optimization_threshold:
                self.logger.info(f"Performance change detected: {improvement:.2%}")
                
                # Apply additional optimizations
                if improvement > 0:  # Performance degraded
                    model = self._apply_aggressive_optimization(model)
                else:  # Performance improved
                    model = self._apply_conservative_optimization(model)
            
            return model
            
        except Exception as e:
            self.logger.error(f"Adaptive optimization failed: {e}")
            return model
    
    def _apply_aggressive_optimization(self, model: nn.Module) -> nn.Module:
        """Apply aggressive optimization when performance degrades."""
        try:
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Reduce batch size
            if self.config.data_config.batch_size > 16:
                self.config.data_config.batch_size //= 2
                self.logger.info(f"Reduced batch size to {self.config.data_config.batch_size}")
            
            # Enable more aggressive memory optimization
            if self.memory_optimizer:
                self.config.memory_config.gc_threshold = 0.5
                self.config.memory_config.max_memory_usage = 0.7
            
            return model
            
        except Exception as e:
            self.logger.error(f"Aggressive optimization failed: {e}")
            return model
    
    def _apply_conservative_optimization(self, model: nn.Module) -> nn.Module:
        """Apply conservative optimization when performance improves."""
        try:
            # Gradually increase batch size
            if self.config.data_config.batch_size < 128:
                self.config.data_config.batch_size = min(128, self.config.data_config.batch_size * 2)
                self.logger.info(f"Increased batch size to {self.config.data_config.batch_size}")
            
            # Relax memory constraints
            if self.memory_optimizer:
                self.config.memory_config.gc_threshold = 0.8
                self.config.memory_config.max_memory_usage = 0.9
            
            return model
            
        except Exception as e:
            self.logger.error(f"Conservative optimization failed: {e}")
            return model

# Utility functions
def optimize_training_pipeline(model: nn.Module, dataset, training_config: Dict[str, Any], config: OptimizationConfig = None) -> Tuple[nn.Module, Dict[str, Any]]:
    """Optimize training pipeline with default configuration."""
    if config is None:
        config = OptimizationConfig()
    
    orchestrator = OptimizationOrchestrator(config)
    
    async def _optimize():
        return await orchestrator.optimize_training_pipeline(model, dataset, training_config)
    
    # Run async optimization
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        optimized_model, optimized_config = loop.run_until_complete(_optimize())
    finally:
        loop.close()
    
    return optimized_model, optimized_config

@contextmanager
def optimization_context(config: OptimizationConfig = None):
    """Context manager for optimization."""
    if config is None:
        config = OptimizationConfig()
    
    orchestrator = OptimizationOrchestrator(config)
    try:
        yield orchestrator
    finally:
        # Cleanup
        pass

def auto_optimize_model(model: nn.Module, config: OptimizationConfig = None) -> nn.Module:
    """Automatically optimize model with default settings."""
    if config is None:
        config = OptimizationConfig()
    
    orchestrator = OptimizationOrchestrator(config)
    
    with orchestrator.optimization_context(model):
        return model






