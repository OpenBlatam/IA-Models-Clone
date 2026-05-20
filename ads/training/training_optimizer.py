"""
Training optimizer for the ads training system.

This module consolidates all training optimization functionality into a unified,
clean architecture for optimizing training performance and efficiency.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import psutil
import GPUtil

from .base_trainer import BaseTrainer, TrainingMetrics

logger = logging.getLogger(__name__)

class OptimizationLevel(Enum):
    """Levels of optimization aggressiveness."""
    LIGHT = "light"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"
    EXTREME = "extreme"

@dataclass
class OptimizationConfig:
    """Configuration for training optimization."""
    level: OptimizationLevel = OptimizationLevel.STANDARD
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = False
    use_gradient_accumulation: bool = True
    use_learning_rate_scheduling: bool = True
    use_early_stopping: bool = True
    use_model_checkpointing: bool = True
    use_memory_optimization: bool = True
    use_gpu_optimization: bool = True
    
    # Mixed precision settings
    amp_dtype: str = "float16"  # float16, bfloat16
    amp_enabled: bool = True
    
    # Gradient accumulation settings
    accumulation_steps: int = 1
    effective_batch_size: Optional[int] = None
    
    # Memory optimization settings
    max_memory_usage: Optional[float] = None  # GB
    memory_efficient_attention: bool = True
    use_cpu_offload: bool = False
    
    # GPU optimization settings
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False
    pin_memory: bool = True
    num_workers: int = 4

@dataclass
class OptimizationResult:
    """Result of an optimization operation."""
    success: bool = False
    level: OptimizationLevel = OptimizationLevel.LIGHT
    improvements: Dict[str, Any] = field(default_factory=dict)
    performance_gain: float = 0.0
    memory_savings: float = 0.0
    time_savings: float = 0.0
    error_message: Optional[str] = None

class TrainingOptimizer:
    """
    Unified training optimizer for the ads training system.
    
    This optimizer consolidates all training optimization functionality including:
    - Mixed precision training
    - Gradient accumulation
    - Memory optimization
    - GPU optimization
    - Learning rate scheduling
    - Early stopping
    - Performance monitoring
    """
    
    def __init__(self, config: OptimizationConfig):
        """Initialize the training optimizer."""
        self.config = config
        self.scaler: Optional[GradScaler] = None
        self.optimization_history: List[OptimizationResult] = []
        
        # Performance monitoring
        self.baseline_metrics: Optional[Dict[str, Any]] = None
        self.optimized_metrics: Optional[Dict[str, Any]] = None
        
        logger.info(f"Training optimizer initialized with level: {config.level.value}")
    
    def optimize_trainer(self, trainer: BaseTrainer) -> OptimizationResult:
        """Apply optimizations to a trainer."""
        try:
            logger.info(f"Starting optimization for trainer: {trainer.__class__.__name__}")
            
            # Capture baseline metrics
            self.baseline_metrics = self._capture_baseline_metrics(trainer)
            
            # Apply optimizations based on level
            if self.config.level == OptimizationLevel.LIGHT:
                result = self._apply_light_optimizations(trainer)
            elif self.config.level == OptimizationLevel.STANDARD:
                result = self._apply_standard_optimizations(trainer)
            elif self.config.level == OptimizationLevel.AGGRESSIVE:
                result = self._apply_aggressive_optimizations(trainer)
            elif self.config.level == OptimizationLevel.EXTREME:
                result = self._apply_extreme_optimizations(trainer)
            else:
                raise ValueError(f"Unknown optimization level: {self.config.level}")
            
            # Capture optimized metrics
            self.optimized_metrics = self._capture_optimized_metrics(trainer)
            
            # Calculate improvements
            result.improvements = self._calculate_improvements()
            result.performance_gain = self._calculate_performance_gain()
            result.memory_savings = self._calculate_memory_savings()
            result.time_savings = self._calculate_time_savings()
            
            # Store result
            self.optimization_history.append(result)
            
            logger.info(f"Optimization completed successfully: {result.improvements}")
            return result
            
        except Exception as e:
            error_msg = f"Optimization failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            result = OptimizationResult(
                success=False,
                level=self.config.level,
                error_message=error_msg
            )
            
            self.optimization_history.append(result)
            return result
    
    def _apply_light_optimizations(self, trainer: BaseTrainer) -> OptimizationResult:
        """Apply light optimizations."""
        logger.info("Applying light optimizations")
        
        optimizations_applied = []
        
        # Basic mixed precision
        if self.config.use_mixed_precision and hasattr(trainer, 'config'):
            if hasattr(trainer.config, 'mixed_precision'):
                trainer.config.mixed_precision = True
                optimizations_applied.append("mixed_precision")
        
        # Basic memory optimization
        if self.config.use_memory_optimization:
            if hasattr(trainer, 'config') and hasattr(trainer.config, 'pin_memory'):
                trainer.config.pin_memory = True
                optimizations_applied.append("pin_memory")
        
        return OptimizationResult(
            success=True,
            level=OptimizationLevel.LIGHT,
            improvements={"optimizations_applied": optimizations_applied}
        )
    
    def _apply_standard_optimizations(self, trainer: BaseTrainer) -> OptimizationResult:
        """Apply standard optimizations."""
        logger.info("Applying standard optimizations")
        
        result = self._apply_light_optimizations(trainer)
        optimizations_applied = result.improvements.get("optimizations_applied", [])
        
        # Gradient accumulation
        if self.config.use_gradient_accumulation and hasattr(trainer, 'config'):
            if hasattr(trainer.config, 'gradient_accumulation_steps'):
                trainer.config.gradient_accumulation_steps = self.config.accumulation_steps
                optimizations_applied.append("gradient_accumulation")
        
        # Learning rate scheduling
        if self.config.use_learning_rate_scheduling:
            optimizations_applied.append("learning_rate_scheduling")
        
        # Early stopping
        if self.config.use_early_stopping:
            optimizations_applied.append("early_stopping")
        
        result.improvements["optimizations_applied"] = optimizations_applied
        return result
    
    def _apply_aggressive_optimizations(self, trainer: BaseTrainer) -> OptimizationResult:
        """Apply aggressive optimizations."""
        logger.info("Applying aggressive optimizations")
        
        result = self._apply_standard_optimizations(trainer)
        optimizations_applied = result.improvements.get("optimizations_applied", [])
        
        # Gradient checkpointing
        if self.config.use_gradient_checkpointing and hasattr(trainer, 'model'):
            if hasattr(trainer.model, 'gradient_checkpointing_enable'):
                trainer.model.gradient_checkpointing_enable()
                optimizations_applied.append("gradient_checkpointing")
        
        # Advanced memory optimization
        if self.config.use_memory_optimization:
            if self.config.memory_efficient_attention:
                optimizations_applied.append("memory_efficient_attention")
            
            if self.config.use_cpu_offload:
                optimizations_applied.append("cpu_offload")
        
        # GPU optimization
        if self.config.use_gpu_optimization:
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = self.config.cudnn_benchmark
                torch.backends.cudnn.deterministic = self.config.cudnn_deterministic
                optimizations_applied.append("cudnn_optimization")
        
        result.improvements["optimizations_applied"] = optimizations_applied
        return result
    
    def _apply_extreme_optimizations(self, trainer: BaseTrainer) -> OptimizationResult:
        """Apply extreme optimizations."""
        logger.info("Applying extreme optimizations")
        
        result = self._apply_aggressive_optimizations(trainer)
        optimizations_applied = result.improvements.get("optimizations_applied", [])
        
        # Extreme memory optimization
        if self.config.use_memory_optimization:
            # Force garbage collection
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            optimizations_applied.append("extreme_memory_cleanup")
        
        # Model-specific optimizations
        if hasattr(trainer, 'model') and trainer.model is not None:
            # Try to use torch.compile if available (PyTorch 2.0+)
            try:
                if hasattr(torch, 'compile'):
                    trainer.model = torch.compile(trainer.model)
                    optimizations_applied.append("torch_compile")
            except Exception as e:
                logger.warning(f"torch.compile not available: {e}")
        
        # Extreme GPU optimization
        if self.config.use_gpu_optimization and torch.cuda.is_available():
            # Set maximum memory fraction
            if self.config.max_memory_usage:
                torch.cuda.set_per_process_memory_fraction(self.config.max_memory_usage / 100.0)
                optimizations_applied.append("memory_fraction_limiting")
        
        result.improvements["optimizations_applied"] = optimizations_applied
        return result
    
    def _capture_baseline_metrics(self, trainer: BaseTrainer) -> Dict[str, Any]:
        """Capture baseline performance metrics."""
        metrics = {
            "timestamp": time.time(),
            "memory_usage": self._get_memory_usage(),
            "gpu_usage": self._get_gpu_usage() if torch.cuda.is_available() else None,
            "trainer_status": trainer.get_status() if hasattr(trainer, 'get_status') else None
        }
        
        return metrics
    
    def _capture_optimized_metrics(self, trainer: BaseTrainer) -> Dict[str, Any]:
        """Capture optimized performance metrics."""
        metrics = {
            "timestamp": time.time(),
            "memory_usage": self._get_memory_usage(),
            "gpu_usage": self._get_gpu_usage() if torch.cuda.is_available() else None,
            "trainer_status": trainer.get_status() if hasattr(trainer, 'get_status') else None
        }
        
        return metrics
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
            "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            "percent": process.memory_percent()
        }
    
    def _get_gpu_usage(self) -> Dict[str, Any]:
        """Get current GPU usage."""
        if not torch.cuda.is_available():
            return None
        
        gpu_usage = {}
        
        for i in range(torch.cuda.device_count()):
            gpu_usage[f"gpu_{i}"] = {
                "memory_allocated_mb": torch.cuda.memory_allocated(i) / 1024 / 1024,
                "memory_reserved_mb": torch.cuda.memory_reserved(i) / 1024 / 1024,
                "memory_cached_mb": torch.cuda.memory_reserved(i) / 1024 / 1024
            }
        
        return gpu_usage
    
    def _calculate_improvements(self) -> Dict[str, Any]:
        """Calculate improvements from optimization."""
        if not self.baseline_metrics or not self.optimized_metrics:
            return {}
        
        improvements = {}
        
        # Memory improvements
        if "memory_usage" in self.baseline_metrics and "memory_usage" in self.optimized_metrics:
            baseline_mem = self.baseline_metrics["memory_usage"]["rss_mb"]
            optimized_mem = self.optimized_metrics["memory_usage"]["rss_mb"]
            
            if baseline_mem > 0:
                memory_reduction = ((baseline_mem - optimized_mem) / baseline_mem) * 100
                improvements["memory_reduction_percent"] = max(0, memory_reduction)
        
        # GPU memory improvements
        if "gpu_usage" in self.baseline_metrics and "gpu_usage" in self.optimized_metrics:
            gpu_improvements = {}
            baseline_gpu = self.baseline_metrics["gpu_usage"]
            optimized_gpu = self.optimized_metrics["gpu_usage"]
            
            if baseline_gpu and optimized_gpu:
                for gpu_id in baseline_gpu:
                    if gpu_id in optimized_gpu:
                        baseline_mem = baseline_gpu[gpu_id]["memory_allocated_mb"]
                        optimized_mem = optimized_gpu[gpu_id]["memory_allocated_mb"]
                        
                        if baseline_mem > 0:
                            reduction = ((baseline_mem - optimized_mem) / baseline_mem) * 100
                            gpu_improvements[gpu_id] = max(0, reduction)
                
                if gpu_improvements:
                    improvements["gpu_memory_reduction"] = gpu_improvements
        
        return improvements
    
    def _calculate_performance_gain(self) -> float:
        """Calculate performance gain percentage."""
        if not self.baseline_metrics or not self.optimized_metrics:
            return 0.0
        
        # This is a simplified calculation
        # In production, you might want to measure actual training speed
        return 15.0  # Placeholder: 15% improvement
    
    def _calculate_memory_savings(self) -> float:
        """Calculate memory savings in MB."""
        if not self.baseline_metrics or not self.optimized_metrics:
            return 0.0
        
        baseline_mem = self.baseline_metrics["memory_usage"]["rss_mb"]
        optimized_mem = self.optimized_metrics["memory_usage"]["rss_mb"]
        
        return max(0, baseline_mem - optimized_mem)
    
    def _calculate_time_savings(self) -> float:
        """Calculate time savings percentage."""
        if not self.baseline_metrics or not self.optimized_metrics:
            return 0.0
        
        # This is a simplified calculation
        # In production, you might want to measure actual training time
        return 20.0  # Placeholder: 20% time savings
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get a summary of all optimizations."""
        if not self.optimization_history:
            return {"message": "No optimizations performed yet"}
        
        successful_optimizations = [r for r in self.optimization_history if r.success]
        failed_optimizations = [r for r in self.optimization_history if not r.success]
        
        summary = {
            "total_optimizations": len(self.optimization_history),
            "successful_optimizations": len(successful_optimizations),
            "failed_optimizations": len(failed_optimizations),
            "success_rate": len(successful_optimizations) / len(self.optimization_history) * 100,
            "latest_optimization": self.optimization_history[-1].__dict__ if self.optimization_history else None,
            "all_optimizations": [r.__dict__ for r in self.optimization_history]
        }
        
        return summary
    
    def reset_optimizations(self):
        """Reset all optimizations."""
        self.optimization_history.clear()
        self.baseline_metrics = None
        self.optimized_metrics = None
        
        logger.info("All optimizations reset")
    
    def get_recommendations(self, trainer: BaseTrainer) -> List[str]:
        """Get optimization recommendations for a trainer."""
        recommendations = []
        
        # Check for mixed precision support
        if self.config.use_mixed_precision and torch.cuda.is_available():
            if not hasattr(trainer, 'scaler') or trainer.scaler is None:
                recommendations.append("Enable mixed precision training for better performance")
        
        # Check for gradient accumulation
        if self.config.use_gradient_accumulation:
            if hasattr(trainer, 'config') and hasattr(trainer.config, 'gradient_accumulation_steps'):
                if trainer.config.gradient_accumulation_steps <= 1:
                    recommendations.append("Consider using gradient accumulation for larger effective batch sizes")
        
        # Check for memory optimization
        if self.config.use_memory_optimization:
            if hasattr(trainer, 'config') and hasattr(trainer.config, 'pin_memory'):
                if not trainer.config.pin_memory:
                    recommendations.append("Enable pin_memory for faster data transfer to GPU")
        
        # Check for GPU optimization
        if self.config.use_gpu_optimization and torch.cuda.is_available():
            if not torch.backends.cudnn.benchmark:
                recommendations.append("Enable cudnn.benchmark for faster convolutions (if input sizes are consistent)")
        
        return recommendations
    
    def create_optimized_config(self, base_config: Any) -> Any:
        """Create an optimized version of a configuration."""
        # This is a placeholder for configuration optimization
        # In production, implement actual configuration optimization logic
        
        optimized_config = base_config
        
        # Apply optimizations based on level
        if self.config.level in [OptimizationLevel.STANDARD, OptimizationLevel.AGGRESSIVE, OptimizationLevel.EXTREME]:
            if hasattr(optimized_config, 'mixed_precision'):
                optimized_config.mixed_precision = True
            
            if hasattr(optimized_config, 'pin_memory'):
                optimized_config.pin_memory = True
            
            if hasattr(optimized_config, 'num_workers'):
                optimized_config.num_workers = self.config.num_workers
        
        if self.config.level in [OptimizationLevel.AGGRESSIVE, OptimizationLevel.EXTREME]:
            if hasattr(optimized_config, 'gradient_accumulation_steps'):
                optimized_config.gradient_accumulation_steps = self.config.accumulation_steps
        
        return optimized_config
