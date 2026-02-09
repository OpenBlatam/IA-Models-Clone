"""
Model Optimizer - Advanced optimization utilities.

Provides:
- Model quantization helpers
- Memory optimization
- Performance tuning
- Batch optimization
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

import torch

logger = logging.getLogger(__name__)


class OptimizationLevel(str, Enum):
    """Optimization levels."""
    NONE = "none"
    BASIC = "basic"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"


@dataclass
class OptimizationConfig:
    """Configuration for model optimization."""
    level: OptimizationLevel = OptimizationLevel.BASIC
    enable_gradient_checkpointing: bool = True
    enable_flash_attention: bool = True
    enable_torch_compile: bool = False
    enable_8bit_optimizer: bool = False
    enable_4bit_quantization: bool = False
    enable_8bit_quantization: bool = False
    memory_efficient_attention: bool = True
    use_channels_last: bool = True
    use_tf32: bool = True
    use_cudnn_benchmark: bool = True


class ModelOptimizer:
    """Model optimization utilities."""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        """
        Initialize optimizer.
        
        Args:
            config: Optimization configuration
        """
        self.config = config or OptimizationConfig()
    
    def optimize_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Apply optimizations to model.
        
        Args:
            model: PyTorch model
            
        Returns:
            Optimized model
        """
        if self.config.level == OptimizationLevel.NONE:
            return model
        
        logger.info(f"Optimizing model with level: {self.config.level}")
        
        # Apply optimizations based on level
        if self.config.level in [OptimizationLevel.BASIC, OptimizationLevel.AGGRESSIVE, OptimizationLevel.MAXIMUM]:
            model = self._apply_basic_optimizations(model)
        
        if self.config.level in [OptimizationLevel.AGGRESSIVE, OptimizationLevel.MAXIMUM]:
            model = self._apply_aggressive_optimizations(model)
        
        if self.config.level == OptimizationLevel.MAXIMUM:
            model = self._apply_maximum_optimizations(model)
        
        return model
    
    def _apply_basic_optimizations(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply basic optimizations."""
        # Gradient checkpointing
        if self.config.enable_gradient_checkpointing:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
            logger.debug("Enabled gradient checkpointing")
        
        # Channels last memory format
        if self.config.use_channels_last and torch.cuda.is_available():
            try:
                model = model.to(memory_format=torch.channels_last)
                logger.debug("Applied channels last memory format")
            except Exception as e:
                logger.warning(f"Failed to apply channels last: {e}")
        
        # cuDNN benchmark
        if self.config.use_cudnn_benchmark and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            logger.debug("Enabled cuDNN benchmark")
        
        # TF32
        if self.config.use_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.debug("Enabled TF32")
        
        return model
    
    def _apply_aggressive_optimizations(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply aggressive optimizations."""
        # Torch compile (PyTorch 2.0+)
        if self.config.enable_torch_compile:
            try:
                model = torch.compile(model, mode="reduce-overhead")
                logger.debug("Applied torch.compile")
            except Exception as e:
                logger.warning(f"Failed to apply torch.compile: {e}")
        
        return model
    
    def _apply_maximum_optimizations(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply maximum optimizations."""
        # Additional optimizations for maximum performance
        # This would include more aggressive settings
        return model
    
    def optimize_memory(self) -> Dict[str, Any]:
        """
        Optimize memory usage.
        
        Returns:
            Dictionary with memory optimization results
        """
        optimizations = {}
        
        if torch.cuda.is_available():
            # Clear cache
            torch.cuda.empty_cache()
            optimizations["cuda_cache_cleared"] = True
            
            # Set memory fraction if needed
            # torch.cuda.set_per_process_memory_fraction(0.9)
        
        return optimizations
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of applied optimizations."""
        return {
            "level": self.config.level.value,
            "gradient_checkpointing": self.config.enable_gradient_checkpointing,
            "flash_attention": self.config.enable_flash_attention,
            "torch_compile": self.config.enable_torch_compile,
            "channels_last": self.config.use_channels_last,
            "tf32": self.config.use_tf32,
            "cudnn_benchmark": self.config.use_cudnn_benchmark,
        }


def create_optimization_config(level: str = "basic") -> OptimizationConfig:
    """
    Create optimization config from level string.
    
    Args:
        level: Optimization level (none, basic, aggressive, maximum)
        
    Returns:
        OptimizationConfig
    """
    opt_level = OptimizationLevel(level.lower())
    
    if opt_level == OptimizationLevel.NONE:
        return OptimizationConfig(level=opt_level, enable_gradient_checkpointing=False)
    elif opt_level == OptimizationLevel.BASIC:
        return OptimizationConfig(level=opt_level)
    elif opt_level == OptimizationLevel.AGGRESSIVE:
        return OptimizationConfig(
            level=opt_level,
            enable_torch_compile=True,
        )
    else:  # MAXIMUM
        return OptimizationConfig(
            level=opt_level,
            enable_torch_compile=True,
            enable_8bit_optimizer=True,
        )












