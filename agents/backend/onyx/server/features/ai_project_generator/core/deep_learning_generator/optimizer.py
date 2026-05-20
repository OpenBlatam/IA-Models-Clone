"""
Optimizer Module

Configuration optimization and auto-tuning utilities.
"""

from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ConfigOptimizer:
    """
    Optimizes generator configurations based on hardware and requirements.
    """
    
    @staticmethod
    def optimize_for_hardware(
        config: Dict[str, Any],
        gpu_memory_gb: Optional[float] = None,
        cpu_cores: Optional[int] = None,
        available_ram_gb: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Optimize configuration based on available hardware.
        
        Args:
            config: Base configuration
            gpu_memory_gb: Available GPU memory in GB
            cpu_cores: Number of CPU cores
            available_ram_gb: Available RAM in GB
            
        Returns:
            Optimized configuration
        """
        optimized = config.copy()
        
        # Optimize batch size based on GPU memory
        if gpu_memory_gb:
            if gpu_memory_gb < 4:
                optimized["batch_size"] = min(optimized.get("batch_size", 32), 8)
                optimized["mixed_precision"] = True
            elif gpu_memory_gb < 8:
                optimized["batch_size"] = min(optimized.get("batch_size", 32), 16)
                optimized["mixed_precision"] = True
            elif gpu_memory_gb < 16:
                optimized["batch_size"] = min(optimized.get("batch_size", 32), 32)
            else:
                optimized["batch_size"] = optimized.get("batch_size", 64)
        
        # Optimize based on CPU cores
        if cpu_cores:
            if cpu_cores < 4:
                optimized["num_workers"] = 2
            elif cpu_cores < 8:
                optimized["num_workers"] = 4
            else:
                optimized["num_workers"] = 8
        
        # Optimize based on RAM
        if available_ram_gb:
            if available_ram_gb < 8:
                optimized["batch_size"] = min(optimized.get("batch_size", 32), 16)
                optimized["pin_memory"] = False
            elif available_ram_gb < 16:
                optimized["pin_memory"] = True
            else:
                optimized["pin_memory"] = True
                optimized["prefetch_factor"] = 2
        
        return optimized
    
    @staticmethod
    def optimize_for_model_type(
        config: Dict[str, Any],
        model_type: str
    ) -> Dict[str, Any]:
        """
        Optimize configuration for specific model type.
        
        Args:
            config: Base configuration
            model_type: Type of model
            
        Returns:
            Optimized configuration
        """
        optimized = config.copy()
        
        model_optimizations = {
            "llm": {
                "batch_size": 8,
                "learning_rate": 5e-5,
                "gradient_clipping": True,
                "gradient_clipping_max_norm": 1.0,
                "gradient_accumulation_steps": 4
            },
            "diffusion": {
                "batch_size": 4,
                "learning_rate": 1e-4,
                "mixed_precision": True,
                "gradient_checkpointing": True
            },
            "gan": {
                "batch_size": 32,
                "learning_rate": 2e-4,
                "mixed_precision": False,
                "gradient_clipping": False
            },
            "transformer": {
                "batch_size": 32,
                "learning_rate": 1e-4,
                "mixed_precision": True,
                "gradient_clipping": True
            },
            "cnn": {
                "batch_size": 64,
                "learning_rate": 1e-3,
                "mixed_precision": True
            }
        }
        
        if model_type in model_optimizations:
            optimized.update(model_optimizations[model_type])
        
        return optimized
    
    @staticmethod
    def auto_tune_batch_size(
        config: Dict[str, Any],
        start_size: int = 32,
        max_size: int = 128,
        factor: int = 2
    ) -> Dict[str, Any]:
        """
        Auto-tune batch size for optimal performance.
        
        Args:
            config: Base configuration
            start_size: Starting batch size
            max_size: Maximum batch size
            factor: Factor to increase/decrease
            
        Returns:
            Configuration with auto-tuned batch size
        """
        optimized = config.copy()
        
        # Simple heuristic: start with base and adjust based on model type
        model_type = config.get("model_type", "transformer")
        
        if model_type == "llm":
            optimized["batch_size"] = min(start_size // 4, 8)
        elif model_type == "diffusion":
            optimized["batch_size"] = min(start_size // 8, 4)
        elif model_type == "cnn":
            optimized["batch_size"] = min(start_size * 2, max_size)
        else:
            optimized["batch_size"] = start_size
        
        return optimized
    
    @staticmethod
    def optimize_learning_rate(
        config: Dict[str, Any],
        model_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Optimize learning rate based on model type and batch size.
        
        Args:
            config: Base configuration
            model_type: Type of model
            
        Returns:
            Configuration with optimized learning rate
        """
        optimized = config.copy()
        model_type = model_type or config.get("model_type", "transformer")
        batch_size = config.get("batch_size", 32)
        
        # Learning rate scaling with batch size
        base_lr = {
            "llm": 5e-5,
            "transformer": 1e-4,
            "cnn": 1e-3,
            "diffusion": 1e-4,
            "gan": 2e-4
        }.get(model_type, 1e-4)
        
        # Scale learning rate with batch size (linear scaling)
        if batch_size > 32:
            scale_factor = batch_size / 32
            optimized["learning_rate"] = base_lr * scale_factor
        else:
            optimized["learning_rate"] = base_lr
        
        return optimized


def optimize_config(
    config: Dict[str, Any],
    hardware_info: Optional[Dict[str, Any]] = None,
    auto_tune: bool = True
) -> Dict[str, Any]:
    """
    Optimize a configuration with all available optimizations.
    
    Args:
        config: Base configuration
        hardware_info: Hardware information (gpu_memory_gb, cpu_cores, etc.)
        auto_tune: Enable auto-tuning
        
    Returns:
        Optimized configuration
    """
    optimizer = ConfigOptimizer()
    optimized = config.copy()
    
    # Optimize for model type
    if "model_type" in optimized:
        optimized = optimizer.optimize_for_model_type(optimized, optimized["model_type"])
    
    # Optimize for hardware
    if hardware_info:
        optimized = optimizer.optimize_for_hardware(
            optimized,
            gpu_memory_gb=hardware_info.get("gpu_memory_gb"),
            cpu_cores=hardware_info.get("cpu_cores"),
            available_ram_gb=hardware_info.get("available_ram_gb")
        )
    
    # Auto-tune
    if auto_tune:
        optimized = optimizer.auto_tune_batch_size(optimized)
        optimized = optimizer.optimize_learning_rate(optimized)
    
    return optimized















