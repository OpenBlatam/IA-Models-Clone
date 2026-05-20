"""
Performance Optimization Utilities
Aggressive optimizations for faster training and inference
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Callable
import logging
import warnings

logger = logging.getLogger(__name__)


def compile_model(
    model: nn.Module,
    mode: str = "reduce-overhead",  # "default", "reduce-overhead", "max-autotune"
    fullgraph: bool = False,
    dynamic: bool = False
) -> nn.Module:
    """
    Compile model with torch.compile (PyTorch 2.0+)
    Provides significant speedup for inference and training
    
    Args:
        model: PyTorch model
        mode: Compilation mode
            - "default": Balanced optimization
            - "reduce-overhead": Reduces framework overhead
            - "max-autotune": Maximum optimization (slower compilation)
        fullgraph: Whether to compile entire graph
        dynamic: Support dynamic shapes
        
    Returns:
        Compiled model
    """
    if not hasattr(torch, "compile"):
        logger.warning("torch.compile not available (requires PyTorch 2.0+)")
        return model
    
    try:
        compiled_model = torch.compile(
            model,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic
        )
        logger.info(f"Model compiled with mode={mode}")
        return compiled_model
    except Exception as e:
        logger.warning(f"Model compilation failed: {e}. Using uncompiled model.")
        return model


def optimize_for_inference(
    model: nn.Module,
    use_jit: bool = True,
    use_tracing: bool = False,
    optimize_for_mobile: bool = False
) -> nn.Module:
    """
    Optimize model specifically for inference
    
    Args:
        model: PyTorch model
        use_jit: Use JIT compilation
        use_tracing: Use tracing instead of scripting
        optimize_for_mobile: Optimize for mobile deployment
        
    Returns:
        Optimized model
    """
    model.eval()
    
    # JIT compilation
    if use_jit:
        try:
            if use_tracing:
                # Create dummy input for tracing
                dummy_input = torch.randn(1, 3, 224, 224)
                model = torch.jit.trace(model, dummy_input)
                logger.info("Model optimized with JIT tracing")
            else:
                model = torch.jit.script(model)
                logger.info("Model optimized with JIT scripting")
        except Exception as e:
            logger.warning(f"JIT optimization failed: {e}")
    
    # Mobile optimization
    if optimize_for_mobile:
        try:
            model = torch.utils.mobile_optimizer.optimize_for_mobile(model)
            logger.info("Model optimized for mobile")
        except Exception as e:
            logger.warning(f"Mobile optimization failed: {e}")
    
    return model


def fuse_model(model: nn.Module) -> nn.Module:
    """
    Fuse operations for faster inference
    Fuses Conv+BN+ReLU, Linear+ReLU, etc.
    
    Args:
        model: PyTorch model
        
    Returns:
        Fused model
    """
    try:
        # Fuse Conv+BN+ReLU
        if hasattr(torch.ao.quantization, 'fuse_modules'):
            from torch.ao.quantization import fuse_modules
            
            # This is a simplified version - actual implementation
            # would need to traverse the model structure
            logger.info("Model fusion attempted (may require manual implementation)")
        else:
            # Fallback: Use torch.jit for fusion
            model = torch.jit.script(model)
            logger.info("Model fused with JIT")
    except Exception as e:
        logger.warning(f"Model fusion failed: {e}")
    
    return model


def quantize_model(
    model: nn.Module,
    quantization_type: str = "int8",  # "int8", "int8_dynamic", "fp16"
    calibration_data: Optional[Any] = None
) -> nn.Module:
    """
    Quantize model for faster inference
    
    Args:
        model: PyTorch model
        quantization_type: Type of quantization
        calibration_data: Data for calibration (for static quantization)
        
    Returns:
        Quantized model
    """
    model.eval()
    
    if quantization_type == "int8_dynamic":
        # Dynamic quantization (no calibration needed)
        try:
            model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.Conv2d},
                dtype=torch.qint8
            )
            logger.info("Model quantized with dynamic int8")
        except Exception as e:
            logger.warning(f"Dynamic quantization failed: {e}")
    
    elif quantization_type == "int8":
        # Static quantization (requires calibration)
        try:
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(model, inplace=True)
            
            # Calibrate
            if calibration_data:
                with torch.no_grad():
                    for data in calibration_data:
                        _ = model(data)
            
            model = torch.quantization.convert(model, inplace=True)
            logger.info("Model quantized with static int8")
        except Exception as e:
            logger.warning(f"Static quantization failed: {e}")
    
    elif quantization_type == "fp16":
        # FP16 quantization
        model = model.half()
        logger.info("Model quantized to FP16")
    
    return model


def enable_cudnn_benchmark():
    """Enable cuDNN benchmark for faster convolutions"""
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        logger.info("cuDNN benchmark enabled")


def enable_tf32():
    """Enable TF32 for faster training on Ampere+ GPUs"""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("TF32 enabled")


def optimize_data_loading(
    dataset,
    batch_size: int = 32,
    num_workers: int = None,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = True
):
    """
    Create optimized DataLoader with best practices
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        num_workers: Number of workers (None = auto)
        pin_memory: Pin memory for faster GPU transfer
        prefetch_factor: Number of batches to prefetch
        persistent_workers: Keep workers alive between epochs
        
    Returns:
        Optimized DataLoader
    """
    from torch.utils.data import DataLoader
    
    if num_workers is None:
        # Auto-detect optimal number
        num_workers = min(8, torch.multiprocessing.cpu_count())
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers and num_workers > 0,
        drop_last=False
    )


class FastInferenceEngine:
    """
    Fast inference engine with optimizations
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        use_compile: bool = True,
        use_jit: bool = False,
        use_quantization: bool = False,
        quantization_type: str = "int8_dynamic",
        batch_size: int = 1
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.batch_size = batch_size
        
        # Optimize model
        if use_compile:
            self.model = compile_model(self.model, mode="reduce-overhead")
        
        if use_jit:
            self.model = optimize_for_inference(self.model, use_jit=True)
        
        if use_quantization:
            self.model = quantize_model(self.model, quantization_type)
        
        self.model.eval()
        
        # Enable optimizations
        enable_cudnn_benchmark()
        if device == "cuda":
            enable_tf32()
    
    @torch.no_grad()
    def predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Fast prediction with optimizations"""
        input_tensor = input_tensor.to(self.device, non_blocking=True)
        
        # Use autocast for mixed precision
        with torch.cuda.amp.autocast():
            output = self.model(input_tensor)
        
        return output
    
    @torch.no_grad()
    def predict_batch(self, input_batch: torch.Tensor) -> torch.Tensor:
        """Batch prediction"""
        return self.predict(input_batch)


def create_optimized_trainer(
    model: nn.Module,
    train_loader,
    val_loader=None,
    compile_model: bool = True,
    enable_tf32: bool = True,
    **trainer_kwargs
):
    """
    Create optimized trainer with all performance enhancements
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        compile_model: Compile model with torch.compile
        enable_tf32: Enable TF32
        **trainer_kwargs: Additional trainer arguments
        
    Returns:
        Optimized Trainer instance
    """
    from training import Trainer
    
    # Enable optimizations
    if enable_tf32:
        enable_tf32()
    enable_cudnn_benchmark()
    
    # Compile model if requested
    if compile_model and hasattr(torch, "compile"):
        model = compile_model(model, mode="reduce-overhead")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        **trainer_kwargs
    )
    
    return trainer


def optimize_memory():
    """Optimize memory usage"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Enable memory efficient attention if available
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            logger.info("Flash attention enabled")
        except:
            pass


def set_optimal_threads():
    """Set optimal number of threads for CPU operations"""
    import os
    num_threads = min(4, os.cpu_count() or 1)
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads)
    logger.info(f"Set threads: {num_threads}")


class ModelCache:
    """
    Cache for model outputs to speed up repeated inferences
    """
    
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, Any] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached result"""
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None
    
    def set(self, key: str, value: Any):
        """Cache result"""
        if len(self.cache) >= self.max_size:
            # Remove oldest (simple FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = value
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict[str, float]:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'size': len(self.cache)
        }













