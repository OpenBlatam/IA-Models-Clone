"""
Speed Optimizations for Music Analyzer AI
Aggressive performance optimizations using PyTorch 2.0+ features
"""

from typing import Optional, Dict, Any, List
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class SpeedOptimizer:
    """
    Aggressive speed optimizations for models and training
    """
    
    @staticmethod
    def compile_model(
        model: nn.Module,
        mode: str = "reduce-overhead",  # "default", "reduce-overhead", "max-autotune"
        fullgraph: bool = False,
        dynamic: bool = False
    ) -> nn.Module:
        """
        Compile model with torch.compile (PyTorch 2.0+) for 2-3x speedup
        
        Args:
            model: PyTorch model
            mode: Compilation mode
            fullgraph: Compile entire graph
            dynamic: Support dynamic shapes
        
        Returns:
            Compiled model
        """
        if not hasattr(torch, 'compile'):
            logger.warning("torch.compile not available (requires PyTorch 2.0+)")
            return model
        
        try:
            compiled_model = torch.compile(
                model,
                mode=mode,
                fullgraph=fullgraph,
                dynamic=dynamic
            )
            logger.info(f"Model compiled with torch.compile (mode={mode})")
            return compiled_model
        except Exception as e:
            logger.warning(f"Model compilation failed: {str(e)}, using original model")
            return model
    
    @staticmethod
    def optimize_dataloader(
        dataloader: DataLoader,
        num_workers: int = 4,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        persistent_workers: bool = True
    ) -> DataLoader:
        """
        Optimize DataLoader for faster data loading
        
        Args:
            dataloader: Original DataLoader
            num_workers: Number of worker processes
            pin_memory: Pin memory for faster GPU transfer
            prefetch_factor: Number of batches to prefetch
            persistent_workers: Keep workers alive between epochs
        
        Returns:
            Optimized DataLoader
        """
        # Create new DataLoader with optimized settings
        optimized_loader = DataLoader(
            dataloader.dataset,
            batch_size=dataloader.batch_size,
            shuffle=dataloader.shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            drop_last=dataloader.drop_last,
            collate_fn=dataloader.collate_fn
        )
        logger.info(f"DataLoader optimized: {num_workers} workers, pin_memory={pin_memory}")
        return optimized_loader
    
    @staticmethod
    def enable_channels_last(model: nn.Module) -> nn.Module:
        """
        Convert model to channels_last memory format for faster convolutions
        """
        try:
            model = model.to(memory_format=torch.channels_last)
            logger.info("Model converted to channels_last format")
            return model
        except Exception as e:
            logger.warning(f"channels_last conversion failed: {str(e)}")
            return model
    
    @staticmethod
    def enable_tf32(model: nn.Module) -> None:
        """
        Enable TF32 for faster training on Ampere+ GPUs (up to 1.5x speedup)
        """
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("TF32 enabled for faster training")
    
    @staticmethod
    def enable_benchmark_mode() -> None:
        """
        Enable cuDNN benchmark mode for consistent input sizes (faster)
        """
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            logger.info("cuDNN benchmark mode enabled")
    
    @staticmethod
    def enable_deterministic(enable: bool = False) -> None:
        """
        Enable/disable deterministic mode (disable for speed, enable for reproducibility)
        """
        torch.backends.cudnn.deterministic = enable
        torch.use_deterministic_algorithms(enable)
        logger.info(f"Deterministic mode: {enable}")


class FastInference:
    """
    Fast inference optimizations
    """
    
    @staticmethod
    def prepare_for_inference(model: nn.Module, device: str = "cuda") -> nn.Module:
        """
        Prepare model for fast inference
        
        Args:
            model: PyTorch model
            device: Target device
        
        Returns:
            Optimized model
        """
        model.eval()
        model = model.to(device)
        
        # Compile if available
        if hasattr(torch, 'compile'):
            model = SpeedOptimizer.compile_model(model, mode="reduce-overhead")
        
        # Enable optimizations
        if device == "cuda":
            SpeedOptimizer.enable_benchmark_mode()
            SpeedOptimizer.enable_tf32(model)
        
        return model
    
    @staticmethod
    def fast_forward(
        model: nn.Module,
        inputs: torch.Tensor,
        use_autocast: bool = True,
        use_no_grad: bool = True
    ) -> torch.Tensor:
        """
        Fast forward pass with optimizations
        
        Args:
            model: PyTorch model
            inputs: Input tensor
            use_autocast: Use mixed precision
            use_no_grad: Disable gradient computation
        
        Returns:
            Model output
        """
        context = torch.no_grad() if use_no_grad else torch.enable_grad()
        
        with context:
            if use_autocast and inputs.is_cuda:
                with torch.cuda.amp.autocast():
                    return model(inputs)
            else:
                return model(inputs)


class FastTraining:
    """
    Fast training optimizations
    """
    
    @staticmethod
    def prepare_for_training(
        model: nn.Module,
        device: str = "cuda",
        compile_model: bool = True
    ) -> nn.Module:
        """
        Prepare model for fast training
        
        Args:
            model: PyTorch model
            device: Target device
            compile_model: Whether to compile model
        
        Returns:
            Optimized model
        """
        model.train()
        model = model.to(device)
        
        # Compile if requested and available
        if compile_model and hasattr(torch, 'compile'):
            model = SpeedOptimizer.compile_model(model, mode="reduce-overhead")
        
        # Enable optimizations
        if device == "cuda":
            SpeedOptimizer.enable_tf32(model)
            SpeedOptimizer.enable_benchmark_mode()
        
        return model
    
    @staticmethod
    def fast_backward(
        loss: torch.Tensor,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        retain_graph: bool = False
    ) -> None:
        """
        Fast backward pass with optimizations
        
        Args:
            loss: Loss tensor
            scaler: GradScaler for mixed precision
            retain_graph: Whether to retain graph
        """
        if scaler is not None:
            scaler.scale(loss).backward(retain_graph=retain_graph)
        else:
            loss.backward(retain_graph=retain_graph)


class MemoryOptimizer:
    """
    Memory optimizations for training
    """
    
    @staticmethod
    def enable_gradient_checkpointing(model: nn.Module) -> nn.Module:
        """
        Enable gradient checkpointing to save memory (slower but uses less memory)
        
        Args:
            model: PyTorch model
        
        Returns:
            Model with gradient checkpointing enabled
        """
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        elif hasattr(model, 'encoder'):
            # For transformer models
            if hasattr(model.encoder, 'gradient_checkpointing_enable'):
                model.encoder.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled for encoder")
        return model
    
    @staticmethod
    def clear_cache():
        """Clear GPU cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


class BatchOptimizer:
    """
    Batch processing optimizations
    """
    
    @staticmethod
    def prepare_batch(batch: Dict[str, Any], device: str = "cuda", non_blocking: bool = True) -> Dict[str, Any]:
        """
        Prepare batch for fast GPU transfer
        
        Args:
            batch: Batch dictionary
            device: Target device
            non_blocking: Non-blocking transfer
        
        Returns:
            Prepared batch
        """
        prepared = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                prepared[key] = value.to(device, non_blocking=non_blocking)
            elif isinstance(value, (list, tuple)):
                prepared[key] = [
                    v.to(device, non_blocking=non_blocking) if isinstance(v, torch.Tensor) else v
                    for v in value
                ]
            else:
                prepared[key] = value
        return prepared
    
    @staticmethod
    def collate_fast(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Fast collate function with optimizations
        
        Args:
            batch: List of samples
        
        Returns:
            Batched dictionary
        """
        # Stack tensors efficiently
        result = {}
        for key in batch[0].keys():
            values = [sample[key] for sample in batch]
            if isinstance(values[0], torch.Tensor):
                result[key] = torch.stack(values)
            else:
                result[key] = values
        return result













