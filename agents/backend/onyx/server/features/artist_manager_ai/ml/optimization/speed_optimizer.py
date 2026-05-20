"""
Speed Optimizer
===============

Optimizations for faster training and inference.
"""

import torch
import torch.nn as nn
import logging
from typing import Optional, Dict, Any
from functools import lru_cache

logger = logging.getLogger(__name__)


class SpeedOptimizer:
    """
    Speed optimization utilities.
    
    Features:
    - Model compilation (torch.compile)
    - JIT scripting
    - DataLoader optimization
    - Inference optimization
    """
    
    @staticmethod
    def compile_model(
        model: nn.Module,
        mode: str = "reduce-overhead",
        fullgraph: bool = False
    ) -> nn.Module:
        """
        Compile model with torch.compile for faster execution.
        
        Args:
            model: PyTorch model
            mode: Compilation mode ("default", "reduce-overhead", "max-autotune")
            fullgraph: Whether to compile full graph
        
        Returns:
            Compiled model
        """
        try:
            # torch.compile is available in PyTorch 2.0+
            if hasattr(torch, 'compile'):
                compiled_model = torch.compile(
                    model,
                    mode=mode,
                    fullgraph=fullgraph
                )
                logger.info(f"Model compiled with mode: {mode}")
                return compiled_model
            else:
                logger.warning("torch.compile not available (requires PyTorch 2.0+)")
                return model
        except Exception as e:
            logger.warning(f"Model compilation failed: {str(e)}")
            return model
    
    @staticmethod
    def script_model(model: nn.Module, example_input: torch.Tensor) -> nn.Module:
        """
        Script model with torch.jit.script for faster inference.
        
        Args:
            model: PyTorch model
            example_input: Example input tensor
        
        Returns:
            Scripted model
        """
        try:
            model.eval()
            scripted_model = torch.jit.script(model)
            logger.info("Model scripted successfully")
            return scripted_model
        except Exception as e:
            logger.warning(f"Model scripting failed: {str(e)}")
            return model
    
    @staticmethod
    def optimize_dataloader(
        dataloader: torch.utils.data.DataLoader,
        num_workers: int = 4,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        persistent_workers: bool = True
    ) -> torch.utils.data.DataLoader:
        """
        Optimize DataLoader for faster data loading.
        
        Args:
            dataloader: DataLoader to optimize
            num_workers: Number of worker processes
            pin_memory: Pin memory for faster GPU transfer
            prefetch_factor: Number of batches to prefetch
            persistent_workers: Keep workers alive between epochs
        
        Returns:
            Optimized DataLoader
        """
        # Create new DataLoader with optimized settings
        optimized_loader = torch.utils.data.DataLoader(
            dataloader.dataset,
            batch_size=dataloader.batch_size,
            shuffle=dataloader.shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            drop_last=dataloader.drop_last
        )
        
        logger.info(f"DataLoader optimized: {num_workers} workers, pin_memory={pin_memory}")
        return optimized_loader
    
    @staticmethod
    def enable_cudnn_benchmark():
        """Enable cuDNN benchmark for faster convolutions."""
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True
            logger.info("cuDNN benchmark enabled")
    
    @staticmethod
    def optimize_inference(
        model: nn.Module,
        device: torch.device,
        use_torchscript: bool = True,
        use_compile: bool = True
    ) -> nn.Module:
        """
        Optimize model for inference.
        
        Args:
            model: Model to optimize
            device: Device
            use_torchscript: Use TorchScript
            use_compile: Use torch.compile
        
        Returns:
            Optimized model
        """
        model = model.to(device)
        model.eval()
        
        # Enable optimizations
        if device.type == "cuda":
            SpeedOptimizer.enable_cudnn_benchmark()
        
        # Compile if available
        if use_compile and hasattr(torch, 'compile'):
            model = SpeedOptimizer.compile_model(model, mode="reduce-overhead")
        
        # Script if requested
        if use_torchscript:
            # Create dummy input for scripting
            dummy_input = torch.randn(1, next(model.parameters()).shape[1] if hasattr(next(model.parameters()), 'shape') else 32).to(device)
            try:
                model = SpeedOptimizer.script_model(model, dummy_input)
            except:
                logger.warning("TorchScript failed, using original model")
        
        return model


class FastInference:
    """Fast inference wrapper."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        batch_size: int = 32,
        use_compile: bool = True
    ):
        """
        Initialize fast inference.
        
        Args:
            model: Model
            device: Device
            batch_size: Batch size for inference
            use_compile: Use compilation
        """
        self.device = device
        self.batch_size = batch_size
        
        # Optimize model
        self.model = SpeedOptimizer.optimize_inference(
            model,
            device,
            use_compile=use_compile
        )
        
        self.model.eval()
    
    @torch.no_grad()
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Fast batch prediction.
        
        Args:
            inputs: Input tensor
        
        Returns:
            Predictions
        """
        inputs = inputs.to(self.device)
        
        # Process in batches if needed
        if inputs.size(0) > self.batch_size:
            results = []
            for i in range(0, inputs.size(0), self.batch_size):
                batch = inputs[i:i + self.batch_size]
                with torch.cuda.amp.autocast():
                    output = self.model(batch)
                results.append(output.cpu())
            return torch.cat(results, dim=0)
        else:
            with torch.cuda.amp.autocast():
                return self.model(inputs).cpu()




