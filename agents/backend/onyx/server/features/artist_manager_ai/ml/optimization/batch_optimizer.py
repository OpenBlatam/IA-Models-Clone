"""
Batch Optimizer
===============

Advanced batch processing optimizations.
"""

import torch
import torch.nn as nn
import logging
from typing import List, Optional, Callable
from collections import deque
import threading

logger = logging.getLogger(__name__)


class SmartBatchProcessor:
    """
    Smart batch processor with adaptive batching.
    
    Features:
    - Adaptive batch sizing
    - Dynamic batching
    - Memory-aware batching
    - Parallel processing
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        initial_batch_size: int = 32,
        max_batch_size: int = 128,
        memory_threshold: float = 0.9
    ):
        """
        Initialize smart batch processor.
        
        Args:
            model: PyTorch model
            device: Device
            initial_batch_size: Initial batch size
            max_batch_size: Maximum batch size
            memory_threshold: Memory threshold (0-1)
        """
        self.model = model.to(device)
        self.device = device
        self.current_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.memory_threshold = memory_threshold
        self._logger = logger
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage."""
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
            total = torch.cuda.get_device_properties(self.device).total_memory
            return (reserved / total) if total > 0 else 0.0
        return 0.0
    
    def _adjust_batch_size(self, success: bool):
        """Adjust batch size based on success."""
        if success:
            # Increase batch size if memory allows
            if self._get_memory_usage() < self.memory_threshold:
                self.current_batch_size = min(
                    self.current_batch_size * 2,
                    self.max_batch_size
                )
        else:
            # Decrease batch size on failure
            self.current_batch_size = max(
                self.current_batch_size // 2,
                1
            )
    
    @torch.no_grad()
    def process_batch(
        self,
        inputs: torch.Tensor,
        use_amp: bool = True
    ) -> torch.Tensor:
        """
        Process batch with adaptive sizing.
        
        Args:
            inputs: Input tensor
            use_amp: Use automatic mixed precision
        
        Returns:
            Predictions
        """
        results = []
        num_samples = inputs.size(0)
        
        for i in range(0, num_samples, self.current_batch_size):
            batch = inputs[i:i + self.current_batch_size].to(self.device)
            
            try:
                if use_amp and self.device.type == "cuda":
                    with torch.cuda.amp.autocast():
                        output = self.model(batch)
                else:
                    output = self.model(batch)
                
                results.append(output.cpu())
                self._adjust_batch_size(True)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    self._adjust_batch_size(False)
                    # Retry with smaller batch
                    if self.current_batch_size > 1:
                        batch = inputs[i:i + self.current_batch_size].to(self.device)
                        if use_amp and self.device.type == "cuda":
                            with torch.cuda.amp.autocast():
                                output = self.model(batch)
                        else:
                            output = self.model(batch)
                        results.append(output.cpu())
                else:
                    raise
        
        return torch.cat(results, dim=0) if results else torch.empty(0)


class ParallelBatchProcessor:
    """
    Parallel batch processor for multi-GPU.
    """
    
    def __init__(
        self,
        model: nn.Module,
        devices: List[torch.device],
        batch_size_per_device: int = 32
    ):
        """
        Initialize parallel batch processor.
        
        Args:
            model: PyTorch model
            devices: List of devices
            batch_size_per_device: Batch size per device
        """
        self.devices = devices
        self.batch_size_per_device = batch_size_per_device
        
        # Replicate model to devices
        if len(devices) > 1:
            self.models = torch.nn.DataParallel(model, device_ids=[d.index for d in devices if d.type == "cuda"])
        else:
            self.models = model.to(devices[0])
        
        self._logger = logger
    
    @torch.no_grad()
    def process_parallel(
        self,
        inputs: torch.Tensor,
        use_amp: bool = True
    ) -> torch.Tensor:
        """
        Process in parallel across devices.
        
        Args:
            inputs: Input tensor
            use_amp: Use automatic mixed precision
        
        Returns:
            Predictions
        """
        if use_amp and any(d.type == "cuda" for d in self.devices):
            with torch.cuda.amp.autocast():
                outputs = self.models(inputs)
        else:
            outputs = self.models(inputs)
        
        return outputs




