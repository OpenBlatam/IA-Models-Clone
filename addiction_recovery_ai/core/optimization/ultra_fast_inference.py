"""
Ultra-Fast Inference Optimizations
Maximum speed with aggressive optimizations
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Any, Callable
import logging
from functools import lru_cache
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import asyncio

logger = logging.getLogger(__name__)


class UltraFastInference:
    """
    Ultra-fast inference with all optimizations
    - Model fusion
    - Pre-computation
    - Async inference
    - Batch optimization
    - Memory optimization
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        enable_all: bool = True
    ):
        """
        Initialize ultra-fast inference
        
        Args:
            model: PyTorch model
            device: Device to use
            enable_all: Enable all optimizations
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        
        if enable_all:
            self._apply_all_optimizations()
    
    def _apply_all_optimizations(self):
        """Apply all possible optimizations"""
        # 1. Fuse operations
        self._fuse_operations()
        
        # 2. Compile with max optimization
        self._compile_model()
        
        # 3. Set to inference mode
        self.model.eval()
        
        # 4. Disable gradient computation
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 5. Enable channels_last for better memory access
        if hasattr(self.model, 'to'):
            try:
                self.model = self.model.to(memory_format=torch.channels_last)
            except:
                pass
        
        # 6. Warmup
        self._warmup()
    
    def _fuse_operations(self):
        """Fuse operations for faster inference"""
        try:
            # Fuse Conv+BN+ReLU
            if hasattr(torch.quantization, 'fuse_modules'):
                torch.quantization.fuse_modules(
                    self.model,
                    [['conv', 'bn', 'relu']],
                    inplace=True
                )
        except Exception as e:
            logger.debug(f"Fusion failed: {e}")
    
    def _compile_model(self):
        """Compile model with maximum optimization"""
        if hasattr(torch, 'compile') and self.device.type == "cuda":
            try:
                self.model = torch.compile(
                    self.model,
                    mode="max-autotune",
                    fullgraph=True,
                    dynamic=False
                )
                logger.info("Model compiled with max-autotune")
            except Exception as e:
                logger.warning(f"Compilation failed: {e}")
    
    def _warmup(self, num_iterations: int = 20):
        """Warmup model"""
        try:
            # Create dummy input
            dummy_input = torch.randn(1, 10).to(self.device)
            
            with torch.inference_mode():
                for _ in range(num_iterations):
                    _ = self.model(dummy_input)
            
            if self.device.type == "cuda":
                torch.cuda.synchronize()
        except Exception as e:
            logger.debug(f"Warmup failed: {e}")
    
    @torch.inference_mode()
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Ultra-fast prediction
        
        Args:
            inputs: Input tensor
            
        Returns:
            Predictions
        """
        if inputs.device != self.device:
            inputs = inputs.to(self.device, non_blocking=True)
        
        # Use autocast for mixed precision
        if self.device.type == "cuda":
            with torch.cuda.amp.autocast():
                outputs = self.model(inputs)
        else:
            outputs = self.model(inputs)
        
        return outputs
    
    def predict_batch_optimized(
        self,
        inputs: List[torch.Tensor],
        batch_size: int = 128
    ) -> List[torch.Tensor]:
        """
        Optimized batch prediction with memory efficiency
        
        Args:
            inputs: List of input tensors
            batch_size: Batch size
            
        Returns:
            List of predictions
        """
        results = []
        
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            batch_tensor = torch.stack(batch).to(self.device, non_blocking=True)
            
            # Predict
            outputs = self.predict(batch_tensor)
            
            # Move to CPU and split
            outputs_cpu = outputs.cpu()
            results.extend(outputs_cpu.split(1))
            
            # Clear cache periodically
            if i % (batch_size * 10) == 0 and self.device.type == "cuda":
                torch.cuda.empty_cache()
        
        return results


class AsyncInferenceEngine:
    """
    Async inference engine for non-blocking predictions
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        max_workers: int = 4
    ):
        """
        Initialize async inference engine
        
        Args:
            model: PyTorch model
            device: Device to use
            max_workers: Maximum worker threads
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def predict_async(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Async prediction
        
        Args:
            inputs: Input tensor
            
        Returns:
            Predictions
        """
        loop = asyncio.get_event_loop()
        
        def _predict():
            with torch.inference_mode():
                if inputs.device != self.device:
                    inputs_device = inputs.to(self.device, non_blocking=True)
                else:
                    inputs_device = inputs
                
                if self.device.type == "cuda":
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs_device)
                else:
                    outputs = self.model(inputs_device)
                
                return outputs.cpu()
        
        return await loop.run_in_executor(self.executor, _predict)
    
    async def predict_batch_async(
        self,
        inputs: List[torch.Tensor],
        batch_size: int = 32
    ) -> List[torch.Tensor]:
        """
        Async batch prediction
        
        Args:
            inputs: List of inputs
            batch_size: Batch size
            
        Returns:
            List of predictions
        """
        tasks = []
        
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            batch_tensor = torch.stack(batch)
            tasks.append(self.predict_async(batch_tensor))
        
        results = await asyncio.gather(*tasks)
        
        # Flatten results
        all_results = []
        for batch_results in results:
            all_results.extend(batch_results.split(1))
        
        return all_results
    
    def shutdown(self):
        """Shutdown executor"""
        self.executor.shutdown(wait=True)


class EmbeddingCache:
    """
    Intelligent embedding cache with LRU eviction
    """
    
    def __init__(self, max_size: int = 10000, device: Optional[torch.device] = None):
        """
        Initialize embedding cache
        
        Args:
            max_size: Maximum cache size
            device: Device for cached embeddings
        """
        self.max_size = max_size
        self.device = device
        self.cache: Dict[str, torch.Tensor] = {}
        self.access_order: List[str] = []
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[torch.Tensor]:
        """
        Get embedding from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached embedding or None
        """
        if key in self.cache:
            # Update access order
            self.access_order.remove(key)
            self.access_order.append(key)
            self.hits += 1
            return self.cache[key]
        
        self.misses += 1
        return None
    
    def put(self, key: str, embedding: torch.Tensor):
        """
        Put embedding in cache
        
        Args:
            key: Cache key
            embedding: Embedding tensor
        """
        # Evict if needed
        if len(self.cache) >= self.max_size and key not in self.cache:
            # Remove least recently used
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]
        
        self.cache[key] = embedding
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate
        }
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.access_order.clear()
        self.hits = 0
        self.misses = 0


class BatchOptimizer:
    """
    Optimize batch processing for maximum throughput
    """
    
    @staticmethod
    def create_optimal_batches(
        items: List[Any],
        batch_size: int,
        sort_by_length: bool = False
    ) -> List[List[Any]]:
        """
        Create optimal batches
        
        Args:
            items: List of items
            batch_size: Batch size
            sort_by_length: Sort by length for better batching
            
        Returns:
            List of batches
        """
        if sort_by_length and hasattr(items[0], '__len__'):
            # Sort by length for better padding efficiency
            items = sorted(items, key=len, reverse=True)
        
        batches = []
        for i in range(0, len(items), batch_size):
            batches.append(items[i:i + batch_size])
        
        return batches
    
    @staticmethod
    def pad_batch(
        sequences: List[torch.Tensor],
        pad_value: float = 0.0,
        max_length: Optional[int] = None
    ) -> torch.Tensor:
        """
        Efficient batch padding
        
        Args:
            sequences: List of sequences
            pad_value: Padding value
            max_length: Maximum length (None for auto)
            
        Returns:
            Padded batch tensor
        """
        if max_length is None:
            max_length = max(len(seq) for seq in sequences)
        
        batch_size = len(sequences)
        feature_dim = sequences[0].shape[-1] if sequences[0].dim() > 1 else 1
        
        # Pre-allocate tensor
        if sequences[0].dim() > 1:
            batch_tensor = torch.full(
                (batch_size, max_length, feature_dim),
                pad_value,
                dtype=sequences[0].dtype,
                device=sequences[0].device
            )
        else:
            batch_tensor = torch.full(
                (batch_size, max_length),
                pad_value,
                dtype=sequences[0].dtype,
                device=sequences[0].device
            )
        
        # Fill with sequences
        for i, seq in enumerate(sequences):
            length = min(len(seq), max_length)
            if seq.dim() > 1:
                batch_tensor[i, :length] = seq[:length]
            else:
                batch_tensor[i, :length] = seq[:length]
        
        return batch_tensor


def create_ultra_fast_inference(
    model: nn.Module,
    device: Optional[torch.device] = None
) -> UltraFastInference:
    """Factory for ultra-fast inference"""
    return UltraFastInference(model, device, enable_all=True)


def create_async_engine(
    model: nn.Module,
    device: Optional[torch.device] = None
) -> AsyncInferenceEngine:
    """Factory for async inference engine"""
    return AsyncInferenceEngine(model, device)


def create_embedding_cache(
    max_size: int = 10000,
    device: Optional[torch.device] = None
) -> EmbeddingCache:
    """Factory for embedding cache"""
    return EmbeddingCache(max_size, device)













