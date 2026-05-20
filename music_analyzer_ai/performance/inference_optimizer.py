"""
Inference Optimization
Batching, caching, and quantization for faster inference
"""

from typing import List, Dict, Any, Optional, Callable
import logging
import time
from collections import deque
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.jit import script
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class BatchRequest:
    """Request for batched inference"""
    input_data: Any
    future: Any
    timestamp: float


class InferenceBatcher:
    """
    Batches inference requests for better GPU utilization
    """
    
    def __init__(
        self,
        model: nn.Module,
        batch_size: int = 32,
        max_wait_time: float = 0.1,
        device: str = "cuda"
    ):
        self.model = model
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.device = device
        
        self.queue: deque = deque()
        self.lock = threading.Lock()
        self.running = False
        
        # Move model to device and set to eval
        self.model.to(device)
        self.model.eval()
    
    def process_batch(self, batch: List[Any]) -> List[Any]:
        """Process a batch of inputs"""
        try:
            # Stack inputs
            if isinstance(batch[0], torch.Tensor):
                batch_tensor = torch.stack(batch).to(self.device)
            else:
                # Convert to tensor if needed
                batch_tensor = torch.tensor(batch).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(batch_tensor)
            
            # Split outputs
            results = [outputs[i].cpu() for i in range(len(batch))]
            return results
        
        except Exception as e:
            logger.error(f"Batch processing error: {str(e)}")
            return [None] * len(batch)
    
    def add_request(self, input_data: Any) -> Any:
        """Add inference request to batch"""
        import asyncio
        future = asyncio.Future()
        
        with self.lock:
            self.queue.append((input_data, future, time.time()))
        
        return future
    
    async def process_queue(self):
        """Process queued requests in batches"""
        while self.running:
            await asyncio.sleep(0.01)  # Small delay
            
            with self.lock:
                if len(self.queue) == 0:
                    continue
                
                # Get batch
                batch = []
                futures = []
                current_time = time.time()
                
                while len(batch) < self.batch_size and len(self.queue) > 0:
                    input_data, future, timestamp = self.queue.popleft()
                    
                    # Check if we should wait for more
                    if len(batch) > 0 and (current_time - timestamp) < self.max_wait_time:
                        batch.append(input_data)
                        futures.append(future)
                    elif len(batch) == 0:
                        batch.append(input_data)
                        futures.append(future)
                    else:
                        # Put back if not ready
                        self.queue.appendleft((input_data, future, timestamp))
                        break
                
                if len(batch) > 0:
                    # Process batch
                    results = self.process_batch(batch)
                    
                    # Set futures
                    for future, result in zip(futures, results):
                        if not future.done():
                            future.set_result(result)


class ModelQuantizer:
    """
    Model quantization for faster inference
    Supports INT8 quantization
    """
    
    @staticmethod
    def quantize_dynamic(model: nn.Module) -> nn.Module:
        """Dynamic quantization (INT8)"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for quantization")
        
        try:
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d},
                dtype=torch.qint8
            )
            logger.info("Model quantized to INT8")
            return quantized_model
        except Exception as e:
            logger.warning(f"Quantization failed: {str(e)}")
            return model
    
    @staticmethod
    def quantize_static(
        model: nn.Module,
        calibration_data: List[torch.Tensor]
    ) -> nn.Module:
        """Static quantization with calibration"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for quantization")
        
        try:
            model.eval()
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(model, inplace=True)
            
            # Calibrate
            with torch.no_grad():
                for data in calibration_data:
                    model(data)
            
            # Convert
            quantized_model = torch.quantization.convert(model, inplace=False)
            logger.info("Model quantized with static quantization")
            return quantized_model
        except Exception as e:
            logger.warning(f"Static quantization failed: {str(e)}")
            return model


class TorchScriptCompiler:
    """
    Compile models to TorchScript for faster inference
    """
    
    @staticmethod
    def compile_trace(model: nn.Module, example_input: torch.Tensor) -> nn.Module:
        """Compile model using tracing"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for TorchScript")
        
        try:
            model.eval()
            traced_model = torch.jit.trace(model, example_input)
            logger.info("Model compiled with TorchScript (trace)")
            return traced_model
        except Exception as e:
            logger.warning(f"TorchScript tracing failed: {str(e)}")
            return model
    
    @staticmethod
    def compile_script(model: nn.Module) -> nn.Module:
        """Compile model using scripting"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for TorchScript")
        
        try:
            scripted_model = torch.jit.script(model)
            logger.info("Model compiled with TorchScript (script)")
            return scripted_model
        except Exception as e:
            logger.warning(f"TorchScript scripting failed: {str(e)}")
            return model


class OptimizedInferenceEngine:
    """
    Optimized inference engine with:
    - Batching
    - Caching
    - Quantization
    - TorchScript compilation
    - torch.compile (PyTorch 2.0+)
    """
    
    def __init__(
        self,
        model: nn.Module,
        use_batching: bool = True,
        use_quantization: bool = False,
        use_torchscript: bool = False,
        use_compile: bool = True,  # Use torch.compile (fastest)
        batch_size: int = 32,
        device: str = "cuda"
    ):
        self.model = model
        self.device = device
        
        # torch.compile (fastest, PyTorch 2.0+)
        if use_compile and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                logger.info("Model compiled with torch.compile for fastest inference")
            except Exception as e:
                logger.warning(f"torch.compile failed: {str(e)}")
        
        # Quantization (if compile not available or as additional optimization)
        if use_quantization and not use_compile:
            self.model = ModelQuantizer.quantize_dynamic(self.model)
        
        # TorchScript (fallback if compile not available)
        if use_torchscript and not use_compile and not use_quantization:
            # Need example input for tracing
            example_input = torch.randn(1, 169).to(device)
            self.model = TorchScriptCompiler.compile_trace(self.model, example_input)
        
        # Enable optimizations
        if device == "cuda":
            torch.backends.cudnn.benchmark = True
        
        # Batching
        self.batcher = None
        if use_batching:
            self.batcher = InferenceBatcher(
                self.model,
                batch_size=batch_size,
                device=device
            )
            self.batcher.running = True
        
        # Cache
        self.cache: Dict[str, Any] = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def infer(self, input_data: Any, use_cache: bool = True) -> Any:
        """Run inference with optimizations"""
        # Cache lookup
        if use_cache:
            cache_key = self._get_cache_key(input_data)
            if cache_key in self.cache:
                self.cache_hits += 1
                return self.cache[cache_key]
            self.cache_misses += 1
        
        # Inference
        if self.batcher:
            # Use batcher
            future = self.batcher.add_request(input_data)
            result = future.result()  # Wait for result
        else:
            # Direct inference with optimizations
            if isinstance(input_data, torch.Tensor):
                input_tensor = input_data.to(self.device, non_blocking=True)
            else:
                input_tensor = torch.tensor(input_data).to(self.device, non_blocking=True)
            
            with torch.no_grad():
                # Use autocast for faster inference
                if self.device == "cuda":
                    with torch.cuda.amp.autocast():
                        result = self.model(input_tensor)
                else:
                    result = self.model(input_tensor)
        
        # Cache result
        if use_cache:
            self.cache[cache_key] = result
        
        return result
    
    def _get_cache_key(self, input_data: Any) -> str:
        """Generate cache key from input"""
        if isinstance(input_data, torch.Tensor):
            return str(input_data.cpu().numpy().tobytes())
        return str(input_data)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        
        return {
            "cache_size": len(self.cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate
        }
    
    def clear_cache(self):
        """Clear inference cache"""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

