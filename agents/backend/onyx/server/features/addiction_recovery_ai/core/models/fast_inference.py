"""
Ultra-Fast Inference Optimizations
Maximum speed optimizations for production deployment
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Any
import logging
from functools import lru_cache
import numpy as np

logger = logging.getLogger(__name__)

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class FastInferenceEngine:
    """
    Ultra-fast inference engine with aggressive optimizations
    - Model compilation
    - Batch processing
    - Caching
    - Quantization
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        use_jit: bool = True,
        use_compile: bool = True,
        use_quantization: bool = True,
        cache_size: int = 1000
    ):
        """
        Initialize fast inference engine
        
        Args:
            model: PyTorch model
            device: Device to use
            use_jit: Use JIT compilation
            use_compile: Use torch.compile (PyTorch 2.0+)
            use_quantization: Use INT8 quantization
            cache_size: LRU cache size
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        
        # Apply optimizations
        if use_quantization and self.device.type == "cpu":
            try:
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {nn.Linear}, dtype=torch.qint8
                )
                logger.info("Model quantized to INT8")
            except Exception as e:
                logger.warning(f"Quantization failed: {e}")
        
        if use_jit:
            try:
                # JIT script for faster inference
                self.model = torch.jit.script(self.model)
                logger.info("Model JIT compiled")
            except Exception as e:
                logger.warning(f"JIT compilation failed: {e}")
        
        if use_compile and hasattr(torch, 'compile') and self.device.type == "cuda":
            try:
                self.model = torch.compile(
                    self.model,
                    mode="max-autotune",  # Most aggressive optimization
                    fullgraph=True
                )
                logger.info("Model compiled with torch.compile (max-autotune)")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")
        
        # Warmup
        self._warmup()
        
        # Cache
        self.cache = {}
        self.cache_size = cache_size
    
    def _warmup(self, num_warmup: int = 10):
        """Warmup model for consistent performance"""
        try:
            dummy_input = torch.randn(1, 10).to(self.device)
            with torch.no_grad():
                for _ in range(num_warmup):
                    _ = self.model(dummy_input)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            logger.info(f"Model warmed up with {num_warmup} iterations")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")
    
    @torch.inference_mode()  # Faster than torch.no_grad()
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Fast prediction with inference mode
        
        Args:
            inputs: Input tensor
            
        Returns:
            Predictions
        """
        if inputs.device != self.device:
            inputs = inputs.to(self.device, non_blocking=True)
        
        # Use autocast for mixed precision if on GPU
        if self.device.type == "cuda":
            with torch.cuda.amp.autocast():
                outputs = self.model(inputs)
        else:
            outputs = self.model(inputs)
        
        return outputs
    
    def predict_batch(
        self,
        inputs: List[torch.Tensor],
        batch_size: int = 64
    ) -> List[torch.Tensor]:
        """
        Batch prediction with optimal batching
        
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
            outputs = self.predict(batch_tensor)
            results.extend(outputs.cpu().split(1))
        
        return results


class CachedTransformer:
    """
    Transformer with aggressive caching for repeated inputs
    """
    
    def __init__(
        self,
        model_name: str,
        device: Optional[torch.device] = None,
        use_mixed_precision: bool = True
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_mixed_precision = use_mixed_precision
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required")
        
        # Load model
        torch_dtype = torch.float16 if use_mixed_precision else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch_dtype
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
        
        # Compile if available
        if hasattr(torch, 'compile') and self.device.type == "cuda":
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
            except:
                pass
        
        # Cache for embeddings
        self.embedding_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    @torch.inference_mode()
    def encode(self, text: str, use_cache: bool = True) -> torch.Tensor:
        """
        Encode text with caching
        
        Args:
            text: Input text
            use_cache: Use cache
            
        Returns:
            Embedding tensor
        """
        # Check cache
        if use_cache and text in self.embedding_cache:
            self.cache_hits += 1
            return self.embedding_cache[text]
        
        # Encode
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        if self.use_mixed_precision:
            with torch.cuda.amp.autocast():
                outputs = self.model(**inputs, output_hidden_states=True)
        else:
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # Use CLS token embedding
        embedding = outputs.hidden_states[-1][0, 0, :].cpu()
        
        # Cache
        if use_cache:
            self.embedding_cache[text] = embedding
            self.cache_misses += 1
        
        return embedding
    
    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        use_cache: bool = True
    ) -> List[torch.Tensor]:
        """Encode batch of texts"""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Check cache for each
            uncached = []
            uncached_indices = []
            
            for idx, text in enumerate(batch_texts):
                if use_cache and text in self.embedding_cache:
                    results.append(self.embedding_cache[text])
                    self.cache_hits += 1
                else:
                    uncached.append(text)
                    uncached_indices.append(len(results) + len(uncached_indices))
            
            # Process uncached texts
            if uncached:
                inputs = self.tokenizer(
                    uncached,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)
                
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**inputs, output_hidden_states=True)
                else:
                    outputs = self.model(**inputs, output_hidden_states=True)
                
                embeddings = outputs.hidden_states[-1][:, 0, :].cpu()
                
                # Store results and cache
                for idx, (text, emb) in enumerate(zip(uncached, embeddings)):
                    if use_cache:
                        self.embedding_cache[text] = emb
                    self.cache_misses += 1
                    results.insert(uncached_indices[idx], emb)
        
        return results
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cache_size": len(self.embedding_cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        }


class OptimizedDataLoader:
    """
    Optimized DataLoader with prefetching and pinning
    """
    
    @staticmethod
    def create_loader(
        dataset,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        persistent_workers: bool = True
    ):
        """
        Create optimized DataLoader
        
        Args:
            dataset: PyTorch dataset
            batch_size: Batch size
            num_workers: Number of worker processes
            pin_memory: Pin memory for faster GPU transfer
            prefetch_factor: Prefetch factor
            persistent_workers: Keep workers alive
            
        Returns:
            Optimized DataLoader
        """
        from torch.utils.data import DataLoader
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # Faster for inference
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            drop_last=False
        )


def create_fast_engine(
    model: nn.Module,
    device: Optional[torch.device] = None,
    **kwargs
) -> FastInferenceEngine:
    """Factory for fast inference engine"""
    return FastInferenceEngine(model, device, **kwargs)


def create_cached_transformer(
    model_name: str,
    device: Optional[torch.device] = None,
    **kwargs
) -> CachedTransformer:
    """Factory for cached transformer"""
    return CachedTransformer(model_name, device, **kwargs)













