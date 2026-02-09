"""
Ultra-Fast Music Generator with Maximum Performance Optimizations

Implements:
- torch.compile with optimal modes
- Advanced caching strategies
- Batch processing optimization
- Async inference
- Memory optimizations
- Model pre-loading
- Optimized tokenization
- JIT compilation where possible
"""

import logging
import asyncio
from typing import Optional, Dict, Any, List, Tuple
from functools import lru_cache
from pathlib import Path
import torch
import torch.nn as nn
import torch._dynamo as dynamo
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import hashlib
import pickle

from config.settings import settings

logger = logging.getLogger(__name__)

# Try to import numba for JIT compilation
try:
    from numba import jit, prange
    USE_NUMBA = True
except ImportError:
    USE_NUMBA = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


@jit(nopython=True, parallel=True) if USE_NUMBA else lambda x: x
def fast_normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Fast audio normalization with numba JIT."""
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        return audio / max_val
    return audio


class UltraFastCache:
    """High-performance cache with LRU and disk persistence."""
    
    def __init__(self, max_size: int = 1000, use_disk: bool = False, cache_dir: str = "./cache"):
        self.memory_cache = {}
        self.max_size = max_size
        self.use_disk = use_disk
        self.cache_dir = Path(cache_dir)
        if use_disk:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_key_hash(self, key: Tuple) -> str:
        """Generate hash for cache key."""
        key_str = str(key).encode('utf-8')
        return hashlib.md5(key_str).hexdigest()
    
    def get(self, key: Tuple) -> Optional[np.ndarray]:
        """Get from cache."""
        # Try memory first
        if key in self.memory_cache:
            return self.memory_cache[key]
        
        # Try disk
        if self.use_disk:
            key_hash = self._get_key_hash(key)
            cache_file = self.cache_dir / f"{key_hash}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                except Exception as e:
                    logger.debug(f"Cache read error: {e}")
        
        return None
    
    def set(self, key: Tuple, value: np.ndarray) -> None:
        """Set cache value."""
        # Memory cache
        if len(self.memory_cache) >= self.max_size:
            # Remove oldest (simple FIFO)
            oldest_key = next(iter(self.memory_cache))
            del self.memory_cache[oldest_key]
        
        self.memory_cache[key] = value
        
        # Disk cache
        if self.use_disk:
            key_hash = self._get_key_hash(key)
            cache_file = self.cache_dir / f"{key_hash}.pkl"
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(value, f)
            except Exception as e:
                logger.debug(f"Cache write error: {e}")
    
    def clear(self) -> None:
        """Clear all caches."""
        self.memory_cache.clear()
        if self.use_disk:
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    cache_file.unlink()
                except Exception:
                    pass


class UltraFastMusicGenerator:
    """
    Ultra-fast music generator with maximum performance optimizations.
    
    Speed optimizations:
    - torch.compile with optimal mode
    - Advanced caching (memory + disk)
    - Batch processing
    - Async inference
    - Optimized tokenization
    - Memory-efficient operations
    """
    
    def __init__(
        self,
        compile_mode: str = "max-autotune",
        use_cache: bool = True,
        use_disk_cache: bool = True,
        enable_async: bool = True,
        max_workers: int = 4,
        preload_model: bool = True
    ):
        """
        Initialize ultra-fast generator.
        
        Args:
            compile_mode: torch.compile mode (default, reduce-overhead, max-autotune)
            use_cache: Enable caching
            use_disk_cache: Enable disk caching
            enable_async: Enable async inference
            max_workers: Max workers for async operations
            preload_model: Pre-load model on initialization
        """
        self.device = self._get_device()
        self.compile_mode = compile_mode
        self.enable_async = enable_async
        self.max_workers = max_workers
        
        # Setup cache
        self.cache = UltraFastCache(
            max_size=1000,
            use_disk=use_disk_cache
        ) if use_cache else None
        
        # Model components
        self.model = None
        self.processor = None
        self.compiled_model = None
        self._model_loaded = False
        
        # Async executor
        self.executor = ThreadPoolExecutor(max_workers=max_workers) if enable_async else None
        
        # Pre-load model
        if preload_model:
            self._load_model()
    
    def _get_device(self) -> str:
        """Get optimal device."""
        if torch.cuda.is_available() and settings.use_gpu:
            device = "cuda"
            # Set optimal GPU settings
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            logger.info("Using CPU")
        return device
    
    def _load_model(self) -> None:
        """Load and optimize model."""
        if self._model_loaded:
            return
        
        try:
            logger.info(f"Loading model: {settings.music_model}")
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(settings.music_model)
            
            # Load model
            self.model = MusicgenForConditionalGeneration.from_pretrained(
                settings.music_model
            )
            self.model.to(self.device)
            self.model.eval()
            
            # Compile model for maximum speed
            if hasattr(torch, 'compile'):
                try:
                    self.compiled_model = torch.compile(
                        self.model,
                        mode=self.compile_mode,
                        fullgraph=False,
                        dynamic=False
                    )
                    logger.info(f"Model compiled with mode: {self.compile_mode}")
                except Exception as e:
                    logger.warning(f"Could not compile model: {e}")
                    self.compiled_model = self.model
            else:
                self.compiled_model = self.model
            
            # Warmup compilation
            self._warmup_model()
            
            self._model_loaded = True
            logger.info("Model loaded and optimized")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            raise
    
    def _warmup_model(self) -> None:
        """Warmup model to trigger compilation."""
        try:
            logger.info("Warming up model...")
            dummy_text = "test music"
            dummy_inputs = self.processor(
                text=[dummy_text],
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            
            with torch.no_grad():
                if torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        _ = self.compiled_model.generate(
                            **dummy_inputs,
                            max_new_tokens=100,
                            do_sample=False
                        )
                else:
                    _ = self.compiled_model.generate(
                        **dummy_inputs,
                        max_new_tokens=100,
                        do_sample=False
                    )
            
            logger.info("Model warmup complete")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")
    
    def _prepare_inputs(self, text: str) -> Dict[str, torch.Tensor]:
        """Optimized input preparation."""
        # Use processor with optimized settings
        inputs = self.processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        ).to(self.device, non_blocking=True)
        return inputs
    
    def generate_from_text(
        self,
        text: str,
        duration: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        use_cache: Optional[bool] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Ultra-fast generation with all optimizations.
        
        Args:
            text: Song description
            duration: Duration in seconds
            guidance_scale: Guidance scale
            temperature: Temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            use_cache: Override cache setting
            **kwargs: Additional parameters
            
        Returns:
            Generated audio array
        """
        # Ensure model is loaded
        if not self._model_loaded:
            self._load_model()
        
        # Check cache
        use_cache = use_cache if use_cache is not None else (self.cache is not None)
        if use_cache and self.cache:
            cache_key = (text, duration, guidance_scale, temperature, top_k, top_p)
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                logger.debug("Cache hit")
                return cached_result
        
        # Prepare parameters
        duration = duration or settings.default_duration
        guidance_scale = guidance_scale or settings.cfg_coef
        temperature = temperature or settings.temperature
        top_k = top_k or settings.top_k
        top_p = top_p or settings.top_p
        
        try:
            # Prepare inputs (optimized)
            inputs = self._prepare_inputs(text)
            
            # Calculate max_new_tokens
            hop_length = getattr(
                self.compiled_model.config.audio_encoder,
                'hop_length',
                512
            )
            max_new_tokens = int(duration * settings.sample_rate / hop_length)
            
            # Generate with all optimizations
            with torch.no_grad():
                if torch.cuda.is_available():
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        audio_values = self.compiled_model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            guidance_scale=guidance_scale,
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                            **kwargs
                        )
                else:
                    audio_values = self.compiled_model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        guidance_scale=guidance_scale,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        **kwargs
                    )
            
            # Convert to numpy (optimized)
            audio = audio_values[0, 0].cpu().numpy()
            
            # Fast normalization
            audio = fast_normalize_audio(audio)
            
            # Cache result
            if use_cache and self.cache:
                cache_key = (text, duration, guidance_scale, temperature, top_k, top_p)
                self.cache.set(cache_key, audio)
            
            return audio
            
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise RuntimeError("GPU out of memory")
        except Exception as e:
            logger.error(f"Generation error: {e}", exc_info=True)
            raise
    
    async def generate_async(
        self,
        text: str,
        duration: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Async generation for non-blocking inference.
        
        Args:
            text: Song description
            duration: Duration in seconds
            **kwargs: Additional parameters
            
        Returns:
            Generated audio array
        """
        if not self.enable_async or not self.executor:
            return self.generate_from_text(text, duration=duration, **kwargs)
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self.generate_from_text,
            text,
            duration,
            **kwargs
        )
        return result
    
    def generate_batch(
        self,
        texts: List[str],
        duration: Optional[int] = None,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> List[np.ndarray]:
        """
        Optimized batch generation.
        
        Args:
            texts: List of prompts
            duration: Duration in seconds
            batch_size: Batch size (auto if None)
            **kwargs: Additional parameters
            
        Returns:
            List of audio arrays
        """
        if not self._model_loaded:
            self._load_model()
        
        # Auto batch size based on GPU memory
        if batch_size is None:
            if torch.cuda.is_available():
                free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
                batch_size = min(4, max(1, int(free_memory / (2 * 1024**3))))  # Rough estimate
            else:
                batch_size = 1
        
        results = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            try:
                # Prepare batch inputs
                batch_inputs = self.processor(
                    text=batch_texts,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device, non_blocking=True)
                
                # Calculate max_new_tokens
                hop_length = getattr(
                    self.compiled_model.config.audio_encoder,
                    'hop_length',
                    512
                )
                max_new_tokens = int((duration or settings.default_duration) * settings.sample_rate / hop_length)
                
                # Generate batch
                with torch.no_grad():
                    if torch.cuda.is_available():
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            audio_batch = self.compiled_model.generate(
                                **batch_inputs,
                                max_new_tokens=max_new_tokens,
                                **kwargs
                            )
                    else:
                        audio_batch = self.compiled_model.generate(
                            **batch_inputs,
                            max_new_tokens=max_new_tokens,
                            **kwargs
                        )
                
                # Convert to numpy
                for j in range(len(batch_texts)):
                    audio = audio_batch[j, 0].cpu().numpy()
                    audio = fast_normalize_audio(audio)
                    results.append(audio)
                    
            except Exception as e:
                logger.error(f"Batch generation error: {e}")
                # Fallback to individual generation
                for text in batch_texts:
                    try:
                        audio = self.generate_from_text(text, duration=duration, **kwargs)
                        results.append(audio)
                    except Exception:
                        results.append(None)
        
        return results
    
    async def generate_batch_async(
        self,
        texts: List[str],
        duration: Optional[int] = None,
        **kwargs
    ) -> List[np.ndarray]:
        """
        Async batch generation.
        
        Args:
            texts: List of prompts
            duration: Duration in seconds
            **kwargs: Additional parameters
            
        Returns:
            List of audio arrays
        """
        if not self.enable_async:
            return self.generate_batch(texts, duration=duration, **kwargs)
        
        # Create tasks for parallel generation
        tasks = [
            self.generate_async(text, duration=duration, **kwargs)
            for text in texts
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error generating for text {i}: {result}")
                processed_results.append(None)
            else:
                processed_results.append(result)
        
        return processed_results
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        if self.cache:
            self.cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def __del__(self):
        """Cleanup."""
        if self.executor:
            self.executor.shutdown(wait=False)


# Global instance
_ultra_fast_generator: Optional[UltraFastMusicGenerator] = None


def get_ultra_fast_generator(
    compile_mode: str = "max-autotune",
    use_cache: bool = True
) -> UltraFastMusicGenerator:
    """
    Get global ultra-fast generator instance.
    
    Args:
        compile_mode: torch.compile mode
        use_cache: Enable caching
        
    Returns:
        UltraFastMusicGenerator instance
    """
    global _ultra_fast_generator
    if _ultra_fast_generator is None:
        _ultra_fast_generator = UltraFastMusicGenerator(
            compile_mode=compile_mode,
            use_cache=use_cache
        )
    return _ultra_fast_generator













