"""
ML Model Manager - Advanced model management with caching, batching, and optimization
"""

from typing import Dict, Optional, Any, List, Callable
from dataclasses import dataclass
from enum import Enum
import logging
import time
import hashlib
import pickle
from pathlib import Path
import threading
from collections import defaultdict

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types of ML models supported"""
    SKIN_ANALYSIS = "skin_analysis"
    TEXTURE_ANALYSIS = "texture_analysis"
    RECOMMENDATION = "recommendation"
    CONDITION_PREDICTION = "condition_prediction"
    AGE_ESTIMATION = "age_estimation"
    CUSTOM = "custom"


@dataclass
class ModelConfig:
    """Configuration for a model"""
    model_id: str
    model_type: ModelType
    model_path: Optional[str] = None
    model_class: Optional[Callable] = None
    device: str = "cpu"  # "cpu", "cuda", "mps"
    batch_size: int = 32
    use_cache: bool = True
    cache_ttl: int = 3600
    precision: str = "float32"  # "float32", "float16", "bfloat16"
    optimize_for_inference: bool = True
    max_memory_mb: Optional[int] = None


@dataclass
class InferenceResult:
    """Result of model inference"""
    prediction: Any
    confidence: float
    processing_time: float
    model_id: str
    cached: bool = False
    metadata: Dict[str, Any] = None


class MLModelManager:
    """
    Advanced ML Model Manager with:
    - Model loading and caching
    - Batch processing optimization
    - GPU/CPU management
    - Inference caching
    - Memory management
    - Lazy loading
    - Prefetching
    """
    
    def __init__(self, cache_dir: Optional[str] = None, lazy_load: bool = True):
        self.models: Dict[str, Any] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        self.inference_cache: Dict[str, tuple] = {}  # hash -> (result, timestamp)
        self.batch_queue: Dict[str, List] = defaultdict(list)
        self.lock = threading.Lock()
        self.cache_dir = Path(cache_dir) if cache_dir else Path("/tmp/ml_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.lazy_load = lazy_load
        self.loading_models: Dict[str, threading.Event] = {}
        
        # Statistics
        self.stats = {
            "inferences": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "batch_processed": 0,
            "total_time": 0.0,
            "lazy_loads": 0
        }
    
    def register_model(self, config: ModelConfig):
        """Register a model configuration"""
        with self.lock:
            self.model_configs[config.model_id] = config
            logger.info(f"Registered model: {config.model_id} ({config.model_type.value})")
    
    def load_model(self, model_id: str, force_reload: bool = False) -> Any:
        """Load a model into memory (with lazy loading support)"""
        if model_id in self.models and not force_reload:
            return self.models[model_id]
        
        if model_id not in self.model_configs:
            raise ValueError(f"Model {model_id} not registered")
        
        # Check if already loading (prevent duplicate loads)
        if model_id in self.loading_models:
            event = self.loading_models[model_id]
            event.wait()  # Wait for loading to complete
            if model_id in self.models:
                return self.models[model_id]
        
        # Mark as loading
        loading_event = threading.Event()
        self.loading_models[model_id] = loading_event
        
        try:
            config = self.model_configs[model_id]
            
            # Load model based on type
            if config.model_class:
                model = config.model_class()
            elif config.model_path:
                model = self._load_from_path(config.model_path)
            else:
                raise ValueError(f"No model source specified for {model_id}")
            
            # Move to device
            if config.device != "cpu":
                try:
                    import torch
                    if torch.cuda.is_available() and config.device == "cuda":
                        model = model.to(config.device)
                    elif hasattr(torch.backends, 'mps') and config.device == "mps":
                        model = model.to(config.device)
                except ImportError:
                    logger.warning("PyTorch not available, using CPU")
                    config.device = "cpu"
            
            # Optimize for inference
            if config.optimize_for_inference:
                model = self._optimize_model(model, config)
            
            # Set to eval mode
            if hasattr(model, "eval"):
                model.eval()
            
            with self.lock:
                self.models[model_id] = model
                self.stats["lazy_loads"] += 1
            
            logger.info(f"Loaded model: {model_id} on {config.device}")
            return model
        
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {str(e)}", exc_info=True)
            raise
        finally:
            # Signal loading complete
            loading_event.set()
            if model_id in self.loading_models:
                del self.loading_models[model_id]
    
    def prefetch_model(self, model_id: str):
        """Prefetch a model in background"""
        if model_id in self.models or model_id not in self.model_configs:
            return
        
        import threading
        thread = threading.Thread(target=self.load_model, args=(model_id,), daemon=True)
        thread.start()
    
    def _load_from_path(self, model_path: str) -> Any:
        """Load model from file path"""
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Try different loading methods
        try:
            import torch
            if path.suffix == ".pt" or path.suffix == ".pth":
                return torch.load(model_path, map_location="cpu")
        except ImportError:
            pass
        
        try:
            import pickle
            if path.suffix == ".pkl":
                with open(model_path, "rb") as f:
                    return pickle.load(f)
        except Exception:
            pass
        
        raise ValueError(f"Unsupported model format: {model_path}")
    
    def _optimize_model(self, model: Any, config: ModelConfig) -> Any:
        """Optimize model for inference with advanced techniques"""
        try:
            import torch
            from utils.optimization import (
                compile_model,
                optimize_for_inference,
                enable_cudnn_benchmark,
                enable_tf32
            )
            
            if hasattr(model, "parameters"):
                # Set to eval mode
                if hasattr(model, "eval"):
                    model.eval()
                
                # Enable optimizations
                enable_cudnn_benchmark()
                if config.device != "cpu":
                    enable_tf32()
                
                # Use torch.compile for optimization (PyTorch 2.0+)
                if config.optimize_for_inference:
                    try:
                        # Try torch.compile first (fastest)
                        if hasattr(torch, "compile"):
                            model = compile_model(model, mode="reduce-overhead")
                            logger.info(f"Model {config.model_id} compiled with torch.compile")
                        else:
                            # Fallback: JIT optimization
                            model = optimize_for_inference(model, use_jit=True)
                            logger.info(f"Model {config.model_id} optimized with JIT")
                    except Exception as e:
                        logger.debug(f"Could not optimize model: {e}")
                
                # Use mixed precision if specified
                if config.device != "cpu":
                    if config.precision == "float16":
                        model = model.half()
                    elif config.precision == "bfloat16":
                        model = model.to(torch.bfloat16)
        except ImportError:
            pass
        
        return model
    
    def predict(
        self,
        model_id: str,
        input_data: Any,
        use_cache: Optional[bool] = None,
        batch: bool = False
    ) -> InferenceResult:
        """Run inference on a model"""
        start_time = time.time()
        
        # Get model config
        if model_id not in self.model_configs:
            raise ValueError(f"Model {model_id} not registered")
        
        config = self.model_configs[model_id]
        use_cache = use_cache if use_cache is not None else config.use_cache
        
        # Check cache
        cache_key = None
        if use_cache:
            cache_key = self._generate_cache_key(model_id, input_data)
            if cache_key in self.inference_cache:
                result, timestamp = self.inference_cache[cache_key]
                if time.time() - timestamp < config.cache_ttl:
                    self.stats["cache_hits"] += 1
                    self.stats["inferences"] += 1
                    processing_time = time.time() - start_time
                    self.stats["total_time"] += processing_time
                    return InferenceResult(
                        prediction=result,
                        confidence=1.0,
                        processing_time=processing_time,
                        model_id=model_id,
                        cached=True
                    )
        
        self.stats["cache_misses"] += 1
        
        # Load model if not loaded
        model = self.load_model(model_id)
        
        # Run inference
        try:
            prediction = self._run_inference(model, input_data, config)
            processing_time = time.time() - start_time
            
            # Cache result
            if use_cache and cache_key:
                with self.lock:
                    self.inference_cache[cache_key] = (prediction, time.time())
            
            self.stats["inferences"] += 1
            self.stats["total_time"] += processing_time
            
            return InferenceResult(
                prediction=prediction,
                confidence=self._calculate_confidence(prediction),
                processing_time=processing_time,
                model_id=model_id,
                cached=False
            )
        
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}", exc_info=True)
            raise
    
    def _run_inference(self, model: Any, input_data: Any, config: ModelConfig) -> Any:
        """Run actual inference with optimizations"""
        try:
            import torch
            from torch.cuda.amp import autocast
            
            # Convert input to tensor if needed
            if isinstance(input_data, (list, tuple, np.ndarray)):
                if isinstance(input_data, np.ndarray):
                    input_data = torch.from_numpy(input_data)
                else:
                    input_data = torch.tensor(input_data)
                
                # Move to device
                if config.device != "cpu":
                    input_data = input_data.to(config.device)
                
                # Add batch dimension if needed
                if input_data.dim() == 3:  # (C, H, W)
                    input_data = input_data.unsqueeze(0)
            
            # Run inference with mixed precision if enabled
            model.eval()
            with torch.no_grad():
                if config.precision in ["float16", "bfloat16"] and config.device != "cpu":
                    with autocast():
                        if hasattr(model, "__call__"):
                            output = model(input_data)
                        else:
                            output = model.predict(input_data)
                else:
                    if hasattr(model, "__call__"):
                        output = model(input_data)
                    else:
                        output = model.predict(input_data)
            
            # Convert back to CPU/numpy if needed
            if isinstance(output, torch.Tensor):
                output = output.cpu()
                # Convert to numpy if it's a single value or small tensor
                if output.numel() == 1:
                    output = output.item()
                elif output.dim() == 0:
                    output = output.item()
                else:
                    output = output.numpy()
            
            return output
        
        except ImportError:
            # Fallback for non-PyTorch models
            if hasattr(model, "predict"):
                return model.predict(input_data)
            elif hasattr(model, "__call__"):
                return model(input_data)
            else:
                raise ValueError("Model does not support inference")
    
    def _calculate_confidence(self, prediction: Any) -> float:
        """Calculate confidence score from prediction"""
        try:
            import numpy as np
            
            if isinstance(prediction, np.ndarray):
                if prediction.ndim == 1:
                    # Probability distribution
                    return float(np.max(prediction))
                else:
                    # Multiple predictions
                    return float(np.mean([np.max(p) for p in prediction]))
            elif isinstance(prediction, (list, tuple)):
                if all(isinstance(p, (int, float)) for p in prediction):
                    return float(max(prediction))
            
            return 0.5  # Default confidence
        
        except Exception:
            return 0.5
    
    def _generate_cache_key(self, model_id: str, input_data: Any) -> str:
        """Generate cache key for input data"""
        try:
            import numpy as np
            import pickle
            
            # Serialize input data
            if isinstance(input_data, np.ndarray):
                data_hash = hashlib.md5(input_data.tobytes()).hexdigest()
            else:
                data_hash = hashlib.md5(pickle.dumps(input_data)).hexdigest()
            
            return f"{model_id}:{data_hash}"
        
        except Exception:
            return f"{model_id}:{hash(str(input_data))}"
    
    def batch_predict(
        self,
        model_id: str,
        input_batch: List[Any],
        batch_size: Optional[int] = None
    ) -> List[InferenceResult]:
        """Run batch inference"""
        if model_id not in self.model_configs:
            raise ValueError(f"Model {model_id} not registered")
        
        config = self.model_configs[model_id]
        batch_size = batch_size or config.batch_size
        
        results = []
        for i in range(0, len(input_batch), batch_size):
            batch = input_batch[i:i + batch_size]
            batch_results = self._process_batch(model_id, batch, config)
            results.extend(batch_results)
            self.stats["batch_processed"] += 1
        
        return results
    
    def _process_batch(self, model_id: str, batch: List[Any], config: ModelConfig) -> List[InferenceResult]:
        """Process a batch of inputs"""
        model = self.load_model(model_id)
        
        try:
            import torch
            import numpy as np
            
            # Prepare batch
            if isinstance(batch[0], np.ndarray):
                batch_tensor = torch.tensor(np.array(batch))
            else:
                batch_tensor = torch.tensor(batch)
            
            if config.device != "cpu":
                batch_tensor = batch_tensor.to(config.device)
            
            # Run batch inference
            start_time = time.time()
            with torch.no_grad():
                if hasattr(model, "__call__"):
                    outputs = model(batch_tensor)
                else:
                    outputs = model.predict(batch_tensor)
            
            processing_time = time.time() - start_time
            
            # Convert outputs
            if isinstance(outputs, torch.Tensor):
                outputs = outputs.cpu().numpy()
            
            # Create results
            results = []
            for i, output in enumerate(outputs):
                results.append(InferenceResult(
                    prediction=output,
                    confidence=self._calculate_confidence(output),
                    processing_time=processing_time / len(batch),
                    model_id=model_id,
                    cached=False
                ))
            
            return results
        
        except ImportError:
            # Fallback: process individually
            return [self.predict(model_id, item, use_cache=False) for item in batch]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about model usage"""
        avg_time = (
            self.stats["total_time"] / self.stats["inferences"]
            if self.stats["inferences"] > 0
            else 0.0
        )
        cache_hit_rate = (
            self.stats["cache_hits"] / (self.stats["cache_hits"] + self.stats["cache_misses"])
            if (self.stats["cache_hits"] + self.stats["cache_misses"]) > 0
            else 0.0
        )
        
        return {
            **self.stats,
            "avg_processing_time": avg_time,
            "cache_hit_rate": cache_hit_rate,
            "loaded_models": len(self.models),
            "registered_models": len(self.model_configs)
        }
    
    def clear_cache(self):
        """Clear inference cache"""
        with self.lock:
            self.inference_cache.clear()
            logger.info("Inference cache cleared")
    
    def unload_model(self, model_id: str):
        """Unload a model from memory"""
        with self.lock:
            if model_id in self.models:
                del self.models[model_id]
                logger.info(f"Unloaded model: {model_id}")

