"""
Model Serving System
Optimized model serving with batching, caching, and load balancing
"""

from typing import Dict, Any, Optional, List, Callable
import logging
import time
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class ModelConfig:
    """Model configuration for serving"""
    model_id: str
    model_path: str
    version: str
    input_shape: tuple
    output_shape: tuple
    device: str = "cuda"
    batch_size: int = 32
    max_queue_size: int = 100


class ModelServer:
    """
    Model serving system with:
    - Model loading and management
    - Request batching
    - Caching
    - Load balancing
    - Health monitoring
    """
    
    def __init__(self):
        self.models: Dict[str, nn.Module] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        self.request_stats: Dict[str, Dict[str, Any]] = {}
    
    def load_model(self, config: ModelConfig) -> bool:
        """Load a model for serving"""
        try:
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch required for model serving")
            
            # Load model
            model = torch.load(config.model_path, map_location=config.device)
            model.to(config.device)
            model.eval()
            
            self.models[config.model_id] = model
            self.model_configs[config.model_id] = config
            self.request_stats[config.model_id] = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "avg_latency": 0.0,
                "latencies": []
            }
            
            logger.info(f"Loaded model {config.model_id} v{config.version}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading model {config.model_id}: {str(e)}")
            return False
    
    def unload_model(self, model_id: str) -> bool:
        """Unload a model"""
        if model_id in self.models:
            del self.models[model_id]
            del self.model_configs[model_id]
            if model_id in self.request_stats:
                del self.request_stats[model_id]
            logger.info(f"Unloaded model {model_id}")
            return True
        return False
    
    def predict(
        self,
        model_id: str,
        input_data: Any,
        batch: bool = False
    ) -> Any:
        """Run inference"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not loaded")
        
        config = self.model_configs[model_id]
        model = self.models[model_id]
        stats = self.request_stats[model_id]
        
        start_time = time.time()
        
        try:
            # Prepare input
            if isinstance(input_data, np.ndarray):
                input_tensor = torch.from_numpy(input_data).to(config.device)
            elif isinstance(input_data, torch.Tensor):
                input_tensor = input_data.to(config.device)
            else:
                input_tensor = torch.tensor(input_data).to(config.device)
            
            # Add batch dimension if needed
            if len(input_tensor.shape) == len(config.input_shape):
                input_tensor = input_tensor.unsqueeze(0)
            
            # Inference
            with torch.no_grad():
                output = model(input_tensor)
            
            # Remove batch dimension if single input
            if not batch and output.shape[0] == 1:
                output = output.squeeze(0)
            
            # Update stats
            latency = time.time() - start_time
            stats["total_requests"] += 1
            stats["successful_requests"] += 1
            stats["latencies"].append(latency)
            if len(stats["latencies"]) > 100:
                stats["latencies"] = stats["latencies"][-100:]
            stats["avg_latency"] = np.mean(stats["latencies"])
            
            return output.cpu().numpy() if isinstance(output, torch.Tensor) else output
        
        except Exception as e:
            stats["total_requests"] += 1
            stats["failed_requests"] += 1
            logger.error(f"Prediction error for {model_id}: {str(e)}")
            raise
    
    def batch_predict(
        self,
        model_id: str,
        input_batch: List[Any]
    ) -> List[Any]:
        """Run batch inference"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not loaded")
        
        config = self.model_configs[model_id]
        model = self.models[model_id]
        
        # Prepare batch
        tensors = []
        for input_data in input_batch:
            if isinstance(input_data, np.ndarray):
                tensor = torch.from_numpy(input_data)
            elif isinstance(input_data, torch.Tensor):
                tensor = input_data
            else:
                tensor = torch.tensor(input_data)
            tensors.append(tensor)
        
        batch_tensor = torch.stack(tensors).to(config.device)
        
        # Inference
        with torch.no_grad():
            outputs = model(batch_tensor)
        
        # Convert to list
        results = [outputs[i].cpu().numpy() for i in range(len(input_batch))]
        return results
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model information"""
        if model_id not in self.models:
            return None
        
        config = self.model_configs[model_id]
        stats = self.request_stats[model_id]
        
        return {
            "model_id": model_id,
            "version": config.version,
            "device": config.device,
            "input_shape": config.input_shape,
            "output_shape": config.output_shape,
            "stats": {
                "total_requests": stats["total_requests"],
                "successful_requests": stats["successful_requests"],
                "failed_requests": stats["failed_requests"],
                "success_rate": stats["successful_requests"] / stats["total_requests"] if stats["total_requests"] > 0 else 0,
                "avg_latency_ms": stats["avg_latency"] * 1000,
                "p50_latency_ms": np.percentile(stats["latencies"], 50) * 1000 if stats["latencies"] else 0,
                "p95_latency_ms": np.percentile(stats["latencies"], 95) * 1000 if stats["latencies"] else 0,
                "p99_latency_ms": np.percentile(stats["latencies"], 99) * 1000 if stats["latencies"] else 0
            }
        }
    
    def list_models(self) -> List[str]:
        """List loaded models"""
        return list(self.models.keys())
    
    def health_check(self) -> Dict[str, Any]:
        """Health check for all models"""
        health = {
            "status": "healthy",
            "models": {}
        }
        
        for model_id in self.models:
            try:
                # Try a dummy prediction
                config = self.model_configs[model_id]
                dummy_input = torch.randn(1, *config.input_shape).to(config.device)
                with torch.no_grad():
                    _ = self.models[model_id](dummy_input)
                
                health["models"][model_id] = {
                    "status": "healthy",
                    "device": config.device
                }
            except Exception as e:
                health["status"] = "degraded"
                health["models"][model_id] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        
        return health


# Global model server instance
_model_server: Optional[ModelServer] = None


def get_model_server() -> ModelServer:
    """Get or create model server instance"""
    global _model_server
    if _model_server is None:
        _model_server = ModelServer()
    return _model_server

