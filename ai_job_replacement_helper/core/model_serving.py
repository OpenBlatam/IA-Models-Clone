"""
Model Serving Service - Servicio de inferencia optimizado
==========================================================

Sistema para servir modelos con optimizaciones de inferencia.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Try to import PyTorch
try:
    import torch
    import torch.jit
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


@dataclass
class InferenceConfig:
    """Configuración de inferencia"""
    model_id: str
    use_torchscript: bool = True
    use_quantization: bool = False
    use_tensorrt: bool = False
    batch_size: int = 1
    max_batch_size: int = 32
    device: Optional[str] = None


@dataclass
class InferenceResult:
    """Resultado de inferencia"""
    predictions: Any
    inference_time: float
    batch_size: int
    device: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelServingService:
    """Servicio de model serving optimizado"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.models: Dict[str, nn.Module] = {}
        self.torchscript_models: Dict[str, torch.jit.ScriptModule] = {}
        self.configs: Dict[str, InferenceConfig] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"ModelServingService initialized on device: {self.device}")
    
    def load_model(
        self,
        model: nn.Module,
        config: InferenceConfig
    ) -> bool:
        """Cargar modelo para serving"""
        if not TORCH_AVAILABLE:
            return False
        
        try:
            device = torch.device(config.device) if config.device else self.device
            model = model.to(device)
            model.eval()
            
            self.models[config.model_id] = model
            self.configs[config.model_id] = config
            
            # Convert to TorchScript if requested
            if config.use_torchscript:
                try:
                    # Trace the model
                    example_input = torch.randn(1, 10).to(device)  # Adjust shape as needed
                    traced_model = torch.jit.trace(model, example_input)
                    self.torchscript_models[config.model_id] = traced_model
                    logger.info(f"Model {config.model_id} converted to TorchScript")
                except Exception as e:
                    logger.warning(f"Failed to convert to TorchScript: {e}")
            
            logger.info(f"Model {config.model_id} loaded for serving")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def quantize_model(
        self,
        model_id: str,
        quantization_type: str = "dynamic"
    ) -> bool:
        """Cuantizar modelo para inferencia más rápida"""
        if not TORCH_AVAILABLE:
            return False
        
        model = self.models.get(model_id)
        if not model:
            return False
        
        try:
            if quantization_type == "dynamic":
                quantized_model = torch.quantization.quantize_dynamic(
                    model,
                    {nn.Linear, nn.LSTM},
                    dtype=torch.qint8
                )
                self.models[model_id] = quantized_model
                logger.info(f"Model {model_id} quantized (dynamic)")
                return True
            elif quantization_type == "static":
                # Static quantization requires calibration data
                logger.warning("Static quantization requires calibration data")
                return False
            else:
                logger.warning(f"Unknown quantization type: {quantization_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error quantizing model: {e}")
            return False
    
    async def predict(
        self,
        model_id: str,
        inputs: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: Optional[int] = None
    ) -> InferenceResult:
        """Realizar predicción optimizada"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        
        start_time = time.time()
        
        config = self.configs.get(model_id)
        if not config:
            raise ValueError(f"Model {model_id} not loaded")
        
        # Use TorchScript if available
        model = self.torchscript_models.get(model_id) or self.models.get(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        device = self.device
        
        # Prepare inputs
        if isinstance(inputs, list):
            # Batch inputs
            batch_size = batch_size or config.batch_size
            batched_inputs = torch.stack(inputs).to(device)
        else:
            batched_inputs = inputs.to(device) if isinstance(inputs, torch.Tensor) else torch.tensor(inputs).to(device)
        
        # Inference
        with torch.no_grad():
            # Use autocast for mixed precision if on GPU
            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    predictions = model(batched_inputs)
            else:
                predictions = model(batched_inputs)
        
        inference_time = time.time() - start_time
        
        # Convert to CPU and numpy if needed
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        
        return InferenceResult(
            predictions=predictions,
            inference_time=inference_time,
            batch_size=batched_inputs.size(0) if isinstance(batched_inputs, torch.Tensor) else len(batched_inputs),
            device=str(device),
            metadata={
                "model_id": model_id,
                "used_torchscript": model_id in self.torchscript_models,
            }
        )
    
    async def predict_batch(
        self,
        model_id: str,
        inputs: List[torch.Tensor],
        batch_size: Optional[int] = None
    ) -> InferenceResult:
        """Realizar predicción en batch"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        
        config = self.configs.get(model_id)
        batch_size = batch_size or config.batch_size
        
        # Process in batches
        all_predictions = []
        total_time = 0.0
        
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            result = await self.predict(model_id, batch, batch_size)
            all_predictions.append(result.predictions)
            total_time += result.inference_time
        
        return InferenceResult(
            predictions=all_predictions,
            inference_time=total_time,
            batch_size=len(inputs),
            device=str(self.device),
            metadata={
                "model_id": model_id,
                "num_batches": (len(inputs) + batch_size - 1) // batch_size,
            }
        )
    
    def get_model_stats(self, model_id: str) -> Dict[str, Any]:
        """Obtener estadísticas del modelo"""
        model = self.models.get(model_id)
        if not model:
            return {"error": "Model not found"}
        
        if not TORCH_AVAILABLE:
            return {"error": "PyTorch not available"}
        
        config = self.configs.get(model_id, InferenceConfig(model_id=model_id))
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            "model_id": model_id,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "has_torchscript": model_id in self.torchscript_models,
            "device": str(self.device),
            "batch_size": config.batch_size,
        }




