"""
Inference Utils - Utilidades de Inferencia
===========================================

Utilidades para optimización y gestión de inferencia.
"""

import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict, Optional, Callable, Any, Iterator
import numpy as np
from contextlib import contextmanager
import time

logger = logging.getLogger(__name__)

# Intentar importar bibliotecas opcionales
try:
    import onnx
    import onnxruntime as ort
    _has_onnx = True
except ImportError:
    _has_onnx = False
    logger.warning("ONNX not available, some inference functions will be limited")


class BatchInferenceManager:
    """
    Gestor de inferencia por batches.
    """
    
    def __init__(
        self,
        model: nn.Module,
        batch_size: int = 32,
        device: str = "cuda",
        use_amp: bool = True
    ):
        """
        Inicializar gestor de inferencia.
        
        Args:
            model: Modelo PyTorch
            batch_size: Tamaño de batch
            device: Dispositivo
            use_amp: Usar mixed precision
        """
        self.model = model
        self.batch_size = batch_size
        self.device = device
        self.use_amp = use_amp
        
        self.model.to(device)
        self.model.eval()
    
    def predict(
        self,
        dataloader: DataLoader,
        return_probs: bool = False
    ) -> np.ndarray:
        """
        Predecir sobre un DataLoader.
        
        Args:
            dataloader: DataLoader
            return_probs: Retornar probabilidades
            
        Returns:
            Predicciones
        """
        all_predictions = []
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(self.device)
                else:
                    x = batch.to(self.device)
                
                with torch.cuda.amp.autocast() if self.use_amp else torch.no_grad():
                    outputs = self.model(x)
                
                if return_probs:
                    predictions = torch.softmax(outputs, dim=1)
                else:
                    predictions = outputs.argmax(dim=1)
                
                all_predictions.append(predictions.cpu().numpy())
        
        return np.concatenate(all_predictions)
    
    def predict_batch(
        self,
        inputs: torch.Tensor,
        return_probs: bool = False
    ) -> torch.Tensor:
        """
        Predecir sobre un batch.
        
        Args:
            inputs: Inputs
            return_probs: Retornar probabilidades
            
        Returns:
            Predicciones
        """
        self.model.eval()
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast() if self.use_amp else torch.no_grad():
                outputs = self.model(inputs)
            
            if return_probs:
                return torch.softmax(outputs, dim=1)
            else:
                return outputs.argmax(dim=1)


class ONNXExporter:
    """
    Exportador de modelos a ONNX.
    """
    
    def __init__(self, model: nn.Module):
        """
        Inicializar exportador ONNX.
        
        Args:
            model: Modelo PyTorch
        """
        if not _has_onnx:
            raise ImportError("ONNX is required for ONNXExporter")
        
        self.model = model
        self.model.eval()
    
    def export(
        self,
        dummy_input: torch.Tensor,
        output_path: str,
        input_names: List[str] = ["input"],
        output_names: List[str] = ["output"],
        dynamic_axes: Optional[Dict] = None,
        opset_version: int = 11
    ):
        """
        Exportar modelo a ONNX.
        
        Args:
            dummy_input: Input de ejemplo
            output_path: Ruta de salida
            input_names: Nombres de inputs
            output_names: Nombres de outputs
            dynamic_axes: Ejes dinámicos
            opset_version: Versión de opset
        """
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True
        )
        logger.info(f"Model exported to {output_path}")
    
    def optimize(self, model_path: str, optimized_path: str):
        """
        Optimizar modelo ONNX.
        
        Args:
            model_path: Ruta del modelo
            optimized_path: Ruta del modelo optimizado
        """
        try:
            from onnxruntime.transformers import optimizer
            from onnxruntime.transformers.fusion_options import FusionOptions
            
            opt_options = FusionOptions('bert')
            opt_model = optimizer.optimize_model(
                model_path,
                'bert',
                num_heads=12,
                hidden_size=768,
                optimization_options=opt_options
            )
            opt_model.save_model_to_file(optimized_path)
            logger.info(f"Optimized model saved to {optimized_path}")
        except ImportError:
            logger.warning("ONNX Runtime Transformers not available for optimization")


class ONNXRuntimeInference:
    """
    Inferencia usando ONNX Runtime.
    """
    
    def __init__(
        self,
        model_path: str,
        providers: Optional[List[str]] = None
    ):
        """
        Inicializar inferencia ONNX.
        
        Args:
            model_path: Ruta del modelo ONNX
            providers: Proveedores de ejecución
        """
        if not _has_onnx:
            raise ImportError("ONNX Runtime is required")
        
        if providers is None:
            providers = ['CPUExecutionProvider']
            if torch.cuda.is_available():
                providers.insert(0, 'CUDAExecutionProvider')
        
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
    
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Predecir con ONNX Runtime.
        
        Args:
            inputs: Inputs numpy
            
        Returns:
            Predicciones
        """
        outputs = self.session.run([self.output_name], {self.input_name: inputs})
        return outputs[0]
    
    def predict_batch(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        """
        Predecir batch con ONNX Runtime.
        
        Args:
            inputs: Lista de inputs
            
        Returns:
            Lista de predicciones
        """
        return [self.predict(inp) for inp in inputs]


class InferenceOptimizer:
    """
    Optimizador de inferencia.
    """
    
    def __init__(self, model: nn.Module):
        """
        Inicializar optimizador.
        
        Args:
            model: Modelo PyTorch
        """
        self.model = model
    
    def optimize_for_inference(self):
        """
        Optimizar modelo para inferencia.
        """
        self.model.eval()
        
        # Fusionar operaciones
        if hasattr(torch.jit, 'script'):
            try:
                self.model = torch.jit.script(self.model)
                logger.info("Model optimized with TorchScript")
            except Exception as e:
                logger.warning(f"TorchScript optimization failed: {e}")
        
        # Compilar con torch.compile (PyTorch 2.0+)
        try:
            self.model = torch.compile(self.model, mode="reduce-overhead")
            logger.info("Model optimized with torch.compile")
        except Exception as e:
            logger.warning(f"torch.compile optimization failed: {e}")
    
    @contextmanager
    def inference_mode(self):
        """
        Context manager para modo de inferencia optimizado.
        """
        self.model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                yield self.model


class TorchServeExporter:
    """
    Exportador para TorchServe.
    """
    
    def __init__(self, model: nn.Module):
        """
        Inicializar exportador TorchServe.
        
        Args:
            model: Modelo PyTorch
        """
        self.model = model
        self.model.eval()
    
    def export(
        self,
        dummy_input: torch.Tensor,
        output_dir: str,
        model_name: str = "model"
    ):
        """
        Exportar modelo para TorchServe.
        
        Args:
            dummy_input: Input de ejemplo
            output_dir: Directorio de salida
            model_name: Nombre del modelo
        """
        # Guardar modelo
        model_path = f"{output_dir}/{model_name}.pth"
        torch.save(self.model.state_dict(), model_path)
        
        # Crear archivo de handler (ejemplo)
        handler_code = f"""
from torchvision import transforms
import torch

def model_fn(model_dir):
    model = YourModelClass()  # Reemplazar con clase del modelo
    model.load_state_dict(torch.load('{model_path}'))
    model.eval()
    return model

def input_fn(request_body, request_content_type):
    # Procesar input
    return processed_input

def predict_fn(input_data, model):
    with torch.no_grad():
        output = model(input_data)
    return output
"""
        
        handler_path = f"{output_dir}/{model_name}_handler.py"
        with open(handler_path, 'w') as f:
            f.write(handler_code)
        
        logger.info(f"Model exported for TorchServe to {output_dir}")


class InferenceBenchmark:
    """
    Benchmark de inferencia.
    """
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        """
        Inicializar benchmark.
        
        Args:
            model: Modelo PyTorch
            device: Dispositivo
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def benchmark(
        self,
        dummy_input: torch.Tensor,
        num_runs: int = 100,
        warmup: int = 10
    ) -> Dict[str, float]:
        """
        Ejecutar benchmark.
        
        Args:
            dummy_input: Input de ejemplo
            num_runs: Número de ejecuciones
            warmup: Número de warmup runs
            
        Returns:
            Estadísticas de benchmark
        """
        dummy_input = dummy_input.to(self.device)
        
        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                _ = self.model(dummy_input)
        
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            if self.device == "cuda":
                torch.cuda.synchronize()
            
            start = time.time()
            with torch.no_grad():
                _ = self.model(dummy_input)
            
            if self.device == "cuda":
                torch.cuda.synchronize()
            
            times.append(time.time() - start)
        
        times = np.array(times)
        
        return {
            "mean_time": float(np.mean(times)),
            "std_time": float(np.std(times)),
            "min_time": float(np.min(times)),
            "max_time": float(np.max(times)),
            "p50_time": float(np.percentile(times, 50)),
            "p95_time": float(np.percentile(times, 95)),
            "p99_time": float(np.percentile(times, 99)),
            "throughput": float(1.0 / np.mean(times))
        }




