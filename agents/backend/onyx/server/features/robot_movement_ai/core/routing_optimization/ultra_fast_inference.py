"""
Ultra-Fast Inference
====================

Optimizaciones extremas para inferencia máxima velocidad.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)

# TensorRT
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    logger.warning("TensorRT no disponible")

# ONNX Runtime
try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False
    logger.warning("ONNX Runtime no disponible")


class UltraFastInference:
    """
    Optimizador ultra-rápido para inferencia.
    """
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        """
        Inicializar optimizador.
        
        Args:
            model: Modelo
            device: Dispositivo
        """
        self.model = model
        self.device = device
        self.optimized_model = None
        self.optimization_method = None
    
    def optimize_torchscript_aggressive(self) -> nn.Module:
        """
        Optimización agresiva con TorchScript.
        
        Returns:
            Modelo optimizado
        """
        self.model.eval()
        
        try:
            # Crear input de ejemplo
            if hasattr(self.model, 'config'):
                input_dim = self.model.config.input_dim
            else:
                input_dim = 20
            
            example_input = torch.randn(1, input_dim).to(self.device)
            
            # Tracing
            traced = torch.jit.trace(self.model, example_input)
            
            # Optimizaciones agresivas
            traced = torch.jit.optimize_for_inference(traced)
            traced = torch.jit.freeze(traced)  # Congelar para optimizaciones adicionales
            
            # Fuse operations
            traced = torch.jit.fuse(traced)
            
            self.optimized_model = traced
            self.optimization_method = "torchscript_aggressive"
            logger.info("Modelo optimizado agresivamente con TorchScript")
            
            return traced
        except Exception as e:
            logger.warning(f"Error en optimización TorchScript: {e}")
            return self.model
    
    def optimize_torch_compile_max(self) -> nn.Module:
        """
        Optimización máxima con torch.compile.
        
        Returns:
            Modelo optimizado
        """
        if not hasattr(torch, 'compile'):
            logger.warning("torch.compile no disponible")
            return self.model
        
        try:
            # Modo máximo de optimización
            self.optimized_model = torch.compile(
                self.model,
                mode="max-autotune",  # Máxima optimización
                fullgraph=True,
                dynamic=False
            )
            
            self.optimization_method = "torch_compile_max"
            logger.info("Modelo compilado con torch.compile (max-autotune)")
            
            return self.optimized_model
        except Exception as e:
            logger.warning(f"Error en torch.compile max: {e}")
            return self.model
    
    def optimize_onnx(self, input_shape: Tuple[int, ...] = (1, 20),
                     output_path: Optional[str] = None) -> Optional[Any]:
        """
        Exportar y optimizar con ONNX Runtime.
        
        Args:
            input_shape: Forma de entrada
            output_path: Ruta para guardar ONNX (opcional)
            
        Returns:
            ONNX Runtime session
        """
        if not ONNXRUNTIME_AVAILABLE:
            logger.warning("ONNX Runtime no disponible")
            return None
        
        try:
            import onnx
            
            # Exportar a ONNX
            dummy_input = torch.randn(*input_shape)
            onnx_path = output_path or "model.onnx"
            
            torch.onnx.export(
                self.model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=13,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            
            # Crear sesión ONNX Runtime optimizada
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == "cuda" else ['CPUExecutionProvider']
            
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.enable_mem_pattern = True
            sess_options.enable_cpu_mem_arena = True
            
            session = ort.InferenceSession(
                onnx_path,
                sess_options=sess_options,
                providers=providers
            )
            
            self.optimization_method = "onnx"
            logger.info("Modelo optimizado con ONNX Runtime")
            
            return session
        except Exception as e:
            logger.warning(f"Error en optimización ONNX: {e}")
            return None
    
    def optimize_tensorrt(self, input_shape: Tuple[int, ...] = (1, 20),
                         output_path: Optional[str] = None) -> Optional[Any]:
        """
        Optimizar con TensorRT (requiere TensorRT).
        
        Args:
            input_shape: Forma de entrada
            output_path: Ruta para guardar TensorRT engine (opcional)
            
        Returns:
            TensorRT engine
        """
        if not TENSORRT_AVAILABLE:
            logger.warning("TensorRT no disponible")
            return None
        
        try:
            # Primero exportar a ONNX
            dummy_input = torch.randn(*input_shape)
            onnx_path = "model_for_tensorrt.onnx"
            
            torch.onnx.export(
                self.model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=13
            )
            
            # Convertir a TensorRT (simplificado - requiere configuración completa)
            logger.info("TensorRT requiere configuración adicional")
            logger.info("Ver documentación de TensorRT para implementación completa")
            
            return None
        except Exception as e:
            logger.warning(f"Error en TensorRT: {e}")
            return None
    
    def apply_all_optimizations(self) -> nn.Module:
        """
        Aplicar todas las optimizaciones disponibles.
        
        Returns:
            Modelo optimizado
        """
        # 1. Compilación máxima
        if hasattr(torch, 'compile'):
            self.model = self.optimize_torch_compile_max()
        
        # 2. TorchScript agresivo
        self.model = self.optimize_torchscript_aggressive()
        
        # 3. Optimizaciones GPU
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
        
        logger.info("Todas las optimizaciones aplicadas")
        return self.model


class StreamInference:
    """
    Inferencia con CUDA streams para paralelización.
    """
    
    def __init__(self, model: nn.Module, num_streams: int = 4):
        """
        Inicializar inferencia con streams.
        
        Args:
            model: Modelo
            num_streams: Número de CUDA streams
        """
        self.model = model
        self.num_streams = num_streams
        self.streams = []
        
        if torch.cuda.is_available():
            for _ in range(num_streams):
                self.streams.append(torch.cuda.Stream())
    
    def predict_parallel(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Predecir en paralelo usando streams.
        
        Args:
            inputs: Lista de inputs
            
        Returns:
            Lista de outputs
        """
        if not torch.cuda.is_available() or len(self.streams) == 0:
            # Fallback secuencial
            with torch.no_grad():
                return [self.model(inp) for inp in inputs]
        
        outputs = [None] * len(inputs)
        
        # Procesar en paralelo con streams
        for i, inp in enumerate(inputs):
            stream = self.streams[i % len(self.streams)]
            with torch.cuda.stream(stream):
                with torch.no_grad():
                    outputs[i] = self.model(inp.cuda(non_blocking=True))
        
        # Sincronizar
        for stream in self.streams:
            stream.synchronize()
        
        return outputs


class PrecompiledModel:
    """
    Modelo pre-compilado para inferencia instantánea.
    """
    
    def __init__(self, model: nn.Module, compile_mode: str = "max"):
        """
        Inicializar modelo pre-compilado.
        
        Args:
            model: Modelo
            compile_mode: Modo de compilación
        """
        self.original_model = model
        self.compiled_model = None
        self.compile_mode = compile_mode
        
        self._compile()
    
    def _compile(self):
        """Compilar modelo."""
        optimizer = UltraFastInference(self.original_model)
        
        if self.compile_mode == "max":
            self.compiled_model = optimizer.apply_all_optimizations()
        elif self.compile_mode == "torchscript":
            self.compiled_model = optimizer.optimize_torchscript_aggressive()
        elif self.compile_mode == "torch_compile":
            self.compiled_model = optimizer.optimize_torch_compile_max()
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if self.compiled_model:
            return self.compiled_model(x)
        return self.original_model(x)
    
    def predict_batch_optimized(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Predecir batch optimizado.
        
        Args:
            batch: Batch de inputs
            
        Returns:
            Batch de outputs
        """
        # Usar no_grad y optimizaciones
        with torch.no_grad():
            if hasattr(torch, 'inference_mode'):
                with torch.inference_mode():  # Más rápido que no_grad
                    return self(batch)
            else:
                return self(batch)

