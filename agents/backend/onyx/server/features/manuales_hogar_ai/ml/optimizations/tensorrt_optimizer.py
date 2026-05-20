"""
TensorRT Optimizer
==================

Optimización con TensorRT para inferencia ultra-rápida en NVIDIA GPUs.
"""

import logging
import torch
from typing import Optional, Any
import os

try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    logging.warning("TensorRT no disponible. Instalar con: pip install nvidia-tensorrt")

logger = logging.getLogger(__name__)


class TensorRTOptimizer:
    """Optimizador usando TensorRT."""
    
    def __init__(self, use_fp16: bool = True):
        """
        Inicializar optimizador TensorRT.
        
        Args:
            use_fp16: Usar FP16
        """
        self.use_fp16 = use_fp16
        self.engine = None
        self.context = None
        self._logger = logger
    
    def build_engine(
        self,
        model: Any,
        input_shape: tuple = (1, 512),
        output_path: Optional[str] = None
    ):
        """
        Construir engine TensorRT.
        
        Args:
            model: Modelo PyTorch
            input_shape: Forma de entrada
            output_path: Ruta para guardar engine
        """
        if not TENSORRT_AVAILABLE:
            raise ImportError("TensorRT no está disponible")
        
        try:
            # Convertir a ONNX primero
            onnx_path = output_path.replace(".trt", ".onnx") if output_path else "model.onnx"
            
            # Exportar a ONNX
            dummy_input = torch.randn(input_shape).cuda()
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
                opset_version=14
            )
            
            # Construir engine TensorRT
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, logger)
            
            with open(onnx_path, 'rb') as model_file:
                parser.parse(model_file.read())
            
            config = builder.create_builder_config()
            if self.use_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            config.max_workspace_size = 1 << 30  # 1GB
            
            self.engine = builder.build_engine(network, config)
            
            if output_path:
                with open(output_path, 'wb') as f:
                    f.write(self.engine.serialize())
            
            self.context = self.engine.create_execution_context()
            self._logger.info(f"Engine TensorRT construido: {output_path}")
        
        except Exception as e:
            self._logger.error(f"Error construyendo engine TensorRT: {str(e)}")
            raise
    
    def infer(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Inferencia con TensorRT.
        
        Args:
            input_tensor: Tensor de entrada
        
        Returns:
            Tensor de salida
        """
        if self.engine is None:
            raise ValueError("Engine TensorRT no construido")
        
        try:
            # Preparar buffers
            bindings = []
            for binding in self.engine:
                size = trt.volume(self.engine.get_binding_shape(binding))
                dtype = trt.nptype(self.engine.get_binding_dtype(binding))
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                bindings.append(int(device_mem))
            
            # Ejecutar
            cuda.memcpy_htod(bindings[0], input_tensor.cpu().numpy())
            self.context.execute_v2(bindings)
            output = cuda.pagelocked_empty(trt.volume(self.engine.get_binding_shape(1)), dtype)
            cuda.memcpy_dtoh(output, bindings[1])
            
            return torch.from_numpy(output).cuda()
        
        except Exception as e:
            self._logger.error(f"Error en inferencia TensorRT: {str(e)}")
            raise




