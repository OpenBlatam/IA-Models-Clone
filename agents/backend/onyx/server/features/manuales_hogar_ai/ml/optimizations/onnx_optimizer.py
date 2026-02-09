"""
Optimizador ONNX
================

Conversión a ONNX para inferencia ultra-rápida.
"""

import logging
import torch
import numpy as np
from typing import Optional, Any
import os

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.warning("ONNX Runtime no disponible. Instalar con: pip install onnxruntime-gpu")

logger = logging.getLogger(__name__)


class ONNXOptimizer:
    """Optimizador usando ONNX Runtime."""
    
    def __init__(self, use_gpu: bool = True):
        """
        Inicializar optimizador ONNX.
        
        Args:
            use_gpu: Usar GPU
        """
        self.use_gpu = use_gpu and ONNX_AVAILABLE
        self.session = None
        self._logger = logger
    
    def convert_to_onnx(
        self,
        model: Any,
        tokenizer: Any,
        output_path: str,
        input_shape: tuple = (1, 512)
    ):
        """
        Convertir modelo a ONNX.
        
        Args:
            model: Modelo PyTorch
            tokenizer: Tokenizer
            output_path: Ruta de salida
            input_shape: Forma de entrada
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime no está disponible")
        
        try:
            import torch.onnx
            
            model.eval()
            
            # Ejemplo de entrada
            dummy_input = torch.randint(0, tokenizer.vocab_size, input_shape)
            
            # Exportar a ONNX
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                input_names=['input_ids'],
                output_names=['logits'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                    'logits': {0: 'batch_size', 1: 'sequence_length'}
                },
                opset_version=14,
                do_constant_folding=True
            )
            
            self._logger.info(f"Modelo convertido a ONNX: {output_path}")
        
        except Exception as e:
            self._logger.error(f"Error convirtiendo a ONNX: {str(e)}")
            raise
    
    def load_onnx_model(self, model_path: str):
        """
        Cargar modelo ONNX.
        
        Args:
            model_path: Ruta del modelo ONNX
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime no está disponible")
        
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.use_gpu else ['CPUExecutionProvider']
            
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.enable_mem_pattern = True
            sess_options.enable_cpu_mem_arena = True
            
            self.session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=providers
            )
            
            self._logger.info(f"Modelo ONNX cargado: {model_path}")
        
        except Exception as e:
            self._logger.error(f"Error cargando modelo ONNX: {str(e)}")
            raise
    
    def infer(self, input_ids: np.ndarray) -> np.ndarray:
        """
        Inferencia con ONNX.
        
        Args:
            input_ids: IDs de entrada
        
        Returns:
            Logits de salida
        """
        if self.session is None:
            raise ValueError("Modelo ONNX no cargado")
        
        try:
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: input_ids})
            return outputs[0]
        
        except Exception as e:
            self._logger.error(f"Error en inferencia ONNX: {str(e)}")
            raise




