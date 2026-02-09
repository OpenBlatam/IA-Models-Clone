"""
ONNX Utilities - Modular ONNX Tools
====================================

Utilidades modulares para trabajar con ONNX.
"""

import logging
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
from pathlib import Path

logger = logging.getLogger(__name__)


class ONNXConverter:
    """Convertidor a ONNX."""
    
    @staticmethod
    def convert(
        model: nn.Module,
        input_shape: tuple,
        output_path: str,
        opset_version: int = 14,
        dynamic_axes: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        Convertir modelo a ONNX.
        
        Args:
            model: Modelo PyTorch
            input_shape: Forma de entrada
            output_path: Ruta de salida
            opset_version: Versión de opset
            dynamic_axes: Ejes dinámicos
            **kwargs: Argumentos adicionales
            
        Returns:
            Ruta del archivo ONNX
        """
        try:
            import torch.onnx
            
            model.eval()
            dummy_input = torch.randn(1, *input_shape)
            
            if dynamic_axes is None:
                dynamic_axes = {
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                opset_version=opset_version,
                input_names=kwargs.get('input_names', ['input']),
                output_names=kwargs.get('output_names', ['output']),
                dynamic_axes=dynamic_axes,
                **{k: v for k, v in kwargs.items() if k not in ['input_names', 'output_names']}
            )
            
            logger.info(f"Model converted to ONNX: {output_path}")
            return output_path
        except ImportError:
            raise ImportError("ONNX not available. Install with: pip install onnx")
        except Exception as e:
            logger.error(f"Error converting to ONNX: {e}")
            raise


class ONNXRuntime:
    """Runtime de ONNX."""
    
    def __init__(self, model_path: str, providers: Optional[List[str]] = None):
        """
        Inicializar runtime ONNX.
        
        Args:
            model_path: Ruta al modelo ONNX
            providers: Proveedores de ejecución
        """
        try:
            import onnxruntime as ort
            
            if providers is None:
                providers = ['CPUExecutionProvider']
                if ort.get_device() == 'GPU':
                    providers.insert(0, 'CUDAExecutionProvider')
            
            self.session = ort.InferenceSession(
                model_path,
                providers=providers
            )
            
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            logger.info(f"ONNX Runtime initialized: {model_path}")
        except ImportError:
            raise ImportError("onnxruntime not available. Install with: pip install onnxruntime")
        except Exception as e:
            logger.error(f"Error initializing ONNX Runtime: {e}")
            raise
    
    def predict(self, input_data: Any) -> Any:
        """
        Realizar predicción.
        
        Args:
            input_data: Datos de entrada
            
        Returns:
            Predicción
        """
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.cpu().numpy()
        elif not isinstance(input_data, np.ndarray):
            import numpy as np
            input_data = np.array(input_data)
        
        result = self.session.run(
            [self.output_name],
            {self.input_name: input_data}
        )
        
        return result[0]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Obtener información del modelo."""
        return {
            'input_name': self.input_name,
            'output_name': self.output_name,
            'inputs': [
                {
                    'name': inp.name,
                    'shape': inp.shape,
                    'type': str(inp.type)
                }
                for inp in self.session.get_inputs()
            ],
            'outputs': [
                {
                    'name': out.name,
                    'shape': out.shape,
                    'type': str(out.type)
                }
                for out in self.session.get_outputs()
            ]
        }


class ONNXOptimizer:
    """Optimizador de modelos ONNX."""
    
    @staticmethod
    def optimize(
        model_path: str,
        output_path: Optional[str] = None,
        optimization_level: str = 'all'
    ) -> str:
        """
        Optimizar modelo ONNX.
        
        Args:
            model_path: Ruta al modelo ONNX
            output_path: Ruta de salida (opcional)
            optimization_level: Nivel de optimización
            
        Returns:
            Ruta del modelo optimizado
        """
        try:
            import onnx
            from onnxoptimizer import optimize_model
            
            if output_path is None:
                output_path = model_path.replace('.onnx', '_optimized.onnx')
            
            model = onnx.load(model_path)
            optimized_model = optimize_model(model, optimization_level)
            onnx.save(optimized_model, output_path)
            
            logger.info(f"ONNX model optimized: {output_path}")
            return output_path
        except ImportError:
            raise ImportError("onnxoptimizer not available. Install with: pip install onnxoptimizer")
        except Exception as e:
            logger.error(f"Error optimizing ONNX model: {e}")
            raise








