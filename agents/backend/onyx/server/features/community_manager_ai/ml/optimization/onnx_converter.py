"""
ONNX Converter - Conversor a ONNX
==================================

Conversión a ONNX para inferencia optimizada.
"""

import logging
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
import os

logger = logging.getLogger(__name__)


class ONNXConverter:
    """Conversor de modelos a ONNX"""
    
    @staticmethod
    def to_onnx(
        model: nn.Module,
        example_input: Dict[str, torch.Tensor],
        output_path: str,
        opset_version: int = 14,
        dynamic_axes: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Convertir modelo a ONNX
        
        Args:
            model: Modelo a convertir
            example_input: Input de ejemplo
            output_path: Ruta de salida
            opset_version: Versión de opset
            dynamic_axes: Ejes dinámicos
            
        Returns:
            True si se convirtió exitosamente
        """
        try:
            import onnx
            
            model.eval()
            
            # Convertir
            torch.onnx.export(
                model,
                example_input,
                output_path,
                opset_version=opset_version,
                input_names=list(example_input.keys()),
                output_names=["output"],
                dynamic_axes=dynamic_axes,
                do_constant_folding=True
            )
            
            # Verificar
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            
            logger.info(f"Modelo convertido a ONNX: {output_path}")
            return True
            
        except ImportError:
            logger.warning("ONNX no disponible")
            return False
        except Exception as e:
            logger.error(f"Error convirtiendo a ONNX: {e}")
            return False
    
    @staticmethod
    def load_onnx_model(onnx_path: str) -> Optional[Any]:
        """
        Cargar modelo ONNX para inferencia
        
        Args:
            onnx_path: Ruta al modelo ONNX
            
        Returns:
            Sesión ONNX Runtime o None
        """
        try:
            import onnxruntime as ort
            
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            session = ort.InferenceSession(
                onnx_path,
                providers=providers
            )
            
            logger.info(f"Modelo ONNX cargado: {onnx_path}")
            return session
            
        except ImportError:
            logger.warning("onnxruntime no disponible")
            return None
        except Exception as e:
            logger.error(f"Error cargando modelo ONNX: {e}")
            return None
    
    @staticmethod
    def optimize_onnx(onnx_path: str, optimized_path: str) -> bool:
        """
        Optimizar modelo ONNX
        
        Args:
            onnx_path: Ruta al modelo ONNX
            optimized_path: Ruta de salida optimizada
            
        Returns:
            True si se optimizó exitosamente
        """
        try:
            import onnx
            from onnxruntime.transformers import optimizer
            
            model = onnx.load(onnx_path)
            optimized_model = optimizer.optimize_model(model)
            optimized_model.save_model_to_file(optimized_path)
            
            logger.info(f"Modelo ONNX optimizado: {optimized_path}")
            return True
            
        except ImportError:
            logger.warning("Optimizador ONNX no disponible")
            return False
        except Exception as e:
            logger.error(f"Error optimizando ONNX: {e}")
            return False




