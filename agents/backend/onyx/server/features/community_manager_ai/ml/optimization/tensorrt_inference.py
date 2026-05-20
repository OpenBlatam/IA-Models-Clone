"""
TensorRT Inference - Inferencia con TensorRT
=============================================

Inferencia ultra-rápida usando TensorRT.
"""

import logging
import torch
import torch.nn as nn
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class TensorRTEngine:
    """Motor TensorRT para inferencia optimizada"""
    
    def __init__(
        self,
        model: nn.Module,
        example_input: Dict[str, torch.Tensor],
        precision: str = "fp16",
        workspace_size: int = 1 << 30  # 1GB
    ):
        """
        Inicializar motor TensorRT
        
        Args:
            model: Modelo PyTorch
            example_input: Input de ejemplo
            precision: Precisión (fp32, fp16, int8)
            workspace_size: Tamaño de workspace
        """
        try:
            import tensorrt as trt
            from torch2trt import torch2trt
            
            # Convertir modelo a TensorRT
            self.trt_model = torch2trt(
                model,
                [list(example_input.values())[0]],
                fp16_mode=(precision == "fp16"),
                max_workspace_size=workspace_size
            )
            
            self.device = next(model.parameters()).device
            logger.info(f"TensorRT Engine inicializado con precisión {precision}")
            
        except ImportError:
            logger.warning("TensorRT no disponible")
            self.trt_model = None
            self.device = None
        except Exception as e:
            logger.error(f"Error inicializando TensorRT: {e}")
            self.trt_model = None
            self.device = None
    
    def infer(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Inferencia con TensorRT
        
        Args:
            inputs: Inputs del modelo
            
        Returns:
            Output del modelo
        """
        if not self.trt_model:
            raise ValueError("TensorRT engine no inicializado")
        
        try:
            # Convertir inputs
            input_tensor = list(inputs.values())[0]
            
            with torch.no_grad():
                output = self.trt_model(input_tensor)
            
            return output
            
        except Exception as e:
            logger.error(f"Error en inferencia TensorRT: {e}")
            raise


class TensorRTConverter:
    """Conversor a TensorRT"""
    
    @staticmethod
    def convert_model(
        model: nn.Module,
        example_input: torch.Tensor,
        output_path: str,
        precision: str = "fp16"
    ) -> bool:
        """
        Convertir modelo a TensorRT
        
        Args:
            model: Modelo PyTorch
            example_input: Input de ejemplo
            output_path: Ruta de salida
            precision: Precisión
            
        Returns:
            True si se convirtió exitosamente
        """
        try:
            import tensorrt as trt
            from torch2trt import torch2trt
            
            trt_model = torch2trt(
                model,
                [example_input],
                fp16_mode=(precision == "fp16")
            )
            
            torch.save(trt_model.state_dict(), output_path)
            logger.info(f"Modelo TensorRT guardado en {output_path}")
            return True
            
        except ImportError:
            logger.warning("TensorRT no disponible")
            return False
        except Exception as e:
            logger.error(f"Error convirtiendo a TensorRT: {e}")
            return False




