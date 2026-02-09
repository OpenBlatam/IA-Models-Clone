"""
Inference Engine - Modular Inference System
===========================================

Motor de inferencia modular para modelos de deep learning.
"""

import logging
from typing import Dict, Any, Optional, List, Union
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np

from ..dl_utils.device_manager import DeviceManager, get_device_manager

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Motor de inferencia modular.
    
    Maneja la inferencia de modelos con optimizaciones
    y batching automático.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device_manager: Optional[DeviceManager] = None,
        use_amp: bool = True,
        batch_size: int = 32,
        max_batch_size: int = 128
    ):
        """
        Inicializar motor de inferencia.
        
        Args:
            model: Modelo PyTorch
            device_manager: Gestor de dispositivos
            use_amp: Usar mixed precision
            batch_size: Tamaño de batch por defecto
            max_batch_size: Tamaño máximo de batch
        """
        self.model = model
        self.device_manager = device_manager or get_device_manager()
        self.use_amp = use_amp and self.device_manager.is_cuda_available
        self.batch_size = batch_size
        self.max_batch_size = max_batch_size
        
        # Mover modelo a dispositivo
        self.model = self.device_manager.move_to_device(self.model)
        self.model.eval()
        
        logger.info("Inference Engine initialized")
    
    @torch.no_grad()
    def predict(
        self,
        inputs: Union[torch.Tensor, np.ndarray, List],
        batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Realizar predicción.
        
        Args:
            inputs: Datos de entrada
            batch_size: Tamaño de batch (opcional)
            
        Returns:
            Predicciones
        """
        batch_size = batch_size or self.batch_size
        
        # Convertir a tensor si es necesario
        if isinstance(inputs, (list, np.ndarray)):
            inputs = torch.from_numpy(np.array(inputs)).float()
        
        # Mover a dispositivo
        inputs = self.device_manager.move_to_device(inputs)
        
        # Batching automático si es necesario
        if len(inputs) > batch_size:
            predictions = []
            for i in range(0, len(inputs), batch_size):
                batch = inputs[i:i + batch_size]
                batch_pred = self._predict_batch(batch)
                predictions.append(batch_pred)
            return torch.cat(predictions, dim=0)
        else:
            return self._predict_batch(inputs)
    
    def _predict_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Predecir un batch."""
        if self.use_amp:
            with torch.cuda.amp.autocast():
                return self.model(batch)
        else:
            return self.model(batch)
    
    @torch.no_grad()
    def predict_sequence(
        self,
        initial_state: torch.Tensor,
        num_steps: int,
        **kwargs
    ) -> torch.Tensor:
        """
        Predecir secuencia autoregresivamente.
        
        Args:
            initial_state: Estado inicial
            num_steps: Número de pasos a predecir
            **kwargs: Argumentos adicionales
            
        Returns:
            Secuencia predicha
        """
        if hasattr(self.model, 'predict_sequence'):
            return self.model.predict_sequence(initial_state, num_steps, **kwargs)
        else:
            # Implementación genérica
            predictions = []
            current_state = self.device_manager.move_to_device(initial_state)
            
            for _ in range(num_steps):
                pred = self._predict_batch(current_state)
                predictions.append(pred)
                # Actualizar estado para siguiente iteración
                if current_state.dim() > 1:
                    # Desplazar ventana
                    current_state = torch.cat([current_state[:, 1:], pred.unsqueeze(1)], dim=1)
                else:
                    current_state = pred
            
            return torch.stack(predictions, dim=1)
    
    def optimize_for_inference(self):
        """Optimizar modelo para inferencia."""
        # Compilar modelo si PyTorch 2.0+
        if hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
                logger.info("Model compiled for inference")
            except Exception as e:
                logger.warning(f"Could not compile model: {e}")
        
        # JIT tracing si es posible
        # (requiere ejemplo de entrada)
    
    def export_to_onnx(
        self,
        output_path: str,
        example_input: torch.Tensor,
        opset_version: int = 14
    ):
        """
        Exportar modelo a ONNX.
        
        Args:
            output_path: Ruta de salida
            example_input: Ejemplo de entrada para tracing
            opset_version: Versión de opset ONNX
        """
        try:
            import torch.onnx
            
            example_input = self.device_manager.move_to_device(example_input)
            
            torch.onnx.export(
                self.model,
                example_input,
                output_path,
                opset_version=opset_version,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            logger.info(f"Model exported to ONNX: {output_path}")
        except ImportError:
            logger.error("ONNX not available")
        except Exception as e:
            logger.error(f"Error exporting to ONNX: {e}")


class BatchInferenceEngine(InferenceEngine):
    """
    Motor de inferencia optimizado para batches grandes.
    """
    
    def __init__(self, *args, **kwargs):
        """Inicializar con optimizaciones adicionales."""
        super().__init__(*args, **kwargs)
        self.optimize_for_inference()
    
    def predict_large_batch(
        self,
        inputs: torch.Tensor,
        chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Predecir batch grande en chunks.
        
        Args:
            inputs: Datos de entrada
            chunk_size: Tamaño de chunk
            
        Returns:
            Predicciones
        """
        chunk_size = chunk_size or self.batch_size
        
        if len(inputs) <= chunk_size:
            return self.predict(inputs)
        
        predictions = []
        for i in range(0, len(inputs), chunk_size):
            chunk = inputs[i:i + chunk_size]
            chunk_pred = self.predict(chunk)
            predictions.append(chunk_pred)
        
        return torch.cat(predictions, dim=0)









