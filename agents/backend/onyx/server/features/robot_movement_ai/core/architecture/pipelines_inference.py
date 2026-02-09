"""
Inference Pipeline Module
==========================

Pipeline profesional para inferencia de modelos de deep learning.
Incluye optimizaciones, batch processing y post-processing.
"""

import logging
from typing import Dict, Any, Optional, Union, List, Callable
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.cuda.amp import autocast
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    logging.warning("PyTorch not available. Inference pipeline will be disabled.")

from .pipelines_config import InferenceConfig

logger = logging.getLogger(__name__)


class InferencePipeline:
    """
    Pipeline profesional para inferencia de modelos.
    
    Características:
    - Batch processing optimizado
    - Mixed precision inference
    - Model compilation (torch.compile)
    - TorchScript optimization
    - Post-processing pipelines
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[InferenceConfig] = None,
        post_process_fn: Optional[Callable] = None
    ):
        """
        Inicializar pipeline de inferencia.
        
        Args:
            model: Modelo PyTorch entrenado
            config: Configuración de inferencia
            post_process_fn: Función de post-procesamiento opcional
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for InferencePipeline")
        
        self.model = model
        self.config = config or InferenceConfig()
        self.post_process_fn = post_process_fn
        
        # Device setup
        self.device = torch.device(self.config.device)
        self.model.to(self.device)
        self.model.eval()
        
        # Model optimization
        if self.config.compile_model and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model)
                logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"Failed to compile model: {e}")
        
        if self.config.use_jit:
            try:
                # Example: trace model (would need example input)
                logger.info("TorchScript optimization available (requires example input)")
            except Exception as e:
                logger.warning(f"Failed to create TorchScript: {e}")
        
        logger.info(f"InferencePipeline initialized on {self.device}")
    
    def predict(
        self,
        inputs: Union[np.ndarray, torch.Tensor, List],
        batch_size: Optional[int] = None,
        return_dict: bool = False
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Realizar predicción en batch.
        
        Args:
            inputs: Inputs para el modelo (numpy array, tensor, o lista)
            batch_size: Tamaño de batch (usa config si None)
            return_dict: Retornar como diccionario
            
        Returns:
            Predicciones como numpy array o dict
        """
        batch_size = batch_size or self.config.batch_size
        
        # Convertir a tensor
        if isinstance(inputs, list):
            inputs = np.array(inputs)
        if isinstance(inputs, np.ndarray):
            inputs = torch.FloatTensor(inputs)
        
        inputs = inputs.to(self.device)
        
        # Batch processing
        predictions = []
        with torch.no_grad():
            for i in range(0, len(inputs), batch_size):
                batch = inputs[i:i + batch_size]
                
                if self.config.mixed_precision:
                    with autocast():
                        batch_pred = self.model(batch)
                else:
                    batch_pred = self.model(batch)
                
                predictions.append(batch_pred.cpu().numpy())
        
        result = np.concatenate(predictions, axis=0)
        
        # Post-processing
        if self.post_process_fn:
            result = self.post_process_fn(result)
        
        if return_dict:
            return {"predictions": result}
        return result
    
    def predict_single(
        self,
        input_data: Union[np.ndarray, torch.Tensor],
        return_dict: bool = False
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Predecir un solo ejemplo.
        
        Args:
            input_data: Input único
            return_dict: Retornar como diccionario
            
        Returns:
            Predicción como numpy array o dict
        """
        if isinstance(input_data, np.ndarray):
            input_data = torch.FloatTensor(input_data)
        
        # Asegurar dimensión de batch
        if input_data.ndim == 1:
            input_data = input_data.unsqueeze(0)
        
        input_data = input_data.to(self.device)
        
        with torch.no_grad():
            if self.config.mixed_precision:
                with autocast():
                    prediction = self.model(input_data)
            else:
                prediction = self.model(input_data)
        
        result = prediction.cpu().numpy().squeeze()
        
        # Post-processing
        if self.post_process_fn:
            result = self.post_process_fn(result)
        
        if return_dict:
            return {"prediction": result}
        return result
    
    def predict_stream(
        self,
        input_stream: List[Union[np.ndarray, torch.Tensor]],
        batch_size: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Predecir stream de inputs.
        
        Args:
            input_stream: Stream de inputs
            batch_size: Tamaño de batch
            
        Returns:
            Lista de predicciones
        """
        batch_size = batch_size or self.config.batch_size
        results = []
        
        for i in range(0, len(input_stream), batch_size):
            batch = input_stream[i:i + batch_size]
            batch_pred = self.predict(batch, batch_size=len(batch))
            results.extend(batch_pred)
        
        return results
    
    def predict_with_uncertainty(
        self,
        inputs: Union[np.ndarray, torch.Tensor],
        num_samples: int = 10,
        return_std: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Predecir con estimación de incertidumbre (Monte Carlo Dropout).
        
        Args:
            inputs: Inputs para el modelo
            num_samples: Número de muestras para MC
            return_std: Retornar desviación estándar
            
        Returns:
            Dict con predicción media y std (opcional)
        """
        # Activar dropout para MC
        self.model.train()
        
        predictions = []
        for _ in range(num_samples):
            pred = self.predict(inputs)
            predictions.append(pred)
        
        # Desactivar dropout
        self.model.eval()
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        
        result = {"mean": mean_pred}
        
        if return_std:
            std_pred = np.std(predictions, axis=0)
            result["std"] = std_pred
        
        return result

