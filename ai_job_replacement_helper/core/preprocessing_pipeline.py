"""
Preprocessing Pipeline - Pipeline de preprocesamiento
=======================================================

Sistema de pipeline para preprocesamiento de datos.
Sigue mejores prácticas de pipelines funcionales.
"""

import logging
from typing import Dict, List, Optional, Any, Callable, Iterator
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class PreprocessingStep(ABC):
    """Paso de preprocesamiento"""
    
    @abstractmethod
    def __call__(self, data: Any) -> Any:
        """Aplicar paso de preprocesamiento"""
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Obtener información del paso"""
        pass


class PreprocessingPipeline:
    """Pipeline de preprocesamiento"""
    
    def __init__(self, steps: Optional[List[PreprocessingStep]] = None):
        """
        Args:
            steps: Lista de pasos de preprocesamiento
        """
        self.steps = steps or []
        logger.info(f"PreprocessingPipeline initialized with {len(self.steps)} steps")
    
    def add_step(self, step: PreprocessingStep) -> None:
        """Agregar paso al pipeline"""
        self.steps.append(step)
        logger.info(f"Added preprocessing step: {step.__class__.__name__}")
    
    def __call__(self, data: Any) -> Any:
        """Aplicar pipeline completo"""
        result = data
        for step in self.steps:
            try:
                result = step(result)
            except Exception as e:
                logger.error(f"Error in preprocessing step {step.__class__.__name__}: {e}")
                raise
        return result
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Obtener información del pipeline"""
        return {
            "num_steps": len(self.steps),
            "steps": [step.get_info() for step in self.steps],
        }


class NormalizationStep(PreprocessingStep):
    """Paso de normalización"""
    
    def __init__(
        self,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        compute_from_data: bool = False
    ):
        """
        Args:
            mean: Media (None = calcular)
            std: Desviación estándar (None = calcular)
            compute_from_data: Calcular desde datos
        """
        self.mean = mean
        self.std = std
        self.compute_from_data = compute_from_data
        self.computed_mean = None
        self.computed_std = None
    
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Normalizar datos"""
        if self.compute_from_data:
            if self.computed_mean is None:
                self.computed_mean = data.mean().item()
                self.computed_std = data.std().item()
            mean_val = self.computed_mean
            std_val = self.computed_std
        else:
            if self.mean is None or self.std is None:
                mean_val = data.mean().item()
                std_val = data.std().item()
            else:
                mean_val = self.mean[0] if isinstance(self.mean, list) else self.mean
                std_val = self.std[0] if isinstance(self.std, list) else self.std
        
        return (data - mean_val) / (std_val + 1e-8)
    
    def get_info(self) -> Dict[str, Any]:
        """Obtener información"""
        return {
            "type": "normalization",
            "mean": self.mean or self.computed_mean,
            "std": self.std or self.computed_std,
        }


class ReshapeStep(PreprocessingStep):
    """Paso de reshape"""
    
    def __init__(self, target_shape: tuple):
        """
        Args:
            target_shape: Forma objetivo
        """
        self.target_shape = target_shape
    
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Reshape datos"""
        return data.reshape(self.target_shape)
    
    def get_info(self) -> Dict[str, Any]:
        """Obtener información"""
        return {
            "type": "reshape",
            "target_shape": self.target_shape,
        }


class PreprocessingPipelineService:
    """Servicio de pipelines de preprocesamiento"""
    
    @staticmethod
    def create_image_pipeline(
        size: tuple = (224, 224),
        normalize: bool = True,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None
    ) -> PreprocessingPipeline:
        """
        Crear pipeline para imágenes.
        
        Args:
            size: Tamaño objetivo
            normalize: Si normalizar
            mean: Media (opcional)
            std: Desviación estándar (opcional)
        
        Returns:
            PreprocessingPipeline
        """
        pipeline = PreprocessingPipeline()
        
        # Reshape step
        pipeline.add_step(ReshapeStep(target_shape=(-1, *size)))
        
        # Normalization step
        if normalize:
            pipeline.add_step(NormalizationStep(mean=mean, std=std))
        
        return pipeline
    
    @staticmethod
    def create_tensor_pipeline(
        normalize: bool = True,
        target_shape: Optional[tuple] = None
    ) -> PreprocessingPipeline:
        """
        Crear pipeline para tensores.
        
        Args:
            normalize: Si normalizar
            target_shape: Forma objetivo (opcional)
        
        Returns:
            PreprocessingPipeline
        """
        pipeline = PreprocessingPipeline()
        
        if target_shape:
            pipeline.add_step(ReshapeStep(target_shape=target_shape))
        
        if normalize:
            pipeline.add_step(NormalizationStep(compute_from_data=True))
        
        return pipeline




