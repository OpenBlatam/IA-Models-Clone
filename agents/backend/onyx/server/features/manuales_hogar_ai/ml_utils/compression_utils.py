"""
Compression Utils - Utilidades de Compresión de Modelos
========================================================

Utilidades para compresión de modelos (pruning, quantization).
"""

import logging
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Optional, Dict, Any, List, Callable, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Intentar importar bibliotecas opcionales
try:
    import torch.quantization as quantization
    _has_quantization = True
except ImportError:
    _has_quantization = False
    logger.warning("torch.quantization not available, quantization features limited")


class ModelPruner:
    """
    Pruner de modelos con diferentes estrategias.
    """
    
    def __init__(self, model: nn.Module):
        """
        Inicializar pruner.
        
        Args:
            model: Modelo a podar
        """
        self.model = model
        self.pruning_config = {}
    
    def magnitude_pruning(
        self,
        module: nn.Module,
        amount: float = 0.2,
        name: str = "weight"
    ):
        """
        Magnitude-based pruning.
        
        Args:
            module: Módulo a podar
            amount: Cantidad de pruning (0-1)
            name: Nombre del parámetro
        """
        prune.l1_unstructured(module, name=name, amount=amount)
        prune.remove(module, name=name)
    
    def structured_pruning(
        self,
        module: nn.Module,
        amount: float = 0.2,
        dim: int = 0
    ):
        """
        Structured pruning.
        
        Args:
            module: Módulo a podar
            amount: Cantidad de pruning
            dim: Dimensión a podar
        """
        prune.ln_structured(module, name="weight", amount=amount, n=2, dim=dim)
    
    def global_pruning(
        self,
        parameters: List[Tuple[nn.Module, str]],
        amount: float = 0.2
    ):
        """
        Global pruning across multiple parameters.
        
        Args:
            parameters: Lista de (módulo, nombre_parametro)
            amount: Cantidad de pruning
        """
        prune.global_unstructured(
            parameters,
            pruning_method=prune.L1Unstructured,
            amount=amount
        )
    
    def iterative_pruning(
        self,
        train_fn: Callable,
        val_fn: Callable,
        initial_amount: float = 0.1,
        final_amount: float = 0.5,
        num_iterations: int = 5
    ) -> nn.Module:
        """
        Pruning iterativo con fine-tuning.
        
        Args:
            train_fn: Función de entrenamiento
            val_fn: Función de validación
            initial_amount: Cantidad inicial
            final_amount: Cantidad final
            num_iterations: Número de iteraciones
            
        Returns:
            Modelo podado
        """
        amounts = np.linspace(initial_amount, final_amount, num_iterations)
        
        for i, amount in enumerate(amounts):
            logger.info(f"Iteration {i + 1}/{num_iterations}, Pruning: {amount:.2%}")
            
            # Aplicar pruning
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    self.magnitude_pruning(module, amount=amount)
            
            # Fine-tuning
            train_fn(self.model)
            
            # Evaluación
            val_metrics = val_fn(self.model)
            logger.info(f"Validation metrics: {val_metrics}")
        
        return self.model


class ModelQuantizer:
    """
    Quantizador de modelos.
    """
    
    def __init__(self, model: nn.Module):
        """
        Inicializar quantizador.
        
        Args:
            model: Modelo a cuantizar
        """
        if not _has_quantization:
            raise ImportError("torch.quantization is required for ModelQuantizer")
        
        self.model = model
    
    def dynamic_quantization(self) -> nn.Module:
        """
        Aplicar dynamic quantization.
        
        Returns:
            Modelo cuantizado
        """
        quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {nn.Linear, nn.LSTM, nn.GRU},
            dtype=torch.qint8
        )
        return quantized_model
    
    def static_quantization(
        self,
        calibration_data: List[torch.Tensor],
        backend: str = "fbgemm"
    ) -> nn.Module:
        """
        Aplicar static quantization.
        
        Args:
            calibration_data: Datos para calibración
            backend: Backend de quantization
            
        Returns:
            Modelo cuantizado
        """
        self.model.eval()
        self.model.qconfig = quantization.get_default_qconfig(backend)
        
        # Preparar modelo
        quantization.prepare(self.model, inplace=True)
        
        # Calibrar
        with torch.no_grad():
            for data in calibration_data:
                _ = self.model(data)
        
        # Convertir
        quantized_model = quantization.convert(self.model, inplace=False)
        return quantized_model
    
    def qat_prepare(
        self,
        backend: str = "fbgemm"
    ) -> nn.Module:
        """
        Preparar modelo para Quantization-Aware Training (QAT).
        
        Args:
            backend: Backend de quantization
            
        Returns:
            Modelo preparado
        """
        self.model.train()
        self.model.qconfig = quantization.get_default_qat_qconfig(backend)
        quantization.prepare_qat(self.model, inplace=True)
        return self.model
    
    def qat_convert(self) -> nn.Module:
        """
        Convertir modelo QAT a quantizado.
        
        Returns:
            Modelo cuantizado
        """
        self.model.eval()
        quantized_model = quantization.convert(self.model, inplace=False)
        return quantized_model


class ModelCompressor:
    """
    Compresor completo de modelos.
    """
    
    def __init__(self, model: nn.Module):
        """
        Inicializar compresor.
        
        Args:
            model: Modelo a comprimir
        """
        self.model = model
        self.pruner = ModelPruner(model)
        self.quantizer = ModelQuantizer(model) if _has_quantization else None
    
    def compress(
        self,
        method: str = "pruning",
        **kwargs
    ) -> nn.Module:
        """
        Comprimir modelo.
        
        Args:
            method: Método ('pruning', 'quantization', 'both')
            **kwargs: Argumentos adicionales
            
        Returns:
            Modelo comprimido
        """
        if method == "pruning":
            amount = kwargs.get('amount', 0.2)
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    self.pruner.magnitude_pruning(module, amount=amount)
        
        elif method == "quantization":
            if self.quantizer is None:
                raise ImportError("Quantization not available")
            self.model = self.quantizer.dynamic_quantization()
        
        elif method == "both":
            # Primero pruning
            amount = kwargs.get('pruning_amount', 0.2)
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    self.pruner.magnitude_pruning(module, amount=amount)
            
            # Luego quantization
            if self.quantizer is not None:
                self.model = self.quantizer.dynamic_quantization()
        
        return self.model
    
    def get_model_size(self) -> Dict[str, Any]:
        """
        Obtener tamaño del modelo.
        
        Returns:
            Diccionario con información de tamaño
        """
        param_size = 0
        buffer_size = 0
        
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_all_mb = (param_size + buffer_size) / 1024**2
        
        return {
            'param_size_mb': param_size / 1024**2,
            'buffer_size_mb': buffer_size / 1024**2,
            'total_size_mb': size_all_mb,
            'num_parameters': sum(p.numel() for p in self.model.parameters())
        }

