"""
NaN Detector
============

Detección de NaN e Inf en modelos.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class NaNDetector:
    """
    Detector de NaN e Inf.
    """
    
    def __init__(self, model: nn.Module):
        """
        Inicializar detector.
        
        Args:
            model: Modelo
        """
        self.model = model
    
    def check_weights(self) -> Dict[str, bool]:
        """
        Verificar NaN/Inf en pesos.
        
        Returns:
            Diccionario {nombre: tiene_nan_o_inf}
        """
        issues = {}
        
        for name, param in self.model.named_parameters():
            has_nan = torch.isnan(param.data).any().item()
            has_inf = torch.isinf(param.data).any().item()
            issues[name] = has_nan or has_inf
            
            if issues[name]:
                logger.warning(f"NaN/Inf detectado en pesos: {name}")
        
        return issues
    
    def check_output(self, output: torch.Tensor) -> Dict[str, Any]:
        """
        Verificar NaN/Inf en output.
        
        Args:
            output: Output del modelo
            
        Returns:
            Información de NaN/Inf
        """
        has_nan = torch.isnan(output).any().item()
        has_inf = torch.isinf(output).any().item()
        
        result = {
            "has_nan": has_nan,
            "has_inf": has_inf,
            "nan_count": torch.isnan(output).sum().item() if has_nan else 0,
            "inf_count": torch.isinf(output).sum().item() if has_inf else 0
        }
        
        if has_nan or has_inf:
            logger.warning(f"NaN/Inf en output: {result}")
        
        return result
    
    def check_gradients(self) -> Dict[str, Any]:
        """
        Verificar NaN/Inf en gradientes.
        
        Returns:
            Información de gradientes problemáticos
        """
        issues = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                has_nan = torch.isnan(param.grad).any().item()
                has_inf = torch.isinf(param.grad).any().item()
                
                if has_nan or has_inf:
                    issues[name] = {
                        "has_nan": has_nan,
                        "has_inf": has_inf
                    }
                    logger.warning(f"NaN/Inf en gradiente: {name}")
        
        return issues
    
    def full_check(self, input_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Verificación completa.
        
        Args:
            input_tensor: Input de prueba
            
        Returns:
            Resultado completo
        """
        result = {
            "weights": self.check_weights(),
            "output": None,
            "gradients": None
        }
        
        # Check output
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)
            result["output"] = self.check_output(output)
        
        # Check gradients
        if input_tensor.requires_grad:
            self.model.train()
            output = self.model(input_tensor)
            loss = output.mean()
            loss.backward()
            result["gradients"] = self.check_gradients()
        
        return result


def detect_nans(model: nn.Module, input_tensor: torch.Tensor) -> Dict[str, Any]:
    """
    Función helper para detectar NaN/Inf.
    
    Args:
        model: Modelo
        input_tensor: Input
        
    Returns:
        Resultado de detección
    """
    detector = NaNDetector(model)
    return detector.full_check(input_tensor)

