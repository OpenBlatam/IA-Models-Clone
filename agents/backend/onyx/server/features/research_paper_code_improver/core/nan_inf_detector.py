"""
NaN/Inf Detection System - Sistema de detección de NaN/Inf
===========================================================
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class AnomalyReport:
    """Reporte de anomalías"""
    layer_name: str
    anomaly_type: str  # "nan", "inf", "both"
    count: int
    percentage: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            "layer_name": self.layer_name,
            "anomaly_type": self.anomaly_type,
            "count": self.count,
            "percentage": self.percentage,
            "timestamp": self.timestamp.isoformat()
        }


class NaNInfDetector:
    """Detector de NaN/Inf"""
    
    def __init__(self, enable_autograd_detection: bool = True):
        self.enable_autograd_detection = enable_autograd_detection
        self.reports: List[AnomalyReport] = []
        self.hooks = []
    
    def enable_detection(self):
        """Habilita detección de anomalías en autograd"""
        if self.enable_autograd_detection:
            torch.autograd.set_detect_anomaly(True)
            logger.info("Detección de anomalías en autograd habilitada")
    
    def disable_detection(self):
        """Deshabilita detección de anomalías"""
        if self.enable_autograd_detection:
            torch.autograd.set_detect_anomaly(False)
        self._remove_hooks()
        logger.info("Detección de anomalías deshabilitada")
    
    def register_hooks(self, model: nn.Module):
        """Registra hooks para detectar NaN/Inf"""
        def check_hook(name: str):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    nan_count = torch.isnan(output).sum().item()
                    inf_count = torch.isinf(output).sum().item()
                    
                    if nan_count > 0 or inf_count > 0:
                        total = output.numel()
                        anomaly_type = "both" if nan_count > 0 and inf_count > 0 else ("nan" if nan_count > 0 else "inf")
                        count = nan_count + inf_count
                        
                        report = AnomalyReport(
                            layer_name=name,
                            anomaly_type=anomaly_type,
                            count=count,
                            percentage=(count / total) * 100
                        )
                        self.reports.append(report)
                        
                        logger.warning(
                            f"Anomalía detectada en {name}: {anomaly_type}, "
                            f"count={count}, percentage={report.percentage:.2f}%"
                        )
            
            return hook
        
        for name, module in model.named_modules():
            hook = module.register_forward_hook(check_hook(name))
            self.hooks.append(hook)
        
        logger.info(f"Hooks registrados en {len(self.hooks)} módulos")
    
    def _remove_hooks(self):
        """Remueve hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def check_tensor(self, tensor: torch.Tensor, name: str = "tensor") -> Dict[str, Any]:
        """Verifica un tensor por NaN/Inf"""
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        total = tensor.numel()
        
        has_nan = nan_count > 0
        has_inf = inf_count > 0
        
        return {
            "name": name,
            "has_nan": has_nan,
            "has_inf": has_inf,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "nan_percentage": (nan_count / total) * 100 if total > 0 else 0,
            "inf_percentage": (inf_count / total) * 100 if total > 0 else 0,
            "total_elements": total
        }
    
    def check_model_parameters(self, model: nn.Module) -> List[Dict[str, Any]]:
        """Verifica todos los parámetros del modelo"""
        results = []
        
        for name, param in model.named_parameters():
            result = self.check_tensor(param.data, name)
            results.append(result)
            
            if result["has_nan"] or result["has_inf"]:
                logger.error(f"Anomalía en parámetro {name}: {result}")
        
        return results
    
    def fix_nan_inf(self, tensor: torch.Tensor, method: str = "zero") -> torch.Tensor:
        """Corrige NaN/Inf en un tensor"""
        if method == "zero":
            tensor = torch.where(torch.isnan(tensor) | torch.isinf(tensor), torch.zeros_like(tensor), tensor)
        elif method == "mean":
            mean_value = tensor[~torch.isnan(tensor) & ~torch.isinf(tensor)].mean()
            tensor = torch.where(torch.isnan(tensor) | torch.isinf(tensor), mean_value, tensor)
        elif method == "clip":
            tensor = torch.clamp(tensor, min=-1e6, max=1e6)
            tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
        
        return tensor
    
    def get_reports(self) -> List[Dict[str, Any]]:
        """Obtiene reportes de anomalías"""
        return [r.to_dict() for r in self.reports]
    
    def clear_reports(self):
        """Limpia reportes"""
        self.reports.clear()




