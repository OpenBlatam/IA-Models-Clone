"""
Model Comparison Tool - Herramienta de comparación de modelos
=============================================================
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

from .base_classes import BaseManager, BaseConfig
from .common_utils import (
    get_device, move_to_device, count_parameters,
    estimate_flops, measure_inference_time, calculate_model_size,
    get_model_output, extract_predictions, calculate_accuracy
)

logger = logging.getLogger(__name__)


@dataclass
class ModelComparisonResult:
    """Resultado de comparación de modelos"""
    model_name: str
    parameters: int
    flops: int
    inference_time: float
    memory_usage: float
    accuracy: float
    loss: float
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            "model_name": self.model_name,
            "parameters": self.parameters,
            "flops": self.flops,
            "inference_time_ms": self.inference_time * 1000,
            "memory_usage_mb": self.memory_usage,
            "accuracy": self.accuracy,
            "loss": self.loss,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat()
        }


class ModelComparator(BaseManager):
    """Comparador de modelos"""
    
    def __init__(self, config: Optional[BaseConfig] = None):
        super().__init__(config or BaseConfig())
        self.comparison_results: List[ModelComparisonResult] = []
    
    def compare_models(
        self,
        models: Dict[str, nn.Module],
        test_loader: Any,
        device: Optional[str] = None,
        loss_fn: Optional[Callable] = None
    ) -> List[ModelComparisonResult]:
        """Compara múltiples modelos usando utilidades compartidas"""
        results = []
        device_obj = get_device(device)
        
        for name, model in models.items():
            logger.info(f"Comparando modelo: {name}")
            result = self._evaluate_model(
                name, model, test_loader, device_obj, loss_fn
            )
            results.append(result)
            self.comparison_results.append(result)
        
        self.log_event("model_comparison", {"num_models": len(models)})
        return results
    
    def _evaluate_model(
        self,
        name: str,
        model: nn.Module,
        test_loader: Any,
        device: torch.device,
        loss_fn: Optional[Callable]
    ) -> ModelComparisonResult:
        """Evalúa un modelo individual"""
        model = model.to(device)
        model.eval()
        
        # Contar parámetros usando utilidades compartidas
        param_info = count_parameters(model)
        total_params = param_info["total"]
        
        # Estimar FLOPs usando utilidades compartidas
        try:
            batch = next(iter(test_loader))
            if isinstance(batch, dict):
                example_input = batch.get("input_ids") or batch.get("inputs")
            else:
                example_input = batch[0] if isinstance(batch, tuple) else batch
            input_shape = tuple(example_input.shape)
            flops = estimate_flops(model, input_shape, str(device))
        except Exception:
            flops = 0
        
        # Medir tiempo de inferencia usando utilidades compartidas
        try:
            inference_time_ms = measure_inference_time(
                model, example_input, num_runs=10, warmup=3, device=str(device)
            )
            inference_time = inference_time_ms / 1000.0  # Convert to seconds
        except Exception:
            inference_time = 0.0
        
        # Medir uso de memoria usando utilidades compartidas
        memory_usage = calculate_model_size(model)
        
        # Evaluar accuracy y loss usando utilidades compartidas
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in test_loader:
                # Mover batch a device usando utilidades compartidas
                batch = move_to_device(batch, device)
                
                # Obtener labels
                if isinstance(batch, dict):
                    labels = batch.get("labels") or batch.get("targets")
                else:
                    labels = batch[1] if isinstance(batch, tuple) else None
                
                # Obtener outputs usando utilidades compartidas
                outputs = get_model_output(model, batch, str(device))
                
                if loss_fn and labels is not None:
                    loss = loss_fn(outputs, labels)
                    total_loss += loss.item()
                    num_batches += 1
                
                # Extraer predicciones usando utilidades compartidas
                predictions = extract_predictions(outputs)
                
                if labels is not None:
                    all_predictions.append(predictions)
                    all_labels.append(labels)
        
        # Calcular accuracy usando utilidades compartidas
        if all_predictions and all_labels:
            predictions_tensor = torch.cat(all_predictions)
            labels_tensor = torch.cat(all_labels)
            accuracy = calculate_accuracy(predictions_tensor, labels_tensor) * 100  # Convert to percentage
        else:
            accuracy = 0.0
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Calcular métricas adicionales
        from sklearn.metrics import precision_score, recall_score, f1_score
        if all_predictions and all_labels:
            predictions_np = torch.cat(all_predictions).cpu().numpy()
            labels_np = torch.cat(all_labels).cpu().numpy()
            metrics = {
                "precision": precision_score(labels_np, predictions_np, average='weighted', zero_division=0),
                "recall": recall_score(labels_np, predictions_np, average='weighted', zero_division=0),
                "f1": f1_score(labels_np, predictions_np, average='weighted', zero_division=0)
            }
        else:
            metrics = {}
        
        return ModelComparisonResult(
            model_name=name,
            parameters=total_params,
            flops=flops,
            inference_time=inference_time,
            memory_usage=memory_usage,
            accuracy=accuracy,
            loss=avg_loss,
            metrics=metrics
        )
    
    def _estimate_flops(self, model: nn.Module) -> int:
        """Estima FLOPs del modelo"""
        flops = 0
        
        def count_flops_hook(module, input, output):
            nonlocal flops
            if isinstance(module, nn.Linear):
                flops += input[0].shape[1] * output.shape[1]
            elif isinstance(module, nn.Conv2d):
                kernel_flops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
                output_elements = output.numel()
                flops += kernel_flops * output_elements
        
        hooks = []
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                hooks.append(module.register_forward_hook(count_flops_hook))
        
        # Forward pass dummy
        dummy_input = torch.randn(1, 3, 224, 224)
        try:
            with torch.no_grad():
                _ = model(dummy_input)
        except:
            pass
        
        for hook in hooks:
            hook.remove()
        
        return flops
    
    def _measure_inference_time(
        self,
        model: nn.Module,
        test_loader: Any,
        device: torch.device,
        num_batches: int = 10
    ) -> float:
        """Mide tiempo de inferencia"""
        model.eval()
        times = []
        
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if i >= num_batches:
                    break
                
                if isinstance(batch, dict):
                    inputs = batch.get("input_ids") or batch.get("inputs")
                    inputs = inputs.to(device)
                else:
                    inputs = batch[0].to(device)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                
                if start:
                    start.record()
                else:
                    import time
                    start_time = time.time()
                
                if isinstance(batch, dict):
                    _ = model(**batch)
                else:
                    _ = model(inputs)
                
                if end:
                    end.record()
                    torch.cuda.synchronize()
                    times.append(start.elapsed_time(end) / 1000.0)
                else:
                    times.append(time.time() - start_time)
        
        return sum(times) / len(times) if times else 0.0
    
    def get_comparison_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de comparación"""
        if not self.comparison_results:
            return {}
        
        # Encontrar mejores modelos por métrica
        best_accuracy = max(self.comparison_results, key=lambda x: x.accuracy)
        best_speed = min(self.comparison_results, key=lambda x: x.inference_time)
        smallest_model = min(self.comparison_results, key=lambda x: x.parameters)
        
        return {
            "total_models": len(self.comparison_results),
            "best_accuracy": best_accuracy.to_dict(),
            "fastest": best_speed.to_dict(),
            "smallest": smallest_model.to_dict(),
            "all_results": [r.to_dict() for r in self.comparison_results]
        }

