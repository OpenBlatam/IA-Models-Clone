"""
Model Comparison - Comparación de modelos
==========================================
Comparación sistemática de múltiples modelos
"""

import logging
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ModelComparisonResult:
    """Resultado de comparación de modelo"""
    model_name: str
    metrics: Dict[str, float]
    training_time: float
    inference_time: float
    model_size_mb: float
    num_parameters: int
    memory_usage_mb: float


class ModelComparator:
    """Sistema de comparación de modelos"""
    
    def __init__(self):
        self.comparison_results: List[ModelComparisonResult] = []
    
    def compare_models(
        self,
        models: Dict[str, torch.nn.Module],
        test_loader: torch.utils.data.DataLoader,
        device: torch.device,
        metrics_fn: Optional[Callable] = None
    ) -> List[ModelComparisonResult]:
        """Compara múltiples modelos"""
        results = []
        
        for model_name, model in models.items():
            logger.info(f"Evaluating model: {model_name}")
            
            # Métricas
            if metrics_fn:
                metrics = metrics_fn(model, test_loader, device)
            else:
                metrics = self._default_metrics(model, test_loader, device)
            
            # Tiempos
            training_time = 0.0  # En producción, medir tiempo real
            inference_time = self._measure_inference_time(model, test_loader, device)
            
            # Tamaño
            model_size_mb = self._calculate_model_size(model)
            num_parameters = sum(p.numel() for p in model.parameters())
            memory_usage_mb = self._measure_memory_usage(model, device)
            
            result = ModelComparisonResult(
                model_name=model_name,
                metrics=metrics,
                training_time=training_time,
                inference_time=inference_time,
                model_size_mb=model_size_mb,
                num_parameters=num_parameters,
                memory_usage_mb=memory_usage_mb
            )
            
            results.append(result)
            self.comparison_results.append(result)
        
        return results
    
    def _default_metrics(
        self,
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> Dict[str, float]:
        """Métricas por defecto"""
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, dict):
                    inputs = batch["input"].to(device)
                    labels = batch["label"].to(device)
                else:
                    inputs, labels = batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1) if outputs.dim() > 1 else outputs
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Métricas básicas
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(all_labels, all_preds)
        
        return {"accuracy": accuracy}
    
    def _measure_inference_time(
        self,
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device,
        num_runs: int = 10
    ) -> float:
        """Mide tiempo de inferencia"""
        model.eval()
        times = []
        
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if i >= num_runs:
                    break
                
                if isinstance(batch, dict):
                    inputs = batch["input"].to(device)
                else:
                    inputs = batch[0].to(device)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                else:
                    import time
                    start_time = time.time()
                
                _ = model(inputs)
                
                if torch.cuda.is_available():
                    end.record()
                    torch.cuda.synchronize()
                    times.append(start.elapsed_time(end) / 1000.0)
                else:
                    times.append(time.time() - start_time)
        
        return np.mean(times) if times else 0.0
    
    def _calculate_model_size(self, model: torch.nn.Module) -> float:
        """Calcula tamaño del modelo en MB"""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        total_size = param_size + buffer_size
        return total_size / (1024 * 1024)  # MB
    
    def _measure_memory_usage(
        self,
        model: torch.nn.Module,
        device: torch.device
    ) -> float:
        """Mide uso de memoria"""
        if torch.cuda.is_available() and device.type == "cuda":
            return torch.cuda.memory_allocated(device) / (1024 * 1024)  # MB
        else:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)  # MB
    
    def generate_comparison_report(self) -> Dict[str, Any]:
        """Genera reporte de comparación"""
        if not self.comparison_results:
            return {"error": "No comparison results available"}
        
        # Encontrar mejores modelos por métrica
        best_accuracy = max(self.comparison_results, key=lambda x: x.metrics.get("accuracy", 0))
        fastest = min(self.comparison_results, key=lambda x: x.inference_time)
        smallest = min(self.comparison_results, key=lambda x: x.model_size_mb)
        
        return {
            "total_models": len(self.comparison_results),
            "best_accuracy": {
                "model": best_accuracy.model_name,
                "accuracy": best_accuracy.metrics.get("accuracy", 0)
            },
            "fastest": {
                "model": fastest.model_name,
                "inference_time": fastest.inference_time
            },
            "smallest": {
                "model": smallest.model_name,
                "size_mb": smallest.model_size_mb
            },
            "all_results": [
                {
                    "model_name": r.model_name,
                    "metrics": r.metrics,
                    "inference_time": r.inference_time,
                    "model_size_mb": r.model_size_mb,
                    "num_parameters": r.num_parameters
                }
                for r in self.comparison_results
            ]
        }
    
    def clear_results(self):
        """Limpia resultados"""
        self.comparison_results.clear()




