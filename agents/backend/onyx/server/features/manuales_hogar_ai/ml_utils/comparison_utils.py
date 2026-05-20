"""
Comparison Utils - Utilidades de Comparación de Modelos
========================================================

Utilidades para comparar modelos y sus resultados.
"""

import logging
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ModelComparison:
    """Resultado de comparación de modelos."""
    model_name: str
    metrics: Dict[str, float]
    parameters: int
    inference_time: float
    memory_usage: float


class ModelComparator:
    """
    Comparador de modelos.
    """
    
    def __init__(self):
        """Inicializar comparador."""
        self.comparisons: List[ModelComparison] = []
    
    def compare_models(
        self,
        models: Dict[str, nn.Module],
        dataloader: torch.utils.data.DataLoader,
        device: str = "cuda",
        num_runs: int = 5
    ) -> Dict[str, ModelComparison]:
        """
        Comparar múltiples modelos.
        
        Args:
            models: Diccionario de modelos {name: model}
            dataloader: DataLoader para evaluación
            device: Dispositivo
            num_runs: Número de ejecuciones para promediar
            
        Returns:
            Diccionario de comparaciones
        """
        results = {}
        
        for name, model in models.items():
            logger.info(f"Evaluating model: {name}")
            
            model = model.to(device)
            model.eval()
            
            # Contar parámetros
            num_params = sum(p.numel() for p in model.parameters())
            
            # Medir tiempo de inferencia
            inference_times = []
            memory_usages = []
            
            with torch.no_grad():
                for _ in range(num_runs):
                    import time
                    if device == "cuda":
                        torch.cuda.synchronize()
                        start_memory = torch.cuda.memory_allocated()
                    
                    start_time = time.time()
                    
                    for batch in dataloader:
                        if isinstance(batch, (list, tuple)):
                            inputs = batch[0].to(device)
                        else:
                            inputs = batch.to(device)
                        _ = model(inputs)
                    
                    if device == "cuda":
                        torch.cuda.synchronize()
                        end_memory = torch.cuda.memory_allocated()
                        memory_usages.append((end_memory - start_memory) / 1024**2)
                    
                    elapsed = time.time() - start_time
                    inference_times.append(elapsed)
            
            avg_inference_time = np.mean(inference_times)
            avg_memory = np.mean(memory_usages) if memory_usages else 0.0
            
            # Calcular métricas
            metrics = self._compute_metrics(model, dataloader, device)
            
            comparison = ModelComparison(
                model_name=name,
                metrics=metrics,
                parameters=num_params,
                inference_time=avg_inference_time,
                memory_usage=avg_memory
            )
            
            results[name] = comparison
            self.comparisons.append(comparison)
        
        return results
    
    def _compute_metrics(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: str
    ) -> Dict[str, float]:
        """
        Calcular métricas.
        
        Args:
            model: Modelo
            dataloader: DataLoader
            device: Dispositivo
            
        Returns:
            Diccionario de métricas
        """
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0], batch[1]
                else:
                    inputs, targets = batch, None
                
                inputs = inputs.to(device)
                outputs = model(inputs)
                
                if targets is not None:
                    targets = targets.to(device)
                    preds = outputs.argmax(dim=1)
                    all_preds.append(preds.cpu())
                    all_targets.append(targets.cpu())
        
        if len(all_preds) > 0:
            all_preds = torch.cat(all_preds)
            all_targets = torch.cat(all_targets)
            
            accuracy = (all_preds == all_targets).float().mean().item()
            
            return {
                'accuracy': accuracy
            }
        
        return {}
    
    def get_best_model(
        self,
        metric: str = "accuracy",
        maximize: bool = True
    ) -> Optional[ModelComparison]:
        """
        Obtener mejor modelo según métrica.
        
        Args:
            metric: Nombre de métrica
            maximize: Maximizar métrica
            
        Returns:
            Mejor modelo
        """
        if not self.comparisons:
            return None
        
        best = self.comparisons[0]
        best_value = best.metrics.get(metric, 0.0)
        
        for comp in self.comparisons[1:]:
            value = comp.metrics.get(metric, 0.0)
            if (maximize and value > best_value) or (not maximize and value < best_value):
                best = comp
                best_value = value
        
        return best
    
    def generate_report(self) -> str:
        """
        Generar reporte de comparación.
        
        Returns:
            Reporte en texto
        """
        if not self.comparisons:
            return "No comparisons available"
        
        report = "Model Comparison Report\n"
        report += "=" * 50 + "\n\n"
        
        for comp in self.comparisons:
            report += f"Model: {comp.model_name}\n"
            report += f"  Parameters: {comp.parameters:,}\n"
            report += f"  Inference Time: {comp.inference_time:.4f}s\n"
            report += f"  Memory Usage: {comp.memory_usage:.2f} MB\n"
            report += "  Metrics:\n"
            for metric, value in comp.metrics.items():
                report += f"    {metric}: {value:.4f}\n"
            report += "\n"
        
        return report




