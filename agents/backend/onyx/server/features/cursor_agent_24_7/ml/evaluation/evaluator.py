"""
Evaluator - Evaluador de modelos
=================================

Clase para evaluar modelos de deep learning.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader

from ..models.base import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Métricas de evaluación"""
    loss: float
    perplexity: Optional[float] = None
    accuracy: Optional[float] = None
    bleu_score: Optional[float] = None
    rouge_score: Optional[Dict[str, float]] = None


class Evaluator:
    """Evaluador de modelos"""
    
    def __init__(self, model: BaseModel):
        self.model = model
    
    def evaluate(
        self,
        data_loader: DataLoader,
        metrics: Optional[List[str]] = None
    ) -> EvaluationMetrics:
        """Evaluar modelo"""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_targets = []
        
        metrics = metrics or ["loss"]
        
        with torch.no_grad():
            for batch in data_loader:
                outputs = self.model(**batch)
                loss = outputs.loss
                total_loss += loss.item()
                num_batches += 1
                
                # Guardar predicciones y targets si es necesario
                if "perplexity" in metrics or "accuracy" in metrics:
                    if hasattr(outputs, 'logits'):
                        predictions = torch.argmax(outputs.logits, dim=-1)
                        all_predictions.extend(predictions.cpu().tolist())
                        
                        if "labels" in batch:
                            all_targets.extend(batch["labels"].cpu().tolist())
        
        avg_loss = total_loss / num_batches
        
        # Calcular métricas adicionales
        perplexity = None
        accuracy = None
        
        if "perplexity" in metrics:
            perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        if "accuracy" in metrics and all_predictions and all_targets:
            correct = sum(p == t for p, t in zip(all_predictions, all_targets))
            accuracy = correct / len(all_predictions)
        
        return EvaluationMetrics(
            loss=avg_loss,
            perplexity=perplexity,
            accuracy=accuracy
        )
    
    def predict(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Hacer predicción"""
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            if hasattr(outputs, 'logits'):
                return torch.argmax(outputs.logits, dim=-1)
            return outputs



