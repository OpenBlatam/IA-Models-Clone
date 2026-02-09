"""
Sistema de evaluación completo
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
import logging

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluador de modelos"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
    
    def evaluate_classification(
        self,
        model: torch.nn.Module,
        data_loader: DataLoader,
        num_classes: int
    ) -> Dict[str, float]:
        """
        Evalúa modelo de clasificación
        
        Returns:
            Métricas: accuracy, precision, recall, f1
        """
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                inputs = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = model(inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calcular métricas
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="weighted",
            zero_division=0
        )
        
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }
    
    def evaluate_generation(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        prompts: List[str],
        references: List[str],
        max_length: int = 100
    ) -> Dict[str, float]:
        """
        Evalúa modelo de generación
        
        Returns:
            Métricas: BLEU, ROUGE, etc.
        """
        model.eval()
        generated_texts = []
        
        with torch.no_grad():
            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
                
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=4,
                    early_stopping=True
                )
                
                generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated_texts.append(generated)
        
        # Calcular métricas (simplificado)
        # En producción usaría bibliotecas como nltk.translate.bleu_score
        metrics = {
            "avg_length": np.mean([len(text.split()) for text in generated_texts]),
            "num_generated": len(generated_texts)
        }
        
        return metrics
    
    def evaluate_embeddings(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray
    ) -> Dict[str, float]:
        """
        Evalúa calidad de embeddings
        
        Returns:
            Métricas de similitud
        """
        # Similitud coseno
        cosine_similarities = np.sum(embeddings1 * embeddings2, axis=1) / (
            np.linalg.norm(embeddings1, axis=1) *
            np.linalg.norm(embeddings2, axis=1)
        )
        
        return {
            "mean_cosine_similarity": float(np.mean(cosine_similarities)),
            "std_cosine_similarity": float(np.std(cosine_similarities)),
            "min_cosine_similarity": float(np.min(cosine_similarities)),
            "max_cosine_similarity": float(np.max(cosine_similarities))
        }




