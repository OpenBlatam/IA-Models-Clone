"""
Evaluation Metrics - Métricas de Evaluación
============================================

Métricas para evaluación de modelos.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error
)


class ClassificationMetrics:
    """Métricas para clasificación"""
    
    @staticmethod
    def compute_metrics(
        predictions: np.ndarray,
        labels: np.ndarray,
        average: str = "weighted"
    ) -> Dict[str, float]:
        """
        Calcular métricas de clasificación
        
        Args:
            predictions: Predicciones
            labels: Labels verdaderos
            average: Tipo de promedio
            
        Returns:
            Dict con métricas
        """
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average=average, zero_division=0
        )
        
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }
    
    @staticmethod
    def compute_confusion_matrix(
        predictions: np.ndarray,
        labels: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Calcular matriz de confusión
        
        Args:
            predictions: Predicciones
            labels: Labels verdaderos
            class_names: Nombres de clases
            
        Returns:
            Dict con matriz de confusión
        """
        cm = confusion_matrix(labels, predictions)
        
        result = {
            "matrix": cm.tolist(),
            "shape": cm.shape
        }
        
        if class_names:
            result["class_names"] = class_names
        
        return result


class RegressionMetrics:
    """Métricas para regresión"""
    
    @staticmethod
    def compute_metrics(
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Calcular métricas de regresión
        
        Args:
            predictions: Predicciones
            labels: Labels verdaderos
            
        Returns:
            Dict con métricas
        """
        mse = mean_squared_error(labels, predictions)
        mae = mean_absolute_error(labels, predictions)
        rmse = np.sqrt(mse)
        
        # R² score
        ss_res = np.sum((labels - predictions) ** 2)
        ss_tot = np.sum((labels - np.mean(labels)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        return {
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2)
        }


class TextGenerationMetrics:
    """Métricas para generación de texto"""
    
    @staticmethod
    def compute_bleu(
        predictions: List[str],
        references: List[str],
        n_gram: int = 4
    ) -> float:
        """
        Calcular BLEU score
        
        Args:
            predictions: Textos generados
            references: Textos de referencia
            n_gram: N-gram máximo
            
        Returns:
            BLEU score
        """
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            
            smoothing = SmoothingFunction().method1
            scores = []
            
            for pred, ref in zip(predictions, references):
                pred_tokens = pred.split()
                ref_tokens = ref.split()
                score = sentence_bleu(
                    [ref_tokens],
                    pred_tokens,
                    smoothing_function=smoothing
                )
                scores.append(score)
            
            return float(np.mean(scores))
        except ImportError:
            return 0.0
    
    @staticmethod
    def compute_perplexity(
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        """
        Calcular perplexity
        
        Args:
            logits: Logits del modelo
            labels: Labels verdaderos
            
        Returns:
            Perplexity
        """
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        perplexity = torch.exp(loss)
        return float(perplexity.item())


class ModelEvaluator:
    """Evaluador completo de modelos"""
    
    def __init__(self, task_type: str = "classification"):
        """
        Inicializar evaluador
        
        Args:
            task_type: Tipo de tarea (classification, regression, generation)
        """
        self.task_type = task_type
    
    def evaluate(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: str = "cuda"
    ) -> Dict[str, Any]:
        """
        Evaluar modelo completo
        
        Args:
            model: Modelo a evaluar
            dataloader: DataLoader de evaluación
            device: Dispositivo
            
        Returns:
            Dict con métricas
        """
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = model(**batch)
                
                if self.task_type == "classification":
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(batch["labels"].cpu().numpy())
                elif self.task_type == "regression":
                    predictions = outputs.logits.squeeze()
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(batch["labels"].cpu().numpy())
        
        # Calcular métricas
        if self.task_type == "classification":
            return ClassificationMetrics.compute_metrics(
                np.array(all_predictions),
                np.array(all_labels)
            )
        elif self.task_type == "regression":
            return RegressionMetrics.compute_metrics(
                np.array(all_predictions),
                np.array(all_labels)
            )
        
        return {}




