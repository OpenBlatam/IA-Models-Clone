"""
Advanced Metrics Module
========================

Métricas profesionales para evaluación de modelos.
Incluye métricas para diferentes tareas de deep learning.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

try:
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        mean_squared_error,
        mean_absolute_error,
        r2_score
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available. Some metrics will be disabled.")

try:
    import torchmetrics
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False
    logging.warning("torchmetrics not available. Install with: pip install torchmetrics")

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Calculadora de métricas profesionales.
    
    Soporta:
    - Clasificación (accuracy, precision, recall, F1, AUC)
    - Regresión (MSE, MAE, R², MAPE)
    - Secuencias (BLEU, ROUGE, perplexity)
    """
    
    def __init__(self, task: str = "classification"):
        """
        Inicializar calculadora de métricas.
        
        Args:
            task: Tipo de tarea ("classification", "regression", "sequence")
        """
        self.task = task
        self.metrics_history: List[Dict[str, float]] = []
        
        # Inicializar métricas de torchmetrics si está disponible
        if TORCHMETRICS_AVAILABLE and task == "classification":
            self.accuracy_metric = torchmetrics.Accuracy()
            self.precision_metric = torchmetrics.Precision()
            self.recall_metric = torchmetrics.Recall()
            self.f1_metric = torchmetrics.F1Score()
    
    def calculate_classification_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        average: str = "weighted"
    ) -> Dict[str, float]:
        """
        Calcular métricas de clasificación.
        
        Args:
            predictions: Predicciones (probabilidades o clases)
            targets: Targets reales
            average: Tipo de promedio para métricas multiclase
            
        Returns:
            Dict con métricas
        """
        # Convertir a clases si son probabilidades
        if predictions.ndim > 1:
            pred_classes = np.argmax(predictions, axis=1)
        else:
            pred_classes = predictions
        
        metrics = {}
        
        if SKLEARN_AVAILABLE:
            metrics["accuracy"] = float(accuracy_score(targets, pred_classes))
            metrics["precision"] = float(precision_score(targets, pred_classes, average=average, zero_division=0))
            metrics["recall"] = float(recall_score(targets, pred_classes, average=average, zero_division=0))
            metrics["f1"] = float(f1_score(targets, pred_classes, average=average, zero_division=0))
            
            # AUC solo para clasificación binaria
            if len(np.unique(targets)) == 2 and predictions.ndim > 1:
                try:
                    metrics["auc"] = float(roc_auc_score(targets, predictions[:, 1]))
                except Exception as e:
                    logger.warning(f"Could not calculate AUC: {e}")
        
        # Usar torchmetrics si está disponible
        if TORCHMETRICS_AVAILABLE:
            try:
                pred_tensor = torch.from_numpy(pred_classes)
                target_tensor = torch.from_numpy(targets)
                
                self.accuracy_metric.update(pred_tensor, target_tensor)
                self.precision_metric.update(pred_tensor, target_tensor)
                self.recall_metric.update(pred_tensor, target_tensor)
                self.f1_metric.update(pred_tensor, target_tensor)
                
                metrics["accuracy_tm"] = float(self.accuracy_metric.compute())
                metrics["precision_tm"] = float(self.precision_metric.compute())
                metrics["recall_tm"] = float(self.recall_metric.compute())
                metrics["f1_tm"] = float(self.f1_metric.compute())
            except Exception as e:
                logger.warning(f"TorchMetrics calculation failed: {e}")
        
        return metrics
    
    def calculate_regression_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """
        Calcular métricas de regresión.
        
        Args:
            predictions: Predicciones
            targets: Targets reales
            
        Returns:
            Dict con métricas
        """
        metrics = {}
        
        if SKLEARN_AVAILABLE:
            metrics["mse"] = float(mean_squared_error(targets, predictions))
            metrics["rmse"] = float(np.sqrt(metrics["mse"]))
            metrics["mae"] = float(mean_absolute_error(targets, predictions))
            metrics["r2"] = float(r2_score(targets, predictions))
        
        # MAPE (Mean Absolute Percentage Error)
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.mean(np.abs((targets - predictions) / targets)) * 100
            metrics["mape"] = float(mape) if not np.isnan(mape) else float('inf')
        
        # R² ajustado
        n = len(targets)
        p = predictions.shape[1] if predictions.ndim > 1 else 1
        if n > p + 1 and "r2" in metrics:
            metrics["r2_adjusted"] = float(
                1 - (1 - metrics["r2"]) * (n - 1) / (n - p - 1)
            )
        
        return metrics
    
    def calculate_sequence_metrics(
        self,
        predictions: List[str],
        targets: List[str]
    ) -> Dict[str, float]:
        """
        Calcular métricas para secuencias (texto).
        
        Args:
            predictions: Lista de predicciones
            targets: Lista de targets
            
        Returns:
            Dict con métricas
        """
        metrics = {}
        
        # BLEU score (requiere nltk)
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            smooth = SmoothingFunction()
            
            bleu_scores = []
            for pred, target in zip(predictions, targets):
                pred_tokens = pred.split()
                target_tokens = [target.split()]
                bleu = sentence_bleu(target_tokens, pred_tokens, smoothing_function=smooth.method1)
                bleu_scores.append(bleu)
            
            metrics["bleu"] = float(np.mean(bleu_scores))
        except ImportError:
            logger.warning("NLTK not available for BLEU score")
        except Exception as e:
            logger.warning(f"BLEU calculation failed: {e}")
        
        # ROUGE score (requiere rouge-score)
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            
            rouge_scores = []
            for pred, target in zip(predictions, targets):
                scores = scorer.score(target, pred)
                rouge_scores.append({
                    'rouge1': scores['rouge1'].fmeasure,
                    'rouge2': scores['rouge2'].fmeasure,
                    'rougeL': scores['rougeL'].fmeasure
                })
            
            metrics["rouge1"] = float(np.mean([s['rouge1'] for s in rouge_scores]))
            metrics["rouge2"] = float(np.mean([s['rouge2'] for s in rouge_scores]))
            metrics["rougeL"] = float(np.mean([s['rougeL'] for s in rouge_scores]))
        except ImportError:
            logger.warning("rouge-score not available")
        except Exception as e:
            logger.warning(f"ROUGE calculation failed: {e}")
        
        return metrics
    
    def calculate(
        self,
        predictions: Union[np.ndarray, List],
        targets: Union[np.ndarray, List]
    ) -> Dict[str, float]:
        """
        Calcular métricas según el tipo de tarea.
        
        Args:
            predictions: Predicciones
            targets: Targets
            
        Returns:
            Dict con métricas
        """
        if self.task == "classification":
            return self.calculate_classification_metrics(
                np.array(predictions),
                np.array(targets)
            )
        elif self.task == "regression":
            return self.calculate_regression_metrics(
                np.array(predictions),
                np.array(targets)
            )
        elif self.task == "sequence":
            return self.calculate_sequence_metrics(
                predictions if isinstance(predictions, list) else predictions.tolist(),
                targets if isinstance(targets, list) else targets.tolist()
            )
        else:
            raise ValueError(f"Unknown task: {self.task}")
    
    def reset(self):
        """Resetear métricas acumuladas."""
        if TORCHMETRICS_AVAILABLE:
            self.accuracy_metric.reset()
            self.precision_metric.reset()
            self.recall_metric.reset()
            self.f1_metric.reset()


class ConfusionMatrix:
    """Matriz de confusión para análisis detallado."""
    
    @staticmethod
    def compute(
        predictions: np.ndarray,
        targets: np.ndarray,
        num_classes: Optional[int] = None
    ) -> np.ndarray:
        """
        Calcular matriz de confusión.
        
        Args:
            predictions: Predicciones
            targets: Targets
            num_classes: Número de clases (None para inferir)
            
        Returns:
            Matriz de confusión
        """
        if predictions.ndim > 1:
            predictions = np.argmax(predictions, axis=1)
        
        if num_classes is None:
            num_classes = max(len(np.unique(predictions)), len(np.unique(targets)))
        
        cm = np.zeros((num_classes, num_classes), dtype=np.int32)
        
        for pred, target in zip(predictions, targets):
            cm[int(target), int(pred)] += 1
        
        return cm
    
    @staticmethod
    def plot(cm: np.ndarray, class_names: Optional[List[str]] = None):
        """
        Visualizar matriz de confusión.
        
        Args:
            cm: Matriz de confusión
            class_names: Nombres de clases
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=class_names or range(len(cm)),
                yticklabels=class_names or range(len(cm))
            )
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            return plt.gcf()
        except ImportError:
            logger.warning("matplotlib/seaborn not available for plotting")
            return None
