"""
Model Evaluation Service - Evaluación de modelos
=================================================

Sistema avanzado para evaluar modelos de deep learning.
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")

try:
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix,
        classification_report,
        mean_squared_error,
        mean_absolute_error,
        r2_score,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available")


@dataclass
class EvaluationMetrics:
    """Métricas de evaluación"""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    roc_auc: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelEvaluationService:
    """Servicio de evaluación de modelos"""
    
    def __init__(self):
        """Inicializar servicio"""
        logger.info("ModelEvaluationService initialized")
    
    def evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        average: str = "weighted"
    ) -> EvaluationMetrics:
        """Evaluar modelo de clasificación"""
        if not SKLEARN_AVAILABLE:
            # Fallback metrics
            accuracy = np.mean(y_true == y_pred)
            return EvaluationMetrics(accuracy=float(accuracy))
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average=average, zero_division=0)
        recall = recall_score(y_true, y_pred, average=average, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
        
        roc_auc = None
        if y_proba is not None:
            try:
                # Handle binary and multiclass
                if len(np.unique(y_true)) == 2:
                    roc_auc = roc_auc_score(y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
                else:
                    roc_auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average=average)
            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC: {e}")
        
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, zero_division=0)
        
        return EvaluationMetrics(
            accuracy=float(accuracy),
            precision=float(precision),
            recall=float(recall),
            f1_score=float(f1),
            roc_auc=float(roc_auc) if roc_auc is not None else None,
            confusion_matrix=cm,
            classification_report=report,
        )
    
    def evaluate_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> EvaluationMetrics:
        """Evaluar modelo de regresión"""
        if not SKLEARN_AVAILABLE:
            # Fallback metrics
            mse = np.mean((y_true - y_pred) ** 2)
            mae = np.mean(np.abs(y_true - y_pred))
            return EvaluationMetrics(mse=float(mse), mae=float(mae))
        
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return EvaluationMetrics(
            mse=float(mse),
            mae=float(mae),
            r2_score=float(r2),
        )
    
    async def evaluate_model(
        self,
        model: nn.Module,
        dataloader: Any,
        device: Optional[torch.device] = None,
        task_type: str = "classification",
        use_mixed_precision: bool = True
    ) -> EvaluationMetrics:
        """
        Evaluar modelo completo con manejo robusto.
        
        Args:
            model: Modelo a evaluar
            dataloader: DataLoader con datos
            device: Dispositivo (None = auto)
            task_type: Tipo de tarea (classification, regression)
            use_mixed_precision: Usar mixed precision
        
        Returns:
            EvaluationMetrics con resultados
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, returning empty metrics")
            return EvaluationMetrics()
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model.eval()
        all_preds = []
        all_labels = []
        all_probas = []
        
        autocast_context = (
            torch.cuda.amp.autocast() if use_mixed_precision and device.type == "cuda"
            else torch.cuda.amp.autocast(enabled=False)
        )
        
        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    try:
                        # Handle different batch formats
                        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                            inputs = batch[0].to(device, non_blocking=True)
                            targets = batch[1].to(device, non_blocking=True)
                        elif isinstance(batch, dict):
                            inputs = batch.get("input_ids", batch.get("inputs", None))
                            targets = batch.get("labels", batch.get("targets", None))
                            if inputs is not None:
                                inputs = inputs.to(device, non_blocking=True)
                            if targets is not None:
                                targets = targets.to(device, non_blocking=True)
                        else:
                            inputs = batch.to(device, non_blocking=True)
                            targets = None
                        
                        if inputs is None:
                            logger.warning(f"Skipping batch {batch_idx}: inputs is None")
                            continue
                        
                        # Forward pass
                        with autocast_context:
                            outputs = model(inputs)
                        
                        # Handle different output formats
                        if isinstance(outputs, dict):
                            logits = outputs.get("logits", outputs.get("output", None))
                            if logits is None:
                                logits = list(outputs.values())[0]
                        else:
                            logits = outputs
                        
                        # Get predictions
                        if task_type == "classification":
                            if logits.dim() > 1:
                                probs = F.softmax(logits, dim=1)
                                preds = torch.argmax(logits, dim=1)
                                all_probas.append(probs.cpu().numpy())
                            else:
                                preds = (logits > 0.5).long()
                                all_probas.append(torch.sigmoid(logits).cpu().numpy())
                            
                            all_preds.append(preds.cpu().numpy())
                        else:  # regression
                            all_preds.append(logits.cpu().numpy())
                        
                        # Store labels
                        if targets is not None:
                            all_labels.append(targets.cpu().numpy())
                    
                    except Exception as e:
                        logger.error(f"Error processing batch {batch_idx}: {e}", exc_info=True)
                        continue
            
            # Concatenate all predictions and labels
            if not all_preds:
                logger.warning("No predictions collected")
                return EvaluationMetrics()
            
            y_pred = np.concatenate(all_preds, axis=0)
            y_proba = np.concatenate(all_probas, axis=0) if all_probas else None
            
            if all_labels:
                y_true = np.concatenate(all_labels, axis=0)
            else:
                logger.warning("No labels found, cannot compute metrics")
                return EvaluationMetrics()
            
            # Compute metrics
            if task_type == "classification":
                return self.evaluate_classification(y_true, y_pred, y_proba)
            else:
                return self.evaluate_regression(y_true, y_pred)
        
        except Exception as e:
            logger.error(f"Error in model evaluation: {e}", exc_info=True)
            return EvaluationMetrics(metadata={"error": str(e)})
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    inputs, labels = batch[0].to(device), batch[1].to(device)
                else:
                    inputs = batch.to(device)
                    labels = None
                
                outputs = model(inputs)
                
                if task_type == "classification":
                    if outputs.dim() > 1:
                        probas = torch.softmax(outputs, dim=1)
                        preds = torch.argmax(outputs, dim=1)
                    else:
                        probas = torch.sigmoid(outputs)
                        preds = (probas > 0.5).long()
                    
                    all_preds.append(preds.cpu().numpy())
                    all_labels.append(labels.cpu().numpy() if labels is not None else None)
                    all_probas.append(probas.cpu().numpy())
                else:
                    all_preds.append(outputs.cpu().numpy())
                    if labels is not None:
                        all_labels.append(labels.cpu().numpy())
        
        all_preds = np.concatenate(all_preds)
        
        if task_type == "classification":
            all_labels = np.concatenate(all_labels) if all_labels[0] is not None else None
            all_probas = np.concatenate(all_probas) if all_probas else None
            
            if all_labels is not None:
                return self.evaluate_classification(all_labels, all_preds, all_probas)
        else:
            all_labels = np.concatenate(all_labels) if all_labels else None
            if all_labels is not None:
                return self.evaluate_regression(all_labels, all_preds)
        
        return EvaluationMetrics()
    
    def calculate_confidence_intervals(
        self,
        metrics: List[float],
        confidence: float = 0.95
    ) -> Dict[str, float]:
        """Calcular intervalos de confianza"""
        if len(metrics) == 0:
            return {}
        
        mean = np.mean(metrics)
        std = np.std(metrics)
        n = len(metrics)
        
        # Z-score for confidence level
        z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
        z = z_scores.get(confidence, 1.96)
        
        margin = z * (std / np.sqrt(n))
        
        return {
            "mean": float(mean),
            "std": float(std),
            "lower": float(mean - margin),
            "upper": float(mean + margin),
            "confidence": confidence,
        }

