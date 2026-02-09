"""
Evaluador avanzado de modelos con métricas completas

Mejoras:
- Métricas extensas para diferentes tareas
- Cross-validation integrado
- Model comparison
- Statistical significance testing
- Confidence intervals
"""

import logging
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from scipy import stats
import json
from pathlib import Path

from ..core.base_service import BaseMLService
from ..core.exceptions import ValidationError

logger = logging.getLogger(__name__)


class AdvancedModelEvaluator(BaseMLService):
    """
    Evaluador avanzado de modelos
    
    Features:
    - Métricas completas para clasificación y regresión
    - Cross-validation
    - Statistical testing
    - Confidence intervals
    - Model comparison
    """
    
    def __init__(self):
        super().__init__()
    
    def _load_model(self) -> None:
        """No requiere modelo específico"""
        pass
    
    def evaluate_classification(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        class_names: Optional[List[str]] = None,
        calculate_confidence_intervals: bool = True
    ) -> Dict[str, Any]:
        """
        Evalúa modelo de clasificación con métricas completas
        
        Args:
            predictions: Predicciones del modelo
            targets: Targets reales
            class_names: Nombres de clases
            calculate_confidence_intervals: Si calcular intervalos de confianza
            
        Returns:
            Diccionario con métricas completas
        """
        # Métricas básicas
        accuracy = accuracy_score(targets, predictions)
        precision = precision_score(targets, predictions, average='weighted', zero_division=0)
        recall = recall_score(targets, predictions, average='weighted', zero_division=0)
        f1 = f1_score(targets, predictions, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        
        # Classification report
        report = classification_report(
            targets,
            predictions,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        
        # Confidence intervals
        ci = {}
        if calculate_confidence_intervals:
            ci = self._calculate_confidence_intervals(
                accuracy,
                len(targets),
                confidence_level=0.95
            )
        
        # Per-class metrics
        per_class_metrics = {}
        if class_names:
            for i, class_name in enumerate(class_names):
                if i < len(cm):
                    tp = cm[i, i]
                    fp = cm[:, i].sum() - tp
                    fn = cm[i, :].sum() - tp
                    tn = cm.sum() - tp - fp - fn
                    
                    per_class_metrics[class_name] = {
                        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
                        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
                        "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
                        "support": int(tp + fn)
                    }
        
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "per_class_metrics": per_class_metrics,
            "confidence_intervals": ci,
            "num_samples": len(targets),
            "num_classes": len(np.unique(targets))
        }
    
    def evaluate_regression(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        calculate_confidence_intervals: bool = True
    ) -> Dict[str, Any]:
        """
        Evalúa modelo de regresión con métricas completas
        
        Args:
            predictions: Predicciones del modelo
            targets: Targets reales
            calculate_confidence_intervals: Si calcular intervalos de confianza
            
        Returns:
            Diccionario con métricas completas
        """
        # Métricas básicas
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
        # Métricas adicionales
        mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
        median_ae = np.median(np.abs(targets - predictions))
        
        # Residuals
        residuals = targets - predictions
        
        # Confidence intervals
        ci = {}
        if calculate_confidence_intervals:
            ci = {
                "rmse": self._calculate_confidence_intervals(
                    rmse,
                    len(targets),
                    confidence_level=0.95
                ),
                "mae": self._calculate_confidence_intervals(
                    mae,
                    len(targets),
                    confidence_level=0.95
                )
            }
        
        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2_score": float(r2),
            "mape": float(mape),
            "median_ae": float(median_ae),
            "residuals_mean": float(np.mean(residuals)),
            "residuals_std": float(np.std(residuals)),
            "confidence_intervals": ci,
            "num_samples": len(targets)
        }
    
    def compare_models(
        self,
        model_results: List[Dict[str, Any]],
        metric_name: str = "accuracy",
        statistical_test: bool = True
    ) -> Dict[str, Any]:
        """
        Compara múltiples modelos
        
        Args:
            model_results: Lista de resultados de modelos
            metric_name: Nombre de métrica a comparar
            statistical_test: Si realizar test estadístico
            
        Returns:
            Comparación de modelos
        """
        if not model_results:
            return {}
        
        # Extraer métricas
        model_names = [r.get("model_name", f"Model_{i}") for i, r in enumerate(model_results)]
        metrics = [r.get(metric_name, 0) for r in model_results]
        
        # Estadísticas
        best_idx = np.argmax(metrics)
        best_model = model_names[best_idx]
        best_metric = metrics[best_idx]
        
        # Statistical test
        statistical_comparison = {}
        if statistical_test and len(model_results) > 1:
            # Paired t-test entre mejor y segundo mejor
            sorted_indices = np.argsort(metrics)[::-1]
            if len(sorted_indices) >= 2:
                best_metrics = model_results[sorted_indices[0]].get("detailed_metrics", {})
                second_metrics = model_results[sorted_indices[1]].get("detailed_metrics", {})
                
                # Si hay datos de validación cruzada
                if "cv_scores" in best_metrics and "cv_scores" in second_metrics:
                    t_stat, p_value = stats.ttest_rel(
                        best_metrics["cv_scores"],
                        second_metrics["cv_scores"]
                    )
                    statistical_comparison = {
                        "t_statistic": float(t_stat),
                        "p_value": float(p_value),
                        "significant": p_value < 0.05
                    }
        
        return {
            "models": model_names,
            "metrics": {name: float(metric) for name, metric in zip(model_names, metrics)},
            "best_model": best_model,
            "best_metric": float(best_metric),
            "metric_name": metric_name,
            "statistical_comparison": statistical_comparison
        }
    
    def _calculate_confidence_intervals(
        self,
        metric: float,
        n_samples: int,
        confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """
        Calcula intervalos de confianza para una métrica
        
        Args:
            metric: Valor de la métrica
            n_samples: Número de muestras
            confidence_level: Nivel de confianza (0.95 = 95%)
            
        Returns:
            Intervalo de confianza
        """
        # Usar aproximación normal para proporciones
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        se = np.sqrt(metric * (1 - metric) / n_samples) if n_samples > 0 else 0
        
        lower = max(0, metric - z_score * se)
        upper = min(1, metric + z_score * se)
        
        return {
            "lower": float(lower),
            "upper": float(upper),
            "confidence_level": confidence_level
        }
    
    def evaluate_with_cross_validation(
        self,
        model: torch.nn.Module,
        dataset: torch.utils.data.Dataset,
        n_splits: int = 5,
        task_type: str = "classification"
    ) -> Dict[str, Any]:
        """
        Evalúa modelo con cross-validation
        
        Args:
            model: Modelo a evaluar
            dataset: Dataset completo
            n_splits: Número de folds
            task_type: Tipo de tarea (classification, regression)
            
        Returns:
            Resultados de cross-validation
        """
        from sklearn.model_selection import KFold, StratifiedKFold
        
        # Seleccionar tipo de CV
        if task_type == "classification":
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Obtener targets
        targets = np.array([item[1] for item in dataset])
        
        cv_scores = []
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(dataset, targets)):
            # Crear dataloaders para este fold
            train_subset = torch.utils.data.Subset(dataset, train_idx)
            val_subset = torch.utils.data.Subset(dataset, val_idx)
            
            # Evaluar (simplificado - en producción entrenarías el modelo)
            # Aquí asumimos que el modelo ya está entrenado
            val_predictions = []
            val_targets = []
            
            model.eval()
            with torch.no_grad():
                for idx in val_idx:
                    item = dataset[idx]
                    # Asumir formato (input, target)
                    if isinstance(item, tuple):
                        input_data, target = item
                        if isinstance(input_data, torch.Tensor):
                            output = model(input_data.unsqueeze(0))
                            if task_type == "classification":
                                pred = output.argmax(dim=1).item()
                            else:
                                pred = output.item()
                            val_predictions.append(pred)
                            val_targets.append(target)
            
            # Calcular métricas para este fold
            if task_type == "classification":
                fold_metrics = self.evaluate_classification(
                    np.array(val_predictions),
                    np.array(val_targets)
                )
                cv_scores.append(fold_metrics["accuracy"])
            else:
                fold_metrics = self.evaluate_regression(
                    np.array(val_predictions),
                    np.array(val_targets)
                )
                cv_scores.append(fold_metrics["r2_score"])
            
            fold_results.append({
                "fold": fold + 1,
                "metrics": fold_metrics
            })
        
        # Estadísticas de CV
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        
        return {
            "cv_scores": [float(score) for score in cv_scores],
            "cv_mean": float(cv_mean),
            "cv_std": float(cv_std),
            "cv_mean_std": f"{cv_mean:.4f} ± {cv_std:.4f}",
            "fold_results": fold_results,
            "n_splits": n_splits
        }




