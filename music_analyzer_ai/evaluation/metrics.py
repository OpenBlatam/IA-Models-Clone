"""
Advanced Evaluation Metrics
Comprehensive metrics for model evaluation
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from sklearn.metrics import (
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        classification_report,
        r2_score
    )
    from sklearn.model_selection import KFold
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available, some metrics will not work")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ClassificationMetrics:
    """
    Metrics for classification tasks
    """
    
    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy"""
        return float(np.mean(y_true == y_pred))
    
    @staticmethod
    def precision(y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> float:
        """Calculate precision"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for precision calculation")
        return float(precision_score(y_true, y_pred, average=average, zero_division=0))
    
    @staticmethod
    def recall(y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> float:
        """Calculate recall"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for recall calculation")
        return float(recall_score(y_true, y_pred, average=average, zero_division=0))
    
    @staticmethod
    def f1_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> float:
        """Calculate F1 score"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for F1 score calculation")
        return float(f1_score(y_true, y_pred, average=average, zero_division=0))
    
    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculate confusion matrix"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for confusion matrix")
        return confusion_matrix(y_true, y_pred)
    
    @staticmethod
    def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Generate classification report"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for classification report")
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        return report
    
    @staticmethod
    def top_k_accuracy(
        y_true: np.ndarray,
        y_pred_probs: np.ndarray,
        k: int = 3
    ) -> float:
        """Calculate top-k accuracy"""
        top_k_preds = np.argsort(y_pred_probs, axis=-1)[:, -k:]
        return float(np.mean([y_true[i] in top_k_preds[i] for i in range(len(y_true))]))


class RegressionMetrics:
    """
    Metrics for regression tasks
    """
    
    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Squared Error"""
        return float(np.mean((y_true - y_pred) ** 2))
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error"""
        return float(np.mean(np.abs(y_true - y_pred)))
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error"""
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R² score"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for R² score")
        return float(r2_score(y_true, y_pred))
    
    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error"""
        mask = y_true != 0
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


class ModelEvaluator:
    """
    Comprehensive model evaluator
    """
    
    def __init__(self):
        self.classification_metrics = ClassificationMetrics()
        self.regression_metrics = RegressionMetrics()
    
    def evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_probs: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Evaluate classification model"""
        metrics = {
            "accuracy": self.classification_metrics.accuracy(y_true, y_pred),
            "precision": self.classification_metrics.precision(y_true, y_pred),
            "recall": self.classification_metrics.recall(y_true, y_pred),
            "f1_score": self.classification_metrics.f1_score(y_true, y_pred),
            "confusion_matrix": self.classification_metrics.confusion_matrix(y_true, y_pred).tolist()
        }
        
        if y_pred_probs is not None:
            metrics["top_3_accuracy"] = self.classification_metrics.top_k_accuracy(
                y_true, y_pred_probs, k=3
            )
            metrics["top_5_accuracy"] = self.classification_metrics.top_k_accuracy(
                y_true, y_pred_probs, k=5
            )
        
        # Per-class metrics
        report = self.classification_metrics.classification_report(y_true, y_pred)
        metrics["per_class"] = report
        
        return metrics
    
    def evaluate_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Evaluate regression model"""
        return {
            "mse": self.regression_metrics.mse(y_true, y_pred),
            "mae": self.regression_metrics.mae(y_true, y_pred),
            "rmse": self.regression_metrics.rmse(y_true, y_pred),
            "r2_score": self.regression_metrics.r2_score(y_true, y_pred),
            "mape": self.regression_metrics.mape(y_true, y_pred)
        }
    
    def evaluate_multi_task(
        self,
        predictions: Dict[str, Any],
        ground_truth: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate multi-task model"""
        results = {}
        
        # Classification tasks
        for task in ["genre", "mood", "instruments"]:
            if task in predictions and task in ground_truth:
                y_true = ground_truth[task]
                y_pred = predictions[task]
                
                if isinstance(y_pred, (list, np.ndarray)) and len(y_pred.shape) > 1:
                    # Probabilities
                    y_pred_classes = np.argmax(y_pred, axis=-1)
                    results[task] = self.evaluate_classification(
                        y_true, y_pred_classes, y_pred
                    )
                else:
                    # Classes
                    results[task] = self.evaluate_classification(y_true, y_pred)
        
        # Regression tasks
        for task in ["energy", "complexity"]:
            if task in predictions and task in ground_truth:
                results[task] = self.evaluate_regression(
                    ground_truth[task],
                    predictions[task]
                )
        
        return results


class CrossValidator:
    """
    Cross-validation utilities
    """
    
    @staticmethod
    def k_fold_cv(
        model_factory: callable,
        X: np.ndarray,
        y: np.ndarray,
        k: int = 5,
        task_type: str = "classification"
    ) -> Dict[str, Any]:
        """K-fold cross-validation"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for cross-validation")
        
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            model = model_factory()
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_val)
            
            if task_type == "classification":
                from sklearn.metrics import accuracy_score
                score = accuracy_score(y_val, y_pred)
            else:
                score = r2_score(y_val, y_pred)
            
            scores.append(score)
            logger.info(f"Fold {fold + 1}/{k}: {score:.4f}")
        
        return {
            "mean": np.mean(scores),
            "std": np.std(scores),
            "scores": scores
        }

