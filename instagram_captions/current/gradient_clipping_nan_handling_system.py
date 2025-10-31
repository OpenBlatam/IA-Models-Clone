import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_auc_score, average_precision_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import warnings
warnings.filterwarnings('ignore')


@dataclass
class GradientConfig:
    """Configuration for gradient clipping and NaN/Inf handling."""
    
    # Gradient clipping
    gradient_clip_val: float = 1.0
    gradient_clip_norm: float = 1.0
    gradient_clip_algorithm: str = "norm"  # "norm", "value"
    
    # NaN/Inf handling
    check_gradients: bool = True
    check_weights: bool = True
    check_loss: bool = True
    nan_threshold: float = 1e-6
    inf_threshold: float = 1e6
    
    # Recovery strategies
    zero_nan_gradients: bool = True
    skip_nan_batches: bool = False
    restart_training_on_nan: bool = False
    max_nan_restarts: int = 3
    
    # Monitoring
    log_gradient_stats: bool = True
    log_weight_stats: bool = True
    gradient_history_size: int = 100


@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics."""
    
    # Classification metrics
    classification_metrics: List[str] = None
    
    # Regression metrics
    regression_metrics: List[str] = None
    
    # Text generation metrics
    generation_metrics: List[str] = None
    
    # Custom metrics
    custom_metrics: Dict[str, Callable] = None
    
    # Thresholds
    classification_threshold: float = 0.5
    confidence_threshold: float = 0.8
    
    def __post_init__(self):
        if self.classification_metrics is None:
            self.classification_metrics = [
                "accuracy", "precision", "recall", "f1", 
                "confusion_matrix", "roc_auc", "pr_auc"
            ]
        
        if self.regression_metrics is None:
            self.regression_metrics = [
                "mse", "mae", "rmse", "r2", "mape"
            ]
        
        if self.generation_metrics is None:
            self.generation_metrics = [
                "perplexity", "bleu", "rouge", "meteor"
            ]


class GradientMonitor:
    """Monitor and handle gradient issues."""
    
    def __init__(self, config: GradientConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.gradient_history = []
        self.weight_history = []
        self.nan_count = 0
        self.inf_count = 0
        self.restart_count = 0
    
    def check_gradients(self, model: nn.Module) -> Dict[str, Any]:
        """Check for NaN/Inf gradients and return statistics."""
        if not self.config.check_gradients:
            return {}
        
        gradient_stats = {
            "has_nan": False,
            "has_inf": False,
            "grad_norm": 0.0,
            "grad_mean": 0.0,
            "grad_std": 0.0,
            "nan_layers": [],
            "inf_layers": []
        }
        
        total_norm = 0.0
        all_gradients = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                
                # Check for NaN
                if torch.isnan(grad).any():
                    gradient_stats["has_nan"] = True
                    gradient_stats["nan_layers"].append(name)
                    self.nan_count += 1
                    
                    if self.config.zero_nan_gradients:
                        param.grad.data.zero_()
                        self.logger.warning(f"NaN gradient detected in {name}, zeroed")
                
                # Check for Inf
                if torch.isinf(grad).any():
                    gradient_stats["has_inf"] = True
                    gradient_stats["inf_layers"].append(name)
                    self.inf_count += 1
                    
                    if self.config.zero_nan_gradients:
                        param.grad.data.zero_()
                        self.logger.warning(f"Inf gradient detected in {name}, zeroed")
                
                # Collect gradient statistics
                if not torch.isnan(grad).any() and not torch.isinf(grad).any():
                    param_norm = grad.norm(2)
                    total_norm += param_norm.item() ** 2
                    all_gradients.extend(grad.cpu().numpy().flatten())
        
        gradient_stats["grad_norm"] = total_norm ** 0.5
        if all_gradients:
            gradient_stats["grad_mean"] = np.mean(all_gradients)
            gradient_stats["grad_std"] = np.std(all_gradients)
        
        # Store in history
        if self.config.log_gradient_stats:
            self.gradient_history.append(gradient_stats)
            if len(self.gradient_history) > self.config.gradient_history_size:
                self.gradient_history.pop(0)
        
        return gradient_stats
    
    def check_weights(self, model: nn.Module) -> Dict[str, Any]:
        """Check for NaN/Inf weights and return statistics."""
        if not self.config.check_weights:
            return {}
        
        weight_stats = {
            "has_nan": False,
            "has_inf": False,
            "weight_norm": 0.0,
            "weight_mean": 0.0,
            "weight_std": 0.0,
            "nan_layers": [],
            "inf_layers": []
        }
        
        all_weights = []
        
        for name, param in model.named_parameters():
            weight = param.data
            
            # Check for NaN
            if torch.isnan(weight).any():
                weight_stats["has_nan"] = True
                weight_stats["nan_layers"].append(name)
                self.logger.error(f"NaN weight detected in {name}")
            
            # Check for Inf
            if torch.isinf(weight).any():
                weight_stats["has_inf"] = True
                weight_stats["inf_layers"].append(name)
                self.logger.error(f"Inf weight detected in {name}")
            
            # Collect weight statistics
            if not torch.isnan(weight).any() and not torch.isinf(weight).any():
                all_weights.extend(weight.cpu().numpy().flatten())
        
        if all_weights:
            weight_stats["weight_norm"] = np.linalg.norm(all_weights)
            weight_stats["weight_mean"] = np.mean(all_weights)
            weight_stats["weight_std"] = np.std(all_weights)
        
        # Store in history
        if self.config.log_weight_stats:
            self.weight_history.append(weight_stats)
            if len(self.weight_history) > self.config.gradient_history_size:
                self.weight_history.pop(0)
        
        return weight_stats
    
    def check_loss(self, loss: torch.Tensor) -> bool:
        """Check if loss is NaN or Inf."""
        if not self.config.check_loss:
            return True
        
        loss_value = loss.item()
        
        if torch.isnan(loss) or torch.isinf(loss):
            self.logger.error(f"Invalid loss detected: {loss_value}")
            return False
        
        if abs(loss_value) > self.config.inf_threshold:
            self.logger.warning(f"Loss value too large: {loss_value}")
            return False
        
        return True
    
    def should_restart_training(self) -> bool:
        """Determine if training should be restarted."""
        if not self.config.restart_training_on_nan:
            return False
        
        if self.restart_count >= self.config.max_nan_restarts:
            self.logger.error("Maximum restart attempts reached")
            return False
        
        return self.nan_count > 0 or self.inf_count > 0


class GradientClipper:
    """Advanced gradient clipping implementation."""
    
    def __init__(self, config: GradientConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def clip_gradients(self, model: nn.Module, optimizer: optim.Optimizer) -> float:
        """Clip gradients using specified algorithm."""
        if self.config.gradient_clip_algorithm == "norm":
            return self._clip_by_norm(model)
        elif self.config.gradient_clip_algorithm == "value":
            return self._clip_by_value(model)
        else:
            raise ValueError(f"Unknown gradient clipping algorithm: {self.config.gradient_clip_algorithm}")
    
    def _clip_by_norm(self, model: nn.Module) -> float:
        """Clip gradients by norm."""
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            self.config.gradient_clip_norm
        )
        return total_norm.item()
    
    def _clip_by_value(self, model: nn.Module) -> float:
        """Clip gradients by value."""
        torch.nn.utils.clip_grad_value_(
            model.parameters(), 
            self.config.gradient_clip_val
        )
        return 0.0  # Value clipping doesn't return norm


class EvaluationMetrics:
    """Comprehensive evaluation metrics for different tasks."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def evaluate_classification(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Evaluate classification metrics."""
        metrics = {}
        
        # Basic metrics
        if "accuracy" in self.config.classification_metrics:
            metrics["accuracy"] = accuracy_score(y_true, y_pred)
        
        if "precision" in self.config.classification_metrics or "recall" in self.config.classification_metrics or "f1" in self.config.classification_metrics:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='weighted', zero_division=0
            )
            metrics.update({
                "precision": precision,
                "recall": recall,
                "f1": f1
            })
        
        # Confusion matrix
        if "confusion_matrix" in self.config.classification_metrics:
            cm = confusion_matrix(y_true, y_pred)
            metrics["confusion_matrix"] = cm.tolist()
        
        # ROC AUC
        if "roc_auc" in self.config.classification_metrics and y_proba is not None:
            try:
                if y_proba.ndim == 1:
                    metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
                else:
                    metrics["roc_auc"] = roc_auc_score(y_true, y_proba, multi_class='ovr')
            except Exception as e:
                self.logger.warning(f"Could not compute ROC AUC: {e}")
        
        # PR AUC
        if "pr_auc" in self.config.classification_metrics and y_proba is not None:
            try:
                if y_proba.ndim == 1:
                    metrics["pr_auc"] = average_precision_score(y_true, y_proba)
                else:
                    metrics["pr_auc"] = average_precision_score(y_true, y_proba, average='weighted')
            except Exception as e:
                self.logger.warning(f"Could not compute PR AUC: {e}")
        
        return metrics
    
    def evaluate_regression(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate regression metrics."""
        metrics = {}
        
        if "mse" in self.config.regression_metrics:
            metrics["mse"] = mean_squared_error(y_true, y_pred)
        
        if "mae" in self.config.regression_metrics:
            metrics["mae"] = mean_absolute_error(y_true, y_pred)
        
        if "rmse" in self.config.regression_metrics:
            metrics["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
        
        if "r2" in self.config.regression_metrics:
            metrics["r2"] = r2_score(y_true, y_pred)
        
        if "mape" in self.config.regression_metrics:
            mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100
            metrics["mape"] = mape
        
        return metrics
    
    def evaluate_text_generation(
        self, 
        references: List[List[str]], 
        predictions: List[str]
    ) -> Dict[str, float]:
        """Evaluate text generation metrics."""
        metrics = {}
        
        if "perplexity" in self.config.generation_metrics:
            # This would require a language model to compute
            metrics["perplexity"] = self._compute_perplexity(predictions)
        
        if "bleu" in self.config.generation_metrics:
            metrics["bleu"] = self._compute_bleu(references, predictions)
        
        if "rouge" in self.config.generation_metrics:
            metrics["rouge"] = self._compute_rouge(references, predictions)
        
        if "meteor" in self.config.generation_metrics:
            metrics["meteor"] = self._compute_meteor(references, predictions)
        
        return metrics
    
    def _compute_perplexity(self, texts: List[str]) -> float:
        """Compute perplexity (placeholder implementation)."""
        # This would require a language model
        return 0.0
    
    def _compute_bleu(self, references: List[List[str]], predictions: List[str]) -> float:
        """Compute BLEU score (placeholder implementation)."""
        # This would require nltk or similar
        return 0.0
    
    def _compute_rouge(self, references: List[List[str]], predictions: List[str]) -> float:
        """Compute ROUGE score (placeholder implementation)."""
        # This would require rouge-score library
        return 0.0
    
    def _compute_meteor(self, references: List[List[str]], predictions: List[str]) -> float:
        """Compute METEOR score (placeholder implementation)."""
        # This would require nltk
        return 0.0


class RobustTrainingSystem:
    """Training system with robust gradient handling and evaluation."""
    
    def __init__(
        self, 
        model: nn.Module,
        gradient_config: GradientConfig,
        evaluation_config: EvaluationConfig
    ):
        self.model = model
        self.gradient_config = gradient_config
        self.evaluation_config = evaluation_config
        
        self.gradient_monitor = GradientMonitor(gradient_config)
        self.gradient_clipper = GradientClipper(gradient_config)
        self.evaluator = EvaluationMetrics(evaluation_config)
        
        self.logger = logging.getLogger(__name__)
        self.device = next(model.parameters()).device
        
        # Training state
        self.optimizer = None
        self.scheduler = None
        self.scaler = GradScaler() if torch.cuda.is_available() else None
        
        # Statistics
        self.training_stats = {
            "epoch": 0,
            "step": 0,
            "total_loss": 0.0,
            "gradient_norms": [],
            "nan_counts": [],
            "inf_counts": []
        }
    
    def setup_training(
        self, 
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        scheduler_type: str = "cosine"
    ):
        """Setup optimizer and scheduler."""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        if scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=1000
            )
        elif scheduler_type == "linear":
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=1000
            )
    
    def train_step(
        self, 
        batch: Dict[str, torch.Tensor],
        task_type: str = "classification"
    ) -> Dict[str, float]:
        """Single training step with robust gradient handling."""
        self.model.train()
        self.optimizer.zero_grad()
        
        try:
            # Forward pass
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(**batch)
                    loss = outputs.loss
            else:
                outputs = self.model(**batch)
                loss = outputs.loss
            
            # Check loss validity
            if not self.gradient_monitor.check_loss(loss):
                if self.gradient_config.skip_nan_batches:
                    return {"loss": float('inf'), "skipped": True}
                else:
                    raise ValueError("Invalid loss detected")
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Check gradients
            grad_stats = self.gradient_monitor.check_gradients(self.model)
            
            # Check weights
            weight_stats = self.gradient_monitor.check_weights(self.model)
            
            # Clip gradients
            grad_norm = self.gradient_clipper.clip_gradients(self.model, self.optimizer)
            
            # Optimizer step
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Update statistics
            self.training_stats["step"] += 1
            self.training_stats["total_loss"] += loss.item()
            self.training_stats["gradient_norms"].append(grad_norm)
            self.training_stats["nan_counts"].append(grad_stats.get("has_nan", False))
            self.training_stats["inf_counts"].append(grad_stats.get("has_inf", False))
            
            return {
                "loss": loss.item(),
                "grad_norm": grad_norm,
                "has_nan": grad_stats.get("has_nan", False),
                "has_inf": grad_stats.get("has_inf", False),
                "lr": self.optimizer.param_groups[0]["lr"]
            }
            
        except Exception as e:
            self.logger.error(f"Error in training step: {e}")
            return {"loss": float('inf'), "error": str(e)}
    
    def evaluate(
        self, 
        dataloader: DataLoader,
        task_type: str = "classification"
    ) -> Dict[str, float]:
        """Evaluate model with appropriate metrics."""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs.loss
                total_loss += loss.item()
                
                if task_type == "classification":
                    logits = outputs.logits
                    probabilities = torch.softmax(logits, dim=-1)
                    predictions = torch.argmax(logits, dim=-1)
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    all_targets.extend(batch["labels"].cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())
                
                elif task_type == "regression":
                    predictions = outputs.logits.squeeze()
                    all_predictions.extend(predictions.cpu().numpy())
                    all_targets.extend(batch["labels"].cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities) if all_probabilities else None
        
        # Compute metrics
        metrics = {"loss": total_loss / len(dataloader)}
        
        if task_type == "classification":
            task_metrics = self.evaluator.evaluate_classification(
                all_targets, all_predictions, all_probabilities
            )
            metrics.update(task_metrics)
        
        elif task_type == "regression":
            task_metrics = self.evaluator.evaluate_regression(
                all_targets, all_predictions
            )
            metrics.update(task_metrics)
        
        return metrics
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training statistics summary."""
        if not self.training_stats["gradient_norms"]:
            return self.training_stats
        
        return {
            **self.training_stats,
            "avg_grad_norm": np.mean(self.training_stats["gradient_norms"]),
            "max_grad_norm": np.max(self.training_stats["gradient_norms"]),
            "total_nan_count": sum(self.training_stats["nan_counts"]),
            "total_inf_count": sum(self.training_stats["inf_counts"]),
            "nan_rate": sum(self.training_stats["nan_counts"]) / len(self.training_stats["nan_counts"]),
            "inf_rate": sum(self.training_stats["inf_counts"]) / len(self.training_stats["inf_counts"])
        }


# Example usage
def create_robust_training_example():
    """Example of using the robust training system."""
    
    # Configuration
    gradient_config = GradientConfig(
        gradient_clip_val=1.0,
        gradient_clip_norm=1.0,
        gradient_clip_algorithm="norm",
        check_gradients=True,
        check_weights=True,
        check_loss=True,
        zero_nan_gradients=True,
        skip_nan_batches=False,
        restart_training_on_nan=False
    )
    
    evaluation_config = EvaluationConfig(
        classification_metrics=["accuracy", "precision", "recall", "f1", "roc_auc"],
        regression_metrics=["mse", "mae", "rmse", "r2"],
        generation_metrics=["perplexity", "bleu"]
    )
    
    # Model setup
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    
    # Training system
    training_system = RobustTrainingSystem(model, gradient_config, evaluation_config)
    training_system.setup_training(learning_rate=2e-5, weight_decay=0.01)
    
    return training_system


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create example
    training_system = create_robust_training_example()
    print("Robust training system created successfully!")
    print(f"Gradient monitoring: {training_system.gradient_monitor.config}")
    print(f"Evaluation metrics: {training_system.evaluator.config}")




