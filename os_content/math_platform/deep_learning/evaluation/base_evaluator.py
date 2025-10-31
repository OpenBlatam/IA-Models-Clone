from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
        from sklearn.metrics import roc_curve, auc
        from sklearn.metrics import precision_recall_curve, average_precision_score
    import torch.nn as nn
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Base Evaluator Class
Foundation class for all evaluation implementations in the modular deep learning system.
"""

    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    # Evaluation parameters
    batch_size: int = 32
    num_workers: int = 4
    device: str = "auto"
    
    # Metrics parameters
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "precision", "recall", "f1"])
    average: str = "weighted"  # micro, macro, weighted, binary
    zero_division: int = 0
    
    # Output parameters
    save_predictions: bool = True
    save_plots: bool = True
    save_report: bool = True
    output_dir: str = "evaluation_results"
    
    # Visualization parameters
    plot_confusion_matrix: bool = True
    plot_roc_curve: bool = True
    plot_precision_recall: bool = True
    plot_learning_curves: bool = True
    
    # Analysis parameters
    error_analysis: bool = True
    feature_importance: bool = False
    model_interpretation: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'device': self.device,
            'metrics': self.metrics,
            'average': self.average,
            'zero_division': self.zero_division,
            'save_predictions': self.save_predictions,
            'save_plots': self.save_plots,
            'save_report': self.save_report,
            'output_dir': self.output_dir,
            'plot_confusion_matrix': self.plot_confusion_matrix,
            'plot_roc_curve': self.plot_roc_curve,
            'plot_precision_recall': self.plot_precision_recall,
            'plot_learning_curves': self.plot_learning_curves,
            'error_analysis': self.error_analysis,
            'feature_importance': self.feature_importance,
            'model_interpretation': self.model_interpretation
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EvaluationConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def save(self, filepath: str):
        """Save config to file."""
        with open(filepath, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'EvaluationConfig':
        """Load config from file."""
        with open(filepath, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class BaseEvaluator(ABC):
    """Base evaluator class for all evaluation implementations."""
    
    def __init__(self, model: nn.Module, test_dataset: data.Dataset, config: EvaluationConfig = None):
        
    """__init__ function."""
self.model = model
        self.test_dataset = test_dataset
        self.config = config or EvaluationConfig()
        
        # Setup device
        self.device = self._setup_device()
        self.model.to(self.device)
        self.model.eval()
        
        # Setup data loader
        self.test_loader = self._create_dataloader()
        
        # Initialize results storage
        self.predictions = []
        self.targets = []
        self.probabilities = []
        self.evaluation_results = {}
        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized evaluator with {len(test_dataset)} test samples")
    
    def _setup_device(self) -> torch.device:
        """Setup device for evaluation."""
        if self.config.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.config.device)
        
        logger.info(f"Using device: {device}")
        return device
    
    def _create_dataloader(self) -> data.DataLoader:
        """Create data loader for test dataset."""
        return data.DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
    
    def evaluate(self) -> Dict[str, Any]:
        """Main evaluation method."""
        logger.info("Starting evaluation...")
        
        # Generate predictions
        self._generate_predictions()
        
        # Calculate metrics
        self._calculate_metrics()
        
        # Generate visualizations
        if self.config.save_plots:
            self._generate_plots()
        
        # Perform error analysis
        if self.config.error_analysis:
            self._perform_error_analysis()
        
        # Save results
        self._save_results()
        
        logger.info("Evaluation completed!")
        return self.evaluation_results
    
    def _generate_predictions(self) -> Any:
        """Generate predictions for all test samples."""
        self.predictions = []
        self.targets = []
        self.probabilities = []
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(self.test_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(data)
                
                # Get predictions
                if outputs.dim() > 1:
                    probabilities = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs, 1)
                else:
                    probabilities = torch.sigmoid(outputs)
                    predicted = (probabilities > 0.5).long()
                
                # Store results
                self.predictions.extend(predicted.cpu().numpy())
                self.targets.extend(targets.cpu().numpy())
                self.probabilities.extend(probabilities.cpu().numpy())
        
        self.predictions = np.array(self.predictions)
        self.targets = np.array(self.targets)
        self.probabilities = np.array(self.probabilities)
        
        logger.info(f"Generated predictions for {len(self.predictions)} samples")
    
    def _calculate_metrics(self) -> Any:
        """Calculate evaluation metrics."""
        metrics = {}
        
        for metric_name in self.config.metrics:
            metric_value = self._calculate_single_metric(metric_name)
            metrics[metric_name] = metric_value
        
        # Add additional metrics
        metrics['total_samples'] = len(self.targets)
        metrics['correct_predictions'] = np.sum(self.predictions == self.targets)
        
        self.evaluation_results['metrics'] = metrics
        
        # Log metrics
        logger.info("Evaluation Metrics:")
        for metric_name, metric_value in metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")
    
    def _calculate_single_metric(self, metric_name: str) -> float:
        """Calculate a single metric."""
        metric_functions = {
            'accuracy': lambda: accuracy_score(self.targets, self.predictions),
            'precision': lambda: precision_score(self.targets, self.predictions, 
                                               average=self.config.average, 
                                               zero_division=self.config.zero_division),
            'recall': lambda: recall_score(self.targets, self.predictions, 
                                         average=self.config.average, 
                                         zero_division=self.config.zero_division),
            'f1': lambda: f1_score(self.targets, self.predictions, 
                                 average=self.config.average, 
                                 zero_division=self.config.zero_division),
            'mse': lambda: mean_squared_error(self.targets, self.predictions),
            'mae': lambda: mean_absolute_error(self.targets, self.predictions),
            'r2': lambda: r2_score(self.targets, self.predictions)
        }
        
        if metric_name in metric_functions:
            try:
                return metric_functions[metric_name]()
            except Exception as e:
                logger.warning(f"Error calculating {metric_name}: {e}")
                return 0.0
        else:
            logger.warning(f"Unknown metric: {metric_name}")
            return 0.0
    
    def _generate_plots(self) -> Any:
        """Generate evaluation plots."""
        plots_dir = Path(self.config.output_dir) / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Confusion matrix
        if self.config.plot_confusion_matrix:
            self._plot_confusion_matrix(plots_dir)
        
        # ROC curve
        if self.config.plot_roc_curve and self.probabilities.shape[1] > 1:
            self._plot_roc_curve(plots_dir)
        
        # Precision-Recall curve
        if self.config.plot_precision_recall and self.probabilities.shape[1] > 1:
            self._plot_precision_recall_curve(plots_dir)
        
        # Prediction distribution
        self._plot_prediction_distribution(plots_dir)
        
        logger.info(f"Plots saved to {plots_dir}")
    
    def _plot_confusion_matrix(self, plots_dir: Path):
        """Plot confusion matrix."""
        cm = confusion_matrix(self.targets, self.predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=range(cm.shape[1]), 
                   yticklabels=range(cm.shape[0]))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(plots_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curve(self, plots_dir: Path):
        """Plot ROC curve."""
        
        n_classes = self.probabilities.shape[1]
        
        plt.figure(figsize=(10, 8))
        
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve((self.targets == i).astype(int), self.probabilities[:, i])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / "roc_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_precision_recall_curve(self, plots_dir: Path):
        """Plot Precision-Recall curve."""
        
        n_classes = self.probabilities.shape[1]
        
        plt.figure(figsize=(10, 8))
        
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(
                (self.targets == i).astype(int), self.probabilities[:, i]
            )
            ap = average_precision_score((self.targets == i).astype(int), self.probabilities[:, i])
            
            plt.plot(recall, precision, label=f'Class {i} (AP = {ap:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / "precision_recall_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_prediction_distribution(self, plots_dir: Path):
        """Plot prediction distribution."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Target distribution
        unique_targets, target_counts = np.unique(self.targets, return_counts=True)
        axes[0].bar(unique_targets, target_counts, alpha=0.7, label='Actual')
        axes[0].set_title('Target Distribution')
        axes[0].set_xlabel('Class')
        axes[0].set_ylabel('Count')
        axes[0].legend()
        
        # Prediction distribution
        unique_preds, pred_counts = np.unique(self.predictions, return_counts=True)
        axes[1].bar(unique_preds, pred_counts, alpha=0.7, label='Predicted', color='orange')
        axes[1].set_title('Prediction Distribution')
        axes[1].set_xlabel('Class')
        axes[1].set_ylabel('Count')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(plots_dir / "prediction_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _perform_error_analysis(self) -> Any:
        """Perform error analysis."""
        error_analysis = {
            'total_errors': np.sum(self.predictions != self.targets),
            'error_rate': np.mean(self.predictions != self.targets),
            'error_indices': np.where(self.predictions != self.targets)[0],
            'error_predictions': self.predictions[self.predictions != self.targets],
            'error_targets': self.targets[self.predictions != self.targets]
        }
        
        # Class-wise error analysis
        class_errors = {}
        for class_idx in np.unique(self.targets):
            class_mask = self.targets == class_idx
            class_errors[class_idx] = {
                'total_samples': np.sum(class_mask),
                'errors': np.sum((self.predictions != self.targets) & class_mask),
                'error_rate': np.mean((self.predictions != self.targets) & class_mask)
            }
        
        error_analysis['class_errors'] = class_errors
        
        self.evaluation_results['error_analysis'] = error_analysis
        
        logger.info(f"Error Analysis: {error_analysis['total_errors']} errors out of {len(self.targets)} samples")
        logger.info(f"Overall error rate: {error_analysis['error_rate']:.4f}")
    
    def _save_results(self) -> Any:
        """Save evaluation results."""
        # Save predictions
        if self.config.save_predictions:
            predictions_df = pd.DataFrame({
                'target': self.targets,
                'prediction': self.predictions,
                'correct': self.predictions == self.targets
            })
            
            # Add probabilities if available
            if len(self.probabilities.shape) > 1:
                for i in range(self.probabilities.shape[1]):
                    predictions_df[f'prob_class_{i}'] = self.probabilities[:, i]
            
            predictions_path = Path(self.config.output_dir) / "predictions.csv"
            predictions_df.to_csv(predictions_path, index=False)
            logger.info(f"Predictions saved to {predictions_path}")
        
        # Save evaluation report
        if self.config.save_report:
            report_path = Path(self.config.output_dir) / "evaluation_report.json"
            with open(report_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                json.dump(self.evaluation_results, f, indent=2, default=str)
            logger.info(f"Evaluation report saved to {report_path}")
    
    def get_classification_report(self) -> str:
        """Get detailed classification report."""
        if len(np.unique(self.targets)) > 1:
            return classification_report(
                self.targets, self.predictions, 
                zero_division=self.config.zero_division
            )
        else:
            return "Classification report not available for single class"
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix."""
        return confusion_matrix(self.targets, self.predictions)
    
    def get_error_samples(self, num_samples: int = 10) -> List[Dict[str, Any]]:
        """Get sample error cases."""
        error_indices = np.where(self.predictions != self.targets)[0]
        
        if len(error_indices) == 0:
            return []
        
        # Get random sample of errors
        sample_indices = np.random.choice(
            error_indices, 
            size=min(num_samples, len(error_indices)), 
            replace=False
        )
        
        error_samples = []
        for idx in sample_indices:
            error_samples.append({
                'index': int(idx),
                'target': int(self.targets[idx]),
                'prediction': int(self.predictions[idx]),
                'confidence': float(np.max(self.probabilities[idx])) if len(self.probabilities.shape) > 1 else 0.0
            })
        
        return error_samples
    
    def compare_with_baseline(self, baseline_predictions: np.ndarray) -> Dict[str, Any]:
        """Compare current model with baseline predictions."""
        baseline_metrics = {}
        current_metrics = {}
        
        # Calculate metrics for both
        for metric_name in self.config.metrics:
            baseline_metrics[metric_name] = self._calculate_metric_for_predictions(
                self.targets, baseline_predictions, metric_name
            )
            current_metrics[metric_name] = self._calculate_metric_for_predictions(
                self.targets, self.predictions, metric_name
            )
        
        # Calculate improvements
        improvements = {}
        for metric_name in self.config.metrics:
            baseline_val = baseline_metrics[metric_name]
            current_val = current_metrics[metric_name]
            
            if baseline_val != 0:
                improvement = (current_val - baseline_val) / baseline_val * 100
            else:
                improvement = float('inf') if current_val > 0 else 0
            
            improvements[metric_name] = improvement
        
        comparison = {
            'baseline_metrics': baseline_metrics,
            'current_metrics': current_metrics,
            'improvements': improvements
        }
        
        return comparison
    
    def _calculate_metric_for_predictions(self, targets: np.ndarray, predictions: np.ndarray, metric_name: str) -> float:
        """Calculate metric for given predictions."""
        metric_functions = {
            'accuracy': lambda: accuracy_score(targets, predictions),
            'precision': lambda: precision_score(targets, predictions, 
                                               average=self.config.average, 
                                               zero_division=self.config.zero_division),
            'recall': lambda: recall_score(targets, predictions, 
                                         average=self.config.average, 
                                         zero_division=self.config.zero_division),
            'f1': lambda: f1_score(targets, predictions, 
                                 average=self.config.average, 
                                 zero_division=self.config.zero_division)
        }
        
        if metric_name in metric_functions:
            try:
                return metric_functions[metric_name]()
            except Exception as e:
                logger.warning(f"Error calculating {metric_name}: {e}")
                return 0.0
        else:
            return 0.0


class RegressionEvaluator(BaseEvaluator):
    """Evaluator for regression tasks."""
    
    def _calculate_metrics(self) -> Any:
        """Calculate regression metrics."""
        metrics = {
            'mse': mean_squared_error(self.targets, self.predictions),
            'mae': mean_absolute_error(self.targets, self.predictions),
            'r2': r2_score(self.targets, self.predictions),
            'rmse': np.sqrt(mean_squared_error(self.targets, self.predictions))
        }
        
        self.evaluation_results['metrics'] = metrics
        
        # Log metrics
        logger.info("Regression Metrics:")
        for metric_name, metric_value in metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")
    
    def _plot_regression_results(self, plots_dir: Path):
        """Plot regression-specific visualizations."""
        # Scatter plot of predictions vs targets
        plt.figure(figsize=(10, 8))
        plt.scatter(self.targets, self.predictions, alpha=0.6)
        plt.plot([self.targets.min(), self.targets.max()], 
                [self.targets.min(), self.targets.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predictions vs Actual Values')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / "regression_scatter.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Residual plot
        residuals = self.targets - self.predictions
        plt.figure(figsize=(10, 8))
        plt.scatter(self.predictions, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / "residual_plot.png", dpi=300, bbox_inches='tight')
        plt.close()


# Example usage
if __name__ == "__main__":
    # Create a simple model for testing
    
    class SimpleModel(nn.Module):
        def __init__(self) -> Any:
            super().__init__()
            self.fc = nn.Linear(784, 10)
        
        def forward(self, x) -> Any:
            return self.fc(x.view(x.size(0), -1))
    
    # Create synthetic test data
    test_data = torch.randn(500, 784)
    test_labels = torch.randint(0, 10, (500,))
    
    class SimpleDataset(data.Dataset):
        def __init__(self, data, labels) -> Any:
            self.data = data
            self.labels = labels
        
        def __len__(self) -> Any:
            return len(self.data)
        
        def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
            return self.data[idx], self.labels[idx]
    
    # Create test dataset
    test_dataset = SimpleDataset(test_data, test_labels)
    
    # Create model
    model = SimpleModel()
    
    # Create evaluation config
    config = EvaluationConfig(
        batch_size=32,
        metrics=["accuracy", "precision", "recall", "f1"],
        output_dir="test_evaluation"
    )
    
    # Create evaluator
    evaluator = BaseEvaluator(model, test_dataset, config)
    
    # Run evaluation
    results = evaluator.evaluate()
    
    print("Evaluation completed successfully!")
    print(f"Results: {results['metrics']}") 