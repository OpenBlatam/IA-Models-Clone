from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import asyncio
import time
import logging
import json
import pickle
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import warnings
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import (
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from .production_transformers import ProductionTransformersEngine, DeviceManager
from .model_training import ModelTrainer, TrainingConfig, ModelType, TrainingMode
        from .model_training import ModelTrainer
from typing import Any, List, Dict, Optional
"""
🚀 Model Evaluation System - Production Ready
=============================================

Enterprise-grade model evaluation system with advanced metrics,
cross-validation, model comparison, and production evaluation pipelines.
"""


    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    log_loss, cohen_kappa_score, matthews_corrcoef,
    hamming_loss, jaccard_score, f1_score
)

# Import our production engines

logger = logging.getLogger(__name__)

class EvaluationType(Enum):
    """Available evaluation types."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    MULTI_LABEL = "multi_label"
    RANKING = "ranking"
    GENERATION = "generation"

@dataclass
class ModelPerformance:
    """Model performance metrics."""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: Optional[float] = None
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: Optional[str] = None
    inference_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    model_size_mb: float = 0.0
    
    # Regression metrics
    mse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None
    
    # Advanced metrics
    cohen_kappa: Optional[float] = None
    matthews_corr: Optional[float] = None
    log_loss: Optional[float] = None

@dataclass
class CrossValidationResult:
    """Cross-validation results."""
    model_name: str
    cv_scores: List[float]
    mean_score: float
    std_score: float
    fold_results: List[Dict[str, float]]
    best_fold: int
    worst_fold: int

@dataclass
class ModelComparison:
    """Model comparison results."""
    models: List[str]
    metrics: Dict[str, List[float]]
    statistical_tests: Dict[str, float]
    ranking: List[Tuple[str, float]]
    best_model: str

class ModelEvaluator:
    """Production-ready model evaluator."""
    
    def __init__(self, device_manager: DeviceManager):
        
    """__init__ function."""
self.device_manager = device_manager
        self.device = device_manager.get_best_device()
        self.logger = logging.getLogger(f"{__name__}.ModelEvaluator")
        
    def calculate_advanced_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 y_prob: Optional[np.ndarray] = None,
                                 task_type: str = "classification") -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        metrics = {}
        
        if task_type == "classification":
            # Basic metrics
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='weighted'
            )
            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['f1_score'] = f1
            
            # Advanced classification metrics
            metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
            metrics['matthews_corr'] = matthews_corrcoef(y_true, y_pred)
            
            if y_prob is not None:
                metrics['log_loss'] = log_loss(y_true, y_prob)
                
                # ROC AUC for binary classification
                if len(np.unique(y_true)) == 2:
                    try:
                        metrics['auc_roc'] = roc_auc_score(y_true, y_prob[:, 1])
                    except:
                        metrics['auc_roc'] = None
                else:
                    # Multi-class ROC AUC
                    try:
                        y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
                        metrics['auc_roc'] = roc_auc_score(y_true_bin, y_prob, average='weighted')
                    except:
                        metrics['auc_roc'] = None
        
        elif task_type == "regression":
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2'] = r2_score(y_true, y_pred)
            
            # Additional regression metrics
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mape'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        elif task_type == "multi_label":
            metrics['hamming_loss'] = hamming_loss(y_true, y_pred)
            metrics['jaccard_score'] = jaccard_score(y_true, y_pred, average='weighted')
            metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
            metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro')
        
        return metrics
    
    def perform_statistical_tests(self, model_scores: Dict[str, List[float]]) -> Dict[str, float]:
        """Perform statistical significance tests between models."""
        tests = {}
        model_names = list(model_scores.keys())
        
        if len(model_names) >= 2:
            # Paired t-test
            scores1 = model_scores[model_names[0]]
            scores2 = model_scores[model_names[1]]
            t_stat, p_value = stats.ttest_rel(scores1, scores2)
            tests['paired_t_test'] = p_value
            
            # Wilcoxon signed-rank test
            w_stat, w_p_value = stats.wilcoxon(scores1, scores2)
            tests['wilcoxon_test'] = w_p_value
            
            # ANOVA for multiple models
            if len(model_names) > 2:
                all_scores = [model_scores[name] for name in model_names]
                f_stat, anova_p_value = stats.f_oneway(*all_scores)
                tests['anova_test'] = anova_p_value
        
        return tests
    
    async def evaluate_model_performance(self, model: nn.Module, test_loader, 
                                       config: TrainingConfig) -> ModelPerformance:
        """Evaluate single model performance."""
        self.logger.info(f"Evaluating model: {config.model_name}")
        
        model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        inference_times = []
        
        # Measure model size
        model_size = sum(p.numel() * p.element_size() for p in model.parameters())
        model_size_mb = model_size / (1024 * 1024)
        
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Measure inference time
                start_time = time.time()
                outputs = model(**batch)
                inference_time = (time.time() - start_time) * 1000
                inference_times.append(inference_time)
                
                # Get predictions and probabilities
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                    probabilities = torch.softmax(logits, dim=-1)
                    predictions = torch.argmax(logits, dim=-1)
                    
                    all_probabilities.extend(probabilities.cpu().numpy())
                else:
                    predictions = outputs
                    all_probabilities = None
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch['labels'].cpu().numpy())
        
        # Calculate metrics
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        probabilities = np.array(all_probabilities) if all_probabilities else None
        
        metrics = self.calculate_advanced_metrics(
            targets, predictions, probabilities, 
            task_type="classification"
        )
        
        # Calculate confusion matrix
        conf_matrix = confusion_matrix(targets, predictions)
        class_report = classification_report(targets, predictions)
        
        # Calculate average inference time
        avg_inference_time = np.mean(inference_times)
        
        # Estimate memory usage (rough approximation)
        memory_usage_mb = model_size_mb * 2  # Rough estimate including activations
        
        performance = ModelPerformance(
            model_name=config.model_name,
            accuracy=metrics['accuracy'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            f1_score=metrics['f1_score'],
            auc_roc=metrics.get('auc_roc'),
            confusion_matrix=conf_matrix,
            classification_report=class_report,
            inference_time_ms=avg_inference_time,
            memory_usage_mb=memory_usage_mb,
            model_size_mb=model_size_mb,
            cohen_kappa=metrics.get('cohen_kappa'),
            matthews_corr=metrics.get('matthews_corr'),
            log_loss=metrics.get('log_loss')
        )
        
        return performance
    
    async def cross_validate_model(self, model_class, config: TrainingConfig, 
                                 dataset, n_folds: int = 5) -> CrossValidationResult:
        """Perform cross-validation on model."""
        self.logger.info(f"Performing {n_folds}-fold cross-validation for {config.model_name}")
        
        # Prepare data
        texts = dataset.texts
        labels = dataset.labels
        
        # Initialize cross-validation
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        cv_scores = []
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
            self.logger.info(f"Training fold {fold + 1}/{n_folds}")
            
            # Split data
            train_texts = [texts[i] for i in train_idx]
            train_labels = [labels[i] for i in train_idx]
            val_texts = [texts[i] for i in val_idx]
            val_labels = [labels[i] for i in val_idx]
            
            # Create datasets
            train_dataset = CustomDataset(train_texts, train_labels)
            val_dataset = CustomDataset(val_texts, val_labels)
            
            # Create model
            num_classes = len(set(labels))
            model, _ = self.create_model(config, num_classes)
            model = model.to(self.device)
            
            # Train model
            trainer = ModelTrainer(self.device_manager)
            fold_config = TrainingConfig(
                model_type=config.model_type,
                training_mode=config.training_mode,
                model_name=f"{config.model_name}_fold_{fold}",
                dataset_path="",  # Not used for cross-validation
                output_dir=config.output_dir,
                num_epochs=config.num_epochs // 2,  # Shorter training for CV
                batch_size=config.batch_size,
                learning_rate=config.learning_rate
            )
            
            # Create dataloaders
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
            
            # Train and evaluate
            best_score = 0.0
            for epoch in range(fold_config.num_epochs):
                # Train
                train_metrics = await trainer.train_epoch(
                    model, train_loader, None, None, fold_config
                )
                
                # Validate
                val_metrics = await trainer.validate_epoch(model, val_loader)
                
                if val_metrics['f1'] > best_score:
                    best_score = val_metrics['f1']
            
            cv_scores.append(best_score)
            fold_results.append({
                'fold': fold + 1,
                'f1_score': best_score,
                'accuracy': val_metrics['accuracy']
            })
        
        # Calculate statistics
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        best_fold = np.argmax(cv_scores) + 1
        worst_fold = np.argmin(cv_scores) + 1
        
        result = CrossValidationResult(
            model_name=config.model_name,
            cv_scores=cv_scores,
            mean_score=mean_score,
            std_score=std_score,
            fold_results=fold_results,
            best_fold=best_fold,
            worst_fold=worst_fold
        )
        
        return result
    
    async def compare_models(self, models: List[Tuple[str, nn.Module]], 
                           test_loader, configs: List[TrainingConfig]) -> ModelComparison:
        """Compare multiple models."""
        self.logger.info(f"Comparing {len(models)} models")
        
        model_performances = []
        model_scores = {}
        
        for (model_name, model), config in zip(models, configs):
            # Evaluate model
            performance = await self.evaluate_model_performance(model, test_loader, config)
            model_performances.append(performance)
            
            # Store scores for statistical testing
            model_scores[model_name] = [performance.f1_score]  # Single score for now
        
        # Perform statistical tests
        statistical_tests = self.perform_statistical_tests(model_scores)
        
        # Create ranking
        model_rankings = [(p.model_name, p.f1_score) for p in model_performances]
        model_rankings.sort(key=lambda x: x[1], reverse=True)
        
        best_model = model_rankings[0][0]
        
        # Compile metrics
        metrics = {
            'accuracy': [p.accuracy for p in model_performances],
            'precision': [p.precision for p in model_performances],
            'recall': [p.recall for p in model_performances],
            'f1_score': [p.f1_score for p in model_performances],
            'inference_time_ms': [p.inference_time_ms for p in model_performances],
            'model_size_mb': [p.model_size_mb for p in model_performances]
        }
        
        comparison = ModelComparison(
            models=[p.model_name for p in model_performances],
            metrics=metrics,
            statistical_tests=statistical_tests,
            ranking=model_rankings,
            best_model=best_model
        )
        
        return comparison
    
    def create_model(self, config: TrainingConfig, num_classes: int) -> Tuple[nn.Module, Any]:
        """Create model (reuse from ModelTrainer)."""
        trainer = ModelTrainer(self.device_manager)
        return trainer.create_model(config, num_classes)
    
    def generate_evaluation_report(self, performance: ModelPerformance, 
                                 output_path: str = "evaluation_report.html"):
        """Generate comprehensive evaluation report."""
        # Create interactive plots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Performance Metrics', 'Confusion Matrix', 
                          'Model Statistics', 'Inference Performance'),
            specs=[[{"type": "bar"}, {"type": "heatmap"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Performance metrics
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [performance.accuracy, performance.precision, 
                 performance.recall, performance.f1_score]
        
        fig.add_trace(
            go.Bar(x=metrics, y=values, name='Metrics'),
            row=1, col=1
        )
        
        # Confusion matrix
        if performance.confusion_matrix is not None:
            fig.add_trace(
                go.Heatmap(z=performance.confusion_matrix, 
                          colorscale='Blues', showscale=True),
                row=1, col=2
            )
        
        # Model statistics
        stats_metrics = ['Model Size (MB)', 'Memory Usage (MB)']
        stats_values = [performance.model_size_mb, performance.memory_usage_mb]
        
        fig.add_trace(
            go.Bar(x=stats_metrics, y=stats_values, name='Statistics'),
            row=2, col=1
        )
        
        # Inference performance
        fig.add_trace(
            go.Scatter(x=[performance.model_size_mb], 
                      y=[performance.inference_time_ms],
                      mode='markers', name='Inference Time',
                      marker=dict(size=15)),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f"Model Evaluation Report: {performance.model_name}",
            height=800,
            showlegend=False
        )
        
        # Save report
        fig.write_html(output_path)
        
        # Save detailed metrics
        metrics_path = output_path.replace('.html', '_metrics.json')
        with open(metrics_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump({
                'model_name': performance.model_name,
                'accuracy': performance.accuracy,
                'precision': performance.precision,
                'recall': performance.recall,
                'f1_score': performance.f1_score,
                'auc_roc': performance.auc_roc,
                'inference_time_ms': performance.inference_time_ms,
                'model_size_mb': performance.model_size_mb,
                'memory_usage_mb': performance.memory_usage_mb,
                'cohen_kappa': performance.cohen_kappa,
                'matthews_corr': performance.matthews_corr,
                'log_loss': performance.log_loss,
                'confusion_matrix': performance.confusion_matrix.tolist() if performance.confusion_matrix is not None else None,
                'classification_report': performance.classification_report
            }, f, indent=2)
        
        self.logger.info(f"Evaluation report saved to {output_path}")
        return output_path

class ProductionEvaluationPipeline:
    """Production evaluation pipeline."""
    
    def __init__(self, device_manager: DeviceManager):
        
    """__init__ function."""
self.device_manager = device_manager
        self.evaluator = ModelEvaluator(device_manager)
        self.logger = logging.getLogger(f"{__name__}.ProductionEvaluationPipeline")
    
    async def run_full_evaluation(self, model_paths: List[str], 
                                test_dataset_path: str,
                                output_dir: str = "evaluation_results") -> Dict[str, Any]:
        """Run comprehensive evaluation pipeline."""
        self.logger.info("Starting production evaluation pipeline")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        results = {
            'evaluation_timestamp': time.time(),
            'models_evaluated': [],
            'comparison_results': None,
            'cross_validation_results': [],
            'reports_generated': []
        }
        
        # Load test dataset
        test_dataset = self.load_test_dataset(test_dataset_path)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        # Evaluate each model
        models = []
        configs = []
        
        for model_path in model_paths:
            try:
                # Load model
                model, config = self.load_model(model_path)
                models.append((config.model_name, model))
                configs.append(config)
                
                # Evaluate performance
                performance = await self.evaluator.evaluate_model_performance(
                    model, test_loader, config
                )
                
                # Generate report
                report_path = output_path / f"{config.model_name}_evaluation_report.html"
                self.evaluator.generate_evaluation_report(performance, str(report_path))
                
                results['models_evaluated'].append({
                    'model_name': config.model_name,
                    'model_path': model_path,
                    'performance': vars(performance),
                    'report_path': str(report_path)
                })
                
                # Cross-validation
                cv_result = await self.evaluator.cross_validate_model(
                    type(model), config, test_dataset
                )
                results['cross_validation_results'].append(vars(cv_result))
                
            except Exception as e:
                self.logger.error(f"Failed to evaluate model {model_path}: {e}")
        
        # Compare models
        if len(models) > 1:
            comparison = await self.evaluator.compare_models(models, test_loader, configs)
            results['comparison_results'] = vars(comparison)
            
            # Generate comparison report
            comparison_report = self.generate_comparison_report(comparison, output_path)
            results['reports_generated'].append(comparison_report)
        
        # Save comprehensive results
        results_path = output_path / "evaluation_results.json"
        with open(results_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Evaluation pipeline completed. Results saved to {output_path}")
        return results
    
    def load_test_dataset(self, dataset_path: str):
        """Load test dataset."""
        # Implementation depends on your dataset format
        if dataset_path.endswith('.csv'):
            df = pd.read_csv(dataset_path)
            texts = df['text'].tolist()
            labels = df['label'].tolist()
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_path}")
        
        return CustomDataset(texts, labels)
    
    def load_model(self, model_path: str) -> Tuple[nn.Module, TrainingConfig]:
        """Load trained model."""
        checkpoint = torch.load(model_path, map_location=self.device_manager.get_best_device())
        
        # Extract config
        config = checkpoint['config']
        
        # Create model
        num_classes = 2  # Default, should be extracted from config
        model, _ = self.evaluator.create_model(config, num_classes)
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, config
    
    def generate_comparison_report(self, comparison: ModelComparison, 
                                 output_path: Path) -> str:
        """Generate model comparison report."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('F1 Score Comparison', 'Inference Time vs Model Size',
                          'Accuracy Comparison', 'Model Ranking'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # F1 Score comparison
        fig.add_trace(
            go.Bar(x=comparison.models, y=comparison.metrics['f1_score'],
                  name='F1 Score'),
            row=1, col=1
        )
        
        # Inference time vs model size
        fig.add_trace(
            go.Scatter(x=comparison.metrics['model_size_mb'],
                      y=comparison.metrics['inference_time_ms'],
                      mode='markers+text',
                      text=comparison.models,
                      name='Inference Performance'),
            row=1, col=2
        )
        
        # Accuracy comparison
        fig.add_trace(
            go.Bar(x=comparison.models, y=comparison.metrics['accuracy'],
                  name='Accuracy'),
            row=2, col=1
        )
        
        # Model ranking
        ranked_models = [model for model, _ in comparison.ranking]
        ranked_scores = [score for _, score in comparison.ranking]
        
        fig.add_trace(
            go.Bar(x=ranked_models, y=ranked_scores,
                  name='Ranking Score'),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Model Comparison Report",
            height=800,
            showlegend=False
        )
        
        report_path = output_path / "model_comparison_report.html"
        fig.write_html(str(report_path))
        
        return str(report_path)

# Factory functions
async def create_model_evaluator(device_manager: DeviceManager) -> ModelEvaluator:
    """Create a model evaluator instance."""
    return ModelEvaluator(device_manager)

async def create_evaluation_pipeline(device_manager: DeviceManager) -> ProductionEvaluationPipeline:
    """Create an evaluation pipeline instance."""
    return ProductionEvaluationPipeline(device_manager)

# Quick evaluation functions
async def quick_model_evaluation(
    model_path: str,
    test_dataset_path: str,
    output_dir: str = "evaluation_results"
) -> Dict[str, Any]:
    """Quick model evaluation."""
    device_manager = DeviceManager()
    evaluator = await create_model_evaluator(device_manager)
    
    # Load model and dataset
    test_dataset = evaluator.load_test_dataset(test_dataset_path)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    model, config = evaluator.load_model(model_path)
    
    # Evaluate
    performance = await evaluator.evaluate_model_performance(model, test_loader, config)
    
    # Generate report
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    report_path = output_path / f"{config.model_name}_evaluation_report.html"
    evaluator.generate_evaluation_report(performance, str(report_path))
    
    return {
        'performance': vars(performance),
        'report_path': str(report_path)
    }

async def quick_model_comparison(
    model_paths: List[str],
    test_dataset_path: str,
    output_dir: str = "comparison_results"
) -> Dict[str, Any]:
    """Quick model comparison."""
    device_manager = DeviceManager()
    pipeline = await create_evaluation_pipeline(device_manager)
    
    return await pipeline.run_full_evaluation(model_paths, test_dataset_path, output_dir)

# Example usage
if __name__ == "__main__":
    async def demo():
        
    """demo function."""
# Quick evaluation example
        result = await quick_model_evaluation(
            model_path="models/distilbert_sentiment_best.pth",
            test_dataset_path="data/test_dataset.csv"
        )
        print(f"Evaluation completed: {result}")
        
        # Model comparison example
        comparison_result = await quick_model_comparison(
            model_paths=[
                "models/distilbert_sentiment_best.pth",
                "models/bert_sentiment_best.pth"
            ],
            test_dataset_path="data/test_dataset.csv"
        )
        print(f"Comparison completed: {comparison_result}")
    
    asyncio.run(demo()) 