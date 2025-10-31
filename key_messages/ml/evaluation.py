from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import structlog
from tqdm import tqdm
import time
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from .models import BaseMessageModel
from .data_loader import MessageDataset
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Evaluation Module for Key Messages Feature - Modular Architecture
"""
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)


logger = structlog.get_logger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation class."""
    
    def __init__(self, model: BaseMessageModel, device: str = "cuda"):
        
    """__init__ function."""
self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        
        # Evaluation results storage
        self.evaluation_results = {}
        self.predictions = []
        self.true_labels = []
        self.confidence_scores = []
        
        logger.info("ModelEvaluator initialized", device=str(self.device))
    
    def evaluate_model(self, test_loader: DataLoader, task_type: str = "generation") -> Dict[str, Any]:
        """Main evaluation method."""
        logger.info("Starting model evaluation", task_type=task_type)
        
        start_time = time.time()
        
        if task_type == "generation":
            results = self._evaluate_generation_task(test_loader)
        elif task_type == "classification":
            results = self._evaluate_classification_task(test_loader)
        elif task_type == "regression":
            results = self._evaluate_regression_task(test_loader)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        evaluation_time = time.time() - start_time
        
        # Add metadata
        results['evaluation_time'] = evaluation_time
        results['task_type'] = task_type
        results['device'] = str(self.device)
        
        self.evaluation_results = results
        
        logger.info("Model evaluation completed", 
                   task_type=task_type,
                   evaluation_time=evaluation_time,
                   metrics=results.get('metrics', {}))
        
        return results
    
    def _evaluate_generation_task(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Evaluate text generation task."""
        generated_texts = []
        reference_texts = []
        generation_times = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating generation"):
                batch_start_time = time.time()
                
                # Get prompts from batch
                prompts = batch.get('original_message', [''] * len(batch))
                
                # Generate text
                for prompt in prompts:
                    try:
                        generated_text = self.model.generate(prompt)
                        generated_texts.append(generated_text)
                        
                        # Get reference text if available
                        reference = batch.get('generated_response', [''])[0]
                        reference_texts.append(reference)
                        
                    except Exception as e:
                        logger.warning("Generation failed", error=str(e), prompt=prompt[:100])
                        generated_texts.append("")
                        reference_texts.append("")
                
                generation_times.append(time.time() - batch_start_time)
        
        # Calculate generation metrics
        metrics = self._calculate_generation_metrics(generated_texts, reference_texts)
        
        # Add performance metrics
        metrics.update({
            'avg_generation_time': np.mean(generation_times),
            'total_generation_time': np.sum(generation_times),
            'samples_per_second': len(generated_texts) / np.sum(generation_times)
        })
        
        return {
            'metrics': metrics,
            'generated_texts': generated_texts,
            'reference_texts': reference_texts,
            'generation_times': generation_times
        }
    
    def _evaluate_classification_task(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Evaluate classification task."""
        all_predictions = []
        all_true_labels = []
        all_confidence_scores = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating classification"):
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch.get('attention_mask', None)
                )
                
                # Get predictions
                if hasattr(self.model, 'classify'):
                    # Use model's classify method if available
                    for text in batch.get('original_message', []):
                        result = self.model.classify(text)
                        all_confidence_scores.append(result)
                        # Get predicted class
                        predicted_class = max(result.items(), key=lambda x: x[1])[0]
                        all_predictions.append(predicted_class)
                else:
                    # Standard classification
                    probabilities = torch.softmax(outputs.logits, dim=-1)
                    predictions = torch.argmax(probabilities, dim=-1)
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    all_confidence_scores.extend(probabilities.cpu().numpy())
                
                # Get true labels
                if 'labels' in batch:
                    all_true_labels.extend(batch['labels'].cpu().numpy())
        
        # Calculate classification metrics
        metrics = self._calculate_classification_metrics(all_predictions, all_true_labels)
        
        return {
            'metrics': metrics,
            'predictions': all_predictions,
            'true_labels': all_true_labels,
            'confidence_scores': all_confidence_scores
        }
    
    def _evaluate_regression_task(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Evaluate regression task."""
        all_predictions = []
        all_true_values = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating regression"):
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch.get('attention_mask', None)
                )
                
                # Get predictions (assuming single value output)
                predictions = outputs.logits.squeeze()
                all_predictions.extend(predictions.cpu().numpy())
                
                # Get true values
                if 'labels' in batch:
                    all_true_values.extend(batch['labels'].cpu().numpy())
        
        # Calculate regression metrics
        metrics = self._calculate_regression_metrics(all_predictions, all_true_values)
        
        return {
            'metrics': metrics,
            'predictions': all_predictions,
            'true_values': all_true_values
        }
    
    def _calculate_generation_metrics(self, generated_texts: List[str], reference_texts: List[str]) -> Dict[str, float]:
        """Calculate metrics for text generation."""
        metrics = {}
        
        # Basic text statistics
        metrics['avg_generated_length'] = np.mean([len(text) for text in generated_texts])
        metrics['avg_reference_length'] = np.mean([len(text) for text in reference_texts])
        
        # BLEU score (simplified)
        metrics['bleu_score'] = self._calculate_bleu_score(generated_texts, reference_texts)
        
        # Perplexity (if applicable)
        if hasattr(self.model, 'calculate_perplexity'):
            metrics['perplexity'] = self.model.calculate_perplexity(generated_texts)
        
        # Diversity metrics
        metrics['unique_ratio'] = len(set(generated_texts)) / len(generated_texts)
        
        # Coherence metrics (simplified)
        metrics['coherence_score'] = self._calculate_coherence_score(generated_texts)
        
        return metrics
    
    def _calculate_classification_metrics(self, predictions: List, true_labels: List) -> Dict[str, float]:
        """Calculate metrics for classification."""
        if not true_labels:
            return {'error': 'No true labels provided'}
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        
        # Basic metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted'
        )
        
        # Per-class metrics
        class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average=None
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'class_precision': class_precision.tolist(),
            'class_recall': class_recall.tolist(),
            'class_f1': class_f1.tolist()
        }
        
        return metrics
    
    def _calculate_regression_metrics(self, predictions: List[float], true_values: List[float]) -> Dict[str, float]:
        """Calculate metrics for regression."""
        if not true_values:
            return {'error': 'No true values provided'}
        
        predictions = np.array(predictions)
        true_values = np.array(true_values)
        
        metrics = {
            'mse': mean_squared_error(true_values, predictions),
            'rmse': np.sqrt(mean_squared_error(true_values, predictions)),
            'mae': mean_absolute_error(true_values, predictions),
            'r2_score': r2_score(true_values, predictions),
            'correlation': np.corrcoef(true_values, predictions)[0, 1]
        }
        
        return metrics
    
    def _calculate_bleu_score(self, generated_texts: List[str], reference_texts: List[str]) -> float:
        """Calculate simplified BLEU score."""
        # This is a simplified BLEU score calculation
        # In practice, you'd use a proper BLEU implementation
        total_score = 0.0
        valid_pairs = 0
        
        for gen, ref in zip(generated_texts, reference_texts):
            if gen and ref:
                # Simple n-gram overlap
                gen_words = set(gen.lower().split())
                ref_words = set(ref.lower().split())
                
                if ref_words:
                    overlap = len(gen_words.intersection(ref_words))
                    score = overlap / len(ref_words)
                    total_score += score
                    valid_pairs += 1
        
        return total_score / valid_pairs if valid_pairs > 0 else 0.0
    
    def _calculate_coherence_score(self, texts: List[str]) -> float:
        """Calculate text coherence score."""
        # Simplified coherence calculation
        # In practice, you'd use more sophisticated methods
        total_score = 0.0
        valid_texts = 0
        
        for text in texts:
            if text:
                # Simple coherence based on sentence structure
                sentences = text.split('.')
                if len(sentences) > 1:
                    # Check if sentences have reasonable length
                    avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
                    coherence = min(avg_sentence_length / 10.0, 1.0)  # Normalize
                    total_score += coherence
                    valid_texts += 1
        
        return total_score / valid_texts if valid_texts > 0 else 0.0
    
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Move batch tensors to device."""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch
    
    def generate_evaluation_report(self, output_path: str = None) -> str:
        """Generate comprehensive evaluation report."""
        if not self.evaluation_results:
            raise ValueError("No evaluation results available. Run evaluate_model first.")
        
        report = self._create_report_content()
        
        if output_path:
            with open(output_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(report)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            logger.info("Evaluation report saved", path=output_path)
        
        return report
    
    def _create_report_content(self) -> str:
        """Create evaluation report content."""
        results = self.evaluation_results
        
        report = f"""
# Model Evaluation Report

## Summary
- **Task Type**: {results.get('task_type', 'Unknown')}
- **Device**: {results.get('device', 'Unknown')}
- **Evaluation Time**: {results.get('evaluation_time', 0):.2f} seconds

## Metrics
"""
        
        metrics = results.get('metrics', {})
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                report += f"- **{metric_name}**: {metric_value:.4f}\n"
            elif isinstance(metric_value, list):
                report += f"- **{metric_name}**: {metric_value}\n"
            else:
                report += f"- **{metric_name}**: {metric_value}\n"
        
        report += "\n## Detailed Analysis\n"
        
        # Add task-specific analysis
        task_type = results.get('task_type', '')
        if task_type == 'generation':
            report += self._create_generation_analysis()
        elif task_type == 'classification':
            report += self._create_classification_analysis()
        elif task_type == 'regression':
            report += self._create_regression_analysis()
        
        return report
    
    def _create_generation_analysis(self) -> str:
        """Create generation-specific analysis."""
        generated_texts = self.evaluation_results.get('generated_texts', [])
        reference_texts = self.evaluation_results.get('reference_texts', [])
        
        analysis = f"""
### Generation Analysis
- **Total Generated Texts**: {len(generated_texts)}
- **Average Generation Time**: {self.evaluation_results.get('metrics', {}).get('avg_generation_time', 0):.4f} seconds
- **Samples per Second**: {self.evaluation_results.get('metrics', {}).get('samples_per_second', 0):.2f}

#### Sample Generations
"""
        
        for i, (gen, ref) in enumerate(zip(generated_texts[:5], reference_texts[:5])):
            analysis += f"""
**Sample {i+1}**:
- **Generated**: {gen[:200]}...
- **Reference**: {ref[:200]}...
"""
        
        return analysis
    
    def _create_classification_analysis(self) -> str:
        """Create classification-specific analysis."""
        predictions = self.evaluation_results.get('predictions', [])
        true_labels = self.evaluation_results.get('true_labels', [])
        
        if not true_labels:
            return "\n### Classification Analysis\nNo true labels available for analysis."
        
        analysis = f"""
### Classification Analysis
- **Total Predictions**: {len(predictions)}
- **Accuracy**: {self.evaluation_results.get('metrics', {}).get('accuracy', 0):.4f}
- **F1 Score**: {self.evaluation_results.get('metrics', {}).get('f1_score', 0):.4f}

#### Confusion Matrix
"""
        
        # Create confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        analysis += f"```\n{cm}\n```\n"
        
        return analysis
    
    def _create_regression_analysis(self) -> str:
        """Create regression-specific analysis."""
        predictions = self.evaluation_results.get('predictions', [])
        true_values = self.evaluation_results.get('true_values', [])
        
        if not true_values:
            return "\n### Regression Analysis\nNo true values available for analysis."
        
        analysis = f"""
### Regression Analysis
- **Total Predictions**: {len(predictions)}
- **R² Score**: {self.evaluation_results.get('metrics', {}).get('r2_score', 0):.4f}
- **RMSE**: {self.evaluation_results.get('metrics', {}).get('rmse', 0):.4f}

#### Prediction vs True Values
- **Correlation**: {self.evaluation_results.get('metrics', {}).get('correlation', 0):.4f}
"""
        
        return analysis
    
    def create_visualizations(self, output_dir: str = "evaluation_plots"):
        """Create evaluation visualizations."""
        if not self.evaluation_results:
            raise ValueError("No evaluation results available. Run evaluate_model first.")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        task_type = self.evaluation_results.get('task_type', '')
        
        if task_type == 'generation':
            self._create_generation_plots(output_path)
        elif task_type == 'classification':
            self._create_classification_plots(output_path)
        elif task_type == 'regression':
            self._create_regression_plots(output_path)
        
        logger.info("Visualizations created", output_dir=str(output_path))
    
    def _create_generation_plots(self, output_path: Path):
        """Create generation-specific plots."""
        metrics = self.evaluation_results.get('metrics', {})
        
        # Length distribution plot
        generated_texts = self.evaluation_results.get('generated_texts', [])
        lengths = [len(text) for text in generated_texts]
        
        plt.figure(figsize=(10, 6))
        plt.hist(lengths, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Text Length')
        plt.ylabel('Frequency')
        plt.title('Distribution of Generated Text Lengths')
        plt.savefig(output_path / 'generation_length_dist.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_classification_plots(self, output_path: Path):
        """Create classification-specific plots."""
        predictions = self.evaluation_results.get('predictions', [])
        true_labels = self.evaluation_results.get('true_labels', [])
        
        if not true_labels:
            return
        
        # Confusion matrix heatmap
        cm = confusion_matrix(true_labels, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(output_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_regression_plots(self, output_path: Path):
        """Create regression-specific plots."""
        predictions = self.evaluation_results.get('predictions', [])
        true_values = self.evaluation_results.get('true_values', [])
        
        if not true_values:
            return
        
        # Scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(true_values, predictions, alpha=0.6)
        plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], 'r--', lw=2)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title('Predictions vs True Values')
        plt.savefig(output_path / 'regression_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()

class EvaluationManager:
    """High-level evaluation management class."""
    
    def __init__(self, model: BaseMessageModel, device: str = "cuda"):
        
    """__init__ function."""
self.evaluator = ModelEvaluator(model, device)
    
    def run_comprehensive_evaluation(self, test_loader: DataLoader, task_type: str = "generation",
                                   output_dir: str = "evaluation_results") -> Dict[str, Any]:
        """Run comprehensive evaluation with report and visualizations."""
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Run evaluation
        results = self.evaluator.evaluate_model(test_loader, task_type)
        
        # Generate report
        report_path = output_path / "evaluation_report.md"
        self.evaluator.generate_evaluation_report(str(report_path))
        
        # Create visualizations
        plots_dir = output_path / "plots"
        self.evaluator.create_visualizations(str(plots_dir))
        
        # Save results as JSON
        results_path = output_path / "evaluation_results.json"
        with open(results_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(results, f, indent=2, default=str)
        
        logger.info("Comprehensive evaluation completed", 
                   output_dir=str(output_path),
                   task_type=task_type)
        
        return results 