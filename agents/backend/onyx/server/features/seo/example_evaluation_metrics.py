from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from evaluation_metrics import (
        from sklearn.metrics import balanced_accuracy_score
        import traceback
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Example: Comprehensive Evaluation Metrics Framework
Demonstrates task-specific evaluation metrics for different deep learning scenarios
"""


# Import our evaluation metrics framework
    EvaluationConfig, EvaluationResult, ModelEvaluator,
    ClassificationMetrics, RegressionMetrics, RankingMetrics,
    SEOMetrics, MultiTaskMetrics, StatisticalAnalysis, EvaluationVisualizer
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SEOSampleDataset(Dataset):
    """Sample SEO dataset for demonstration"""
    
    def __init__(self, num_samples=1000, num_features=768, num_classes=3) -> Any:
        self.num_samples = num_samples
        self.num_features = num_features
        self.num_classes = num_classes
        self._generate_data()
    
    def _generate_data(self) -> Any:
        """Generate synthetic SEO data"""
        np.random.seed(42)
        
        # Generate features
        self.features = torch.randn(self.num_samples, self.num_features)
        
        # Generate labels with some class imbalance
        # Class 0: 60%, Class 1: 30%, Class 2: 10%
        class_weights = [0.6, 0.3, 0.1]
        self.labels = torch.tensor(
            np.random.choice(self.num_classes, self.num_samples, p=class_weights),
            dtype=torch.long
        )
        
        # Add some noise to make training challenging
        noise = torch.randn_like(self.features) * 0.1
        self.features += noise
    
    def __len__(self) -> Any:
        return self.num_samples
    
    def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
        return {
            'features': self.features[idx],
            'labels': self.labels[idx]
        }

def example_classification_evaluation():
    """Example: Classification evaluation with comprehensive metrics"""
    logger.info("=== Classification Evaluation Example ===")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_classes = 3
    
    # Generate true labels and predictions
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = np.random.randint(0, n_classes, n_samples)
    
    # Generate probabilities
    y_prob = np.random.rand(n_samples, n_classes)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    
    # Create evaluation configuration
    config = EvaluationConfig(
        task_type="classification",
        classification_metrics=["accuracy", "precision", "recall", "f1", "auc", "ap", "confusion_matrix"],
        average_method="weighted",
        statistical_analysis=True,
        confidence_intervals=True,
        bootstrap_samples=100,
        create_plots=True,
        save_plots=True,
        verbose=True
    )
    
    # Create evaluator
    evaluator = ModelEvaluator(config)
    
    # Evaluate
    result = evaluator.evaluate(y_true, y_pred, y_prob)
    
    # Print results
    logger.info("Classification Evaluation Results:")
    for metric, value in result.metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Print confidence intervals
    if result.confidence_intervals:
        logger.info("Confidence Intervals:")
        for metric, (lower, upper) in result.confidence_intervals.items():
            logger.info(f"  {metric}: ({lower:.4f}, {upper:.4f})")
    
    # Save results
    evaluator.save_results(result, "classification_evaluation.json")
    
    return result

def example_regression_evaluation():
    """Example: Regression evaluation with comprehensive metrics"""
    logger.info("=== Regression Evaluation Example ===")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate true values and predictions
    y_true = np.random.normal(0, 1, n_samples)
    y_pred = y_true + np.random.normal(0, 0.3, n_samples)  # Add some noise
    
    # Create evaluation configuration
    config = EvaluationConfig(
        task_type="regression",
        regression_metrics=["mse", "rmse", "mae", "r2", "mape", "smape"],
        statistical_analysis=True,
        confidence_intervals=True,
        bootstrap_samples=100,
        create_plots=True,
        save_plots=True,
        verbose=True
    )
    
    # Create evaluator
    evaluator = ModelEvaluator(config)
    
    # Evaluate
    result = evaluator.evaluate(y_true, y_pred)
    
    # Print results
    logger.info("Regression Evaluation Results:")
    for metric, value in result.metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Print statistical tests
    if result.statistical_tests:
        logger.info("Statistical Tests:")
        for test_name, test_result in result.statistical_tests.items():
            logger.info(f"  {test_name}: {test_result}")
    
    # Save results
    evaluator.save_results(result, "regression_evaluation.json")
    
    return result

def example_ranking_evaluation():
    """Example: Ranking evaluation with comprehensive metrics"""
    logger.info("=== Ranking Evaluation Example ===")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate relevance scores and predicted scores
    y_true = np.random.binomial(1, 0.3, n_samples)  # Binary relevance
    y_pred = np.random.rand(n_samples)  # Predicted relevance scores
    
    # Create evaluation configuration
    config = EvaluationConfig(
        task_type="ranking",
        ranking_metrics=["ndcg", "mrr", "map", "precision_at_k", "recall_at_k"],
        k_values=[1, 3, 5, 10],
        create_plots=True,
        save_plots=True,
        verbose=True
    )
    
    # Create evaluator
    evaluator = ModelEvaluator(config)
    
    # Evaluate
    result = evaluator.evaluate(y_true, y_pred)
    
    # Print results
    logger.info("Ranking Evaluation Results:")
    for metric, value in result.metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Save results
    evaluator.save_results(result, "ranking_evaluation.json")
    
    return result

def example_seo_evaluation():
    """Example: SEO-specific evaluation metrics"""
    logger.info("=== SEO Evaluation Example ===")
    
    # Create sample SEO data
    np.random.seed(42)
    n_samples = 1000
    
    # Ranking data
    ranking_data = {
        'y_true': np.random.randint(1, 11, n_samples),  # True rankings 1-10
        'y_pred': np.random.randint(1, 11, n_samples)   # Predicted rankings 1-10
    }
    
    # Traffic data
    traffic_data = {
        'clicks': np.random.poisson(50, n_samples),
        'impressions': np.random.poisson(1000, n_samples),
        'bounces': np.random.poisson(20, n_samples),
        'sessions': np.random.poisson(100, n_samples),
        'time_spent': np.random.exponential(120, n_samples),  # seconds
        'page_views': np.random.poisson(2, n_samples),
        'conversions': np.random.poisson(5, n_samples),
        'organic_sessions': np.random.poisson(80, n_samples),
        'total_sessions': np.random.poisson(100, n_samples)
    }
    
    # Content data
    content_data = {
        'text': "This is a sample SEO content with relevant keywords for search engine optimization.",
        'keyword': "SEO",
        'readability_score': 75.5,
        'word_count': 1500,
        'keyword_density': 0.02,
        'internal_links': 8
    }
    
    # Create evaluation configuration
    config = EvaluationConfig(
        task_type="seo",
        seo_metrics=["ranking_accuracy", "click_through_rate", "bounce_rate", 
                    "time_on_page", "conversion_rate", "organic_traffic", 
                    "keyword_density", "content_quality"],
        create_plots=False,
        verbose=True
    )
    
    # Create evaluator
    evaluator = ModelEvaluator(config)
    
    # Evaluate
    result = evaluator.evaluate(None, None, ranking_data=ranking_data, 
                              traffic_data=traffic_data, content_data=content_data)
    
    # Print results
    logger.info("SEO Evaluation Results:")
    for metric, value in result.metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Save results
    evaluator.save_results(result, "seo_evaluation.json")
    
    return result

def example_multitask_evaluation():
    """Example: Multi-task evaluation metrics"""
    logger.info("=== Multi-task Evaluation Example ===")
    
    # Create sample multi-task data
    np.random.seed(42)
    n_samples = 1000
    
    # Task 1: Classification (3 classes)
    y_true_task1 = np.random.randint(0, 3, n_samples)
    y_pred_task1 = np.random.randint(0, 3, n_samples)
    
    # Task 2: Binary classification
    y_true_task2 = np.random.randint(0, 2, n_samples)
    y_pred_task2 = np.random.randint(0, 2, n_samples)
    
    # Task 3: Regression
    y_true_task3 = np.random.normal(0, 1, n_samples)
    y_pred_task3 = y_true_task3 + np.random.normal(0, 0.3, n_samples)
    
    # Combine into dictionaries
    y_true = {
        'classification': y_true_task1,
        'binary_classification': y_true_task2,
        'regression': y_true_task3
    }
    
    y_pred = {
        'classification': y_pred_task1,
        'binary_classification': y_pred_task2,
        'regression': y_pred_task3
    }
    
    # Create evaluation configuration
    config = EvaluationConfig(
        task_type="multitask",
        multitask_metrics=["task_accuracy", "overall_accuracy", "task_f1", "overall_f1"],
        create_plots=False,
        verbose=True
    )
    
    # Create evaluator
    evaluator = ModelEvaluator(config)
    
    # Evaluate
    result = evaluator.evaluate(y_true, y_pred)
    
    # Print results
    logger.info("Multi-task Evaluation Results:")
    for metric, value in result.metrics.items():
        if isinstance(value, dict):
            logger.info(f"  {metric}:")
            for task, task_value in value.items():
                logger.info(f"    {task}: {task_value:.4f}")
        else:
            logger.info(f"  {metric}: {value:.4f}")
    
    # Save results
    evaluator.save_results(result, "multitask_evaluation.json")
    
    return result

def example_advanced_statistical_analysis():
    """Example: Advanced statistical analysis"""
    logger.info("=== Advanced Statistical Analysis Example ===")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate data with known relationship
    x = np.random.normal(0, 1, n_samples)
    y_true = 2 * x + 1 + np.random.normal(0, 0.5, n_samples)
    y_pred = 1.8 * x + 1.2 + np.random.normal(0, 0.6, n_samples)  # Slightly different model
    baseline_pred = 1.5 * x + 0.8 + np.random.normal(0, 0.7, n_samples)  # Baseline model
    
    # Perform statistical analysis
    statistical_tests = StatisticalAnalysis.perform_statistical_tests(y_true, y_pred, baseline_pred)
    
    logger.info("Statistical Analysis Results:")
    for test_name, test_result in statistical_tests.items():
        logger.info(f"  {test_name}:")
        if isinstance(test_result, dict):
            for key, value in test_result.items():
                logger.info(f"    {key}: {value:.4f}")
        else:
            logger.info(f"    {test_result:.4f}")
    
    # Bootstrap confidence intervals
    confidence_intervals = StatisticalAnalysis.bootstrap_confidence_intervals(
        y_true, y_pred, lambda y_t, y_p: r2_score(y_t, y_p), 
        n_bootstrap=100, confidence_level=0.95
    )
    
    logger.info("Bootstrap Confidence Intervals:")
    for metric, result in confidence_intervals.items():
        if isinstance(result, tuple):
            logger.info(f"  {metric}: ({result[0]:.4f}, {result[1]:.4f})")
        else:
            logger.info(f"  {metric}: {result:.4f}")
    
    return statistical_tests, confidence_intervals

def example_custom_metrics():
    """Example: Custom evaluation metrics"""
    logger.info("=== Custom Metrics Example ===")
    
    # Define custom metric function
    def custom_balanced_accuracy(y_true, y_pred) -> Any:
        """Custom balanced accuracy metric"""
        return balanced_accuracy_score(y_true, y_pred)
    
    def custom_hamming_loss(y_true, y_pred) -> Any:
        """Custom Hamming loss metric"""
        return np.mean(y_true != y_pred)
    
    # Create sample data
    np.random.seed(42)
    y_true = np.random.randint(0, 3, 1000)
    y_pred = np.random.randint(0, 3, 1000)
    
    # Calculate custom metrics
    balanced_acc = custom_balanced_accuracy(y_true, y_pred)
    hamming_loss = custom_hamming_loss(y_true, y_pred)
    
    logger.info("Custom Metrics Results:")
    logger.info(f"  Balanced Accuracy: {balanced_acc:.4f}")
    logger.info(f"  Hamming Loss: {hamming_loss:.4f}")
    
    # Compare with standard metrics
    standard_metrics = ClassificationMetrics.calculate_all_metrics(y_true, y_pred)
    
    logger.info("Standard Metrics for Comparison:")
    for metric, value in standard_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    return {
        'balanced_accuracy': balanced_acc,
        'hamming_loss': hamming_loss,
        'standard_metrics': standard_metrics
    }

def example_model_comparison():
    """Example: Compare multiple models using evaluation metrics"""
    logger.info("=== Model Comparison Example ===")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    y_true = np.random.randint(0, 3, n_samples)
    
    # Generate predictions for different models
    models = {
        'Model_A': np.random.randint(0, 3, n_samples),
        'Model_B': np.random.randint(0, 3, n_samples),
        'Model_C': np.random.randint(0, 3, n_samples)
    }
    
    # Generate probabilities for each model
    model_probs = {}
    for model_name in models.keys():
        probs = np.random.rand(n_samples, 3)
        probs = probs / probs.sum(axis=1, keepdims=True)
        model_probs[model_name] = probs
    
    # Create evaluation configuration
    config = EvaluationConfig(
        task_type="classification",
        classification_metrics=["accuracy", "precision", "recall", "f1", "auc"],
        average_method="weighted",
        create_plots=True,
        save_plots=True,
        verbose=False
    )
    
    # Evaluate each model
    results = {}
    evaluator = ModelEvaluator(config)
    
    for model_name, y_pred in models.items():
        logger.info(f"Evaluating {model_name}...")
        result = evaluator.evaluate(y_true, y_pred, model_probs[model_name])
        results[model_name] = result.metrics
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results).T
    
    logger.info("Model Comparison Results:")
    logger.info(comparison_df.round(4))
    
    # Find best model for each metric
    logger.info("Best Models by Metric:")
    for metric in comparison_df.columns:
        if metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
            best_model = comparison_df[metric].idxmax()
            best_score = comparison_df[metric].max()
            logger.info(f"  {metric}: {best_model} ({best_score:.4f})")
    
    # Save comparison results
    comparison_df.to_csv("model_comparison.csv")
    logger.info("Model comparison saved to model_comparison.csv")
    
    return comparison_df

def example_visualization_demo():
    """Example: Demonstration of visualization capabilities"""
    logger.info("=== Visualization Demo ===")
    
    # Create sample data for different visualizations
    np.random.seed(42)
    
    # Classification data
    y_true_clf = np.random.randint(0, 3, 1000)
    y_pred_clf = np.random.randint(0, 3, 1000)
    y_prob_clf = np.random.rand(1000, 3)
    y_prob_clf = y_prob_clf / y_prob_clf.sum(axis=1, keepdims=True)
    
    # Regression data
    y_true_reg = np.random.normal(0, 1, 1000)
    y_pred_reg = y_true_reg + np.random.normal(0, 0.3, 1000)
    
    # Ranking data
    y_true_rank = np.random.binomial(1, 0.3, 1000)
    y_pred_rank = np.random.rand(1000)
    
    # Create visualizations
    logger.info("Creating visualizations...")
    
    # Classification visualizations
    EvaluationVisualizer.plot_confusion_matrix(y_true_clf, y_pred_clf, 
                                             class_names=['Class 0', 'Class 1', 'Class 2'])
    EvaluationVisualizer.plot_roc_curve(y_true_clf, y_prob_clf, 
                                       class_names=['Class 0', 'Class 1', 'Class 2'])
    EvaluationVisualizer.plot_precision_recall_curve(y_true_clf, y_prob_clf)
    
    # Regression visualizations
    EvaluationVisualizer.plot_regression_results(y_true_reg, y_pred_reg)
    
    # Ranking visualizations
    ranking_metrics = RankingMetrics.calculate_all_metrics(y_true_rank, y_pred_rank, [1, 3, 5, 10])
    EvaluationVisualizer.plot_ranking_metrics(ranking_metrics, [1, 3, 5, 10])
    
    logger.info("Visualizations completed!")

def main():
    """Run all evaluation examples"""
    logger.info("Starting Comprehensive Evaluation Metrics Examples")
    
    try:
        # Basic evaluation examples
        classification_result = example_classification_evaluation()
        regression_result = example_regression_evaluation()
        ranking_result = example_ranking_evaluation()
        
        # Specialized evaluation examples
        seo_result = example_seo_evaluation()
        multitask_result = example_multitask_evaluation()
        
        # Advanced analysis examples
        statistical_tests, confidence_intervals = example_advanced_statistical_analysis()
        custom_metrics = example_custom_metrics()
        
        # Model comparison
        comparison_df = example_model_comparison()
        
        # Visualization demo
        example_visualization_demo()
        
        logger.info("\n=== Summary ===")
        logger.info("All evaluation examples completed successfully!")
        logger.info("Key features demonstrated:")
        logger.info("  ✓ Classification metrics with confidence intervals")
        logger.info("  ✓ Regression metrics with statistical analysis")
        logger.info("  ✓ Ranking metrics with multiple k-values")
        logger.info("  ✓ SEO-specific metrics")
        logger.info("  ✓ Multi-task evaluation")
        logger.info("  ✓ Advanced statistical analysis")
        logger.info("  ✓ Custom metrics implementation")
        logger.info("  ✓ Model comparison capabilities")
        logger.info("  ✓ Comprehensive visualization")
        
        # Print sample results
        logger.info("\nSample Results:")
        logger.info(f"Classification F1: {classification_result.metrics['f1']:.4f}")
        logger.info(f"Regression R²: {regression_result.metrics['r2']:.4f}")
        logger.info(f"Ranking NDCG@10: {ranking_result.metrics['ndcg@10']:.4f}")
        logger.info(f"SEO CTR: {seo_result.metrics.get('click_through_rate', 0):.4f}")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        traceback.print_exc()

match __name__:
    case "__main__":
    main() 